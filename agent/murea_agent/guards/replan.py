"""Check the director's action plan; send concrete errors back for a re-plan."""

from __future__ import annotations

import math
from typing import Any
from collections.abc import Callable

from murea_agent import config
from murea_agent import reqlog
from murea_agent.memory.world import World
from murea_agent.util import _Clock, _noop, _say
from murea_agent.guards.action_sequence import (
    _action_sequence_repair_hint,
    _LEDGER_ASSIGNMENT_ERROR,
    _action_sequence_semantic_errors,
    _merge_unassigned_events,
    _normalize_action_sequence,
    _rebuild_event_ledger,
    _prepared_motion_errors,
    _repair_directive_physics,
)
from murea_agent.guards.density_audit import (
    _audit_single_event_density,
    _density_audit_repair_hint,
    _density_replan_schema,
    _single_event_needs_density_audit,
)
from murea_agent.director import director


class _PreparedDecision(dict):
    """In-memory prefetch result whose action ledger already passed validation."""


def _checked_plan(d: dict[str, Any], segment_seconds: int, base: str) -> tuple[dict[str, Any], list[str]]:
    """The plan's semantic errors, after fixing an incomplete event ledger without the model.

    What is left (an over-packed step) goes to the re-plan with its precise cause. Applied to every
    reply, re-plans included: a re-plan that fixed the real problem used to be re-run once more
    only for this bookkeeping error.
    """
    errors = _action_sequence_semantic_errors(d.get("action_sequence"), segment_seconds)
    if _LEDGER_ASSIGNMENT_ERROR not in errors:
        return d, errors
    merged = _merge_unassigned_events(d.get("action_sequence"))
    if merged is None:
        return d, errors
    remaining = _action_sequence_semantic_errors(merged, segment_seconds)
    reqlog.record(
        "action_ledger_merged", ok=not remaining, errors=errors, remaining=remaining, background=bool(base)
    )
    return {**d, "action_sequence": merged}, remaining


def _prepare_action_plan(
    world: World,
    user_text: str,
    d: dict[str, Any],
    on_partial: Callable[[dict[str, Any]], None],
    *,
    duration: float | None,
    asset_mentions: list[dict[str, str]] | None,
    clock: _Clock,
    progress: Callable[[str, str], None] = _noop,
    base: str = "",
    is_continuation: bool = False,
) -> tuple[dict[str, Any], bool]:
    # Only our in-memory prefetch wrapper is trusted; JSON/model fields cannot
    # claim that semantic validation happened. It is never serialized as metadata.
    if isinstance(d, _PreparedDecision):
        return dict(d), False
    # The GM, rather than a service-side action vocabulary, decides which visible
    # events exist.  Require its own event-duration ledger to fit the real video
    # budget.  If it compressed a causal chain, give the GM one semantic re-plan
    # before any image or video request is started.
    replanned = False
    segment_seconds = director._segment_seconds(world, duration)
    d, sequence_semantic_errors = _checked_plan(d, segment_seconds, base)
    density_audit = None
    # Auto steps still need goal coverage: the model can replace "release the
    # restraints" with "help the already freed captive". Density alone must not
    # expand a trusted queue; an explicitly missing goal may repair this event.
    if not sequence_semantic_errors and _single_event_needs_density_audit(
        d.get("action_sequence"),
        segment_seconds,
    ):
        with clock.stage("gm_density_audit"):
            matching = (
                isinstance(d, director._DecisionWithAudit)
                and d.sequence == d.get("action_sequence")
                and d.goal == user_text
                and d.seconds == segment_seconds
            )
            if matching:
                try:
                    density_audit = d.future.result()
                    reqlog.record("action_density_overlap", reused=True)
                except Exception:
                    # Normal auditor failures already return None; if its worker
                    # itself failed, perform the original check synchronously.
                    matching = False
            if not matching:
                density_audit = _audit_single_event_density(
                    d.get("action_sequence"),
                    user_text,
                    segment_seconds,
                    base=base,
                )
    if isinstance(d, director._DecisionWithAudit):
        # No future can escape into a saved directive or prefetch deepcopy.
        d = dict(d)
    current_steps = (
        _normalize_action_sequence(
            d.get("action_sequence"),
        ).get("steps")
        or []
    )
    audited_segments = int((density_audit or {}).get("required_segments") or 0)
    density_requires_replan = (not is_continuation and audited_segments > len(current_steps)) or (
        density_audit or {}
    ).get("goal_covered") is False
    if (
        density_audit
        and audited_segments == 1
        and len(current_steps) == 1
        and density_audit.get("continuous_shot") is True
    ):
        # The initial plan may allocate 6s to an action the auditor estimates
        # at 7s. Extend its runtime instead of rewriting/splitting the action.
        audited_seconds = sum(
            float(event.get("estimated_seconds") or 0) for event in density_audit.get("events", [])
        )
        if duration is None:
            d["duration_seconds"] = max(
                director._generation_seconds(world, d, duration),
                min(segment_seconds, math.ceil(audited_seconds)),
            )
    # Only the first round is about density; after that, quote the semantic errors the last
    # round actually reported. Repeating the same density hint only resamples the same
    # distribution and wastes a call.
    count_check = density_requires_replan
    attempts = 0
    while (sequence_semantic_errors or density_requires_replan) and attempts < config.GM_PLAN_REPAIR_TRIES:
        attempts += 1
        progress("gm", _say("正在校正动作时长与完整终点", "Adjusting the action timing"))
        if density_requires_replan:
            repair_suffix = _density_audit_repair_hint(density_audit or {}, segment_seconds)
        else:
            repair_suffix = _action_sequence_repair_hint(
                sequence_semantic_errors,
                segment_seconds,
            )
        repair_text = user_text + repair_suffix
        reqlog.record(
            "action_plan_repair",
            attempt=attempts,
            errors=sequence_semantic_errors,
            required_segments=audited_segments,
            background=bool(base),
        )
        with clock.stage("gm_plan_repair"):
            # Enforce count and event assignments in decoding, not only prose.
            # Other providers keep their established schema contract.
            repair_args = {}
            if (density_audit or {}).get("goal_covered") is False and config.LLM_BACKEND == "gemini":
                repair_args["model"] = config.ACTION_AUDIT_MODEL
            narrowed = density_requires_replan and config.LLM_BACKEND == "openai"
            if narrowed:
                repair_args["schema"] = _density_replan_schema(world, density_audit)
            repaired = director._decide(
                world,
                repair_text,
                on_partial,
                duration=duration,
                asset_mentions=asset_mentions,
                base=base,
                **repair_args,
            )
            # The narrowed path only asks for the fields that change (see `_REPLAN_FIELDS`) and
            # keeps the rest from the first answer. The non-narrowed path (old schema / non-OpenAI
            # backend) still replaces the whole result.
            d = {**d, **repaired} if narrowed else repaired
        replanned = True
        if is_continuation and (density_audit or {}).get("goal_covered") is False:
            d["_repair_current_event"] = True
        d, sequence_semantic_errors = _checked_plan(d, segment_seconds, base)
        repaired_steps = (
            _normalize_action_sequence(
                d.get("action_sequence"),
            ).get("steps")
            or []
        )
        if count_check and len(repaired_steps) != audited_segments:
            sequence_semantic_errors.append(
                f"density audit asked for {audited_segments} segments, the GM returned {len(repaired_steps)}"
            )
        density_requires_replan = False
    if sequence_semantic_errors:
        # Re-planning did not fix it. Rebuild the event ledger from the steps, and if that
        # still fails keep only the first step, instead of failing the whole beat.
        for max_steps in (3, 1):
            rebuilt = _rebuild_event_ledger(d.get("action_sequence"), max_steps)
            rebuilt_errors = _action_sequence_semantic_errors(rebuilt, segment_seconds)
            if not rebuilt_errors:
                reqlog.record(
                    "action_plan_rebuilt",
                    errors=sequence_semantic_errors,
                    steps=len(rebuilt.get("steps") or []),
                )
                d = {**d, "action_sequence": rebuilt}
                sequence_semantic_errors = []
                break
    if sequence_semantic_errors:
        raise ValueError(
            f"Action timing plan still invalid after {attempts} attempts: "
            + "; ".join(sequence_semantic_errors)
        )

    motion_errors = _prepared_motion_errors(world, d, user_text)
    if motion_errors:
        progress("gm", _say("正在补全本拍物理字段", "Completing the motion plan"))
        with clock.stage("gm_physics_repair"):
            d = _repair_directive_physics(world, d, user_text, motion_errors, base=base)
        reqlog.record("motion_plan_repair", errors=motion_errors, background=bool(base))

    return d, replanned
