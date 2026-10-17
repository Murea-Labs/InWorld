"""Motion plans and action sequences: normalisation, validation and repair."""

from __future__ import annotations

import re
from typing import Any
from murea_agent.util import _clean_motion_text
from murea_agent.director.agent_schema import (
    MOTION_EVIDENCE,
    MOTION_FRAMINGS,
    MOTION_PHASES,
    MOTION_RELATIONS,
    MOTION_TYPES,
    SECTION_ORDER,
)


_EMPTY_MOTION_PLAN: dict[str, Any] = {
    "interaction_type": "none",
    "phase": "none",
    "relation_change": "none",
    "actors": [],
    "instrument": "",
    "target": "",
    "start_state": "",
    "path": "",
    "contact_points": "",
    "end_state": "",
    "camera_side": "",
    "framing": "none",
    "visual_evidence": [],
    "must_show": [],
    "invariants": [],
}

_PHASE_RELATIONS = {
    "none": {"none"},
    "approach": {"distance"},
    "engage": {"contact"},
    "transform": {"containment", "support", "possession", "configuration"},
    # Gaining distance may be an approach before contact or backing off from contact; the latter is
    # disengage, and a different phase name must not make the user's separation illegal.
    "disengage": {"contact", "distance", "support", "possession"},
    "settle": {"none", "distance", "support"},
}

_RELATION_DEFAULT_PHASE = {
    "distance": "approach",
    "contact": "engage",
    "containment": "transform",
    "support": "transform",
    "possession": "transform",
    "configuration": "transform",
}

_RELATION_EVIDENCE = {
    "none": set(),
    "distance": {"actor_full_body", "target_full_structure", "support_surface"},
    "contact": {"contact_point", "target_full_structure"},
    "containment": {
        "actor_full_body",
        "actor_feet",
        "target_full_structure",
        "boundary_opening",
        "support_surface",
    },
    "support": {"actor_full_body", "contact_point", "support_surface", "target_full_structure"},
    "possession": {"actor_hands", "contact_point", "target_full_structure"},
    "configuration": {"actor_hands", "contact_point", "target_full_structure"},
}

_EVIDENCE_TEXT = {
    "actor_full_body": ("the moving actor from head to both feet, with visible ground beneath both shoes"),
    "actor_hands": "both acting hands and their grip on the target",
    "actor_feet": "both feet and each planted or moving footfall",
    "target_full_structure": "the target's complete silhouette and relevant rigid structure",
    "contact_point": "every active contact point without occlusion",
    "support_surface": "the continuous support surface directly beneath the actor",
    "boundary_opening": "the complete open passage and both of its solid edges",
    "both_actors_contact": "both actors and the exact point where their bodies meet",
    "instrument_full_structure": ("the complete moving instrument or projectile as a single rigid object"),
    "instrument_path": ("the instrument's continuous visible path from its start position to contact"),
}

_RELATION_PHYSICS = {
    "distance": (
        "A visible gap of air or ground continuously separates the actor and target; "
        "the gap narrows smoothly to the stated end distance."
    ),
    "contact": (
        "The participants meet first and only at the named contact points, while their "
        "remaining rigid structures and silhouettes stay spatially distinct."
    ),
    "containment": (
        "The crossing follows the named open passage: the leading body part, torso, "
        "and trailing feet pass its plane in order while both solid edges stay rigid."
    ),
    "support": (
        "Weight transfers only through the named contact points; the actor's center "
        "of mass stays over the visible support surface."
    ),
    "possession": (
        "The target starts moving only after the hand reaches it and closes a visible "
        "grip; it then follows that same hand as one rigid object."
    ),
    "configuration": (
        "Only the named movable part changes configuration, following its visible "
        "hinge, joint, or track while the rest of the target stays rigid."
    ),
}


def _distance_progress_clause(plan: dict[str, Any]) -> str:
    """Add visible travel requirements to continuous movement.

    With only "walk forward a bit", a short clip often shows one or two steps; this writes the key
    acceptance criteria of a movement segment into the video prompt: keep moving along a clear path
    and stop in the end state. Destination name, path and end state are still decided by the
    director; the server never guesses places.
    """
    if plan.get("relation_change") != "distance":
        return ""
    return (
        "Travel progression — keep the actor moving forward along the named path "
        "for the shot, visibly covering a meaningful stretch instead of taking only "
        "one or two steps. Reach the exact end state at the endpoint and settle there; "
        "if the endpoint is reached early, remain there while preserving the target "
        "and the continuous route."
    )


_CAMERA_SCALE_WORDS = re.compile(
    r"\b(?:extreme\s+wide|wide|medium[- ]wide|medium|close[- ]?up|close)\s+shot\b",
    re.I,
)


def _normalize_motion_plan(raw: Any) -> dict[str, Any]:
    """Turn the director's motion plan into a stable structure; old data or ordinary static shots fall back to an empty plan."""
    if not isinstance(raw, dict):
        return dict(_EMPTY_MOTION_PLAN)
    plan = dict(_EMPTY_MOTION_PLAN)
    for key, allowed in (
        ("interaction_type", MOTION_TYPES),
        ("phase", MOTION_PHASES),
        ("relation_change", MOTION_RELATIONS),
        ("framing", MOTION_FRAMINGS),
    ):
        value = _clean_motion_text(raw.get(key)).lower()
        plan[key] = value if value in allowed else "none"
    for key in (
        "instrument",
        "target",
        "start_state",
        "path",
        "contact_points",
        "end_state",
        "camera_side",
    ):
        plan[key] = _clean_motion_text(raw.get(key))
    # camera_side only describes the spatial side; the director sometimes slips "medium shot" in,
    # contradicting the wide framing the server raised. Shot size is controlled only by framing, so
    # strip it here.
    plan["camera_side"] = _CAMERA_SCALE_WORDS.sub("", plan["camera_side"]).strip(" ,;:")
    for key in ("actors", "must_show", "invariants"):
        values = raw.get(key)
        plan[key] = (
            [text for text in (_clean_motion_text(x) for x in values or []) if text]
            if isinstance(values, list)
            else []
        )
    # The director sometimes copies visual_evidence enum values into must_show; structured evidence
    # is already rendered as natural English, and keeping both would put internal tokens like
    # actor_full_body into the final prompt.
    plan["must_show"] = [text for text in plan["must_show"] if text.lower() not in MOTION_EVIDENCE]
    evidence = raw.get("visual_evidence")
    plan["visual_evidence"] = (
        [
            value
            for value in (_clean_motion_text(x).lower() for x in evidence or [])
            if value in MOTION_EVIDENCE
        ]
        if isinstance(evidence, list)
        else []
    )
    # The relation type already fixes the minimum evidence; missing items need not fail the round.
    # The server fills them from a fixed vocabulary so full body, support surface, boundary or
    # contact point really reach the prompt.
    required_evidence = set(_RELATION_EVIDENCE.get(plan["relation_change"], set()))
    if plan["interaction_type"] == "impact":
        required_evidence.update({"instrument_full_structure", "instrument_path"})
    if (
        len(plan["actors"]) > 1
        and plan["relation_change"] == "contact"
        and plan["interaction_type"] != "impact"
    ):
        required_evidence.add("both_actors_contact")
    for value in MOTION_EVIDENCE:
        if value in required_evidence and value not in plan["visual_evidence"]:
            plan["visual_evidence"].append(value)
    # In real clips medium-wide still cut everything below the knees. Whenever the full-body path,
    # crossing order or load point must be visible, framing is raised to wide deterministically
    # instead of trusting the model with feet.
    if _framing_is_promoted(plan) and plan["framing"] in {"medium", "medium-wide"}:
        plan["framing"] = "wide"
    return plan


def _motion_active(plan: dict[str, Any]) -> bool:
    return any(
        plan.get(key) != "none"
        for key in (
            "interaction_type",
            "phase",
            "relation_change",
        )
    )


def _repair_motion_plan(
    plan: dict[str, Any],
    directive: dict[str, Any] | None = None,
    *,
    reason: str = "motion plan phase and relation disagree",
) -> dict[str, Any]:
    """Compile phase conflicts decidable from the relation into an executable plan instead of stopping the user with an error.

    relation_change is the physical fact to show; phase is only where that fact sits in the action's
    lifecycle. The director sometimes writes "back off" as disengage + distance, or "open" as engage
    + configuration; both can be repaired deterministically from the relation and the action text.
    The repair is recorded on the directive for the inspector instead of silently swallowing model
    output.
    """
    repaired = dict(plan)
    relation = repaired.get("relation_change", "none")
    phase = repaired.get("phase", "none")
    if repaired.get("interaction_type") == "impact" and relation != "contact":
        # `impact` already states the semantic relationship: a structured
        # instrument reaches a structured target. Keep later deformation or
        # ignition as effects, but compile the primary relation as contact.
        repaired["relation_change"] = "contact"
        repaired["phase"] = "engage"
        relation = "contact"
        phase = "engage"
        if directive is not None:
            directive.setdefault("_server_repairs", []).append("impact motion plan aligned to engage/contact")
    if relation in _RELATION_DEFAULT_PHASE:
        allowed = _PHASE_RELATIONS.get(phase, set())
        if relation not in allowed:
            # The server only repairs structural conflicts between enums and never reads path /
            # start / end_state to guess a "retreat phase"; if disengage is needed, the director
            # must give it explicitly in the structured phase.
            target_phase = _RELATION_DEFAULT_PHASE[relation]
            repaired["phase"] = target_phase
            if directive is not None:
                directive.setdefault("_server_repairs", []).append(
                    f"{reason}: phase={phase}, relation={relation} -> {target_phase}"
                )
    elif relation == "none" and phase not in {"none", "settle"}:
        # A shot without a relation change can only be static / dialogue; keep the action text but
        # return the plan to the valid settle end state, so a meaningless phase name does not raise
        # ValueError.
        repaired["phase"] = "settle"
        if directive is not None:
            directive.setdefault("_server_repairs", []).append(
                f"phase without a relation change adjusted: phase={phase} -> settle"
            )
    return repaired


def _motion_plan_errors(plan: dict[str, Any]) -> list[str]:
    """Check relations only, never specific objects; so one rule set covers any interaction target."""
    if not _motion_active(plan):
        return (
            []
            if all(
                plan.get(k) == "none"
                for k in (
                    "interaction_type",
                    "phase",
                    "relation_change",
                )
            )
            else ["an empty plan must set all three category fields to none"]
        )
    errors: list[str] = []
    if plan["interaction_type"] == "none":
        errors.append("interaction_type cannot be none when there is motion")
    if plan["phase"] == "none":
        errors.append("phase cannot be none when there is motion")
    if plan["relation_change"] == "none" and plan["phase"] != "settle":
        errors.append("relation_change cannot be none when there is motion, except for settle")
    if plan["interaction_type"] == "impact":
        if plan["relation_change"] != "contact":
            errors.append("impact must use relation contact for the instrument reaching the target")
        if not plan.get("instrument"):
            errors.append("impact motion plan has no instrument")
    allowed = _PHASE_RELATIONS.get(plan["phase"], set())
    if plan["relation_change"] not in allowed:
        errors.append(f"phase={plan['phase']} cannot change relation={plan['relation_change']}")
    for key in (
        "actors",
        "target",
        "start_state",
        "path",
        "end_state",
        "camera_side",
        "visual_evidence",
        "invariants",
    ):
        if not plan.get(key):
            errors.append(f"motion plan has no {key}")
    if plan["framing"] == "none":
        errors.append("framing cannot be none when there is motion")
    if plan["relation_change"] in {"distance", "containment", "support"} and plan["framing"] not in {
        "wide",
        "medium-wide",
    }:
        errors.append(
            f"relation={plan['relation_change']} needs wide or medium-wide framing to show the evidence"
        )
    evidence = set(plan["visual_evidence"])
    required = set(_RELATION_EVIDENCE.get(plan["relation_change"], set()))
    if plan["interaction_type"] == "impact":
        required.update({"instrument_full_structure", "instrument_path"})
    if (
        len(plan["actors"]) > 1
        and plan["relation_change"] == "contact"
        and plan["interaction_type"] != "impact"
    ):
        required.add("both_actors_contact")
    missing = sorted(required - evidence)
    if missing:
        errors.append("visual_evidence is missing " + ", ".join(missing))
    if plan["relation_change"] not in {"none", "distance"} and not plan["contact_points"]:
        errors.append("contact, crossing, support, holding or configuration changes need contact_points")
    return errors


# Articles, quantifiers, demonstratives: what models add or drop most when copying free text; they
# do not change the referent.
_AUDIT_FILLER = re.compile(
    r"\b(the|a|an|this|that|these|those|其|该|那个|这个|一个)\b|[\s，,.。;；:：'\"()（）\-]+",
    re.I,
)


def _audit_norm(value: Any) -> str:
    return _AUDIT_FILLER.sub("", _clean_motion_text(value).casefold())


# The server raises framing to wide for these relations (see `_repair_motion_plan`). The audit must
# use the same criterion, or it would blame the model for not copying the raised value.
_FRAMING_PROMOTED_RELATIONS = frozenset({"distance", "containment", "support"})


def _framing_is_promoted(plan: dict[str, Any]) -> bool:
    return plan.get("relation_change") in _FRAMING_PROMOTED_RELATIONS


def _motion_audit_error(result: dict[str, Any], plan: dict[str, Any]) -> str:
    """Whether the merged call contradicts its own motion plan.

    ⚠️ The two kinds of fields must be judged differently; that is the whole point of this function:

    - `motion_phase` / `motion_relation_change` are **enums** (approach/transform…, distance/containment…). A
      mismatch is a real contradiction: the model says "approach" while the plan says "cross", and that job must not go
      to the video model. Compare strictly.
    - `motion_target` / `motion_framing` are **free text**. Requiring a verbatim copy is unrealistic: "the arched
      airlock door" and "arched airlock door" are the same door, and calling it a conflict only falls back to the
      separate rewriter for nothing.

    ⚠️ **The plan passed in must be the model's own, not the one the server modified.**

    The question is whether the model contradicts **its own** plan. Before the audit the server
    modifies it three times: `_advance_completed_transition` (consumes a completed approach),
    `_reconcile_dialogue_motion` (pure dialogue beats become none), `_repair_motion_plan` (impact
    aligned to contact, framing raised to wide). Comparing the modified plan would flag a model that
    wrote the same thing in both places.

    In 305 beats there were 61 fallbacks (20%), 25 of them on `motion_relation_change`, mostly
    "expected 'none', got 'contact'", left by `_reconcile_dialogue_motion`. Each cost 1.1–1.9 s plus
    an LLM call.

    All four fields used to be compared exactly; in 258 segments **51%** fell back, a median of 2.60
    s and one more call each, for what the comments called a rare path. Free text is now compared by
    mutual containment after normalization: wording differences pass, a different noun is still
    caught.
    """
    # ⚠️ **motion_phase is not audited.** Two reasons, both required:
    #
    # 1. It is a function of relation, not an independent fact. `_repair_motion_plan` sets
    #    phase = _RELATION_DEFAULT_PHASE[relation]; auditing it audits relation twice.
    # 2. It is an **internal field**, removed before the video prompt is built (see agent.md, Output). A mismatch makes
    #    no difference on screen.
    #
    # And it caused **real** fallbacks: the merged call emits motion_plan and the six sections
    # together, the model copies what it wrote, and the server **afterwards** changes phase in
    # `_repair_motion_plan` and compares the changed value, so a faithful copy was judged a
    # conflict. Each cost 2.6 s and a call.
    #
    # relation_change / target / framing are still audited: they decide what is on screen.
    # ⚠️ With no relation change there is nothing to audit; let it pass.
    #
    # With `relation_change == "none"`, `_motion_active(plan)` is False, `_motion_plan_clause`
    # returns an empty string and `_lock_motion_plan` is a no-op: **no physical constraint appears
    # in the prompt at all**. The audit still asked the model to copy `none / '' / none` and fell
    # back on a mismatch; 10 of 25 fallbacks in 102 beats were this path, 2.45–3.18 s and one more
    # call each, buying nothing.
    if not _motion_active(plan):
        return ""
    mismatches: list[str] = []
    for key, wanted in (("motion_relation_change", plan["relation_change"]),):
        got = _clean_motion_text(result.get(key)).casefold()
        wanted_norm = _clean_motion_text(wanted).casefold()
        if wanted_norm in {"", "none"} and got in {"", "none", "n/a"}:
            continue
        if got != wanted_norm:
            mismatches.append(f"{key} should be {wanted!r}, got {result.get(key)!r}")

    # ⚠️ Decide "no target" **before stripping filler words**. `_audit_norm` drops the lone a in "n/a" as an article
    # (`\ba\b` has a word boundary at the slash), leaving "n/" that matches no sentinel, so plan
    # none vs model N/A, actually consistent, was judged a conflict. Recognize sentinels on the raw
    # text first, then normalize.
    def audit_field(value: Any) -> str:
        raw = _clean_motion_text(value).casefold().strip()
        return "" if raw in {"", "none", "na", "n/a"} else _audit_norm(value)

    # ⚠️ **Do not audit framing the server raised**: the same trap as `motion_phase` above.
    #
    # `_repair_motion_plan` raises framing to wide for distance / containment / support (in real
    # clips medium-wide cut everything below the knees). The model writes the same `medium` in
    # `motion_plan.framing` and `motion_framing`, the server **afterwards** changes only the former
    # and compares it to the latter, so a consistent model is judged a conflict, wasting about 1.9 s
    # of extra calls. Historically 21 of 59 fallbacks were this. Beats without a raise are still
    # audited: then both sides should match.
    audited = [("motion_target", plan["target"])]
    if not _framing_is_promoted(plan):
        audited.append(("motion_framing", plan["framing"]))
    for key, wanted in audited:
        got = audit_field(result.get(key))
        want = audit_field(wanted)
        if not got and not want:
            continue
        if got and want and (got == want or got in want or want in got):
            continue
        mismatches.append(f"{key} should be {wanted!r}, got {result.get(key)!r}")
    return "；".join(mismatches)


def _strip_motion_audit(result: dict[str, Any]) -> dict[str, str]:
    return {k: str(result.get(k) or "") for k in SECTION_ORDER if k in result}
