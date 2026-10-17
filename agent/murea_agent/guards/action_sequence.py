"""Turning one player input into 1-3 short clips: the action sequence, its checks and repairs."""

from __future__ import annotations

import copy
import re
from typing import Any
from murea_agent import config
from murea_agent.providers import llm
from murea_agent.memory.world import World
from murea_agent.guards.motion import (
    _EMPTY_MOTION_PLAN,
    _PHASE_RELATIONS,
    _RELATION_DEFAULT_PHASE,
    _motion_active,
    _motion_plan_errors,
    _normalize_motion_plan,
    _repair_motion_plan,
)
from murea_agent.util import _clean_motion_text, _pretty, _say, cap
from murea_agent.director.agent_schema import (
    MOTION_PHASES,
    MOTION_PLAN_SCHEMA,
    MOTION_RELATIONS,
)
from murea_agent.shot.definitions import (
    _PendingImage,
    _english_asset_prompt,
)
from murea_agent.guards.density_audit import (
    _motion_replan_schema,
)


def _canonical_event_ids(raw: dict[str, Any]) -> dict[str, Any]:
    """Renumber an unambiguous event ledger without changing any planned action.

    Some models count from zero. Renumber both ends of the references together;
    changing only events[0].id made its step look orphaned and re-ran the whole GM.
    Missing, duplicated or reordered assignments still require semantic repair.
    """
    events, steps = raw.get("events"), raw.get("steps")
    if not isinstance(events, list) or not events or not isinstance(steps, list):
        return raw
    if not all(isinstance(event, dict) for event in events):
        return raw
    ids = [event.get("id") for event in events]
    if any(type(value) is not int or value < 0 for value in ids):
        return raw
    if len(set(ids)) != len(ids):
        return raw
    assigned = []
    for step in steps:
        if not isinstance(step, dict) or not isinstance(step.get("event_ids"), list):
            return raw
        if len(step["event_ids"]) != 1 or type(step["event_ids"][0]) is not int:
            return raw
        assigned.extend(step["event_ids"])
    if assigned != ids or ids == list(range(1, len(ids) + 1)):
        return raw
    mapping = {old: new for new, old in enumerate(ids, 1)}
    return {
        **raw,
        "events": [{**event, "id": mapping[event["id"]]} for event in events],
        "steps": [{**step, "event_ids": [mapping[value] for value in step["event_ids"]]} for step in steps],
    }


def _normalize_action_sequence(raw: Any) -> dict[str, Any]:
    """Turn one user intent into 1–3 short video events; only the first is executed now."""
    if not isinstance(raw, dict):
        return {"goal": "", "budget_reason": "", "events": [], "steps": []}
    raw = _canonical_event_ids(raw)

    def event_record(event: Any, fallback_id: int = 0) -> dict[str, Any] | None:
        if not isinstance(event, dict):
            return None
        try:
            event_id = int(event.get("id") or fallback_id)
        except (TypeError, ValueError):
            event_id = fallback_id
        try:
            seconds = float(event.get("estimated_seconds") or 0)
        except (TypeError, ValueError):
            seconds = 0.0
        return {
            "id": max(0, event_id),
            "description": _clean_motion_text(event.get("description"))[:300],
            "visible_result": _clean_motion_text(event.get("visible_result"))[:300],
            "estimated_seconds": max(0.0, seconds),
        }

    master_events: list[dict[str, Any]] = []
    raw_master = raw.get("events")
    if isinstance(raw_master, list):
        for index, event in enumerate(raw_master, 1):
            normalized = event_record(event, index)
            if normalized is not None:
                master_events.append(normalized)
    event_by_id = {event["id"]: event for event in master_events if event["id"] > 0}

    steps: list[dict[str, Any]] = []
    candidates = raw.get("steps")
    if isinstance(candidates, list):
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            phase = _clean_motion_text(candidate.get("phase")).lower()
            relation = _clean_motion_text(candidate.get("relation_change")).lower()
            kind = _clean_motion_text(candidate.get("kind")).lower()
            raw_ids = candidate.get("event_ids")
            event_ids = (
                [
                    int(value)
                    for value in raw_ids or []
                    if isinstance(value, int) and not isinstance(value, bool)
                ]
                if isinstance(raw_ids, list)
                else []
            )
            events = [event_by_id[value] for value in event_ids if value in event_by_id]
            # Compatibility with the first semantic-ledger revision and hand-built
            # fixtures. Fresh GM output uses the master list plus event_ids.
            if not events and isinstance(candidate.get("events"), list):
                for index, event in enumerate(candidate["events"], 1):
                    normalized = event_record(event, index)
                    if normalized is not None:
                        events.append(normalized)

            def role_values(key: str, candidate: dict[str, Any] = candidate) -> list[str]:
                values = candidate.get(key)
                return (
                    [text for text in (_clean_motion_text(value)[:80] for value in values or []) if text][:6]
                    if isinstance(values, list)
                    else []
                )

            steps.append(
                {
                    "instruction": _clean_motion_text(candidate.get("instruction"))[:500],
                    "place": _clean_motion_text(candidate.get("place"))[:120],
                    "visible_subjects": [
                        _clean_motion_text(value)[:80]
                        for value in (candidate.get("visible_subjects") or [])
                        if _clean_motion_text(value)
                    ][:6]
                    if isinstance(candidate.get("visible_subjects"), list)
                    else [],
                    "actors": role_values("actors"),
                    "instruments": role_values("instruments"),
                    "targets": role_values("targets"),
                    "events": events[:4],
                    "event_ids": event_ids[:1],
                    "kind": kind if kind in {"motion", "dialogue", "observation"} else "motion",
                    "phase": phase if phase in MOTION_PHASES else "none",
                    "relation_change": relation if relation in MOTION_RELATIONS else "none",
                    "end_state": _clean_motion_text(candidate.get("end_state"))[:500],
                    "continuity_anchor": _clean_motion_text(candidate.get("continuity_anchor"))[:500],
                }
            )
    return {
        "goal": _clean_motion_text(raw.get("goal"))[:500],
        "budget_reason": _clean_motion_text(raw.get("budget_reason"))[:500],
        "events": master_events[:3],
        "steps": steps,
    }


_LEDGER_ASSIGNMENT_ERROR = "each event must be assigned, in order, to exactly one video step"


def _merge_unassigned_events(raw: Any) -> dict[str, Any] | None:
    """Fold events a step left out of its `event_ids`, or listed beyond the first, into that step.

    The most common ledger failure (8 of 8 sampled): the model lists two events, writes one step
    whose instruction films both, and references only the first, or both in one step (the schema's
    one-id limit is not enforced on every backend). The whole director call used to be re-run for
    it, 15-25 s each, often three times. Merging changes nothing that is filmed (instructions stay
    as written) and keeps the time check honest: the step's event takes the sum of the durations,
    so an over-packed step still fails, now with "needs about Ns" instead of a bookkeeping error.

    An unreferenced event goes to the step that references the event before it (the first step if
    none does). Returns None when the ledger is not merely incomplete (out of order, duplicated,
    a step with no event), which still needs the model.
    """
    if not isinstance(raw, dict):
        return None
    raw = _canonical_event_ids(raw)
    events, steps = raw.get("events"), raw.get("steps")
    if not isinstance(events, list) or not events or not isinstance(steps, list) or not steps:
        return None
    if not all(isinstance(e, dict) for e in events) or not all(isinstance(s, dict) for s in steps):
        return None
    ids = [e.get("id") for e in events]
    if ids != list(range(1, len(ids) + 1)):
        return None
    groups: list[list[int]] = []
    for step in steps:
        refs = step.get("event_ids")
        if (
            not isinstance(refs, list)
            or not refs
            or not all(type(i) is int and 1 <= i <= len(ids) for i in refs)
        ):
            return None
        groups.append(list(refs))
    flat = [i for group in groups for i in group]
    if flat != sorted(set(flat)):
        return None
    for orphan in sorted(set(ids) - set(flat)):
        owner = max((k for k, group in enumerate(groups) if group[0] < orphan), default=0)
        groups[owner].append(orphan)
        groups[owner].sort()
    if all(len(group) == 1 for group in groups):
        return None
    by_id = {e["id"]: e for e in events}
    merged = []
    for k, group in enumerate(groups, 1):
        parts = [by_id[i] for i in group]
        seconds = 0.0
        for part in parts:
            try:
                seconds += max(0.0, float(part.get("estimated_seconds") or 0))
            except (TypeError, ValueError):
                pass
        merged.append(
            {
                "id": k,
                "description": "; then ".join(
                    text for text in (_clean_motion_text(p.get("description")) for p in parts) if text
                ),
                "visible_result": _clean_motion_text(parts[-1].get("visible_result")),
                "estimated_seconds": seconds,
            }
        )
    return {
        **raw,
        "events": merged,
        "steps": [{**step, "event_ids": [k]} for k, step in enumerate(steps, 1)],
    }


def _rebuild_event_ledger(raw: Any, max_steps: int = 3) -> Any:
    """Rebuild the event ledger from the steps themselves: step i owns event i.

    A last resort after re-planning: models sometimes keep miscounting the master event list
    while every step is complete. Each step keeps the event it referenced (or one built from
    its own instruction), so nothing that will be filmed changes; only the bookkeeping does.
    `max_steps=1` keeps just the first step, for when even that does not validate.
    """
    if not isinstance(raw, dict) or not isinstance(raw.get("steps"), list):
        return raw
    by_id = {e.get("id"): e for e in raw.get("events") or [] if isinstance(e, dict)}
    events, steps = [], []
    for index, step in enumerate([st for st in raw["steps"] if isinstance(st, dict)][:max_steps], 1):
        ids = step.get("event_ids") if isinstance(step.get("event_ids"), list) else []
        own = next((by_id[i] for i in ids if i in by_id), None)
        if own is None and isinstance(step.get("events"), list) and step["events"]:
            own = step["events"][0] if isinstance(step["events"][0], dict) else None
        text = _clean_motion_text(step.get("instruction"))
        event = {
            "description": _clean_motion_text((own or {}).get("description")) or text,
            "visible_result": _clean_motion_text((own or {}).get("visible_result"))
            or _clean_motion_text(step.get("end_state"))
            or text,
            "estimated_seconds": (own or {}).get("estimated_seconds") or 4,
            "id": index,
        }
        events.append(event)
        steps.append({**step, "event_ids": [index]})
    return {**raw, "events": events, "steps": steps}


_CONTINUATION_GOAL = re.compile(r"继续完成同一个目标[“\"]([^”\"]+)[”\"]")


def _sequence_goal(request_text: str, fallback: str) -> str:
    """On auto-continue take the original goal from the quotes; on the first beat keep the user's own words and never let the director swap the goal."""
    text = _clean_motion_text(request_text)
    matched = _CONTINUATION_GOAL.search(text)
    return _clean_motion_text(matched.group(1) if matched else text or fallback)[:500]


_GOAL_RELATION_PATTERNS = {
    "distance": re.compile(
        r"靠近|走到|跑到|接近|移动|移向|来到|抵达|向前|往前|前行|"
        r"继续前进|\bapproach\b|\bwalk to\b|\bmove toward\b",
        re.I,
    ),
    "configuration": re.compile(r"打开|关上|关闭|拉开|推开|掀开|合上|\bopen\b|\bclose\b", re.I),
    "possession": re.compile(
        r"拿起|拿取|捡起|拾起|抓起|接过|取出|取物|拿到|取得|"
        r"\bpick up\b|\btake\b|\bgrab\b",
        re.I,
    ),
    "containment": re.compile(
        r"进入|进去|走进|走入|踏入|踏进|跨入|迈入|迈进|穿过|跨过|跨进|"
        r"退出|退回|返回|回到|走出|走回|出来|撤出|出去|离开|\benter\b|"
        r"\b(?:go|walk|step|move|cross)(?:s|ed|ing)?\s+(?:inside|into)\b|"
        r"\bcross through\b|\b(?:exit|leave|return|go back|walk back|step out|come back)\b",
        re.I,
    ),
    "support": re.compile(
        r"坐下|坐进|坐到|躺下|躺到|骑上|爬上|登上|\bsit\b|\blie down\b|\bmount\b|"
        r"\bclimb onto\b",
        re.I,
    ),
}


_OBSERVATION_ACTION = re.compile(
    r"定场镜头|观察|查看|看向|看着|张望|凝视|注视|搜寻|搜索|寻找|检查|"
    r"翻阅|阅读|浏览|环顾|审视|\b(?:look|watch|inspect|search|examine|read|scan)\b",
    re.I,
)


_DIALOGUE_ACTION = re.compile(
    r"交谈|对话|询问|问道|回答|回应|告诉|说明|召见|禀报|劝说|争辩|谈论|"
    r"\b(?:talk|talks|talking|speak|speaks|speaking|ask|asks|asking|"
    r"answer|answers|tell|tells|telling|conversation|dialogue)\b",
    re.I,
)


_ACTION_DURATION_SECONDS = {
    # Conservative estimates of the time to complete one visible result, not model generation times;
    # used to decide whether a user sentence is split into several clips. Each relation is still
    # checked physically by motion_plan.
    "distance": 3.0,
    "configuration": 3.5,
    "containment": 3.5,
    "support": 3.0,
    "possession": 3.0,
    "contact": 2.5,
    "observation": 2.5,
    "dialogue": 2.5,
}


_GOAL_RELATION_PHASE = {
    "distance": "approach",
    "configuration": "transform",
    "possession": "transform",
    "containment": "transform",
    "support": "transform",
    "contact": "engage",
}


# ⚠️ The text of this table **is shown to users** and fed back into the director's history as "the user said"
# (on auto-continued beats). So it must be plain language.
#
# It used to contain audit language ("complete the configuration change along the visible joint or
# track (continuity anchor: same person, same target…)"). After one sentence was split into four
# segments, 6 of 14 history lines were such machine text, 3 of them as "the user said", and the
# director continued as if the player were issuing CAD commands. Physical constraints belong in
# motion_plan for the rewriter, not here.
_GOAL_RELATION_INSTRUCTION_ZH = {
    "distance": "走到跟前，停稳",
    "configuration": "把它完全打开",
    "possession": "伸手握住它，稳稳拿在手里",
    "containment": "走进去",
    "support": "把重心压上去，站稳",
    "contact": "伸手碰到它",
}
_GOAL_RELATION_INSTRUCTION_EN = {
    "distance": "Walk right up to it and stop",
    "configuration": "Open it all the way",
    "possession": "Reach out and take it firmly in hand",
    "containment": "Step inside",
    "support": "Put your weight on it and steady yourself",
    "contact": "Reach out and touch it",
}


# The relations that have a plain-language fallback step.
_GOAL_RELATIONS = frozenset(_GOAL_RELATION_INSTRUCTION_EN)


def _goal_relation_instruction(relation: str, default: str = "") -> str:
    table = _GOAL_RELATION_INSTRUCTION_ZH if config.LANGUAGE == "zh" else _GOAL_RELATION_INSTRUCTION_EN
    return table.get(relation, default)


_GOAL_RELATION_END = {
    "distance": "The actor reaches the required position and remains outside the target",
    "configuration": "The requested movable part reaches its final configuration",
    "possession": "The target is visibly held and follows the same gripping hand",
    "containment": "The actor's entire body is on the destination side of the open boundary",
    "support": "The actor's weight is visibly settled on the target support surface",
    "contact": "The named body parts visibly meet at the intended contact point",
}


_CONTAINMENT_EXIT_ACTION = re.compile(
    r"退出|退回|返回|回到|走出|走回|出来|撤出|出去|离开|"
    r"\b(?:exit|leave|return|go back|walk back|step out|come back)\b",
    re.I,
)


def _ensure_action_sequence_budget(
    sequence: dict[str, Any],
    motion_plan: dict[str, Any],
    segment_seconds: int = 4,
) -> None:
    """Respect the director's semantic split; do not pad repeated steps by keyword count.

    The director already decides ``steps`` from length, independence and information density. This
    used to treat several observation words as several segments, splitting "establish the scene and
    observe the same target" into two nearly identical narrations. Now only the director produces
    1–3 semantic steps; the server still checks physical relations, phase order and completeness
    later but no longer adds observation / dialogue by wording.
    """
    return


def _action_sequence_semantic_errors(
    raw: Any,
    segment_seconds: int = 4,
) -> list[str]:
    """Validate the GM's semantic duration ledger without parsing natural language.

    The GM owns the semantic decision: it names atomic visible events and estimates
    their duration.  The service only verifies that the returned ledger is complete,
    fits the actual backend duration, and does not repeat the same visible result.
    No action-word or object-name list participates in this check.
    """
    if raw is None:
        return []
    # Saved worlds, in-memory prefetches and unit-test fixtures from before this
    # schema revision do not contain the semantic ledger. They remain executable;
    # every fresh GM response is protected because the current schema requires it.
    raw_steps = raw.get("steps") if isinstance(raw, dict) else None
    if (
        "events" not in raw
        and isinstance(raw_steps, list)
        and raw_steps
        and all(isinstance(step, dict) and "events" not in step for step in raw_steps)
    ):
        return []
    sequence = _normalize_action_sequence(raw)
    steps = sequence.get("steps") or []
    if not 1 <= len(steps) <= 3:
        return ["the action sequence must have 1-3 short video steps"]

    errors: list[str] = []
    master_events = sequence.get("events") or []
    if "events" in raw:
        master_ids = [event.get("id") for event in master_events]
        if not master_events:
            errors.append("the full user intent lists no atomic visible events")
        if any(not isinstance(event_id, int) or event_id <= 0 for event_id in master_ids):
            errors.append("the event list has an invalid id")
        if len(master_ids) != len(set(master_ids)):
            errors.append("event ids must be unique")
        if master_ids != list(range(1, len(master_ids) + 1)):
            errors.append("event ids must count up from 1 without gaps")
        assigned = [event_id for step in steps for event_id in (step.get("event_ids") or [])]
        if assigned != master_ids:
            errors.append(_LEDGER_ASSIGNMENT_ERROR)
    seen_results: set[str] = set()
    for step_index, step in enumerate(steps, 1):
        events = step.get("events") or []
        if not events:
            errors.append(f"step {step_index} lists no atomic visible events")
            continue
        total = 0.0
        for event_index, event in enumerate(events, 1):
            if not event.get("description"):
                errors.append(f"step {step_index}, event {event_index} has no description")
            result = _clean_motion_text(event.get("visible_result"))
            if not result:
                errors.append(f"step {step_index}, event {event_index} has no visible result")
            seconds = float(event.get("estimated_seconds") or 0)
            if seconds <= 0:
                errors.append(f"step {step_index}, event {event_index} has no plausible duration")
            total += max(0.0, seconds)
            result_key = re.sub(r"[\W_]+", "", result.casefold())
            if result_key and result_key in seen_results:
                errors.append(f"step {step_index} repeats a visible result that already happened: {result}")
            if result_key:
                seen_results.add(result_key)
        if total > float(segment_seconds) + 0.25:
            errors.append(f"step {step_index} needs about {total:g}s, over the {segment_seconds}s budget")
    return errors


def _action_sequence_repair_hint(errors: list[str], segment_seconds: int) -> str:
    """Ask the GM to re-plan its own semantic ledger once before any media work."""
    details = "; ".join(errors)
    return _say(
        "\n\n【动作时长规划校正】你上一次返回的结构化动作计划无法执行："
        f"{details}。请重新理解用户的完整意图，把每个必须在画面中看见的动作或状态"
        "变化列为 events，再按因果顺序分配到 1–3 个短视频步骤：每一步的 event_ids 只含"
        "一个 id，每个 event 恰好用一次，events 的数量等于步骤数。每一步只拍这一个 event，"
        f"其 estimated_seconds 不超过 {segment_seconds} 秒；装不下就拆到下一步，"
        "不要压缩物理过程，不要重复同一动作或终态。reply、beat、motion_plan、"
        "state_updates 和任务状态只能描述重新规划后的第一步。",
        # This hint used to say events "may share a step", which the check rejects (one event id
        # per step): the model did exactly that and the same error came back three times.
        "\n\n[Action timing correction] The structured action plan you returned cannot be "
        f"executed: {details}. Re-read the user's full intent, list every action or state change "
        "that must be seen on screen as an event, then assign the events in causal order to 1-3 "
        "short video steps: each step's event_ids holds exactly one id, every event is used "
        "exactly once, and there are as many events as steps. Each step films only its own "
        f"event, whose estimated_seconds is at most {segment_seconds}; what does not fit goes to "
        "the next step. Do not compress physical processes and do not repeat an action or end "
        "state. reply, beat, motion_plan, state_updates and quest updates describe only the "
        "first step of the new plan.",
    )


def _prepared_motion_errors(world: World, decision: dict[str, Any], user_text: str) -> list[str]:
    # Legacy fixtures may omit this field. Fresh model schemas always require it.
    if "motion_plan" not in decision:
        return []
    preview = copy.deepcopy(dict(decision))
    plan = _normalize_motion_plan(preview["motion_plan"])
    plan = _advance_completed_transition(world, preview, plan, user_text)
    plan = _reconcile_dialogue_motion(world, preview, plan, user_text)
    return _motion_plan_errors(_repair_motion_plan(plan, preview))


def _repair_directive_physics(
    world: World,
    decision: dict[str, Any],
    user_text: str,
    errors: list[str],
    *,
    base: str = "",
) -> dict[str, Any]:
    """Correct only geometry; a missing field must not regenerate the whole story."""
    schema = _motion_replan_schema(
        {
            "type": "object",
            "properties": {"motion_plan": copy.deepcopy(MOTION_PLAN_SCHEMA)},
            "required": ["motion_plan"],
            "additionalProperties": False,
        }
    )
    inactive, active = schema["properties"]["motion_plan"]["anyOf"]
    schema = {
        "anyOf": [
            {
                "type": "object",
                "properties": {"has_body_motion": {"const": moving}, "motion_plan": variant},
                "required": ["has_body_motion", "motion_plan"],
                "additionalProperties": False,
            }
            for moving, variant in ((False, inactive), (True, active))
        ]
    }
    scene = {
        key: copy.deepcopy(decision.get(key))
        for key in (
            "beat",
            "dialogue",
            "place",
            "carry",
            "state_updates",
            "subjects",
            "new_subjects",
            "implicit_player",
            "new_places",
            "detailed_description",
        )
    }
    sequence = decision.get("action_sequence") or {}
    scene["action_sequence"] = {
        "events": sequence.get("events"),
        "steps": [
            {key: value for key, value in step.items() if key not in ("kind", "phase", "relation_change")}
            for step in sequence.get("steps", [])
            if isinstance(step, dict)
        ],
    }
    result = llm.chat(
        system=(
            "Repair only the physical annotation of the supplied immutable scene. "
            "First answer has_body_motion: does an ON-SCREEN character or object actually "
            "move/change a physical relation in this shot? A stationary person plus a "
            "moving camera is false. Off-screen footsteps or voices alone are false. "
            "The user goal, action instructions, dialogue, assets, location, state updates "
            "and video description must remain unchanged. Never invent a new event or "
            "move a character to another place to satisfy a missing field. Camera "
            "dolly/pan/zoom is not character locomotion: if only the camera moves, use "
            "interaction_type=phase=relation_change=none. The invalid_annotation and its "
            "classification labels are NOT immutable facts: correct them from the scene's "
            "actual words. A camera approaching a person does not make that person move. "
            "Preserve real bodily motion, "
            "including getting up or crossing a boundary, with complete start/path/contact/"
            "end geometry. Use the existing participant names. Return has_body_motion "
            "and its matching motion_plan; nothing else."
        ),
        user=_pretty(
            {
                "user_goal": user_text,
                "errors": errors,
                "snapshot": world.snapshot(),
                "immutable_scene": scene,
                "invalid_annotation": decision.get("motion_plan"),
            }
        ),
        schema=schema,
        model=config.GM_MODEL,
        thinking=0,
        temperature=0.2,
        base=base,
        tag="motion_field_repair",
    )
    raw = result.get("motion_plan") if isinstance(result, dict) else None
    if not isinstance(raw, dict) or not set(MOTION_PLAN_SCHEMA["required"]).issubset(raw):
        raise ValueError("Motion repair returned an incomplete plan")
    fixed = copy.deepcopy(dict(decision))
    fixed["motion_plan"] = _normalize_motion_plan(raw)
    if not isinstance(result.get("has_body_motion"), bool) or result["has_body_motion"] != _motion_active(
        fixed["motion_plan"]
    ):
        raise ValueError("Motion repair disagrees with the plan")
    remaining = _prepared_motion_errors(world, fixed, user_text)
    if remaining:
        raise ValueError("Invalid motion plan: " + "; ".join(remaining))
    return fixed


def _ensure_goal_relations(sequence: dict[str, Any]) -> None:
    """Segment count and relations come entirely from the director's structured judgement; no steps are added from user text.

    In natural language a state and an action can share a word. Adding relations by regex would turn
    "walk through the already open entrance" into "open it again and walk through again". Missing
    steps are fixed through the prompt and structured output; this function remains only for
    existing callers and tests.
    """
    return


def _complete_sequence_step_fields(sequence: dict[str, Any]) -> None:
    """Fill visible end states and continuity anchors for later steps so old cached decisions do not break auto-continue."""
    for index, step in enumerate(sequence.get("steps") or []):
        if not isinstance(step, dict) or index == 0:
            continue
        relation = step.get("relation_change", "none")
        if not step.get("end_state"):
            step["end_state"] = _GOAL_RELATION_END.get(
                relation,
                "The character keeps their current state and finishes this action",
            )
        if not step.get("continuity_anchor"):
            step["continuity_anchor"] = (
                "Same person, same place, and the last visible state of the previous clip"
            )


_CONFIGURATION_ACTION = re.compile(
    r"打开|推开|拉开|掀开|关上|关闭|合上|"
    r"\b(?:opens|closes)\b|"
    r"\b(?:opening|closing)\b\s+(?:(?:the|a|an|his|her|its)\s+)?"
    r"(?:door|gate|hatch|lid|panel)\b|"
    r"\b(?:pulls?|pulling|pushes?|pushing|swings?|swinging|slides?|sliding)\b"
    r".{0,28}\b(?:open|closed|outward|inward|aside)\b",
    re.I,
)


_BOUNDARY_CROSSING_ACTION = re.compile(
    r"进入|进去|走进|走入|踏入|踏进|跨入|迈入|迈进|穿过|跨过|跨进|"
    r"退出|退回|返回|回到|走出|走回|出来|撤出|出去|离开|"
    r"\b(?:enter(?:s|ed|ing)?)\b|"
    r"\b(?:cross(?:es|ed|ing)?|passes?|passing|steps?|stepping)\b"
    r".{0,36}\b(?:through|across|inside|into|threshold|boundary|interior)\b|"
    r"\b(?:through|across)\b.{0,24}\b(?:door|gate|hatch|opening|threshold|boundary)\b",
    re.I,
)


# The director sometimes writes "already inside another space" as a continue in the current place.
# Instead of guessing vehicle or room types, only the high-confidence signal "crosses a boundary +
# names a destination space" is detected; a hit makes the director judge the place again from the
# snapshot, so no video is silently generated in the old scene.
_SCENE_ENTRY_ACTION = re.compile(
    r"进入|进去|走进|走入|踏入|踏进|跨入|迈入|迈进|进到|跨进|跨过门槛|穿过|"
    r"退出|退回|返回|回到|走出|走回|出来|撤出|出去|离开|"
    r"\b(?:enter(?:s|ed|ing)?|go(?:es|ing)?\s+(?:inside|into)|"
    r"(?:go|walk|step|move|cross|pass)(?:s|es|ed|ing)?\s+"
    r"(?:through|across|past|over|inside|into)|"
    r"cross(?:es|ed|ing)?\s+into|"
    r"(?:cross(?:es|ed|ing)?|pass(?:es|ed|ing)?|step(?:s|ping)?|"
    r"walk(?:s|ing)?|move(?:s|d|ing)?)\s+.{0,48}?\s+"
    r"(?:through|across|past|over|inside|into)\b|exit)\b",
    re.I,
)


_SCENE_TAIL_BOUNDARY = re.compile(
    r"(?:并且?|并|然后|随后|接着|同时|再|之后|以后|开始|继续|来到|抵达|到达|"
    r"\b(?:and|then|before|while|as|after|to)\b)",
    re.I,
)


_SPACE_CONNECTOR = re.compile(
    r"^(?:到|向|往|从|进入|走进|穿过|the|a|an|to|into|inside|through)\s+",
    re.I,
)


def _stage_declared_places(
    world: World,
    directive: dict[str, Any],
    pending: list[_PendingImage],
) -> set[str]:
    """Register places the model explicitly wrote into ``new_places``; no guessing of asset types."""
    raw_places = [x for x in (directive.get("new_places") or []) if isinstance(x, dict)]

    staged: set[str] = set()
    seen_pending = {p.key for p in pending if p.kind == "place"}
    for candidate in raw_places:
        name = _clean_motion_text(candidate.get("name"))
        if not name:
            continue
        prompt = _clean_motion_text(candidate.get("image_prompt")) or (
            f"A wide establishing shot of {name}, showing its spatial layout, architecture, "
            "surfaces, and primary light sources."
        )
        prompt = _english_asset_prompt(
            prompt,
            fallback="A wide establishing shot of the location, showing spatial layout, surfaces, and primary light sources.",
        )
        brief = cap(_clean_motion_text(candidate.get("brief")), 60)
        if name not in world.places:
            world.add_place(
                name,
                "",
                prompt,
                _clean_motion_text(candidate.get("main_scene")) or name,
                brief=brief,
            )
        elif not world.places[name].get("ref"):
            # An old cache may have registered the place without a reference image yet; draw it once
            # in this beat.
            world.places[name].update({"desc": prompt, "brief": brief or world.places[name].get("brief", "")})
        if name not in seen_pending and not world.places[name].get("ref"):
            pending.append(_PendingImage("place", name, prompt, world.visual_style, brief))
            seen_pending.add(name)
        staged.add(name)

    if staged:
        directive["subjects"] = [
            value for value in (directive.get("subjects") or []) if _clean_motion_text(value) not in staged
        ]
    directive["new_places"] = [
        candidate for candidate in raw_places if _clean_motion_text(candidate.get("name")) in staged
    ]
    return staged


def _same_space_label(left: str, right: str) -> bool:
    """Compare only normalized full place names; no fixed list of place words to guess the same space."""

    def norm(value: str) -> str:
        return re.sub(r"[\s、，。！？!?,.:：;；]", "", value or "").casefold()

    a, b = norm(left), norm(right)
    return bool(a and b and a == b)


def _previous_motion_plan(world: World) -> dict[str, Any]:
    head = world.head_node
    if head is None:
        return dict(_EMPTY_MOTION_PLAN)
    return _normalize_motion_plan((head.directive or {}).get("motion_plan"))


def _advance_completed_transition(
    world: World,
    directive: dict[str, Any],
    plan: dict[str, Any],
    request_text: str,
) -> dict[str, Any]:
    """Consume the approach step the previous clip completed and go straight to the relation the user asks for now.

    Not filtering the director's output, but feeding the clip tree's completed state into the action
    compiler: if the previous clip ended outside the entrance, the first step of "enter the inner
    hall" can only be crossing, not approaching again.
    """
    raw_sequence = directive.get("action_sequence")
    # An explicit event ledger has already passed duration/goal review. A prior
    # distance action does not prove its endpoint was reached (e.g. moving toward
    # a bridge). Never delete ledger events using this legacy step-only shortcut.
    if isinstance(raw_sequence, dict) and raw_sequence.get("events"):
        return plan
    raw_steps = raw_sequence.get("steps") if isinstance(raw_sequence, dict) else None
    semantic_relations = {
        _clean_motion_text(step.get("relation_change")).lower()
        for step in (raw_steps or [])
        if isinstance(step, dict)
    }
    if "containment" not in semantic_relations or plan.get("relation_change") not in {"none", "distance"}:
        return plan
    head = world.head_node
    previous = _previous_motion_plan(world)
    if head is None or previous.get("relation_change") != "distance":
        return plan
    current = head.place or world.here
    destination = _clean_motion_text(directive.get("place"))
    if not destination or _same_space_label(current, destination):
        steps = raw_steps
        destination = next(
            (
                _clean_motion_text(step.get("place"))
                for step in (steps or [])
                if isinstance(step, dict)
                and step.get("relation_change") == "containment"
                and _clean_motion_text(step.get("place"))
            ),
            "",
        )
    if not destination or _same_space_label(current, destination):
        return plan

    steps = raw_steps
    if isinstance(steps, list):
        # Remove the completed distance step from the head; the order of the remaining steps is
        # still the director's.
        remaining = [step for step in steps if isinstance(step, dict)]
        remaining = [step for step in remaining if step.get("relation_change") != "distance"]
        containment = next(
            (step for step in remaining if step.get("relation_change") == "containment"),
            None,
        )
        if containment is None:
            containment = _safe_relation_step("containment")
        raw_sequence["steps"] = [containment] + [step for step in remaining if step is not containment]
        directive.setdefault("_server_repairs", []).append(
            "approach finished in the previous clip; this step crosses straight in"
        )

    safe = _safe_motion_for_relation({**plan, "target": destination}, "containment")
    directive["mode"] = "cut"
    directive["transition"] = True
    return safe


def _dialogue_step_from(step: dict[str, Any]) -> dict[str, Any]:
    """Compile an explicitly typed dialogue step without reading action words."""
    instruction = _clean_motion_text(step.get("instruction"))
    if not instruction:
        instruction = _say(
            "停稳后与眼前的人自然交谈", "Settle and talk naturally with the person in front of you"
        )
    return {
        **step,
        "instruction": instruction,
        "kind": "dialogue",
        "phase": "none",
        "relation_change": "none",
        "end_state": "The two keep talking naturally",
        "continuity_anchor": "Both keep their positions and facing",
    }


def _reconcile_dialogue_motion(
    world: World,
    directive: dict[str, Any],
    plan: dict[str, Any],
    request_text: str,
) -> dict[str, Any]:
    """Separate dialogue from contact using the GM's structured ``kind`` only.

    Absence from an action-word list is not evidence of dialogue. Object impact,
    tool use and body contact can all legitimately use relation=contact. Only a
    step explicitly typed as dialogue may clear a contact motion plan.
    """
    if plan.get("relation_change") != "contact":
        return plan
    raw_sequence = directive.get("action_sequence")
    raw_steps = raw_sequence.get("steps") if isinstance(raw_sequence, dict) else None
    if not isinstance(raw_steps, list) or not raw_steps:
        return plan
    first = raw_steps[0] if isinstance(raw_steps[0], dict) else {}
    if _clean_motion_text(first.get("kind")).lower() != "dialogue":
        return plan
    raw_sequence["steps"] = [
        _dialogue_step_from(step)
        if isinstance(step, dict) and _clean_motion_text(step.get("kind")).lower() == "dialogue"
        else step
        for step in raw_steps
    ]
    directive.setdefault("_server_repairs", []).append(
        "cleared a contact plan contradicting the GM's dialogue kind"
    )
    return dict(_EMPTY_MOTION_PLAN)


def _annotate_sequence_context(
    world: World,
    directive: dict[str, Any],
    sequence: dict[str, Any],
    plan: dict[str, Any],
    request_text: str,
) -> set[str]:
    """Write the end place and visible subjects of each segment, forming an executable stage plan."""
    steps = sequence.get("steps") or []
    current = world.head_node.place if world.head_node else world.here
    # Place semantics only read the model's structured fields. This used to cut the phrase after
    # "leave / enter" from the user's text and override the model's place; "leave the post office to
    # look for clues" became a nonexistent place. The code only chains the per-step places the model
    # gave.
    directive_place = _clean_motion_text(directive.get("place"))
    cursor = current
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        explicit_place = _clean_motion_text(step.get("place"))
        if explicit_place:
            cursor = explicit_place
        elif index == 0 and directive_place:
            cursor = directive_place
        step["place"] = cursor

    first = steps[0] if steps and isinstance(steps[0], dict) else {}
    if first.get("place") and first["place"] != directive_place:
        # The current step is more specific than the whole-command destination.
        # A rescue in the courtyard must not bind bridge references or drop its
        # latent merely because a later step will reach the bridge.
        directive["place"] = first["place"]
        directive["mode"] = "continue" if first["place"] == current else "cut"
        if first["place"] == current:
            directive["transition"] = False
        directive.setdefault("_server_repairs", []).append(
            "this shot uses the first step's place; later destinations stay in the plan"
        )

    declared_visible = first.get("visible_subjects") if isinstance(first, dict) else None
    names = [
        str(value).strip()
        for value in (
            declared_visible
            if isinstance(declared_visible, list) and declared_visible
            else directive.get("subjects") or []
        )
        if str(value).strip()
    ]
    deferred: set[str] = set()
    destination = next(
        (
            _clean_motion_text(step.get("place"))
            for step in steps
            if isinstance(step, dict)
            and step.get("relation_change") == "containment"
            and _clean_motion_text(step.get("place"))
        ),
        "",
    )
    if (
        first.get("relation_change") == "distance"
        and destination
        and not _same_space_label(current, destination)
    ):
        declared = [x for x in (directive.get("new_subjects") or []) if isinstance(x, dict)]
        # Delayed appearances follow only the director's structured target/targets; no more scanning
        # the user's sentence, reply or brief for words like "waiting / inner hall / inside" to
        # guess whether someone is far away.
        structured_targets = {
            _clean_motion_text(plan.get("target")),
            *(
                _clean_motion_text(value)
                for step in steps
                if isinstance(step, dict)
                for value in (step.get("targets") or [])
            ),
        }
        target_keys = {re.sub(r"\s+", "", value).casefold() for value in structured_targets if value}
        for candidate in declared:
            name = _clean_motion_text(candidate.get("name"))
            if not name:
                continue
            name_key = re.sub(r"\s+", "", name).casefold()
            if name_key in target_keys:
                deferred.add(name)
        if deferred:
            names = [name for name in names if _clean_motion_text(name) not in deferred]
            directive.setdefault("_server_repairs", []).append(
                "subjects of a later place keep their assets but appear after entering: "
                + ", ".join(sorted(deferred))
            )
    directive["subjects"] = names
    return deferred


_DISTANCE_ACTION = re.compile(
    r"走近|走到|走向|跑向|靠近|接近|缩短距离|"
    r"\b(?:approach(?:es|ed|ing)?|walks?|walking|runs?|running|moves?|moving)\b"
    r".{0,28}\b(?:toward|towards|up to|closer|near)\b|\bcloses? the (?:gap|distance)\b",
    re.I,
)


_SUPPORT_ACTION = re.compile(
    r"坐下|坐进|躺下|骑上|爬上|登上|转移重心|"
    r"\b(?:sits?|sitting|lies?|lying|mounts?|mounting)\b|"
    r"\b(?:climbs?|climbing|transfers?|transferring)\b.{0,24}"
    r"\b(?:onto|weight|support)\b",
    re.I,
)


_POSSESSION_ACTION = re.compile(
    r"拿起|拿取|捡起|拾起|抓起|接过|取出|取物|拿到|取得|"
    r"\b(?:picks?|picking)\b.{0,8}\bup\b|"
    r"\b(?:takes?|taking|lifts?|lifting)\b.{0,24}\b(?:object|item|tool|weapon|cup|key)\b",
    re.I,
)


_PRIMARY_ACTION_PATTERNS = {
    "distance": _DISTANCE_ACTION,
    "configuration": _CONFIGURATION_ACTION,
    "containment": _BOUNDARY_CROSSING_ACTION,
    "support": _SUPPORT_ACTION,
    "possession": _POSSESSION_ACTION,
}


def _safe_relation_step(relation: str, anchor: str = "") -> dict[str, str]:
    return {
        "instruction": _goal_relation_instruction(relation),
        "phase": _GOAL_RELATION_PHASE[relation],
        "relation_change": relation,
        "end_state": _GOAL_RELATION_END[relation],
        "continuity_anchor": anchor
        or "Same person, same rigid boundary, same direction of travel, and the last visible state of the previous clip",
    }


def _safe_motion_for_relation(
    plan: dict[str, Any],
    relation: str,
) -> dict[str, Any]:
    """Trim a mixed action to a single relation so a wrong path never reaches the final video prompt as is."""
    safe = dict(plan)
    if relation == "distance":
        safe.update(
            {
                "interaction_type": "locomotion",
                "phase": "approach",
                "relation_change": "distance",
                "path": (
                    "the actor moves along the continuous visible ground toward the target "
                    "and stops outside it before any contact or interaction begins"
                ),
                "contact_points": "",
                "end_state": _GOAL_RELATION_END["distance"],
                "visual_evidence": [],
                "must_show": ["the actor's full body, both feet, the continuous ground path, and the target"],
            }
        )
    elif relation == "configuration":
        safe.update(
            {
                "interaction_type": "manipulation",
                "phase": "transform",
                "relation_change": "configuration",
                "path": (
                    "the acting hand first reaches the visible handle or movable part, then "
                    "moves only that part along its hinge, joint, or track until fully open; "
                    "the actor's entire body remains on the starting side of the boundary"
                ),
                "contact_points": "the acting hand on the visible handle or movable part",
                "end_state": (
                    "the movable boundary is fully open and the actor remains entirely on "
                    "the starting side, outside the destination space"
                ),
                "visual_evidence": [],
                "must_show": [
                    "the complete movable part and its hinge, joint, or track",
                    "the acting hand and the actor remaining on the starting side",
                ],
            }
        )
    elif relation == "containment":
        safe.update(
            {
                "interaction_type": "boundary",
                "phase": "transform",
                "relation_change": "containment",
                "path": (
                    "through the already-open passage, the leading body part, torso, and "
                    "trailing feet cross the boundary plane in order"
                ),
                "contact_points": (
                    "both feet alternately supported by the visible surfaces before and after the passage"
                ),
                "end_state": _GOAL_RELATION_END["containment"],
                "visual_evidence": [],
                "must_show": [
                    "the complete open passage, both rigid edges, the actor's full body, "
                    "and both support surfaces"
                ],
            }
        )
    elif relation == "support":
        safe.update(
            {
                "interaction_type": "support",
                "phase": "transform",
                "relation_change": "support",
                "path": (
                    "the actor transfers body weight onto the single visible support surface "
                    "through the named contact points and settles without changing any other relation"
                ),
                "contact_points": "the actor's body and feet on the visible support surfaces",
                "end_state": _GOAL_RELATION_END["support"],
                "visual_evidence": [],
                "must_show": [
                    "the actor's full body, both feet, every load-bearing contact point, and the support"
                ],
            }
        )
    elif relation == "possession":
        safe.update(
            {
                "interaction_type": "manipulation",
                "phase": "transform",
                "relation_change": "possession",
                "path": (
                    "the acting hand reaches the target, closes a visible grip, and only then "
                    "moves the target as one rigid object with that same hand"
                ),
                "contact_points": "the acting hand's visible grip on the target",
                "end_state": _GOAL_RELATION_END["possession"],
                "visual_evidence": [],
                "must_show": ["the acting hand, the complete target, and the grip point"],
            }
        )
    elif relation == "contact":
        if safe.get("interaction_type") == "impact" and safe.get("instrument"):
            # Preserve the GM's concrete path and result.  Replacing them with the
            # body-contact fallback would erase the throw/tool motion we are trying
            # to protect.
            safe.update(
                {
                    "phase": "engage",
                    "relation_change": "contact",
                    "visual_evidence": list(safe.get("visual_evidence") or []),
                    "must_show": list(safe.get("must_show") or []),
                }
            )
        else:
            safe.update(
                {
                    "interaction_type": "contact",
                    "phase": "engage",
                    "relation_change": "contact",
                    "path": (
                        "the actors close only the final contact gap and meet at the named body parts "
                        "while all other silhouettes remain spatially distinct"
                    ),
                    "contact_points": "the named body parts at the intended visible contact point",
                    "end_state": _GOAL_RELATION_END["contact"],
                    "visual_evidence": [],
                    "must_show": ["both actors and the intended contact point"],
                }
            )
    return _normalize_motion_plan(safe)


def _repair_action_sequence_phases(
    sequence: dict[str, Any],
    directive: dict[str, Any] | None = None,
) -> None:
    """Rewrite decidable phase / relation conflicts in the action queue into valid steps.

    action_sequence is the auto-continue queue; a wrong lifecycle phase on one step must not stop
    the whole round. The relation stays and only the phase is recomputed from the action's meaning;
    the repair is saved with the directive so the inspector shows what the compiler did.
    """
    for step in sequence.get("steps") or []:
        if not isinstance(step, dict):
            continue
        relation = _clean_motion_text(step.get("relation_change")).lower()
        phase = _clean_motion_text(step.get("phase")).lower() or "none"
        if relation in _RELATION_DEFAULT_PHASE:
            if relation in _PHASE_RELATIONS.get(phase, set()):
                continue
            # Repair consistency only from the schema's relation / phase enums; never guess
            # "retreat" vs ordinary approach from the instruction's wording.
            target_phase = _RELATION_DEFAULT_PHASE[relation]
            step["phase"] = target_phase
            if directive is not None:
                directive.setdefault("_server_repairs", []).append(
                    f"action step phase repaired: phase={phase}, relation={relation} -> {target_phase}"
                )
        elif relation == "none" and phase not in {"none", "settle"}:
            step["phase"] = "none"
            if directive is not None:
                directive.setdefault("_server_repairs", []).append(
                    f"action step without a relation change adjusted: phase={phase} -> none"
                )


def _action_sequence_errors(
    sequence: dict[str, Any],
    motion_plan: dict[str, Any],
) -> list[str]:
    """The action budget and the current motion plan must describe the same thing, or no video job is sent."""
    errors: list[str] = []
    if not sequence.get("goal"):
        errors.append("the action sequence has no goal")
    steps = sequence.get("steps") or []
    if not 1 <= len(steps) <= 3:
        errors.append("the action sequence must have 1-3 short video steps")
        return errors
    for index, step in enumerate(steps, 1):
        # `end_state` and `continuity_anchor` are only read for later steps: the next beat's
        # instruction is built from steps[1:] by `_sequence_continue_with`, and boundary repair
        # looks at the last step. Nobody reads them on the first step (what goes to the rewriter is
        # motion_plan's path/end_state, see `_apply_action_sequence`), so an empty value must not
        # fail the beat. Requiring them would make the director write two English paragraphs for
        # nothing.
        required = ("instruction",) if index == 1 else ("instruction", "end_state", "continuity_anchor")
        for key in required:
            if not step.get(key):
                errors.append(f"action step {index} has no {key}")
        allowed = _PHASE_RELATIONS.get(step.get("phase", "none"), set())
        if step.get("relation_change") not in allowed:
            errors.append(
                f"action step {index}: phase={step.get('phase')} cannot change "
                f"relation={step.get('relation_change')}"
            )
    first = steps[0]
    if first.get("phase") != motion_plan.get("phase"):
        errors.append("the first action step's phase differs from motion_plan")
    if first.get("relation_change") != motion_plan.get("relation_change"):
        errors.append("the first action step's relation_change differs from motion_plan")
    return errors
