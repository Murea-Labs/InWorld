"""Auditing whether one action fits in one clip, and the narrowed schemas used to replan it."""

from __future__ import annotations

import copy
import re
import math
from typing import Any
from murea_agent import config
from murea_agent.providers import llm
from murea_agent import reqlog
from murea_agent.memory.world import World
from murea_agent.util import _pretty, _say
from murea_agent.director.agent_schema import (
    _MOTION_DETAIL_FIELDS,
    _agent_schema,
)
from murea_agent.shot.definitions import (
    _locked_definitions,
)


_ACTION_DENSITY_AUDIT_SYSTEM = """你是短视频动作时长审计 GM。只判断语义和物理时长，
不续写剧情，不改写用户目标。先识别事件，再判断能否在给定时长内组成同一连续镜头。
事件数不等于视频段数。若一句话要求依次看清多个不同结果，按因果顺序
列为 2–3 个事件；前后只是同一
观察、同一动作或同一终态的重复描述时保持一段。按真实身体、物体运动和画面可读性估时，
不能为了卡进预算而把时间写得不现实。

先区分已有状态与本次变化，再判断是否需要分段。站在某处、坐着、持有物品、保持目光、
天气和背景等可以是动作发生时的持续状态，不代表这次还要走过去、坐下、拿起或转头。
不能给原文补出“先站稳”“收起踉跄”“从移动变静止”等未要求的前置事件。仅有四秒
时长、两个人、多个动词或并列的状态描写，都不是需要两段的证据；说一句话或持续观察
可以占满四秒。只有原目标或计划明确要求发生不同的变化、而且必须依次看见其各自结果，
才拆段。拆出的每个事件必须能从原文找到依据，不得续写动作。

例如：“甲站在火炉旁，乙问他从哪里来”只有询问；“他拿着信看向对方”没有重新拿信；
“她坐在椅子上观察门口”没有坐下过程。上述状态可以与同一动作同时呈现。
同一句问话中，说者开口与听者保持聆听是同时发生的对话，不是先说后听两个结果；
若原文要求先提问再回答，才是两个独立话轮。
定场、运镜、景别变化、对焦和展示背景是同一事件的拍摄方式，不是额外的剧情动作。
“定场镜头展示大厅并聚焦正在惊醒的人”只有人物惊醒这一事件；大厅、钟声和人物可以
在同一个连续镜头里同时呈现，不要拆成“扫过大厅”和“聚焦人物”两个事件。镜头推进
不等于人物走近，视线或画面露出某物不等于人物拿起它。只有原文明确要求先后展示的
独立世界变化才拆分；开场仍要保留真实的起身、走动、拿取等有因果依赖的身体动作。
但“他走到火炉旁站定，然后回答”确有移动和回答；“她坐下，再拿起信”确有坐下和拿信；
“投掷、击碎、倒塌”有各自不可省略的可见结果。这些须按因果链列出事件，不可省略。
continuous_shot 表示所有事件能否在同一场景、同一连续镜头里自然完成。
例如走近并扶起同伴的 3 秒接近 + 4 秒起立，可在 7 秒以上的预算里连续拍完；
提醒对方并抓紧绳索、同一座桥上从入口走到对岸，也不要仅因先后因果就拆视频。
先使用允许的较长时长，只有超出预算、真正换场景或独立叙事镜头才分段。
保留各事件真实时长，continuous_shot=true 也不能省略动作或掩盖终态缺失。
还要检查现有 steps 的最终 end_state 是否完成了用户的完整目标，填 goal_covered。
例如用户要求“通过桥到对岸”，计划却止于“桥中央”，必须为 false；用户只要求
“走到桥中央”则为 true。事件标题声称“通过”不能掩盖实际终态只走了一半。
用户要求发生的变化不能被当成已完成的前提：要求“束缚断裂、人物脱困”，计划却
只写“扶起已经脱困的人”，即使终态用了“脱困”也必须为 false，漏掉了核心动作。
不完整时在 coverage_reason 说明缺少的终态，并在 events 中给出能真正完成目标的
完整行动（可在一段完成就保留一段，需要多段则拆分），不能把剩余动作丢给推荐选项。
返回 events、continuous_shot、goal_covered、coverage_reason 的 JSON。完整保留每个事件的动作、可见结果和真实时长；
不用再写段数或长篇解释，服务端根据事件清单计算段数并记录审计依据。"""


_ACTION_DENSITY_AUDIT_SYSTEM_EN = """You audit how long the actions of a short video take. Judge only meaning
and physical duration; do not continue the story or rewrite the user's goal. First identify
the events, then decide whether they fit in one continuous shot within the given duration.
The number of events is not the number of video segments. If one sentence asks to see several
different results in turn, list them as 2-3 events in causal order; when the parts only
restate the same observation, action or end state, keep one segment. Estimate time from real
body and object motion and on-screen readability; never make a duration unrealistic to fit
the budget.

Separate existing state from change before deciding to split. Standing somewhere, sitting,
holding an object, keeping one's gaze, weather and background can be ongoing states while the
action happens; they do not mean the character still has to walk there, sit down, pick it up
or turn. Do not add unrequested preliminary events such as "steadies first" or "stops
moving". A four-second budget, two people, several verbs or a list of states are not by
themselves evidence for two segments; saying one line or watching can fill four seconds.
Split only when the goal or plan explicitly asks for different changes whose results must be
seen one after another. Every event must be grounded in the original text; never continue the
action.

For example, "A stands by the stove; B asks where he came from" is only the question; "he
looks at the other person holding a letter" does not pick the letter up again; "she sits in a
chair watching the door" has no sitting down. Those states can show alongside the action.
In one question, the speaker talking and the listener listening happen together, not as two
results; only "ask, then answer" is two turns.
Establishing shots, camera moves, shot-size changes, focus and showing the background are how
one event is filmed, not extra story actions. "An establishing shot shows the hall and settles
on a man waking with a start" is one event, the man waking; the hall, the bell and the man can
all appear in one continuous shot. A camera push-in is not a character walking closer, and a
glance revealing something is not picking it up. Split only independent world changes the text
explicitly orders; an opening still keeps real causal body actions such as getting up, walking
and taking things.
But "he walks to the stove, stops, then answers" really has a move and an answer; "she sits
down, then picks up the letter" has sitting and taking the letter; "throw, shatter, collapse"
each have a result that cannot be dropped. List these in causal order; omit none.
continuous_shot says whether all events can happen naturally in one place in one continuous
shot. For example, a 3-second approach plus a 4-second lift of a companion fits a budget of 7
seconds or more; warning someone while gripping a rope, or crossing one bridge from end to end,
should not be split just because one thing follows another.
Use the longest allowed duration first; split only when over budget, on a real change of place
or for an independent narrative shot. Keep each event's real duration; continuous_shot=true
never permits dropping an action or hiding a missing end state.
Also check whether the final end_state of the existing steps completes the user's whole goal,
and fill goal_covered. If the user asked to "cross the bridge to the far side" and the plan
stops at "the middle of the bridge", it is false; if the user only asked to "walk to the middle
of the bridge" it is true. An event titled "crosses" cannot hide an end state halfway across.
A change the user asked for cannot be treated as an already met precondition: for "the bonds
snap and the man breaks free", a plan that only "helps up the man who is already free" is false
even if its end state says "free"; it left out the core action.
When incomplete, explain the missing end state in coverage_reason and give, in events, the full
action that really completes the goal (one segment if it fits, otherwise split); never leave the
rest to the suggested options.
Return JSON with events, continuous_shot, goal_covered and coverage_reason. Keep every event's
action, visible result and real duration in full; do not write a segment count or long
explanations, the server derives the count from the events and records the audit. Write all
text in English."""


def _audit_system() -> str:
    return _ACTION_DENSITY_AUDIT_SYSTEM if config.LANGUAGE == "zh" else _ACTION_DENSITY_AUDIT_SYSTEM_EN


_ACTION_DENSITY_AUDIT_SCHEMA = {
    "type": "object",
    "properties": {
        "goal_covered": {"type": "boolean"},
        "continuous_shot": {"type": "boolean"},
        "coverage_reason": {"type": "string"},
        # Extract actual events first. Their count is derived by the service;
        # asking the model to guess it first can commit it to inventing an event.
        "events": {
            "type": "array",
            "minItems": 1,
            "maxItems": 3,
            "items": {
                "type": "object",
                "properties": {
                    "description": {"type": "string"},
                    "visible_result": {"type": "string"},
                    "estimated_seconds": {"type": "number"},
                },
                "required": ["description", "visible_result", "estimated_seconds"],
            },
        },
    },
    "required": ["events", "continuous_shot", "goal_covered", "coverage_reason"],
}


def _single_event_needs_density_audit(raw: Any, segment_seconds: int) -> bool:
    """Select saturated one-step plans for semantic review, without action words."""
    from murea_agent.guards.action_sequence import _normalize_action_sequence  # circular at import time

    sequence = _normalize_action_sequence(raw)
    events = sequence.get("events") or []
    steps = sequence.get("steps") or []
    if len(events) != 1 or len(steps) != 1:
        return False
    # A short action can still stop before the requested destination. With a
    # long automatic budget (10-20 s), saturation alone misses ordinary 6–12s moves.
    if steps[0].get("kind") == "motion":
        return True
    event = events[0]
    seconds = float(event.get("estimated_seconds") or 0)
    description_size = len(re.sub(r"\s+", "", event.get("description") or ""))
    return seconds >= max(1.0, float(segment_seconds) * 0.9) or description_size >= 42


def _audit_single_event_density(
    raw: Any,
    goal: str,
    segment_seconds: int,
    *,
    base: str = "",
) -> dict[str, Any] | None:
    """Let a compact GM pass review a suspicious one-step causal chain."""
    from murea_agent.guards.action_sequence import _normalize_action_sequence  # circular at import time

    if not _single_event_needs_density_audit(raw, segment_seconds):
        return None
    sequence = _normalize_action_sequence(raw)
    try:
        result = llm.chat(
            system=_audit_system(),
            user=_pretty(
                {
                    "user_goal": goal,
                    "segment_budget_seconds": segment_seconds,
                    "plan_to_audit": {
                        "events": sequence.get("events"),
                        "steps": sequence.get("steps"),
                    },
                }
            ),
            schema=_ACTION_DENSITY_AUDIT_SCHEMA,
            temperature=0.2,
            model=(config.ACTION_AUDIT_MODEL if config.LLM_BACKEND == "gemini" else config.GM_MODEL),
            thinking=0,
            tag="action_density_audit",
            base=base,
        )
    except Exception as exc:  # noqa: BLE001 -- keep the original plan if the audit fails
        reqlog.record("action_density_audit", ok=False, error=f"{type(exc).__name__}: {exc}")
        return None
    if not isinstance(result, dict):
        return None
    events = result.get("events")
    if not isinstance(events, list) or not 1 <= len(events) <= 3:
        reqlog.record("action_density_audit", ok=False, error="the audit returned no 1-3 complete events")
        return None
    # A causal event boundary is not automatically a video boundary. Preserve
    # the auditor's semantic decision and enforce the actual duration budget.
    try:
        seconds = [float(event.get("estimated_seconds") or 0) for event in events]
    except (TypeError, ValueError, AttributeError):
        return None
    if any(not math.isfinite(value) or value <= 0 for value in seconds):
        return None
    continuous = result.get("continuous_shot") is True and sum(seconds) <= segment_seconds
    required = 1 if continuous else len(events)
    result = {**result, "required_segments": required}
    reqlog.record("action_density_audit", ok=True, **result)
    return result


def _density_audit_repair_hint(audit: dict[str, Any], segment_seconds: int) -> str:
    """Lock a semantic audit's distinct events into the one allowed re-plan."""
    events = audit.get("events") or []
    lines = "\n".join(
        f"{index}. {item.get('description')} → {item.get('visible_result')} "
        + _say(f"（约 {item.get('estimated_seconds')} 秒）", f"(about {item.get('estimated_seconds')}s)")
        for index, item in enumerate(events, 1)
        if isinstance(item, dict)
    )
    required = int(audit.get("required_segments") or len(events) or 1)
    if config.LANGUAGE != "zh":
        completion = (
            "The end state does not cover the user's goal: fix the events' visible_result and the "
            "last step's end_state so they show visible evidence that the goal is done. If the "
            "events below still describe only a midway state, follow the user's full goal and the "
            "end-state check; do not copy the incomplete result or leave the rest to the options. "
            if audit.get("goal_covered") is False
            else ""
        )
        return (
            "\n\n[GM action plan review] The plan must be corrected for duration and for the end "
            f"result the user asked for. End-state check: "
            f"{audit.get('coverage_reason') or 'follow the full events in order'}. "
            f"Keep all {len(events)} events below and produce exactly {required} steps. "
            + (
                "Bind every event_id, in order, to the single step; its duration covers all events. "
                if required == 1
                else "Bind one event_id per step. "
            )
            + f"Budget per segment {segment_seconds}s:\n{lines}\n"
            f"{completion}"
            "reply, beat, motion_plan, state_updates, carry and the video prompt show only the "
            "events bound to step 1; results of later steps must not appear early."
        )
    completion = (
        "终态未覆盖用户目标时，必须修正事件的 visible_result 和最后一步 end_state，"
        "明确写出目标已完成的可见证据；若下列事件仍只描述途中状态，以用户完整目标"
        "和终态校验为准，不能照抄不完整结果，也不能把剩余动作丢到推荐选项。"
        if audit.get("goal_covered") is False
        else ""
    )
    return (
        "\n\n【GM 动作计划复核】原计划需要按时长与用户要求的最终结果校正。"
        f"终态校验：{audit.get('coverage_reason') or '按完整事件顺序执行'}。"
        f"必须完整保留下面 {len(events)} 个事件，生成恰好 {required} 个 steps。"
        + (
            "全部 event_ids 按顺序绑定同一步，镜头时长覆盖所有事件之和。"
            if required == 1
            else "每段只绑定一个 event_id。"
        )
        + f"单段预算 {segment_seconds} 秒：\n{lines}\n"
        f"{completion}"
        "当前 reply、beat、motion_plan、state_updates、carry 和视频 prompt 只能表现第 1 个"
        "step 绑定的全部事件；其它 steps 的结果不能提前出现。"
    )


# Re-splitting only changes these fields. The rest (place, assets, mainline, quests, options…) does
# not depend on the number of segments; the first answer still holds and need not be rewritten.
#
# A replan took about 3.4 s / ~1400 tokens, of which only these (~650 tokens) actually change.
_REPLAN_FIELDS = (
    "reply",
    "beat",
    "action_sequence",
    "motion_plan",
    "dialogue",
    "state_updates",
    "subject_definitions",
    "summary",
    "detailed_description",
    "overall_soundscape",
    "non_diegetic_music",
    "motion_phase",
    "motion_relation_change",
    "motion_target",
    "motion_framing",
)


def _density_replan_schema(world: World, audit: dict[str, Any]) -> dict[str, Any]:
    """Make the accepted audit's events and ordered step assignments obligatory.

    Used only for an OpenAI constrained replan. A prose hint alone allowed the
    model to list two events but emit only the first step, aborting world opening.
    Physical realization and asset choices remain model decisions; downstream
    semantic/physics checks still run on the complete response.
    """
    schema = _agent_schema(bool(_locked_definitions(world)))
    events = [
        {"id": index, **{key: item[key] for key in ("description", "visible_result", "estimated_seconds")}}
        for index, item in enumerate(audit["events"], 1)
    ]
    sequence = schema["properties"]["action_sequence"]
    # ⚠️ **Constrain only the structure; never put event content into the schema as const.**
    #
    # The previous version used `{"const": events}`, putting the events' text into the schema, so
    # every beat had a unique grammar that the constrained decoder **compiled every time** and never
    # cached.
    #
    # Measured on one server, changing only the grammar:
    #
    #     plain schema (same every beat)      0.86 → 0.39 → 0.36 → 0.36s
    #     const events (different each beat)  3.71 → 3.55 → 3.63 → 3.66s   ← compiled every time
    #     same const schema repeated          3.70 → 0.38 → 0.38s          ← fast from the second call
    #
    # With only structural constraints (how many events, how many segments, which goes where), the
    # grammar is identical across beats and compiled once.
    #
    # Content is no longer enforced by the grammar but by two existing checks:
    # `_action_sequence_semantic_errors` rechecks the ledger, and "the segment count must match the
    # audit" below. The prompt still carries the audit's event text (`_density_audit_repair_hint`).
    sequence["properties"]["events"] = {
        **copy.deepcopy(sequence["properties"]["events"]),
        "minItems": len(events),
        "maxItems": len(events),
    }
    steps = sequence["properties"]["steps"]
    item = steps.pop("items")
    fixed_steps = []
    groups = (
        [list(range(1, len(events) + 1))]
        if int(audit.get("required_segments") or len(events)) == 1
        else [[event["id"]] for event in events]
    )
    for event_ids in groups:
        step = copy.deepcopy(item)
        step["properties"]["event_ids"] = {"const": event_ids}
        # The replan path is **stricter than usual**: after splitting into two segments, who moves
        # in each must be explicit, the second segment's actor cannot be vague. Normally `actors` is
        # empty 78% of the time and not required (see ACTION_SEQUENCE_SCHEMA), but here it is added
        # back explicitly.
        step["required"] = list(dict.fromkeys([*step["required"], "actors"]))
        fixed_steps.append(step)
    steps.update(minItems=len(groups), maxItems=len(groups), prefixItems=fixed_steps, items=False)
    # ⚠️ Ask only for the fields that change. The caller merges the result into the first decision instead of replacing it.
    keep = [k for k in schema["properties"] if k in _REPLAN_FIELDS]
    schema["properties"] = {k: schema["properties"][k] for k in keep}
    schema["required"] = [
        k for k in schema["properties"] if k in set(schema.get("required") or []) | set(_REPLAN_FIELDS)
    ]
    return schema


def _motion_replan_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Require coherent inactive/active physics in the initial response as well."""
    schema = copy.deepcopy(schema)
    motion = schema["properties"]["motion_plan"]
    if "anyOf" in motion:
        return schema
    inactive, active = copy.deepcopy(motion), copy.deepcopy(motion)
    for key in ("interaction_type", "phase", "relation_change"):
        inactive["properties"][key] = {"const": "none"}
    # Without body movement those 11 fields are empty **by definition**; the model need not even
    # write the keys.
    #
    # In 300 recorded beats (166 static / dialogue): `actors` `instrument` `path`
    # `visual_evidence` `must_show` were empty in 166/166 static beats, `start_state` `end_state`
    # `camera_side` `invariants` in 164–165/166, and almost always filled in the 134 action beats.
    # So this is not cutting fields but not writing empty values.
    #
    # It saves about 90 tokens, 12% of a static beat's output. `_normalize_motion_plan` starts from
    # `_EMPTY_MOTION_PLAN` and uses `.get()`, so a missing key equals an empty string.
    # `additionalProperties` is off so it **cannot** write them; otherwise the model would keep
    # filling them out of habit.
    for key in _MOTION_DETAIL_FIELDS:
        inactive["properties"].pop(key, None)
    inactive["required"] = [key for key in inactive["required"] if key not in _MOTION_DETAIL_FIELDS]
    inactive["additionalProperties"] = False
    for key in ("interaction_type", "phase"):
        active["properties"][key]["enum"] = [
            item for item in active["properties"][key]["enum"] if item != "none"
        ]
    active["properties"]["framing"]["enum"] = [
        item for item in active["properties"]["framing"]["enum"] if item != "none"
    ]
    for key in ("target", "start_state", "path", "end_state", "camera_side"):
        active["properties"][key]["minLength"] = 1
    for key in ("actors", "visual_evidence", "invariants"):
        active["properties"][key]["minItems"] = 1
    schema["properties"]["motion_plan"] = {"anyOf": [inactive, active]}
    return schema
