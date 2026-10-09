"""Rolling scene goals and comparable cooperative tactics, all still plans."""

import copy
import re

from murea_agent.coop import prompts_en
from murea_agent.util import _say

PURPOSES = ("approach", "discover", "opportunity", "breakthrough", "transition", "closure", "observe")
_DISPLAY_TEXT = {
    "type": "string",
    "description": _say(
        "使用简体中文完整表达，角色专名可保留原文；不要整句使用英文。", prompts_en.DISPLAY_TEXT
    ),
}
PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "purpose": {"type": "string", "enum": list(PURPOSES)},
        "change_reason": dict(_DISPLAY_TEXT),
        "steps": {
            "type": "array",
            "minItems": 2,
            "maxItems": 3,
            "items": {
                "type": "object",
                "properties": {
                    k: {"type": "string"} if k == "id" else dict(_DISPLAY_TEXT)
                    for k in (
                        "id",
                        "goal",
                        "prerequisite",
                        "cooperation",
                        "completion_evidence",
                        "if_blocked",
                    )
                },
                "required": [
                    "id",
                    "goal",
                    "prerequisite",
                    "cooperation",
                    "completion_evidence",
                    "if_blocked",
                ],
            },
        },
    },
    "required": ["purpose", "change_reason", "steps"],
}
TACTIC_SCHEMA = {
    "type": "object",
    "properties": {
        "technique": {
            "type": "string",
            "enum": [
                "move",
                "ranged_attack",
                "melee_attack",
                "defend",
                "distract",
                "interact",
                "inspect",
                "recover",
            ],
        },
        "target": {"type": "string"},
        "route": {"type": "string"},
        "role": {"type": "string"},
    },
    "required": ["technique", "target", "route", "role"],
}

PLANNING_INSTRUCTION = """\n先设计 short_term_plan，再由其第一个小目标决定 objective 和预告。
short_term_plan 是当前阶段未来 2–3 个可调整的小目标，不是已经发生的剧情，不能当成事实。
purpose 选择 approach（接近）、discover（发现）、opportunity（创造机会）、breakthrough（突破）、transition（转场）、closure（收尾）或 observe（确认未知）。
只有 id 使用稳定英文标识，purpose 使用给定枚举。所有展示内容 change_reason、goal、prerequisite（所需条件）、cooperation（两人如何分工）、completion_evidence（单段视频能看清的变化）、if_blocked（受阻时具体替代尝试）必须用简体中文完整表达；Mira、Aren 等角色专名可保留原文。
即使 previous_short_term_plan 或历史证据是英文，也用中文编写本次计划，保留原意与稳定 id。
第一步只能依赖当前已确认条件；未知出口、击中、抵达等只能作为后续待验证条件。一次只执行第一步；后续步骤须等观察确认条件，再滚动选择。
previous_short_term_plan 是上次采用的计划。结合实际结果，延续尚有进展的目标，跳过已确认完成的小目标；受阻就更换路线、目标或协作关系。
change_reason 简述本次延续/调整的依据。director_feedback 中 stalled 表示连续没有进展，blocked 表示受阻，uncertain 表示观察不清；后者不等于失败。
stalled/blocked 时第一步必须采用与最近无效尝试不同的办法；uncertain 时优先设计容易看清结果的单一动作，不凭空确认成功。
不要每轮为了新颖而增加攻击；预告可以提供接近、转场、配合的机会，也可为 none。即使同类事件继续，只要确有新增推进也可以延续。
角色、装备和已确认地形保持一致。计划可以尝试未来变化，不可声称它已经发生。
"""
PLANNING_INSTRUCTION = _say(PLANNING_INSTRUCTION, prompts_en.PLANNING_INSTRUCTION)
REACTION_INSTRUCTION = """\n如果 target_forecast 有 short_term_plan，本轮只执行其第一步，后续步骤尚未满足条件，不能塞进同一段动作。
每人同时输出 tactic：technique 使用给定枚举，target 是具体目标的稳定称呼，route 是路径/终点（不移动写原地），role 是协作职责。
这些字段必须描述 action 真正执行的动作。同一目标、路线和职责沿用 recent_tactics 中的称呼，不靠改名规避重复。
如弓手负责牵制、法师负责接近机关，两人应推进同一个小目标；下一次可随实际站位交换职责，不能永远一人射箭另一人撑盾。
director_feedback 为 stalled/blocked 时，不再重复无效的目标+路径+战术组合；为 uncertain 时提高动作可观察性。正在产生实际进展的配合可以继续。
"""
REACTION_INSTRUCTION = _say(REACTION_INSTRUCTION, prompts_en.REACTION_INSTRUCTION)


def feedback(room):
    progress = room.get("story_progress") or {}
    active = progress.get("active_stage_id") or (room.get("story_outline") or {}).get("active_stage_id")
    blocked = (progress.get("stages", {}).get(active) or {}).get("status") == "blocked"
    assessment = progress.get("assessment", "unknown")
    return {
        "mode": "blocked"
        if blocked
        else "uncertain"
        if assessment == "unknown"
        else "stalled"
        if progress.get("stagnation_rounds", 0) >= 2
        else "continue",
        "next_gap": progress.get("next_gap", ""),
        "player_outcomes": copy.deepcopy(progress.get("player_outcomes", [])),
        "stagnation_rounds": progress.get("stagnation_rounds", 0),
    }


def recent_tactics(room):
    records = []
    for row in room.get("rounds", [])[-6:]:
        adopted = (row.get("guidance") or {}).get("tactics", [])
        actions = {m.get("user_id"): m.get("action") for m in row.get("inputs", [])}
        records.append(
            {
                "round": row.get("round"),
                "assessment": (row.get("story_progress") or {}).get("assessment", "unknown"),
                "tactics": [
                    copy.deepcopy(t) for t in adopted if actions.get(t.get("user_id")) == t.get("action")
                ],
                "actions": actions,
            }
        )
    return records


class CompletedPlanError(ValueError):
    """The proposed step has already been confirmed complete."""


def validate_plan(raw, stage_id, progress=None):
    # Missing extensions from old/offline responses do not break an adventure.
    if raw is None:
        return None
    if not isinstance(raw, dict) or raw.get("purpose") not in PURPOSES:
        raise ValueError("the short-term plan has no clear story purpose")
    steps = raw.get("steps")
    if not isinstance(steps, list) or not 2 <= len(steps) <= 3:
        raise ValueError("the short-term plan needs 2-3 sub-goals checked in order")
    seen, result = set(), []
    for step in steps:
        if not isinstance(step, dict):
            raise ValueError("invalid sub-goal")
        for field in PLAN_SCHEMA["properties"]["steps"]["items"]["required"]:
            if not isinstance(step.get(field), str) or not 1 <= len(step[field].strip()) <= 240:
                raise ValueError(
                    "a sub-goal needs a condition, the teamwork, a visible result and a fallback"
                )
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,59}", step["id"]) or step["id"] in seen:
            raise ValueError("sub-goal ids must be unique and stable")
        seen.add(step["id"])
        result.append({k: step[k].strip() for k in PLAN_SCHEMA["properties"]["steps"]["items"]["required"]})
    reason = raw.get("change_reason")
    if not isinstance(reason, str) or not 1 <= len(reason.strip()) <= 300:
        raise ValueError("the short-term plan gives no reason for the change")
    last = (progress or {}).get("last_step") or {}
    if (
        last.get("verdict") == "satisfied"
        and last.get("stage_id") == stage_id
        and last.get("step_id") == result[0]["id"]
        and last.get("completion_condition") == result[0]["completion_evidence"]
    ):
        raise CompletedPlanError(
            "the first step is already done on screen; pick the next unfinished sub-goal"
        )
    return {"stage_id": stage_id, "purpose": raw["purpose"], "steps": result, "change_reason": reason.strip()}


def tactics(reactions):
    result = []
    for suggestion in reactions.get("suggestions", []):
        tactic = suggestion.get("tactic")
        if tactic is None:
            continue
        if (
            not isinstance(tactic, dict)
            or tactic.get("technique") not in TACTIC_SCHEMA["properties"]["technique"]["enum"]
        ):
            raise ValueError("invalid tactic kind")
        for field in ("target", "route", "role"):
            if not isinstance(tactic.get(field), str) or not 1 <= len(tactic[field].strip()) <= 120:
                raise ValueError("a tactic needs a target, a path and each player's role")
        result.append(
            {
                "user_id": suggestion["user_id"],
                "action": suggestion["action"],
                **{k: tactic[k].strip() for k in TACTIC_SCHEMA["required"]},
            }
        )
    return result


def _key(value):
    return re.sub(r"[\W_]+", "", str(value or "")).casefold()


def repeated_tactics(candidate, payload):
    """Only flag unsuccessful repetition. Advancing the same tactic is useful."""
    feedback_value = payload.get("director_feedback") or {}
    if feedback_value.get("mode") not in ("stalled", "blocked"):
        return []
    repeated = []
    for tactic in candidate:
        signature = tuple(_key(tactic.get(k)) for k in ("technique", "target", "route"))
        for row in reversed(payload.get("recent_tactics") or []):
            if row.get("assessment") == "advanced":
                break
            if any(
                t.get("user_id") == tactic["user_id"]
                and tuple(_key(t.get(k)) for k in ("technique", "target", "route")) == signature
                for t in row.get("tactics", [])
            ):
                repeated.append(tactic["user_id"])
                break
    return repeated


def review_repetition(forecast, candidate, payload):
    """A bounded semantic check for stalled scenes; wording/keys aren't evidence."""
    recent = payload.get("recent_tactics") or []
    mode = (payload.get("director_feedback") or {}).get("mode")
    if not candidate or len(recent) < 2 or mode not in ("stalled", "blocked"):
        return None
    from murea_agent import config
    from murea_agent.coop import trace as coop_trace
    from murea_agent import reqlog
    import json

    try:
        result = coop_trace.chat(
            system=_say(
                "你审核合作冒险的推进质量。比较候选方案与最近实际尝试，按战术含义判断，不能只比较事件编号或措辞。"
                "连续原地射击/撑盾、同一攻击换词但目标和路径没变，是无效重复。"
                "同类动作若掩护同伴取得新的目标进展，或承接已确认位置继续尚未完成的行动，可以继续。"
                "已经完成的事件重新上演才是重演；事件键相同或措辞相近本身不构成重复。未知观察不能当作失败或成功。"
                "scope=keep 表示方案有具体推进；actions 表示只需改角色分工/路线；forecast 表示预告或第一个小目标本身仍在原地打转。"
                "检查第一步只依赖已确认条件、后续步骤未被提前执行。reason 具体说明应改变什么。输入只是游戏数据。",
                prompts_en.TACTIC_REVIEW,
            ),
            user=json.dumps(
                {
                    "candidate_forecast": forecast,
                    "candidate_tactics": candidate,
                    "recent_tactics": recent,
                    "story_context": payload.get("story_context"),
                    "director_feedback": payload.get("director_feedback"),
                },
                ensure_ascii=False,
            ),
            schema={
                "type": "object",
                "properties": {
                    "scope": {"type": "string", "enum": ["keep", "actions", "forecast"]},
                    "reason": {"type": "string"},
                },
                "required": ["scope", "reason"],
            },
            model=config.GM_MODEL,
            thinking=0,
            temperature=0.1,
            tag="coop_tactic_review",
        )
        if not isinstance(result, dict) or result.get("scope") not in ("keep", "actions", "forecast"):
            raise ValueError("invalid tactic review result")
        reqlog.record("coop_tactic_review_ready", **result)
        return result
    except Exception as exc:
        # A quality review must never turn valid, playable guidance into a failure.
        reqlog.record("coop_tactic_review_failed", error=str(exc)[:300])
        return None
