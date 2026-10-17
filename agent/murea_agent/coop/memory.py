"""Observed state slots, separate from the chronological story ledger."""

import copy
import hashlib
import json
import re
from murea_agent.coop import boss as coop_boss
from murea_agent.coop import prompts_en
from murea_agent.util import _say


FACT_SCHEMA = {
    "type": "array",
    "maxItems": 12,
    "items": {
        "type": "object",
        "properties": {
            "entity": {"type": "string"},
            "attribute": {
                "type": "string",
                "enum": ["location", "posture", "condition", "passage", "mechanism", "equipment", "identity"],
            },
            "value": {"type": "string"},
            "evidence": {"type": "string"},
            "frame_indices": {"type": "array", "items": {"type": "integer"}},
        },
        "required": ["entity", "attribute", "value", "evidence", "frame_indices"],
    },
}

OBSERVATION_INSTRUCTION = """\n同时给出 fact_updates，只记录关键帧直接确认的状态变化。
通常只需 3–8 条关键变化，最多 12 条；取值和证据各用一句短语，避免复述整轮视频。
entity 使用 scene、boss、给定玩家 user_id，或 object:英文稳定名称（例如 object:stone_gate）。
沿用 current_state.facts 已有实体名和属性；同一扇门不能每轮换名。新物体必须在画面中清晰可见。
attribute 为 location/posture/condition/passage/mechanism/equipment/identity。
value 写精简的当前取值，状态不变时沿用旧值，不能把整轮故事塞进一个字段。
evidence 写可见依据，frame_indices 指向从 1 开始的本轮关键帧；持续状态取最后清晰可见的状态。
新值替换同一实体同一属性的旧值；不同实体的状态独立。没拍到就不更新，不能据此删除已知通路、装备或伤情。
如门已打开，更新 object:stone_gate.passage；不要同时保留“关闭”和“打开”为当前状态。
environment_state/boss_state 是当前可见场景的简述，历史变化由事实账本保留，不要拼接互相矛盾的旧姿态。
"""
OBSERVATION_INSTRUCTION = _say(OBSERVATION_INSTRUCTION, prompts_en.MEMORY_OBSERVATION)


def validate_updates(raw, members, frame_count):
    if not isinstance(raw, list) or len(raw) > 12:
        raise ValueError("state updates must be at most 12 records grounded in the frames")
    known = {"scene", "boss", *(m["user_id"] for m in members)}
    attributes = FACT_SCHEMA["items"]["properties"]["attribute"]["enum"]
    updates, seen = [], set()
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("invalid state update")
        entity, attribute = item.get("entity"), item.get("attribute")
        if (
            not isinstance(entity, str)
            or (entity not in known and not re.fullmatch(r"object:[a-z][a-z0-9_]{0,59}", entity))
            or attribute not in attributes
            or (entity, attribute) in seen
        ):
            raise ValueError("state update entity or attribute is invalid or repeated")
        indices = item.get("frame_indices")
        if (
            not isinstance(indices, list)
            or not indices
            or any(type(i) is not int or not 1 <= i <= frame_count for i in indices)
        ):
            raise ValueError("state update has no valid keyframe evidence")
        for field in ("value", "evidence"):
            if not isinstance(item.get(field), str) or not 1 <= len(item[field].strip()) <= 300:
                raise ValueError("invalid state value or evidence")
        seen.add((entity, attribute))
        updates.append(
            {k: copy.deepcopy(item[k]) for k in ("entity", "attribute", "value", "evidence", "frame_indices")}
        )
    return updates


def apply_updates(state, updates, node_id, number):
    facts = state.setdefault("facts", {})
    for update in updates:
        key = update["entity"] + "." + update["attribute"]
        if facts.get(key, {}).get("round", -1) > number:
            continue
        facts[key] = {**copy.deepcopy(update), "node_id": node_id, "round": number}


def current(state):
    """A small current snapshot; never embed the ledger recursively."""
    result = {
        k: copy.deepcopy(state[k])
        for k in ("phase", "environment", "boss", "players", "boss_visible", "boss_known", "facts")
        if k in state
    }
    result["boss_life"] = coop_boss.current(state)
    if state.get("facts"):
        result["facts"] = {
            key: {k: copy.deepcopy(value[k]) for k in ("entity", "attribute", "value", "round") if k in value}
            for key, value in state["facts"].items()
        }
    return result


def material(state):
    facts = state.get("facts") or {}
    if not facts:
        return current(state)
    # Evidence wording and timestamps don't change a decision. Keep the visual
    # summaries too: a partial slot update (e.g. equipment only) must not hide
    # a newly observed player position or environmental change.
    result = {"facts": {key: value.get("value") for key, value in facts.items()}}
    result.update(
        players=state.get("players", {}), environment=state.get("environment", ""), boss=state.get("boss", "")
    )
    result["boss_life"] = coop_boss.current(state)
    return result


def fingerprint(state):
    return hashlib.sha256(
        json.dumps(material(state), ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


def history(state, limit=8):
    """Keep recent evidence without repeating old forecasts and requests."""
    return [
        {
            "round": row.get("round"),
            "node_id": row.get("node_id"),
            "observation": copy.deepcopy(row.get("observation") or {}),
        }
        for row in state.get("persistent_history", [])[-limit:]
    ]
