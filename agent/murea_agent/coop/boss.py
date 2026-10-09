"""Observed Boss life state, independent of reference ownership and visibility."""

import copy
import re
from murea_agent.coop import prompts_en
from murea_agent.util import _say


LIFE_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["alive", "dead", "unknown"]},
        "evidence": {"type": "string"},
        "frame_indices": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["status", "evidence", "frame_indices"],
}

OBSERVATION_INSTRUCTION = """
boss_life 单独核对当前 shared_creature 对应 BOSS 的生死，不能把新生物当成旧 BOSS。
status 为 alive/dead/unknown，evidence 是可见依据，frame_indices 使用本轮从 1 开始的关键帧编号。
dead 需要明确死亡画面证据；倒地、击败、昏迷、静止、暂未出镜或玩家要求击杀都不能单独证明死亡。
本轮前面的关键帧已证实死亡而末帧尸体离开镜头时，仍可返回 dead；boss_visible 只表示末帧是否可见。
没有清晰证据或尚无已绑定 BOSS 时返回 unknown，不根据预告、指令、旁白推断死亡。
"""
OBSERVATION_INSTRUCTION = _say(OBSERVATION_INSTRUCTION, prompts_en.BOSS_OBSERVATION)

PLANNING_INSTRUCTION = """
boss_life is the observed life state of the existing creature. If dead, it is a corpse,
not an active opponent. Its living reference is intentionally absent and has no Subject token.
Use the preceding frames and observed state for the remains' position. Only mention the
motionless remains when they are still in the shot or a player interacts with them; keep
them outside the composition once the players and camera have left. Never stage another
attack, breathing, standing up or pursuit by this dead creature. Follow the players' current
positions and ongoing actions rather than returning them to the corpse. A new creature is
a separate identity, never a revival of the old one.
"""

# Only explicit assertions from old *observations* are migrated. In particular,
# "defeated", "motionless", and "not visible" are not death evidence.
_DEATH = re.compile(
    r"(?:已(?:经)?|确认|确定|被)(?:被)?(?:击杀|杀死|死亡|死去)|死透|尸体|遗骸|"
    r"(?:^|[，。；;\s])(?:死亡|死去)(?:[，。；;\s]|$)|"
    r"\b(?:dead|killed|slain|corpse|carcass)\b",
    re.I,
)
_UNCERTAIN = re.compile(
    r"未|没有|并非|不是|尚|无|可能|疑似|仿佛|似乎|如果|若|试图|将要|假死|濒死|失败|接近|避免|"
    r"\b(?:not|no|never|nearly|almost|might|may|seems?|appears?|if|undead|failed|playing)\b",
    re.I,
)


def identity(monster):
    return str(monster.get("subject_id") or monster.get("ref") or "")


def _legacy_status(state):
    # Do not read the forecast, reference description or requested actions.
    text = str(state.get("boss") or "")
    return "dead" if _DEATH.search(text) and not _UNCERTAIN.search(text) else "unknown"


def status(room, monster=None):
    key = identity((room.get("monster") or {}) if monster is None else monster)
    if not key:
        return "unknown"
    state = room.get("story_state") or {}
    record = (state.get("boss_lifecycle") or {}).get(key)
    if record:
        return record.get("status", "unknown")
    if state.get("boss_identity", key) == key:
        return _legacy_status(state)
    return "unknown"


def context(room):
    return {"identity": identity(room.get("monster") or {}), "status": status(room)}


def current(state):
    record = (state.get("boss_lifecycle") or {}).get(state.get("boss_identity")) or {}
    return {"status": record.get("status") or _legacy_status(state)}


def validate(raw, frame_count, *, has_boss=True):
    """Optional enrichment must never discard an otherwise valid observation."""
    if not isinstance(raw, dict) or not has_boss:
        return {}
    status_value, indices = raw.get("status"), raw.get("frame_indices")
    if status_value not in ("alive", "dead", "unknown"):
        return {}
    if (
        not isinstance(indices, list)
        or any(type(i) is not int or not 1 <= i <= frame_count for i in indices)
        or (status_value != "unknown" and not indices)
    ):
        return {}
    evidence = raw.get("evidence")
    if not isinstance(evidence, str) or (status_value != "unknown" and not evidence.strip()):
        return {}
    return {"status": status_value, "evidence": evidence.strip()[:500], "frame_indices": list(indices)}


def publish(room, observation, node_id, number):
    key = identity(room.get("monster") or {})
    if not key:
        return
    state = room.setdefault("story_state", {})
    records = state.setdefault("boss_lifecycle", {})
    prior = records.get(key) or {}
    life = observation.get("boss_life") or {}
    if prior.get("status") == "dead" or prior.get("round", -1) > number:
        return  # Model drift cannot silently resurrect a confirmed dead identity.
    if life.get("status") not in ("alive", "dead"):
        # Compatibility for observations created before the structured field.
        if "boss_life" in observation:
            # A missing/invalid evidence extension must not be bypassed by
            # reading its accompanying prose through the legacy migration.
            state["boss_identity"] = key
            records.setdefault(key, {"status": "unknown", "node_id": node_id, "round": number})
            return
        if _legacy_status({"boss": observation.get("boss_state")}) != "dead":
            return
        life = {"status": "dead", "evidence": observation["boss_state"], "frame_indices": []}
    state["boss_identity"] = key
    records[key] = {**copy.deepcopy(life), "node_id": node_id, "round": number}


def remember(room):
    """Persist explicit legacy death before later observations replace prose."""
    if status(room) == "dead":
        state = room.setdefault("story_state", {})
        key = identity(room.get("monster") or {})
        state["boss_identity"] = key
        state.setdefault("boss_lifecycle", {}).setdefault(
            key,
            {
                "status": "dead",
                "evidence": state.get("boss", ""),
                "node_id": state.get("last_node_id", ""),
                "round": -1,
            },
        )


def bind(room, *, reset=False):
    """Bind a new asset without transferring the previous Boss's life/facts."""
    state = room.setdefault("story_state", {})
    key = identity(room.get("monster") or {})
    if reset or state.get("boss_identity") != key:
        state.update(boss_identity=key, boss="", boss_visible=False, boss_known=bool(key))
        state["facts"] = {k: v for k, v in (state.get("facts") or {}).items() if not k.startswith("boss.")}
    if reset and key:
        state.setdefault("boss_lifecycle", {}).pop(key, None)


def retired_reference(room):
    return (room.get("monster") or {}).get("ref", "") if status(room) == "dead" else ""
