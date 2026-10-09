"""Quests: the locked mainline, its stages, and side quests, stored on the world and snapshotted per clip."""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any

from murea_agent import config
from murea_agent.util import _say


QUEST_TYPES = {"main", "side"}
QUEST_STATUSES = {"active", "completed", "failed"}
QUEST_LIMIT = 24
QUEST_TITLE_LIMIT = 80
QUEST_TEXT_LIMIT = 300
MAINLINE_LIMIT = 200
MAINLINE_PHASE_LIMIT = 80
MAINLINE_MILESTONE_LIMIT = 6
MAINLINE_MILESTONE_TEXT_LIMIT = 120


# Older versions wrote "what is happening this second" straight into the long-term mainline. Only
# high-confidence momentary states are repaired automatically: explicit physical predicaments or
# remaining-seconds wording, and no long-term goal verbs such as find / uncover / stop. Ordinary
# mystery hooks and user-written mainlines are left alone.
_IMMEDIATE_MAINLINE_RE = re.compile(
    r"还剩|只剩|倒计时(?:显示)?|被卡|被困|困在|压在|双腿|双脚|"
    r"此刻|眼前|正在|刚刚|传来|站在|坐在|躺在|手里|脚下",
    re.I,
)


_DURABLE_MAINLINE_RE = re.compile(
    r"必须|需要|要在|找出|查明|找到|阻止|逃离|拯救|保护|关闭|恢复|"
    r"调查|追查|完成|揭开|解决|夺回|守住|修复|带回|摧毁|拯救",
    re.I,
)


def _quest_text(value: Any, limit: int = QUEST_TEXT_LIMIT) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.split())[:limit]


def _quest_title(value: Any) -> str:
    text = _quest_text(value, QUEST_TITLE_LIMIT)
    if len(text) <= 42:
        return text
    return text[:41].rstrip() + "…"


def _mainline_phase(value: Any) -> str:
    return _quest_text(value, MAINLINE_PHASE_LIMIT)


def _mainline_milestones(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    out: list[str] = []
    for item in value:
        text = _quest_text(item, MAINLINE_MILESTONE_TEXT_LIMIT)
        if text and text not in out:
            out.append(text)
        if len(out) >= MAINLINE_MILESTONE_LIMIT:
            break
    return out


def _looks_like_immediate_mainline(value: Any) -> bool:
    text = _quest_text(value, MAINLINE_LIMIT)
    return bool(text and _IMMEDIATE_MAINLINE_RE.search(text) and not _DURABLE_MAINLINE_RE.search(text))


def _setting_place_hint(setting: str) -> str:
    """Extract a stable place word from the world setting, for repairing older mainlines."""
    for keyword in (
        "医院",
        "病院",
        "列车",
        "火车",
        "酒馆",
        "旅店",
        "神殿",
        "宫殿",
        "学校",
        "城市",
        "小镇",
        "村落",
        "港口",
        "空间站",
        "实验室",
        "密室",
        "地铁",
    ):
        if keyword in setting:
            return keyword
    text = _quest_text(setting, 36).rstrip("。！？.!?")
    return text or "这片土地"


def _durable_mainline_fallback(raw: Any, setting: str) -> str:
    """Turn an obviously one-off opening state into a long-term goal that can span many beats."""
    text = _quest_text(raw, MAINLINE_LIMIT)
    if config.LANGUAGE != "zh":
        # The keyword tables below read Chinese text; English worlds get one neutral goal.
        return "Find out what is really going on here, and resolve it or get out safely."
    place = _setting_place_hint(setting)
    if re.search(r"爆破|炸弹|爆炸|倒计时", text, re.I):
        return f"查明{place}里的爆破危机，并在爆炸前逃出生天。"
    if re.search(r"谋杀|凶手|尸体|命案", text, re.I):
        return f"查明{place}里的谋杀真相，并阻止凶手继续行动。"
    if re.search(r"失踪|找不到|绑架", text, re.I):
        return f"查明{place}中的失踪事件，并找到被带走的人。"
    if re.search(r"追兵|敌人|入侵|怪物", text, re.I):
        return f"查明{place}面临的威胁，并找到安全脱身的办法。"
    return f"查明{place}正在发生的核心危机，并找到解决或脱身的办法。"


def _durable_milestones_fallback(
    raw: Any,
    setting: str,
    objective: Any = "",
) -> list[str]:
    """A minimal path of milestones for mainlines migrated without, or missing, milestones."""
    text = _quest_text(raw, MAINLINE_LIMIT)
    if config.LANGUAGE != "zh":
        first = _quest_text(objective, MAINLINE_MILESTONE_TEXT_LIMIT)
        return (
            [first, "Find the key to the core danger", "Complete the main goal"]
            if first
            else [
                "Understand the danger and find the next lead",
                "Resolve the core danger",
            ]
        )
    place = _setting_place_hint(setting)
    if re.search(r"爆破|炸弹|爆炸|倒计时", text, re.I):
        return ["脱离开场危机", f"在{place}找到爆破装置", "关闭倒计时", f"逃离{place}"]
    if re.search(r"谋杀|凶手|尸体|命案", text, re.I):
        return [f"确认{place}的案发线索", "锁定幕后凶手", "阻止凶手继续行动"]
    if re.search(r"失踪|找不到|绑架", text, re.I):
        return [f"找到{place}中的关键线索", "确认失踪者下落", "救出并带回失踪者"]
    if re.search(r"追兵|敌人|入侵|怪物", text, re.I):
        return [f"确认{place}面临的威胁", "找到安全脱身或反制办法", "解决核心威胁"]
    first = _quest_text(objective, MAINLINE_MILESTONE_TEXT_LIMIT)
    return (
        [first, "取得解决核心危机的关键线索", "完成主线目标"]
        if first
        else [
            "确认危机并找到下一步线索",
            "解决核心危机",
        ]
    )


def _normalise_quest(value: Any, fallback_id: str) -> dict[str, Any] | None:
    """Normalize quests from old data or from the model into stable records that are safe to show."""
    if not isinstance(value, dict):
        return None
    title = _quest_title(value.get("title"))
    if not title:
        return None
    quest_type = str(value.get("type") or "side").lower()
    status = str(value.get("status") or "active").lower()
    return {
        "id": _quest_text(value.get("id"), 40) or fallback_id,
        "type": quest_type if quest_type in QUEST_TYPES else "side",
        "title": title,
        "summary": _quest_text(value.get("summary")),
        "status": status if status in QUEST_STATUSES else "active",
        "progress": _quest_text(value.get("progress")),
        "place": _quest_text(value.get("place"), 120),
        "created_node": _quest_text(value.get("created_node"), 40),
        "updated_node": _quest_text(value.get("updated_node"), 40),
        "phase": _mainline_phase(value.get("phase")),
        "milestones": _mainline_milestones(value.get("milestones")),
    }


class QuestsMixin:
    """Quest methods of World. Kept apart so world.py stays about the story tree and assets."""

    def _next_quest_id(self) -> str:
        used = {str(quest.get("id") or "") for quest in self.quests}
        number = 1
        while f"Q{number}" in used:
            number += 1
        return f"Q{number}"

    def _migrate_quests(self) -> None:
        """Normalize quests and migrate old momentary opening mainlines into long-term goals."""
        self._migrate_mainline()
        normalised: list[dict[str, Any]] = []
        used: set[str] = set()
        for index, raw in enumerate(self.quests if isinstance(self.quests, list) else [], 1):
            quest = _normalise_quest(raw, f"Q{index}")
            if quest is None:
                continue
            quest_id = quest["id"]
            if quest_id in used:
                quest_id = self._first_free_quest_id(used)
                quest["id"] = quest_id
            used.add(quest_id)
            normalised.append(quest)
            if len(normalised) >= QUEST_LIMIT:
                break
        self.quests = normalised

        if self.mainline and not any(q["type"] == "main" for q in self.quests):
            self.quests.insert(
                0,
                {
                    "id": self._next_quest_id(),
                    "type": "main",
                    "title": _quest_title(self.mainline) or _say("推进主线", "Main quest"),
                    "summary": _quest_text(self.mainline),
                    "status": "active",
                    "progress": _quest_text(self.objective),
                    "place": self.here if self.here in self.places else "",
                    "created_node": "",
                    "updated_node": self.head or "",
                    "phase": self.mainline_phase,
                    "milestones": deepcopy(self.mainline_milestones),
                },
            )
        main = next((q for q in self.quests if q["type"] == "main"), None)
        if main is not None:
            # The mainline text may change during migration or when first set; quest cards always
            # use the long-term goal, never leftovers of the old momentary situation.
            main["title"] = _quest_title(self.mainline) or _say("推进主线", "Main quest")
            main["summary"] = _quest_text(self.mainline)
            main["phase"] = self.mainline_phase
            main["milestones"] = deepcopy(self.mainline_milestones)

    def _migrate_mainline(self) -> None:
        """Migrate old "instant crisis" mainlines without raising, keeping the original opening situation."""
        self.mainline = _quest_text(self.mainline, MAINLINE_LIMIT)
        self.mainline_phase = _mainline_phase(self.mainline_phase)
        self.mainline_milestones = _mainline_milestones(self.mainline_milestones)
        self.opening_situation = _quest_text(self.opening_situation, 360)
        raw_mainline = self.mainline
        was_immediate = bool(raw_mainline and _looks_like_immediate_mainline(raw_mainline))
        if was_immediate:
            if not self.opening_situation:
                self.opening_situation = raw_mainline
            self.mainline = _durable_mainline_fallback(raw_mainline, self.setting)
            if not self.mainline_phase:
                self.mainline_phase = _say("脱离开场危机", "Escape the opening danger")
        if self.mainline and not self.mainline_milestones:
            self.mainline_milestones = _durable_milestones_fallback(
                raw_mainline if was_immediate else self.mainline,
                self.setting,
                self.objective,
            )

    def set_mainline(
        self,
        value: Any,
        *,
        phase: Any = "",
        milestones: Any = None,
        opening_situation: Any = "",
    ) -> None:
        """Set or automatically repair the mainline; rewrite momentary states into long-term goals."""
        candidate = _quest_text(value, MAINLINE_LIMIT)
        if not self.mainline:
            immediate = bool(candidate and _looks_like_immediate_mainline(candidate))
            # The model sometimes returns a momentary mainline together with an ordinary reply. For
            # an old mainline the original sentence is the most reliable opening fact and must not
            # be lost to a reply placeholder first.
            if immediate:
                if not self.opening_situation:
                    self.opening_situation = candidate
                candidate = _durable_mainline_fallback(candidate, self.setting)
            elif opening_situation and not self.opening_situation:
                self.opening_situation = _quest_text(opening_situation, 360)
            self.mainline = candidate
        if phase:
            self.mainline_phase = _mainline_phase(phase)
        if milestones is not None:
            parsed = _mainline_milestones(milestones)
            if parsed:
                self.mainline_milestones = parsed
        self._migrate_mainline()

    @staticmethod
    def _first_free_quest_id(used: set[str]) -> str:
        number = 1
        while f"Q{number}" in used:
            number += 1
        return f"Q{number}"

    def quest_state_snapshot(self) -> dict[str, Any]:
        self._migrate_quests()
        return {
            "quests": deepcopy(self.quests),
            "objective": self.objective,
            "mainline": self.mainline,
            "mainline_phase": self.mainline_phase,
            "mainline_milestones": deepcopy(self.mainline_milestones),
            "opening_situation": self.opening_situation,
        }

    def restore_quest_state(self, snapshot: dict[str, Any] | None) -> None:
        if not isinstance(snapshot, dict):
            return
        self.quests = deepcopy(snapshot.get("quests") or [])
        self.objective = _quest_text(snapshot.get("objective"), 120)
        if "mainline" in snapshot:
            self.mainline = _quest_text(snapshot.get("mainline"), MAINLINE_LIMIT)
        if "mainline_phase" in snapshot:
            self.mainline_phase = _mainline_phase(snapshot.get("mainline_phase"))
        if "mainline_milestones" in snapshot:
            self.mainline_milestones = _mainline_milestones(snapshot.get("mainline_milestones"))
        if "opening_situation" in snapshot:
            self.opening_situation = _quest_text(snapshot.get("opening_situation"), 360)
        self._migrate_quests()

    def _resolve_quest(self, value: Any) -> dict[str, Any] | None:
        token = _quest_text(value, QUEST_TITLE_LIMIT).casefold()
        if not token:
            return None
        for quest in self.quests:
            if (
                str(quest.get("id") or "").casefold() == token
                or str(quest.get("title") or "").casefold() == token
            ):
                return quest
        return None

    def apply_quest_updates(
        self,
        updates: Any,
        *,
        current_place: str = "",
        node_id: str = "",
    ) -> dict[str, list[str]]:
        """Apply the director's quest delta; unknown quests and unknown places never pollute the world."""
        self._migrate_quests()
        raw = updates if isinstance(updates, dict) else {}
        changed = {"created": [], "updated": []}

        creates = raw.get("create") if isinstance(raw.get("create"), list) else []
        for candidate in creates:
            if not isinstance(candidate, dict) or len(self.quests) >= QUEST_LIMIT:
                continue
            title = _quest_title(candidate.get("title"))
            if not title or self._resolve_quest(title) is not None:
                continue
            quest_type = str(candidate.get("type") or "side").lower()
            if quest_type not in QUEST_TYPES:
                quest_type = "side"
            # One mainline per world; additional optional goals can only be side quests.
            if quest_type == "main" and any(q["type"] == "main" for q in self.quests):
                quest_type = "side"
            requested_place = _quest_text(candidate.get("place"), 120)
            place = (
                requested_place
                if requested_place in self.places
                else (current_place if current_place in self.places else "")
            )
            quest = {
                "id": self._next_quest_id(),
                "type": quest_type,
                "title": title,
                "summary": _quest_text(candidate.get("summary")),
                "status": "active",
                "progress": _quest_text(candidate.get("progress")),
                "place": place,
                "created_node": node_id,
                "updated_node": node_id,
            }
            self.quests.append(quest)
            changed["created"].append(quest["id"])

        patches = raw.get("update") if isinstance(raw.get("update"), list) else []
        for patch in patches:
            if not isinstance(patch, dict):
                continue
            quest = self._resolve_quest(patch.get("id") or patch.get("title"))
            if quest is None:
                continue
            before = dict(quest)
            status = str(patch.get("status") or "").lower()
            if status in QUEST_STATUSES:
                quest["status"] = status
            progress = _quest_text(patch.get("progress"))
            if progress:
                quest["progress"] = progress
            place = _quest_text(patch.get("place"), 120)
            if place in self.places:
                quest["place"] = place
            if quest != before:
                quest["updated_node"] = node_id
                changed["updated"].append(quest["id"])

        # The mainline's current progress follows the saved objective, so even if the director
        # forgets the quest delta the drawer does not stay on the previous beat. Explicitly
        # completed / failed mainlines are never reactivated.
        main = next((q for q in self.quests if q["type"] == "main"), None)
        if main is not None and main["status"] == "active":
            if self.mainline:
                main["summary"] = _quest_text(self.mainline)
            if self.objective:
                main["progress"] = _quest_text(self.objective)
            main["phase"] = self.mainline_phase
            main["milestones"] = deepcopy(self.mainline_milestones)
            if current_place in self.places:
                main["place"] = current_place
            main["updated_node"] = node_id
        return changed
