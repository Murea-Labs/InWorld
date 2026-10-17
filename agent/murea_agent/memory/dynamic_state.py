"""Dynamic state of characters and places: what may change from one beat to the next (clothes, injuries, props, mood, weather), kept apart from the locked reference descriptions."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


# Dynamic state is deliberately separate from the immutable reference-image
# description (`desc` / `definition`).  The latter keeps pixels stable; this is
# what is allowed to change from one beat to the next.
SUBJECT_STATE_DEFAULTS: dict[str, Any] = {
    "location": "",
    "outfit": "",
    "condition": "",
    "mood": "",
    "goal": "",
    "knowledge": [],
    "relations": {},
}


PLACE_STATE_DEFAULTS: dict[str, Any] = {
    "time_of_day": "",
    "weather": "",
    "lighting": "",
    "condition": "",
    "props": [],
}


SUBJECT_STATE_TEXT_FIELDS = ("location", "outfit", "condition", "mood", "goal")
PLACE_STATE_TEXT_FIELDS = ("time_of_day", "weather", "lighting", "condition")
STATE_TEXT_LIMIT = 240
STATE_LIST_LIMIT = 24
STATE_LIST_ITEM_LIMIT = 160


def _state_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.split())[:STATE_TEXT_LIMIT]


def _state_list(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    out: list[str] = []
    for item in value:
        text = _state_text(item)[:STATE_LIST_ITEM_LIMIT]
        if text and text not in out:
            out.append(text)
        if len(out) >= STATE_LIST_LIMIT:
            break
    return out


def _normalise_state(value: Any, defaults: dict[str, Any]) -> dict[str, Any]:
    """Return a bounded, JSON-safe state object while preserving old worlds."""
    raw = value if isinstance(value, dict) else {}
    out: dict[str, Any] = {}
    for key in defaults:
        if key in ("knowledge", "props"):
            out[key] = _state_list(raw.get(key))
        elif key == "relations":
            relations: dict[str, str] = {}
            candidate = raw.get(key)
            if isinstance(candidate, dict):
                for target, relation in candidate.items():
                    t, r = _state_text(target)[:80], _state_text(relation)[:STATE_LIST_ITEM_LIMIT]
                    if t and r:
                        relations[t] = r
            out[key] = relations
        else:
            out[key] = _state_text(raw.get(key))
    return out


class DynamicStateMixin:
    """Dynamic-state methods of World."""

    def _migrate_dynamic_state(self) -> None:
        """Give legacy assets an empty dynamic-state object on load."""
        for subject in self.subjects.values():
            if isinstance(subject, dict):
                subject["state"] = _normalise_state(subject.get("state"), SUBJECT_STATE_DEFAULTS)
        for place in self.places.values():
            if isinstance(place, dict):
                place["state"] = _normalise_state(place.get("state"), PLACE_STATE_DEFAULTS)

    def _ensure_dynamic_state(self) -> None:
        # This also makes worlds constructed directly in tests behave like loaded worlds.
        self._migrate_dynamic_state()

    def dynamic_state_snapshot(self) -> dict[str, Any]:
        """Deep-copy all mutable asset state for a node checkpoint."""
        self._ensure_dynamic_state()
        return {
            "subjects": {sid: deepcopy(subject["state"]) for sid, subject in self.subjects.items()},
            "places": {name: deepcopy(place["state"]) for name, place in self.places.items()},
        }

    def restore_dynamic_state(self, snapshot: dict[str, Any] | None) -> None:
        """Restore a checkpoint, resetting assets created after that checkpoint."""
        if not isinstance(snapshot, dict):
            return
        self._ensure_dynamic_state()
        subjects = snapshot.get("subjects") if isinstance(snapshot.get("subjects"), dict) else {}
        places = snapshot.get("places") if isinstance(snapshot.get("places"), dict) else {}
        for sid, subject in self.subjects.items():
            subject["state"] = _normalise_state(subjects.get(sid), SUBJECT_STATE_DEFAULTS)
        for name, place in self.places.items():
            place["state"] = _normalise_state(places.get(name), PLACE_STATE_DEFAULTS)

    @staticmethod
    def _updates(raw: Any, key: str) -> list[dict[str, Any]]:
        """Accept the documented list form and a forgiving keyed-map form."""
        value = raw.get(key) if isinstance(raw, dict) else None
        if isinstance(value, list):
            return [x for x in value if isinstance(x, dict)]
        if isinstance(value, dict):
            out: list[dict[str, Any]] = []
            for target, patch in value.items():
                if isinstance(patch, dict):
                    out.append({"name": str(target), **patch})
            return out
        return []

    def apply_dynamic_updates(
        self,
        updates: Any,
        *,
        current_place: str = "",
        visible_subject_ids: list[str] | tuple[str, ...] = (),
    ) -> dict[str, Any]:
        """Apply the GM's end-of-beat state patch deterministically.

        Unknown entities and unknown fields are ignored.  Empty strings mean
        "unchanged"; list fields support explicit *_add / *_remove operations.
        Visible subjects are always stamped with the scene they appear in, so a
        model forgetting `location` cannot leave the world map stale.
        """
        self._ensure_dynamic_state()
        subject_changes: dict[str, dict[str, Any]] = {}
        for update in self._updates(updates, "subjects"):
            target = update.get("id") or update.get("name")
            sid = self._resolve_subject(target)
            if sid is None:
                continue
            state = self.subjects[sid]["state"]
            before = deepcopy(state)
            for key in SUBJECT_STATE_TEXT_FIELDS:
                text = _state_text(update.get(key))
                if text:
                    state[key] = text
            self._apply_list_patch(
                state, "knowledge", update.get("knowledge_add"), update.get("knowledge_remove")
            )
            relations = update.get("relation_updates")
            if isinstance(relations, dict):
                relations = [{"target": k, "relation": v} for k, v in relations.items()]
            if isinstance(relations, list):
                for relation in relations:
                    if not isinstance(relation, dict):
                        continue
                    target_name = _state_text(relation.get("target"))[:80]
                    relation_text = _state_text(relation.get("relation"))[:STATE_LIST_ITEM_LIMIT]
                    if target_name and relation_text:
                        target_sid = self._resolve_subject(target_name)
                        state["relations"][target_sid or target_name] = relation_text
            if state != before:
                subject_changes[sid] = {"before": before, "after": deepcopy(state)}

        # A visible person is in the current camera space at the end of this beat.
        if current_place:
            for sid in visible_subject_ids:
                if sid not in self.subjects:
                    continue
                state = self.subjects[sid]["state"]
                if state.get("location") != current_place:
                    before = deepcopy(state)
                    state["location"] = current_place
                    subject_changes[sid] = {
                        "before": subject_changes.get(sid, {}).get("before", before),
                        "after": deepcopy(state),
                    }

        place_changes: dict[str, dict[str, Any]] = {}
        for update in self._updates(updates, "places"):
            name = _state_text(update.get("name"))
            if name not in self.places:
                continue
            state = self.places[name]["state"]
            before = deepcopy(state)
            for key in PLACE_STATE_TEXT_FIELDS:
                text = _state_text(update.get(key))
                if text:
                    state[key] = text
            self._apply_list_patch(state, "props", update.get("props_add"), update.get("props_remove"))
            if state != before:
                place_changes[name] = {"before": before, "after": deepcopy(state)}

        return {"subjects": subject_changes, "places": place_changes}

    @staticmethod
    def _apply_list_patch(state: dict[str, Any], key: str, additions: Any, removals: Any) -> None:
        current = _state_list(state.get(key))
        for item in _state_list(additions):
            if item not in current:
                current.append(item)
        for item in _state_list(removals):
            current = [x for x in current if x != item]
        state[key] = current[:STATE_LIST_LIMIT]

    def _resolve_subject(self, value: Any) -> str | None:
        token = _state_text(value)
        if token in self.subjects:
            return token
        wanted = " ".join(token.split()).casefold()
        if not wanted:
            return None
        for sid, subject in self.subjects.items():
            if " ".join(str(subject.get("name") or "").split()).casefold() == wanted:
                return sid
        return None

    def dynamic_state_for_prompt(
        self, place: str, subject_ids: list[str] | tuple[str, ...]
    ) -> dict[str, Any]:
        """The state relevant to the next shot, including non-visual GM facts."""
        self._ensure_dynamic_state()
        return {
            "subjects": {
                sid: {
                    "name": self.subjects[sid].get("name", sid),
                    "state": deepcopy(self.subjects[sid]["state"]),
                }
                for sid in subject_ids
                if sid in self.subjects
            },
            "scene": {
                "name": place,
                "state": deepcopy((self.places.get(place) or {}).get("state") or {}),
            },
        }
