"""Typed facts and deterministic compilation for the co-op video boundary.

The planner is allowed to decide what each participant intends to do.  This
module owns the wire objects that cross into the video compiler so a planned
action cannot be confused with an observed result and a model generated
timeline cannot silently disagree with the per-player actions.

The classes are intentionally small and dependency free.  They accept the
legacy dictionaries used by old room files and expose a compatibility
``to_legacy`` adapter while new callers can keep the immutable object around.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import re
import threading
from typing import Any
from collections.abc import Iterable, Mapping
from types import MappingProxyType

from murea_agent import reqlog
from murea_agent.coop import boss as coop_boss


@dataclass(frozen=True, slots=True)
class AssetReference:
    """A stable asset identity resolved to one concrete reference file."""

    asset_id: str
    kind: str
    name: str
    ref: str
    description: str = ""
    picture_index: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "asset_id": self.asset_id,
            "kind": self.kind,
            "name": self.name,
            "ref": self.ref,
            "description": self.description,
            "picture_index": self.picture_index,
        }


@dataclass(frozen=True, slots=True)
class WorldSnapshot:
    """Immutable facts used to plan one turn.

    ``revision`` is the room's logical state revision, not the UI write counter.
    ``node_id`` anchors a continuation to the exact preceding video.  Actions
    are copied as tuples so a worker cannot observe a later room edit halfway
    through planning.
    """

    world_id: str
    room_code: str
    revision: int
    round_number: int
    node_id: str
    setting: str
    visual_style: str
    scene: AssetReference | None
    creature: AssetReference | None
    players: tuple[dict[str, str], ...] = field(default_factory=tuple)
    story_state: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_room(
        cls, room: Mapping[str, Any], *, node_id: str = "", revision: int | None = None
    ) -> WorldSnapshot:
        members = tuple(
            MappingProxyType(
                {
                    "user_id": str(m.get("user_id", "")),
                    "subject_id": str(m.get("subject_id", "")),
                    "character": str(m.get("character", "")),
                    "description": str(m.get("description", "")),
                    "action": str(m.get("action", "")),
                    "ref": str(m.get("ref", "")),
                }
            )
            for m in (room.get("members") or [])
        )
        scene = None
        if room.get("scene_ref"):
            scene = AssetReference(
                asset_id=f"scene:{room.get('scene_name', '')}",
                kind="scene",
                name=str(room.get("scene_name", "")),
                ref=str(room["scene_ref"]),
                description=str(room.get("scene_description", "")),
            )
        monster = room.get("monster") or {}
        creature = None
        if monster.get("ref") and coop_boss.status(room) != "dead":
            creature = AssetReference(
                asset_id=str(monster.get("subject_id") or f"creature:{monster.get('name', '')}"),
                kind="creature",
                name=str(monster.get("name", "")),
                ref=str(monster.get("ref", "")),
                description=str(monster.get("description", "")),
            )
        return cls(
            world_id=str(room.get("world_id", "")),
            room_code=str(room.get("code", "")),
            revision=int(
                room.get("state_revision", revision if revision is not None else room.get("revision", 0)) or 0
            ),
            round_number=int(room.get("round", 0) or 0),
            node_id=str(node_id),
            setting=str(room.get("setting", "")),
            visual_style=str(room.get("visual_style", "")),
            scene=scene,
            creature=creature,
            players=members,
            story_state=_freeze_json(room.get("story_state") or {}),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "world_id": self.world_id,
            "room_code": self.room_code,
            "revision": self.revision,
            "round_number": self.round_number,
            "node_id": self.node_id,
            "setting": self.setting,
            "visual_style": self.visual_style,
            "scene": self.scene.as_dict() if self.scene else None,
            "creature": self.creature.as_dict() if self.creature else None,
            "players": [_copy_json(dict(player)) for player in self.players],
            "story_state": _copy_json(dict(self.story_state)),
        }


@dataclass(slots=True)
class CancellationToken:
    """Request-owned cancellation state shared by its background work."""

    _event: threading.Event = field(default_factory=threading.Event)
    reason: str = ""

    def cancel(self, reason: str = "") -> None:
        self.reason = reason
        self._event.set()

    def cancelled(self) -> bool:
        return self._event.is_set()


@dataclass(slots=True)
class TurnContext:
    """All transient ownership for one co-op turn."""

    request_id: str
    snapshot: WorldSnapshot
    assets: tuple[AssetReference, ...]
    cancellation: CancellationToken = field(default_factory=CancellationToken)
    image_futures: dict[str, Any] = field(default_factory=dict)
    log_context: dict[str, Any] = field(default_factory=dict)
    final_video_request: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class PlannedAction:
    """One player's requested motion, translated once into English."""

    user_id: str
    source_action: str
    appearance: str
    action: str
    beat: str = ""
    reference_layout: str = "single full-body view"
    visibility: str = "full body"
    retention_level: str = "fully_preserved"
    retention: str = ""
    action_id: str = ""
    target: str = ""
    path: str = ""
    end_state: str = ""
    start_state: str = ""

    @staticmethod
    def stable_id(user_id: str, source_action: str) -> str:
        value = f"{user_id}\0{source_action}".encode()
        return "action-" + hashlib.sha256(value).hexdigest()[:16]

    def as_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "source_action": self.source_action,
            "appearance": self.appearance,
            "action": self.action,
            "beat": self.beat,
            "reference_layout": self.reference_layout,
            "visibility": self.visibility,
            "retention_level": self.retention_level,
            "retention": self.retention,
            "action_id": self.action_id or self.stable_id(self.user_id, self.source_action),
            "target": self.target,
            "path": self.path,
            "end_state": self.end_state,
            "start_state": self.start_state,
        }


@dataclass(frozen=True, slots=True)
class PlannedEvent:
    """A future or current event with an explicit source boundary."""

    key: str
    kind: str
    text: str
    source: str = "forecast"
    observed: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "kind": self.kind,
            "text": self.text,
            "source": self.source,
            "observed": self.observed,
        }


@dataclass(frozen=True, slots=True)
class ObservedEvent:
    """A fact supported by a completed video's keyframes."""

    node_id: str
    summary: str
    confidence: str = "unknown"
    event_outcome: str = "unknown"
    boss_visible: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "summary": self.summary,
            "confidence": self.confidence,
            "event_outcome": self.event_outcome,
            "boss_visible": self.boss_visible,
        }


@dataclass(frozen=True, slots=True)
class WorldDelta:
    """A proposed state patch; it is committed only after version checking."""

    values: tuple[tuple[str, Any], ...] = field(default_factory=tuple)
    source: str = "planner"
    base_revision: int = 0

    def as_dict(self) -> dict[str, Any]:
        # ⚠️ `values` is the only field in this dataclass family that holds arbitrary JSON, and patches
        # often come straight from `WorldSnapshot.story_state`, a MappingProxyType frozen by
        # `_freeze_json`. Stored as is, it travels via directive['scene_plan'] into the Node and
        # only fails when `asdict(node)` deep-copies it:
        #
        #     round_failed  cannot pickle 'mappingproxy' object
        #
        # By then the video is already rendered, the user waited a whole round for nothing, and a
        # retry fails again. So it must be unfrozen here.
        return {
            "values": {key: _copy_json(value) for key, value in self.values},
            "source": self.source,
            "base_revision": self.base_revision,
        }


@dataclass(frozen=True, slots=True)
class ScenePlan:
    """The one immutable plan consumed by :class:`ShotCompiler`."""

    base_revision: int
    duration: int
    continuation: bool
    actions: tuple[PlannedAction, ...]
    scene: str
    sound: str
    narration: str = ""
    event: PlannedEvent | None = None
    event_action: str = ""
    world_delta: WorldDelta = field(default_factory=WorldDelta)
    assets: tuple[AssetReference, ...] = field(default_factory=tuple)
    # Subject numbering follows the immutable reference-picture order.  The
    # submission order is separate UI/history metadata and must never be used
    # to renumber a character in the video prompt.
    subject_order: tuple[str, ...] = field(default_factory=tuple)
    submission_order: tuple[str, ...] = field(default_factory=tuple)
    source: str = "planner"

    @classmethod
    def from_legacy(
        cls,
        raw: Mapping[str, Any],
        members: Iterable[Mapping[str, Any]],
        *,
        base_revision: int = 0,
        duration: int = 10,
        continuation: bool = False,
        assets: Iterable[AssetReference] = (),
        submission_order: Iterable[str] = (),
    ) -> ScenePlan:
        member_list = list(members)
        by_id = {str(m.get("user_id", "")): m for m in member_list}
        actions: list[PlannedAction] = []
        for entry in raw.get("actions") or []:
            if not isinstance(entry, Mapping):
                continue
            uid = str(entry.get("user_id", ""))
            member = by_id.get(uid, {})
            actions.append(
                PlannedAction(
                    user_id=uid,
                    source_action=str(entry.get("source_action", member.get("action", ""))),
                    appearance=str(entry.get("appearance", "")),
                    action=str(entry.get("action", "")),
                    beat=str(entry.get("beat", "")),
                    reference_layout=str(entry.get("reference_layout", "single full-body view")),
                    visibility=str(entry.get("visibility", "full body")),
                    retention_level=str(entry.get("retention_level", "fully_preserved")),
                    retention=str(entry.get("retention", "")),
                    action_id=str(
                        entry.get("action_id", "")
                        or PlannedAction.stable_id(
                            uid, str(entry.get("source_action", member.get("action", "")))
                        )
                    ),
                    target=str(entry.get("target", "") or ""),
                    path=str(entry.get("path", "") or ""),
                    end_state=str(entry.get("end_state", "") or ""),
                    start_state=str(entry.get("start_state", "") or ""),
                )
            )
        event_text = str(raw.get("world_event", "") or "").strip()
        event = None
        if event_text:
            event = PlannedEvent(
                key=str(raw.get("world_event_key", "") or ""),
                kind=str(raw.get("world_event_type", "environment") or "environment"),
                text=event_text,
            )
        return cls(
            base_revision=base_revision,
            duration=int(duration),
            continuation=bool(continuation),
            actions=tuple(actions),
            scene=str(raw.get("scene", "") or ""),
            sound=str(raw.get("sound", "") or ""),
            narration=str(raw.get("narration", "") or ""),
            event=event,
            event_action=str(raw.get("event_action", "") or ""),
            assets=tuple(assets),
            subject_order=tuple(str(m.get("user_id", "")) for m in member_list),
            submission_order=tuple(str(uid) for uid in submission_order if str(uid) in by_id),
            source="planner",
        )

    def to_legacy(self) -> dict[str, Any]:
        result = {
            "scene": self.scene,
            "sound": self.sound,
            "narration": self.narration,
            "actions": [action.as_dict() for action in self.actions],
            "timeline": ShotCompiler.timeline(self),
        }
        if self.event:
            result.update(
                world_event=self.event.text, world_event_type=self.event.kind, world_event_key=self.event.key
            )
        return result

    def as_dict(self) -> dict[str, Any]:
        """Return a stable, fully structured serialization for diagnostics.

        The legacy adapter intentionally keeps the old prompt-planning shape;
        this method preserves the typed boundary so persisted directives can be
        inspected without reconstructing a plan from a timeline string.
        """
        return {
            "base_revision": self.base_revision,
            "duration": self.duration,
            "continuation": self.continuation,
            "actions": [action.as_dict() for action in self.actions],
            "scene": self.scene,
            "sound": self.sound,
            "narration": self.narration,
            "event": self.event.as_dict() if self.event else None,
            "event_action": self.event_action,
            "world_delta": self.world_delta.as_dict(),
            "assets": [asset.as_dict() for asset in self.assets],
            "subject_order": list(self.subject_order),
            "submission_order": list(self.submission_order),
            "source": self.source,
            "timeline_source": "server:ShotCompiler",
            "timeline": ShotCompiler.timeline(self),
        }


class AssetResolver:
    """Resolve stable references into deterministic Picture order."""

    @staticmethod
    def resolve(
        members: Iterable[Mapping[str, Any]],
        room: Mapping[str, Any],
        shared: Iterable[Mapping[str, Any]] | None = None,
    ) -> tuple[AssetReference, ...]:
        assets: list[AssetReference] = []
        for member in members:
            assets.append(
                AssetReference(
                    asset_id=str(member.get("subject_id", "")),
                    kind="character",
                    name=str(member.get("character", "")),
                    ref=str(member.get("ref", "")),
                    description=str(member.get("description", "")),
                    picture_index=len(assets) + 1,
                )
            )
        shared_list = list(shared) if shared is not None else []
        if shared is None:
            monster = room.get("monster") or {}
            if monster.get("ref") and coop_boss.status(room) != "dead":
                shared_list.append({"kind": "monster", **monster})
            if room.get("scene_ref"):
                shared_list.append(
                    {
                        "kind": "scene",
                        "name": room.get("scene_name", ""),
                        "ref": room.get("scene_ref", ""),
                        "description": room.get("scene_description", ""),
                    }
                )
        for item in shared_list:
            if item.get("kind") == "scene":
                assets.append(
                    AssetReference(
                        asset_id=f"scene:{item.get('name', '')}",
                        kind="scene",
                        name=str(item.get("name", "")),
                        ref=str(item.get("ref", "")),
                        description=str(item.get("description", "")),
                        picture_index=len(assets) + 1,
                    )
                )
            else:
                assets.append(
                    AssetReference(
                        asset_id=str(item.get("subject_id") or item.get("name", "")),
                        kind="creature",
                        name=str(item.get("name", "")),
                        ref=str(item.get("ref", "")),
                        description=str(item.get("description", "")),
                        picture_index=len(assets) + 1,
                    )
                )
        return tuple(assets)


class ShotCompiler:
    """Compile each action exactly once into the single shared timeline."""

    @staticmethod
    def action_clause(action: PlannedAction, subject: str) -> str:
        """Bind one locked action to its immutable actor exactly once."""
        text = action.action.strip()
        # Models sometimes include the binding in the translated action.
        # The compiler owns that prefix, so keep exactly one token.
        if text.casefold().startswith(subject.casefold()):
            text = text[len(subject) :].lstrip(" ,:;")
        text, dropped = _strip_leading_subject(text)
        if dropped:
            reqlog.record("coop_timeline_subject_deduped", subject=subject, dropped=dropped)
        return f"{subject} {text}".strip() if text else subject

    @staticmethod
    def _phases(action: PlannedAction, subject: str) -> dict[str, str]:
        """Keep the complete action as an overview, never as the starting pose.

        A route spans the movement; it is not an event that happens halfway
        through the shot. Only an explicitly supplied initial state is timed
        at the start. Unknown starts and unspecified outcomes stay absent.
        """

        def clean(text):
            return text.strip().rstrip(" .;")

        def state_clause(text, phase):
            text = clean(text)
            if not text:
                return ""
            # New plans return complements of "is". Legacy plans sometimes
            # contain bare locations ("its rear flank") or lists of poses;
            # keep those facts without inventing a preposition or movement.
            if re.match(
                r"^(?:at|in|on|near|beside|behind|between|inside|outside|with|standing|"
                r"facing|kneeling|crouching|holding|seated|lying|ready|upright|stopped)\b",
                text,
                re.I,
            ):
                return f"{subject} is {text}"
            return f"{subject}'s {phase} state is: {text}"

        route = ""
        if clean(action.path):
            route = f"{subject} follows {clean(action.path)}"
            if clean(action.target):
                route += f", relative to {clean(action.target)}"
        elif clean(action.target):
            route = f"{subject}'s movement is relative to {clean(action.target)}"
        return {
            "start": state_clause(action.start_state, "starting"),
            "action": clean(ShotCompiler.action_clause(action, subject)),
            "path": route,
            "end": state_clause(action.end_state, "final"),
        }

    @staticmethod
    def _join_phase(prefix: str, clauses: list[str]) -> str:
        return prefix + ". ".join(clauses) + "." if clauses else ""

    @classmethod
    def _timeline(cls, actions, duration: int, event_action: str, *, caption: bool) -> str:
        # State and path metadata stay for internal checks; the video model only gets the visible
        # performance the model wrote.
        if caption:
            clauses = [
                cls.action_clause(action, subject).strip().rstrip(" .;")
                for action, subject in actions
                if action.action.strip()
            ]
            if event_action.strip():
                clauses.append(event_action.strip().rstrip(" .;"))
            return ". ".join(clauses) + "." if clauses else ""
        phases = [cls._phases(action, subject) for action, subject in actions if action.action.strip()]
        total = max(4, int(duration or 4))
        prefixes = {
            "start": "At the start, " if caption else "At 0.00s, ",
            "action": "Over the shot, " if caption else f"From 0.00s to {total:.2f}s, ",
            "path": "As the movement unfolds, "
            if caption
            else f"Along the movement from 0.00s to {total:.2f}s, ",
            "end": "By the end, " if caption else f"By {total:.2f}s, ",
        }
        # Group all players by phase, retaining the immutable reference order.
        # No actor finishes before another actor's start is introduced.
        lines = [
            cls._join_phase(prefixes[key], [row[key] for row in phases if row[key]])
            for key in ("start", "action", "path")
        ]
        if event_action.strip():
            lines.append(event_action.strip().rstrip(" .;") + ".")
        lines.append(cls._join_phase(prefixes["end"], [row["end"] for row in phases if row["end"]]))
        return (
            " ".join(line for line in lines if line)
            or "The established subjects remain visible in the environment."
        )

    @classmethod
    def timeline(cls, plan: ScenePlan, *, caption: bool = False) -> str:
        subject_order = plan.subject_order or tuple(action.user_id for action in plan.actions)
        subject_by_user = {uid: f"<Subject {index}>" for index, uid in enumerate(subject_order, 1)}
        actions = [(action, subject_by_user[action.user_id]) for action in plan.actions]
        return cls._timeline(actions, plan.duration, plan.event_action, caption=caption)

    @classmethod
    def compile_timeline(
        cls,
        plan: ScenePlan,
        *,
        camera: str,
        cast_count: str = "",
        creature_clause: str = "",
        environment: str = "",
        style: str = "",
    ) -> str:
        timeline = cls.timeline(plan, caption=True)
        return " ".join(
            part
            for part in (
                "[Shot 1]",
                environment,
                camera,
                cast_count,
                creature_clause,
                timeline,
                style,
            )
            if part
        ).strip()


# The server **prepends** `<Subject N>` to each action in the timeline, so `action` must be a bare
# predicate. Models often write a full sentence ("The character stands ..."), which becomes:
#
#     <Subject 2> The character uses a weapon to block the arm.
#
# In a sample of 110 timeline entries with a capitalized word after `<Subject N>`, 63 (57%) repeated
# the subject ("The character", "The player", or role names like "The archer"). The number exists to
# disambiguate, and this defeats it; worse, the same sentence may use a role name for **another**
# subject ("The swordswoman follows closely behind the archer"), so the video model cannot tell who
# is who.
#
# Only strip a phrase starting with a definite article (The / This / That) or a bare pronoun, and
# only when a lowercase word follows. "A massive stone fist ..." stays (an indefinite article
# usually introduces a new object), and so does "The dragon's tail ..." (the apostrophe does not
# match).
_SUBJECT_PRONOUNS = ("She", "He", "They", "It")
# Actions are present tense, third person, so the verb ends in s (stands / rolls / sprints /
# follows / grips / raises / is ...). That gives a reliable anchor without POS tagging: the first
# lowercase word ending in s after `The` is the verb. All 43 real samples above match, none wrongly.
_LEADING_DEFINITE = re.compile(r"^(?:The|This|That)\s+")
_PREDICATE_WORD = re.compile(r"^[a-z][\w-]*s$")


def _strip_leading_subject(text: str) -> tuple[str, str]:
    """Strip a repeated subject at the start of an action; returns (bare predicate, stripped part)."""
    for pronoun in _SUBJECT_PRONOUNS:
        if text.startswith(pronoun + " ") and text[len(pronoun) + 1 : len(pronoun) + 2].islower():
            return text[len(pronoun) + 1 :], pronoun
    head = _LEADING_DEFINITE.match(text)
    if not head:
        return text, ""
    rest = text[head.end() :]
    words = rest.split(" ")
    # The subject phrase takes at most 3 more words; without a verb, leave it alone rather than cut
    # wrongly.
    for i, word in enumerate(words[:4]):
        if not _PREDICATE_WORD.match(word):
            continue
        if i == 0:  # "The stands ..." is not a subject phrase
            return text, ""
        stripped = " ".join(words[i:])
        if not stripped:
            return text, ""
        return stripped, text[: len(text) - len(stripped)].strip()
    return text, ""


def _copy_json(value: Any) -> Any:
    # ``_freeze_json`` stores every nested object as ``MappingProxyType``.
    # Converting only the outer proxy with ``dict(...)`` leaves nested proxies
    # behind; dataclasses.asdict/deepcopy then fails after a rendered video with
    # ``cannot pickle 'mappingproxy' object`` while committing the Node.
    if isinstance(value, Mapping):
        return {str(k): _copy_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_copy_json(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_copy_json(v) for v in value)
    return value


def _freeze_json(value: Any) -> Any:
    """Recursively freeze JSON-like state carried by a snapshot."""
    if isinstance(value, dict):
        return MappingProxyType({str(k): _freeze_json(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(v) for v in value)
    if isinstance(value, tuple):
        return tuple(_freeze_json(v) for v in value)
    return value


__all__ = [
    "AssetReference",
    "WorldSnapshot",
    "PlannedAction",
    "PlannedEvent",
    "ObservedEvent",
    "WorldDelta",
    "ScenePlan",
    "AssetResolver",
    "ShotCompiler",
    "CancellationToken",
    "TurnContext",
]
