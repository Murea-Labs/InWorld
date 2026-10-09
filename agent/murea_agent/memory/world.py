"""World state.

Two things live here and they are deliberately different kinds of thing:

  history  -- the multi-turn conversation. Manages STORY coherence. Never compacted:
              a 5-minute demo is ~60 turns of ~100 tokens, it cannot overflow.
              The six-section prompts are NOT put in here -- letting the GM see 20 of
              its own previous prompts makes it copy its own phrasing instead of
              responding to the user. Only a one-line beat summary goes back in.

  assets   -- subjects (people) and places, each bound to a real image FILE.
              Manages PIXEL coherence. This is not memory: no amount of remembering
              lets a language model produce a JPEG. When the GM says "back to the
              meeting room", the agent looks the name up here and attaches that exact
              file. That lookup is the whole trick behind places staying themselves.

Segments form a TREE, not a list, because the video continuation artifacts do
(each records its parent). Undo is `head = node.parent`; branching is free.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import time
import uuid
from copy import deepcopy
import dataclasses
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from murea_agent import config
from murea_agent.memory import voices
from murea_agent.memory.dynamic_state import (
    PLACE_STATE_DEFAULTS,
    SUBJECT_STATE_DEFAULTS,
    DynamicStateMixin,
    _normalise_state,
)
from murea_agent.memory.quests import QuestsMixin

DEFAULT_USER_ID = "u-default"  # owner of worlds created without accounts


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _sid(prefix: str) -> str:
    return f"{prefix}{uuid.uuid4().hex[:8]}"


@dataclass
class Node:
    """One generated segment.

    Everything a developer needs to understand or reproduce this segment is stored
    on the node itself -- the GM's decision, the exact prompt the video model received, which
    reference files were attached in which Picture order, the sampling params, the
    continuation handle it resumed from, and where the wall clock went. The UI's
    inspector reads these fields directly; nothing is recomputed.
    """

    id: str
    parent: str | None
    video_id: str  # video continuation handle for this segment's tail latents
    clip: str  # filename under data/clips
    place: str
    beat: str  # one line, this is what goes back into GM history
    mode: str  # "continue" | "cut"
    # What must still be true on the people at the end of this beat: held items, clothing, visible
    # state. One English sentence. It is the only temporary state carried across scene changes;
    # without it the coffee in someone's hand vanishes when they walk from the café into the lobby
    # (locked descriptions never mention coffee).
    carry: str = ""
    # Correlate a disconnected browser request with its committed result.
    request_id: str = ""
    # Where to go next, exactly 3 options offered by the world; one click plays that way.
    options: list[str] = field(default_factory=list)
    # Which of the three the director recommends (index). The UI marks it; following it is the
    # mainline. Default 0 rather than -1: old clips lack the field and land on the first option, so
    # the UI needs no "no recommendation" branch.
    recommend: int = 0
    # What the world wants the user to do after this beat. A short sentence stored with the clip, so
    # reviewing shows the goal of that moment.
    objective: str = ""
    # What the next beat plays while the user's sentence is not finished yet; the frontend continues
    # on its own. "Go downstairs, cross the street to the restaurant" is one sentence and five
    # spaces; one beat cannot hold it, and without this field the user would type five times, using
    # a text-to-video page, not an agent.
    continue_with: str = ""
    # Steps after `continue_with` not played yet, one sentence each, in order.
    # ⚠️ This carries the "done up to step N" state. The director replans the whole goal every beat and has
    # no idea which steps were played; that is the server's job. With the remaining queue pinned on
    # the node, continuing just walks it; the queue only shrinks, so it always converges. Without
    # it, "open the door and walk out" once spent four continuations on the same door.
    plan: list[str] = field(default_factory=list)
    subjects: list[str] = field(default_factory=list)
    six_section: dict[str, str] = field(default_factory=dict)
    refs: list[str] = field(default_factory=list)
    elapsed: float = 0.0
    created: str = field(default_factory=_now)

    # --- developer inspection ------------------------------------------------
    # The narrative sentence fed back into the director's history. `beat` has two consumers with
    # opposite needs: the rewriter wants English physical detail ("the acting hand first reaches the
    # visible handle…"), the history wants a plain sentence. Sharing `beat` filled the history with
    # mechanical English and the director continued in that style. Old clips lack it;
    # `_history_beat` falls back to beat / reply.
    recap: str = ""
    user_text: str = ""  # what the user typed
    reply: str = ""  # what the GM said back
    directive: dict[str, Any] = field(default_factory=dict)  # raw GM JSON
    pictures: list[dict[str, Any]] = field(default_factory=list)  # Picture N -> which ref
    # <Audio N> -> which voice, bound to which <Subject N>. Like pictures, this is what was
    # **actually sent** for this clip and cannot be derived later from world.subjects: if the voice
    # changed later, reviewing this clip would show the new one while the video has the old one. Old
    # clips default to an empty list.
    audios: list[dict[str, Any]] = field(default_factory=list)
    model_params: dict[str, Any] = field(default_factory=dict)  # steps / shift / seed / target
    continued_from: str | None = None  # artifact the latents were resumed from
    artifact_id: str | None = None  # this segment's artifact, for the next one
    timings: dict[str, float] = field(default_factory=dict)  # seconds per stage
    # Reference images are drawn alongside the rewriter, so their time is inside
    # `timings["rewrite"]` rather than a stage of its own.
    images_drawn: int = 0
    image_seconds: float = 0.0
    # Reference images drawn in this beat (shown in the chat as "what I made for you"). Not
    # derivable from pictures later: when reviewing, the image is in world.places as if it had
    # always been there; only the moment of generation knows it was new.
    sketches: list[dict[str, Any]] = field(default_factory=list)
    # Frames from what this clip **really rendered**, stored in data/frames. Sent to the director
    # with the next snapshot; before this the director only saw its own beat (what it meant to film)
    # and knew nothing of what the video model produced, so it kept building on an imagined history.
    frames: list[str] = field(default_factory=list)
    prompt: str = ""  # the exact string the video model received
    origin: str = "turn"  # "turn" | "regenerate"
    note: str = ""  # free-form, set from the inspector
    # Dynamic world state before and after this beat.  Optional so nodes written by
    # older checkouts remain loadable; new nodes always record both snapshots.
    state_before: dict[str, Any] | None = None
    state_after: dict[str, Any] | None = None
    # Quests, like character state, belong to the current story branch. Nodes keep before / after
    # snapshots so undo turns "completed" back into "active" instead of leaking results from another
    # timeline.
    quest_before: dict[str, Any] | None = None
    quest_after: dict[str, Any] | None = None
    # The agent steps that produced this clip (`murea_agent.events`): what it saw,
    # planned, checked and did, with timings. Empty for older clips.
    trace: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class World(QuestsMixin, DynamicStateMixin):
    id: str = field(default_factory=lambda: _sid("w"))
    # Old session files have no owner; they deliberately belong to the default
    # user so upgrading never hides or loses an existing world.
    owner_id: str = DEFAULT_USER_ID
    title: str = "未命名世界"
    setting: str = ""  # the world in one sentence
    # Mainline: a long-term goal spanning many beats. The predicament of the opening second is
    # stored separately as opening_situation, so momentary states ("pinned down", "seconds left")
    # are no longer mistaken for the mainline.
    #
    # Without it the world only responds: one line, one clip, and after a few steps there is nowhere
    # to go. With a mainline every beat has a direction and the director can lead instead of waiting
    # for orders.
    #
    # The user can write it when creating the world; otherwise the director sets one in the first
    # beat, and it is locked from then on (changing it midway would be a different world and void
    # earlier clips).
    mainline: str = ""
    # Current objective: where the mainline stands next, updated by the director every beat.
    objective: str = ""
    # Mainline stages and milestones are dynamic: the title stays stable while the story moves
    # through stages.
    mainline_phase: str = ""
    mainline_milestones: list[str] = field(default_factory=list)
    # Old worlds keep their original opening hook as background facts for the director, no longer as
    # a long-term quest.
    opening_situation: str = ""
    visual_style: str = ""  # immutable rendering Style Bible
    # One video route per world: continuation handles of different providers are incompatible, and
    # switching midway breaks the chain.
    video_model: str = ""
    here: str = ""
    head: str | None = None
    subjects: dict[str, dict[str, Any]] = field(default_factory=dict)
    places: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Scene switches persist as simple paths. Legacy entries may have extra
    # type/source keys; loading them remains harmless.
    place_links: list[dict[str, Any]] = field(default_factory=list)
    # Structured state of the mainline and side quests. Quest places use stable names from places,
    # so the frontend can pin quest badges on map nodes.
    quests: list[dict[str, Any]] = field(default_factory=list)
    nodes: dict[str, dict[str, Any]] = field(default_factory=dict)
    history: list[dict[str, str]] = field(default_factory=list)
    created: str = field(default_factory=_now)

    # ---------- persistence ----------

    @property
    def path(self) -> Path:
        return config.SESSIONS / f"{self.id}.json"

    def save(self) -> None:
        # Written to a temporary file and renamed, so a reader never sees a
        # half-written world (deletion would take it for one that uses no media).
        text = json.dumps(asdict(self), ensure_ascii=False, indent=2)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=f".{self.id}-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as handle:
                handle.write(text)
            os.replace(tmp, self.path)
        finally:
            Path(tmp).unlink(missing_ok=True)

    @classmethod
    def load(cls, world_id: str) -> World:
        raw = json.loads((config.SESSIONS / f"{world_id}.json").read_text())
        if not isinstance(raw, dict):
            raise ValueError(f"world {world_id!r} must contain a JSON object")

        # Session files are user data and can outlive the checkout that wrote
        # them.  Keep fields known by this version (including the legacy
        # ``place_links`` field above), while ignoring additive fields from a
        # newer/experimental checkout instead of making the whole world
        # unloadable.  Dataclass defaults supply fields introduced later, such
        # as ``visual_style``, when loading an older file.
        known = {f.name for f in fields(cls)}
        payload = {key: value for key, value in raw.items() if key in known}
        payload.setdefault("id", world_id)
        world = cls(**payload)
        world._migrate_scene_groups()
        world._migrate_dynamic_state()
        world._migrate_quests()
        return world

    @classmethod
    def list_all(cls, owner_id: str | None = None) -> list[dict[str, Any]]:
        # This used to be sorted(...glob(...), reverse=True), sorting by **file name**; ids are
        # random hex, so the order (and the world opened by default) was random. Sort by creation
        # time, newest first.
        out = []
        for p in config.SESSIONS.glob("*.json"):
            try:
                d = json.loads(p.read_text())
            except Exception:
                continue
            actual_owner = d.get("owner_id") or DEFAULT_USER_ID
            if owner_id is not None and actual_owner != owner_id:
                continue
            # length of the CURRENT lineage, not every node ever made -- undone
            # segments stay on disk so they can be branched from again.
            nodes, cur, n = d.get("nodes", {}), d.get("head"), 0
            while cur and cur in nodes:
                n += 1
                cur = nodes[cur].get("parent")
            out.append(
                {
                    "id": d["id"],
                    "title": d.get("title", ""),
                    "here": d.get("here", ""),
                    "segments": n,
                    "total_nodes": len(nodes),
                    "created": d.get("created", ""),
                    "owner_id": actual_owner,
                }
            )
        out.sort(key=lambda w: w["created"], reverse=True)  # newest first
        return out

    # ---------- assets ----------

    def add_subject(
        self,
        name: str,
        ref_filename: str,
        desc: str = "",
        control: str = "ai",
        brief: str = "",
        kind: str = "人物",
        voice: dict[str, Any] | None = None,
    ) -> str:
        """`control`: who drives this character.

        "user" is the one the user plays, "ai" is driven by the world. This is the line between "one
        person moving some puppets" and "a world with people in it": without it every character
        moves only when the user speaks and nobody in the world has intentions of their own.
        """
        # After deleting an item, never reuse an old ID via len + 1: historical nodes still point to
        # the asset of their time, and reuse would let the inspector and @ bindings mix up two
        # different assets.
        used_numbers = [
            int(match.group(1)) for key in self.subjects if (match := re.fullmatch(r"S(\d+)", str(key)))
        ]
        for node in self.nodes.values():
            historical = [*(node.get("subjects") or [])]
            historical.extend(
                picture.get("id") for picture in (node.get("pictures") or []) if isinstance(picture, dict)
            )
            for key in historical:
                match = re.fullmatch(r"S(\d+)", str(key or ""))
                if match:
                    used_numbers.append(int(match.group(1)))
        sid = f"S{max(used_numbers, default=0) + 1}"
        self.subjects[sid] = {
            "name": name,
            "ref": ref_filename,
            "desc": desc,
            "control": control,
            "brief": brief,
            # 人物 / 生物 / 物件 (character / creature / object). Anything that appears repeatedly needs its own
            # reference image, not only people: that mechanical monster must be the same one the
            # second time.
            "kind": kind,
            # Locked voice: a library entry or an upload. `ref` is the file name under data/voices
            # and is sent to the video model every time, so the same person always sounds the same,
            # just as reference images lock the look.
            # See murea_agent/memory/voices.py. (Old worlds store `{"instruct": "男，中年", "ref": ...}`; readers use .get(),
            # so they still work.)
            "voice": dict(voice or {"id": "", "name": "", "ref": "", "source": ""}),
            "state": _normalise_state(None, SUBJECT_STATE_DEFAULTS),
        }
        return sid

    def drop_item(self, sid: str) -> dict[str, Any]:
        """Remove an item asset without rewriting immutable generated segments."""
        subject = self.subjects.get(sid)
        if subject is None:
            raise KeyError(sid)
        if str(subject.get("kind") or "").strip() not in {"物件", "物品", "道具"}:
            raise ValueError("Only item assets can be deleted here")

        removed = self.subjects.pop(sid)
        name = str(removed.get("name") or "").strip()

        # Dynamic state describes only the current world. Historical nodes keep their subjects /
        # pictures / refs, otherwise the inspector of old clips would lose the reference assets
        # actually used.
        for other in self.subjects.values():
            state = other.get("state")
            relations = state.get("relations") if isinstance(state, dict) else None
            if isinstance(relations, dict):
                relations.pop(sid, None)
                if name:
                    relations.pop(name, None)
        if name:
            for place in self.places.values():
                state = place.get("state")
                props = state.get("props") if isinstance(state, dict) else None
                if isinstance(props, list):
                    state["props"] = [prop for prop in props if str(prop).strip() != name]

            # The critic appends an exact "holding {name}" part when registering a
            # picked-up item. Remove only that part, never clothing, injuries or other
            # continuity in the same carry.
            head = self.nodes.get(self.head or "")
            if isinstance(head, dict):
                held = {f"手里拿着{name}", f"holding {name}"}
                parts = [part.strip() for part in re.split(r"[；;]", str(head.get("carry") or ""))]
                sep = "；" if config.LANGUAGE == "zh" else "; "
                head["carry"] = sep.join(part for part in parts if part and part not in held)
        return removed

    def add_place(
        self,
        name: str,
        ref_filename: str,
        desc: str = "",
        main_scene: str = "",
        brief: str = "",
    ) -> None:
        requested = main_scene.strip()
        root = self.main_scene_for(requested) if requested in self.places else name
        self.places[name] = {
            "ref": ref_filename,
            "desc": desc,
            "brief": brief,
            "first_seen": _now(),
            "main_scene": root,
            "state": _normalise_state(None, PLACE_STATE_DEFAULTS),
        }

    # ---------- quests -------------------------------------------------------

    # ---------- dynamic state ------------------------------------------------

    def main_scene_for(self, name: str) -> str:
        """Return a valid group root; malformed/legacy values fall back to self."""
        if name not in self.places:
            return name
        root = str(self.places[name].get("main_scene") or "").strip()
        return root if root in self.places else name

    def set_place_main_scene(self, name: str, main_scene: str) -> str:
        """Move one scene (or an entire root group) under a valid main scene."""
        if name not in self.places:
            raise KeyError(name)
        requested = main_scene.strip() or name
        if requested not in self.places:
            raise ValueError("main scene not found")
        old_root = self.main_scene_for(name)
        new_root = name if requested == name else self.main_scene_for(requested)
        if name == old_root:
            for child_name, child in self.places.items():
                if child_name != name and self.main_scene_for(child_name) == old_root:
                    child["main_scene"] = new_root
        self.places[name]["main_scene"] = new_root
        return new_root

    def _migrate_scene_groups(self) -> None:
        """Collapse the old area/building/floor model into one main-scene key."""
        roots_by_location: dict[tuple[str, str, str], str] = {}
        for name, place in self.places.items():
            location = place.pop("location", None)
            current = str(place.get("main_scene") or "").strip()
            if current in self.places:
                continue
            if isinstance(location, dict):
                key = tuple(str(location.get(k) or "").strip() for k in ("area", "building", "floor"))
                if any(key):
                    place["main_scene"] = roots_by_location.setdefault(key, name)
                    continue
            place["main_scene"] = name
        # A child pointing at another child resolves to the same stable root.
        for name, place in self.places.items():
            current = name
            seen: set[str] = set()
            while True:
                if current in seen:
                    root = name
                    break
                seen.add(current)
                parent = str(self.places[current].get("main_scene") or "").strip()
                if parent not in self.places:
                    root = name
                    break
                if parent == current:
                    root = current
                    break
                current = parent
            place["main_scene"] = root

    # ---------- tree ----------

    @property
    def head_node(self) -> Node | None:
        if not self.head:
            return None
        return Node(**self.nodes[self.head])

    @staticmethod
    def node_of(raw: dict[str, Any]) -> Node:
        """Stored as a dict, used as a Node. Missing fields of old worlds get defaults."""
        fields = {f.name for f in dataclasses.fields(Node)}
        return Node(**{k: v for k, v in raw.items() if k in fields})

    def add_node(self, node: Node) -> None:
        prev = self.here
        self.nodes[node.id] = asdict(node)
        self.head = node.id
        self.here = node.place
        if prev and prev != node.place:
            self._link(prev, node.place)

    def _link(self, a: str, b: str) -> None:
        """Record "b is directly reachable from a".

        This is knowledge about **the world**, not the current story line: undoing a clip does not
        make the corridor and the meeting room non-adjacent, so links are only added.

        Adjacency used to be computed only in the frontend map dialog (WorldMap.tsx), never stored
        or sent back; the director only got an alphabetical list of place names. It did not know
        what connects to what, so "go back to the meeting room" could teleport, skipping spaces that
        had their own reference images.
        """
        wanted = frozenset((a, b))
        for e in self.place_links:
            if frozenset((e.get("from", ""), e.get("to", ""))) == wanted:
                e["n"] = int(e.get("n", 1)) + 1
                return
        self.place_links.append({"from": a, "to": b, "n": 1})

    def adjacency(self) -> dict[str, list[str]]:
        """Undirected adjacency. Walked once means connected; doors work both ways.

        Besides `place_links`, it is also derived from the clip tree: the field was added later and
        is empty in older worlds, which would otherwise never get a map. The union gives old worlds
        a map right away.
        """
        adj: dict[str, set[str]] = {name: set() for name in self.places}

        def link(a: str | None, b: str | None) -> None:
            if not a or not b or a == b:
                return
            adj.setdefault(a, set()).add(b)
            adj.setdefault(b, set()).add(a)

        for e in self.place_links:
            link(e.get("from"), e.get("to"))
        for raw in self.nodes.values():
            parent = self.nodes.get(raw.get("parent") or "")
            if parent:
                link(parent.get("place"), raw.get("place"))
        return {k: sorted(v) for k, v in sorted(adj.items())}

    def lineage(self) -> list[Node]:
        """Root -> head. This is the movie as currently assembled."""
        chain: list[Node] = []
        cur = self.head
        while cur:
            n = Node(**self.nodes[cur])
            chain.append(n)
            cur = n.parent
        return list(reversed(chain))

    def rollback(self) -> bool:
        """Undo one segment. The clip and its artifact stay on disk -- redoing or
        branching later just points at them again."""
        node = self.head_node
        if node is None:
            return False
        self.head = node.parent
        parent = self.head_node
        self.here = parent.place if parent else ""
        # Dynamic asset state follows the selected branch just like the video
        # latent.  Older nodes have no checkpoint and retain legacy behaviour.
        if node.state_before is not None:
            self.restore_dynamic_state(node.state_before)
        if node.quest_before is not None:
            self.restore_quest_state(node.quest_before)
        elif parent is not None:
            # Old nodes have no quest snapshot; at least restore the objective recorded by the
            # parent beat.
            self.objective = parent.objective
        # drop the last exchange so the GM does not think the undone beat happened
        while self.history and self.history[-1]["role"] == "model":
            self.history.pop()
        if self.history and self.history[-1]["role"] == "user":
            self.history.pop()
        return True

    # ---------- what the GM sees ----------

    def snapshot(self) -> dict[str, Any]:
        self._ensure_dynamic_state()
        self._migrate_quests()
        head = self.head_node
        started = bool(self.nodes)
        # A movement target is a visible anchor across turns. Carry only the previous beat's
        # distance plan, not the whole raw directive; when the user keeps walking the director
        # reuses the same destination and route instead of treating "walk forward" as a new action
        # without an end.
        previous_motion = (head.directive or {}).get("motion_plan") if head else None
        previous_motion_anchor = {}
        if isinstance(previous_motion, dict) and previous_motion.get("relation_change") == "distance":
            previous_motion_anchor = {
                "target": str(previous_motion.get("target") or "").strip(),
                "path": str(previous_motion.get("path") or "").strip(),
                "end_state": str(previous_motion.get("end_state") or "").strip(),
            }
        return {
            "setting": self.setting or "(not set)",
            # The mainline says where this world is heading; the objective says
            # where this beat should push. An unset mainline is defined this turn.
            "mainline": self.mainline or "(not set: define it this turn)",
            "mainline_phase": self.mainline_phase or "(not started)",
            "mainline_milestones": deepcopy(self.mainline_milestones),
            "opening_situation": self.opening_situation or "(none recorded)",
            "objective": self.objective or "(not set)",
            "quests": deepcopy(self.quests),
            "visual_style": self.visual_style or "(not locked)",
            "here": self.here or "(no scene yet)",
            "opening_hint": (
                ""
                if started or not self.here
                else f"The user supplied a reference image for {self.here!r}; the story "
                "must start there. Use it as the first segment's place."
            ),
            "segments_so_far": len(self.lineage()),
            "player_character": (
                [s.get("name", sid) for sid, s in self.subjects.items() if s.get("control") == "user"]
                or "(none yet: create one the first time the player's body is on screen)"
            ),
            "subjects": {
                sid: {
                    "name": s["name"],
                    "kind": s.get("kind", "人物"),
                    "voice": voices.label(s.get("voice")),
                    "description": s.get("desc", ""),
                    "state": deepcopy(s.get("state") or {}),
                    # Only the player's own character waits for commands; everyone
                    # else should act and react on their own every beat.
                    "driven_by": "user" if s.get("control") == "user" else "world (you play them)",
                }
                for sid, s in self.subjects.items()
            },
            "places": {
                name: (place.get("desc") or "").strip()[:70] or "(no description)"
                for name, place in sorted(self.places.items())
            },
            "place_states": {
                name: deepcopy(place.get("state") or {}) for name, place in sorted(self.places.items())
            },
            "characters_at_place": {
                name: [
                    subject.get("name", sid)
                    for sid, subject in self.subjects.items()
                    if (subject.get("state") or {}).get("location") == name
                ]
                for name in sorted(self.places)
            },
            "scene_groups": {name: self.main_scene_for(name) for name in sorted(self.places)},
            "connections": self.adjacency(),
            "previous_beat": head.beat if head else "(not started)",
            "previous_motion_anchor": previous_motion_anchor or "(the last beat was not a continuous move)",
            "carried": (head.carry if head and head.carry else "(nothing in particular)"),
        }
