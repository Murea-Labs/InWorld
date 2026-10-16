"""Undo a committed co-op clip without deleting its media or latent artifact."""

import copy
import fcntl
from contextlib import ExitStack

from fastapi import HTTPException

from app.api.sse import GenerationBusy, mutation_slot
from murea_agent.coop import room as coop, events as coop_events, intro as coop_intro
from murea_agent.memory.world import World
from murea_agent.coop.messages import msg

_FIELDS = (
    "scene_ref",
    "scene_name",
    "scene_description",
    "scene_asset_source",
    "scene_asset_from_node",
    "monster",
    "asset_edit_revision",
    "story_state",
    "story_outline",
    "story_progress",
    "story_progress_history",
    "story_chapters",
    "chapter_transition",
    "asset_extraction",
)


def checkpoint(room, snapshot, node):
    saved = room.setdefault("_rollback_states", {})
    saved[node["id"]] = {key: copy.deepcopy(snapshot[key]) for key in _FIELDS if key in snapshot}
    while len(saved) > 51:
        del saved[next(iter(saved))]


def _legacy_state(room, node, world):
    """Reconstruct older rooms from persisted references and observed history."""
    history = (room.get("story_state") or {}).get("persistent_history", [])
    valid = {(r.get("node") or {}).get("id") for r in room.get("rounds", [])}
    valid.add((room.get("opening_node") or {}).get("id"))
    room["story_state"] = {}
    for item in history:
        if item.get("node_id") in valid and item.get("observation"):
            observation = copy.deepcopy(item["observation"])
            observation["_fact_updates"] = observation.pop("fact_updates", [])
            coop_intro.publish_observation(room, observation, item["node_id"])
    parent = (room.get("rounds") or [{}])[-1]
    room["story_progress"] = copy.deepcopy(parent.get("story_progress") or {})
    room["story_progress_history"] = [
        r for r in room.get("story_progress_history", []) if r.get("node_id") in valid
    ]
    chapter = room["story_progress"].get("chapter")
    if chapter and chapter != (room.get("story_outline") or {}).get("chapter"):
        old = next(
            (
                c
                for c in reversed(room.get("story_chapters", []))
                if (c.get("outline") or {}).get("chapter") == chapter
            ),
            None,
        )
        if old:
            room["story_outline"] = copy.deepcopy(old["outline"])
    room.pop("chapter_transition", None)
    for picture in node.get("pictures") or []:
        if picture.get("kind") == "scene":
            name = picture["name"]
            room.update(
                scene_name=name,
                scene_ref=picture["ref"],
                scene_description=(world.places.get(name) or {}).get("desc", ""),
            )
    if "monster" in (node.get("directive") or {}):
        room["monster"] = copy.deepcopy(node["directive"]["monster"])


def rollback(code, uid, expected_node):
    code = coop._code(code)
    with coop._edit(code) as room, ExitStack() as stack:
        coop._host(room, uid)
        coop._waiting(room)
        rounds = room.get("rounds") or []
        latest = rounds[-1]["node"] if rounds else room.get("opening_node")
        if not latest or latest["id"] != expected_node:
            raise HTTPException(409, msg("最新片段已变化，请刷新后再撤回"))
        # These workers write observations/assets; do not race their commits.
        for job in ("run", "story", "observation", "opening-assets", "round-boss"):
            guard = stack.enter_context((coop._directory() / f"{code}.{job}.lock").open("a"))
            try:
                fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise HTTPException(409, msg("片段仍在处理，请稍后再撤回")) from None
        try:
            stack.enter_context(mutation_slot(room["world_id"]))
        except GenerationBusy:
            raise HTTPException(409, msg("世界正在生成，请稍后再撤回")) from None
        world = World.load(room["world_id"])
        if world.head != expected_node:
            raise HTTPException(409, msg("世界片段已变化，请刷新后再撤回"))
        if world.head_node is None or not world.head_node.parent:
            raise HTTPException(409, msg("世界第一段不能撤回"))
        removed = rounds.pop() if rounds else None
        if removed is None:
            room.pop("opening_node", None)
        saved = (room.get("_rollback_states") or {}).pop(expected_node, None)
        if saved is not None:
            for key in _FIELDS:
                room.pop(key, None)
            room.update(copy.deepcopy(saved))
        else:
            _legacy_state(room, latest, world)
        world.rollback()
        world.here = room.get("scene_name", world.here)
        world.save()
        coop._unready(room)
        inputs = {p["user_id"]: p.get("action", "") for p in (removed or {}).get("inputs", [])}
        for member in room["members"]:
            member["action"] = inputs.get(member["user_id"], "")
            member.pop("immersive_session", None)
            member.pop("immersive_owner", None)
        for key in (
            "intro",
            "pending_guidance",
            "story_outline_job",
            "story_observation",
            "_story_observation_pending",
            "_opening_extraction_needed",
            "_round_boss_detection_needed",
            "boss_detection",
            "preview_clip",
            "generation_prompt",
            "generation_narration",
            "inflight",
            "immersive_sessions",
            "recommendation_prefetch",
        ):
            room.pop(key, None)
        if removed is None:
            room.pop("asset_extraction", None)
        room.update(
            round=removed["round"] if removed else 1, status="waiting", error="", progress="", request_id=""
        )
    coop_events.notify(code)
    return coop.get_room(code, uid, prepare_intro=False)
