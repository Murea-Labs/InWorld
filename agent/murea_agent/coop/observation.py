"""Durable, ordered video observations, independent of reference-asset jobs."""

from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import secrets

from fastapi import HTTPException

from murea_agent import config
from murea_agent.coop import guidance_checks
from murea_agent.coop import (
    room as coop,
    generation as coop_generation,
    intro as coop_intro,
    trace as coop_trace,
    story as coop_story,
    memory as coop_memory,
    boss as coop_boss,
)
from murea_agent import reqlog
from murea_agent.coop.messages import msg

_POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="coop-observation")
_PENDING = "_story_observation_pending"


def _cast(members):
    return sorted(
        tuple(m.get(key, "") for key in ("user_id", "subject_id", "ref", "character", "description"))
        for m in members
    )


def enqueue(room, snapshot, node):
    """Called under the room lock when a video commits, never for speculation."""
    node_id = node.get("id")
    if not node_id:
        return
    pending = room.setdefault(_PENDING, [])
    if any(item["node"]["id"] == node_id for item in pending) or any(
        item.get("node_id") == node_id
        for item in (room.get("story_state") or {}).get("persistent_history") or []
    ):
        return
    # Preserve the cast, requested actions and adopted forecast of this video;
    # the next round may start or the host may edit assets before we read it.
    job = {
        "code": room["code"],
        "world_id": room["world_id"],
        "round": 0 if snapshot.get("opening") else snapshot["round"],
        "node": {"id": node_id, "frames": node.get("frames") or []},
        "members": [
            {
                key: member.get(key, "")
                for key in ("user_id", "subject_id", "character", "description", "ref", "action")
            }
            for member in snapshot.get("members", [])
        ],
        "current_forecast": {} if snapshot.get("opening") else coop_intro.round_forecast(snapshot),
        "setting": snapshot.get("setting", ""),
        "shared_scene": {key: snapshot.get(key, "") for key in ("scene_name", "scene_description")},
        "shared_creature": {
            key: (snapshot.get("monster") or {}).get(key, "")
            for key in ("name", "description", "subject_id", "ref")
        },
        "asset_edit_revision": snapshot.get("asset_edit_revision", 0),
    }
    if snapshot.get("opening"):
        for member in job["members"]:
            member["action"] = ""
    pending.append(json.loads(json.dumps(job)))
    if (room.get("story_observation") or {}).get("status") != "generating":
        room["story_observation"] = {
            "status": "queued",
            "node_id": node_id,
            "round": job["round"],
            "error": "",
        }


def queue(code):
    """Reserve one worker per room; room reads also recover interrupted jobs."""
    guard = (coop._directory() / f"{coop._code(code)}.observation.lock").open("a")
    handed_off = False
    try:
        try:
            fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        with coop._edit(code) as room:
            if room.get("status") == "closed":
                room.pop(_PENDING, None)
                return False
            pending = room.get(_PENDING) or []
            if not pending:
                return False
            job = pending[0]
            room["story_observation"] = {
                "status": "generating",
                "node_id": job["node"]["id"],
                "round": job["round"],
                "error": "",
            }
        try:
            _POOL.submit(_run, code, guard)
            handed_off = True
        except Exception as exc:
            # Keep the durable queue so a later read can resume it. This state
            # is never used to gate playback, guidance or the next video.
            with coop._edit(code) as room:
                room["story_observation"].update(status="queued", error=msg("剧情观察等待重试。"))
            reqlog.record("coop_observation_submit_failed", ok=False, error=str(exc)[:300])
        return handed_off
    finally:
        if not handed_off:
            guard.close()


def _observe(job, state):
    images = coop_generation.frame_images(job["node"])
    if not images:
        return None
    payload = {
        "video": {"id": job["node"]["id"], "frame_count": len(images), "frames_are_ordered": True},
        "setting": job["setting"],
        "shared_scene": job["shared_scene"],
        "shared_creature": job["shared_creature"],
        "players": [
            {key: member[key] for key in ("user_id", "character", "description")} for member in job["members"]
        ],
        "current_forecast": job["current_forecast"],
        "requested_actions": [
            {key: member[key] for key in ("user_id", "character", "action")}
            for member in job["members"]
            if member["action"]
        ],
        "story_context": {"current_state": coop_memory.current(state), "visual_evidence_available": True},
    }
    story_plan = job.get("story_plan") or {}
    if story_plan:
        payload["story_plan"] = story_plan
    for attempt in range(2):
        try:
            plan = coop_trace.chat(
                system=coop_intro.observation_system() + (coop_story._PROGRESS_SYSTEM if story_plan else ""),
                schema=coop_story.observation_schema(
                    coop_intro.observation_schema(), story_plan, job["current_forecast"]
                ),
                model=config.GM_MODEL,
                thinking=0,
                temperature=0.1,
                tag="coop_story_observation",
                user=json.dumps(payload, ensure_ascii=False),
                images=images,
            )
            observation = guidance_checks._validate_observation(
                plan, job["members"], payload, allow_empty_players=True
            )
            if observation is None:
                raise ValueError("the story observation has no observation")
            raw_life = (plan.get("observation") or {}).get("boss_life")
            if raw_life is not None:
                observation["boss_life"] = coop_boss.validate(
                    raw_life, len(images), has_boss=bool(job["shared_creature"].get("name"))
                )
                if not observation["boss_life"]:
                    reqlog.record("coop_boss_life_invalid", ok=False, node_id=job["node"]["id"])
            try:
                observation["_fact_updates"] = coop_memory.validate_updates(
                    (plan.get("observation") or {}).get("fact_updates", []), job["members"], len(images)
                )
            except (ValueError, TypeError) as exc:
                reqlog.record("coop_state_updates_invalid", ok=False, error=str(exc)[:300])
            if not job["current_forecast"].get("world_event"):
                observation["event_outcome"] = "unknown"
            if story_plan:
                try:
                    observation["_story_progress"] = coop_story.validate_progress(
                        plan.get("story_progress"),
                        story_plan,
                        len(images),
                        job["members"],
                        job["current_forecast"],
                    )
                except (ValueError, TypeError, AttributeError) as exc:
                    # Keep valid visual evidence even if the optional progress
                    # extension is malformed. It must not fail the whole audit.
                    reqlog.record("coop_story_progress_invalid", ok=False, error=str(exc)[:300])
            return observation
        except Exception as exc:
            if attempt:
                raise
            payload["correction_required"] = str(getattr(exc, "detail", exc))[:300]


def _record_reflection(room, node_id, status, observation, error):
    """Close the round's agent trace with the observer's verdict (reflect phase)."""
    observed = observation or {}
    step = {
        "id": "reflect-" + node_id,
        "phase": "reflect",
        "name": "critic",
        "status": "done" if status in ("ready", "skipped") else "error",
        "summary": "看真实画面，更新剧情状态",
        "data": {
            k: v
            for k, v in {
                "status": status,
                "summary": observed.get("summary") or observed.get("narration"),
                "event_outcome": observed.get("event_outcome"),
                "error": error,
            }.items()
            if v
        },
        "started": None,
        "seconds": None,
    }
    clips = [room.get("opening_node") or {}] + [item.get("node") or {} for item in room.get("rounds") or []]
    for clip in clips:
        if clip.get("id") == node_id:
            clip["trace"] = [s for s in clip.get("trace") or [] if s.get("id") != step["id"]] + [step]


def _run(code, guard):
    try:
        while True:
            with coop._edit(code) as room:
                pending = room.get(_PENDING) or []
                if not pending or room.get("status") == "closed":
                    room.pop(_PENDING, None)
                    # Release while holding the room lock, so a simultaneous
                    # commit cannot enqueue work after we checked the queue
                    # and mistake this departing worker for its new owner.
                    guard.close()
                    return
                job = pending[0]
                job["story_plan"] = coop_story.context(room) if coop_story.enabled() else {}
                state = json.loads(json.dumps(room.get("story_state") or {}))
                request_id = "observation-" + secrets.token_hex(12)
                room["story_observation"] = {
                    "status": "generating",
                    "node_id": job["node"]["id"],
                    "round": job["round"],
                    "request_id": request_id,
                    "error": "",
                }
            with coop_trace.operation(job["world_id"], job["round"], request_id):
                observation = None
                error = ""
                try:
                    observation = _observe(job, state)
                    status = "ready" if observation else "skipped"
                except Exception as exc:
                    status = "failed"
                    error = "本轮剧情观察未完成，不影响继续冒险。"
                    reqlog.record(
                        "coop_story_observation_failed",
                        ok=False,
                        node_id=job["node"]["id"],
                        error=str(exc)[:500],
                    )
                with coop._edit(code) as room:
                    was_dead = coop_boss.status(room) == "dead"
                    pending = room.get(_PENDING) or []
                    if not pending or pending[0]["node"]["id"] != job["node"]["id"]:
                        continue
                    nodes = [(room.get("opening_node") or {}).get("id")]
                    nodes.extend((item.get("node") or {}).get("id") for item in room.get("rounds") or [])
                    if (
                        room.get("status") == "closed"
                        or room.get("world_id") != job["world_id"]
                        or job["node"]["id"] not in nodes
                    ):
                        status = "superseded"
                    else:
                        # Historical evidence still belongs to its video, but
                        # must not overwrite a newly uploaded asset or cast.
                        update_current = room.get("asset_edit_revision", 0) == job[
                            "asset_edit_revision"
                        ] and _cast(room.get("members", [])) == _cast(job["members"])
                        if observation:
                            coop_intro.publish_observation(
                                room, observation, job["node"]["id"], update_current=update_current
                            )
                        if coop_story.enabled():
                            coop_story.record_progress(
                                room, job, observation, status, update_current=update_current
                            )
                    pending.pop(0)
                    room["story_observation"].update(status=status, error=error)
                    _record_reflection(room, job["node"]["id"], status, observation, error)
                    boss_retired = not was_dead and coop_boss.status(room) == "dead"
                reqlog.record(
                    "coop_story_observation_complete",
                    ok=status == "ready",
                    node_id=job["node"]["id"],
                    status=status,
                )
                coop_story.queue(code)
                if boss_retired:
                    coop_intro.queue(code, history_only=True)
                coop_intro.queue_state_refresh(code)
                from murea_agent.coop import prefetch as coop_prefetch

                coop_prefetch.resume_after_observation(code)
    except HTTPException as exc:
        if exc.status_code != 404:
            raise
    finally:
        guard.close()
