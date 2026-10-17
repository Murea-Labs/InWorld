"""Latency hiding for co-op action planning and recommendation speculation.

All cached work is guarded by signatures over the current world head, assets,
cast and exact action text.  A cache hit may skip work; it never relaxes the
normal reference checks or commits a video before the players accept it.
"""

from __future__ import annotations

from concurrent import futures
import copy
import fcntl
import hashlib
import json
import os
import threading
import time

from murea_agent import config
from murea_agent.coop import (
    duration as coop_duration,
    generation as coop_generation,
    trace as coop_trace,
    boss as coop_boss,
)
from murea_agent import reqlog
from murea_agent.providers import video
from murea_agent.memory.world import World
from murea_agent.memory.world_media import has_media, media_name, media_path
from murea_agent.coop.messages import msg

_POOL = futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="coop-prefetch")
_CANCEL_POOL = futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="coop-prefetch-cancel")
_LOCK = threading.Lock()
_PENDING_SUBMISSIONS: set[str] = set()
_ACTION_FUTURES: dict[str, futures.Future] = {}
_RECOMMENDATION_FUTURES: dict[str, futures.Future] = {}
_CANCELLED: set[str] = set()
_OBSERVATION_GRACE_SECONDS = 6.0


def _enabled(name: str) -> bool:
    # Prefetch the recommended prompt and its video by default so an exact
    # recommendation match can advance immediately. Deployments with limited
    # video capacity can still disable either stage explicitly.
    default = "1"
    return config.coop_env(name, default) not in ("0", "false", "False", "")


def _latest_node(room):
    rounds = room.get("rounds") or []
    return rounds[-1].get("node", {}) if rounds else room.get("opening_node", {}) or {}


def state_signature(room) -> str:
    """Hash every input that can change an action translation or video prompt."""
    node = _latest_node(room)
    cast = [
        {key: member.get(key, "") for key in ("user_id", "subject_id", "character", "description", "ref")}
        for member in room.get("members", [])
    ]
    monster = {
        key: room.get("monster", {}).get(key, "")
        for key in ("name", "description", "ref", "subject_id", "source", "from_node")
    }
    value = [
        room.get("world_id"),
        room.get("state_revision", room.get("revision", 0)),
        room.get("round"),
        node.get("id"),
        room.get("setting", ""),
        room.get("visual_style", ""),
        room.get("scene_name", ""),
        room.get("scene_description", ""),
        room.get("scene_ref", ""),
        room.get("scene_asset_source", ""),
        room.get("scene_asset_from_node", ""),
        room.get("asset_edit_revision", 0),
        monster,
        cast,
        coop_boss.retired_reference(room),
        (room.get("intro") or {}).get("world_event", ""),
        (room.get("intro") or {}).get("world_event_type", "none"),
        (room.get("intro") or {}).get("world_event_key", ""),
        (room.get("intro") or {}).get("boss_candidate", {}),
        {
            "duration_policy": coop_duration.policy_signature(),
            "prompt_version": coop_generation.PROMPT_VERSION,
            "planner_model": config.GM_MODEL,
            "video_model": config.VIDEO_MODEL,
        },
    ]
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def action_signature(room, user_id: str, action: str) -> str:
    return hashlib.sha256(f"{state_signature(room)}\0{user_id}\0{action}".encode()).hexdigest()


def _action_order(room, *, fallback_members: bool = True) -> list[str]:
    """Return the order in which actions were accepted.

    A recommendation is created before players submit anything, so it uses a
    deterministic member-order fallback.  Once players submit, the persisted
    ``action_order`` is authoritative and a different order deliberately makes
    the recommendation miss instead of silently changing narration/history.
    """
    members = room.get("members") or []
    known = [m.get("user_id") for m in members if m.get("user_id")]
    order = [uid for uid in (room.get("action_order") or []) if uid in known]
    if fallback_members:
        order.extend(uid for uid in known if uid not in order)
    return order


def recommendation_signature(room, actions: dict[str, str]) -> str:
    value = [state_signature(room), sorted(actions.items()), _action_order(room, fallback_members=True)]
    return hashlib.sha256(json.dumps(value, ensure_ascii=False).encode()).hexdigest()


def _suggested_actions(room) -> dict[str, str]:
    intro = room.get("intro") or {}
    if intro.get("status") != "ready":
        return {}
    result = {}
    expected = {member.get("user_id") for member in room.get("members", [])}
    for item in intro.get("suggestions") or []:
        if not isinstance(item, dict) or item.get("user_id") not in expected:
            return {}
        action = item.get("action")
        if not isinstance(action, str) or not action.strip() or item["user_id"] in result:
            return {}
        result[item["user_id"]] = action.strip()
    return result if set(result) == expected else {}


def _forget(mapping, key, future, request_id=""):
    """Remove only the future that was registered for this exact key."""
    with _LOCK:
        current = mapping.get(key)
        same_future = current is future
        if same_future:
            mapping.pop(key, None)
        # An invalidated request can be replaced by a new request with the same
        # deterministic signature. The old Future's callback must not clear the
        # new request's cancellation marker.
        if request_id and future.done() and same_future:
            _CANCELLED.discard(request_id)


def _register(mapping, key, future, request_id=""):
    with _LOCK:
        mapping[key] = future
        cancelled_before_register = bool(request_id and request_id in _CANCELLED)
    # ``Future.add_done_callback`` also invokes immediately for an already
    # completed Future. Register first so the immediate test/inline executor
    # path cannot leave a stale entry behind.
    future.add_done_callback(lambda done: _forget(mapping, key, done, request_id=request_id))
    # A room edit can supersede a recommendation in the tiny window between
    # ``submit`` and registry insertion. Preserve that cancellation marker and
    # withdraw a queued Future as soon as it becomes visible here.
    if cancelled_before_register:
        future.cancel()


def _future(mapping, key):
    with _LOCK:
        return mapping.get(key)


def _cancel_remote(job_id, world_id, round_number, request_id):
    with coop_trace.operation(world_id, round_number, request_id, "generation"):
        return video.cancel(job_id)


def cancel_job(job_id, room=None):
    if not job_id:
        return
    if room is None:
        _CANCEL_POOL.submit(video.cancel, job_id)
    else:
        request_id = (
            (room.get("inflight") or {}).get("request_id")
            or room.get("request_id")
            or (room.get("recommendation_prefetch") or {}).get("request_id", "")
        )
        _CANCEL_POOL.submit(_cancel_remote, job_id, room["world_id"], room["round"], request_id)


def _mark_cancelled(request_id: str, job_id: str = "", room=None) -> None:
    if not request_id:
        return
    with _LOCK:
        future = _RECOMMENDATION_FUTURES.get(request_id)
        # Keep the marker while a producer is between ``submit`` and registry
        # insertion. The pending set is updated under the same lock, so a late
        # invalidation cannot be mistaken for a completed request.
        if future is None and request_id not in _PENDING_SUBMISSIONS:
            _CANCELLED.discard(request_id)
        elif future is None or not future.done():
            _CANCELLED.add(request_id)
        else:
            _CANCELLED.discard(request_id)
    if future and not future.done():
        future.cancel()
    if job_id:
        cancel_job(job_id, room)


def _is_cancelled(request_id: str) -> bool:
    with _LOCK:
        return request_id in _CANCELLED


def _cleanup_rendered(rendered) -> None:
    if not rendered:
        return
    generation = rendered.get("generation") or {}
    clip = (
        generation.video_path.name if hasattr(generation, "video_path") else str(generation.get("clip", ""))
    )
    if media_name("clips", clip):
        media_path("clips", clip).unlink(missing_ok=True)
    for name in rendered.get("frames") or []:
        if media_name("frames", name):
            media_path("frames", name).unlink(missing_ok=True)


def invalidate(room, reason="房间或动作已变化", *, clear_actions=True) -> None:
    """Invalidate speculative work while holding the room lock; cancel remotely async."""
    if clear_actions:
        room.pop("action_prefetch", None)
    item = room.get("recommendation_prefetch") or {}
    if item.get("status") in ("planning", "prompt_ready", "rendering", "ready"):
        # A custom action or a world change invalidates the prefetch; never keep playing the stale
        # speculative clip.
        if room.get("preview_clip") == item.get("clip"):
            room.pop("preview_clip", None)
        request_id = item.get("request_id", "")
        rendered = item.get("_rendered")
        item.update(status="superseded", error=reason)
        _mark_cancelled(request_id, item.get("job_id", ""), room)
        item.pop("_prepared", None)
        item.pop("_rendered", None)
        item.pop("clip", None)
        item.pop("video_id", None)
        _cleanup_rendered(rendered)


def cancel_if_action_mismatch(room) -> None:
    item = room.get("recommendation_prefetch") or {}
    expected = item.get("actions") or {}
    if not expected or item.get("status") not in ("planning", "prompt_ready", "rendering", "ready"):
        return
    for member in room.get("members", []):
        action = member.get("action", "").strip()
        if action and action != expected.get(member.get("user_id")):
            invalidate(room, "玩家修改了系统推荐动作", clear_actions=False)
            return


def queue_action(code: str, user_id: str) -> bool:
    """Translate one exact ready action without waiting for the other players."""
    if not _enabled("COOP_ACTION_PREFETCH"):
        return False
    from murea_agent.coop import room as coop

    with coop._edit(code) as room:
        member = next((m for m in room.get("members", []) if m.get("user_id") == user_id), None)
        if (
            not member
            or not member.get("ready")
            or not member.get("action")
            or room.get("status") not in ("waiting", "failed")
            or not _latest_node(room)
        ):
            return False
        sig = action_signature(room, user_id, member["action"])
        old = (room.get("action_prefetch") or {}).get(user_id, {})
        if old.get("signature") == sig and old.get("status") in ("planning", "ready"):
            return True
        room.setdefault("action_prefetch", {})[user_id] = {
            "status": "planning",
            "signature": sig,
            "round": room["round"],
            "error": "",
        }
        snapshot = copy.deepcopy(room)
        selected = copy.deepcopy(member)
    lock_name = hashlib.sha256(sig.encode()).hexdigest()[:16]
    guard = (coop._directory() / f"{coop._code(code)}.action-{lock_name}.lock").open("a")
    try:
        try:
            fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            guard.close()
            return True
        # The action worker never acquires ``_LOCK``, so submit and registry
        # insertion can be one atomic section. A replacement action that lands
        # concurrently will then always find and cancel this exact Future.
        with _LOCK:
            future = _POOL.submit(_run_action, snapshot, selected, sig, guard)
            _ACTION_FUTURES[sig] = future
        future.add_done_callback(lambda done: _forget(_ACTION_FUTURES, sig, done))
        # If a newly queued translation is superseded before a worker starts,
        # ``Future.cancel`` skips ``_run_action`` and therefore its finally
        # block. Release the per-action lock from the cancellation callback.
        future.add_done_callback(lambda done: guard.close() if done.cancelled() else None)
        return True
    except BaseException as exc:
        from murea_agent.coop import room as coop

        with coop._edit(code) as room:
            current = (room.get("action_prefetch") or {}).get(user_id, {})
            if current.get("signature") == sig:
                current.update(status="failed", error=str(exc)[:300])
        guard.close()
        raise


def cancel_action_prefetch(room, user_id: str) -> None:
    """Cancel a superseded per-player translation, best effort.

    ``Future.cancel`` cannot stop a request already inside the model client, but
    removing it from the registry prevents a later consumer from treating it as
    current; the worker also performs an exact room/action signature check.
    """
    entry = (room.get("action_prefetch") or {}).pop(user_id, None)
    sig = entry.get("signature") if isinstance(entry, dict) else ""
    if not sig:
        return
    with _LOCK:
        future = _ACTION_FUTURES.get(sig)
    if future is not None:
        future.cancel()


def _run_action(snapshot, member, sig, guard):
    from murea_agent.coop import room as coop

    try:
        with coop_trace.operation(snapshot["world_id"], snapshot["round"], "action-" + sig[:16]):
            world = World.load(snapshot["world_id"])
            result = coop_generation.prefetch_member_action(world, snapshot, member)
            with coop._edit(snapshot["code"]) as room:
                current = (room.get("action_prefetch") or {}).get(member["user_id"], {})
                actual = next(
                    (m for m in room.get("members", []) if m.get("user_id") == member["user_id"]), {}
                )
                if (
                    current.get("signature") != sig
                    or state_signature(room) != state_signature(snapshot)
                    or not actual.get("ready")
                    or actual.get("action") != member.get("action")
                ):
                    reqlog.record("coop_action_prefetch_discarded", player=member["user_id"])
                    return None
                current.update(status="ready", result=result, error="")
                reqlog.record("coop_action_prefetch_ready", player=member["user_id"])
                return result
    except Exception as exc:
        with coop._edit(snapshot["code"]) as room:
            current = (room.get("action_prefetch") or {}).get(member["user_id"], {})
            if current.get("signature") == sig:
                current.update(status="failed", error=str(exc)[:300])
        reqlog.record("coop_action_prefetch_failed", ok=False, player=member["user_id"], error=str(exc)[:300])
        return None
    finally:
        guard.close()


def matching_actions(room) -> dict[str, dict]:
    entries = room.get("action_prefetch") or {}
    result = {}
    for member in room.get("members", []):
        item = entries.get(member.get("user_id"), {})
        expected = action_signature(room, member.get("user_id", ""), member.get("action", ""))
        if item.get("status") == "ready" and item.get("signature") == expected:
            result[member["user_id"]] = copy.deepcopy(item.get("result"))
    return result


def queue_recommendation(code: str) -> bool:
    """Prepare and optionally render the exact system-recommended joint action."""
    from murea_agent.coop import chapter as coop_chapter

    if not _enabled("COOP_RECOMMENDATION_PREFETCH"):
        return False
    from murea_agent.coop import room as coop, intro as coop_intro

    if coop_intro.queue_state_refresh(code):
        return True
    guard = (coop._directory() / f"{coop._code(code)}.prefetch.lock").open("a")
    handed_off = False
    resumed_from = ""
    resume_prepared = None
    try:
        try:
            fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        with coop._edit(code) as room:
            actions = _suggested_actions(room)
            node = _latest_node(room)
            if (
                coop_chapter.blocked(room)
                or not actions
                or not node.get("id")
                or room.get("status") not in ("waiting", "failed")
                or (room.get("guidance_refresh") or {}).get("status") == "planning"
                or (room.get("intro") or {}).get("signature") != coop_intro.signature(room)
                or (room.get("asset_extraction") or {}).get("status") == "generating"
                or (room.get("boss_detection") or {}).get("status") == "generating"
            ):
                return False
            # Once somebody types a different action this guess can no longer hit.
            if any(
                m.get("action") and m.get("action") != actions.get(m.get("user_id"))
                for m in room.get("members", [])
            ):
                return False
            sig = recommendation_signature(room, actions)
            old = room.get("recommendation_prefetch") or {}
            if old.get("signature") == sig:
                old_status = old.get("status")
                old_request_id = old.get("request_id", "")
                future = _future(_RECOMMENDATION_FUTURES, old_request_id)
                if old_status in ("planning", "prompt_ready", "rendering") and future is not None:
                    if not (
                        old_status == "prompt_ready"
                        and (old.get("awaiting_guidance") or old.get("awaiting_observation"))
                        and future.done()
                    ):
                        return True
                if old_status in ("ready", "consuming"):
                    return True
                # The prompt is persisted before video rendering starts. A deploy
                # can therefore leave a valid prompt_ready entry after its
                # in-memory Future disappears. Continue from that exact prompt
                # instead of treating the cache as live or planning it again.
                if old_status == "prompt_ready" and isinstance(old.get("_prepared"), dict):
                    if (
                        old.get("awaiting_observation")
                        and _observation_pending(room)
                        and time.time() < old.get("observation_wait_until", 0)
                    ):
                        return True
                    if not _enabled("COOP_SPECULATIVE_VIDEO") or room.get("recommendation_video_paused"):
                        return True
                    resumed_from = old_request_id
                    resume_prepared = copy.deepcopy(old["_prepared"])
                else:
                    invalidate(room, "新的下一步建议已生成")
            else:
                invalidate(room, "新的下一步建议已生成")
            # The signature remains the cache key; the request id is an
            # ownership token and must be unique across a supersede/retry of
            # the same action set. Otherwise a late callback from the old
            # Future can cancel or clear the new producer.
            request_id = (
                "recommend-" + hashlib.sha256(sig.encode()).hexdigest()[:16] + "-" + os.urandom(4).hex()
            )
            room["recommendation_prefetch"] = {
                "status": "prompt_ready" if resume_prepared is not None else "planning",
                "signature": sig,
                "request_id": request_id,
                "round": room["round"],
                "actions": actions,
                "error": "",
                "job_id": "",
            }
            if resume_prepared is not None:
                room["recommendation_prefetch"].update(
                    prompt=resume_prepared.get("prompt", ""),
                    _prepared=copy.deepcopy(resume_prepared),
                    observation_wait_until=old.get("observation_wait_until", 0),
                )
            snapshot = copy.deepcopy(room)
            for member in snapshot["members"]:
                member.update(action=actions[member["user_id"]], ready=True)
            # Keep the canonical order used by the recommendation signature.
            # If players later submit in another order, the signature will miss
            # and the normal path will preserve their real submission order.
            snapshot["action_order"] = _action_order(room, fallback_members=True)
            snapshot["request_id"] = request_id
            # Publish the pending marker before releasing the room lock. An
            # invalidating player edit that lands before ``ThreadPoolExecutor``
            # accepts the work can then leave a cancellation token for the
            # eventual Future registration.
            with _LOCK:
                _PENDING_SUBMISSIONS.add(request_id)
        try:
            # The pending marker was published with the room entry above; hold
            # it until both submission and registry insertion are complete.
            try:
                future = _POOL.submit(_run_recommendation, snapshot, sig, request_id, guard, resume_prepared)
                _register(_RECOMMENDATION_FUTURES, request_id, future, request_id=request_id)
            finally:
                with _LOCK:
                    _PENDING_SUBMISSIONS.discard(request_id)
        except BaseException as exc:
            with coop._edit(code) as room:
                item = room.get("recommendation_prefetch") or {}
                if item.get("request_id") == request_id:
                    item.update(status="failed", error=str(exc)[:300])
            with _LOCK:
                _CANCELLED.discard(request_id)
                _PENDING_SUBMISSIONS.discard(request_id)
            reqlog.record("coop_recommendation_prefetch_submit_failed", ok=False, error=str(exc)[:300])
            return False
        if resume_prepared is not None:
            reqlog.record(
                "coop_recommendation_prefetch_resumed",
                previous_request_id=resumed_from,
                request_id=request_id,
                prompt=resume_prepared.get("prompt", ""),
            )
        # If a custom action cancels it while still queued in the pool, the producer never runs its
        # finally.
        future.add_done_callback(lambda done: guard.close() if done.cancelled() else None)
        future.add_done_callback(
            lambda done: _resume_after_guidance(code, request_id) if not done.cancelled() else None
        )
        handed_off = True
        return True
    finally:
        if not handed_off:
            guard.close()


def _resume_after_guidance(code, request_id):
    from murea_agent.coop import room as coop

    with coop._edit(code, persist=False) as room:
        item = room.get("recommendation_prefetch") or {}
        resume = (
            item.get("request_id") == request_id
            and (item.get("awaiting_guidance") or item.get("awaiting_observation"))
            and item.get("status") == "prompt_ready"
            and (room.get("guidance_refresh") or {}).get("status") != "planning"
        )
    if resume:
        queue_recommendation(code)


def _observation_pending(room):
    node_id = _latest_node(room).get("id")
    observation = room.get("story_observation") or {}
    return (
        observation.get("node_id") == node_id and observation.get("status") in ("queued", "generating")
    ) or any(job.get("node", {}).get("id") == node_id for job in room.get("_story_observation_pending", []))


def resume_after_observation(code):
    from murea_agent.coop import room as coop

    with coop._edit(code, persist=False) as room:
        request_id = (room.get("recommendation_prefetch") or {}).get("request_id", "")
    if request_id:
        _resume_after_guidance(code, request_id)


def _wake_observation_wait(code, request_id):
    try:
        _resume_after_guidance(code, request_id)
    except Exception as exc:
        reqlog.record("coop_observation_wait_resume_failed", error=str(exc)[:300])


def _serialize_rendered(rendered):
    gen = rendered["generation"]
    return {
        "node_id": rendered["node_id"],
        "frames": rendered["frames"],
        "frame_seconds": rendered["frame_seconds"],
        "submitted_prompt": rendered.get("submitted_prompt", ""),
        "submitted_six": rendered.get("submitted_six") or {},
        "used_prompt_fallback": bool(rendered.get("used_prompt_fallback")),
        "generation": {
            "video_id": gen.video_id,
            "clip": gen.video_path.name,
            "duration": gen.duration,
            "elapsed": gen.elapsed,
            "params": gen.params,
            "continued": gen.continued,
            "artifact_id": gen.artifact_id,
            "submitted_prompt": gen.submitted_prompt,
        },
    }


def _serialize_prepared(prepared):
    """Persist a prepared request without thread-local ownership objects."""
    result = {
        key: copy.deepcopy(value)
        for key, value in prepared.items()
        if key not in ("scene_plan", "world_snapshot", "turn_context")
    }
    scene_plan = prepared.get("scene_plan")
    if scene_plan is not None:
        result["scene_plan"] = scene_plan.to_legacy()
        result["scene_plan"]["base_revision"] = scene_plan.base_revision
    snapshot = prepared.get("world_snapshot")
    if snapshot is not None:
        result["world_snapshot"] = {
            "world_id": snapshot.world_id,
            "room_code": snapshot.room_code,
            "revision": snapshot.revision,
            "round_number": snapshot.round_number,
            "node_id": snapshot.node_id,
        }
    return result


def _restore_rendered(value):
    if not value:
        return None
    raw = value["generation"]
    return {
        **value,
        "generation": video.Generation(
            raw["video_id"],
            media_path("clips", raw["clip"]),
            raw["duration"],
            raw["elapsed"],
            raw.get("params") or {},
            raw.get("continued", False),
            raw.get("artifact_id"),
            raw.get("submitted_prompt", ""),
        ),
    }


def _still_current(code, request_id, sig):
    from murea_agent.coop import room as coop

    if _is_cancelled(request_id):
        return False
    with coop._edit(code, persist=False) as room:
        item = room.get("recommendation_prefetch") or {}
        return (
            item.get("request_id") == request_id
            and item.get("signature") == sig
            and sig == recommendation_signature(room, item.get("actions") or {})
            and item.get("status") in ("planning", "prompt_ready", "rendering", "ready", "consuming")
        )


def _run_recommendation(snapshot, sig, request_id, guard, prepared=None):
    from murea_agent.coop import room as coop

    rendered = None
    try:
        with coop_trace.operation(snapshot["world_id"], snapshot["round"], request_id):
            world = World.load(snapshot["world_id"])
            node = _latest_node(snapshot)
            if world.head != node.get("id"):
                raise ValueError("the node the next step was planned on changed")
            # Invalidation can race the handoff between room persistence and
            # worker submission. Check before the expensive planner call so a
            # stale recommendation is discarded without consuming model time.
            if not _still_current(snapshot["code"], request_id, sig):
                return None
            if prepared is None:
                prepared = coop_generation.prepare(
                    world,
                    "",
                    room=snapshot,
                    mentions=[m["subject_id"] for m in snapshot["members"]],
                )
                if not _still_current(snapshot["code"], request_id, sig):
                    return None
                with coop._edit(snapshot["code"]) as room:
                    item = room.get("recommendation_prefetch") or {}
                    if item.get("request_id") != request_id or item.get("status") not in (
                        "planning",
                        "consuming",
                    ):
                        return None
                    # Once a player accepts it the round owns the result; later stages must not fall
                    # back to a prefetch that can be invalidated.
                    if item.get("status") != "consuming":
                        item["status"] = "prompt_ready"
                    item.update(prompt=prepared["prompt"], _prepared=_serialize_prepared(prepared), error="")
                reqlog.record("coop_recommendation_prompt_ready", prompt=prepared["prompt"])
            if not _enabled("COOP_SPECULATIVE_VIDEO") or snapshot.get("recommendation_video_paused"):
                return {"prepared": prepared, "rendered": None}
            if not _still_current(snapshot["code"], request_id, sig):
                return None
            with coop._edit(snapshot["code"]) as room:
                item = room.get("recommendation_prefetch") or {}
                if item.get("request_id") != request_id or item.get("status") not in (
                    "prompt_ready",
                    "consuming",
                ):
                    return None
                if item.get("status") != "consuming":
                    if (room.get("guidance_refresh") or {}).get("status") == "planning":
                        item["awaiting_guidance"] = True
                        return {"prepared": prepared, "rendered": None}
                    if snapshot.get("intro", {}).get("state_basis") and _observation_pending(room):
                        deadline = (
                            item.get("observation_wait_until") or time.time() + _OBSERVATION_GRACE_SECONDS
                        )
                        item["observation_wait_until"] = deadline
                        if time.time() < deadline:
                            item["awaiting_observation"] = True
                            # Preparation and playback continue. A one-shot wakeup
                            # caps this wait even if the visual model is unavailable.
                            timer = threading.Timer(
                                max(0.01, deadline - time.time()),
                                _wake_observation_wait,
                                args=(snapshot["code"], request_id),
                            )
                            timer.daemon = True
                            timer.start()
                            return {"prepared": prepared, "rendered": None}
                    item["status"] = "rendering"

            def status(job):
                job_id = str(job.get("id") or "")
                if not job_id:
                    return
                with coop._edit(snapshot["code"]) as room:
                    item = room.get("recommendation_prefetch") or {}
                    current = item.get("request_id") == request_id and item.get("status") in (
                        "rendering",
                        "consuming",
                    )
                    if item.get("request_id") == request_id:
                        item["job_id"] = job_id
                if not current or _is_cancelled(request_id):
                    cancel_job(job_id, snapshot)

            def preview_ready(clip):
                # Publish a preview as soon as the video model returns a playable file; frame
                # extraction and committing to the world continue in the background.
                with coop._edit(snapshot["code"]) as room:
                    item = room.get("recommendation_prefetch") or {}
                    current = item.get("request_id") == request_id and item.get("status") in (
                        "rendering",
                        "consuming",
                    )
                    if not current or _is_cancelled(request_id):
                        return
                    item["clip"] = clip
                    if item.get("status") == "consuming":
                        room["preview_clip"] = clip
                        room["progress"] = msg("提前生成的视频已就绪，正在完成收尾")
                    reqlog.record("coop_speculative_video_preview_ready", clip=clip)

            rendered = coop_generation.render_prepared(
                prepared,
                progress=lambda *_: None,
                cancelled=lambda: _is_cancelled(request_id),
                on_status=status,
                on_video_ready=preview_ready,
            )
            if not _still_current(snapshot["code"], request_id, sig):
                _cleanup_rendered(rendered)
                return None
            serialized = _serialize_rendered(rendered)
            with coop._edit(snapshot["code"]) as room:
                item = room.get("recommendation_prefetch") or {}
                if item.get("request_id") != request_id or item.get("status") not in (
                    "rendering",
                    "consuming",
                ):
                    _cleanup_rendered(rendered)
                    return None
                # Keep consuming when the players accepted while rendering.
                if item.get("status") != "consuming":
                    item["status"] = "ready"
                item.update(
                    clip=rendered["generation"].video_path.name,
                    video_id=rendered["generation"].video_id,
                    _rendered=serialized,
                    error="",
                )
            reqlog.record(
                "coop_speculative_video_ready",
                video_id=rendered["generation"].video_id,
                prompt=prepared["prompt"],
            )
            return {"prepared": prepared, "rendered": rendered}
    except Exception as exc:
        if rendered:
            _cleanup_rendered(rendered)
        with coop._edit(snapshot["code"]) as room:
            item = room.get("recommendation_prefetch") or {}
            if item.get("request_id") == request_id and item.get("status") != "superseded":
                item.update(status="failed", error=str(exc)[:300])
        reqlog.record("coop_recommendation_prefetch_failed", ok=False, error=str(exc)[:300])
        raise
    finally:
        guard.close()


def take(room):
    """Claim an exact recommendation hit while the caller holds the room lock."""
    item = room.get("recommendation_prefetch") or {}
    actions = {m.get("user_id"): m.get("action", "").strip() for m in room.get("members", [])}
    exact_actions = bool(actions) and actions == item.get("actions")
    exact_signature = exact_actions and item.get("signature") == recommendation_signature(room, actions)
    if (
        not exact_actions
        or not exact_signature
        or item.get("status") not in ("planning", "prompt_ready", "rendering", "ready")
    ):
        # A different submission order has the same action text but a different
        # canonical timeline. Treat it like any other recommendation miss and
        # clean up a rendered draft; otherwise the stale clip can survive beside
        # the real round and its producer may continue writing into the room.
        if exact_actions and item.get("signature") and not exact_signature:
            invalidate(room, "玩家提交顺序与预生成顺序不同", clear_actions=False)
        cancel_if_action_mismatch(room)
        return None
    request_id = item.get("request_id", "")
    future = _future(_RECOMMENDATION_FUTURES, request_id)
    if future is None and item.get("_prepared"):
        # A completed prompt/video survives a process restart through room JSON.
        future = futures.Future()
        future.set_result(
            {
                "prepared": copy.deepcopy(item["_prepared"]),
                "rendered": _restore_rendered(item.get("_rendered")),
            }
        )
    if future is None:
        item.update(status="failed", error=msg("预生成工作进程已重启，已回退正常生成。"))
        return None
    item["status"] = "consuming"
    reqlog.record("coop_recommendation_prefetch_hit", status="waiting" if not future.done() else "ready")
    return request_id, future


def complete(request_id: str) -> None:
    with _LOCK:
        future = _RECOMMENDATION_FUTURES.get(request_id)
        # A timed-out consumer may call complete while the producer is still
        # unwinding. Keep the cancellation marker until that Future actually
        # finishes; otherwise the producer could pass ``_still_current`` and
        # publish a speculative result after the normal path has started.
        if future is not None and not future.done():
            _CANCELLED.add(request_id)
            return
        _RECOMMENDATION_FUTURES.pop(request_id, None)
        _CANCELLED.discard(request_id)


def abandon(room, request_id: str, reason: str = "推荐预生成未及时完成，已回退正常生成") -> None:
    """Invalidate a recommendation owned by a consumer that is falling back.

    ``invalidate`` intentionally leaves an accepted ``consuming`` result alone
    so a process restart can recover it. A foreground timeout is different: the
    normal round is about to start, so the old result and its serialized media
    must be removed and its producer must no longer be allowed to publish.
    """
    item = room.get("recommendation_prefetch") or {}
    if item.get("request_id") != request_id:
        return
    rendered = item.get("_rendered")
    job_id = item.get("job_id", "")
    item.update(status="superseded", error=reason)
    _mark_cancelled(request_id, job_id, room)
    item.pop("_prepared", None)
    item.pop("_rendered", None)
    item.pop("clip", None)
    item.pop("video_id", None)
    _cleanup_rendered(rendered)


def discard(bundle) -> None:
    if isinstance(bundle, dict):
        _cleanup_rendered(bundle.get("rendered"))


def recover(room) -> None:
    """Recover prefetches from the real run locks; removing a finished Future does not mean the process exited."""
    item = room.get("recommendation_prefetch") or {}
    if item.get("status") in ("planning", "prompt_ready", "rendering", "ready") and item.get(
        "signature"
    ) != recommendation_signature(room, item.get("actions") or {}):
        invalidate(room, "本轮生成依据已更新，重新准备推荐视频", clear_actions=False)
        return
    if item.get("status") not in ("planning", "rendering", "consuming"):
        return
    from murea_agent.coop import room as coop

    request_id = item.get("request_id", "")
    code = coop._code(room["code"])
    # The producer holds prefetch.lock before submitting to the pool, and the receiver holds
    # run.lock until commit. Together they cover the registration gap, completion callbacks,
    # cross-process polling and receiving results already on disk.
    with (
        (coop._directory() / f"{code}.run.lock").open("a") as run_guard,
        (coop._directory() / f"{code}.prefetch.lock").open("a") as prefetch_guard,
    ):
        try:
            for guard in (run_guard, prefetch_guard):
                fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return

        rendered = item.get("_rendered") or {}
        job_id = item.get("job_id") or rendered.get("generation", {}).get("video_id")
        if job_id:
            # Once World.save succeeded but the room is not updated yet, the video belongs to the
            # world; never cancel it remotely.
            world = World.load(room["world_id"])
            if any(node.get("video_id") == job_id for node in world.nodes.values()):
                room.pop("recommendation_prefetch", None)
                return
        if item.get("_prepared"):
            clip = rendered.get("generation", {}).get("clip", "")
            if has_media("clips", clip):
                item.update(status="ready", error="")
                return
            if not job_id:
                item.update(status="prompt_ready", error="")
                return

        # Reclaim only orphan jobs with no worker, no result to receive and nothing committed to the
        # world.
        item.update(status="failed", error=msg("服务重启中断了下一轮预生成，将在玩家提交时正常生成。"))
        _mark_cancelled(request_id, item.get("job_id", ""), room)
