"""Explicit ownership and cancellation for automatically accepted co-op turns."""

from __future__ import annotations

from contextlib import contextmanager
from concurrent.futures import TimeoutError

from fastapi import HTTPException

from murea_agent.coop import (
    room as coop,
    ai as coop_ai,
    intro as coop_intro,
    prefetch as coop_prefetch,
)
from murea_agent import reqlog
from murea_agent.coop.messages import msg


def start(code, uid, session_id):
    with coop._edit(code) as room:
        coop._member(room, uid)
        if room["status"] == "closed":
            raise HTTPException(410, msg("房间已结束"))
        room.setdefault("immersive_sessions", {})[uid] = session_id
        room.pop("recommendation_video_paused", None)
    # A single entry sync also recovers guidance in historical rooms.
    return coop.get_room(code, uid)


def advance(code, uid, session_id, round_number):
    with coop._edit(code) as room:
        coop._member(room, uid)
        if room.get("immersive_sessions", {}).get(uid) != session_id:
            raise HTTPException(409, msg("沉浸模式已退出，请重新进入"))
        # Idempotent against duplicate requests and SSE arriving before HTTP.
        if room["round"] != round_number or room["status"] == "generating":
            return coop._public(room)
        coop._waiting(room)
        intro = room.get("intro") or {}
        actions = coop_prefetch._suggested_actions(room)
        if (
            not coop_prefetch._latest_node(room)
            or not actions
            or intro.get("signature") != coop_intro.signature(room)
        ):
            raise HTTPException(409, msg("下一步建议尚未就绪"))
        controlled = [m for m in room["members"] if room.get("mode") == "local" or m["user_id"] == uid]
        for member in controlled:
            coop._controlled(room, uid, member["user_id"])
            if not member.get("ref"):
                raise HTTPException(409, msg("请先设置角色形象"))
        for member in controlled:
            coop_prefetch.cancel_action_prefetch(room, member["user_id"])
            member.update(
                action=actions[member["user_id"]],
                ready=True,
                immersive_session=session_id,
                immersive_owner=uid,
            )
            coop._remember_action(room, member)
        coop_prefetch.cancel_if_action_mismatch(room)
        coop_ai.invalidate(room)
        room["error"] = ""
    if coop_ai.bots(room):
        coop_ai.queue(code)
    # Unlike a manual submit, immersive advancement also works when automatic
    # generation is disabled for an offline deployment.
    coop._launch(code, None, round_number, None, auto=True)
    return coop.get_room(code, uid)


def stop(code, uid, session_id):
    with coop._edit(code) as room:
        coop._member(room, uid)
        coop._recover(room)
        if room.get("immersive_sessions", {}).get(uid) != session_id:
            return coop._public(room)
        room["immersive_sessions"].pop(uid, None)
        inflight = room.get("inflight") or {}
        owns_turn = any(
            m.get("immersive_session") == session_id and m.get("immersive_owner") == uid
            for m in inflight.get("members", [])
        )
        can_cancel = owns_turn and not inflight.get("committed")
        if not inflight or owns_turn:
            room["recommendation_video_paused"] = True
        # An unrelated manual/other-player turn must keep its video and preview.
        if not inflight or can_cancel:
            if can_cancel:
                inflight["cancel_requested"] = True
                room["progress"] = msg("正在取消沉浸模式的下一段视频")
                if inflight.get("video_job_id"):
                    coop_prefetch.cancel_job(inflight["video_job_id"], room)
            item = room.get("recommendation_prefetch") or {}
            if item.get("status") == "consuming":
                coop_prefetch.abandon(room, item.get("request_id", ""), "已退出沉浸模式")
            elif item.get("status") != "ready":
                coop_prefetch.invalidate(room, "已退出沉浸模式", clear_actions=False)
            room.pop("preview_clip", None)
            room.pop("generation_narration", None)
        for member in room["members"]:
            if member.get("immersive_session") == session_id and member.get("immersive_owner") == uid:
                member["ready"] = False
                member.pop("immersive_session", None)
                member.pop("immersive_owner", None)
                coop._remember_action(room, member)
        coop_ai.invalidate(room)
        # intro is the valid forecast for the unplayed turn. Keep it unchanged;
        # pending_guidance predicts AFTER that turn and must stay private.
        reqlog.record("coop_immersive_stopped", cancelled=can_cancel)
    return coop.get_room(code, uid)


def cancelled(code, snapshot):
    with coop._edit(code, persist=False) as room:
        current = room.get("inflight") or {}
        return current.get("request_id") != snapshot["request_id"] or bool(current.get("cancel_requested"))


def check(code, snapshot):
    if cancelled(code, snapshot):
        raise RuntimeError("immersive generation was cancelled")


def status_callback(code, snapshot):
    def status(job):
        job_id = str(job.get("id") or "")
        if not job_id:
            return
        with coop._edit(code) as room:
            current = room.get("inflight") or {}
            current_turn = current.get("request_id") == snapshot["request_id"]
            if current_turn:
                current["video_job_id"] = job_id
            stop_job = not current_turn or current.get("cancel_requested")
        if stop_job:
            coop_prefetch.cancel_job(job_id, snapshot)

    return status


@contextmanager
def commit_guard(code, snapshot):
    # Cancellation and World.save are serialized: whichever wins this lock
    # determines whether the clip exists. A late exit never deletes a saved clip.
    with coop._edit(code) as room:
        current = room.get("inflight") or {}
        if current.get("request_id") != snapshot["request_id"] or current.get("cancel_requested"):
            raise RuntimeError("immersive generation was cancelled")
        yield
        current["committed"] = True


def finish_cancelled(code, snapshot):
    with coop._edit(code) as room:
        current = room.get("inflight") or {}
        if (
            current.get("request_id") != snapshot["request_id"]
            or not current.get("cancel_requested")
            or current.get("committed")
        ):
            return False
        room.pop("inflight", None)
        for key in ("preview_clip", "generation_prompt", "generation_narration"):
            room.pop(key, None)
        room.update(status="waiting", error="", progress="")
        reqlog.record("coop_round_cancelled", request_id=snapshot["request_id"])
        return True


def await_prefetch(code, snapshot, future):
    while True:
        check(code, snapshot)
        try:
            return future.result(timeout=0.25)
        except TimeoutError:
            if future.done():
                raise
