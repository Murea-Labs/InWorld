"""Generating segments, and inspecting/re-rolling them."""

from __future__ import annotations

import concurrent.futures as futures
import threading
import time
import uuid
from dataclasses import asdict

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool

from murea_agent import config
from murea_agent import critic, loop, reqlog
from murea_agent.director import prefetch as director_prefetch
from app import option_prefetch
from murea_agent.providers import imagegen
from app.api import sse
from murea_agent.memory.world import World
from murea_agent.util import _say
from app.api.auth import CurrentUser, load_owned_world
from app.api.sse import GenerationBusy, mutation_slot, stream

router = APIRouter(prefix="/api/worlds", tags=["turns"])


# Items are discovered by the critic while the video plays. The last frame works as a temporary
# reference right away; the item's own image is drawn slowly in this small pool, never holding the
# video lock and never blocking /adjudicate. The job key includes the prompt, so after the user
# replaces the asset an old job is rejected by the idempotency check before it commits.
_ITEM_REF_POOL = futures.ThreadPoolExecutor(
    max_workers=2,
    thread_name_prefix="item-ref",
)
_ITEM_REF_GUARD = threading.Lock()
_ITEM_REF_JOBS: dict[tuple[str, str, str], futures.Future[None]] = {}
_ITEM_REF_COMMIT_RETRIES = 180


def _item_ref_key(world_id: str, sid: str, prompt: str) -> tuple[str, str, str]:
    return world_id, sid, " ".join(prompt.split())


def _mutate_item_with_retry(world_id: str, fn) -> bool:
    """Run a short world-file mutation after the image request completes.

    A new turn may own the per-world lock for its video call. Waiting here is harmless
    because this is a background worker; holding the lock during the remote image
    request would be harmful because it would reject/queue the user's next turn.
    """
    for _ in range(_ITEM_REF_COMMIT_RETRIES):
        try:
            with mutation_slot(world_id):
                return bool(fn())
        except GenerationBusy:
            time.sleep(1.0)
    return False


def _mark_item_ref_failed(
    world_id: str,
    owner_id: str,
    sid: str,
    prompt: str,
    source_ref: str,
    reason: str,
) -> None:
    def mutate() -> bool:
        try:
            world = World.load(world_id)
        except (FileNotFoundError, ValueError):
            return False
        if world.owner_id != owner_id:
            return False
        subject = world.subjects.get(sid)
        if not isinstance(subject, dict):
            return False
        if (
            subject.get("ref_status") != "pending"
            or subject.get("ref_prompt") != prompt
            or subject.get("source_ref") != source_ref
        ):
            return False
        subject["ref_status"] = "failed"
        subject["ref_source"] = "frame"
        subject["ref_error"] = reason[:240]
        world.save()
        return True

    try:
        _mutate_item_with_retry(world_id, mutate)
    except Exception as exc:  # noqa: BLE001 -- failure status is best-effort
        reqlog.record("item_reference_status", ok=False, sid=sid, error=f"{type(exc).__name__}: {exc}")


def _run_item_reference(
    world_id: str,
    owner_id: str,
    sid: str,
    name: str,
    prompt: str,
    source_ref: str,
    visual_style: str,
) -> None:
    token = reqlog.bind(world_id)
    generated_path = None
    temporary_path = None
    try:
        started = time.time()
        # ``prompt`` is the English appearance the critic wrote from the real last frame; object
        # kinds add reference constraints such as "single item, plain background, no people".
        raw = imagegen.generate(prompt, kind="object", visual_style=visual_style)
        ext, _ = imagegen.sniff(raw)
        filename = f"item_{uuid.uuid4().hex[:8]}{ext}"
        generated_path = config.REFS / filename
        temporary_path = config.REFS / f".{filename}.tmp"
        temporary_path.write_bytes(raw)
        temporary_path.replace(generated_path)

        def commit() -> bool:
            try:
                world = World.load(world_id)
            except (FileNotFoundError, ValueError):
                return False
            if world.owner_id != owner_id:
                return False
            subject = world.subjects.get(sid)
            if not isinstance(subject, dict):
                return False
            # The user may replace the image or delete the item while it is being drawn; never
            # overwrite the user's new choice or bring back a deleted asset.
            if (
                subject.get("ref_status") != "pending"
                or subject.get("ref_prompt") != prompt
                or subject.get("source_ref") != source_ref
            ):
                return False
            subject["ref"] = filename
            subject["ref_status"] = "ready"
            subject["ref_source"] = "generated"
            subject.pop("ref_error", None)
            subject["ref_generated_at"] = time.time()
            world.save()
            return True

        committed = _mutate_item_with_retry(world_id, commit)
        if not committed:
            generated_path.unlink(missing_ok=True)
            # If the world lock stays busy (rather than the user having replaced / deleted the
            # asset), do not leave the frontend on pending forever; this marker gives up by itself
            # when the check fields differ.
            _mark_item_ref_failed(
                world_id,
                owner_id,
                sid,
                prompt,
                source_ref,
                _say(
                    "世界持续生成，独立参考图未能提交",
                    "The world kept generating; the item reference was not submitted",
                ),
            )
            return
        reqlog.record(
            "item_reference",
            ok=True,
            sid=sid,
            name=name,
            source_ref=source_ref,
            seconds=round(time.time() - started, 3),
        )
    except Exception as exc:  # noqa: BLE001 -- a failed background enhancement must never affect the video path
        if generated_path is not None:
            generated_path.unlink(missing_ok=True)
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        _mark_item_ref_failed(
            world_id,
            owner_id,
            sid,
            prompt,
            source_ref,
            f"{type(exc).__name__}: {exc}",
        )
        reqlog.record("item_reference", ok=False, sid=sid, name=name, error=f"{type(exc).__name__}: {exc}")
    finally:
        reqlog.unbind(token)


def _schedule_item_reference(
    world_id: str,
    owner_id: str,
    sid: str,
    name: str,
    prompt: str,
    source_ref: str,
    visual_style: str,
) -> bool:
    prompt = " ".join(prompt.split())[:300]
    key = _item_ref_key(world_id, sid, prompt)
    with _ITEM_REF_GUARD:
        current = _ITEM_REF_JOBS.get(key)
        if current is not None and not current.done():
            return False
        future = _ITEM_REF_POOL.submit(
            _run_item_reference,
            world_id,
            owner_id,
            sid,
            name,
            prompt,
            source_ref,
            visual_style,
        )
        _ITEM_REF_JOBS[key] = future

    def forget(done: futures.Future[None]) -> None:
        with _ITEM_REF_GUARD:
            if _ITEM_REF_JOBS.get(key) is done:
                _ITEM_REF_JOBS.pop(key, None)

    future.add_done_callback(forget)
    return True


def _enqueue_picked_up_reference(
    wid: str,
    user: dict,
    node_id: str,
    picked_name: str,
) -> None:
    """Find the just-created pending item and enqueue its clean reference image."""
    if not picked_name:
        return
    try:
        world = load_owned_world(wid, user)
    except HTTPException:
        return
    except Exception as exc:  # noqa: BLE001 -- scheduling is best-effort
        reqlog.record("item_reference_schedule", ok=False, error=f"{type(exc).__name__}: {exc}")
        return
    for sid, subject in world.subjects.items():
        if not isinstance(subject, dict):
            continue
        if (
            str(subject.get("kind") or "").strip() not in {"物件", "物品", "道具"}
            or str(subject.get("name") or "").strip() != picked_name
            or str(subject.get("from_node") or "") != node_id
            or subject.get("ref_status") != "pending"
        ):
            continue
        prompt = str(subject.get("ref_prompt") or subject.get("desc") or "").strip()
        source_ref = str(subject.get("source_ref") or subject.get("ref") or "").strip()
        if not prompt or not source_ref:
            continue
        _schedule_item_reference(
            wid,
            str(user["id"]),
            sid,
            picked_name,
            prompt,
            source_ref,
            world.visual_style,
        )


def _done(events, wid: str, node, user: CurrentUser) -> None:
    world = load_owned_world(wid, user)
    # The saved video's frames and final world state are now available. Queue
    # planning before sending done; browser download/playback is not a prerequisite.
    option_prefetch.warm(world, source="video_done", expected_head=node.id, allow_busy=True)
    events.put({"type": "done", "reply": node.reply, "node": asdict(node), "world": asdict(world)})


@router.post("/{wid}/turn")
async def turn(wid: str, req: Request, user: CurrentUser) -> StreamingResponse:
    body = await req.json()
    request_id = body.get("request_id") or ""
    if not isinstance(request_id, str) or len(request_id) > 80:
        raise HTTPException(400, "invalid request_id")
    text = (body.get("text") or "").strip()
    if not text:
        raise HTTPException(400, "empty text")
    mentions = body.get("mentions")
    if mentions is not None and not isinstance(mentions, list):
        raise HTTPException(400, "mentions must be an array")
    if mentions is not None and any(not isinstance(value, str) for value in mentions):
        raise HTTPException(400, "mentions must contain asset IDs")
    # IDs are validated against the loaded world inside loop.turn. Keeping
    # this as a separate field means the visible @name text remains natural
    # language while the server still gets an unambiguous asset binding.
    mentions = [value.strip() for value in (mentions or []) if value.strip()]
    duration = body.get("duration")
    # "auto" = the frontend continued from the last beat's continue_with; not typed by the user
    origin = "auto" if body.get("auto") else "turn"
    load_owned_world(wid, user)

    def job(emit, events):
        w = load_owned_world(wid, user)
        # ⚠️ The binding must cover the **whole** turn, including the video part.
        # With bind/unbind inside loop.turn, the unbind ran in the finally of the image step, before
        # the video call, so no video_submit was ever logged. Request-scoped things belong in the
        # route.
        token = reqlog.bind(wid)
        try:
            # notify uses the same SSE: the director's line and new reference images arrive before
            # the video
            _done(
                events,
                wid,
                loop.turn(
                    w,
                    text,
                    progress=emit,
                    notify=events.put,
                    cancelled=sse.cancelled,
                    duration=duration,
                    origin=origin,
                    mentions=mentions,
                    request_id=request_id,
                ),
                user,
            )
        finally:
            reqlog.unbind(token)

    return stream(job, owner_id=user["id"], world_id=wid, request_id=request_id)


@router.get("/{wid}/turn-status/{request_id}")
def turn_status(wid: str, request_id: str, user: CurrentUser) -> dict:
    """Read-only recovery: never submit the video again after a transport failure."""
    # Read live status first: if the worker finishes during the subsequent load,
    # the saved node below wins. Ownership is still checked before returning data.
    status = sse.request_status(wid, request_id)
    world = load_owned_world(wid, user)
    for node in world.nodes.values():
        if node.get("request_id") == request_id:
            return {
                "status": "done",
                "event": {
                    "type": "done",
                    "node": node,
                    "world": asdict(world),
                    "reply": node.get("reply", ""),
                },
            }
    return status


@router.post("/{wid}/prefetch")
async def prefetch(wid: str, req: Request, user: CurrentUser) -> dict:
    """Run the director step for this text ahead of time and keep it until the user actually picks it.

    While the user still watches the previous clip, the recommended choice can already be planned,
    so generation starts the moment they click.

    Deliberately **not through `stream` and without the global lock**: the lock queues video jobs;
    prefetch is a single LLM call and joining that queue would only delay real generation.
    """
    body = await req.json()
    text = (body.get("text") or "").strip()
    # No prefetch while generating: head is about to change and the result would be stale.
    if not text or sse.busy(wid):
        return {"ok": True, "cached": False}
    w = load_owned_world(wid, user)
    # A "choice" is one of three fixed texts sent as is when clicked; a "draft" is a half-typed
    # sentence almost always invalidated by the next keystroke. Their cache value differs by an
    # order of magnitude, so they are kept apart (see `director.prefetch`: drafts once evicted the
    # recommended choice).
    kind = "draft" if body.get("kind") == "draft" else "option"
    cached = await run_in_threadpool(director_prefetch.prefetch, w, text, None, kind)
    return {"ok": True, "cached": cached}


@router.post("/cancel")
def cancel(user: CurrentUser) -> dict:
    """The user changed their mind: stop the running beat.

    The backend runs one beat at a time, so no world id is needed. A video request already sent
    cannot be recalled; what is guaranteed is that **this clip is not written into the world**, and
    the images and clip just made are removed.
    """
    return {"ok": True, "was_running": sse.cancel_current(user["id"])}


@router.post("/{wid}/regenerate")
async def regenerate(wid: str, req: Request, user: CurrentUser) -> StreamingResponse:
    """Re-roll the head segment with an edited prompt and/or a new seed."""
    body = await req.json()
    six, seed, duration = body.get("six_section") or None, body.get("seed"), body.get("duration")
    load_owned_world(wid, user)

    def job(emit, events):
        w = load_owned_world(wid, user)
        token = reqlog.bind(wid)  # the re-render's calls belong in the world's request log
        try:
            node = loop.regenerate(
                w, six_section=six, seed=seed, duration=duration, progress=emit, notify=events.put
            )
        finally:
            reqlog.unbind(token)
        _done(events, wid, node, user)

    return stream(job, owner_id=user["id"], world_id=wid)


@router.post("/{wid}/nodes/{nid}/note")
async def set_note(wid: str, nid: str, req: Request, user: CurrentUser) -> dict:
    body = await req.json()
    w = load_owned_world(wid, user)
    if nid not in w.nodes:
        raise HTTPException(404, "node not found")
    w.nodes[nid]["note"] = (body.get("note") or "")[:2000]
    w.save()
    return {"ok": True}


@router.get("/{wid}/siblings/{nid}")
def siblings(wid: str, nid: str, user: CurrentUser) -> list[dict]:
    """Every take of the same beat: the original plus each re-roll."""
    w = load_owned_world(wid, user)
    if nid not in w.nodes:
        raise HTTPException(404, "node not found")
    parent = w.nodes[nid].get("parent")
    return sorted(
        (n for n in w.nodes.values() if n.get("parent") == parent), key=lambda n: n.get("created", "")
    )


@router.post("/{wid}/adjudicate")
async def adjudicate(wid: str, req: Request, user: CurrentUser) -> dict:
    """Look at the frames just rendered and replace the choices and the objective with ones that match the picture.

    The frontend calls it on `done`, when the clip starts playing; the user is watching and then
    thinking, so the call is free for them.

    Deliberately **not through `stream` and without the global lock**, like `prefetch`: the lock
    queues video jobs, and holding it would delay the next beat.

    Failures return `{"ok": false}` instead of an error: the node keeps the director's choices and
    play goes on.
    """
    body = await req.json()
    node_id = str(body.get("node") or "").strip()
    world = load_owned_world(wid, user)
    raw = world.nodes.get(node_id) if node_id else None
    node = World.node_of(raw) if raw else world.head_node
    if node is None:
        raise HTTPException(404, "No such clip")

    def run() -> dict:
        # ⚠️ Bind like turn does: without it reqlog drops entries with no world, and the critic and its
        # LLM call would be invisible.
        token = reqlog.bind(wid)
        try:
            if world.head != node.id or sse.busy(wid):
                return {"ok": False}
            before = asdict(world)

            def commit() -> bool:
                # Model inference and new-asset drawing run outside this lock.
                # Head alone is insufficient: users can edit assets or undo while
                # the judge works. Compare the entire snapshot before writing.
                try:
                    with mutation_slot(wid):
                        current = load_owned_world(wid, user)
                        if asdict(current) != before:
                            return False
                        world.save()
                        return True
                except (GenerationBusy, HTTPException):
                    return False

            verdict = critic.adjudicate(world, node, commit=commit)
            if verdict:
                option_prefetch.warm(world, source="adjudicated", expected_head=node.id)
        finally:
            reqlog.unbind(token)
        # Items are registered with the last frame first and their own image is drawn in the
        # background. Queued after the critic is saved, so drawing never adds to the critical path.
        if verdict and verdict.get("picked_up"):
            _enqueue_picked_up_reference(
                wid,
                user,
                node.id,
                str(verdict.get("picked_up") or "").strip(),
            )
        out: dict = {"ok": bool(verdict), **(verdict or {})}
        # When something is picked up, return the world too: the item card must show in the assets
        # right away, not on the next beat.
        if verdict and (verdict.get("picked_up") or verdict.get("dropped") or verdict.get("appeared")):
            out["world"] = asdict(load_owned_world(wid, user))
        return out

    return await run_in_threadpool(run)
