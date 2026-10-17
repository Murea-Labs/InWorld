"""Server-sent events for long-running generation, with a global generation lock."""

from __future__ import annotations

from collections.abc import Iterator
from collections import OrderedDict
import contextvars
from contextlib import contextmanager
import json
import queue
import threading
import time
import traceback
from collections.abc import Callable

from fastapi.responses import StreamingResponse

from murea_agent import reqlog
from murea_agent.util import _say

# ⚠️ The lock covers only the **video** part, not the whole beat.
#
# This used to be one global lock around the whole turn (agent + video) shared by all users and
# worlds. A beat takes tens of seconds, so anyone else playing made you wait a whole beat, and
# N players meant waiting N beats.
#
# The agent's seconds are pure LLM calls and can run concurrently; only two things must be serial:
#
#   1. two beats of the same world  → concurrent writes would corrupt the world file. Lock per world.
#   2. requests to the video model  → one job at a time. Global, but only around video.generate.
#
# Split like this, agents of different people run concurrently and only queue to submit video.
_world_locks: dict[str, threading.Lock] = {}
_world_locks_guard = threading.Lock()


def _world_lock(world_id: str) -> threading.Lock:
    with _world_locks_guard:
        lock = _world_locks.get(world_id)
        if lock is None:
            lock = _world_locks[world_id] = threading.Lock()
        return lock


# Which world this beat belongs to. Undo and busy checks are per world, not a global flag;
# otherwise one person pressing stop would cancel someone else's beat.
_world_of: contextvars.ContextVar[str] = contextvars.ContextVar("sse_world", default="")

# Video jobs occasionally hang on the provider side (submitted but status never returns). New
# requests used to wait here forever while the lock holder waited for the client's 15 minute
# timeout. Queueing is normal; waiting forever is not.
_QUEUE_POLL_SECONDS = 2.0
_QUEUE_WAIT_LIMIT_SECONDS = 300.0

# The user changed their mind before the beat finished.
_cancel: set[str] = set()
_owner_guard = threading.Lock()
# world_id -> owner_id of the beats running now
_running: dict[str, str | None] = {}
# Track the request before its worker starts, including time spent in the queue.
# Keep only small failure/cancellation events; completed clips live in world nodes.
_active_requests: set[tuple[str, str]] = set()
_request_results: OrderedDict[tuple[str, str], dict] = OrderedDict()
_MAX_REQUEST_RESULTS = 256


def request_status(world_id: str, request_id: str) -> dict:
    with _owner_guard:
        key = (world_id, request_id)
        event = _request_results.get(key)
        if event is not None:
            return {"status": "failed" if event["type"] == "error" else "cancelled", "event": dict(event)}
        return {"status": "running" if key in _active_requests else "unknown"}


Emit = Callable[[str, str], None]


class GenerationBusy(RuntimeError):
    """Raised when a world mutation would race an in-flight generation."""


from murea_agent.events import Cancelled  # noqa: E402,F401  (re-exported for routes)


def cancel_current(owner_id: str | None = None) -> bool:
    """Ask to stop the running beat. Returns whether something was running.

    Only raises a flag, never kills the thread: a video request already sent cannot be recalled;
    what can be saved is the time **before** it, and not writing the clip into the world after.
    """
    with _owner_guard:
        mine = [w for w, o in _running.items() if owner_id is None or o == owner_id]
        if not mine:
            return False
        _cancel.update(mine)
    return True


def cancelled() -> bool:
    """Whether this beat was cancelled. **Per world.**

    This used to be one global Event: one person pressing stop cancelled every running beat.
    """
    world_id = _world_of.get()
    with _owner_guard:
        return world_id in _cancel if world_id else bool(_cancel)


def busy(world_id: str = "") -> bool:
    """Whether anything is running in this world.

    Used by the director prefetch: while generating, head is about to change and any prefetch is
    stale.
    ⚠️ Per world: another user's world running has nothing to do with this prefetch.
    """
    with _owner_guard:
        return (world_id in _running) if world_id else bool(_running)


@contextmanager
def mutation_slot(world_id: str = "") -> Iterator[None]:
    """Serialize short world mutations with generation's load/save window.

    ⚠️ Per world as well: editing assets of world A must not be blocked by world B generating.
    Old calls without world_id keep the global behavior (refuse while any beat runs).
    """
    if world_id:
        lock = _world_lock(world_id)
        if not lock.acquire(blocking=False):
            raise GenerationBusy("generation in progress")
        try:
            yield
        finally:
            lock.release()
        return
    with _owner_guard:
        running = bool(_running)
    if running:
        raise GenerationBusy("generation in progress")
    yield


def stream(
    job: Callable[[Emit, queue.Queue], None],
    owner_id: str | None = None,
    world_id: str = "",
    request_id: str = "",
) -> StreamingResponse:
    events: queue.Queue[dict | None] = queue.Queue()
    key = (world_id, request_id)
    if request_id:
        with _owner_guard:
            _active_requests.add(key)
            _request_results.pop(key, None)

    def terminal(event: dict) -> None:
        if request_id:
            # Save before sending: a disconnected browser can miss the SSE event.
            with _owner_guard:
                _request_results[key] = event
                while len(_request_results) > _MAX_REQUEST_RESULTS:
                    _request_results.popitem(last=False)
            reqlog.record("turn_terminal", request_id=request_id, event=event)
        events.put(event)

    def run() -> None:
        token = _world_of.set(world_id)
        log_token = reqlog.bind(world_id)
        lock = _world_lock(world_id)
        try:
            # ⚠️ Wait only for the previous beat of **the same world**; other worlds are unrelated.
            if not lock.acquire(blocking=False):
                events.put(
                    {
                        "type": "progress",
                        "stage": "queue",
                        "detail": _say(
                            "这个世界的上一段还在生成", "The previous clip of this world is still generating"
                        ),
                    }
                )
                queued_at = time.monotonic()
                while not lock.acquire(timeout=_QUEUE_POLL_SECONDS):
                    waited = time.monotonic() - queued_at
                    # The queue stage used to emit a single event; if the previous beat failed the
                    # user could not tell queued from dead. Periodic refreshes make the state
                    # visible and give the timeout a clear bound.
                    events.put(
                        {
                            "type": "progress",
                            "stage": "queue",
                            "detail": _say(
                                f"这个世界的上一段仍在生成，已等待 {int(waited)}s",
                                f"The previous clip is still generating; waited {int(waited)}s",
                            ),
                        }
                    )
                    if waited >= _QUEUE_WAIT_LIMIT_SECONDS:
                        raise GenerationBusy(
                            _say(
                                "上一段生成超过 5 分钟仍未结束，已停止等待；请点击停止后重新发送。",
                                "The previous clip has been generating for over 5 minutes; stopped "
                                "waiting. Press stop, then send again.",
                            )
                        )
            with _owner_guard:
                _running[world_id] = owner_id
                # Clear the flag after taking the lock: a flag raised for the previous beat must not
                # cancel this one.
                _cancel.discard(world_id)
            try:
                job(lambda s, d="": events.put({"type": "progress", "stage": s, "detail": d}), events)
            finally:
                with _owner_guard:
                    _running.pop(world_id, None)
                    _cancel.discard(world_id)
                lock.release()
        except Cancelled:
            # Undo is the user's action, not a failure: a separate event, so the frontend shows no
            # red bar
            terminal({"type": "cancelled"})
        except Exception as exc:
            terminal(
                {
                    "type": "error",
                    "message": f"{type(exc).__name__}: {exc}",
                    "trace": traceback.format_exc()[-1500:],
                }
            )
        finally:
            if request_id:
                with _owner_guard:
                    _active_requests.discard(key)
            reqlog.unbind(log_token)
            _world_of.reset(token)
            events.put(None)

    threading.Thread(target=run, daemon=True).start()

    def gen():
        while True:
            try:
                item = events.get(timeout=10)
            except queue.Empty:
                yield ": keep-alive\n\n"
                continue
            if item is None:
                break
            yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
