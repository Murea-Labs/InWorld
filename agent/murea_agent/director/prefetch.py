"""Plan likely next turns while the current clip plays, and hand them over when picked."""

from __future__ import annotations

import contextvars
import threading
import time
from concurrent import futures
from typing import Any
from collections.abc import Callable

from murea_agent import config
from murea_agent import reqlog
from murea_agent.memory.world import World
from murea_agent.util import _Clock
from murea_agent.shot.beat_assets import (
    _warm_images,
)
from murea_agent.events import bail
from murea_agent.director import director
from murea_agent.guards import replan


# ---- Director prefetch ---------------------------------------------------
#
# When the user clicks the recommended choice, video generation should start **right away** instead
# of waiting for the agent to think first. The director is a pure LLM call and can run while the
# user still watches the previous clip.
#
# ⚠️ Only the director runs ahead, **never the video**. The video model renders one job at a time
# (the global lock in api/sse.py exists for it). Rendering a clip nobody picked delays the one the
# user does pick; a right guess saves a little, a wrong one costs a whole clip, so the expectation
# is negative. A wrong director guess only wastes one LLM call.
#
# **All three choices are prefetched**, not just the recommended one. With only one, a sample of
# 109 beats hit 8 times (7%); users picking the second or third choice still waited for the
# director. Prefetching all three costs two extra small calls per beat, run while the user watches.
#
# ⚠️ Not free: three times the director calls, and shared endpoints rate limit (the backoff in
# `llm.py` exists for that). The frontend sends the recommended one first so that, under a tight
# quota, the most likely pick is still covered.
#
# The cache key includes head: undoing or moving the world forward invalidates it naturally. At
# most 4 entries (three choices + one typed draft), dropping the oldest.
_PREFETCH_MAX = 4

_prefetch_lock = threading.Lock()

# Prefetch runs in its own pool: three choices in parallel plus some room.
_prefetch_pool = futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="prefetch")

# Clicking a queued choice reuses the same future: take a finished one at once, wait for the rest of
# a running one. A timeout only ends this wait and keeps the future for a retry; never start a
# duplicate request.
_PREFETCH_JOIN_TIMEOUT = 30.0

_PREFETCH_JOIN_POLL = 0.1

_prefetched: dict[tuple[str, str | None, str, int], futures.Future[dict[str, Any] | None]] = {}


def _prefetch_key(
    world: World,
    text: str,
    duration: float | None = None,
) -> tuple[str, str | None, str, int]:
    # Normalize whitespace: spacing differences in how the frontend joins option text must not miss
    # the cache.
    budget = director._segment_seconds(world, duration)
    budget_key = -budget if duration is None else budget
    return (world.id, world.head, " ".join(text.split()), budget_key)


def prefetch(world: World, text: str, duration: float | None = None, kind: str = "option") -> bool:
    """Run the director step ahead and keep it for when the user clicks. Returns whether it was **queued**.

    ⚠️ What is stored is **the call itself** (a future), not its result.
    It used to be cached only after finishing, so a running prefetch was invisible to the server.
    Users clicking before it finished (each takes seconds, and many click within 10 s) made
    `_take_prefetched` see nothing and **run the same call again**: wasted, and the user waited
    anyway.

    With the future stored there are only two outcomes: finished → take it; running → join it.
    Never "running, yet run again".

    Fault tolerance is required: a failed prefetch must **never** affect a normal turn; it is a
    speed-up, not a feature.
    """
    if not world.head:  # not started yet, there is no next beat to prefetch
        return False
    key = _prefetch_key(world, text, duration)
    with _prefetch_lock:
        if key in _prefetched:
            return True  # already there or running; do not spend another call

    # ⚠️ Attach `_warm_images`. The user is watching the previous clip, so anything that can be done
    # now should be. Prefetch used to run only the director and reference images were drawn only
    # after the click, so beats needing a new image still paid seconds that could have been hidden.
    # Cost: if the two choices not picked each need a new image, those are drawn for nothing. That
    # only happens for beats with a new subject / place; otherwise `_warm_images` is a no-op.
    # `_WARM_MAX` still caps it, and unclaimed ones are cancelled by `_drop_warm` at the end of the
    # turn.
    def _warm_only(part: dict[str, Any]) -> None:
        try:
            _warm_images(world, part)
        except Exception:  # noqa: BLE001 -- a failed speed-up must not affect anything
            pass

    started, started_at = threading.Event(), []

    def _run() -> dict[str, Any] | None:
        started_at.append(time.monotonic())
        started.set()
        # bind / unbind must happen in the same Context. Binding in the calling thread and unbinding
        # only in the copy given to the background thread left this world id on the calling thread
        # forever, and later unrelated logs were written into the previous world.
        log_token = reqlog.bind(world.id)
        try:
            reqlog.record("prefetch_started", head=world.head, user_text=text, origin=kind)
            # Likewise: prefetch is background work during playback; use the background replica, do
            # not compete with the visible path.
            clock = _Clock()
            base = config._background_base()
            with clock.stage("gm"):
                d, tail = director._decide_early(
                    world, text, _warm_only, duration=duration, asset_mentions=None, base=base
                )
                if tail is not None:
                    # Prefetch must cache the fully validated response, while
                    # allowing its density audit to overlap streaming output.
                    full = tail.result()
                    d.clear()
                    d.update(full)
            if not isinstance(d, dict):
                return None
            d, _ = replan._prepare_action_plan(
                world,
                text,
                d,
                _warm_only,
                duration=duration,
                asset_mentions=None,
                clock=clock,
                base=base,
                is_continuation=(text == (world.nodes.get(world.head or "") or {}).get("continue_with")),
            )
            reqlog.record(
                "prefetch_ready", head=world.head, user_text=text, seconds=clock.total(), timings=clock.marks
            )
            return replan._PreparedDecision(d)
        except Exception as exc:  # optional acceleration may fail
            reqlog.record(
                "prefetch_failed", head=world.head, user_text=text, error=f"{type(exc).__name__}: {exc}"
            )
            return None
        finally:
            reqlog.unbind(log_token)

    # ⚠️ The thread pool does not carry contextvars; without copy this call's log goes to no world
    ctx = contextvars.copy_context()
    with _prefetch_lock:
        if key in _prefetched:
            return True
        fut = _prefetch_pool.submit(ctx.run, _run)
        # Record when it **really starts**. `_take_prefetched` uses it to judge whether waiting is
        # worth it; the submit time is not enough, the third queued choice may start well after its
        # submission.
        fut.murea_agent_started, fut.murea_agent_started_at = started, started_at
        fut.murea_agent_kind = kind
        # ⚠️ Keep only one draft, and drafts only evict drafts; never a choice.
        #
        # This was why the hit rate was low. It used to be one FIFO: after playback three choices
        # were queued (the recommended one **first**); typing added a draft every 400 ms pause, the
        # second draft exceeded the limit, and the oldest entry was dropped and cancelled: **the
        # recommended choice**. By the time the user clicked a choice its prefetch was gone. The
        # least useful entry evicted the most likely one.
        #
        # A draft is a half-typed sentence almost always invalidated by the next keystroke; a choice
        # is one of three fixed texts sent as is.
        if kind == "draft":
            for other in [
                k
                for k, f in _prefetched.items()
                if k != key and k[0] == world.id and getattr(f, "murea_agent_kind", "option") == "draft"
            ]:
                _prefetched.pop(other).cancel()
        _prefetched[key] = fut
        # Safety cap: only drop drafts; if all are choices let it exceed slightly rather than evict
        # a choice.
        while len(_prefetched) > _PREFETCH_MAX:
            drafts = [
                k
                for k, f in _prefetched.items()
                if getattr(f, "murea_agent_kind", "option") == "draft" and k != key
            ]
            if not drafts:
                break
            _prefetched.pop(drafts[0]).cancel()
    return True


def _join_budget(fut: futures.Future[Any]) -> float:
    """Bound the user's wait, not the background task's already elapsed work."""
    return 0.0 if fut.done() else _PREFETCH_JOIN_TIMEOUT


def _take_prefetched(
    world: World,
    text: str,
    duration: float | None = None,
    *,
    cancelled: Callable[[], bool] = lambda: False,
    on_wait: Callable[[], None] = lambda: None,
) -> dict[str, Any] | None:
    """Reuse the selected task, including its in-flight planning and audit.

    Only failed/empty results fall back to normal planning. Timeout/cancellation
    retain the task for a retry rather than silently starting duplicate work.
    Other worlds' tasks remain untouched.
    """
    key = _prefetch_key(world, text, duration)
    with _prefetch_lock:
        fut = _prefetched.get(key)
        if fut is None:
            return None
        for other in list(_prefetched):
            if other[0] == world.id and other != key:
                _prefetched.pop(other).cancel()
    started = time.monotonic()
    deadline = started + _join_budget(fut)
    joined = not fut.done()
    if joined:
        on_wait()
        reqlog.record("prefetch_join", state="waiting", head=world.head, user_text=text)
    consumed = False
    try:
        while True:
            bail(cancelled)
            remaining = deadline - time.monotonic()
            if not fut.done() and remaining <= 0:
                reqlog.record(
                    "prefetch_join",
                    state="timeout",
                    head=world.head,
                    user_text=text,
                    seconds=round(time.monotonic() - started, 3),
                )
                raise TimeoutError(
                    "This option is still being planned in the background; try again in a moment."
                )
            try:
                result = fut.result(timeout=min(_PREFETCH_JOIN_POLL, max(0.0, remaining)))
            except futures.TimeoutError:
                if not fut.done() or (not fut.cancelled() and fut.exception() is None):
                    continue
                # A worker itself raised TimeoutError: this is a failed task.
                consumed = True
                reqlog.record("prefetch_join", state="failed", head=world.head, user_text=text)
                return None
            except Exception:
                consumed = True
                reqlog.record("prefetch_join", state="failed", head=world.head, user_text=text)
                return None
            consumed = True
            reqlog.record(
                "prefetch_join",
                state="ready" if result is not None else "failed",
                joined=joined,
                head=world.head,
                user_text=text,
                seconds=round(time.monotonic() - started, 3),
            )
            return result
    finally:
        if consumed:
            with _prefetch_lock:
                if _prefetched.get(key) is fut:
                    _prefetched.pop(key)
