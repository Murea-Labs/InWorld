"""Push persisted room changes; one event per revision, with reconnect recovery."""

import asyncio
from contextlib import contextmanager
import json
import threading
import time
from fastapi import HTTPException

_lock = threading.Lock()
_watchers = {}


def notify(code):
    with _lock:
        watchers = tuple(_watchers.get(code, ()))
    for loop, event in watchers:
        try:
            loop.call_soon_threadsafe(event.set)
        except RuntimeError:
            pass


@contextmanager
def subscribe(code):
    watcher = (asyncio.get_running_loop(), asyncio.Event())
    with _lock:
        _watchers.setdefault(code, set()).add(watcher)
    try:
        yield watcher[1]
    finally:
        with _lock:
            _watchers[code].discard(watcher)
            if not _watchers[code]:
                _watchers.pop(code)


async def stream(code, uid):
    from murea_agent.coop import room as coop

    path = coop._directory() / f"{code}.json"
    revision = -1
    stamp = None
    heartbeat = time.monotonic()
    with subscribe(code) as event:
        while True:
            event.clear()
            try:
                changed = path.stat().st_mtime_ns
                if changed != stamp:
                    # Atomic replacement makes an unlocked read safe. Recheck membership
                    # on every change, including when someone leaves the room.
                    room = await asyncio.to_thread(coop.get_room, code, uid, prepare_intro=False)
                    stamp = changed
                    if room["revision"] != revision:
                        revision = room["revision"]
                        yield f"id: {revision}\nevent: room\ndata: {json.dumps(room, ensure_ascii=False)}\n\n"
                if time.monotonic() - heartbeat >= 15:
                    yield ": keepalive\n\n"
                    heartbeat = time.monotonic()
            except (HTTPException, FileNotFoundError):
                yield "event: revoked\ndata: {}\n\n"
                return
            try:
                # Notifications are immediate within a worker. Stat-only fallback
                # also observes writes from another process without polling room JSON.
                await asyncio.wait_for(event.wait(), timeout=1)
            except TimeoutError:
                pass
