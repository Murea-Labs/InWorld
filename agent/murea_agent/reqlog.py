"""Each world's outbound call log: what was sent, what came back, how long it took.

Why: a beat makes four kinds of external calls (director / rewriter / image / video). Nodes
store the final `prompt` and `six_section`, but those are results; without the real request
bodies of the calls in between, whether the model wrote it wrong or we sent it wrong is a guess.

Three deliberate rules:

1. **Memory plus one jsonl per world**, never the world file. The world file is user data;
   debug logs do not belong in it, nor in the exported zip.
2. **Never raise.** A logging failure must let the main flow continue; it is observation,
   not a feature.
3. **Under `config.DATA`**, already ignored by `data/*` in `.gitignore`. Request bodies
   contain raw prompts, as private as the world file.
"""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
from functools import wraps
import json
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

from murea_agent import config

# Which world an outbound call belongs to. A ContextVar instead of a parameter: `llm._post`
# is deep in the call stack, and threading a world id down would change a dozen signatures
# for something that is pure observation.
_world: contextvars.ContextVar[str] = contextvars.ContextVar("reqlog_world", default="")
_scope: contextvars.ContextVar[Mapping[str, Any]] = contextvars.ContextVar(
    "reqlog_scope", default=MappingProxyType({})
)


@contextmanager
def scope(**metadata):
    """Optional co-op operation context; single-player logging is unchanged."""
    token = _scope.set({**_scope.get(), **metadata})
    try:
        yield
    finally:
        _scope.reset(token)


def full() -> bool:
    return bool(_scope.get().get("full"))


def complete_text_request(formatter):
    """Opt into full co-op text while leaving the single-player formatter intact."""

    @wraps(formatter)
    def wrapped(body):
        if not full():
            return formatter(body)

        def text_only(value):
            if isinstance(value, dict):
                return {
                    k: text_only(v) for k, v in value.items() if k not in ("image_url", "inlineData", "data")
                }
            if isinstance(value, list):
                return [text_only(v) for v in value]
            return value

        # Whitelist text request fields; never headers, keys or image bytes.
        return text_only(
            {
                k: body[k]
                for k in (
                    "model",
                    "messages",
                    "contents",
                    "systemInstruction",
                    "generationConfig",
                    "response_format",
                    "temperature",
                    "max_tokens",
                    "chat_template_kwargs",
                    "stream",
                )
                if k in body
            }
        )

    return wrapped


_lock = threading.Lock()
_trim_ticks: dict[Path, int] = {}
# How many lines per world. About 4 per beat, so 500 is about 125 beats.
MAX_LINES = 500
# Truncate request / response bodies. The director's system prompt is tens of thousands of
# characters; storing it all would make this file bigger than the world, and debugging needs
# "what was sent this time", not the prompt template.
MAX_TEXT = 4000


def bind(world_id: str) -> contextvars.Token:
    return _world.set(world_id or "")


def unbind(token: contextvars.Token) -> None:
    try:
        _world.reset(token)
    except ValueError:  # reset from another thread, ignore
        pass


def current() -> str:
    return _world.get()


def _dir() -> Path:
    return config.DATA / "logs"


def path(world_id: str) -> Path:
    # the world id comes from a path parameter; block `../`
    safe = "".join(c for c in (world_id or "") if c.isalnum() or c in "-_")
    return _dir() / f"{safe}.jsonl"


def _clip(value: Any) -> Any:
    if isinstance(value, str) and len(value) > MAX_TEXT:
        return value[:MAX_TEXT] + f"… ({len(value) - MAX_TEXT} more characters)"
    return value


def record(kind: str, **fields: Any) -> None:
    """Record one entry. Any failure is swallowed."""
    wid = _world.get()
    if not wid:
        return
    try:
        row = {"t": time.time(), "kind": kind, **{k: v for k, v in _scope.get().items() if k != "full"}}
        row.update({k: v if full() else _clip(v) for k, v in fields.items()})
        line = json.dumps(row, ensure_ascii=False)
        p = path(wid)
        with _lock:
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
            _trim(p)
    except Exception:  # noqa: BLE001 -- observation must never affect the main flow
        pass


def _trim(p: Path) -> None:
    """Keep only the last MAX_LINES lines. Checked every 64 writes, not on every write."""
    try:
        tick = _trim_ticks.get(p, 0) + 1
        _trim_ticks[p] = tick % 64
        if len(_trim_ticks) > 1024:
            _trim_ticks.pop(next(iter(_trim_ticks)))
        if tick % 64:
            return
        if p.stat().st_size < 256_000:
            return
        lines = p.read_text(encoding="utf-8").splitlines()
        if len(lines) <= MAX_LINES:
            return
        p.write_text("\n".join(lines[-MAX_LINES:]) + "\n", encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass


def read(world_id: str, limit: int = 200) -> list[dict[str, Any]]:
    """Newest first."""
    p = path(world_id)
    if not p.is_file():
        return []
    try:
        lines = p.read_text(encoding="utf-8").splitlines()
    except Exception:  # noqa: BLE001
        return []
    out: list[dict[str, Any]] = []
    for line in reversed(lines[-max(1, limit) * 2 :]):
        try:
            out.append(json.loads(line))
        except Exception:  # noqa: BLE001 -- skip a half-written line
            continue
        if len(out) >= limit:
            break
    return out


def clear(world_id: str) -> None:
    try:
        path(world_id).unlink(missing_ok=True)
    except Exception:  # noqa: BLE001
        pass
