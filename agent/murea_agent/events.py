"""What the agent reports while it works.

A turn is a sequence of named steps grouped in phases:

    perceive  look at the world and the last frames
    plan      the director decides what happens next
    check     guards validate the plan; errors go back to the director
    act       tools draw references, write the video prompt, render the clip
    commit    the world is updated and saved
    reflect   after playback, the critic compares the clip with the plan

Every step is sent to the caller as an event while it runs and is saved on the
clip (`Node.trace`), so a past beat can be inspected the same way as a live one.
"""

from __future__ import annotations

import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Literal
from collections.abc import Callable, Iterator

Phase = Literal["perceive", "plan", "check", "act", "commit", "reflect"]
PHASES: tuple[Phase, ...] = ("perceive", "plan", "check", "act", "commit", "reflect")


class Cancelled(RuntimeError):
    """The user withdrew this turn. Not an error: nothing is shown as a failure."""


def bail(cancelled: Callable[[], bool]) -> None:
    if cancelled():
        raise Cancelled("cancelled by the user")


class Step:
    """One unit of agent work. Fill `summary` / `data` while it runs."""

    def __init__(self, emitter: Emitter, phase: Phase, name: str, summary: str = "") -> None:
        self._emitter = emitter
        self.id = uuid.uuid4().hex[:8]
        self.phase = phase
        self.name = name
        self.summary = summary
        self.data: dict[str, Any] = {}
        self.status = "running"
        self.started = emitter.elapsed()
        self.seconds: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "phase": self.phase,
            "name": self.name,
            "status": self.status,
            "summary": self.summary,
            "data": self.data,
            "started": self.started,
            "seconds": self.seconds,
        }


class Emitter:
    """Turns progress, steps and user-facing messages into events.

    `notify` receives every event dict (the server forwards them as SSE);
    `progress(stage, detail)` is the older one-line status channel, kept for
    the status bar.
    """

    def __init__(
        self,
        notify: Callable[[dict[str, Any]], None] | None = None,
        progress: Callable[[str, str], None] | None = None,
        earlier: list[dict[str, Any]] | None = None,
    ) -> None:
        # Steps recorded by an earlier emitter of the same beat, e.g. a plan
        # prepared in the background before the beat was actually played.
        self.earlier = [dict(step) for step in earlier or []]
        self._notify = notify or (lambda _event: None)
        self._progress = progress or (lambda _stage, _detail="": None)
        self._t0 = time.monotonic()
        self.steps: list[Step] = []

    def elapsed(self) -> float:
        return round(time.monotonic() - self._t0, 3)

    def progress(self, stage: str, detail: str = "") -> None:
        self._progress(stage, detail)

    def notify(self, event: dict[str, Any]) -> None:
        try:
            self._notify(event)
        except Exception:  # noqa: BLE001 -- a broken listener must not break the turn
            pass

    @contextmanager
    def step(self, phase: Phase, name: str, summary: str = "") -> Iterator[Step]:
        step = Step(self, phase, name, summary)
        self.steps.append(step)
        self.notify({"type": "step", **step.as_dict()})
        try:
            yield step
        except Cancelled:
            step.status = "cancelled"
            raise
        except BaseException as exc:
            step.status = "error"
            step.data.setdefault("error", f"{type(exc).__name__}: {exc}"[:300])
            raise
        else:
            if step.status == "running":
                step.status = "done"
        finally:
            step.seconds = round(self.elapsed() - step.started, 3)
            self.notify({"type": "step", **step.as_dict()})

    def record(self, phase: Phase, name: str, summary: str = "", **data: Any) -> None:
        """A step that already happened (no duration), e.g. a skipped stage."""
        with self.step(phase, name, summary) as step:
            step.data.update(data)

    def trace(self) -> list[dict[str, Any]]:
        return self.earlier + [s.as_dict() for s in self.steps]


_current: ContextVar[Emitter | None] = ContextVar("murea_agent_emitter", default=None)


def current() -> Emitter | None:
    """The emitter of the turn running in this context (tools report to it)."""
    return _current.get()


@contextmanager
def active(emitter: Emitter) -> Iterator[Emitter]:
    token = _current.set(emitter)
    try:
        yield emitter
    finally:
        _current.reset(token)
