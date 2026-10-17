"""`MureaMind`: the public entry point for using the agent from Python.

    from murea_agent import MureaMind

    agent = MureaMind()
    world = agent.create_world(title="Rainy Harbour", setting="A harbour town at night.")
    for event in agent.stream(world, "I step into the inn and look around."):
        print(event["type"], event.get("name", ""), event.get("summary", ""))

The web server (`app/`) is a thin layer over the same calls.
"""

from __future__ import annotations

import queue
import threading
from typing import Any
from collections.abc import Callable, Iterator

from murea_agent import config, critic, loop, reqlog, tools
from murea_agent.director import prefetch
from murea_agent.events import Cancelled
from murea_agent.memory.world import Node, World

Listener = Callable[[dict[str, Any]], None]


class MureaMind:
    """Plays an open world as generated video, one beat per user action.

    Each beat runs the fixed loop perceive -> plan -> check -> act -> commit
    (`murea_agent.loop`); `reflect` runs the critic on a finished clip. Every step is
    reported as an event and saved on the clip as its trace.
    """

    # ---- worlds ----------------------------------------------------------------

    def create_world(
        self,
        *,
        title: str = "",
        setting: str = "",
        mainline: str = "",
        visual_style: str = "",
        owner_id: str | None = None,
    ) -> World:
        world = World(
            title=title or "Untitled world",
            setting=setting,
            mainline=mainline,
            visual_style=visual_style,
            video_model=config.VIDEO_MODEL,
        )
        if owner_id:
            world.owner_id = owner_id
        world.save()
        return world

    def load_world(self, world_id: str) -> World:
        return World.load(world_id)

    # ---- one beat -------------------------------------------------------------

    def play(
        self,
        world: World,
        text: str,
        *,
        on_event: Listener | None = None,
        cancelled: Callable[[], bool] = lambda: False,
        duration: float | None = None,
        mentions: list[str] | None = None,
        auto: bool = False,
        request_id: str = "",
    ) -> Node:
        """Play one beat and return the saved clip. Blocks until the video is done.

        `on_event` receives every event: `step` (agent steps), `say` (the early
        reply), `sketch` (new references), `sequence`, `progress`.
        """
        listener = on_event or (lambda _event: None)
        token = reqlog.bind(world.id)  # every call of this beat goes to the world's log
        try:
            return loop.turn(
                world,
                text,
                progress=lambda stage, detail="": listener(
                    {"type": "progress", "stage": stage, "detail": detail}
                ),
                notify=listener,
                cancelled=cancelled,
                duration=duration,
                origin="auto" if auto else "turn",
                mentions=mentions,
                request_id=request_id,
            )
        finally:
            reqlog.unbind(token)

    def stream(self, world: World, text: str, **kwargs: Any) -> Iterator[dict[str, Any]]:
        """Like `play`, as a generator of events ending with `done` or `error`."""
        events: queue.Queue[dict[str, Any] | None] = queue.Queue()

        def run() -> None:
            try:
                node = self.play(world, text, on_event=events.put, **kwargs)
                events.put({"type": "done", "node": node})
            except Cancelled:
                events.put({"type": "cancelled"})
            except Exception as exc:  # noqa: BLE001 -- reported to the consumer
                events.put({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
            finally:
                events.put(None)

        threading.Thread(target=run, daemon=True).start()
        while (event := events.get()) is not None:
            yield event

    def continue_story(self, world: World, **kwargs: Any) -> Node | None:
        """Play the next queued step of a multi-clip action, if there is one."""
        head = world.head_node
        if head is None or not head.continue_with:
            return None
        return self.play(world, head.continue_with, auto=True, **kwargs)

    # ---- around a beat ----------------------------------------------------------

    def reflect(self, world: World, node: Node | None = None) -> dict[str, Any] | None:
        """Look at the rendered frames and reconcile the world (options, items)."""
        node = node or world.head_node
        if node is None:
            return None
        token = reqlog.bind(world.id)
        try:
            return critic.adjudicate(world, node)
        finally:
            reqlog.unbind(token)

    def plan_ahead(self, world: World, text: str) -> bool:
        """Plan a likely next action in the background while the clip plays."""
        return prefetch.prefetch(world, text)

    def regenerate(
        self,
        world: World,
        *,
        prompt_sections: dict[str, str] | None = None,
        seed: int | None = None,
        on_event: Listener | None = None,
    ) -> Node:
        """Re-render the latest clip, optionally from a hand-edited prompt."""
        listener = on_event or (lambda _event: None)
        return loop.regenerate(
            world,
            six_section=prompt_sections,
            seed=seed,
            notify=listener,
            progress=lambda stage, detail="": listener(
                {"type": "progress", "stage": stage, "detail": detail}
            ),
        )

    def undo(self, world: World) -> bool:
        """Step back one clip (the branch is kept and can be resumed)."""
        done = world.rollback()
        if done:
            world.save()
        return done

    @property
    def tools(self) -> list[tools.Tool]:
        return tools.all_tools()


# Backwards-compatible alias for callers using the former class name.
MureaAgent = MureaMind
