"""The co-op round loop: the same agent loop as single player, for a party.

    perceive  room, every player's submitted action, last frames, shared assets
    plan      the director locks each player's action and plans the environment
    check     world events and boss forecasts are validated
    act       stage the cast, compile one shared shot, render it
    commit    write the round into the world
    reflect   after playback, the observer watches the clip (`murea_agent.coop.observation`)

Every step is reported to `events` (the room shows them live to all players)
and saved on the clip as its trace. The stages live in `murea_agent.coop.generation`
(`prepare`, `render_prepared`, `commit_prepared`) because the speculative
recommendation path runs them separately; this module runs them in order.
"""

from __future__ import annotations

from contextlib import nullcontext

from murea_agent.coop import generation
from murea_agent.events import Emitter
from murea_agent.coop.messages import msg


def play_round(
    world,
    user_text,
    *,
    room,
    progress,
    cancelled,
    origin,
    mentions,
    on_video_ready=None,
    on_prompt_ready=None,
    on_narration_ready=None,
    pretranslated=None,
    on_status=None,
    commit_guard=None,
    events=None,
):
    """Play one round for the whole party and return the committed clip."""
    events = events or Emitter()
    progress("merge", msg("逐人保留原始动作，编排同一个镜头"))
    prepared = generation.prepare(
        world, user_text, room=room, mentions=mentions, pretranslated=pretranslated, events=events
    )
    if on_narration_ready:
        on_narration_ready(prepared["plan"]["narration"])
    if on_prompt_ready:
        on_prompt_ready(prepared["prompt"])
    rendered = generation.render_prepared(
        prepared,
        progress=progress,
        cancelled=cancelled,
        on_video_ready=on_video_ready,
        on_status=on_status,
        events=events,
    )
    try:
        with commit_guard() if commit_guard else nullcontext():
            if not commit_guard and cancelled():
                raise RuntimeError("joint generation was cancelled")
            return generation.commit_prepared(
                world,
                user_text,
                room=room,
                mentions=mentions,
                prepared=prepared,
                rendered=rendered,
                origin=origin,
                events=events,
            )
    except Exception:
        from murea_agent.coop import prefetch as coop_prefetch

        coop_prefetch._cleanup_rendered(rendered)
        raise
