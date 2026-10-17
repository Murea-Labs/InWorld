"""The director: one structured LLM call that plans the next beat and writes its video prompt."""

from __future__ import annotations

import contextvars
import copy
import threading
import math
from concurrent import futures
from typing import Any
from collections.abc import Callable

from murea_agent import config, tools
from murea_agent.director import agent_delta
from murea_agent.providers import llm, video
from murea_agent import reqlog
from murea_agent.memory import voices
from murea_agent.memory.world import World
from murea_agent.util import load_prompt, _pretty
from murea_agent.director.agent_schema import (
    AGENT_LATE_FIELDS,
    _agent_partial_ready,
    _agent_schema,
    _density_pool,
    _early_pool,
)
from murea_agent.guards.action_sequence import (
    _action_sequence_semantic_errors,
    _normalize_action_sequence,
)
from murea_agent.guards.density_audit import (
    _audit_single_event_density,
    _motion_replan_schema,
    _single_event_needs_density_audit,
)
from murea_agent.shot.definitions import (
    _locked_definitions,
)
from murea_agent.director.narration import (
    _clean_history,
)
from murea_agent.shot.soundscape import (
    SOUNDSCAPE_LOCKED,
    _place_soundscape,
)
from murea_agent.tools.frames import _frames_for_gm


# One call returns both the GM decision and the video prompt; see `_agent_step`.
AGENT_SYSTEM = load_prompt("agent.md").replace("{{VOICE_LIBRARY}}", voices.prompt_table())


class _DecisionWithAudit(dict):
    """One request's in-flight audit, never metadata accepted from the model."""

    def __init__(self, value, sequence, goal, seconds, future):
        super().__init__(value)
        self.sequence = sequence
        self.goal = goal
        self.seconds = seconds
        self.future = future


def _decide_early(
    world: World,
    user_text: str,
    on_partial: Callable[[dict[str, Any]], None],
    *,
    duration: float | None,
    asset_mentions: list[dict[str, Any]] | None,
    base: str = "",
) -> tuple[dict[str, Any], futures.Future[dict[str, Any]] | None]:
    """Run the director but return **before it finishes the whole JSON**.

    The model writes field by field in schema order. Submitting the video only needs the six-section
    prompt and the motion plan; `options` / `quest_updates` / `carry` are used after the video and
    are about 16% of the output. They come last (see `AGENT_LATE_FIELDS`), so the server can move on
    as soon as the six sections are written and let the rest finish while the video renders.

    Returns (a decision good enough to proceed, future for the remaining output). A None future
    means the model already finished and the old path is used.

    ⚠️ When merging back **take only `AGENT_LATE_FIELDS`**. Code in between rewrites `beat`, `recap`,
    `continue_with` and `_server_repairs` in place; updating with the whole result would overwrite
    the server's values with the model's originals.
    """
    # Must match the schema actually sent by _agent_step. Locked definitions
    # are omitted there; waiting for that absent field disables early delivery.
    schema = _agent_schema(bool(_locked_definitions(world)))
    seen: dict[str, Any] = {}
    ready = threading.Event()
    audit_future = None
    audit_sequence = None
    segment_seconds = _segment_seconds(world, duration)
    # Gemini accepts independent concurrent requests; never pass a leftover
    # OpenAI URL, which would silently send this audit to another backend.
    audit_base = config.OPENAI_BASE_BOOK if config.LLM_BACKEND == "openai" else ""
    overlap_audit = config.LLM_BACKEND == "gemini" or (
        config.LLM_BACKEND == "openai" and bool(audit_base) and audit_base != config.OPENAI_BASE
    )

    def watch(part: dict[str, Any]) -> None:
        nonlocal audit_future, audit_sequence
        seen.update(part)
        on_partial(part)
        # The auditor consumes only this completed ledger and the user goal.
        # Let the same semantic check run while GM writes the physical/video
        # details, on the separate background replica. The video model still waits for it.
        if overlap_audit and audit_future is None and "action_sequence" in seen:
            sequence = seen["action_sequence"]
            if not _action_sequence_semantic_errors(
                sequence, segment_seconds
            ) and _single_event_needs_density_audit(sequence, segment_seconds):
                audit_sequence = copy.deepcopy(sequence)
                audit_ctx = contextvars.copy_context()
                audit_future = _density_pool.submit(
                    audit_ctx.run,
                    _audit_single_event_density,
                    audit_sequence,
                    user_text,
                    segment_seconds,
                    base=audit_base,
                )
        if not ready.is_set() and _agent_partial_ready(seen, schema):
            ready.set()

    # ⚠️ The thread pool does not carry contextvars; without copy this call's log goes to no world
    ctx = contextvars.copy_context()
    fut = _early_pool.submit(
        ctx.run,
        lambda: _decide(world, user_text, watch, duration=duration, asset_mentions=asset_mentions, base=base),
    )
    while not ready.wait(0.02):
        if fut.done():
            break
    if fut.done():
        result, tail = fut.result(), None  # already finished (or failed; re-raise as is)
    else:
        result, tail = dict(seen), fut
    if audit_future is not None:
        result = _DecisionWithAudit(result, audit_sequence, user_text, segment_seconds, audit_future)
    return result, tail


def _merge_late(d: dict[str, Any], fut: futures.Future[dict[str, Any]] | None) -> None:
    """Merge back the tail written while the video rendered. Late fields only; see the warning in `_decide_early`.

    The code in between already rewrote `beat`/`recap`/`continue_with`/`_server_repairs` in place;
    updating with the whole result would overwrite the server's values with the model's originals.
    """
    if fut is None:
        return
    try:
        full = fut.result(timeout=120)
    except Exception as exc:  # noqa: BLE001
        # The six sections already went to the video model and the clip comes out as usual; what is
        # missing are options / carry etc.
        reqlog.record("agent_late", ok=False, error=f"{type(exc).__name__}: {exc}")
        return
    for key in AGENT_LATE_FIELDS:
        if key in full:
            d[key] = full[key]


# ---------------------------------------------------------------- the director step


def _segment_seconds(world: World, duration: float | None = None) -> int:
    """The clip length the director can really use, clamped the same way as video.generate."""
    longest = video.model(world.video_model).max_seconds
    if duration is None:
        return min(config.VIDEO_MAX_SECONDS, longest)
    return max(4, min(longest, int(round(float(duration or config.VIDEO_SECONDS)))))


def _auto_duration(world: World, duration: float | None) -> bool:
    """Whether the director picks the length of this beat."""
    return duration is None


def _segment_floor(world: World, duration: float | None = None) -> int:
    """Minimum clip length. With a fixed or explicit length the minimum is that value."""
    budget = _segment_seconds(world, duration)
    if _auto_duration(world, duration):
        return max(4, min(config.VIDEO_MIN_SECONDS, budget))
    return budget


def _duration_instruction(world: World, duration: float | None) -> str:
    budget = _segment_seconds(world, duration)
    if _auto_duration(world, duration):
        floor = _segment_floor(world, duration)
        return (
            f"Choose {floor}-{budget} seconds for the first segment (hard limit {budget}, "
            f"minimum {floor}) and put the integer in duration_seconds. Play one complete "
            "action per segment; split into steps only what cannot fit, at natural progress "
            "points, never claiming arrival early. Same-scene steps use mode=continue. "
            "events.estimated_seconds are realistic durations: never speed actions up or "
            f"cram several segments into one. If the content is shorter than {floor} "
            "seconds, extend the same action naturally instead of adding new events. "
            "Spread dialogue over steps at natural pauses so no line is cut off."
        )
    return f"Fixed {budget} seconds: duration_seconds must be {budget}."


def _generation_seconds(world: World, directive: dict[str, Any], duration: float | None) -> int:
    if not _auto_duration(world, duration):
        return _segment_seconds(world, duration)
    sequence = _normalize_action_sequence(directive.get("action_sequence"))
    steps = sequence.get("steps") or []
    events = (steps[0].get("events") or []) if steps else []
    estimated = sum(float(event.get("estimated_seconds") or 0) for event in events)
    budget = _segment_seconds(world, duration)
    if estimated > budget + 0.25:
        raise ValueError(
            f"This action needs about {estimated:g}s, over the {budget}s budget; it must be split"
        )
    floor = _segment_floor(world, duration)
    default = floor
    try:
        chosen = float(directive.get("duration_seconds") or estimated or default)
        if not math.isfinite(chosen):
            chosen = default
    except (TypeError, ValueError):
        chosen = estimated or default
    return max(floor, min(budget, math.ceil(max(chosen, estimated))))


def _agent_step(
    world: World,
    user_text: str,
    on_partial: Callable[[dict[str, Any]], None] | None = None,
    *,
    duration: float | None = None,
    asset_mentions: list[dict[str, str]] | None = None,
    schema: dict[str, Any] | None = None,
    tag: str = "agent",
    base: str = "",
    model: str = "",
) -> dict[str, Any]:
    """Merged version: one call returns the director's decision **and** the six-section video prompt.

    These used to be two sequential calls (~6.0 s); merged it is ~3.3 s. The saving is mostly the
    second call's fixed overhead, not generation time.

    The cost is that the model cannot see the `<Subject N>` numbers (known only after the server
    picks cast and place), so it writes names and the server substitutes them, see
    `_substitute_refs`.
    """
    seen_frames = _frames_for_gm(world.head_node)
    snapshot = world.snapshot()
    snapshot["duration_budget"] = _duration_instruction(world, duration)
    prompt_limit = video.model(world.video_model).max_prompt_chars
    if prompt_limit < config.VIDEO_MAX_PROMPT_CHARS:
        # Say it up front: a long shot used to be cut afterwards, and the cut is the shot's end.
        # The server adds the reference and retention sections, about half of a full prompt.
        shot = max(800, prompt_limit * 2 // 5)
        snapshot["video_prompt_budget"] = (
            f"The video model reads about {prompt_limit} characters in all. Keep "
            f"detailed_description within about {shot} characters ({shot // 6} words): every "
            "action and sound fact once, no restating."
        )
    locked = _locked_definitions(world)
    if locked:
        snapshot["locked_definitions"] = locked
    # The place soundscape is locked → the model writes one word, saving output on the critical
    # path. See SOUNDSCAPE_LOCKED.
    locked_bed = _place_soundscape(world, world.here or "")
    sound_hint = ""
    if locked_bed:
        sound_hint = (
            f"\n\nThe ambient sound of {world.here!r} is locked. If this beat stays there, "
            f"write only `{SOUNDSCAPE_LOCKED}` in overall_soundscape; the server fills in "
            "the locked bed. Write it in full only when the beat moves to another place: "
            "that text becomes the new place's locked bed."
        )
    mention_hint = ""
    if asset_mentions:
        lines = "\n".join(
            f"- {item['kind']} {item['name']} (stable id: {item['id']})" for item in asset_mentions
        )
        mention_hint = (
            "\n\nThe user @-mentioned these assets; they must appear in this beat. "
            "Put characters and objects in subjects and a place in place. You may add "
            "other assets, but never drop these:\n" + lines
        )
    compact_gemini = config.LLM_BACKEND == "gemini" and schema is None
    use_delta = schema is None
    canonical = copy.deepcopy(schema) if schema is not None else _agent_schema(bool(locked))
    if "duration_seconds" in canonical.get("properties", {}):
        canonical["properties"]["duration_seconds"]["maximum"] = _segment_seconds(world, duration)
        canonical["properties"]["duration_seconds"]["minimum"] = _segment_floor(world, duration)
    if config.LLM_BACKEND == "openai" or compact_gemini:
        canonical = _motion_replan_schema(canonical)
    inherited = agent_delta.defaults(world) if use_delta else {}

    def expand_partial(part: dict[str, Any]) -> None:
        if on_partial is not None:
            expanded = agent_delta.expand(part, inherited)
            if _agent_partial_ready(expanded, canonical):
                agent_delta.validate_assets(expanded, world)
            on_partial(expanded)

    history, memory = _context(world)
    prompt = (
        (
            "The attached frames are from the previous clip; the last one is where the "
            "world stands now. Where they disagree with the beat you wrote last turn, "
            "trust the frames.\n\n"
            if seen_frames
            else ""
        )
        + f"World snapshot:\n{_pretty(snapshot)}{sound_hint}{mention_hint}{memory['note']}\n\n"
        f"The user says: {user_text}"
    )
    for attempt in range(2):
        result = llm.chat(
            system=AGENT_SYSTEM + (agent_delta.RULES if use_delta else ""),
            history=history,
            images=seen_frames,
            user=prompt,
            tools=memory["tools"],
            call_tool=memory["call"],
            # When every entity's wording is locked, do not ask for definitions: the server overwrites
            # the whole section and anything the model wrote is discarded. See `_agent_schema`.
            schema=agent_delta.schema(canonical, inherited) if use_delta else canonical,
            temperature=0.9,
            model=model or config.GM_MODEL,
            thinking=config.GM_THINKING,
            tag=tag,
            base=base,
            on_partial=expand_partial if use_delta and on_partial is not None else on_partial,
        )
        if not use_delta:
            return result
        try:
            if "world_delta" not in result:
                raise ValueError("Agent did not finish world_delta")
            expanded = agent_delta.expand(result, inherited)
            if not _agent_partial_ready(expanded, canonical):
                raise ValueError("Agent delta response is missing fields the video needs")
            agent_delta.validate_assets(expanded, world)
        except ValueError as exc:
            # Nothing of this answer reached the player: partials go through the same checks and
            # are never ready when the whole answer fails them. One re-plan with the concrete
            # problem is better than failing the beat (models without strict JSON schema, such as
            # DeepSeek, slip on structure now and then).
            if attempt:
                raise
            if isinstance(exc, agent_delta.UnregisteredAssetsError):
                reqlog.record("agent_unregistered_assets", ok=False, names=exc.names)
                prompt += f"\n\n{exc}"
            else:
                reqlog.record("agent_replan", ok=False, error=str(exc)[:300])
                prompt += (
                    f"\n\nYour last answer was invalid ({exc}). Answer again, following the schema exactly."
                )
            continue
        return expanded


def _decide(
    world: World,
    user_text: str,
    on_partial: Callable[[dict[str, Any]], None] | None = None,
    *,
    duration: float | None = None,
    asset_mentions: list[dict[str, str]] | None = None,
    base: str = "",
    schema: dict[str, Any] | None = None,
    model: str = "",
) -> dict[str, Any]:
    """What to ask the model for this beat. Merged: decision + video prompt in one call; otherwise only the decision.

    Prefetch uses **the same function**: if prefetching and the real beat ran different code, a
    cache hit would let the two paths drift apart silently.
    """
    return _agent_step(
        world,
        user_text,
        on_partial,
        duration=duration,
        asset_mentions=asset_mentions,
        base=base,
        **({"schema": schema} if schema is not None else {}),
        **({"model": model} if model else {}),
    )


def _context(world: World) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """The recent story the director reads, and its memory tools when needed.

    Only the last `agent.history_window` messages are sent. When older beats
    were left out, the director gets `recall_story` / `inspect_asset` to look
    them up instead of carrying the whole story in every prompt.
    """
    history = _clean_history(world.history)
    window = history[-config.HISTORY_WINDOW :]
    while window and window[0].get("role") != "user":
        window = window[1:]  # a conversation starts with the user
    trimmed = len(window) < len(history)
    offered = config.AGENT_TOOLS == "on" or (config.AGENT_TOOLS == "auto" and trimmed)
    if not offered:
        return window, {"note": "", "tools": (), "call": None}
    specs = tools.director_tools()
    note = (
        (
            "\n\nOnly the most recent part of the story is included above. To check an "
            "earlier event, person, place or item, call "
            + " / ".join(t.name for t in specs)
            + " before you answer."
        )
        if trimmed
        else ""
    )
    return window, {
        "note": note,
        "tools": [t.declaration() for t in specs],
        "call": lambda name, args: tools.call(name, world, args),
    }
