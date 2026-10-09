"""After a clip renders: reconcile the world with what was actually filmed."""

from __future__ import annotations

import re
import uuid
from typing import Any
from collections.abc import Callable
from murea_agent import config
from murea_agent.providers import imagegen, llm
from murea_agent import reqlog
from murea_agent.director.prompt_language import ensure_english_fields
from murea_agent.memory.world import Node, World
from murea_agent.memory.world_media import has_media, media_path
from murea_agent.util import _pretty, _say, cap, carrying, drop_carried
from murea_agent.guards.motion import (
    _clean_motion_text,
)
from murea_agent.shot.definitions import (
    _ITEM_KINDS,
)
from murea_agent.director.narration import _recommend_index
from murea_agent.shot.beat_assets import _timed_draw
from murea_agent.tools.frames import _frames_for_gm


# ─────────── Critic: look at the frames really rendered and say what this beat actually did ───────────
#
# Why a separate call: the three choices are written by the director **before rendering**, from its
# own plan, without knowing what the video model actually filmed. The result is choices that do not
# match the picture.
#
# The critic runs **during playback**: the clip is on disk, the frames are real, and the user is
# watching and thinking about the next step. So it is free for the user.
#
# ⚠️ It is an **upgrade**, not a replacement: the director still produces choices as a fallback. If the
# critic fails, times out or returns nothing, the beat is unaffected; same rule as prefetch
# (speed-ups and quality passes must never become new failure sources).
_ADJUDICATOR_SCHEMA = {
    "type": "object",
    "properties": {
        "observed": {"type": "string"},
        "options": {"type": "array", "items": {"type": "string"}},
        "recommend": {"type": "integer"},
        "objective": {"type": "string"},
        # Names of items **visibly** dropped / destroyed / used up in the frames. Empty array if
        # none.
        "dropped": {"type": "array", "items": {"type": "string"}},
        # Characters or creatures that appear in the narration but are not assets yet. Empty name if
        # none.
        "appeared": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "look": {"type": "string"},
            },
            "required": ["name", "look"],
        },
        # Things **really** picked up in the picture. Empty name if none.
        "picked_up": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "look": {"type": "string"},
                "scale_anchor": {"type": "string"},
            },
            "required": ["name", "look", "scale_anchor"],
        },
    },
    "required": ["observed", "options", "recommend", "objective", "picked_up", "dropped", "appeared"],
}

_ADJUDICATOR_SYSTEM = (
    "You are the referee of this world. You are given the frame that was **actually "
    "rendered** a moment ago, and what this beat was meant to show. Your job is not to "
    "re-plan but to **settle the facts**: whatever the frame shows is what happened.\n\n"
    "Write every human-readable field (`observed`, `options`, `objective`, every "
    "`name`) in {LANGUAGE}. `look` and `scale_anchor` are always English.\n\n"
    "- `observed`: one sentence of fact about the frame right now (where the people "
    "are, what they hold, what is in front of them). Only what is visible; no motives "
    "or feelings.\n"
    "- `options`: exactly three things the player can do next, **starting from the "
    "situation in this frame** -- if the frame shows them already inside the pipe, do "
    'not offer "jump into the pipe". One short sentence each. Keep any provided '
    "original option word for word, in the same order, as long as it still fits the "
    "frame; replace only options that conflict with the frame or are already done.\n"
    "- `recommend`: the index (0/1/2) of the option that best advances the main quest.\n"
    "- `objective`: one short sentence, what the world wants the player to do next.\n"
    "- `picked_up`: something that is **newly in a character's hands** in this frame.\n"
    '  - `name`: a short name ("brass lantern", "rusty wrench").\n'
    "  - `look`: one English sentence on its appearance -- material, colour, shape, "
    "light. It goes into later video prompts so the item keeps looking the same.\n"
    "  - `scale_anchor`: one English phrase for its relative size (`handheld, about "
    "25 cm tall`, `palm-sized`, `backpack-sized`). Only what the frame shows.\n"
    "  Only count items **visibly held**. Items on the ground, in someone else's hands "
    "or merely mentioned: leave `name` empty. Better to miss one than to record a "
    "wrong one -- a wrong item follows the character into every later shot. Items "
    "already in the inventory listed below: leave empty, even if still held. At most "
    "one item per beat.\n"
    "- `dropped`: inventory items that are **visibly** thrown away, broken, burnt or "
    "used up in this frame; copy names exactly from the inventory below, or an empty "
    "array. Judge by the frame, not by the story: no longer holding it does not count "
    "(it may be on their back or out of frame). A false report deletes the item and "
    "its reference image from the world.\n"
    "- `appeared`: characters or creatures that **entered in the narration** but are "
    "not yet among the known characters below.\n"
    "  - `name`: a short name.\n"
    "  - `look`: one specific English sentence on their appearance (build, material, "
    "colour, distinctive features); it becomes their reference image.\n"
    '  Judge by the **narration**, not the frame -- this field catches "the narration '
    'says a giant beast arrived but the frame shows nothing". Known characters, names '
    "that are only mentioned, and ambient sounds: leave empty.\n\n"
    "Output exactly these seven fields."
)

# Optional modifiers in names: the model writes "copper lantern" in one beat and "brass lamp" in the
# next for the same thing.
_ITEM_FILLER = re.compile(r"[一二三四五六七八九十]?[个只把件盏支根条]|的|旧|老|小|大|[\s，,。.·、]+")


def _item_key(name: str) -> str:
    return _ITEM_FILLER.sub("", " ".join(str(name or "").split()).casefold())


def _held_items(world: World) -> list[str]:
    """What is in the inventory now. The critic needs it to know what not to report again."""
    return [
        str(v.get("name") or "").strip()
        for v in world.subjects.values()
        if str(v.get("kind") or "").strip() in _ITEM_KINDS and str(v.get("name") or "").strip()
    ]


def _already_have(world: World, name: str) -> bool:
    """Whether this thing is already in the world (characters, places and items all count).

    ⚠️ Exact name matching is not enough: one beat says "copper lantern", the next "brass lamp", and one
    lamp becomes two items (this happened in a real story). So after normalization check containment
    both ways: drop quantifiers and modifiers so that "lantern" ⊂ "copper lantern" matches. Better
    to miss a new item than list the same thing twice: the extra one brings its own reference image
    into every later clip.
    """
    want = _item_key(name)
    if not want:
        return True
    names = [str((v or {}).get("name") or "") for v in world.subjects.values()]
    names += list(world.places)
    for other in names:
        key = _item_key(other)
        if not key:
            continue
        if key == want or key in want or want in key:
            return True
    return False


def _drop_reported_items(world: World, node: Node, raw: Any) -> list[str]:
    """Remove items visibly lost in the picture from the inventory.

    Items used to only ever be added, so the inventory grew forever and `carry` kept mentioning
    things long gone, which flowed into every video prompt.

    ⚠️ A name must **already be in the inventory**. The critic must copy names verbatim from the list,
    and the same normalization is applied as a guard, but **only among items**, never people or
    places: a wrong removal makes that thing and its reference image vanish from the world (past
    clips are unaffected; `drop_item` only changes the current assets).
    """
    if not isinstance(raw, list):
        return []
    items = {
        sid: str(v.get("name") or "").strip()
        for sid, v in world.subjects.items()
        if str(v.get("kind") or "").strip() in _ITEM_KINDS
    }
    gone: list[str] = []
    for said in raw[:3]:
        want = _item_key(said)
        if not want:
            continue
        hit = next(
            (
                sid
                for sid, name in items.items()
                if _item_key(name)
                and (_item_key(name) == want or _item_key(name) in want or want in _item_key(name))
            ),
            None,
        )
        if hit is None:
            continue
        name = items.pop(hit)
        try:
            world.drop_item(hit)
        except Exception:  # noqa: BLE001 -- if it cannot be removed, treat it as not reported
            continue
        gone.append(name)
        # Remove it from carry as well, or it keeps flowing into the video prompt
        carry = str(node.carry or "")
        if name and name in carry:
            node.carry = drop_carried(carry, name)
    return gone


def _known_characters(world: World) -> list[str]:
    """Characters / creatures that are already assets. The critic needs them to know who is new."""
    return [
        str(v.get("name") or "").strip()
        for v in world.subjects.values()
        if str(v.get("kind") or "人物").strip() not in _ITEM_KINDS and str(v.get("name") or "").strip()
    ]


def _register_appeared(world: World, node: Node, raw: Any) -> str:
    """Register characters who appear in the narration but were never registered.

    ⚠️ This fixes a real gap: in a sample of 353 clips, **2.5%** had narration saying something appears while
    `new_subjects` was empty. No asset means no reference image, so the six sections avoided drawing
    it; one beat said `fleeing from an unseen threat` and the video model faithfully rendered
    someone running from thin air, while the narration said a mechanical beast burst out.

    Why in the critic and not in the same beat: **no added latency**. The critic runs during
    playback, which is free time. The cost is that the character appears **one beat later**, but
    from then on it exists with a locked reference image instead of never.

    Images are drawn in `_warm_pool`, the same pool as early drawing, also off the user's time.
    """
    if not isinstance(raw, dict):
        return ""
    name = cap(raw.get("name"), 24)
    look = " ".join(str(raw.get("look") or "").split())[:300]
    if not name or not look or _already_have(world, name):
        return ""
    try:
        raw_png, _ = _timed_draw(look, "subject", world.visual_style)
        fn = f"subj_{uuid.uuid4().hex[:8]}{imagegen.sniff(raw_png)[0]}"
        (config.REFS / fn).write_bytes(raw_png)
    except Exception as exc:  # noqa: BLE001 -- a failed late registration must not affect anything
        reqlog.record("appeared", ok=False, name=name, error=f"{type(exc).__name__}: {exc}")
        return ""
    sid = world.add_subject(name, fn, desc=look, control="ai", brief=cap(look, 80))
    world.subjects[sid]["definition"] = look
    world.subjects[sid]["from_node"] = node.id
    reqlog.record("appeared", ok=True, name=name, node=node.id)
    return name


def _register_picked_up(world: World, node: Node, raw: Any) -> str:
    """Register things really picked up in the picture as item assets.

    The reference starts as the last frame right away; the critic route then queues ``look`` for a
    background image job. The asset appears immediately even with a slow image API, and later shots
    use a clean standalone item image once the job finishes.

    Items used to come from the director's plan ("there should be a lantern here"), drawn fresh from
    its `image_prompt` (seconds plus cost). But that drawn lantern was **not the one** the video
    model actually filmed: the director only mentioned it without knowing what it looked like.

    The critic looks at real frames and registers an item only when it is really in hand; the frame
    is already on disk (`node.frames`) and serves as ``source_ref`` and the temporary ``ref``. When
    the background job succeeds only ``ref`` changes; ``source_ref`` is kept as evidence.

    Returns the registered item name, or an empty string.
    """
    if not isinstance(raw, dict):
        return ""
    name = cap(raw.get("name"), 24)
    look = " ".join(str(raw.get("look") or "").split())[:300]
    scale_anchor = " ".join(str(raw.get("scale_anchor") or "").split())[:100]
    if not name or not look:
        return ""
    if _already_have(world, name):
        return ""  # already there (including under another wording); do not register twice
    frame = next((media_path("frames", fn) for fn in reversed(node.frames) if has_media("frames", fn)), None)
    if frame is None:
        return ""  # no frames, so nothing was really seen
    try:
        fn = f"item_{uuid.uuid4().hex[:8]}{frame.suffix or '.jpg'}"
        (config.REFS / fn).write_bytes(frame.read_bytes())
    except Exception:  # noqa: BLE001 -- a failed registration does not affect the beat
        return ""
    sid = world.add_subject(name, fn, desc=look, control="ai", brief=cap(look, 80), kind="物件")
    world.subjects[sid]["definition"] = look
    if scale_anchor:
        world.subjects[sid]["scale_anchor"] = scale_anchor
    # These fields make the background job idempotent: if the user replaces / deletes the asset
    # while it is being drawn, the old job cannot write its result back. Old worlds lack them and
    # keep working from the current ``ref``.
    world.subjects[sid]["source_ref"] = fn
    world.subjects[sid]["ref_prompt"] = look
    world.subjects[sid]["ref_status"] = "pending"
    world.subjects[sid]["ref_source"] = "frame"
    # ⚠️ Source frame: the item card must show which clip it came from.
    world.subjects[sid]["from_node"] = node.id
    # Things held in hand carry across scenes, or the lantern vanishes when the person leaves the
    # room
    carry = str(node.carry or "").strip()
    if name not in carry:
        node.carry = carrying(carry, name)
    return name


def adjudicate(
    world: World,
    node: Node,
    *,
    commit: Callable[[], bool] | None = None,
) -> dict[str, Any] | None:
    """Look at real frames and replace the choices and the objective with ones that match the picture. Edits node in place and saves.

    The model call and image drawing run on a private snapshot; the HTTP route uses commit to check
    the snapshot and save under a short write lock. Stale results cannot overwrite the next beat or
    hold the world lock while waiting for the model. Direct calls save by default.
    """
    frames = _frames_for_gm(node)
    if not frames:
        # Returning silently once made it impossible to tell whether the critic ran at all. Say so.
        reqlog.record("adjudicate", ok=False, node=node.id, error="this clip has no frames to look at")
        return None
    plan_line = _clean_motion_text(node.beat)
    ask = (
        f"Place: {node.place}\n"
        f"This beat was meant to show: {plan_line}\n"
        f"Main quest: {world.mainline or '(not set yet)'}\n"
        f"Objective given last beat: {node.objective or '(none)'}\n"
        f"Original options (check each; keep valid ones word for word): {_pretty(node.options)}\n"
        # ⚠️ It must see the inventory. Otherwise it meets the lantern "for the first time" every beat, names it
        # differently each time, and one thing becomes several. (Same approach as other narrative
        # games: "Items held (N/MAX)" in the prompt plus "Never re-grant an item already on the
        # sheet".)
        f"Inventory: {', '.join(_held_items(world)) or '(empty)'}\n"
        f"Known characters: {', '.join(_known_characters(world)) or '(none yet)'}\n"
        # ⚠️ It must see the narration. `appeared` means "the narration says so but the picture does not show it";
        # without the narration there is nothing to compare.
        f"Narration of this beat: {node.reply or '(none)'}\n\n"
        "Look at the frame and answer in the schema."
    )
    try:
        # The critic runs during playback, so latency does not matter, but it **must never use the
        # visible replica**, see `config._background_base()`.
        # ⚠️ `model` must be explicit: otherwise it falls back to `config.TEXT_MODEL`, and after switching the agent
        # to another model the critic would silently keep calling the old one.
        out = llm.chat(
            system=_ADJUDICATOR_SYSTEM.replace("{LANGUAGE}", config.LANGUAGE_NAME),
            user=ask,
            model=config.GM_MODEL,
            base=config._background_base(),
            schema=_ADJUDICATOR_SCHEMA,
            temperature=0.7,
            thinking=0,
            tag="adjudicator",
            images=frames,
        )
    except Exception as exc:  # noqa: BLE001 -- a quality pass must never become a failure source
        reqlog.record("adjudicate", ok=False, error=f"{type(exc).__name__}: {exc}")
        return None
    if not isinstance(out, dict):
        return None
    options = [cap(x, 40) for x in (out.get("options") or []) if str(x).strip()]
    if len(options) < 3:
        # Fewer than three: drop them all. The UI is laid out for three, and padding would produce
        # robotic text
        reqlog.record("adjudicate", ok=False, error=f"only {len(options)} options returned")
        return None
    # `appeared.look` and `picked_up.look/scale_anchor` are persisted as English
    # asset definitions and later sent to the image/video models.  The adjudicator
    # schema documents that contract but cannot enforce the language on Qwen, so
    # repair those leaves before they are registered.
    appeared_raw = out.get("appeared") if isinstance(out.get("appeared"), dict) else {}
    picked_raw = out.get("picked_up") if isinstance(out.get("picked_up"), dict) else {}
    guarded = ensure_english_fields(
        {
            "appeared_look": appeared_raw.get("look", ""),
            "picked_up_look": picked_raw.get("look", ""),
            "picked_up_scale": picked_raw.get("scale_anchor", ""),
        },
        fallbacks={
            "appeared_look": "A distinct nonhuman subject with a clear stable silhouette and visible surface details.",
            "picked_up_look": "A single handheld object with clearly visible material, color, and shape.",
            "picked_up_scale": "handheld, realistic human scale",
        },
        model=config.GM_MODEL,
        thinking=0,
        tag="adjudicator_asset_english_guard",
    )
    out = dict(out)
    out["appeared"] = {**appeared_raw, "look": guarded["appeared_look"]}
    out["picked_up"] = {
        **picked_raw,
        "look": guarded["picked_up_look"],
        "scale_anchor": guarded["picked_up_scale"],
    }
    refs_before = {subject.get("ref") for subject in world.subjects.values()}
    appeared = _register_appeared(world, node, out.get("appeared"))
    dropped = _drop_reported_items(world, node, out.get("dropped"))
    picked = _register_picked_up(world, node, out.get("picked_up"))
    node.options = options[:3]
    node.recommend = _recommend_index(out.get("recommend"), len(node.options))
    objective = cap(out.get("objective"), 120)
    if objective:
        node.objective = objective
    # The reflect phase closes this clip's trace: what the critic saw and changed.
    node.trace = [
        *(node.trace or []),
        {
            "id": uuid.uuid4().hex[:8],
            "phase": "reflect",
            "name": "critic",
            "status": "done",
            "summary": _say("看真实画面，修正世界", "Compared the clip with the plan"),
            "data": {
                "observed": str(out.get("observed") or "").strip(),
                "options": node.options,
                "objective": node.objective,
                "picked_up": picked,
                "dropped": dropped,
                "appeared": appeared,
            },
            "started": None,
            "seconds": None,
        },
    ]
    world.nodes[node.id] = _as_dict(node)
    if commit is None:
        world.save()
    elif not commit():
        # These unique files were created only for this discarded snapshot.
        # Never remove pre-existing references, including an item the judge dropped.
        for subject in world.subjects.values():
            ref = subject.get("ref")
            if ref and ref not in refs_before:
                (config.REFS / ref).unlink(missing_ok=True)
        reqlog.record(
            "adjudicate",
            ok=False,
            node=node.id,
            error="the world changed or is generating; discarded a stale verdict",
        )
        return None
    verdict = {
        "observed": str(out.get("observed") or "").strip(),
        "options": node.options,
        "recommend": node.recommend,
        "objective": node.objective,
        "picked_up": picked,
        "dropped": dropped,
        "appeared": appeared,
    }
    reqlog.record("adjudicate", ok=True, node=node.id, **verdict)
    return verdict


def _as_dict(node: Node) -> dict[str, Any]:
    from dataclasses import asdict

    return asdict(node)
