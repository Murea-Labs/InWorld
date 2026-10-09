"""Reference images and voices for a beat: selection, drawing, binding."""

from __future__ import annotations

import contextvars
import re
import time
from concurrent import futures
import uuid
from pathlib import Path
from typing import Any
from collections.abc import Sequence
from murea_agent import config
from murea_agent.util import _say, cap
from murea_agent.providers import imagegen
from murea_agent.memory import voices
from murea_agent.memory.world import World
from murea_agent.director.agent_schema import SECTION_ORDER
from murea_agent.guards.motion import (
    _clean_motion_text,
)
from murea_agent.shot.definitions import (
    _ITEM_KINDS,
    _PendingImage,
    _english_asset_prompt,
    _is_item_record,
)
from murea_agent.shot.video_prompt import (
    _asset_text_match,
    _item_is_close_up,
)
from murea_agent.shot.dialogue import (
    _dialogue_speaker_labels,
    _resolve_dialogue_speaker,
)


# ---------------------------------------------------------------- pieces


# The video provider accepts 1–5 images (video.py); the place always takes the last one.
MAX_CHARACTER_REFS = 4


def _item_needs_reference(
    subject: dict[str, Any],
    *,
    motion_plan: dict[str, Any] | None,
    focus_text: str,
    action_step: dict[str, Any] | None = None,
) -> bool:
    """Decide whether an item gets an independent image in this segment.

    Passive carried props are intentionally omitted. Their appearance is carried by the
    parent latent when available and by `carry_over` in the prompt; an independent catalog
    image is reserved for a structured instrument/target or an explicit object close-up.
    An `@` mention still disambiguates the asset, but by itself does not promote a
    passively carried object into an independently conditioned video subject.
    """
    plan = motion_plan or {}
    step = action_step or {}
    semantic_roles: list[Any] = [plan.get("instrument"), plan.get("target")]
    for key in ("instruments", "targets"):
        values = step.get(key)
        if isinstance(values, list):
            semantic_roles.extend(values)
    if any(_asset_text_match(value, subject) for value in semantic_roles):
        return True
    return _item_is_close_up(subject, focus_text)


def _prune_passive_object_pending(
    world: World,
    pending: list[Any],
    *,
    motion_plan: dict[str, Any] | None,
    focus_text: str,
    action_step: dict[str, Any] | None = None,
) -> None:
    """Avoid generating a standalone image for a non-focal newly declared object."""
    kept: list[Any] = []
    for task in pending:
        if task.kind != "subject":
            kept.append(task)
            continue
        subject = world.subjects.get(str(task.key))
        if not _is_item_record(subject):
            kept.append(task)
            continue
        if _item_needs_reference(
            {**(subject or {}), "id": task.key},
            motion_plan=motion_plan,
            focus_text=focus_text,
            action_step=action_step,
        ):
            kept.append(task)
    pending[:] = kept


def _queue_missing_focal_object_refs(
    world: World,
    subject_ids: list[str],
    pending: list[Any],
    *,
    motion_plan: dict[str, Any] | None,
    focus_text: str,
    action_step: dict[str, Any] | None = None,
) -> None:
    """Generate a missing object ref on demand when it becomes focal later."""
    queued = {str(task.key) for task in pending if task.kind == "subject"}
    for sid in subject_ids:
        subject = world.subjects.get(str(sid))
        if not _is_item_record(subject) or str((subject or {}).get("ref") or "").strip():
            continue
        if sid in queued or not _item_needs_reference(
            {**(subject or {}), "id": sid},
            motion_plan=motion_plan,
            focus_text=focus_text,
            action_step=action_step,
        ):
            continue
        prompt = " ".join(
            str(
                (subject or {}).get("ref_prompt")
                or (subject or {}).get("desc")
                or (subject or {}).get("brief")
                or (subject or {}).get("name")
                or sid
            ).split()
        )
        subject["ref_status"] = "pending"
        subject["ref_source"] = "generated"
        subject["ref_prompt"] = prompt
        pending.append(
            _PendingImage(
                "subject",
                sid,
                prompt,
                world.visual_style,
                str((subject or {}).get("brief") or ""),
                asset_kind="object",
            )
        )
        queued.add(sid)


def _pictures_and_refs(
    world: World,
    place: str,
    subject_ids: list[str],
    required_subject_ids: set[str] | None = None,
    *,
    motion_plan: dict[str, Any] | None = None,
    focus_text: str = "",
    action_step: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    """Reference files for this segment, in `<Picture N>` order.

    Characters and explicitly referenced objects first (director order), destination scene
    last. This order is also the order the files are sent to the video backend; the
    two must stay in sync.

    ⚠️ The provider accepts 1–5 images and the place takes one, so at most 4 subjects share a shot. Extra
    ones are **dropped** instead of failing the turn: a missing reference makes that person look
    less alike, but the clip still exists; a VideoError would show a red bar, a wasted wait and an
    orphan file on disk. Who was dropped goes into `dropped` and the caller reports it. The director
    prompt also says not to crowd shots.
    """
    pictures: list[dict[str, Any]] = []
    refs: list[str] = []
    dropped: list[str] = []
    required = required_subject_ids or set()
    subject_count = 0
    for sid in subject_ids:
        s = world.subjects.get(sid)
        if not s:
            continue
        # A passive carried item is intentionally not uploaded as an independent
        # reference. The parent latent and carry_over contain its current state;
        # sending the isolated catalog image encourages a second, oversized instance.
        if _is_item_record(s) and not _item_needs_reference(
            {**s, "id": sid},
            motion_plan=motion_plan,
            focus_text=focus_text,
            action_step=action_step,
        ):
            continue
        if subject_count >= MAX_CHARACTER_REFS:
            if sid in required:
                raise ValueError(
                    f"More than {MAX_CHARACTER_REFS} characters/items are bound to this shot "
                    f"(cannot fit: {s.get('name', sid)})"
                )
            dropped.append(s.get("name", sid))
            continue
        refs.append(s["ref"])
        subject_count += 1
        asset_kind = "object" if _is_item_record(s) else "character"
        pictures.append(
            {
                "n": len(pictures) + 1,
                "kind": asset_kind,
                "id": sid,
                "name": s.get("name", sid),
                "desc": s.get("desc", ""),
                "ref": s["ref"],
            }
        )
    p = world.places[place]
    refs.append(p["ref"])
    pictures.append(
        {
            "n": len(pictures) + 1,
            "kind": "scene",
            "id": place,
            "name": place,
            "desc": p.get("desc", ""),
            "ref": p["ref"],
        }
    )
    return pictures, refs, dropped


def _subject_id(world: World, value: Any) -> str | None:
    """Resolve either a stable subject id or the subject's display name."""
    token = str(value or "").strip()
    if token in world.subjects:
        return token
    wanted = " ".join(token.split()).casefold()
    if not wanted:
        return None
    for sid, subject in world.subjects.items():
        name = " ".join(str(subject.get("name") or "").split()).casefold()
        if name == wanted:
            return sid
    return None


def _mention_present(text: str, alias: str) -> bool:
    """Return whether ``@alias`` appears as a standalone mention in text.

    Mentions inserted by the UI are followed by a space, but accepting Chinese
    punctuation and adjacent Chinese text keeps manually typed ``@小明去门口``
    useful as well.  Requiring the left boundary to be start or a non-ASCII-word
    character avoids treating email addresses or prose such as ``foo@bar`` as an
    asset binding while allowing natural Chinese ``让@小明走过去``.
    """
    needle = "@" + alias
    if not alias:
        return False
    start = 0
    while True:
        index = text.find(needle, start)
        if index < 0:
            return False
        before = text[index - 1] if index else ""
        after_index = index + len(needle)
        after = text[after_index] if after_index < len(text) else ""
        left_ok = not before or not (before.isascii() and (before.isalnum() or before == "_"))
        # ASCII letters/numbers are the only characters that can continue an
        # English asset name.  Chinese text and punctuation naturally delimit it.
        right_ok = not after or not (after.isascii() and (after.isalnum() or after == "_"))
        if left_ok and right_ok:
            return True
        start = index + 1


def _asset_mentions(
    world: World,
    user_text: str,
    requested: list[str] | None,
) -> list[dict[str, str]]:
    """Resolve explicit asset bindings to stable, world-local IDs.

    The UI sends IDs separately from the natural-language text.  For API clients
    and users who type a known ``@name`` manually, the same resolver also discovers
    matching names in the text.  Unknown explicit IDs fail closed so a binding is
    never silently ignored.
    """
    values = [str(value).strip() for value in (requested or []) if str(value).strip()]
    known: list[tuple[str, str, str]] = []  # id, kind, display name
    for sid, subject in world.subjects.items():
        known.append((sid, "subject", str(subject.get("name") or sid).strip()))
    for name in world.places:
        known.append((name, "place", name))

    resolved: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def add(asset_id: str, kind: str, name: str) -> None:
        key = (kind, asset_id)
        if key in seen:
            return
        seen.add(key)
        resolved.append({"id": asset_id, "kind": kind, "name": name})

    if values:
        for value in values:
            token = " ".join(value.split()).casefold()
            match = next(
                (item for item in known if item[0].casefold() == token or item[2].casefold() == token),
                None,
            )
            if match is None:
                raise ValueError(f"@-mentioned asset does not exist: {value}")
            add(*match)
    else:
        # Longest aliases first prevents a short name from stealing a longer
        # selected name that starts with the same characters.
        aliases: list[tuple[str, str, str, str]] = []
        for asset_id, kind, name in known:
            aliases.append((name, asset_id, kind, name))
            if asset_id != name:
                aliases.append((asset_id, asset_id, kind, name))
        found: list[tuple[int, int, str, str, str]] = []
        for alias, asset_id, kind, name in aliases:
            if not _mention_present(user_text, alias):
                continue
            position = user_text.find("@" + alias)
            if position >= 0:
                # Sort by text position for deterministic subject order; for
                # overlapping aliases, prefer the longer one at that position.
                found.append((position, -len(alias), asset_id, kind, name))
        for _, _, asset_id, kind, name in sorted(found):
            add(asset_id, kind, name)

    places = [item for item in resolved if item["kind"] == "place"]
    if len(places) > 1:
        names = "、".join(item["name"] for item in places)
        raise ValueError(f"A shot can bind at most one place; got: {names}")
    return resolved


def _apply_asset_mentions(
    world: World,
    directive: dict[str, Any],
    mentions: list[dict[str, str]],
) -> list[str]:
    """Make explicit mentions authoritative while preserving GM-selected extras."""
    if not mentions:
        return []
    subject_ids = [item["id"] for item in mentions if item["kind"] == "subject"]
    current_subjects = list(directive.get("subjects") or [])
    # Put required assets first so the reference limit can never evict an
    # explicitly mentioned subject in favour of a GM-added extra.
    directive["subjects"] = [*subject_ids, *current_subjects]
    place = next((item["id"] for item in mentions if item["kind"] == "place"), "")
    if place:
        # A selected scene is the scene reference for this shot.  The normal
        # place-based mode check below will turn it into a cut when necessary.
        directive["place"] = place
        directive["main_scene"] = world.main_scene_for(place)
    directive["asset_mentions"] = [dict(item) for item in mentions]
    return subject_ids


def _user_subject_id(world: World) -> str | None:
    """The established player avatar, if this world already has one."""
    return next(
        (sid for sid, rec in world.subjects.items() if rec.get("control") == "user"),
        None,
    )


def _player_actor_name(world: World) -> str:
    """The current player character's name, so subtitles can use the second person consistently."""
    sid = _user_subject_id(world)
    if not sid:
        return ""
    record = world.subjects.get(sid) or {}
    return _clean_motion_text(record.get("name") or sid)


def _unique_player_name(world: World, requested: Any) -> str:
    """Keep an implicit player distinct from every world-driven asset."""
    base = cap(requested, 40) or _say("玩家角色", "Player")
    if _subject_id(world, base) is None:
        return base
    candidate = _say(f"{base}（玩家）", f"{base} (player)")
    suffix = 2
    while _subject_id(world, candidate) is not None:
        candidate = _say(f"{base}（玩家{suffix}）", f"{base} (player {suffix})")
        suffix += 1
    return candidate


def _inject_implicit_player(world: World, directive: dict[str, Any]) -> None:
    """Turn the first visible, previously implicit player into a real asset.

    A world is allowed to start without a user-controlled character. The GM may
    still turn an embodied command such as "退后两步" into a visible traveler. If
    that person is left as prose, the video model invents a face that cannot be
    remembered. `implicit_player` makes the GM's visibility decision structured;
    this function owns the deterministic asset registration request.
    """
    if _user_subject_id(world) is not None:
        return
    raw = directive.get("implicit_player")
    if not isinstance(raw, dict) or raw.get("appears") is not True:
        return

    name = _unique_player_name(world, raw.get("name"))
    candidate = {
        "name": name,
        "kind": "人物",
        "control": "user",
        "voice": str(raw.get("voice") or "").strip(),
        "image_prompt": str(raw.get("image_prompt") or "").strip(),
        "brief": str(raw.get("brief") or "").strip(),
    }
    declared = [x for x in (directive.get("new_subjects") or []) if isinstance(x, dict)]
    duplicate = next(
        (x for x in declared if str(x.get("name") or "").strip() == name),
        None,
    )
    if duplicate is None:
        declared.append(candidate)
    else:
        duplicate.update({k: v for k, v in candidate.items() if v or k == "control"})
    directive["new_subjects"] = declared

    visible = list(directive.get("subjects") or [])
    if name not in visible:
        visible.insert(0, name)
    directive["subjects"] = visible
    raw["name"] = name


def _prepare_subjects(
    world: World,
    directive: dict[str, Any],
    pending: list[_PendingImage],
    visible_names: set[str] | None = None,
) -> list[str]:
    """Register new on-screen people and return the ids in screen order.

    Registration only; the images themselves are queued in `pending` so they can be
    drawn while the rewriter runs. Name lookup is deliberately repeated before
    creating anything so a model that reports an existing person as new cannot
    create a duplicate asset.
    """
    # A location already has its own place reference. The GM occasionally repeats
    # that same room/street/building in new_subjects as kind=物件; accepting it both
    # wastes a character-reference slot and sends two competing images of one
    # space. Place identity wins deterministically, including for legacy duplicates.
    place_names = {" ".join(str(name).split()).casefold() for name in world.places}
    declared: list[dict[str, Any]] = []
    for candidate in directive.get("new_subjects") or []:
        if not isinstance(candidate, dict):
            continue
        name_key = " ".join(str(candidate.get("name") or "").split()).casefold()
        if name_key and name_key in place_names:
            continue
        declared.append(candidate)
    # Store the directive that was actually executable, not the rejected duplicate.
    directive["new_subjects"] = declared

    created: list[str] = []
    has_player = _user_subject_id(world) is not None
    for candidate in declared:
        name = str(candidate.get("name") or "").strip()
        if not name:
            continue
        sid = _subject_id(world, name)
        if sid is None:
            image_prompt = str(candidate.get("image_prompt") or "").strip() or (
                f"{name}, a character from {world.setting or 'the current story world'}"
            )
            image_prompt = _english_asset_prompt(
                image_prompt,
                fallback="A single complete full-body character with clear stable appearance and neutral lighting.",
            )
            brief = cap(candidate.get("brief"), 60)
            kind = str(candidate.get("kind") or "人物").strip() or "人物"
            # Only people and creatures need a voice; objects do not talk. The director picks a
            # library id (the library is listed in its prompt); this is only a lookup. If it gave
            # none, or an id not in the library, pick one by name and description. Not picking is
            # worse than picking wrong: the video model generates audio with the video, and without
            # a reference voice it invents one every clip. A wrong pick is at least stable, visible
            # in the inspector and replaceable in one click.
            picked = None
            if kind not in _ITEM_KINDS:
                picked = voices.resolve(candidate.get("voice") or "") or voices.suggest(
                    name, brief or image_prompt
                )
            control = "user" if candidate.get("control") == "user" and not has_player else "ai"
            sid = world.add_subject(
                name,
                "",
                image_prompt,
                brief=brief,
                kind=kind,
                control=control,
                voice=voices.record(picked) if picked else None,
            )
            has_player = has_player or control == "user"
            pending.append(
                _PendingImage(
                    "subject",
                    sid,
                    image_prompt,
                    world.visual_style,
                    brief,
                    asset_kind="object" if kind in _ITEM_KINDS else "subject",
                )
            )
        if sid not in created:
            created.append(sid)

    ordered: list[str] = []
    for value in directive.get("subjects") or []:
        sid = _subject_id(world, value)
        rec = world.subjects.get(sid or "") or {}
        subject_name = " ".join(str(rec.get("name") or "").split()).casefold()
        if sid is not None and subject_name not in place_names and sid not in ordered:
            ordered.append(sid)
    # Be defensive if the GM defined a new visible person but forgot to repeat the
    # name in subjects: the new asset must still condition the segment that made it.
    for sid in created:
        if visible_names is not None:
            rec = world.subjects.get(sid) or {}
            if str(rec.get("name") or "").strip() not in visible_names:
                continue
        if sid not in ordered:
            ordered.append(sid)
    return ordered


def _timed_draw(prompt: str, kind: str, visual_style: str) -> tuple[bytes, float]:
    t = time.time()
    raw = imagegen.generate(prompt, kind=kind, visual_style=visual_style)
    return raw, round(time.time() - t, 2)


def _lock_audio_refs(
    six: dict[str, str], bind: list[dict[str, Any]], pictures: list[dict[str, Any]]
) -> dict[str, str]:
    """Bind <Audio N> to a pictured subject or an established character, appended to the definitions.

    This text goes straight into the video model's English subject definitions, so it must be
    English; it only binds the voice and must not let the model repeat words from the reference
    audio.
    """
    idx = {p["id"]: p["n"] for p in pictures if p["kind"] == "character"}
    # Record which <Subject N> it is bound to; the inspector shows it. With more than 4 people in
    # the shot someone's reference image is dropped, so they have no Subject number and the voice
    # cannot bind (0).
    for b in bind:
        b["subject_n"] = idx.get(b["sid"], 0)
    lines = [
        f"<Audio {b['n']}> identifies <Subject {idx[b['sid']]}> "
        f"(S{idx[b['sid']]})'s voice. Keep its timbre, but follow the shot's "
        "vocal performance; never repeat the reference words."
        + (f" Voice: {b['voice_desc'].rstrip('.')}." if b.get("voice_desc") else "")
        for b in bind
        if b["sid"] in idx
    ]
    if not lines:
        return six
    out = dict(six)
    body = out.get("subject_definitions", "")
    out["subject_definitions"] = "\n".join(x for x in (body, *lines) if x.strip())
    return out


# ── Early drawing ─────────────────────────────────────────────────────── Before the merge,
# reference images were drawn in parallel with the rewriter call. With one merged call, drawing
# became sequential after the director: in a sample of 60 clips 17% needed new images, with a median
# of 3.23 s of drawing.
#
# But what to draw is decided the moment the director's streamed output **finishes the new_subjects
# field**, before the whole JSON arrives. The images are queued at that moment; when the flow
# reaches `_draw_pending`, the same prompt reuses the future already running (or finished). The
# saving is the overlap between drawing and the rest of the director's output.
#
# ⚠️ Only warm names **not yet in the world**. Image generation costs money, and warming a name that is
# later deduplicated wastes an image. Name matching uses the same rules as `_prepare_subjects`
# (strip whitespace + casefold), or the two would disagree.
_WARM_MAX = 8

_warm: dict[tuple[str, str, str], futures.Future[tuple[bytes, float]]] = {}

_warm_pool = futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="warm-draw")


def _warm_key(kind: str, prompt: str, style: str) -> tuple[str, str, str]:
    return (kind, " ".join(prompt.split()), style)


def _warm_images(world: World, part: dict[str, Any]) -> None:
    """While the director is still writing, queue the reference images this beat will need."""
    style = world.visual_style
    known = {" ".join(str(n).split()).casefold() for n in world.places}
    known |= {" ".join(str((v or {}).get("name") or "").split()).casefold() for v in world.subjects.values()}
    todo: list[tuple[str, str]] = []
    for item in part.get("new_subjects") or []:
        if not isinstance(item, dict):
            continue
        name = " ".join(str(item.get("name") or "").split())
        prompt = str(item.get("image_prompt") or "").strip()
        if name and prompt and name.casefold() not in known:
            asset_kind = "object" if str(item.get("kind") or "").strip() in _ITEM_KINDS else "subject"
            todo.append((asset_kind, prompt))
    place = " ".join(str(part.get("place") or "").split())
    prompt = str(part.get("place_image_prompt") or "").strip()
    if place and prompt and place.casefold() not in known:
        todo.append(("place", prompt))
    for kind, prompt in todo:
        key = _warm_key(kind, prompt, style)
        if key in _warm or len(_warm) >= _WARM_MAX:
            continue
        # ⚠️ The thread pool does not carry contextvars. Without copy these images' calls are logged to no world
        # (reqlog drops entries without a world), exactly the slowest part when debugging.
        ctx = contextvars.copy_context()
        _warm[key] = _warm_pool.submit(ctx.run, _timed_draw, prompt, kind, style)


def _drop_warm() -> None:
    """Clear at the end of a beat. Unclaimed ones would only let the next beat pick up this beat's images.

    Cancel those not started yet: image generation costs money, and an early undo saves real money.
    Running ones cannot be cancelled; they were already drawing before the undo anyway.
    """
    for future in _warm.values():
        future.cancel()
    _warm.clear()


def _draw_pending(
    pending: list[_PendingImage], pool: futures.ThreadPoolExecutor
) -> dict[str, futures.Future[tuple[bytes, float]]]:
    out: dict[str, futures.Future[tuple[bytes, float]]] = {}
    for p in pending:
        asset_kind = p.asset_kind or ("place" if p.kind == "place" else "subject")
        key = _warm_key(asset_kind, p.prompt, p.visual_style)
        warm = _warm.pop(key, None)
        # Claim the warmed future directly; it may be finished or still running, and `.result()` in
        # `_collect_pending` handles both.
        out[f"{p.kind}:{p.key}"] = warm or pool.submit(_timed_draw, p.prompt, asset_kind, p.visual_style)
    return out


def _audio_refs(
    world: World,
    subject_ids: list[str],
    dialogue: str,
    also_speaking: Sequence[str] = (),
) -> tuple[list[Path], list[dict[str, Any]]]:
    """The reference voices to send to the video model for this clip, and who they are bound to.

    Only voices of characters who actually speak. A silent character's reference audio would
    condition the model on its rhythm and mood too, wasting a slot and disturbing the current
    speaker's performance. The returned order is the order of <Audio 1..N> in the prompt (the
    provider numbers images and audios separately).

    ⚠️ No dialogue in this clip means no voices at all. `_ensure_soundscape` already writes "No speech,
    no voice-over, no narration -- nobody talks in this shot.", but that is only **text**; a voice
    reference is a **real audio file** labelled "voice and speaking style of S1", and for the model
    it outweighs the text, like saying "nobody talks here" while handing over "this is how this
    person talks". In one real world 22 of 24 silent clips carried voices and everyone moved their
    lips. The fix is to stop sending the contradictory signal.
    """
    if not str(dialogue or "").strip() and not also_speaking:
        return [], []
    speaking: list[str] = []
    for label in _dialogue_speaker_labels(dialogue):
        sid = _resolve_dialogue_speaker(world, subject_ids, label)
        if sid and sid not in speaking:
            speaking.append(sid)
    # `dialogue` lists only the speaker of the line the director picked, while `<d>` blocks in the
    # six sections often have several speakers. Anyone missed has no reference voice this beat, and
    # the model invents one or reuses the only voice present (an elder once spoke with a hot-blooded
    # teenager's voice).
    for sid in also_speaking:
        if sid and sid in subject_ids and sid not in speaking:
            speaking.append(sid)
    # Older GM output occasionally omitted the named-dialogue delimiter but still
    # included the display name. Keep that narrow compatibility path without
    # uploading every silent character's speaking style.
    if not speaking:
        speaking = [
            sid
            for sid in subject_ids
            if not _is_item_record(world.subjects.get(sid))
            and str((world.subjects.get(sid) or {}).get("name") or "") in dialogue
        ]
    paths: list[Path] = []
    bind: list[dict[str, Any]] = []
    for sid in speaking:
        rec = world.subjects.get(sid) or {}
        if _is_item_record(rec):
            continue
        # Old worlds store ids of a deleted voice library. Without repair the person has no
        # reference voice and the model invents one every clip; the same character sounds different,
        # silently.
        v = voices.heal(rec.get("voice")) or {}
        p = voices.path(v.get("ref") or "")
        if p is None:
            continue
        paths.append(p)
        bind.append(
            {
                "n": len(paths),
                "sid": sid,
                "name": rec.get("name", sid),
                "file": v.get("ref") or "",
                "voice_id": v.get("id") or "",
                "voice_name": voices.label(v),
                # The reference audio carries timbre, but its reading tone is learned too.
                # Describe the voice in one English sentence (age, gender, texture, way of
                # speaking) as a second anchor for the video model.
                "voice_desc": str(rec.get("voice_desc") or "").strip(),
            }
        )
        if len(paths) >= 3:
            break
    return paths, bind


def _six_section_speakers(six: dict[str, str]) -> list[str]:
    """The `(Sn)` speakers actually marked in the six sections: the people who speak in this beat.

    In order of appearance, deduplicated; only sections with `<d>` count, so numbers merely
    mentioned in prose are not taken as speakers.
    """
    found: list[str] = []
    for key in SECTION_ORDER:
        text = six.get(key) or ""
        if "<d>" not in text:
            continue
        for match in re.finditer(r"\((S\d+)\)\s*<d>", text):
            sid = match.group(1)
            if sid not in found:
                found.append(sid)
    return found


def _collect_pending(
    world: World,
    pending: list[_PendingImage],
    jobs: dict[str, futures.Future[tuple[bytes, float]]],
) -> float:
    """Write the drawn images to disk and point the assets at them.

    Returns the wall time the slowest draw actually took -- these run alongside the
    rewriter, so it must NOT be added to the turn's stage timings or the breakdown
    would count the same seconds twice.
    """
    slowest = 0.0
    for p in pending:
        try:
            raw, took = jobs[f"{p.kind}:{p.key}"].result()
            slowest = max(slowest, took)
        except imagegen.ImageRefused as exc:
            what = "place" if p.kind == "place" else ("object" if p.asset_kind == "object" else "character")
            raise RuntimeError(f"Reference image for {p.key!r} ({what}) failed: {exc}") from exc
        prefix = "place" if p.kind == "place" else "subj"
        fn = f"{prefix}_{uuid.uuid4().hex[:8]}{imagegen.sniff(raw)[0]}"
        (config.REFS / fn).write_bytes(raw)
        store = world.places if p.kind == "place" else world.subjects
        store[p.key]["ref"] = fn
        if p.asset_kind == "object":
            store[p.key]["ref_status"] = "ready"
            store[p.key]["ref_source"] = "generated"
            store[p.key]["ref_prompt"] = p.prompt
    return slowest
