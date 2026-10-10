"""Compiling the six-section video prompt: definitions, locks, budget."""

from __future__ import annotations

import re
from concurrent import futures
from typing import Any
from murea_agent import config
from murea_agent.providers import llm
from murea_agent import reqlog
from murea_agent.director.prompt_language import ensure_english_fields, ensure_english_text
from murea_agent.memory.world import Node, World
from murea_agent.util import load_prompt, _STATE_PREFIX, _STYLE_CLAUSE, _pretty
from murea_agent.director.agent_schema import (
    REWRITER_SCHEMA,
    REWRITER_SCHEMA_NO_DEFS,
    SECTION_ORDER,
    _MODEL_SECTIONS,
)
from murea_agent.guards.motion import (
    _EVIDENCE_TEXT,
    _RELATION_PHYSICS,
    _clean_motion_text,
    _distance_progress_clause,
    _motion_active,
    _motion_audit_error,
    _motion_plan_errors,
    _normalize_motion_plan,
    _strip_motion_audit,
)
from murea_agent.shot.definitions import (
    _CJK_TEXT,
    _DEF_LEAK,
    _definition_body,
    _is_item_record,
    _object_appearance,
    _subject_name_token,
)
from murea_agent.shot.soundscape import (
    SOUNDSCAPE_LOCKED,
)
from murea_agent.shot.prompt_budget import (
    _D_BLOCK,
)


REWRITER_SYSTEM = load_prompt("rewriter.md")


# framing is the model's judgement for this beat (wide / medium-wide / medium), visibility the
# vocabulary retention_analysis requires. They map one to one, so the server translates.
_FRAMING_VISIBILITY = {
    "wide": "full body",
    "medium-wide": "most of the body",
    "medium": "waist-up",
    "none": "most of the body",
}


def _build_retention(
    pictures: list[dict[str, Any]], world: World, motion_plan: dict[str, Any], carry: str = ""
) -> str:
    """Describe visibility and reference changes, leaving stable appearance in subject definitions.

    Foreground identity is retained while pose, framing and background belong to this shot.
    Objects also retain their physical scale, with occlusion when held.
    """
    scale = _FRAMING_VISIBILITY.get(str(motion_plan.get("framing") or "none"), "most of the body")
    scene = next((f"<Subject {p['n']}>" for p in pictures if p.get("kind") == "scene"), "")
    environment = scene or "the current story scene"
    lines: list[str] = []
    for pic in pictures:
        n = pic.get("n")
        if pic.get("kind") == "scene":
            lines.append(
                f"<Subject {n}> (appears in [Shot 1] (wide scene view)): "
                "partially_preserved - layout, materials and lighting retained; framing follows the shot."
            )
            continue
        rec = world.subjects.get(str(pic.get("id") or "")) or {}
        if pic.get("kind") == "object" or _is_item_record(rec):
            held = _object_is_held({**rec, "id": pic.get("id")}, motion_plan, carry)
            vis = "occlusion" if held else "whole object"
            object_scale = _object_scale_anchor(rec, held=held)
            what = f"shape, material and color retained at {object_scale} scale"
        else:
            vis = scale
            what = "identity and visible appearance retained"
        lines.append(
            f"<Subject {n}> (appears in [Shot 1] ({vis})): "
            f"partially_preserved - {what}; pose and framing follow the action; "
            f"{environment} provides the background and lighting."
        )
    return "\n".join(lines)


def _lock_visual_style(six: dict[str, str], visual_style: str) -> dict[str, str]:
    """Deterministically put the world's immutable Style Bible into the video model's prompt."""
    style = " ".join(_english_visual_style(visual_style).split()).rstrip(" .")
    if not style:
        return six
    out = dict(six)
    detail = (out.get("detailed_description") or "").strip()
    clause = f"Visual style: {style}."
    if clause.casefold() in detail.casefold():
        return out
    shot = "[Shot 1]"
    if detail.startswith(shot):
        detail = detail[len(shot) :].lstrip()
    out["detailed_description"] = f"{shot} {clause} {detail}".rstrip()
    return out


# An item deserves a reference slot only when this beat really interacts with it or explicitly asks
# for a close-up. Ordinary holding is kept by the previous latent + carry_over; putting a standalone
# item image in every beat makes the video model draw the same thing both as a foreground subject
# and as the prop in hand.
_CLOSE_UP_HINT = re.compile(
    r"特写|近景|近距离|细节(?:镜头)?|聚焦|放大|微距|"
    r"\b(?:extreme\s+)?close[- ]?up\b|\bmacro\b|\b(?:insert|detail|product)\s+shot\b",
    re.I,
)


def _asset_text_match(value: Any, subject: dict[str, Any]) -> bool:
    """Match an asset name/id inside GM's free-form motion text."""
    raw = re.sub(r"\s+", "", str(value or "")).casefold()
    if not raw:
        return False
    for candidate in (subject.get("name"), subject.get("id")):
        token = re.sub(r"\s+", "", str(candidate or "")).casefold()
        if token and len(token) >= 2 and token in raw:
            return True
    return False


def _item_is_close_up(subject: dict[str, Any], focus_text: str) -> bool:
    """Whether the request explicitly makes this item the close-up subject."""
    text = str(focus_text or "")
    for match in _CLOSE_UP_HINT.finditer(text):
        # A character close-up can mention a carried prop in the same sentence. Stop
        # at clause punctuation so "close-up of 小明，他手里拿着铜灯笼" does not
        # promote the lantern as a focal object.
        before = text[max(0, match.start() - 48) : match.start()]
        after = text[match.end() : match.end() + 48]
        before = re.split(r"[,，。.!！?？;；:：]", before)[-1]
        after = re.split(r"[,，。.!！?？;；:：]", after)[0]
        if _asset_text_match(before, subject) or _asset_text_match(after, subject):
            return True
    return False


def _picture_store(world: World, picture: dict[str, Any]) -> dict[str, Any]:
    """Resolve a picture kind without conflating object references with scenes."""
    return world.places if picture.get("kind") == "scene" else world.subjects


_PROMPT_ENGLISH_FALLBACKS = {
    "subject_definitions": "The established subject shown in the attached reference.",
    "summary": "A continuous cinematic shot shows the established subjects in the current story environment.",
    "retention_analysis": "The attached references retain stable subject identity and visible appearance.",
    "detailed_description": "The established subjects perform the requested actions in one continuous shot in the current story environment.",
    "overall_soundscape": "Diegetic ambience of the current story environment.",
    "non_diegetic_music": "N/A",
}


def _ensure_prompt_english(fields: dict[str, Any]) -> dict[str, str]:
    """Guard every video caption section immediately before it is assembled.

    Explicit ``<d>`` dialogue blocks are intentionally allowed to contain the
    world's Chinese spoken lines; all other prose must be English.
    """
    return ensure_english_fields(
        {key: fields.get(key, "") for key in SECTION_ORDER},
        allow_dialogue=True,
        fallbacks=_PROMPT_ENGLISH_FALLBACKS,
        model=config.REWRITER_MODEL,
        thinking=config.REWRITER_THINKING,
        tag="video_prompt_english_guard",
    )


def _english_visual_style(value: str) -> str:
    """Translate a Chinese Style Bible once before it is copied into media prompts."""
    return ensure_english_text(
        value,
        fallback="cinematic, coherent character and environment design, detailed lighting",
        model=config.REWRITER_MODEL,
        thinking=config.REWRITER_THINKING,
        tag="visual_style_english_guard",
        translate=False,
    )


_MOTION_TRANSLATE_SYSTEM = """Translate physical-plan prose into concise natural English.
Return only the requested JSON object. Preserve every __BOUND_SUBJECT_N__ placeholder
exactly; these placeholders identify reference-bound Subject tokens and must never
be translated, removed, or renamed. Do not output Chinese or any other non-English
prose. Keep the physical meaning, start/end state, contact geometry, camera side, and
visible evidence precise. Do not add a second action or a new relation change."""

_MOTION_TEXT_KEYS = (
    "actors",
    "instrument",
    "target",
    "start_state",
    "path",
    "contact_points",
    "end_state",
    "camera_side",
    "must_show",
    "invariants",
)


def _mask_bound_subject_names(
    value: Any,
    pictures: list[dict[str, Any]] | None,
) -> tuple[str, dict[str, str]]:
    """Protect asset names with untranslatable placeholders, then restore them as Subject tokens after translation."""
    text = _clean_motion_text(value)
    masks: dict[str, str] = {}
    if not pictures:
        return text, masks
    aliases = sorted(
        [
            (
                _clean_motion_text(pic.get("name")),
                f"__BOUND_SUBJECT_{pic.get('n')}__",
            )
            for pic in pictures
            if isinstance(pic.get("n"), int) and _clean_motion_text(pic.get("name"))
        ],
        key=lambda item: len(item[0]),
        reverse=True,
    )
    for alias, marker in aliases:
        if re.search(r"[A-Za-z0-9_]", alias):
            pattern = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(alias)}(?![A-Za-z0-9_])", re.I)
        else:
            pattern = re.compile(re.escape(alias))
        if pattern.search(text):
            text = pattern.sub(marker, text)
            number = marker.removeprefix("__BOUND_SUBJECT_").removesuffix("__")
            masks[marker] = f"<Subject {number}>"
    return text, masks


def _restore_subject_masks(value: Any, masks: dict[str, str]) -> str:
    text = _clean_motion_text(value)
    for marker, token in masks.items():
        text = text.replace(marker, token)
    return text


def _translate_motion_plan(plan: dict[str, Any], pictures: list[dict[str, Any]] | None) -> dict[str, Any]:
    """Translate Chinese physics fields the director wrote by mistake into English, keeping bound asset tokens.

    motion_plan is structured input for compiling physical constraints; one beat must not fail with
    ValueError just because the model occasionally ignored "English". If translation fails, use a
    Chinese-free physical fallback so the beat continues and the problem stays visible in the
    inspectable prompt.
    """
    if not any(
        _CJK_TEXT.search(_clean_motion_text(plan.get(key)))
        or any(_CJK_TEXT.search(_clean_motion_text(x)) for x in (plan.get(key) or []))
        for key in _MOTION_TEXT_KEYS
    ):
        return {
            **plan,
            "actors": [_subject_name_token(x, pictures) for x in plan.get("actors", [])],
            "instrument": _subject_name_token(plan.get("instrument"), pictures),
            "target": _subject_name_token(plan.get("target"), pictures),
            "start_state": _subject_name_token(plan.get("start_state"), pictures),
            "path": _subject_name_token(plan.get("path"), pictures),
            "contact_points": _subject_name_token(plan.get("contact_points"), pictures),
            "end_state": _subject_name_token(plan.get("end_state"), pictures),
            "camera_side": _subject_name_token(plan.get("camera_side"), pictures),
            "must_show": [_subject_name_token(x, pictures) for x in plan.get("must_show", [])],
            "invariants": [_subject_name_token(x, pictures) for x in plan.get("invariants", [])],
        }

    masked_plan = dict(plan)
    masks: dict[str, str] = {}
    for key in _MOTION_TEXT_KEYS:
        value = plan.get(key)
        if isinstance(value, list):
            masked: list[str] = []
            for item in value:
                text, local = _mask_bound_subject_names(item, pictures)
                masked.append(text)
                masks.update(local)
            masked_plan[key] = masked
        else:
            text, local = _mask_bound_subject_names(value, pictures)
            masked_plan[key] = text
            masks.update(local)

    schema = {
        "type": "object",
        "properties": {
            key: (
                {"type": "array", "items": {"type": "string"}}
                if isinstance(masked_plan.get(key), list)
                else {"type": "string"}
            )
            for key in _MOTION_TEXT_KEYS
        },
        "required": list(_MOTION_TEXT_KEYS),
    }
    try:
        translated = llm.chat(
            system=_MOTION_TRANSLATE_SYSTEM,
            user=_pretty({"fields": {key: masked_plan.get(key) for key in _MOTION_TEXT_KEYS}}),
            schema=schema,
            temperature=0.1,
            model=config.REWRITER_MODEL,
            thinking=config.REWRITER_THINKING,
            tag="motion_plan_english",
        )
        out = dict(plan)
        for key in _MOTION_TEXT_KEYS:
            value = translated.get(key)
            if isinstance(plan.get(key), list):
                values = value if isinstance(value, list) else []
                out[key] = [_restore_subject_masks(x, masks) for x in values]
            else:
                out[key] = _restore_subject_masks(value, masks)
        # Even if the translation model drops a token, bound assets must never be written back as
        # bare Chinese names.
        out["actors"] = [_subject_name_token(x, pictures) for x in out.get("actors", [])]
        for key in (
            "instrument",
            "target",
            "start_state",
            "path",
            "contact_points",
            "end_state",
            "camera_side",
        ):
            out[key] = _subject_name_token(out.get(key), pictures)
        out["must_show"] = [_subject_name_token(x, pictures) for x in out.get("must_show", [])]
        out["invariants"] = [_subject_name_token(x, pictures) for x in out.get("invariants", [])]
        if all(
            not _CJK_TEXT.search(_clean_motion_text(out.get(key)))
            and not any(_CJK_TEXT.search(_clean_motion_text(x)) for x in (out.get(key) or []))
            for key in _MOTION_TEXT_KEYS
        ):
            return out
        raise ValueError("Translated motion plan still contains Chinese")
    except Exception as exc:  # noqa: BLE001 -- a failed translation must not block this beat
        reqlog.record("motion_translate", ok=False, error=f"{type(exc).__name__}: {exc}")

    # The fallback keeps only English geometric facts; bound assets still point to their references
    # by token.
    fallback = dict(plan)
    actor_tokens = [_subject_name_token(x, pictures) for x in plan.get("actors", [])]
    actor_tokens = [x for x in actor_tokens if not _CJK_TEXT.search(x)]
    actor = actor_tokens[0] if actor_tokens else "the actor"
    target = _subject_name_token(plan.get("target"), pictures) or "the target"
    if _CJK_TEXT.search(target):
        target = "the target"
    fallback.update(
        {
            "actors": actor_tokens or ["the actor"],
            "instrument": "the instrument" if plan.get("instrument") else "",
            "target": target,
            "start_state": f"{actor} begins in the stated starting position relative to {target}",
            "path": f"{actor} follows only the stated path toward {target}",
            "contact_points": f"{actor} uses the named contact points on {target}",
            "end_state": f"{actor} reaches the stated end relation to {target}",
            "camera_side": "the camera remains on the same side of the boundary as the actor",
            "must_show": ["the actor's required body parts", f"{target} and its relevant structure"],
            "invariants": ["all named rigid shapes, identities, and supports remain unchanged"],
        }
    )
    return fallback


def _lock_bound_subject_tokens(
    fields: dict[str, str],
    pictures: list[dict[str, Any]] | None = None,
) -> dict[str, str]:
    """Replace bare names of bound assets in the six sections with ``<Subject N>``.

    Intermediate director / rewriter output may use Chinese display names so machine translation
    never changes names; but the text the video model receives must point to the uploaded references
    through Subject tokens. Dialogue blocks stay verbatim: the Chinese there is what people actually
    say.
    """
    if not pictures:
        return fields
    out = dict(fields)
    for key, value in out.items():
        text = str(value or "")
        if not text:
            continue
        cursor = 0
        chunks: list[str] = []
        for match in _D_BLOCK.finditer(text):
            chunks.append(_subject_name_token(text[cursor : match.start()], pictures))
            chunks.append(match.group(0))
            cursor = match.end()
        chunks.append(_subject_name_token(text[cursor:], pictures))
        out[key] = "".join(chunks)
    return out


def _motion_plan_clause(
    plan: dict[str, Any],
    pictures: list[dict[str, Any]] | None = None,
) -> str:
    if not _motion_active(plan):
        return ""
    plan = _translate_motion_plan(plan, pictures)

    def clean(value: Any) -> str:
        return _subject_name_token(
            _clean_motion_text(value).rstrip(" .;:"),
            pictures,
        )

    invariants = "; ".join(clean(x) for x in plan["invariants"])
    must_show = "; ".join(
        [
            *(_EVIDENCE_TEXT[x] for x in plan["visual_evidence"] if x in _EVIDENCE_TEXT),
            *(clean(x) for x in plan["must_show"]),
        ]
    )
    relation_physics = _RELATION_PHYSICS.get(plan["relation_change"], "")
    if plan["interaction_type"] == "impact":
        instrument = clean(plan["instrument"])
        relation_physics = (
            f"The moving instrument {instrument} remains one visible rigid object along "
            "a continuous path from its stated start position to the named contact point. "
            "It does not teleport, disappear before contact, or become part of the actor's "
            "body. The target reacts only after visible contact."
        )
    if plan["relation_change"] == "distance" and plan["phase"] == "disengage":
        relation_physics = (
            "A visible gap of air or ground opens smoothly between the actor and target; "
            "the gap widens continuously to the stated end distance."
        )
    contact = f" Contact geometry: {clean(plan['contact_points'])}." if plan["contact_points"] else ""
    progress = _distance_progress_clause(plan)
    instrument_role = f" Active instrument: {clean(plan['instrument'])}." if plan.get("instrument") else ""
    return (
        f"Physical continuity — Start state: {clean(plan['start_state'])}. "
        f"{instrument_role} "
        f"The only relation change involving {clean(plan['target'])} is "
        f"{plan['relation_change']}: {clean(plan['path'])}."
        f"{contact} End state: {clean(plan['end_state'])}. "
        f"Camera geometry: {clean(plan['camera_side'])}. The camera stays on that same "
        f"side of every rigid boundary in a {plan['framing']} composition. "
        f"The framing continuously keeps {must_show} visible from the first frame "
        f"through the final frame. {relation_physics} "
        f"{progress} "
        f"Throughout the shot, {invariants}."
    )


def _lock_motion_plan(
    fields: dict[str, str],
    raw_plan: Any,
    *,
    pictures: list[dict[str, Any]] | None = None,
) -> dict[str, str]:
    """Apply the physical constraints deterministically; even if the rewriter drops a sentence, no unconstrained prompt reaches the video model."""
    plan = _normalize_motion_plan(raw_plan)
    clause = _motion_plan_clause(plan, pictures)
    if not clause:
        return fields
    out = _lock_bound_subject_tokens(fields, pictures)
    detail = (out.get("detailed_description") or "").strip()
    if clause.casefold() in detail.casefold():
        return out
    out["detailed_description"] = f"{detail} {clause}".strip() if detail else f"[Shot 1] {clause}"
    return out


def _lock_required_ref_assets(
    fields: dict[str, str],
    pictures: list[dict[str, Any]],
    required_ids: set[str],
) -> dict[str, str]:
    """Ensure @-bound references are named in the video model's primary shot description."""
    if not required_ids:
        return fields
    missing = [
        pic
        for pic in pictures
        if pic.get("id") in required_ids
        and f"<Subject {pic.get('n')}>".casefold()
        not in (fields.get("detailed_description") or "").casefold()
    ]
    if not missing:
        return fields
    out = dict(fields)
    detail = (out.get("detailed_description") or "").strip()
    clause = (
        "The shot visibly includes "
        + ", ".join(f"<Subject {pic['n']}> as shown in <Picture {pic['n']}>" for pic in missing)
        + "."
    )
    if detail:
        out["detailed_description"] = f"{detail} {clause}".strip()
    else:
        out["detailed_description"] = f"[Shot 1] {clause}"
    return out


def _object_scale_anchor(subject: dict[str, Any], *, held: bool = False) -> str:
    """Return a short, stable scale cue for an object reference."""
    for key in ("scale_anchor", "size_anchor", "scale"):
        value = " ".join(str(subject.get(key) or "").split())
        if value:
            return value[:100].rstrip(" .")
    return "realistic handheld scale" if held else "realistic physical scale"


def _object_is_held(
    subject: dict[str, Any],
    motion_plan: dict[str, Any] | None,
    carry: str,
) -> bool:
    """Whether the current shot already establishes the object as a held prop."""
    plan = motion_plan or {}
    target = _asset_text_match(plan.get("target"), subject)
    relation = str(plan.get("relation_change") or "none").strip().lower()
    if target and relation == "possession":
        return True
    if not _asset_text_match(carry, subject):
        return False
    return bool(re.search(r"手里|拿着|携带|握着|holding|held|carried|carry", str(carry), re.I))


def _object_actor_token(
    motion_plan: dict[str, Any] | None,
    pictures: list[dict[str, Any]],
) -> str:
    """Resolve the first actor in a possession plan to its Subject token."""
    plan = motion_plan or {}
    actors = plan.get("actors") if isinstance(plan.get("actors"), list) else []
    for actor in actors:
        for pic in pictures:
            if pic.get("kind") != "character":
                continue
            if _asset_text_match(actor, {"name": pic.get("name"), "id": pic.get("id")}):
                return f"<Subject {pic['n']}>"
    return "the visible character"


def _object_definition_line(
    body: str,
    picture_n: int,
    subject: dict[str, Any],
    *,
    held: bool,
) -> str:
    """Normalize one object definition to an explicitly inanimate, scaled form."""

    appearance = _object_appearance(body)
    scale = _object_scale_anchor(subject, held=held)
    return (
        f"the single inanimate object prop shown in <Picture {picture_n}> as a single "
        f"isolated object reference; {appearance}; scale anchor: {scale}"
    )


def _lock_object_definitions(
    fields: dict[str, str],
    pictures: list[dict[str, Any]],
    world: World,
    motion_plan: dict[str, Any] | None,
    carry: str,
) -> dict[str, str]:
    """Make every sent object definition unambiguously a prop, never a character."""
    text = str(fields.get("subject_definitions") or "")
    if not text:
        return fields
    out = dict(fields)
    for pic in pictures:
        if pic.get("kind") != "object":
            continue
        subject = world.subjects.get(str(pic.get("id") or "")) or {}
        held = _object_is_held(subject, motion_plan, carry)
        pattern = re.compile(
            rf"(<Subject\s*{int(pic['n'])}>\s+is\s+)(.*?)(?=<Subject\s*\d+>\s+is\s+|\Z)",
            re.S | re.I,
        )

        # Bind this iteration's values; `pattern.sub` calls it right away.
        def replace(match: re.Match[str], pic=pic, subject=subject, held=held) -> str:
            return (
                match.group(1)
                + _object_definition_line(
                    match.group(2),
                    int(pic["n"]),
                    subject,
                    held=held,
                )
                + "\n"
            )

        text = pattern.sub(replace, text)
    out["subject_definitions"] = text
    return out


def _object_constraint_clause(
    picture: dict[str, Any],
    subject: dict[str, Any],
    pictures: list[dict[str, Any]],
    motion_plan: dict[str, Any] | None,
    carry: str,
    focus_text: str,
) -> str:
    """Return a deterministic relation/scale sentence for one object reference."""
    n = int(picture["n"])
    token = f"<Subject {n}>"
    picture_token = f"<Picture {n}>"
    held = _object_is_held(subject, motion_plan, carry)
    scale = _object_scale_anchor(subject, held=held)
    plan = motion_plan or {}
    relation = str(plan.get("relation_change") or "none").strip().lower()
    actor = _object_actor_token(plan, pictures)
    is_impact_instrument = str(
        plan.get("interaction_type") or ""
    ).strip().lower() == "impact" and _asset_text_match(plan.get("instrument"), subject)
    if is_impact_instrument:
        target = _subject_name_token(plan.get("target"), pictures) or "the visible target"
        placement = (
            f"It follows the continuous planned path from {actor} to {target}, remains "
            "visible as one rigid object until the exact contact point, and the target "
            "reacts only after that visible impact"
        )
    elif held:
        placement = (
            f"It is the one handheld prop held by {actor}, in continuous contact with "
            "and moving only with that same hand"
        )
    elif relation == "support":
        placement = (
            f"It keeps the planned support/contact relation with {actor} and remains "
            "physically grounded at the visible contact surface"
        )
    elif relation == "containment":
        placement = (
            f"It keeps the planned containment relation with {actor} at the visible "
            "container boundary, without becoming a separate display object"
        )
    elif relation in {"contact", "configuration", "possession"}:
        placement = (
            f"It keeps the exact planned physical relation with {actor} at the visible "
            "contact or support point, without floating or being staged separately"
        )
    else:
        placement = (
            "It remains physically connected to the actor, support surface, or mounting "
            "established by the shot, and is never staged as an unrelated foreground copy"
        )
    prominence = (
        "The explicit object close-up may control the composition"
        if _item_is_close_up(subject, focus_text)
        else "The object reference does not make it the main character or camera-dominant subject"
    )
    return (
        f"{token} is exactly one inanimate rigid object prop with scale anchor: {scale}; "
        "it is not a character. "
        f"{placement}. Match {picture_token} only for its identity, shape, material, and "
        "color; do not copy the isolated reference image's background, framing, placement, "
        f"apparent scale, or prominence. {prominence}. Keep exactly one instance in the "
        "entire shot, at realistic scale, with no duplicate foreground copy and no body, "
        "face, limbs, clothing, independent pose, or autonomous action."
    )


def _lock_object_reference_constraints(
    fields: dict[str, str],
    pictures: list[dict[str, Any]],
    world: World,
    motion_plan: dict[str, Any] | None,
    carry: str,
    *,
    focus_text: str = "",
) -> dict[str, str]:
    """Add object-only constraints after the model's prompt has been rewritten."""
    object_pictures = [p for p in pictures if p.get("kind") == "object"]
    if not object_pictures:
        return fields
    out = _lock_object_definitions(fields, object_pictures, world, motion_plan, carry)

    detail = str(out.get("detailed_description") or "").strip()
    for pic in object_pictures:
        subject = world.subjects.get(str(pic.get("id") or "")) or {}
        clause = _object_constraint_clause(
            pic,
            subject,
            pictures,
            motion_plan,
            carry,
            " ".join((focus_text, detail)),
        )
        if clause.casefold() not in detail.casefold():
            detail = f"{detail} {clause}".strip()
    out["detailed_description"] = detail
    return out


def _rewrite(
    *,
    pictures: list[dict[str, Any]],
    place: str,
    beat: str,
    dialogue: str,
    continues: bool,
    previous_beat: str,
    setting: str,
    visual_style: str,
    previous_shot: str = "",
    carry: str = "",
    needs_definition: list[dict[str, Any]] | None = None,
    dynamic_state: dict[str, Any] | None = None,
    motion_plan: dict[str, Any] | None = None,
    walked_from: str = "",
) -> dict[str, str]:
    """Six-section prompt for one segment.

    `locked` is the wording earlier segments settled on for each entity; reusing it
    verbatim stops the description (and therefore the character) from drifting.
    """
    plan = _normalize_motion_plan(motion_plan)
    plan_errors = _motion_plan_errors(plan)
    if plan_errors:
        raise ValueError("Invalid motion plan: " + "; ".join(plan_errors))

    payload: dict[str, Any] = {
        "pictures": [{k: v for k, v in pic.items() if k != "ref"} for pic in pictures],
        "place": place,
        "beat": beat,
        "motion_plan": plan,
        "dialogue": dialogue,
        "continues_previous": continues,
        "previous_beat": previous_beat,
        # How the previous segment actually ended, so a continuation opens from
        # the real state instead of re-establishing the shot from scratch.
        "previous_shot_ending": previous_shot,
        # On a scene change the previous beat's ending is not carried (that was another room), but
        # things carried on the person must be: the only temporary state that survives scene
        # changes.
        "carry_over": carry,
        "world_setting": setting,
        "visual_style": visual_style,
        "dynamic_state": dynamic_state or {},
        # Non-empty = this beat is a **transition**: the camera goes from walked_from to place, and
        # both spaces must appear in these seconds. With only the destination, the model's picture
        # stays at the start and it must finish there and then cut.
        "transition_from": walked_from,
        # Only these still need a definition; the rest are composed server-side.
        "needs_definition": [
            {"n": p["n"], "name": p["name"], "kind": p["kind"], "desc": p.get("desc", "")}
            for p in (needs_definition or [])
        ],
    }
    schema = REWRITER_SCHEMA if needs_definition else REWRITER_SCHEMA_NO_DEFS
    tag = "rewrite_defs" if needs_definition else "rewrite"
    audit_error = ""
    for attempt in range(2):
        request = dict(payload)
        if audit_error:
            request["motion_correction"] = (
                "Your last physics audit disagrees with motion_plan. Fix only this conflict: " + audit_error
            )
        result = llm.chat(
            system=REWRITER_SYSTEM,
            user=_pretty(request),
            schema=schema,
            temperature=0.4,
            model=config.REWRITER_MODEL,
            thinking=config.REWRITER_THINKING,
            tag=tag if attempt == 0 else f"{tag}_motion_retry",
        )
        audit_error = _motion_audit_error(result, plan)
        if not audit_error:
            return _ensure_prompt_english(_strip_motion_audit(result))
    raise ValueError("Rewriter ignored the motion plan twice: " + audit_error)


# A definition ends at the next `<Subject N> is`, not at a newline: the rewriter sometimes writes
# two on one line, and splitting by line lets the first swallow the second; `_learn_definitions`
# only writes when no definition exists yet, so that world would carry contradictory conditioning
# forever. 3 of 31 worlds broke this way.
_DEF_SPAN = re.compile(r"<Subject\s*(\d+)>\s+is\s+(.*?)(?=<Subject\s*\d+>\s+is\s|\Z)", re.S)

_PICTURE_REF = re.compile(r"<Picture\s*\d+>")


def _merge_definitions(locked: str, generated: str) -> str:
    """Keep each binding once, preferring the established asset definition."""
    seen = {int(m.group(1)) for m in _DEF_SPAN.finditer(locked)}

    def missing_only(match: re.Match) -> str:
        n = int(match.group(1))
        if n in seen:
            return ""
        seen.add(n)
        return match.group(0)

    generated = _DEF_SPAN.sub(missing_only, generated).strip()
    return "\n".join(part for part in (locked.strip(), generated) if part)


def _definition_from_desc(rec: dict[str, Any], picture: dict[str, Any]) -> str:
    """Entities without settled wording get a definition from their own reference-image prompt instead of another model call.

    ⚠️ This fixes a mismatch: `_agent_schema(bool(_locked_definitions(world)))` stops asking for `subject_definitions`
    once **the world** has any locked description, while the server's need is computed from **this
    beat's references**. If one image has no settled wording (e.g. an asset registered this beat),
    `_six_from_merged` sees a missing section and falls back to a separate rewriter call, about 1 s
    each time.

    `desc` is the English prompt that drew that reference image, the most accurate appearance
    anchor, closer to the image than another description from memory. The result is then stored as
    the locked description by `_learn_definitions`, the same path as model-written ones.

    ⚠️ Not when `desc` is Chinese (co-op AI teammates' random looks are Chinese). Definitions go verbatim into the
    English prompt; Chinese there would move the "translate the physics plan" problem onto subject
    definitions.
    """
    desc = " ".join(str(rec.get("desc") or "").split())
    if not desc or _CJK_TEXT.search(desc):
        return ""
    if picture.get("kind") == "object":
        return _object_definition_line(desc, int(picture["n"]), rec, held=False)
    view = "single wide scene view" if picture.get("kind") == "scene" else "single full-body view"
    noun = "location" if picture.get("kind") == "scene" else "character"
    # `desc` usually starts with framing words (single full-body view of… / establishing wide scene
    # view of… / a wide shot of…); strip them so the sentence does not say it twice.
    tail = re.sub(
        r"^(?:an?\s+)?(?:single|establishing|wide|full[- ]body|close[- ]up|"
        r"medium)?[\s,]*(?:establishing\s+)?(?:full[- ]body|wide\s+scene|wide|"
        r"scene|solo|clean)?\s*(?:view|shot)\s+of\s+",
        "",
        desc,
        flags=re.I,
    ).lstrip(" ,")
    return f"the {noun} shown in <Picture {picture['n']}> as a {view}; {tail}"


def _compose_definitions(world: World, pictures: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """Rebuild the locked part of `subject_definitions` server-side.

    Locked wording is text we already have, so making the rewriter retype it every
    turn burns ~30% of its output tokens for an exact copy. Compose those lines here
    and only ask the model for entities it has never described.

    Each stored body references its own picture, so renumbering is a straight
    substitution of `<Picture N>` with this segment's number for that entity.
    """
    lines: list[str] = []
    missing: list[dict[str, Any]] = []
    for pic in pictures:
        store = _picture_store(world, pic)
        rec = store.get(pic["id"]) or {}
        body = _definition_body(rec.get("definition") or "")
        if body:
            body = _PICTURE_REF.sub(f"<Picture {pic['n']}>", body)
        else:
            # Without settled wording, build one from its own reference-image prompt; cheaper than
            # another rewriter call.
            body = _definition_from_desc(rec, pic)
        if body:
            lines.append(f"<Subject {pic['n']}> is {body}")
        else:
            missing.append(pic)
    return "\n".join(lines), missing


def _ending_of(node: Node | None, budget: int = 1100) -> str:
    """How the previous shot actually ended, for a continuation's rewrite.

    A continuation shares latents with the previous clip, but the prompt was only
    told the previous *beat* (one line), so the rewriter re-established the framing
    and the described state drifted from where the clip actually ended.

    ⚠️ **Strip the boilerplate we add before cutting.** `detailed_description` looks like
    `[Shot 1] Visual style: <style>. <the actual action> Dynamic world state — …`, with both ends
    built by us (`_style_clause` / `_dynamic_visual_clause`). Taking the tail would get depth of
    field, grain and grading plus a state line, nothing about how the camera is placed.

    Seen in practice: the previous clip was `A cinematic wide establishing shot` with a small robot
    far away; the rewriter did not see it, wrote `medium-wide framing`, and the picture jumped from
    a wide shot to a medium close-up. To users, "continuation did not work".

    Shot size is almost always in the **first** sentence of the action and the end state in the
    **last**, so keep both.

    `budget` is measured: over 101 clips the stripped text had a median of 748 characters, p99 1036,
    max 1072, so 1100 almost always fits; the head-and-tail cut below is only for extremes. (The old
    value was 320 and cut **before** stripping: none of the 101 clips fit.) The extra ~250 tokens
    are spent only on continuation beats.
    """
    if node is None:
        return ""
    text = " ".join((((node.six_section or {}).get("detailed_description") or "").replace("\n", " ")).split())
    text = _STYLE_CLAUSE.sub("", text).split(_STATE_PREFIX)[0].strip()
    if len(text) <= budget:
        return text
    head, sep, rest = text.partition(". ")
    keep = budget - len(head) - 4
    # If the head is too long to leave room for the ending, fall back to the tail only: better lose
    # the shot size than keep half an opening sentence.
    return f"{head}. … {rest[-keep:].lstrip()}" if sep and keep > 120 else text[-budget:]


def _learn_definitions(world: World, pictures: list[dict[str, Any]], six: dict[str, str]) -> None:
    """Remember the wording this segment settled on, for every entity that had none.

    `<Subject N>` numbering is per-segment, so only the body after `is` is stored;
    the rewriter renumbers it against the current picture list next time.
    """
    by_n = {p["n"]: p for p in pictures}
    for m in _DEF_SPAN.finditer(six.get("subject_definitions") or ""):
        pic = by_n.get(int(m.group(1)))
        if not pic:
            continue
        body = _definition_body(m.group(2))
        if not body:
            continue
        store = _picture_store(world, pic)
        rec = store.get(pic["id"])
        if rec is None:
            continue
        stored = rec.get("definition") or ""
        # Settled wording is not touched: that is the key against drift. But contaminated wording
        # counts as unsettled, so a clean version replaces it this round.
        if not stored or _DEF_LEAK.search(stored):
            rec["definition"] = body


def _six_from_merged(
    d: dict[str, Any],
    *,
    definitions_locked: bool = False,
) -> dict[str, str] | None:
    """Take the six sections from the merged call's result. None if missing; the caller then calls the rewriter separately.

    Two cases must be covered: `MERGED_AGENT` is a runtime switch and the cache may hold a result
    from the other mode; and the model may leave `detailed_description` empty, the one section that
    cannot be missing (`assemble` sends only it when there are no definitions and no summary).
    """
    # ⚠️ Check `_MODEL_SECTIONS` (the sections the model really writes), not `SECTION_ORDER`. Sections the server builds
    # (retention_analysis) never appear in model output; checking SECTION_ORDER would judge
    # **every** beat incomplete, void the merged path and add 2–3 s per beat for a separate rewriter
    # call.
    required = [k for k in _MODEL_SECTIONS if k != "subject_definitions" or not definitions_locked]
    if not all(k in d for k in required):
        return None
    six = {k: str(d.get(k) or "") for k in _MODEL_SECTIONS}
    # LOCKED is valid only for the ambient sound bed, never for the new shot.
    # A real model response copied it into every section; accepting it would
    # remove the action instead of saving a rewrite.
    if six["detailed_description"].strip() in ("", SOUNDSCAPE_LOCKED):
        return None
    # The merged Agent shares a schema with the Chinese narrative fields.  Run
    # the media-only sections through the same boundary guard used by the
    # standalone rewriter before they can reach the video model.
    return ensure_english_fields(
        six,
        allow_dialogue=True,
        fallbacks=_PROMPT_ENGLISH_FALLBACKS,
        model=config.REWRITER_MODEL,
        thinking=config.REWRITER_THINKING,
        tag="merged_prompt_english_guard",
    )


_NUMBERED_SUBJECT_TOKEN = re.compile(r"<Subject\s+(\d+)>", re.I)

_NUMBERED_PICTURE_TOKEN = re.compile(r"<Picture\s+(\d+)>", re.I)


def _reference_token_error(
    fields: dict[str, str],
    pictures: list[dict[str, Any]],
) -> str:
    """Return why numbered image tokens do not match the files for this request.

    The video prompt must use the current request's ``<Subject N>`` / ``<Picture N>``
    numbers. A stale or out-of-range number is much worse than rejecting the prompt:
    the video model could bind an action or a prop to the wrong image without reporting an error.
    """
    text = "\n".join(str(value or "") for value in fields.values())
    subjects = {int(x) for x in _NUMBERED_SUBJECT_TOKEN.findall(text)}
    picture_tokens = {int(x) for x in _NUMBERED_PICTURE_TOKEN.findall(text)}
    # Very old nodes did not persist the Picture mapping even though their prompt
    # already contained tokens.  Regeneration must remain loadable; new turns and
    # modern nodes always provide ``pictures`` and receive the strict check below.
    if not pictures:
        return ""
    allowed = {int(pic["n"]) for pic in pictures if isinstance(pic.get("n"), int)}
    invalid = sorted((subjects | picture_tokens) - allowed)
    if invalid:
        return "references image numbers that were not uploaded: " + ", ".join(str(x) for x in invalid)
    return ""


def _repair_reference_tokens(
    fields: dict[str, str],
    pictures: list[dict[str, Any]],
) -> tuple[dict[str, str], list[str]]:
    """Replace out-of-range `<Subject N>` / `<Picture N>` with plain nouns; valid ones stay as they are.

    Why repair instead of reject: **23 of 57** merged-path fallbacks (40%) failed here, and it is
    not a semantic error. The merged call cannot know the numbers (they exist only after the server
    picks and orders images), so the prompt asks for `<Ref:name>` and the model occasionally invents
    a `<Picture 3>`.

    Falling back to the separate rewriter cost a p90 of 2.36 s, and **a rewrite does not guarantee
    correct numbers** either. With plain nouns the video model gets "unbound" rather than "wrongly
    bound", strictly better than the case `_reference_token_error` worries about, at no cost.

    Only out-of-range ones change: valid numbers must stay or the reference bindings would be
    undone.
    """
    allowed = {int(pic["n"]) for pic in pictures if isinstance(pic.get("n"), int)}
    notes: list[str] = []
    fixed = dict(fields)

    def swap(pattern: re.Pattern[str], kind: str, generic: str) -> None:
        def one(match: re.Match[str]) -> str:
            n = int(match.group(1))
            if n in allowed:
                return match.group(0)
            notes.append(f"{kind} {n}")
            return generic

        for key, value in list(fixed.items()):
            if isinstance(value, str) and "<" in value:
                fixed[key] = pattern.sub(one, value)

    # Same wording as `_plain_continuation_from_merged`: the same problem (a number pointing
    # nowhere) should look the same on both paths.
    swap(_NUMBERED_SUBJECT_TOKEN, "Subject", "the established character")
    swap(_NUMBERED_PICTURE_TOKEN, "Picture", "the current visual state")
    if not notes:
        return fields, []
    uniq = sorted(set(notes))
    return fixed, [f"out-of-range reference numbers restored to plain nouns: {', '.join(uniq)}"]


_ORDINALS = ("first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth")
# "Hoshi · Star Ranger", "Aren · Axe Warrior · AI": the given name comes before the separator.
_NAME_SEPARATOR = re.compile(r"\s*[·•・|｜/／]\s*")


def _english_name(raw: Any) -> str:
    """The name as it may enter an English video prompt, or "" when it has no Latin form."""
    name = " ".join(str(raw or "").split())
    if name.isascii():
        return name
    head = _NAME_SEPARATOR.split(name, maxsplit=1)[0]
    return head if head.isascii() and any(c.isalpha() for c in head) else ""


def _plain_continuation_from_merged(
    fields: dict[str, str],
    pictures: list[dict[str, Any]],
    world: World | None = None,
) -> dict[str, str]:
    """Compile the merged Agent's shot into a reference-free latent continuation.

    This avoids a second LLM call on the common same-room path. Legacy ``<Ref:...>``
    placeholders and any prematurely invented numeric placeholders become natural
    entity labels; the five reference-only sections are intentionally discarded.
    """
    by_number = {int(pic["n"]): pic for pic in pictures if isinstance(pic.get("n"), int)}

    # Without images these words are all that ties an action to a person, so two characters
    # whose names cannot be written in English must not both become "the established
    # character": they are told apart by order, and the appearance sentences below use the
    # same words.
    ordinal: dict[int, str] = {}
    for group in ("object", "character"):
        unnamed = [
            pic
            for pic in pictures
            if str(pic.get("kind") or "") != "scene"
            and ("object" if pic.get("kind") == "object" else "character") == group
            and not _english_name(pic.get("name"))
        ]
        if len(unnamed) > 1:
            for i, pic in enumerate(unnamed):
                ordinal[id(pic)] = f"the {_ORDINALS[i]} {group}" if i < len(_ORDINALS) else f"{group} {i + 1}"

    def label(pic: dict[str, Any] | None, fallback: str = "the visible subject") -> str:
        if not pic:
            return fallback
        kind = str(pic.get("kind") or "")
        if kind == "scene":
            return "the current story environment"
        # An English name reads better than a role; other scripts never enter a video prompt.
        name = _english_name(pic.get("name"))
        if name:
            return name
        if id(pic) in ordinal:
            return ordinal[id(pic)]
        if kind == "object":
            return "the visible object"
        return "the established character"

    named: dict[str, dict[str, Any]] = {}
    for pic in pictures:
        for key in (pic.get("name"), pic.get("id")):
            normalized = " ".join(str(key or "").split()).casefold()
            if normalized:
                named.setdefault(normalized, pic)
        if pic.get("kind") == "scene" and world is not None:
            main_scene = (world.places.get(str(pic.get("id") or "")) or {}).get("main_scene")
            normalized = " ".join(str(main_scene or "").split()).casefold()
            if normalized:
                named.setdefault(normalized, pic)

    def named_label(match: re.Match[str]) -> str:
        raw = match.group(1).strip()
        return label(named.get(" ".join(raw.split()).casefold()), raw)

    detail = str(fields.get("detailed_description") or "").strip()
    detail = _REF_TOKEN.sub(named_label, detail)
    detail = _NUMBERED_SUBJECT_TOKEN.sub(
        lambda match: label(by_number.get(int(match.group(1)))),
        detail,
    )
    detail = _NUMBERED_PICTURE_TOKEN.sub(
        lambda match: label(by_number.get(int(match.group(1))), "the current visual state"),
        detail,
    )
    detail = re.sub(r"\[Shot\s+\d+]", "", detail, flags=re.I)
    detail = " ".join(detail.split())
    continuity = (
        "The image remains a continuous moving shot inside the current story "
        "environment through the final frame, preserving the parent video's "
        "camera geometry, lighting, background, and subject appearance; every "
        "character stays naturally integrated into that environment in the "
        "action pose reached by this shot."
    )
    if continuity.casefold() not in detail.casefold():
        detail = f"{detail} {continuity}".strip()
    looks = _continuation_appearance(fields.get("subject_definitions") or "", by_number, label)
    if looks:
        detail = f"{detail} {looks}"
    return {
        "subject_definitions": "",
        "summary": "",
        "retention_analysis": "",
        "detailed_description": detail,
        "overall_soundscape": str(fields.get("overall_soundscape") or ""),
        "non_diegetic_music": str(fields.get("non_diegetic_music") or ""),
    }


_DEFINITION_START = re.compile(r"<Subject\s*(\d+)>\s+is\s+", re.I)


def _continuation_appearance(definitions: str, by_number: dict[int, dict[str, Any]], label) -> str:
    """The subject definitions as plain sentences, for a continuation that carries no images.

    The frames carry who is who; the words keep details a close-up may need ("a curved copper
    handle"). The picture-binding part ("shown in <Picture 2> as a single view;") is dropped.
    """
    out = []
    # Definitions may share a line, so split on each "<Subject N> is".
    starts = list(_DEFINITION_START.finditer(definitions))
    for i, hit in enumerate(starts):
        pic = by_number.get(int(hit.group(1)))
        if pic and pic.get("kind") == "scene":
            continue
        text = definitions[hit.end() : starts[i + 1].start() if i + 1 < len(starts) else None].strip()
        if ";" in text and "<Picture" in text.split(";", 1)[0]:
            text = text.split(";", 1)[1]
        text = re.sub(r"\s*(?:as )?shown in <Picture\s*\d+>[^;,.]*", "", text, flags=re.I)
        text = re.sub(r"<(?:Subject|Picture)\s*\d+>", "", text).strip(" .;,")
        if text:
            who = label(pic)
            out.append(f"{who[:1].upper()}{who[1:]}: {text}.")
    return " ".join(out)


_REF_TOKEN = re.compile(r"<Ref:\s*([^<>]+?)\s*>")


def _substitute_refs(
    six: dict[str, str],
    pictures: list[dict[str, Any]],
    world: World | None = None,
) -> tuple[dict[str, str], list[str]]:
    """Compatibility with old caches: replace leftover `<Ref:name>` with the real `<Subject N>`.

    The current prompt asks the model to write numbers directly in the order of `subjects`; this
    conversion exists only for old caches and is not an output protocol.

    Unknown names **become plain text** instead of staying as broken tokens: a leftover `<Ref:Zhang
    San>` would be read literally into the picture by the video model; "Zhang San" is at least a
    fluent sentence, only without a reference image for that person in this clip. Unknown names are
    returned so the caller can decide whether to report them.
    """

    def norm(x: Any) -> str:
        return " ".join(str(x or "").split()).casefold()

    index: dict[str, int] = {}
    for pic in pictures:
        keys = [pic.get("name"), pic.get("id")]
        # The place image must also match its `main_scene` alias: the snapshot may show a place as
        # "inside the industrial warehouse" while its key is "industrial warehouse corridor". The
        # model copies the snapshot, and a mismatch would drop a reference image.
        if world is not None and pic.get("kind") == "scene":
            keys.append((world.places.get(str(pic.get("id"))) or {}).get("main_scene"))
        for key in keys:
            key = norm(key)
            if key:
                index.setdefault(key, pic["n"])
    unknown: list[str] = []

    def one(m: re.Match[str]) -> str:
        raw = m.group(1).strip()
        n = index.get(norm(raw))
        if n is None:
            if raw not in unknown:
                unknown.append(raw)
            return raw
        return f"<Subject {n}>"

    return {k: _REF_TOKEN.sub(one, str(v or "")) for k, v in six.items()}, unknown


# ─────────── Parallel split: visuals and bookkeeping as two concurrent calls ───────────
#
# The only approach measured to bring the agent side under 1 s **without dropping anything**.
#
# Measured (one OpenAI-compatible server, the same payload each time):
#     one call writing 33 fields           3.60 s   1356–1905 tokens out
#     visuals only, 15 fields              1.17 s (TP=1) / 0.93 s (TP=2)
#     bookkeeping, 18 fields (parallel)    2.0–2.8 s
#
# The video model waits only for the visual call. Bookkeeping finishes during rendering and users
# never notice; the same pattern as the critic (`adjudicate` runs during playback).
#
# ⚠️ The cost: this beat's six sections get **no motion plan injected** and **no multi-step split** (continue_with
# waits for the bookkeeping call and only helps the next beat). So it is off by default.
#

# ⚠️ **Two pools; never merge them.**
#
# The bookkeeping call takes 2–3 s and **cannot be cancel()ed once running**. Sharing a pool with
# the visual calls, it fills the workers within a few beats and visual jobs cannot even start. The
# symptoms are very misleading:
#
#     shared pool (16 workers)  median 0.01 s, all 20 beats used the fallback: looks like a broken deadline
#     shared pool (4 workers)   hedging median 1.91 s: looks like hedging does not work
#
_split_pool = futures.ThreadPoolExecutor(max_workers=16, thread_name_prefix="agent-book")
