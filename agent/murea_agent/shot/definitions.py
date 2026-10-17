"""Locked English definitions of assets: how each character, creature, object and place is described to the video model."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any
from murea_agent.util import _clean_motion_text
from murea_agent import config
from murea_agent.director.prompt_language import ensure_english_text
from murea_agent.memory.world import World


_DEF_HEAD = re.compile(r"^<Subject\s+\d+>\s+is\s+", re.I)


# The whole "shown in <Picture 3> as a single full-body view" phrase plus the semicolon or comma
# right after it. View phrases never contain a semicolon / comma, so cutting there is safe.
_DEF_MIDDLE = re.compile(r"\s*shown\s+in\s+<Picture\s*\d+>\s*(?:as\s+[^;,<]*)?[;,]?\s*", re.I)


_DEF_BOILERPLATE = re.compile(
    r"^the\s+(single\s+)?"
    r"(full-body\s+view\s+)?(wide\s+)?(scene\s+)?(portrait\s+)?"
    r"(character\s+sheet|scene\s+view|view|reference)\s*(of\s+)?",
    re.I,
)


# The retention line keeps the full visible appearance; it must not be cut at 90 characters into
# ``standing upright with a.`` like the old version. The cap is only a guard for abnormally long
# model output; normal character / place definitions (usually 150--320 characters) stay complete.
_RETENTION_DEFINITION_MAX_CHARS = 360


def _short_definition(rec: dict[str, Any]) -> str:
    """Clean a locked English definition so the retention line can reuse the full visible appearance."""
    text = " ".join(str(rec.get("definition") or "").split())
    # Old worlds may have persisted a Chinese definition before the language
    # boundary was enforced.  Do not echo it into retention_analysis; the caller
    # will use a generic English description until the asset is rewritten.
    if _CJK_TEXT.search(text):
        return "the established subject shown in the attached reference"
    text = _DEF_HEAD.sub("", text)
    # ⚠️ Definitions come in at least **three** formats, each seen in real runs:
    #
    #   A  is <full appearance> shown in <Picture N> as <view>; <pose>
    #   B  is the single character sheet shown in <Picture N> as <view>; <full appearance>
    #   C  is <phrase> shown in <Picture N> as <view>, <full appearance>
    #
    # The stable appearance may come before or after the reference phrase; the other side is often
    # only this image's temporary pose such as ``standing upright``. Clean both sides and take the
    # more informative one: it covers all three formats without locking a pose into every beat.
    parts = [_DEF_BOILERPLATE.sub("", part).strip(" ,.;") for part in _DEF_MIDDLE.split(text)]
    text = max((part for part in parts if part), key=len, default="")
    if len(text) > _RETENTION_DEFINITION_MAX_CHARS:
        # Prefer ending at a full sentence; only without a period fall back to a word boundary,
        # never leaving half a word or phrase. This branch only guards against very dirty output and
        # never shortens normal definitions.
        window = text[: _RETENTION_DEFINITION_MAX_CHARS + 1]
        endings = [m.end() for m in re.finditer(r"[.!?](?:\s|$)", window)]
        if endings and endings[-1] >= 120:
            text = text[: endings[-1]]
        else:
            text = text[:_RETENTION_DEFINITION_MAX_CHARS].rsplit(" ", 1)[0]
    return text.rstrip(" ,.") or "as shown in the reference"


_ITEM_KINDS = {"物件", "物品", "道具"}


def _is_item_record(subject: dict[str, Any] | None) -> bool:
    return str((subject or {}).get("kind") or "").strip() in _ITEM_KINDS


def _subject_name_token(value: Any, pictures: list[dict[str, Any]] | None = None) -> str:
    """Resolve names of assets with bound reference images into the video model's stable Subject tokens.

    Names are display names from the world data and may be Chinese; they must not be translated just
    to build the prompt. If the asset has a bound reference image in this beat, use its ``<Subject
    N>``; unbound names stay as they are for the upstream English rewriter or the video model to
    interpret.
    """
    text = _clean_motion_text(value)
    if not text or not pictures:
        return text
    aliases: list[tuple[str, str]] = []
    for picture in pictures:
        number = picture.get("n")
        if not isinstance(number, int):
            continue
        token = f"<Subject {number}>"
        # Only replace display names the model wrote. Asset IDs may be ``S1``, which clashes with
        # the video model's ``(S1)`` speaker token; legacy ``<Ref:ID>`` is handled by the
        # compatibility function, so bare IDs are not aliases in the text.
        for alias in (picture.get("name"),):
            alias = _clean_motion_text(alias)
            if alias and alias.casefold() != token.casefold():
                aliases.append((alias, token))
    # Longer names first, so "hall" is not replaced inside "the dim hall of the old mansion".
    aliases.sort(key=lambda item: len(item[0]), reverse=True)
    for alias, token in aliases:
        if re.search(r"[A-Za-z0-9_]", alias):
            pattern = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(alias)}(?![A-Za-z0-9_])", re.I)
        else:
            pattern = re.compile(re.escape(alias))
        text = pattern.sub(token, text)
    return text


_CJK_TEXT = re.compile(r"[\u3400-\u9fff]")


def _english_asset_prompt(value: Any, *, fallback: str) -> str:
    """Normalize a newly declared image prompt before storing it as an asset.

    ``brief``/name fields remain Chinese UI data, but ``image_prompt`` is later
    reused as an English visual definition.  Persisting a repaired value avoids
    repeatedly asking the rewriter to recover the same contaminated asset.
    """
    return ensure_english_text(
        value,
        fallback=fallback,
        model=config.GM_MODEL,
        thinking=config.GM_THINKING,
        tag="asset_prompt_english_guard",
    )


def _object_appearance(body: Any) -> str:
    """Strip our deterministic object wrapper back to its stable visual description."""
    appearance = " ".join(str(body or "").split())
    appearance = (
        re.sub(
            r"^the\s+single\s+inanimate\s+(?:object(?:\s+prop)?|prop)\b"
            r"(?:\s+shown\s+in\s+<Picture\s*\d+>\s+as\s+a\s+single\s+isolated\s+"
            r"object(?:\s+(?:view|reference))?)?\s*;?",
            "",
            appearance,
            flags=re.I,
        ).strip(" ;,:")
        or "the object shown in the reference"
    )
    appearance = (
        re.split(
            r"\s*;?\s*scale anchor\s*:",
            appearance,
            maxsplit=1,
            flags=re.I,
        )[0].strip(" ;,:")
        or "the object shown in the reference"
    )
    # An older backfill wrote items as "character shown in Picture ...; appearance" and wrapped them
    # again later. Reuse the definition parser and keep only the appearance, never the inner type /
    # framing text.
    return _short_definition({"definition": appearance})


_DEF_LEAK = re.compile(r"<Subject\s*\d+>\s+is\s")


def _definition_body(text: str) -> str:
    """One entity's description, with anything that leaked in from the next cut off.

    Also applied on the way *out* so worlds whose store was polluted before this
    fix stop emitting the bad text; `_learn_definitions` then rewrites them clean.
    """
    body = " ".join(_DEF_LEAK.split(text or "", maxsplit=1)[0].split())
    return "" if _CJK_TEXT.search(body) else body


@dataclass
class _PendingImage:
    """A reference image that still has to be drawn."""

    kind: str  # "place" | "subject"
    key: str  # place name or subject id
    prompt: str
    visual_style: str = ""
    brief: str = ""  # a short note shown in the chat after drawing, saying what this is
    # `kind` is the storage bucket; `asset_kind` is the image prompt contract.
    # Objects live in world.subjects but must use imagegen's object suffix.
    asset_kind: str = ""


def _locked_definitions(world: World) -> dict[str, str]:
    """Entity descriptions already settled, by name, for the merged call.

    With two calls this was assembled by `_compose_definitions` before the second call and given to
    the rewriter. With one merged call there is no second call, so it must be in the snapshot up
    front; otherwise the model describes everyone again, which is slow and conflicts with the locked
    wording.
    """
    out: dict[str, str] = {}
    for sid, rec in world.subjects.items():
        body = _definition_body(rec.get("definition") or "")
        if body:
            # The director reads the item's stable appearance, not character / framing wrappers
            # stacked on it historically.
            out[rec.get("name", sid)] = _object_appearance(body) if _is_item_record(rec) else body
    for name, place in world.places.items():
        body = _definition_body(place.get("definition") or "")
        if body:
            out[name] = body
    return out
