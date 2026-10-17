"""Language guards for prompts sent to English-only media models.

The text models are allowed to return Chinese UI/story fields.  The media
caption fields are a separate contract: the video caption and the English image prompt
paths must not receive Chinese outside an explicit dialogue block.  This module
keeps that boundary in one place and provides a best-effort repair with a
deterministic English fallback when the repair model is unavailable.
"""

from __future__ import annotations

import json
import re
from typing import Any
from collections.abc import Mapping

from murea_agent import config

_CJK_TEXT = re.compile(r"[\u3400-\u9fff]")
_DIALOGUE_BLOCK = re.compile(r"<d\b[^>]*>.*?</d>", re.I | re.S)
_PROTECTED_TOKEN = re.compile(
    r"<\s*(?:Subject|Picture|Audio)\s+\d+\s*>|\(S\d+\)|"
    r"\bPicture\s+\d+\s*:\s*|__BOUND_SUBJECT_\d+__",
    re.I,
)


def contains_non_latin_text(value: Any, *, allow_dialogue: bool = False) -> bool:
    """Detect non-Latin letters (Han, kana, Hangul, Cyrillic, ...).

    A script check, not language identification: accented Latin names and
    typographic punctuation are fine in English captions.
    """
    import unicodedata

    text = str(value or "")
    if allow_dialogue:
        text = re.sub(r"<d\b[^>]*>.*?</d>", "", text, flags=re.I | re.S)
    return any(
        char.isalpha() and not char.isascii() and "LATIN" not in unicodedata.name(char, "") for char in text
    )


def validate_english_fields(fields: Mapping[str, Any], *, allow_dialogue: bool = False) -> None:
    """Reject media prompt fields that still contain non-Latin prose."""
    bad = [
        key for key, value in fields.items() if contains_non_latin_text(value, allow_dialogue=allow_dialogue)
    ]
    if bad:
        raise ValueError(
            "Untranslated non-Latin text in prompt fields: "
            + ", ".join(bad)
            + ". Write them in English; spoken lines go in <d>[Language] ...</d>."
        )


def contains_cjk(value: Any, *, allow_dialogue: bool = False) -> bool:
    """Return whether *value* contains CJK text outside an optional <d> block."""
    text = str(value or "")
    if allow_dialogue:
        text = _DIALOGUE_BLOCK.sub("", text)
    return bool(_CJK_TEXT.search(text))


def _mask(value: Any, *, allow_dialogue: bool, prefix: str = "") -> tuple[str, dict[str, str]]:
    text = " ".join(str(value or "").split())
    masks: dict[str, str] = {}
    if allow_dialogue:
        index = 0

        def dialogue(match: re.Match[str]) -> str:
            nonlocal index
            marker = f"__{prefix}DIALOGUE_{index}__"
            index += 1
            masks[marker] = match.group(0)
            return marker

        text = _DIALOGUE_BLOCK.sub(dialogue, text)

    index = 0

    def token(match: re.Match[str]) -> str:
        nonlocal index
        marker = f"__{prefix}BOUND_TOKEN_{index}__"
        index += 1
        masks[marker] = match.group(0)
        return marker

    # Protect model-facing binding tokens.  They are ASCII already, but a
    # translation pass must not renumber or rewrite them.
    text = _PROTECTED_TOKEN.sub(token, text)
    return text, masks


def _restore(value: Any, masks: Mapping[str, str]) -> str:
    text = " ".join(str(value or "").split())
    # Restore longest markers first so a marker can never be partially matched.
    for marker, original in sorted(masks.items(), key=lambda item: -len(item[0])):
        text = text.replace(marker, original)
    return text


def _strip_cjk(value: Any, *, allow_dialogue: bool = False) -> str:
    """Keep any usable ASCII prose as a last-resort fallback."""
    text = str(value or "")
    dialogue_masks: dict[str, str] = {}
    if allow_dialogue:

        def dialogue(match: re.Match[str]) -> str:
            marker = f"__SAFE_DIALOGUE_{len(dialogue_masks)}__"
            dialogue_masks[marker] = match.group(0)
            return marker

        text = _DIALOGUE_BLOCK.sub(dialogue, text)
    text = _CJK_TEXT.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    for marker, original in dialogue_masks.items():
        text = text.replace(marker, original)
    return text


def ensure_english_fields(
    fields: Mapping[str, Any],
    *,
    allow_dialogue: bool = False,
    fallbacks: Mapping[str, str] | None = None,
    model: str | None = None,
    thinking: int | None = 0,
    tag: str = "english_prompt_repair",
    translate: bool = True,
) -> dict[str, str]:
    """Return *fields* with CJK removed from English-only values.

    Only contaminated fields trigger an LLM call, so normal English output is
    zero-cost.  Protected prompt tokens and explicit ``<d>`` dialogue blocks are
    restored verbatim after translation.  If translation fails or still contains
    CJK, an English fallback is used instead of sending unsafe text downstream.
    """
    out = {key: str(value or "") for key, value in fields.items()}
    bad = [key for key, value in out.items() if contains_cjk(value, allow_dialogue=allow_dialogue)]
    if not bad:
        return out

    # Style bibles are often short Chinese labels (for example "电影感").
    # Translating those labels with an additional model call would add latency to
    # every co-op preparation and also makes media generation depend on a second
    # provider.  Callers can request the deterministic fallback for this case.
    if not translate:
        defaults = fallbacks or {}
        for key in bad:
            candidate = _strip_cjk(out[key], allow_dialogue=allow_dialogue)
            out[key] = (
                candidate
                if len(re.sub(r"[^A-Za-z]", "", candidate)) >= 3
                else str(defaults.get(key) or "cinematic, coherent visual design, detailed lighting")
            )
        return out

    masked: dict[str, str] = {}
    masks: dict[str, str] = {}
    field_masks: dict[str, dict[str, str]] = {}
    for key in bad:
        safe_prefix = re.sub(r"[^A-Za-z0-9]", "_", str(key)).upper() + "_"
        text, local = _mask(out[key], allow_dialogue=allow_dialogue, prefix=safe_prefix)
        masked[key] = text
        masks.update(local)
        field_masks[key] = local

    schema = {
        "type": "object",
        "properties": {key: {"type": "string"} for key in bad},
        "required": bad,
        "additionalProperties": False,
    }
    translated: dict[str, Any] = {}
    error = ""
    try:
        # Lazy import avoids the imagegen -> llm -> service import cycle.
        from murea_agent.providers import llm
        from murea_agent import reqlog

        translated = llm.chat(
            system=(
                "Translate the supplied media-prompt fields into concise natural English. "
                "Return only the requested JSON object. Preserve every protected token "
                "exactly, including markers matching __*_BOUND_TOKEN_N__ and "
                "__*_DIALOGUE_N__. Do not output "
                "Chinese or any non-English prose outside dialogue placeholders. Preserve "
                "the original visual facts and physical actions; do not invent events."
            ),
            user=json.dumps({"fields": masked}, ensure_ascii=False),
            schema=schema,
            temperature=0.1,
            model=model or config.REWRITER_MODEL,
            thinking=thinking if thinking is not None else config.REWRITER_THINKING,
            tag=tag,
        )
        if not isinstance(translated, dict):
            raise ValueError("English repair did not return an object")
        defaults = fallbacks or {}
        for key in bad:
            raw_value = str(translated.get(key) or "")
            missing_markers = [marker for marker in field_masks[key] if marker not in raw_value]
            if missing_markers:
                raise ValueError(f"{key} dropped protected media tokens")
            value = _restore(raw_value, masks)
            if contains_cjk(value, allow_dialogue=allow_dialogue):
                raise ValueError(f"{key} still contains CJK")
            if not value.strip() and defaults.get(key):
                value = str(defaults[key])
            out[key] = value
        reqlog.record(tag, ok=True, fields=bad)
        return out
    except Exception as exc:  # noqa: BLE001 - fallback must not block a turn
        error = f"{type(exc).__name__}: {exc}"
        try:
            from murea_agent import reqlog

            reqlog.record(tag, ok=False, fields=bad, error=error)
        except Exception:
            pass

    defaults = fallbacks or {}
    for key in bad:
        candidate = _strip_cjk(out[key], allow_dialogue=allow_dialogue)
        # Keep a stripped value only when it still carries meaningful prose.
        out[key] = (
            candidate
            if len(re.sub(r"[^A-Za-z]", "", candidate)) >= 3
            else str(defaults.get(key) or "The current story action is shown in one continuous shot.")
        )
    return out


def ensure_english_text(
    value: Any,
    *,
    fallback: str,
    allow_dialogue: bool = False,
    model: str | None = None,
    thinking: int | None = 0,
    tag: str = "english_text_repair",
    translate: bool = True,
) -> str:
    """Convenience wrapper for one prompt or style string."""
    return ensure_english_fields(
        {"value": value},
        allow_dialogue=allow_dialogue,
        fallbacks={"value": fallback},
        model=model,
        thinking=thinking,
        tag=tag,
        translate=translate,
    )["value"]
