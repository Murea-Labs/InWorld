"""Dialogue in the video prompt: language tags and which character speaks each line."""

from __future__ import annotations

import re
from typing import Any
from murea_agent import config
from murea_agent.memory.world import World
from murea_agent.director.agent_schema import (
    SECTION_ORDER,
)
from murea_agent.shot.definitions import (
    _is_item_record,
)


_D_NO_LANG = re.compile(r"<d>\s*(?!\[)", re.I)


def _lock_dialogue_language(six: dict[str, str], lang: str = "") -> dict[str, str]:
    """Add a language tag to every dialogue line.

    The video model's dialogue format is `(S1) <d>[Chinese] words</d>`; the language name in
    brackets is the **only** thing telling it which language to pronounce. Without it the model
    invents an unrecognizable language. The rule is in the rewriter prompt, but the model does not
    always follow it, and adding the tag is fully deterministic, so it is not left to the model's
    memory.
    """
    lang = lang or config.LANGUAGE_NAME
    out = dict(six)
    for k, v in out.items():
        if v and "<d>" in v:
            out[k] = _D_NO_LANG.sub(f"<d>[{lang}] ", v)
    return out


def _has_dialogue(six: dict[str, str]) -> bool:
    return any("<d>" in (v or "") for v in six.values())


# Before a name there can be the start of a line, a semicolon, or the previous sentence's final
# punctuation or closing quote: the director often writes two people's lines on one line ("A shouts:
# now! B grits her teeth: don't give up!"). Recognizing only line starts would lose the second
# speaker, and one_speaker below would lock both lines to the first (several opening lines of one
# character were voiced with another's voice).
_DIALOGUE_SPEAKER_PREFIX = re.compile(
    r"(?:^|[\r\n；;。！？!?”」』…])\s*([^：:\r\n；;。！？!?“”「」『』…]{1,48})\s*[：:]"
)


_SPEECH_VERB_SUFFIX = re.compile(
    r"(?:压低声音|低声|轻声|小声|大声|高声|沉声|冷冷地|缓缓地|平静地|"
    r"严肃地|急促地|悄声|笑着)?(?:说道|说|问道|问|答道|回答|喊道|喊|"
    r"叫道|叫|回应|低语)\s*$"
)


_DIALOGUE_SPEAKER_MARK = re.compile(
    r"(?:(?P<onscreen>(?:<Subject\s+\d+>|S\d+)\s+speaks\s+on-screen\s+as\s+S\d+)"
    r"|(?P<offscreen>an\s+off-screen\s+voice\s+identified\s+as\s+S\d+\s+says)"
    r")?(?P<separator>\s*:\s*)?(?P<tag>\(S\d+\))(?=\s*<d>)",
    re.I,
)


def _compact_speaker_label(value: Any) -> str:
    """Normalize a human-facing role/name for conservative alias matching."""
    return re.sub(r"[\s\"'“”‘’（）()，,。.!！?？、]+", "", str(value or "")).casefold()


def _dialogue_speaker_labels(dialogue: str) -> list[str]:
    """Read ordered speaker labels from the GM's documented named-dialogue form."""
    labels: list[str] = []
    for match in _DIALOGUE_SPEAKER_PREFIX.finditer(dialogue or ""):
        label = _SPEECH_VERB_SUFFIX.sub("", match.group(1).strip()).strip()
        if label:
            labels.append(label)
    return labels


def _resolve_dialogue_speaker(world: World, subject_ids: list[str], label: str) -> str | None:
    """Resolve a GM speaker label to one visible character without guessing ties.

    New NPCs occasionally arrive with a stable-id-looking name such as S3 while
    their human role only appears in brief (for example 地下酒吧的酒保). Match the
    id/name first and the brief second. Objects are excluded: an accidentally
    registered scene asset must never become the voice source.
    """
    wanted = _compact_speaker_label(label)
    if not wanted:
        return None
    scored: list[tuple[int, str]] = []
    for sid in subject_ids:
        rec = world.subjects.get(sid) or {}
        if _is_item_record(rec):
            continue
        sid_key = _compact_speaker_label(sid)
        name = _compact_speaker_label(rec.get("name"))
        brief = _compact_speaker_label(rec.get("brief"))
        score = 0
        if wanted == sid_key:
            score = 120
        elif wanted == name:
            score = 110
        elif len(name) >= 2 and (name in wanted or wanted in name):
            score = 95
        elif len(wanted) >= 2 and brief and wanted in brief:
            score = 85
        if score:
            scored.append((score, sid))
    if not scored:
        return None
    best = max(score for score, _ in scored)
    winners = [sid for score, sid in scored if score == best]
    return winners[0] if len(winners) == 1 else None


def _align_dialogue_anchor(six: dict[str, str]) -> dict[str, str]:
    """Make the speaker anchor and the `(Sn)` right before `<d>` point to the same person.

    Only the number in the anchor changes, never the line or `(Sn)` itself: the server does not know
    who speaks here, only that the two must not disagree.
    """
    out = dict(six)

    def fix(match: re.Match[str]) -> str:
        tag = match.group("tag")
        n = tag[2:-1]
        if match.group("onscreen"):
            anchor = f"<Subject {n}> speaks on-screen as S{n}"
        elif match.group("offscreen"):
            anchor = f"an off-screen voice identified as S{n} says"
        else:
            return match.group(0)
        return f"{anchor}{match.group('separator') or ''}{tag}"

    for key in SECTION_ORDER:
        text = out.get(key) or ""
        if "<d>" in text:
            out[key] = _DIALOGUE_SPEAKER_MARK.sub(fix, text)
    return out


def _lock_dialogue_speakers(
    six: dict[str, str],
    dialogue: str,
    world: World,
    pictures: list[dict[str, Any]],
) -> dict[str, str]:
    """Deterministically align speaker tokens with character picture order.

    The rewriter can describe Subject 2 (the NPC) correctly yet emit (S1) beside
    the spoken line. The video model follows that token and makes the player talk. The GM's
    named dialogue and picture ordering are authoritative, so fix all correlated
    subject, voice-source, and dialogue markers immediately before generation.
    """
    character_pictures = [p for p in pictures if p.get("kind") == "character"]
    subject_ids = [str(p.get("id") or "") for p in character_pictures]
    picture_n = {
        str(p.get("id") or ""): int(p["n"])
        for p in character_pictures
        if p.get("id") and isinstance(p.get("n"), int)
    }
    speaker_ns = [
        picture_n.get(_resolve_dialogue_speaker(world, subject_ids, label) or "")
        for label in _dialogue_speaker_labels(dialogue)
    ]
    if not any(speaker_ns):
        # When the speaker could not be resolved this used to return unchanged, leaving the model's
        # anchor in the prompt, often self-contradictory: `<Subject 3> speaks on-screen as S1:
        # (S3)<d>…` names two people for one line, and the video model then puts lips and voice on
        # the wrong person. The (Sn) right before <d> is who really speaks; align the anchor to it
        # so the two at least agree.
        return _align_dialogue_anchor(six)

    # A four-second beat normally has one speaker. If a line contains multiple
    # speakers, preserve their documented order and align it with the <d> order.
    known = [n for n in speaker_ns if n is not None]
    # Only when every name was recognized and they are the same person can all lines be locked to
    # them. If one is unknown there may be a second person: match in order and leave mismatches as
    # they are.
    one_speaker = known[0] if known and len(known) == len(speaker_ns) and len(set(known)) == 1 else None
    cursor = 0
    out = dict(six)
    for key in SECTION_ORDER:
        text = out.get(key) or ""
        if "<d>" not in text:
            continue

        def replace(match: re.Match[str]) -> str:
            nonlocal cursor
            n = one_speaker or (speaker_ns[cursor] if cursor < len(speaker_ns) else None)
            cursor += 1
            if n is None:
                return match.group(0)
            if match.group("onscreen"):
                anchor = f"<Subject {n}> speaks on-screen as S{n}"
            elif match.group("offscreen"):
                anchor = f"an off-screen voice identified as S{n} says"
            else:
                anchor = ""
            return f"{anchor}{match.group('separator') or ''}(S{n})"

        out[key] = _DIALOGUE_SPEAKER_MARK.sub(replace, text)
    return out


def _lock_dialogue_performance(
    six: dict[str, str],
    dialogue: str,
) -> dict[str, str]:
    """Tell the video model to act the line instead of reading it in the reference clip's mood."""
    if not str(dialogue or "").strip() or not _has_dialogue(six):
        return six
    out = dict(six)
    detail = (out.get("detailed_description") or "").strip()
    direction = "Match vocal emotion, breath, volume, pace, and effort to the speaker's immediate state."
    if direction.casefold() not in detail.casefold():
        out["detailed_description"] = f"{detail} {direction}".strip()
    return out
