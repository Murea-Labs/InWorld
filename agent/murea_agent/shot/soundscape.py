"""Ambience and speech directions of the video prompt: each place keeps one locked room tone."""

from __future__ import annotations

import re
from murea_agent.memory.world import World


def _place_soundscape(world: World, place: str) -> str:
    asset = world.places.get(place) or {}
    raw = str(asset.get("soundscape") or "").strip()
    bed = _strip_speech_directive(raw)
    # Older versions may have written each beat's no-speech constraint (often cut at [:400]) into
    # the place asset. Migrate on read; after the next save the disk only holds the ambience that
    # belongs to the place.
    if raw != bed and place in world.places:
        if bed:
            world.places[place]["soundscape"] = bed
        else:
            world.places[place].pop("soundscape", None)
    return bed


def _learn_soundscape(world: World, place: str, six: dict[str, str]) -> None:
    """The first time a place is entered, lock the ambience the model wrote onto the place asset.

    Locked once, then reused verbatim: the same mechanism as locking the visual style and character
    definitions.
    """
    if not place or _place_soundscape(world, place):
        return
    text = _strip_speech_directive(six.get("overall_soundscape") or "")
    if len(text) >= 20 and place in world.places:
        world.places[place]["soundscape"] = text[:400]


_SPEECH_DIRECTIVE = re.compile(
    r"\s*(?:No speech[^.]*\.|Diegetic ambience[^.]*\.|"
    r"[^.]*\bnobody talks in this shot\.|[^.]*\bspeech written in the shot\.)\s*",
    re.I,
)


# Compatibility with long no-speech text from old caches / models. The current format keeps only a
# short `No speech ... nobody talks in this shot.` and never sends internal anti-crosstalk notes to
# the model.
_NO_VOICE_DIRECTIVE = re.compile(
    r"\s*ABSOLUTE NO-VOICE RULE\s*:.*$",
    re.I | re.S,
)


def _strip_extra_no_voice(text: str) -> str:
    """Remove the long no-speech text added by older versions, keeping the ambience and the short speech instruction."""
    return _NO_VOICE_DIRECTIVE.sub("", str(text or "")).strip()


def _strip_speech_directive(text: str) -> str:
    """Strip the "does anyone speak" sentence; what remains is this place's ambience."""
    without_lock = _NO_VOICE_DIRECTIVE.sub(" ", str(text or ""))
    return _SPEECH_DIRECTIVE.sub(" ", without_lock).strip()


def _speech_directive(dialogue: str) -> str:
    return (
        "Plus the speech written in the shot."
        if str(dialogue or "").strip()
        else ("No speech, no voice-over, no narration -- nobody talks in this shot.")
    )


def _lock_soundscape(six: dict[str, str], world: World, place: str, dialogue: str) -> dict[str, str]:
    """The same place reuses its ambience verbatim.

    ⚠️ Not only to save tokens; it fixes an audible problem. Three consecutive beats in the same room:

        beat 1  low hum of a high-speed train engine + hissing vents + distant metallic footsteps
        beat 2  train engine hum + flickering fluorescent lights + creaking metal locker doors
        beat 3  low rumble of wheels on rails + fluorescent buzz + medical equipment

    The same room sounded different every few seconds, the same class of problem as visual style
    drift, solved the same way: by locking.

    Why action sounds can be dropped: the video model **sees the picture**. Footsteps, doors, metal
    scraping follow the action and it generates them itself; it cannot infer only two things: the
    room tone of this space, and whether anyone speaks in this beat. So the room tone is locked and
    the speech instruction recomputed each beat.

    Without a lock (first visit, or an old world without the field) it returns the input unchanged.
    """
    bed = _place_soundscape(world, place)
    if not bed:
        return six
    out = dict(six)
    out["overall_soundscape"] = f"{bed} {_speech_directive(dialogue)}".strip()
    return out


# When the place ambience is locked, the model writes only this word and the server fills it in. It
# saves **output on the critical path**: `_lock_soundscape` replaces whatever the model wrote
# anyway, and most consecutive beats are in the same place, so that text was almost always written
# and thrown away.
#
# Why not remove the field from the schema: the place is decided by the model itself and unknown
# when the schema is built. And **the beat that changes place needs the model's full text most**:
# `_learn_soundscape` locks it onto the new place, and a bad one stays bad forever.
SOUNDSCAPE_LOCKED = "LOCKED"


def _ensure_soundscape(fields: dict[str, str], dialogue: str = "") -> None:
    """Complete the audio direction in the original six-section format. In place.

    `overall_soundscape` is the video model's only audio direction. When empty it still states
    whether this beat has dialogue, without pasting long internal anti-crosstalk details into the
    video prompt.

    "Does anyone speak" needs no model judgement: `dialogue` already says so, so the fallback text
    is deterministic.
    """
    raw = fields.get("overall_soundscape") or ""
    # The sentinel must never reach the prompt. If the place really is locked, `_lock_soundscape`
    # overwrites it next; if the sentinel was written without a lock (the model misjudged the
    # place), treat it as empty and use the deterministic fallback.
    if raw.strip().upper() == SOUNDSCAPE_LOCKED:
        raw = ""
    value = _strip_extra_no_voice(raw)
    has_dialogue = bool(str(dialogue or "").strip())
    if has_dialogue:
        if not value or value.casefold() == "n/a":
            fields["overall_soundscape"] = (
                "Diegetic ambience of the location, plus the speech written in the shot."
            )
        else:
            fields["overall_soundscape"] = value
        return

    # Keep the wind, rain, footsteps, metal scraping etc. the rewriter wrote, and only add back the
    # short speech sentence the format requires; repeated calls (e.g. rerunning a clip) do not keep
    # growing it.
    if not value or value.casefold() == "n/a":
        value = (
            "Diegetic ambience of the location only. No speech, no voice-over, "
            "no narration -- nobody talks in this shot."
        )
    elif not re.search(r"\bNo speech\s*,\s*no voice-over\s*,\s*no narration\b", value, re.I):
        value = f"{value} No speech, no voice-over, no narration -- nobody talks in this shot."
    fields["overall_soundscape"] = value
