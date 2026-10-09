"""Small helpers shared by the beat pipeline modules."""

from __future__ import annotations

import re
import shutil
import time
from pathlib import Path
from typing import Any
from murea_agent import config


PROMPTS = Path(__file__).parent / "director" / "prompts"


def load_prompt(name: str) -> str:
    """A system prompt from prompts/, with the shared video-prompt rules and the
    story language filled in."""
    text = (PROMPTS / name).read_text()
    text = text.replace("{{VIDEO_PROMPT_RULES}}", (PROMPTS / "video_prompt.md").read_text())
    return text.replace("{{LANGUAGE}}", config.LANGUAGE_NAME)


def _clean_motion_text(value: Any) -> str:
    """Collapse whitespace; None becomes an empty string."""
    return " ".join(str(value or "").split())


def cap(text: str, zh_limit: int) -> str:
    """Shorten text to a length budget written for Chinese. English needs about 2.5x
    the characters for the same content, and is cut at a word boundary."""
    text = " ".join(str(text or "").split())
    limit = zh_limit if config.LANGUAGE == "zh" else int(zh_limit * 2.5)
    if len(text) <= limit:
        return text
    cut = text[:limit]
    if config.LANGUAGE != "zh" and " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    return cut.rstrip(" ,;:")


CARRY_SEP = re.compile(r"\s*[；;]\s*")


def carrying(carry: str, name: str) -> str:
    """Add \"holding <name>\" to a node's carry text, in the story language."""
    item = f"手里拿着{name}" if config.LANGUAGE == "zh" else f"holding {name}"
    sep = "；" if config.LANGUAGE == "zh" else "; "
    return (f"{carry}{sep}{item}" if carry else item)[: 200 if config.LANGUAGE == "zh" else 500]


def drop_carried(carry: str, name: str) -> str:
    """Remove every part of a carry text that mentions name."""
    sep = "；" if config.LANGUAGE == "zh" else "; "
    return sep.join(part for part in CARRY_SEP.split(carry) if part and name not in part)


def _say(zh: str, en: str) -> str:
    """A progress message in the story language (shown live in the UI)."""
    return zh if config.LANGUAGE == "zh" else en


# Boilerplate wrapped around detailed_description. `_ending_of` strips these, so it must use
# the same literals; writing them in two places breaks silently when one changes.
_STATE_PREFIX = "Dynamic world state — preserve these visible facts exactly: "

_STYLE_CLAUSE = re.compile(r"^\s*(?:\[Shot 1\])?\s*Visual style:.*?\.\s*", re.I)


def _no_event(_event: dict[str, Any]) -> None:
    """Anything beyond progress (what the director said, which images it drew) is not sent out by default."""


def _noop(_stage: str, _detail: str = "") -> None:
    pass


def _pretty(obj: Any) -> str:
    import json

    return json.dumps(obj, ensure_ascii=False, indent=2)


class _Clock:
    """Per-stage wall clock, so the inspector can say where the time actually went."""

    def __init__(self) -> None:
        self.t0 = time.time()
        self.marks: dict[str, float] = {}

    def stage(self, name: str):
        clock = self

        class _Ctx:
            def __enter__(self):
                self._s = time.time()
                return self

            def __exit__(self, *_):
                elapsed = time.time() - self._s
                clock.marks[name] = round(clock.marks.get(name, 0.0) + elapsed, 2)
                return False

        return _Ctx()

    def total(self) -> float:
        return round(time.time() - self.t0, 2)


def _in_story_language(text: str) -> bool:
    """Whether this text is in the story language (agent.language), so a reader can be shown it.

    The test is **most word characters are Chinese**, not "contains Chinese": the real failing
    case was 「晨雾旅人 wearing canvas jacket, holding train metal handrail」, a Chinese
    name followed by a whole English sentence.
    """
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return True  # no letters, only symbols and digits: fine
    zh = sum(1 for c in letters if "\u4e00" <= c <= "\u9fff")
    if config.LANGUAGE != "zh":
        # English stories: the text is presentable when it is mostly Latin script.
        return (len(letters) - zh) * 2 >= len(letters)
    return zh * 2 >= len(letters)


def ffmpeg_exe() -> str | None:
    """Where ffmpeg is: the system one, else the static binary of the optional imageio-ffmpeg."""
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None
