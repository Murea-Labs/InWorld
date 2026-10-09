"""Character voices: one locked reference recording per speaking character.

A character's voice sample is sent with every beat in which they speak, the
same way their reference image locks their appearance.

Voices come from two places:

* **Uploads.** Any wav/mp3 a user attaches to a character in the UI; stored
  under `data/voices/`.
* **An optional library** you provide in `data/voices/library/`: an
  `index.json` plus the audio files it lists. Entries with `"gm": true` are
  offered to the agent, which then picks a voice for new characters itself::

      {"voices": [{"id": "old_man_calm", "file": "old_man_calm.wav",
                   "name": "Calm old man", "desc": "slow, warm, gravelly",
                   "gender": "male", "age": "old", "group": "elders", "gm": true}]}

  Use recordings you have the rights to. Without a library the agent leaves
  voices empty and the video model chooses one per clip.
"""

from __future__ import annotations

import json
import re
import subprocess
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any

from murea_agent import config
from murea_agent.util import _say, ffmpeg_exe

LIBRARY = config.VOICES / "library"
INDEX = LIBRARY / "index.json"

# Uploaded voices are converted to: mono, 24 kHz (what OmniVoice and the video model use), at most
# 15 seconds (the video model's reference_audios accept 2–15 s each).
_SR = 24000
_MAX_SECONDS = 15


class VoiceError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def library() -> list[dict[str, Any]]:
    """Library entries. Entries whose file is missing are not listed; a silent entry is worse."""
    try:
        raw = json.loads(INDEX.read_text())
    except Exception:  # noqa: BLE001 -- must run without a library, just without voices to pick
        return []
    out = []
    for e in raw.get("voices") or []:
        fn = str(e.get("file") or "")
        if not fn or not (LIBRARY / fn).exists():
            continue  # in the index but the wav was never synced: treat as missing
        gender = str(e.get("gender") or "")
        age = str(e.get("age") or "")
        accent = str(e.get("accent") or "")
        out.append(
            {
                "id": str(e.get("id") or Path(fn).stem),
                "name": str(e.get("name") or ""),
                "desc": str(e.get("desc") or ""),
                "gender": gender,
                "age": age,
                "accent": accent,
                "group": str(e.get("group") or ""),
                "gm": bool(e.get("gm")),
                # Used by search and by resolve()'s fallback. Split the `少年/少女` age band,
                # or searching for "少女" finds nothing.
                "tags": [
                    t
                    for t in (
                        [gender, accent, *age.split("/"), *(str(c) for c in (e.get("categories") or []))]
                    )
                    if t
                ],
                "seconds": float(e.get("seconds") or 0),
                "ref": f"library/{fn}",
            }
        )
    return out


def gm_pool() -> list[dict[str, Any]]:
    """The curated tier the director picks from. If the library is not installed (wavs not synced), fall back
    to everything; better than nothing.
    """
    lib = library()
    return [e for e in lib if e["gm"]] or lib


def heal(voice: dict[str, Any] | None) -> dict[str, Any] | None:
    """The voice record stored in a world, or None if its file is gone."""
    v = voice or {}
    ref = str(v.get("ref") or "")
    return v if ref and path(ref) is not None else None


def get(voice_id: str) -> dict[str, Any] | None:
    return next((e for e in library() if e["id"] == voice_id), None)


def resolve(spec: str) -> dict[str, Any] | None:
    """The director's string -> a library entry.

    Normally it is a library id. But the model sometimes writes old-style attribute words
    (`男，中年，低音调`), and old worlds store those; then pick the entry with the most overlapping tags.
    None if nothing fits: this character has no voice in this clip, not an error.
    """
    spec = (spec or "").strip()
    if not spec:
        return None
    hit = get(spec)
    if hit:
        return hit
    words = {w for w in re.split(r"[，,、\s]+", spec) if w}
    if not words:
        return None
    # Fall back within the curated tier only. Across the whole library "male, young" would hit
    # customer service and audiobook voices, which suit reading, not characters.
    best, best_score = None, 0
    for e in gm_pool():
        score = len(words & set(e["tags"]))
        if score > best_score:
            best, best_score = e, score
    return best


# Gender and age from the name and description. Only words **stated explicitly**; when unsure, leave
# that band open. A larger candidate set beats a wrong gender.
_GENDER_WORDS = {
    "男": (
        "男",
        "他",
        "先生",
        "大叔",
        "大爷",
        "爷爷",
        "少爷",
        "公子",
        "小伙",
        "哥",
        "父",
        "爸",
        "叔",
        "伯",
        "兄",
        "郎",
        "汉子",
        "书生",
        "和尚",
        "道士",
        "混混",
        "壮汉",
        "老板",
        "掌柜",
        "师傅",
        "侠客",
        "将军",
        "士兵",
        "警察",
        "医生",
        "boy",
        "man",
        "male",
        "he ",
        "his ",
    ),
    "女": (
        "女",
        "她",
        "小姐",
        "阿姨",
        "奶奶",
        "婆婆",
        "姑娘",
        "少女",
        "妹",
        "姐",
        "母",
        "妈",
        "婶",
        "娘",
        "夫人",
        "太太",
        "尼姑",
        "老板娘",
        "girl",
        "woman",
        "female",
        "she ",
        "her ",
    ),
}
_AGE_WORDS = (
    ("儿童", ("儿童", "小孩", "孩子", "小男孩", "小女孩", "娃", "幼")),
    ("少年/少女", ("少年", "少女", "学生", "中学生", "高中生", "小子", "丫头")),
    ("老年", ("老年", "老人", "爷爷", "奶奶", "婆婆", "大爷", "老者", "长者", "苍老")),
    ("中年", ("中年", "大叔", "阿姨", "婶", "伯", "母亲", "父亲", "老板娘")),
    ("青年", ("青年", "年轻", "小伙", "姑娘", "公子", "书生")),
)


def suggest(name: str, desc: str = "") -> dict[str, Any] | None:
    """Pick a voice from the curated tier by name and description. None if nothing fits.

    **Why pick for the user.** The video model generates audio with the video; without a reference
    voice it invents a new voice every clip and the same character sounds like different people. A
    wrong pick is at least **consistently** wrong, visible in the inspector and replaceable in one
    click.

    Pure table lookup, no model call: creating a character must return right away and not wait on
    the network for an editable default. The same name always gets the same voice (hash of the
    name), so recreating a character does not change its voice.
    """
    pool = gm_pool()
    if not pool:
        return None
    text = f"{name} {desc}"
    gender = next((g for g, words in _GENDER_WORDS.items() if any(w in text for w in words)), "")
    age = next((a for a, words in _AGE_WORDS if any(w in text for w in words)), "")

    def narrow(entries: list[dict[str, Any]], key: str, want: str) -> list[dict[str, Any]]:
        if not want:
            return entries
        hit = [e for e in entries if e.get(key) == want]
        return hit or entries  # nobody in this band: do not filter, never filter to empty

    cand = narrow(narrow(pool, "gender", gender), "age", age)
    # Stable pick: the same name always lands on the same entry. A name instead of a random number,
    # so "delete and recreate" and another machine give the same voice.
    import hashlib

    n = int(hashlib.sha1(name.strip().encode()).hexdigest()[:8], 16)
    out = record(cand[n % len(cand)])
    # Mark it auto so the UI says "picked automatically, please check". **Without a recognized
    # gender this pick is a coin flip on gender**; the word list can never be complete. Showing it
    # is better than pretending the pick is right.
    out["source"] = "auto"
    out["guessed"] = not gender  # a slightly wrong age is fine, a wrong gender is jarring
    return out


def record(entry: dict[str, Any]) -> dict[str, Any]:
    """A library entry -> the voice record stored on the subject."""
    return {"id": entry["id"], "name": entry["name"], "ref": entry["ref"], "source": "library"}


NONE: dict[str, Any] = {"id": "", "name": "", "ref": "", "source": ""}


def choose(voice_id: str = "", ref: str = "", name: str = "") -> dict[str, Any] | None:
    """The entry picked by the user / API -> a voice record. None if both are empty (= no voice).

    Every entry point (adding a character, changing a voice, others later) goes through here, so
    "library or upload first" is decided once.
    """
    voice_id, ref = voice_id.strip(), ref.strip()
    if voice_id:
        entry = get(voice_id)
        if entry is None:
            raise VoiceError(f"No voice {voice_id} in the library")
        return record(entry)
    if ref:
        if path(ref) is None:
            raise VoiceError("This uploaded voice is gone; upload it again")
        return {
            "id": "",
            "name": (name.strip() or _say("上传的音色", "Uploaded voice"))[:40],
            "ref": ref,
            "source": "upload",
        }
    return None


def path(ref: str) -> Path | None:
    """Stored file name -> real path. None if missing.

    `library/xxx.wav` is from the library; everything else is an upload or an old generated voice at
    the root of `data/voices/`. ⚠️ The value may come from a world file or a request body; block
    `../`.
    """
    ref = (ref or "").strip().replace("\\", "/")
    if not ref or ".." in ref.split("/"):
        return None
    if ref.startswith("library/"):
        p = LIBRARY / Path(ref).name
    else:
        p = config.VOICES / Path(ref).name
    return p if p.is_file() else None


def label(voice: dict[str, Any] | None) -> str:
    """Display name. Old worlds only have instruct; use that."""
    v = voice or {}
    return str(v.get("name") or v.get("instruct") or "").strip()


# How many entries the library had when prompt_table() ran. **Not** how many it has now: the system
# prompts render the table at import time, and `library()` is lru_cached. If the library is synced
# after startup, the process keeps a prompt saying "no voice library installed" while `library()` is
# full; different numbers mean a restart is needed. Otherwise every new character silently gets no
# voice.
_prompt_pool = 0


def prompt_pool_size() -> int:
    """Number of voices rendered into the current prompt. For /api/health."""
    return _prompt_pool


def prompt_table() -> str:
    """The table inserted into the director prompt. It follows the library, no second copy to edit.

    **Only the curated tier**, not the whole library: all entries would add tens of KB to every
    director call, and with hundreds of options the model picks worse. The whole library is for
    manual changes in the UI.

    Grouped by `group` with the group names written out, because the voice names are marketing
    labels; without a stated purpose the director picks them literally. Groups follow **narrative
    role**.

    The two-space indent is required by the prompt: the table is inside a list item.
    """
    global _prompt_pool
    pool = gm_pool()
    _prompt_pool = len(pool)
    if not pool:
        return "  | | (no voice library installed: always write `voice` as an empty string) | |"
    # Group first, then render. Headings **cannot** be written while scanning: the index is sorted
    # by gender and age, so one group is split into several runs and later runs would lack a
    # heading.
    groups: dict[str, list[dict[str, Any]]] = {}
    for e in pool:
        groups.setdefault(e["group"] or _say("其他", "Other"), []).append(e)
    lines: list[str] = []
    for grp, members in groups.items():
        lines.append(f"  | | **{grp}** | |")
        for e in members:
            extra = f"（{e['accent']}）" if e["accent"] else ""
            lines.append(f"  | `{e['id']}` | {e['name']}{extra} | {e['desc']} |")
    return "\n".join(lines)


# ---------------------------------------------------------------- upload


@lru_cache(maxsize=1)
def _is_wav(raw: bytes) -> bool:
    return raw[:4] == b"RIFF" and raw[8:12] == b"WAVE"


MAX_UPLOAD_BYTES = 40 * 1024 * 1024


def store_upload(raw: bytes, filename: str = "") -> tuple[str, str]:
    """An uploaded audio clip -> file name under `data/voices/` + a display label.

    Always converted to mono 24 kHz wav and cut to 15 s: the video model's `reference_audios` are
    sent as wav (see video.py), and sending an mp3 labelled `audio/wav` would bet on the provider
    sniffing it.
    """
    if not raw:
        raise VoiceError("Empty file")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise VoiceError("File too large; a few seconds of audio is enough")
    fn = f"voice_{uuid.uuid4().hex[:8]}.wav"
    out = config.VOICES / fn
    exe = ffmpeg_exe()
    if exe is None:
        if not _is_wav(raw):
            raise VoiceError("ffmpeg is not available; upload a wav file")
        out.write_bytes(raw)
    else:
        r = subprocess.run(
            [
                exe,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                "pipe:0",
                "-t",
                str(_MAX_SECONDS),
                "-ac",
                "1",
                "-ar",
                str(_SR),
                "-c:a",
                "pcm_s16le",
                "-f",
                "wav",
                str(out),
            ],
            input=raw,
            capture_output=True,
        )
        if r.returncode != 0 or not out.exists() or out.stat().st_size < 1024:
            out.unlink(missing_ok=True)
            raise VoiceError(
                "Could not decode this file; try a wav or mp3"
                + (f" (ffmpeg: {r.stderr.decode(errors='replace')[:120]})" if r.stderr else "")
            )
    stem = Path(filename or "").name.rsplit(".", 1)[0][:24]
    return fn, (_say(f"上传：{stem}", f"Upload: {stem}") if stem else _say("上传的音色", "Uploaded voice"))
