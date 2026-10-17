"""What every video provider shares: the result type, errors, and media helpers."""

from __future__ import annotations

import io
import subprocess
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from murea_agent import config, reqlog
from murea_agent.util import ffmpeg_exe


class VideoError(RuntimeError):
    pass


class PromptTooLongError(VideoError):
    """The provider rejected a prompt before creating a video job.

    This is a deterministic input-size failure, not a transient submit error. The
    pipeline budgets the final caption before submission. This distinct type
    prevents a rejected request from being retried without its latent handle.
    """


@dataclass
class Generation:
    video_id: str
    video_path: Path
    duration: float
    elapsed: float
    params: dict[str, Any] = field(default_factory=dict)
    continued: bool = False
    # Artifact id to pass as `continue_from` next time. NOT the job id.
    artifact_id: str | None = None
    # Exact caption accepted by the provider (may differ on a fresh fallback
    # after an expired continuation artifact).
    submitted_prompt: str = ""


StatusCallback = Callable[[dict[str, Any]], None]


class VideoProvider:
    """One video generation service.

    `generate` takes the prompt and the reference images in `<Picture N>` order,
    plus keyword options. Options a provider does not support are ignored, so
    the pipeline can pass the same request to any of them; it asks the
    capability flags below before relying on one.
    """

    id: str = ""
    label: str = ""
    note: str = ""
    # Reference images accepted per request.
    max_references: int = 5
    # Accepts locked voice samples (`ref_audios`, `<Audio N>` in the prompt).
    voice_references: bool = False
    # A continuation may also carry the reference images. When it cannot, the pipeline sends a
    # reference-free prompt with the continuation (and the full one as the fallback).
    references_on_continuation: bool = True

    @property
    def max_prompt_chars(self) -> int:
        """Prompt budget in English characters (see `prompt_budget.prompt_size`)."""
        return config.VIDEO_MAX_PROMPT_CHARS

    @property
    def max_seconds(self) -> int:
        """The longest clip one request can return; the director never plans a step past it."""
        return 15

    def is_configured(self) -> bool:
        """The deployment points at this provider (it can be offered as a choice)."""
        raise NotImplementedError

    def ready(self) -> bool:
        """Every credential a request needs is set."""
        return self.is_configured()

    def healthy(self) -> bool:
        """The provider answers right now (cheap; called by /api/health)."""
        return self.ready()

    def supports_continuation(self) -> bool:
        """The next beat can continue from the previous clip's latent."""
        return False

    def continuation_note(self) -> str:
        """Why continuation is unavailable, for the health endpoint ("" when it is fine)."""
        return ""

    def generate(self, prompt: str, refs: Sequence[Path | bytes] = (), **options: Any) -> Generation:
        raise NotImplementedError

    def cancel(self, job_id: str) -> bool:
        """Withdraw a queued or running job. False when the provider cannot."""
        return False


def clip_seconds(duration: float | None) -> int:
    """The requested clip length, clamped to the hard limit (4-20 s); each provider caps it further."""
    return max(4, min(config.VIDEO_HARD_MAX_SECONDS, int(round(float(duration or config.VIDEO_SECONDS)))))


def save_clip(content: bytes, name: str, out_dir: Path | None) -> Path:
    out_dir = out_dir or config.CLIPS
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.mp4"
    path.write_bytes(content)
    faststart(path)
    return path


# ── Recompress reference images before upload ──────────────────────────────
# Measured: a beat sent a 1.4 MB multipart and the POST alone took **20 seconds**, longer than
# the render itself. Reference images come straight from the image model at full quality
# (~670 KB median each).
#
# **The resolution stays**, only recompress: clips are about the same size as the refs, and
# downscaling would lose facial detail. On the 12 largest, q92 cut size to 34% at a median PSNR of
# 40.0 dB, invisible to the eye, with two thirds less upload.
#
# The file on disk is untouched: the UI shows and users download / replace the original; only the
# copy sent out is recompressed.
_UPLOAD_Q = 92
_shrunk: dict[tuple, bytes] = {}
_shrink_lock = threading.Lock()
_SHRINK_MAX = 64


def for_upload(raw: bytes, key: tuple | None = None) -> bytes:
    """Recompress one reference image. Returns the original if it cannot, gains nothing, or anything fails."""
    if len(raw) < 120_000:  # already small, not worth it
        return raw
    if key is not None:
        with _shrink_lock:
            hit = _shrunk.get(key)
        if hit is not None:
            return hit
    try:
        from PIL import Image, ImageOps

        # A phone photo keeps its rotation in EXIF, which re-encoding drops: apply it first.
        im = ImageOps.exif_transpose(Image.open(io.BytesIO(raw)))
        if im.mode in ("RGBA", "LA", "P"):
            # With alpha, put it on white first: transparent areas mean nothing to the video model
            im = im.convert("RGBA")
            bg = Image.new("RGB", im.size, (255, 255, 255))
            bg.paste(im, mask=im.split()[-1])
            im = bg
        else:
            im = im.convert("RGB")
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=_UPLOAD_Q, optimize=True, progressive=True)
        out = buf.getvalue()
    except Exception:  # send the original if recompression fails
        return raw
    if len(out) >= len(raw):
        out = raw
    if key is not None:
        with _shrink_lock:
            _shrunk[key] = out
            if len(_shrunk) > _SHRINK_MAX:
                _shrunk.pop(next(iter(_shrunk)))
    return out


def reference_bytes(ref: Path | bytes) -> bytes:
    """One reference image, recompressed for upload (cached per file version)."""
    if isinstance(ref, (bytes, bytearray)):
        return for_upload(bytes(ref))
    path = Path(ref)
    st = path.stat()
    return for_upload(path.read_bytes(), (str(path), st.st_mtime_ns, st.st_size))


def faststart(path: Path) -> None:
    """Move the moov atom to the start of the file so the browser can play while downloading.

    ⚠️ **Provider mp4s often have moov at the end.** Then the player must **download the whole file** before
    it can decode the first frame; no progressive playback. Users see a finished video still
    spinning, and seeking waits again.

    Remuxing is lossless (`-c copy`, only the index moves), a few hundred ms, negligible next to the
    render. On failure keep the original: it is a speed-up and must never break the beat.
    """
    exe = ffmpeg_exe()
    if not exe:
        return
    tmp = path.with_suffix(".fast.mp4")
    try:
        r = subprocess.run(
            [exe, "-nostdin", "-loglevel", "error", "-y", "-i", str(path)]
            + ["-c", "copy", "-movflags", "+faststart", str(tmp)],
            capture_output=True,
            timeout=120,
        )
        if r.returncode == 0 and tmp.exists() and tmp.stat().st_size > 0:
            tmp.replace(path)
            return
        reqlog.record(
            "faststart_skipped",
            ok=False,
            clip=path.name,
            error=(r.stderr or b"")[:160].decode("utf-8", "replace"),
        )
    except Exception as exc:
        reqlog.record("faststart_skipped", ok=False, clip=path.name, error=str(exc)[:160])
    finally:
        tmp.unlink(missing_ok=True)
