"""Bounded immutable-image cache for co-op's LLM requests."""

from functools import lru_cache
from io import BytesIO
from pathlib import Path
import subprocess
from murea_agent.tools.frames import _grab_frames
from murea_agent.util import ffmpeg_exe

grab_frames = _grab_frames


@lru_cache(maxsize=48)
def _read_image(path, modified, size):
    return Path(path).read_bytes()


def read_image(path):
    st = path.stat()
    # Keep retained bytes bounded even for large uploaded references.
    if st.st_size > 512 * 1024:
        return path.read_bytes()
    return _read_image(str(path), st.st_mtime_ns, st.st_size)


def _white_ratio(clip: Path, at: float) -> float:
    """Return the fraction of near-white pixels in a sampled video frame."""
    exe = ffmpeg_exe()
    if not exe or at < 0:
        return 0.0
    try:
        result = subprocess.run(
            [
                exe,
                "-v",
                "error",
                "-ss",
                f"{at:.2f}",
                "-i",
                str(clip),
                "-frames:v",
                "1",
                "-f",
                "image2pipe",
                "-vcodec",
                "mjpeg",
                "-",
            ],
            capture_output=True,
            timeout=10,
            check=True,
        )
        from PIL import Image

        image = Image.open(BytesIO(result.stdout)).convert("RGB")
        pixels = list(image.resize((96, 54)).getdata())
        return sum(1 for r, g, b in pixels if r > 238 and g > 238 and b > 238) / len(pixels)
    except Exception:  # noqa: BLE001 - cleanup is best effort
        return 0.0


def remove_white_reference_tail(clip: Path, duration: float) -> float:
    """Trim a generated co-op clip if its tail collapses into a white reference plate.

    the video model occasionally treats the repeated character/creature reference upload as a
    late-frame target.  Keep the valid in-world portion and avoid exposing the
    white-background plate to the player; latent continuation metadata is unchanged.
    """
    if duration < 1.0 or not clip.is_file():
        return duration
    tail = _white_ratio(clip, duration - 0.05)
    prior = _white_ratio(clip, max(0.0, duration - 0.65))
    if tail < 0.58 or tail - prior < 0.25:
        return duration
    cut = max(1.0, duration - 0.38)
    tmp = clip.with_suffix(".trim.mp4")
    exe = ffmpeg_exe()
    if not exe:
        return duration
    try:
        subprocess.run(
            [
                exe,
                "-y",
                "-v",
                "error",
                "-i",
                str(clip),
                "-t",
                f"{cut:.3f}",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "18",
                "-c:a",
                "aac",
                str(tmp),
            ],
            capture_output=True,
            timeout=60,
            check=True,
        )
        tmp.replace(clip)
        return cut
    except Exception:  # noqa: BLE001 - never fail a generated turn on cleanup
        tmp.unlink(missing_ok=True)
        return duration
