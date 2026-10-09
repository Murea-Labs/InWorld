"""Frame extraction from generated clips."""

from __future__ import annotations

import subprocess
from functools import lru_cache
from pathlib import Path
from murea_agent import config
from murea_agent.util import ffmpeg_exe
from murea_agent.memory.world import Node
from murea_agent.memory.world_media import has_media, media_path


# Three frames per clip: early, middle and the last one. The last frame matters most: it is
# where the world stands now and the next beat continues from it. 512px is enough, and each
# image costs a fixed number of tokens regardless of resolution.
_FRAME_AT = (0.25, 0.6, 0.97)

_FRAME_W = 512


@lru_cache(maxsize=1)
def _grab_frames(clip: Path, node_id: str, duration: float) -> list[str]:
    """Extract a few frames from a fresh clip and save them. Failing here must never break the beat.

    The duration comes from the provider's `duration_seconds` instead of running ffprobe:
    one subprocess less, and one dependency imageio-ffmpeg does not ship.
    """
    exe = ffmpeg_exe()
    if not exe or duration <= 0:
        return []
    out: list[str] = []
    dur = duration
    for i, frac in enumerate(_FRAME_AT):
        fn = f"{node_id}_{i}.jpg"
        try:
            subprocess.run(
                [
                    exe,
                    "-y",
                    "-v",
                    "error",
                    "-ss",
                    f"{max(0.0, dur * frac):.2f}",
                    "-i",
                    str(clip),
                    "-frames:v",
                    "1",
                    "-vf",
                    f"scale={_FRAME_W}:-1",
                    "-q:v",
                    "6",
                    str(config.FRAMES / fn),
                ],
                capture_output=True,
                timeout=20,
                check=True,
            )
        except Exception:  # noqa: BLE001
            continue
        if (config.FRAMES / fn).exists():
            out.append(fn)
    return out


def _frames_for_gm(node: Node | None) -> list[tuple[str, bytes]]:
    """What the previous clip really showed, for the director. **Only the last frame.**

    Sending every extracted frame cost about 0.5 s per call. The last frame loses nothing:
    the director needs to know where the world stands now, which only the last frame decides;
    the earlier frames show what already happened, which its own beat already describes.
    """
    if node is None:
        return []
    for fn in reversed(node.frames):
        if has_media("frames", fn):
            return [("image/jpeg", media_path("frames", fn).read_bytes())]
    return []
