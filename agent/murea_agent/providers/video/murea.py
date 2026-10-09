"""Your own Murea video model, served by SGLang (inference/ in this repository).

All Murea servers expose POST /v1/videos -> poll GET /v1/videos/{id} ->
GET /v1/videos/{id}/content; the pipeline reported at /v1/models decides how requests are written.

Murea (the full model, not released yet): `task: ref2va` with image (and voice) `conditions`,
4-15 s, and latent continuation: `return_continuation` makes a job continuable and its
`continuation_id` is passed to the next one, together with the references. Builds without
continuation simply never return an id.

Murea-Fast (LTX2 pipeline, the released model in inference/): 1280x704, 24 fps, 8 steps, CFG 1,
8n+1 frames up to 20 s, a continuation's prefix included. Reference generation is sent up to five
images (three validated, five tested end to end); continuation takes the last 17 frames of the
previous clip instead, cannot carry images, and returns those 17 frames again, which are cut off
here.

Only `MUREA_URL` needs to be set. Nothing is passed by path, so the server may run on this
machine, on another one or behind an SSH tunnel: images travel as data URLs and the
continuation prefix as an upload.
"""

from __future__ import annotations

import base64
import hashlib
import io
import random
import re
import subprocess
import tempfile
import threading
import time
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import httpx

from murea_agent import config, reqlog
from murea_agent.providers import imagegen
from murea_agent.providers.video.base import (
    Generation,
    StatusCallback,
    VideoError,
    VideoProvider,
    clip_seconds,
    reference_bytes,
    save_clip,
)
from murea_agent.util import ffmpeg_exe

# The model's fixed contract: 1280x704 at 24 fps, 8 steps, CFG 1, a 17-frame continuation prefix.
# Clip length is free up to the model's 20 s positional range, prefix included. Measured on 4 H20s,
# same shot: 5/10/15/20 s took 17/35/56/81 s with identity and scene held (one H20 only at 5 s).
FPS = 24
WIDTH, HEIGHT = 1280, 704
MAX_FRAMES = 20 * FPS + 1  # the whole timeline, a continuation's prefix included
PREFIX_FRAMES = 17
_FULL_MAX_SECONDS = 15  # the full Murea model takes 4-15 s
# A Murea-Fast service on one GPU has only been validated at 5 s clips (121 frames); /v1/models
# reports the GPU count, and such a service gets no longer clips.
_SINGLE_GPU_MAX_SECONDS = 5
_MAX_REFERENCES = 5
# Murea-Fast's text encoder reads 1024 Gemma tokens and silently drops the rest, which is the end of
# the prompt: its sound direction. Real prompts take 4.3-5 characters per token, so 4200 is under
# 1000 tokens.
FAST_MAX_PROMPT_CHARS = 4200
_CONTRACT = {"fps": FPS, "num_inference_steps": 8, "guidance_scale": 1.0, "width": WIDTH, "height": HEIGHT}


def _frames(n: int) -> int:
    """The nearest valid frame count (8n+1, at least 9)."""
    return max(9, 8 * round((n - 1) / 8) + 1)


def _new_frames(duration: float | None, prefix_frames: int = 0) -> int:
    """New frames for the requested length; with a prefix the whole timeline stays in MAX_FRAMES."""
    limit = MAX_FRAMES - max(0, prefix_frames - 1)
    return min(_frames(clip_seconds(duration) * FPS), limit)


def _client() -> httpx.Client:
    headers = {"Authorization": f"Bearer {config.MUREA_API_KEY}"} if config.MUREA_API_KEY else {}
    return httpx.Client(base_url=config.MUREA_URL, headers=headers, timeout=httpx.Timeout(120.0, read=300.0))


def _data_uri(raw: bytes, mime: str) -> str:
    return f"data:{mime};base64," + base64.b64encode(raw).decode()


def _reference(ref: Path | bytes, *, fit: bool = False) -> str:
    raw = reference_bytes(ref)
    if fit:
        raw = _fit_reference(raw)
    return _data_uri(raw, imagegen.sniff(raw)[1])


# ── Reference size (Murea-Fast) ──────────────────────────────────────────────
# Murea-Fast encodes each reference at its own pixel size, (W/32)·(H/32) tokens, and every DiT step
# attends over them together with the clip (14,080 tokens for 5 s). Measured on four H20s, 5 s,
# two characters and a place: 1.8 MP references (what the image model draws) 18.5 s, 2048² 26.5 s,
# one 12 MP phone photo among them 29.9 s, three 12 MP photos 58.5 s. Larger references did not
# look more like the person; at 768² the face looked less like it. So the copy sent is
#   - at most _REF_MAX_PIXELS, downscaled only (the image model's own 1328² rounds to 1344²);
#   - then cropped to the figure when it stands on a plain light backdrop, as drawn portraits
#     do: 8-12% faster with the same look, and the figure keeps its resolution.
# With both, the three 12 MP photos took 17.4 s. Files on disk stay as they are; the full Murea
# model gets the references unchanged.
_REF_MAX_PIXELS = 1344 * 1344
_EXIF_ORIENTATION = 0x0112
_fitted: dict[str, bytes] = {}
_fitted_lock = threading.Lock()


def _fit_reference(raw: bytes) -> bytes:
    """The reference as Murea-Fast should get it; the original bytes when nothing changes."""
    key = hashlib.sha1(raw).hexdigest()
    with _fitted_lock:
        hit = _fitted.get(key)
    if hit is not None:
        return hit
    out = raw
    try:
        from PIL import Image, ImageOps

        im = Image.open(io.BytesIO(raw))
        im.load()
        # A phone photo keeps its rotation in EXIF, which re-encoding drops: apply it before
        # measuring, resizing or cropping, and re-encode even if nothing else changes.
        turned = im.getexif().get(_EXIF_ORIENTATION, 1) != 1
        im = ImageOps.exif_transpose(im)
        if im.mode in ("RGBA", "LA", "P"):
            rgba = im.convert("RGBA")
            im = Image.new("RGB", rgba.size, (255, 255, 255))
            im.paste(rgba, mask=rgba.split()[-1])
        else:
            im = im.convert("RGB")
        fitted = im
        area = im.width * im.height
        if area > _REF_MAX_PIXELS:
            scale = (_REF_MAX_PIXELS / area) ** 0.5
            fitted = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))), Image.LANCZOS)
        box = _figure_box(fitted)
        if box is not None:
            fitted = fitted.crop(box)
        if fitted is not im or turned:
            buf = io.BytesIO()
            fitted.save(buf, "JPEG", quality=92, optimize=True)
            out = buf.getvalue()
    except Exception:  # noqa: BLE001 -- an unreadable image goes as it is; the server reports it
        out = raw
    with _fitted_lock:
        _fitted[key] = out
        if len(_fitted) > 64:
            _fitted.pop(next(iter(_fitted)))
    return out


def _figure_box(im) -> tuple[int, int, int, int] | None:
    """The box around a figure on a plain light backdrop (a drawn character or object portrait).

    A backdrop is plain when the top, left and right edges are one light colour; a photo or a
    place has something at its edges and is never cropped. The bottom may hold feet, a cut-off
    body or a floor shadow. Columns and rows count as the figure when more than 1% of their
    pixels differ clearly from the backdrop; a margin is kept around it.
    """
    from PIL import Image, ImageChops, ImageFilter

    small = im.convert("L")
    small.thumbnail((512, 512))
    small = small.filter(ImageFilter.GaussianBlur(1))
    w, h = small.size
    if w < 64 or h < 64:
        return None
    edge = 4
    strips = [small.crop((0, 0, w, edge)), small.crop((0, 0, edge, h)), small.crop((w - edge, 0, w, h))]
    histogram = [sum(counts) for counts in zip(*(strip.histogram() for strip in strips))]
    half, seen, backdrop = sum(histogram) / 2, 0, 255
    for value, count in enumerate(histogram):
        seen += count
        if seen >= half:
            backdrop = value
            break
    if backdrop < 200:
        return None
    figure = ImageChops.difference(small, Image.new("L", small.size, backdrop)).point(
        lambda p: 255 if p > 28 else 0
    )
    for strip in ((0, 0, w, edge), (0, 0, edge, h), (w - edge, 0, w, h)):
        part = figure.crop(strip)
        if part.width * part.height - part.histogram()[0] > 0.02 * part.width * part.height:
            return None
    columns = figure.resize((w, 1), Image.BOX).tobytes()
    rows = figure.resize((1, h), Image.BOX).tobytes()
    xs = [x for x, v in enumerate(columns) if v > 2.55]
    ys = [y for y, v in enumerate(rows) if v > 2.55]
    if not xs or not ys:
        return None
    left, right, top, bottom = xs[0], xs[-1] + 1, ys[0], ys[-1] + 1
    mx, my = int((right - left) * 0.06) + 2, int((bottom - top) * 0.06) + 2
    left, top, right, bottom = max(0, left - mx), max(0, top - my), min(w, right + mx), min(h, bottom + my)
    if (right - left) * (bottom - top) > 0.85 * w * h:
        return None  # nothing worth cropping
    sx, sy = im.width / w, im.height / h
    return (
        int(left * sx),
        int(top * sy),
        min(im.width, round(right * sx)),
        min(im.height, round(bottom * sy)),
    )


# ── Clips ────────────────────────────────────────────────────────────────────


def _ffmpeg(*args: str) -> subprocess.CompletedProcess:
    exe = ffmpeg_exe()
    if not exe:
        raise VideoError("Murea continuation needs ffmpeg (install it, or `pip install imageio-ffmpeg`)")
    done = subprocess.run(
        [exe, "-nostdin", "-v", "error", *args], capture_output=True, text=True, timeout=300
    )
    if done.returncode and "-fps_mode" in args and "fps_mode" in done.stderr:
        # ffmpeg before 5.1 (e.g. Ubuntu 22.04's 4.4) calls it -vsync, with the same values.
        args = tuple("-vsync" if a == "-fps_mode" else a for a in args)
        done = subprocess.run(
            [exe, "-nostdin", "-v", "error", *args], capture_output=True, text=True, timeout=300
        )
    return done


def _frame_count(clip: Path) -> int:
    # Decode and read `-progress`: the only form that reports frames on every ffmpeg build
    # (stream copy prints nothing on some).
    out = _ffmpeg("-progress", "pipe:1", "-nostats", "-i", str(clip), "-map", "0:v:0", "-f", "null", "-")
    found = re.findall(r"^frame=(\d+)", out.stdout, re.M)
    if not found:
        raise VideoError(f"Could not count the frames of {clip.name}")
    return int(found[-1])


def _tail(parent: Path, frames: int) -> bytes:
    """The last `frames` frames of the parent clip at 24 fps, near-losslessly encoded, without audio.

    CRF 4, not lossless: a third of the size to upload (about 1 MB for 17 frames at 1280x704) at
    57 dB PSNR against lossless, far below the compression noise the parent clip already has.
    """
    total = _frame_count(parent)
    if total < frames:
        raise VideoError(f"{parent.name} has {total} frames, fewer than the {frames}-frame prefix")
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "prefix.mp4"
        done = _ffmpeg(
            "-i", str(parent),
            "-vf", f"trim=start_frame={total - frames},setpts=PTS-STARTPTS",
            "-frames:v", str(frames), "-fps_mode", "passthrough", "-an",
            "-c:v", "libx264", "-crf", "4", "-pix_fmt", "yuv420p", str(out),
        )  # fmt: skip
        if done.returncode or not out.exists():
            raise VideoError(f"Could not cut the continuation prefix: {done.stderr[:200]}")
        return out.read_bytes()


def _drop_prefix(content: bytes, frames: int) -> bytes:
    """Remove the repeated prefix from a continuation (picture and the matching audio).

    The output length is set exactly (frame count, and the audio cut to the same span) rather
    than left to -shortest or frame-rate padding, which differ between ffmpeg versions.
    """
    with tempfile.TemporaryDirectory() as tmp:
        src, out = Path(tmp) / "in.mp4", Path(tmp) / "out.mp4"
        src.write_bytes(content)
        keep = _frame_count(src) - frames
        if keep <= 0:
            raise VideoError("Murea returned no frames after the continuation prefix")
        start, end = frames / FPS, (frames + keep) / FPS
        done = _ffmpeg(
            "-i", str(src),
            "-vf", f"trim=start_frame={frames},setpts=PTS-STARTPTS",
            "-af", f"atrim=start={start:.6f}:end={end:.6f},asetpts=PTS-STARTPTS",
            "-frames:v", str(keep), "-fps_mode", "passthrough",
            "-c:v", "libx264", "-crf", "16", "-preset", "fast", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k", str(out),
        )  # fmt: skip
        if done.returncode or not out.exists():
            raise VideoError(f"Could not trim the continuation prefix: {done.stderr[:200]}")
        return out.read_bytes()


# ── Requests ─────────────────────────────────────────────────────────────────


def _submit(c: httpx.Client, body: dict[str, Any], prefix: bytes | None) -> dict[str, Any]:
    if prefix is not None:
        # Upload the prefix clip; every other field travels as a form field. In a multipart
        # request the server reads the resolution only from `size` (width and height are dropped).
        form = {k: str(v) for k, v in body.items() if k not in ("width", "height")}
        form["size"] = f"{body['width']}x{body['height']}"
        r = c.post("/v1/videos", data=form, files={"video_reference": ("prefix.mp4", prefix, "video/mp4")})
    else:
        r = c.post("/v1/videos", json=body)
    reqlog.record(
        "video_submit",
        model="murea",
        status=r.status_code,
        continuation=prefix is not None or bool(body.get("continuation_id")),
        refs=len(body.get("reference_image_paths") or body.get("conditions") or []),
        # Images travel inline; log the request without them.
        payload={k: v for k, v in body.items() if k not in ("prompt", "reference_image_paths", "conditions")},
        prompt=body.get("prompt", ""),
    )
    if r.status_code >= 400:
        raise VideoError(f"Murea submit failed {r.status_code}: {r.text[:300]}")
    job = r.json()
    if not job.get("id"):
        raise VideoError(f"Murea returned no job id: {r.text[:200]}")
    return job


def _wait(
    c: httpx.Client, job: dict[str, Any], started: float, timeout: float, poll_interval: float, on_status
):
    job_id = job["id"]
    while True:
        status = str(job.get("status") or "")
        if on_status:
            # SGLang reports only queued -> completed. Murea Agent sends one video at a time, so a
            # submitted job is almost always rendering; say so instead of "queued behind others".
            on_status({"status": status, "started_at": started})
        if status == "completed":
            return job
        if status == "failed":
            raise VideoError(f"Murea job failed: {(job.get('error') or {}).get('message') or job}")
        if time.time() - started > timeout:
            raise VideoError(f"Murea job {job_id} timed out")
        time.sleep(poll_interval)
        r = c.get(f"/v1/videos/{job_id}")
        if r.status_code >= 400:
            raise VideoError(f"Murea status failed {r.status_code}: {r.text[:200]}")
        job = r.json()


def _parent_clip(continue_from: str | None) -> Path | None:
    if not continue_from:
        return None
    path = config.CLIPS / f"{Path(continue_from).name}.mp4"
    return path if path.exists() else None


def _generate_fast(
    prompt: str,
    refs: Sequence[Path | bytes] = (),
    *,
    continue_from: str | None = None,
    require_continuation: bool = False,
    continuation_fallback_prompt: str | None = None,
    duration: float | None = None,
    seed: int | None = None,
    out_dir: Path | None = None,
    on_status: StatusCallback | None = None,
    poll_interval: float = 1.0,
    timeout: float = 1800.0,
    **_unsupported: Any,
) -> Generation:
    """One Murea-Fast beat. With `continue_from` (a previous clip id) it continues that clip;
    `prompt` is then the reference-free continuation prompt and, if the parent clip is gone,
    `continuation_fallback_prompt` is filmed from the references instead."""
    prefix_frames = PREFIX_FRAMES
    parent = _parent_clip(continue_from)
    new = _new_frames(duration, prefix_frames if parent is not None else 0)
    if continue_from and parent is None:
        if require_continuation and not continuation_fallback_prompt:
            raise VideoError(f"The previous clip {continue_from!r} is gone; cannot continue it")
        reqlog.record("continuation_fallback", ok=True, reason="parent_clip_missing", model="murea")
        prompt = continuation_fallback_prompt or prompt
    body: dict[str, Any] = {
        "prompt": prompt,
        "seed": seed if seed is not None else random.randrange(2**31),
        **_CONTRACT,
    }
    prefix = None
    if parent is not None:
        prefix = _tail(parent, prefix_frames)
        body.update(num_frames=prefix_frames + new - 1, continuation_prefix_frames=prefix_frames)
    else:
        if not refs:
            raise VideoError("Murea needs reference images or a previous clip to continue")
        body.update(num_frames=new, reference_image_paths=[_reference(r, fit=True) for r in refs])

    started = time.time()
    with _client() as c:
        job = _wait(c, _submit(c, body, prefix), started, timeout, poll_interval, on_status)
        content = c.get(f"/v1/videos/{job['id']}/content", timeout=300.0)
        if content.status_code >= 400:
            raise VideoError(f"Murea download failed {content.status_code}")
    data = content.content if prefix is None else _drop_prefix(content.content, prefix_frames)
    video_id = f"mu-{uuid.uuid4().hex[:12]}"
    seconds = (body["num_frames"] - (prefix_frames if prefix else 0)) / FPS
    return Generation(
        video_id=video_id,
        video_path=save_clip(data, video_id, out_dir),
        duration=seconds,
        elapsed=time.time() - started,
        params={
            "model": "murea",
            "num_frames": body["num_frames"],
            # Read by frame grabbing (the critic's and the director's view of the clip) and regenerate.
            "duration_seconds": seconds,
            "size": f"{WIDTH}x{HEIGHT}",
            "seed": body["seed"],
            "inference_time_s": job.get("inference_time_s"),
            "reference_images_sent": 0 if prefix else len(refs),
            "continuation": prefix is not None,
            "prompt_mode": "latent_continuation" if prefix else "reference_generation",
        },
        continued=prefix is not None,
        # The next beat continues from this clip's file.
        artifact_id=video_id,
        submitted_prompt=prompt,
    )


# ── Murea ───────────────────────────────────────────────────────────────────────

_MAX_VOICES = 3
# Servers whose build returned no continuation id: later beats are filmed from references.
_no_continuation: set[str] = set()


def _about_continuation(error: Exception) -> bool:
    return "continu" in str(error).casefold()


def _generate_murea(
    prompt: str,
    refs: Sequence[Path | bytes] = (),
    *,
    ref_audios: Sequence[Path] = (),
    continue_from: str | None = None,
    send_refs_on_continuation: bool | None = None,
    require_continuation: bool = False,
    allow_continuation_fallback: bool = False,
    continuation_fallback_prompt: str | None = None,
    duration: float | None = None,
    aspect_ratio: str = "16:9",
    seed: int | None = None,
    steps: int | None = None,
    out_dir: Path | None = None,
    on_status: StatusCallback | None = None,
    poll_interval: float = 1.0,
    timeout: float = 1800.0,
    **_unsupported: Any,
) -> Generation:
    """One Murea beat; continues `continue_from` (a previous job's continuation id) when given."""
    secs = min(clip_seconds(duration), _FULL_MAX_SECONDS)
    images = [{"type": "image", "uri": _reference(r), "role": "reference"} for r in refs]
    voices = [
        {"type": "audio", "uri": _data_uri(Path(a).read_bytes(), "audio/wav"), "role": "reference"}
        for a in list(ref_audios)[:_MAX_VOICES]
    ]

    def body(continuing: bool, text: str) -> dict[str, Any]:
        keep_refs = send_refs_on_continuation is None or bool(send_refs_on_continuation)
        out: dict[str, Any] = {
            "task": "ref2va",
            "prompt": text,
            "conditions": (images if not continuing or keep_refs else []) + voices,
            "target": {"aspect_ratio": aspect_ratio, "duration_seconds": secs},
            # Make every clip continuable, so the next beat can carry on from its latent.
            "return_continuation": True,
        }
        if continuing:
            out["continuation_id"] = continue_from
        if seed is not None:
            out["seed"] = seed
        if steps is not None:
            out["num_inference_steps"] = steps
        return out

    continuing = bool(continue_from) and config.MUREA_URL not in _no_continuation
    if require_continuation and not continuing:
        raise VideoError("This beat requires latent continuation, which this Murea server does not offer")
    started = time.time()
    with _client() as c:

        def run(request: dict[str, Any]) -> dict[str, Any]:
            return _wait(c, _submit(c, request, None), started, timeout, poll_interval, on_status)

        try:
            job = run(body(continuing, prompt))
        except VideoError as exc:
            # The parent latent can expire. Film the beat from the references instead, unless a
            # co-op round requires continuity and did not allow the fallback.
            if not (continuing and _about_continuation(exc)):
                raise
            if require_continuation and not allow_continuation_fallback:
                raise
            reqlog.record("continuation_fallback", ok=True, reason="parent_latent_unavailable", model="murea")
            continuing, prompt = False, continuation_fallback_prompt or prompt
            job = run(body(False, prompt))
        content = c.get(f"/v1/videos/{job['id']}/content", timeout=300.0)
        if content.status_code >= 400:
            raise VideoError(f"Murea download failed {content.status_code}")
    artifact = job.get("continuation_id")
    if not artifact:
        _no_continuation.add(config.MUREA_URL)
    video_id = f"mu-{uuid.uuid4().hex[:12]}"
    return Generation(
        video_id=video_id,
        video_path=save_clip(content.content, video_id, out_dir),
        duration=float(job.get("seconds") or secs),
        elapsed=time.time() - started,
        params={
            "model": "murea",
            "backend": "murea",
            "duration_seconds": secs,
            "size": job.get("size"),
            "inference_time_s": job.get("inference_time_s"),
            "reference_images_sent": len(images)
            if (not continuing or send_refs_on_continuation is not False)
            else 0,
            "continuation": continuing,
            "prompt_mode": "latent_continuation" if continuing else "reference_generation",
        },
        continued=continuing,
        artifact_id=artifact,
        submitted_prompt=prompt,
    )


# ── Which model the server runs ──────────────────────────────────────────────

_backends: dict[str, str] = {}
# URL -> GPUs the service reports (`num_gpus` in /v1/models), detected with the backend.
_gpus: dict[str, int] = {}
# URL -> when it was last unreachable. The director asks for the clip length several times per
# beat; without this a server that does not answer cost a 10 s timeout on every one of them.
_unreachable: dict[str, float] = {}
_RETRY_UNREACHABLE_AFTER = 30.0


def backend() -> str:
    """ "murea" or "fast" (Murea-Fast, the LTX2 pipeline), from /v1/models (cached per URL)."""
    url = config.MUREA_URL
    if url in _backends:
        return _backends[url]
    if time.monotonic() - _unreachable.get(url, -_RETRY_UNREACHABLE_AFTER) < _RETRY_UNREACHABLE_AFTER:
        return "fast"
    try:
        with _client() as c:
            r = c.get("/v1/models", timeout=10.0)
            # A 401 (wrong MUREA_API_KEY) or a 5xx says nothing about the model; caching
            # its empty list would pin the full-model format until a restart.
            r.raise_for_status()
            models = r.json().get("data") or []
    except (httpx.HTTPError, ValueError, AttributeError):
        # Unreachable or failing: not cached for good, so the real model is detected once it answers.
        _unreachable[url] = time.monotonic()
        return "fast"
    names = " ".join(f"{m.get('pipeline_class', '')} {m.get('pipeline_name', '')}" for m in models)
    gpus = next((m.get("num_gpus") for m in models if isinstance(m, dict)), None)
    if type(gpus) is int and gpus > 0:
        _gpus[url] = gpus
    _backends[url] = "fast" if "LTX2" in names else "murea"
    return _backends[url]


def gpu_count() -> int | None:
    """GPUs the service runs on, as /v1/models reports it; None while unknown."""
    backend()
    return _gpus.get(config.MUREA_URL)


def generate(prompt: str, refs: Sequence[Path | bytes] = (), **options: Any) -> Generation:
    if not config.MUREA_URL:
        raise VideoError("MUREA_URL is not set")
    if len(refs) > _MAX_REFERENCES:
        raise VideoError(f"Murea takes at most {_MAX_REFERENCES} reference images, got {len(refs)}")
    if backend() == "murea":
        return _generate_murea(prompt, refs, **options)
    return _generate_fast(prompt, refs, **options)


class Murea(VideoProvider):
    id = "murea"
    label = "Murea"
    note = "Your own Murea model on SGLang; up to 5 references; continues from the previous clip"
    max_references = _MAX_REFERENCES

    # Capabilities depend on the model the server runs.
    @property
    def references_on_continuation(self) -> bool:  # type: ignore[override]
        return backend() == "murea"

    @property
    def voice_references(self) -> bool:  # type: ignore[override]
        return backend() == "murea"

    @property
    def max_prompt_chars(self) -> int:  # type: ignore[override]
        if backend() == "murea":
            return config.VIDEO_MAX_PROMPT_CHARS
        return min(config.VIDEO_MAX_PROMPT_CHARS, FAST_MAX_PROMPT_CHARS)

    @property
    def max_seconds(self) -> int:  # type: ignore[override]
        # Murea-Fast's positional range is 20 s, but on one GPU only 5 s is validated; the full
        # model takes 4-15 s.
        if backend() == "murea":
            return _FULL_MAX_SECONDS
        if gpu_count() == 1:
            return _SINGLE_GPU_MAX_SECONDS
        return (MAX_FRAMES - 1) // FPS

    def is_configured(self) -> bool:
        return bool(config.MUREA_URL)

    def healthy(self) -> bool:
        try:
            with _client() as c:
                return c.get("/health", timeout=5.0).status_code == 200
        except httpx.HTTPError:
            return False

    def supports_continuation(self) -> bool:
        return backend() != "murea" or config.MUREA_URL not in _no_continuation

    def generate(self, prompt, refs=(), **options):
        return generate(prompt, refs, **options)


PROVIDER = Murea()
