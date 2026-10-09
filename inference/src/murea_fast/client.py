from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path


# The model's positional range is 20 s at 24 fps; a continuation's prefix counts toward it.
MAX_FRAMES = 481


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def request_json(method, url, payload=None):
    request = urllib.request.Request(
        url, data=None if payload is None else json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code}: {exc.read().decode(errors='replace')}") from exc


def server_file_fields(payload: dict) -> list[str]:
    """Fields of `payload` that name files on the service host rather than URLs."""
    def local(value) -> bool:
        return isinstance(value, str) and bool(value) and not value.lower().startswith(
            ("data:", "http://", "https://"))

    names = [key for key in ("output_path", "perf_dump_path") if payload.get(key)]
    names += [key for key in ("video_path", "image_path") if local(payload.get(key))]
    if any(local(path) for path in payload.get("reference_image_paths") or []):
        names.append("reference_image_paths")
    return names


def complete_request(payload: dict) -> dict:
    """Fill in the fields fixed by the distilled checkpoint, as `generate` sends them."""
    return {"fps": 24, "num_inference_steps": 8, "guidance_scale": 1.0,
            "generator_device": "cuda", "seed": 42, **payload}


def validate_frames(frames) -> None:
    if type(frames) is not int or frames < 9 or frames % 8 != 1:
        raise ValueError("num_frames must be 8n+1 and >=9")
    if frames > MAX_FRAMES:
        raise ValueError(f"num_frames must be at most {MAX_FRAMES} (20 s, the model's positional range)")


def validate_prefix(prefix, frames: int) -> None:
    if type(prefix) is not int or prefix < 1 or prefix % 8 != 1 or prefix >= frames:
        raise ValueError("prefix must be 8n+1 and smaller than num_frames")


def validate_request(payload: dict) -> None:
    if not isinstance(payload.get("prompt"), str) or not payload["prompt"].strip():
        raise ValueError("prompt is required")
    for key in ("width", "height"):
        value = payload.get(key)
        if type(value) is not int or value <= 0 or value % 32:
            raise ValueError(f"{key} must be a positive multiple of 32")
    frames = payload.get("num_frames")
    validate_frames(frames)
    if payload.get("num_inference_steps", 8) != 8:
        raise ValueError("Murea-Fast requires exactly 8 steps")
    if payload.get("guidance_scale", 1.0) != 1.0 or payload.get("fps", 24) != 24:
        raise ValueError("Murea-Fast contract: guidance_scale=1, fps=24")
    refs = payload.get("reference_image_paths")
    if refs is not None and (not isinstance(refs, list) or not refs or
                             not all(isinstance(p, str) and p for p in refs)):
        raise ValueError("reference_image_paths must be a non-empty ordered list")
    if payload.get("video_path"):
        if refs or payload.get("image_path"):
            raise ValueError("Continuation cannot combine video and reference images")
        validate_prefix(payload.get("continuation_prefix_frames"), frames)


def probe(path: Path) -> dict:
    return json.loads(subprocess.check_output([
        media_tool("ffprobe"), "-v", "error", "-count_frames", "-show_streams", "-show_format",
        "-of", "json", str(path),
    ], text=True, stdin=subprocess.DEVNULL))


def media_tool(name: str) -> str:
    candidate = os.environ.get("MUREA_" + name.upper(), name)
    resolved = shutil.which(candidate)
    if resolved is None:
        raise FileNotFoundError(f"{name} is required; put it on PATH or set MUREA_{name.upper()}")
    return resolved


def generate(base_url: str, payload: dict, output: Path, timeout: float = 1800) -> dict:
    payload = complete_request(payload)
    validate_request(payload)
    media_tool("ffmpeg")
    media_tool("ffprobe")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "request.json", payload)
    started = time.monotonic()
    try:
        submitted = request_json("POST", base_url.rstrip("/") + "/v1/videos", payload)
    except RuntimeError as exc:
        named = server_file_fields(payload)
        if named and str(exc).startswith("HTTP 400"):
            raise RuntimeError(
                f"{exc}\nThis request names files on the service host ({', '.join(named)}). "
                'The service refuses them unless its launcher configuration sets '
                '"allow_server_paths": true; see docs/deployment.md.') from exc
        raise
    write_json(output / "submitted.json", submitted)
    job_url = base_url.rstrip("/") + "/v1/videos/" + submitted["id"]
    deadline = started + timeout
    while True:
        result = request_json("GET", job_url)
        write_json(output / "latest.json", result)
        if result["status"] == "failed":
            raise RuntimeError(f"Generation failed: {result.get('error')}")
        if result["status"] == "completed":
            break
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Job remains on server: {job_url}")
        time.sleep(1)
    completed_seconds = time.monotonic() - started
    media = output / "video.mp4"
    with urllib.request.urlopen(job_url + "/content", timeout=timeout) as response:
        with media.open("wb") as stream:
            shutil.copyfileobj(response, stream)
    record = {"job_id": submitted["id"], "completion_wall_seconds": completed_seconds,
              "download_wall_seconds": time.monotonic() - started - completed_seconds,
              "inference_time_s": result.get("inference_time_s"),
              "peak_memory_mb": result.get("peak_memory_mb"), "bytes": media.stat().st_size}
    write_json(output / "timing.json", record)
    return validate_output(output, payload)


def validate_output(output: Path, payload: dict) -> dict:
    media = output / "video.mp4"
    record = json.loads((output / "timing.json").read_text())
    media_info = probe(media)
    write_json(output / "ffprobe.json", media_info)
    video = next(s for s in media_info["streams"] if s["codec_type"] == "video")
    if (video["width"], video["height"], int(video["nb_read_frames"])) != (
        payload["width"], payload["height"], payload["num_frames"]
    ):
        raise RuntimeError("Output dimensions/frame count differ from request")
    if not any(s["codec_type"] == "audio" for s in media_info["streams"]):
        raise RuntimeError("Output has no audio stream")
    # ffmpeg reads stdin for keyboard commands; without DEVNULL it eats the input of a
    # calling shell loop (`while read ...; do murea-fast generate ...; done`).
    subprocess.run([media_tool("ffmpeg"), "-v", "error", "-xerror", "-i", str(media),
                    "-f", "null", "-"], check=True, stdin=subprocess.DEVNULL)
    record["media_decode_verified"] = True
    write_json(output / "timing.json", record)
    return record


def tail_info(parent: Path, prefix_frames: int) -> dict:
    """Probe the parent video and check that it can supply the prefix. Writes nothing."""
    if type(prefix_frames) is not int or prefix_frames < 1 or prefix_frames % 8 != 1:
        raise ValueError("prefix_frames must be 8n+1")
    info = probe(parent)
    video = next(s for s in info["streams"] if s["codec_type"] == "video")
    from fractions import Fraction
    if Fraction(video["avg_frame_rate"]) != 24:
        raise ValueError("Parent must be 24 fps; resample explicitly before continuation")
    count = int(video["nb_read_frames"])
    if count < prefix_frames:
        raise ValueError("Parent is shorter than requested prefix")
    return {"width": video["width"], "height": video["height"],
            "source_frames": count, "prefix_frames": prefix_frames}


def prepare_tail(parent: Path, target: Path, prefix_frames: int, info: dict | None = None) -> dict:
    """Write the last `prefix_frames` frames of `parent` to `target`; `info` is `tail_info`'s result."""
    if info is None:
        info = tail_info(parent, prefix_frames)
    if info["prefix_frames"] != prefix_frames:
        raise ValueError("tail_info was computed for a different prefix length")
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        media_tool("ffmpeg"), "-v", "error", "-n", "-i", str(parent), "-vf",
        f"trim=start_frame={info['source_frames'] - prefix_frames},setpts=PTS-STARTPTS",
        "-frames:v", str(prefix_frames), "-an", "-c:v", "libx264", "-crf", "0",
        "-pix_fmt", "yuv420p", str(target),
    ], check=True, stdin=subprocess.DEVNULL)
    return info
