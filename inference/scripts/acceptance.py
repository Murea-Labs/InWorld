#!/usr/bin/env python3
"""End-to-end acceptance run against a live Murea-Fast service.

Runs one multi-reference generation twice (first and repeated request), then a
clean-latent continuation from the tail of the generated video, validating
every output. Run it on the service host: reference paths are read server-side and
the timing files are written by the service, so its launcher configuration must set
``"allow_server_paths": true``.

Case directory layout: ``prompt.txt``, ordered ``ref_00.jpg``/``ref_01.png``...,
and optionally ``continuation_prompt.txt``.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from murea_fast.client import generate, prepare_tail, write_json, validate_output, media_tool


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", type=Path, required=True, help="directory with prompt.txt and ordered ref_* images")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--base-url", default="http://127.0.0.1:30260")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=704)
    p.add_argument("--resume", action="store_true", help="re-validate finished outputs and continue with the remaining steps")
    args = p.parse_args()
    output = args.output.resolve()
    media_tool("ffmpeg")
    media_tool("ffprobe")
    output.mkdir(parents=True, exist_ok=args.resume)

    def run(payload, directory):
        if args.resume and directory.exists():
            previous = json.loads((directory / "request.json").read_text())
            if any(previous.get(k) != v for k, v in payload.items()):
                raise ValueError(f"Cannot resume with changed inputs: {directory}")
            return validate_output(directory, payload)
        return generate(args.base_url, payload, directory)
    refs = sorted(p for p in args.case.resolve().glob("ref_*")
                  if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"))
    if not refs:
        raise ValueError("No ordered reference images")
    prompt = (args.case / "prompt.txt").read_text().strip()
    continuation = args.case / "continuation_prompt.txt"
    continuation_prompt = (continuation.read_text().strip() if continuation.is_file() else
                           "The scene continues naturally from the previous shot. "
                           "The camera stays steady. The ambient sound continues.")
    versions = {}
    for name in ("torch", "torchvision", "torchaudio", "transformers", "diffusers", "safetensors", "triton", "cache-dit", "sglang-kernel"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    write_json(output / "environment.json", versions)
    payload = {"prompt": prompt, "reference_image_paths": list(map(str, refs)),
               "width": args.width, "height": args.height, "num_frames": 121,
               "seed": 42, "perf_dump_path": str(output / "generation-perf.json")}
    results = {"generation_first": run(payload, output / "generation-first")}
    payload["perf_dump_path"] = str(output / "generation-warm-perf.json")
    results["generation_warm"] = run(payload, output / "generation-warm")
    parent = output / "generation-warm/video.mp4"
    prefix = output / "prefix.mp4"
    if not (args.resume and prefix.exists()):
        prepare_tail(parent, prefix, 17)
    payload = {"prompt": continuation_prompt,
               "video_path": str(prefix), "continuation_prefix_frames": 17,
               "width": args.width, "height": args.height, "num_frames": 137,
               "seed": 42, "perf_dump_path": str(output / "continuation-perf.json")}
    results["continuation"] = run(payload, output / "continuation")
    write_json(output / "summary.json", results)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
