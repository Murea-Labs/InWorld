from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Murea-Fast deployment, generation and clean-latent continuation")
    sub = parser.add_subparsers(dest="action", required=True)
    for name in ("prepare", "start", "stop", "status", "command"):
        p = sub.add_parser(name)
        p.add_argument("--config", type=Path, default=Path("configs/single-gpu.json"))
    p = sub.add_parser("verify")
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--sha256", action="store_true")
    p = sub.add_parser("generate")
    p.add_argument("--request", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--base-url", default="http://127.0.0.1:30260")
    p.add_argument("--timeout", type=float, default=1800)
    p = sub.add_parser("continue", help="Run on the server host or a shared filesystem: trim the parent video "
                       "tail, then continue it. The service must run with \"allow_server_paths\": true")
    p.add_argument("--parent", type=Path, required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--prefix-frames", type=int, default=17)
    p.add_argument("--new-frames", type=int, default=120)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--base-url", default="http://127.0.0.1:30260")
    p.add_argument("--timeout", type=float, default=1800)
    args = parser.parse_args()
    if args.action == "verify":
        from .checkpoint import verify
        print(json.dumps(verify(args.checkpoint, args.sha256), indent=2))
    elif args.action == "generate":
        from .client import generate
        print(json.dumps(generate(args.base_url, json.loads(args.request.read_text()),
                                  args.output, args.timeout), indent=2))
    elif args.action == "continue":
        from .client import (complete_request, generate, media_tool, prepare_tail, tail_info,
                             validate_frames, validate_prefix, validate_request, write_json)
        if args.new_frames <= 0 or args.new_frames % 8:
            parser.error("--new-frames must be a positive multiple of 8")
        total = args.prefix_frames + args.new_frames
        try:
            validate_frames(total)
            validate_prefix(args.prefix_frames, total)
        except ValueError as exc:
            parser.error(f"--prefix-frames {args.prefix_frames} + --new-frames {args.new_frames}"
                         f" = {total} frames: {exc}")
        if args.output.exists():
            raise FileExistsError(args.output)
        # Check everything the service would reject before writing anything, so a failed
        # attempt leaves no output directory behind.
        media_tool("ffmpeg")
        info = tail_info(args.parent, args.prefix_frames)
        tail = args.output.resolve() / "prefix.mp4"
        payload = complete_request({
            "prompt": args.prompt, "video_path": str(tail),
            "continuation_prefix_frames": args.prefix_frames,
            "width": info["width"], "height": info["height"],
            "num_frames": total, "seed": args.seed,
            "perf_dump_path": str(args.output.resolve() / "perf.json")})
        validate_request(payload)
        args.output.mkdir(parents=True)
        prepare_tail(args.parent, tail, args.prefix_frames, info)
        write_json(args.output / "parent.json", {"parent": str(args.parent.resolve()), **info})
        print(json.dumps(generate(args.base_url, payload, args.output / "generated", args.timeout), indent=2))
    else:
        from . import service
        c = service.config(args.config)
        if args.action == "command":
            import shlex
            print(shlex.join(service.command(c)))
        else:
            result = getattr(service, "state" if args.action == "status" else args.action)(c)
            if args.action == "status":
                result = {key: result[key] for key in ("running", "pid", "host", "created_at") if key in result}
                result["health_url"] = service.health_url(c)
                result["manifest"] = str(service.runtime(c) / "service.json")
                result["log"] = str(service.runtime(c) / "server.log")
            if result is not None:
                print(json.dumps(result, indent=2))
