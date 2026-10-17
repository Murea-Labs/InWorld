#!/usr/bin/env python3
"""Materialize the Murea-Fast connector overlay.

The SFT checkpoint is a transformer-only file from SGLang's point of view, but
it also contains the 258 trained connector tensors.  This tool merges those
tensors into the base Diffusers connector shards while preserving the four
base-only text projection tensors.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from safetensors import safe_open
from safetensors.torch import save_file


CHECKPOINT_SIZE = 42_018_192_440
CHECKPOINT_SHA256 = "1a2c0d8e23032994b52f2c67e14c930d3be8224f321c51353f0a2954ccbb9385"
CONNECTOR_PREFIXES = (
    "model.diffusion_model.audio_embeddings_connector.",
    "model.diffusion_model.video_embeddings_connector.",
)
EXPECTED_PRESERVED = {
    "audio_text_proj_in.bias",
    "audio_text_proj_in.weight",
    "video_text_proj_in.bias",
    "video_text_proj_in.weight",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}


def fingerprint(path: Path) -> dict[str, int]:
    info = path.stat()
    return {"size": info.st_size, "mtime_ns": info.st_mtime_ns,
            "ctime_ns": info.st_ctime_ns, "inode": info.st_ino}


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def remap_key(key: str) -> str:
    for prefix, replacement in (
        (CONNECTOR_PREFIXES[0], "audio_connector."),
        (CONNECTOR_PREFIXES[1], "video_connector."),
    ):
        if key.startswith(prefix):
            key = replacement + key[len(prefix) :]
            break
    else:
        raise ValueError(f"Not an LTX connector key: {key}")
    return (
        key.replace("transformer_1d_blocks.", "transformer_blocks.")
        .replace(".attn1.k_norm.", ".attn1.norm_k.")
        .replace(".attn1.q_norm.", ".attn1.norm_q.")
    )


def checkpoint_connector_inventory(
    checkpoint: Path,
) -> tuple[dict[str, str], dict[str, tuple[int, ...]], dict[str, str]]:
    mapped: dict[str, str] = {}
    shapes: dict[str, tuple[int, ...]] = {}
    dtypes: dict[str, str] = {}
    with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
        for source_key in handle.keys():
            if not source_key.startswith(CONNECTOR_PREFIXES):
                continue
            target_key = remap_key(source_key)
            if target_key in mapped.values():
                raise ValueError(f"Duplicate mapped connector key: {target_key}")
            mapped[source_key] = target_key
            shapes[target_key] = tuple(handle.get_slice(source_key).get_shape())
            dtypes[target_key] = str(handle.get_slice(source_key).get_dtype())
    return mapped, shapes, dtypes


def base_inventory(
    base_dir: Path,
) -> tuple[dict[str, str], dict[str, tuple[int, ...]], dict[str, str]]:
    index = load_json(base_dir / "diffusion_pytorch_model.safetensors.index.json")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError(f"Invalid connector index: {base_dir}")
    shapes: dict[str, tuple[int, ...]] = {}
    dtypes: dict[str, str] = {}
    for shard_name in sorted(set(weight_map.values())):
        with safe_open(str(base_dir / shard_name), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                shapes[key] = tuple(handle.get_slice(key).get_shape())
                dtypes[key] = str(handle.get_slice(key).get_dtype())
    return weight_map, shapes, dtypes


def valid_existing(
    output_dir: Path,
    checkpoint: Path,
    checkpoint_sha: str,
    base_dir: Path,
    base_index_sha256: str,
    *,
    verify_hashes: bool,
) -> bool:
    manifest_path = output_dir / "connector-overlay-manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = load_json(manifest_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    base_index_record = next(
        (
            item
            for item in manifest.get("base_files", [])
            if item.get("name") == "diffusion_pytorch_model.safetensors.index.json"
        ),
        {},
    )
    if (
        manifest.get("checkpoint_sha256") != checkpoint_sha
        or manifest.get("checkpoint_path") != str(checkpoint)
        or manifest.get("checkpoint_bytes") != checkpoint.stat().st_size
        or manifest.get("base_connector_path") != str(base_dir)
        or base_index_record.get("sha256") != base_index_sha256
        or manifest.get("overlay_key_count") != 258
    ):
        return False
    for item in manifest.get("output_files", []):
        path = output_dir / item["name"]
        if (
            not path.is_file()
            or path.stat().st_size != item["bytes"]
            or (verify_hashes and sha256(path) != item["sha256"])
        ):
            return False
    return True


def materialize(
    checkpoint: Path,
    base_dir: Path,
    output_dir: Path,
    *,
    expected_checkpoint_sha256: str = CHECKPOINT_SHA256,
    verify_existing_hashes: bool = False,
) -> dict[str, Any]:
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if checkpoint.stat().st_size != CHECKPOINT_SIZE:
        raise ValueError(f"Unexpected checkpoint size: {checkpoint} ({checkpoint.stat().st_size})")
    if not (base_dir / "config.json").is_file():
        raise FileNotFoundError(f"Missing connector config: {base_dir / 'config.json'}")
    base_index_path = base_dir / "diffusion_pytorch_model.safetensors.index.json"
    base_index_sha256 = sha256(base_index_path)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir.parent / f".{output_dir.name}.lock"
    lock = lock_path.open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX)
    if valid_existing(
        output_dir,
        checkpoint,
        expected_checkpoint_sha256,
        base_dir,
        base_index_sha256,
        verify_hashes=verify_existing_hashes,
    ):
        manifest_path = output_dir / "connector-overlay-manifest.json"
        manifest = load_json(manifest_path)
        current = fingerprint(checkpoint)
        if manifest.get("checkpoint_fingerprint") != current:
            if sha256(checkpoint) != expected_checkpoint_sha256:
                raise ValueError("Checkpoint changed since connector materialization")
            manifest["checkpoint_fingerprint"] = current
            temporary_manifest = output_dir / ".connector-overlay-manifest.json.tmp"
            temporary_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
            temporary_manifest.replace(manifest_path)
        return manifest
    if output_dir.exists():
        raise RuntimeError(
            f"Refusing to overwrite incomplete connector overlay: {output_dir}"
        )
    checkpoint_sha = sha256(checkpoint)
    if checkpoint_sha != expected_checkpoint_sha256:
        raise ValueError(
            f"Checkpoint SHA-256 mismatch: expected {expected_checkpoint_sha256}, "
            f"got {checkpoint_sha}"
        )
    index = load_json(base_dir / "diffusion_pytorch_model.safetensors.index.json")
    weight_map, base_shapes, base_dtypes = base_inventory(base_dir)
    mapped, overlay_shapes, overlay_dtypes = checkpoint_connector_inventory(checkpoint)
    if len(mapped) != 258:
        raise ValueError(f"Expected 258 connector tensors, found {len(mapped)}")
    mapped_targets = set(mapped.values())
    if not mapped_targets.issubset(weight_map):
        raise ValueError(
            "Checkpoint connectors not in base component: "
            f"{sorted(mapped_targets - set(weight_map))[:10]}"
        )
    preserved = set(weight_map) - mapped_targets
    if preserved != EXPECTED_PRESERVED:
        raise ValueError(f"Unexpected unoverlaid connector keys: {sorted(preserved)}")
    for key in mapped_targets:
        if overlay_shapes[key] != base_shapes[key] or overlay_dtypes[key] != base_dtypes[key]:
            raise ValueError(
                f"Connector shape/dtype mismatch for {key}: "
                f"checkpoint={overlay_shapes[key]}/{overlay_dtypes[key]} "
                f"base={base_shapes[key]}/{base_dtypes[key]}"
            )
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent))
    try:
        shutil.copy2(base_dir / "config.json", temporary / "config.json")
        shutil.copy2(
            base_dir / "diffusion_pytorch_model.safetensors.index.json",
            temporary / "diffusion_pytorch_model.safetensors.index.json",
        )
        for shard_name in sorted(set(weight_map.values())):
            base_shard = base_dir / shard_name
            tensors: dict[str, Any] = {}
            with safe_open(str(base_shard), framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    tensors[key] = handle.get_tensor(key)
            with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
                for source_key, target_key in mapped.items():
                    if weight_map[target_key] == shard_name:
                        tensors[target_key] = handle.get_tensor(source_key)
            target = temporary / shard_name
            save_file(tensors, str(target))

        base_files = [
            file_record(base_dir / name)
            for name in (
                "config.json",
                "diffusion_pytorch_model.safetensors.index.json",
                *sorted(set(weight_map.values())),
            )
        ]
        output_files = [
            file_record(temporary / item["name"])
            for item in base_files
        ]

        manifest = {
            "schema_version": 1,
            "checkpoint_path": str(checkpoint),
            "checkpoint_bytes": checkpoint.stat().st_size,
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_fingerprint": fingerprint(checkpoint),
            "base_connector_path": str(base_dir),
            "base_files": base_files,
            "overlay_key_count": len(mapped),
            "overlay_keys": sorted(mapped.values()),
            "preserved_base_only_keys": sorted(preserved),
            "output_files": output_files,
        }
        (temporary / "connector-overlay-manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.rename(output_dir)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--base-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-checkpoint-sha256", default=CHECKPOINT_SHA256)
    parser.add_argument("--verify-existing-hashes", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    manifest = materialize(
        args.checkpoint.resolve(),
        args.base_dir.resolve(),
        output_dir,
        expected_checkpoint_sha256=args.expected_checkpoint_sha256,
        verify_existing_hashes=args.verify_existing_hashes,
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "overlay_key_count": manifest["overlay_key_count"],
                "checkpoint_sha256": manifest["checkpoint_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
