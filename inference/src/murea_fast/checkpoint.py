from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

CHECKPOINT_BYTES = 42_018_192_440
CHECKPOINT_SHA256 = "1a2c0d8e23032994b52f2c67e14c930d3be8224f321c51353f0a2954ccbb9385"
SIGMAS = [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0]


def read_header(path: Path) -> dict:
    with path.open("rb") as stream:
        raw = stream.read(8)
        if len(raw) != 8:
            raise ValueError("Truncated safetensors header")
        length = struct.unpack("<Q", raw)[0]
        if length > 32 * 1024 * 1024 or length > path.stat().st_size - 8:
            raise ValueError("Invalid safetensors header length")
        return json.loads(stream.read(length))


def verify(path: Path, full_hash: bool = False) -> dict:
    path = path.resolve(strict=True)
    if path.stat().st_size != CHECKPOINT_BYTES:
        raise ValueError("Checkpoint byte count does not match the Murea-Fast checkpoint")
    header = read_header(path)
    keys = set(header) - {"__metadata__"}
    if len(keys) != 4349 or any("reference_id_embedding" in k for k in keys):
        raise ValueError("Wrong checkpoint architecture: expected the Murea-Fast checkpoint (no Ref-ID embedding)")
    metadata = header.get("__metadata__", {})
    contract = json.loads(metadata["distillation_contract"])
    if contract["denoising_sigmas"] != SIGMAS or contract["num_transitions"] != 8:
        raise ValueError("Wrong distilled sampling schedule")
    if contract["sampler_mode"] != "deterministic_euler":
        raise ValueError("Expected deterministic Euler")
    if not contract["exclude_reference_tokens_from_av_cross_attention"]:
        raise ValueError("Missing reference AV attention exclusion")
    if contract["reference_temporal_positioning"] != {
        "mode": "negative_slot_center", "offset": -1.0, "stride": 1.0
    }:
        raise ValueError("Wrong ordered-reference temporal positioning")
    sampling = contract["guidance"]["student_sampling"]
    for modality in ("audio", "video"):
        if sampling[modality] != {
            "cfg_scale": 1.0, "modality_scale": 1.0, "rescale_scale": 0.0,
            "skip_step": 0, "stg_blocks": [], "stg_scale": 0.0
        }:
            raise ValueError("Unexpected distilled guidance contract")
    result = {"path": str(path), "bytes": path.stat().st_size,
              "tensor_count": len(keys), "sha256_verified": False,
              "expected_sha256": CHECKPOINT_SHA256, "contract": contract}
    if full_hash:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != CHECKPOINT_SHA256:
            raise ValueError("Checkpoint SHA-256 mismatch")
        result["sha256_verified"] = True
    return result
