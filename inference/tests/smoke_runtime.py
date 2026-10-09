#!/usr/bin/env python3
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.multimodal_gen.configs.pipeline_configs.ltx_2_5 import (
    LTX25PipelineConfig,
)
from sglang.multimodal_gen.configs.sample.ltx_2_5 import LTX25SamplingParams
from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    VideoGenerationsRequest,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.video_api import (
    _build_video_sampling_params,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.image_encoding import (
    LTX2ImageEncodingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.ltx_2.denoising import (
    LTX2DenoisingStage,
)


def check_sampling_params() -> None:
    params = LTX25SamplingParams(
        video_path="parent.mp4",
        continuation_prefix_frames=121,
        num_frames=241,
    )
    assert params.video_path == "parent.mp4"
    assert params.continuation_prefix_frames == 121


def check_http_lowering() -> None:
    request = VideoGenerationsRequest(
        prompt="Continue the story.",
        video_path="parent.mp4",
        num_frames=241,
        continuation_prefix_frames=121,
    )
    server_args = SimpleNamespace(
        backend="auto",
        model_id=None,
        model_path=("/path/to/LTX-2.5-Diffusers"),
        output_path="outputs",
        comfyui_mode=False,
        num_gpus=1,
        pipeline_config=LTX25PipelineConfig(),
        pipeline_class_name="LTX2Pipeline",
    )
    with (
        patch(
            "sglang.multimodal_gen.runtime.entrypoints.openai.video_api.get_global_server_args",
            return_value=server_args,
        ),
        patch(
            "sglang.multimodal_gen.runtime.entrypoints.openai.utils.get_global_server_args",
            return_value=server_args,
        ),
    ):
        params = _build_video_sampling_params("continuation-smoke", request)
    assert params.video_path == "parent.mp4"
    assert params.num_frames == 241
    assert params.continuation_prefix_frames == 121


def check_frame_alignment() -> None:
    assert (
        LTX2ImageEncodingStage._resolve_ltx2_prefix_frame_count(
            decoded_frames=125,
            target_frames=241,
            requested_frames=None,
        )
        == 121
    )


def check_ulysses_prefix_slicing() -> None:
    condition = torch.arange(16, dtype=torch.float32).reshape(1, 8, 2)
    rank0 = SimpleNamespace(
        did_sp_shard_latents=True,
        sp_video_tokens_per_frame=2,
        raw_latent_shape=(1, 16, 2),
        sp_video_start_frame=0,
        sp_video_latent_num_frames=2,
    )
    rank1 = SimpleNamespace(
        did_sp_shard_latents=True,
        sp_video_tokens_per_frame=2,
        raw_latent_shape=(1, 16, 2),
        sp_video_start_frame=2,
        sp_video_latent_num_frames=2,
    )
    rank2 = SimpleNamespace(
        did_sp_shard_latents=True,
        sp_video_tokens_per_frame=2,
        raw_latent_shape=(1, 16, 2),
        sp_video_start_frame=4,
        sp_video_latent_num_frames=2,
    )
    spans0 = LTX2DenoisingStage._get_ltx2_condition_spans(
        rank0, torch.zeros(1, 4, 2), condition, 8
    )
    spans1 = LTX2DenoisingStage._get_ltx2_condition_spans(
        rank1, torch.zeros(1, 4, 2), condition, 8
    )
    spans2 = LTX2DenoisingStage._get_ltx2_condition_spans(
        rank2, torch.zeros(1, 4, 2), condition, 8
    )
    assert spans0[0][0] == 0
    torch.testing.assert_close(spans0[0][1], condition[:, :4])
    assert spans1[0][0] == 0
    torch.testing.assert_close(spans1[0][1], condition[:, 4:8])
    assert spans2 == []


def check_clean_mask() -> None:
    condition = torch.tensor([[[101.0], [102.0], [103.0], [104.0]]])
    batch = SimpleNamespace(
        did_sp_shard_latents=True,
        sp_video_tokens_per_frame=2,
        raw_latent_shape=(1, 16, 1),
        sp_video_start_frame=1,
        sp_video_latent_num_frames=2,
    )
    latents = torch.zeros(1, 4, 1)
    conditioned, denoise_mask, clean_latent = (
        LTX2DenoisingStage._prepare_ltx2_ti2v_clean_state(
            batch=batch,
            latents=latents,
            image_latent=condition,
            num_img_tokens=4,
            zero_clean_latent=True,
        )
    )
    torch.testing.assert_close(conditioned[:, :2], condition[:, 2:4])
    torch.testing.assert_close(clean_latent[:, :2], condition[:, 2:4])
    torch.testing.assert_close(denoise_mask[:, :2], torch.zeros(1, 2, 1))
    torch.testing.assert_close(denoise_mask[:, 2:], torch.ones(1, 2, 1))
    updated = torch.full_like(conditioned, -5.0)
    context = SimpleNamespace(denoise_mask=denoise_mask, clean_latent=clean_latent)
    restored = LTX2DenoisingStage._ltx2_apply_clean_latent_mask(updated, context)
    torch.testing.assert_close(restored[:, :2], condition[:, 2:4])
    torch.testing.assert_close(restored[:, 2:], torch.full((1, 2, 1), -5.0))


def main() -> None:
    checks = (
        check_sampling_params,
        check_http_lowering,
        check_frame_alignment,
        check_ulysses_prefix_slicing,
        check_clean_mask,
    )
    for check in checks:
        check()
        print(f"PASS {check.__name__}", flush=True)
    print(f"PASS total={len(checks)}", flush=True)


if __name__ == "__main__":
    main()
