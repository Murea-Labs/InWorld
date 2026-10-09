from types import SimpleNamespace
from unittest.mock import patch

import torch
from sglang.multimodal_gen.runtime.layers.attention import layer as attention_layer
from sglang.multimodal_gen.runtime.layers.attention.layer import USPAttention
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import (
    DenoisingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.ltx_2.denoising import (
    LTX2DenoisingStage,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum


def test_ti2v_request_delegates_to_cache_dit_mount_or_refresh():
    stage = object.__new__(LTX2DenoisingStage)
    batch = SimpleNamespace(image_path="/tmp/reference.png")

    with patch.object(
        DenoisingStage, "_maybe_enable_cache_dit", autospec=True
    ) as parent:
        stage._maybe_enable_cache_dit(30, batch)

    parent.assert_called_once_with(stage, 30, batch)


def test_sp_padding_mask_is_omitted_when_global_sequence_has_no_padding():
    batch = SimpleNamespace(
        sp_video_has_padding=False,
        sp_video_valid_token_count=8,
    )

    mask = LTX2DenoisingStage._build_ltx2_sp_padding_mask(
        batch,
        seq_len=8,
        batch_size=1,
        key="sp_video_valid_token_count",
        padding_key="sp_video_has_padding",
        device=torch.device("cpu"),
    )

    assert mask is None


def test_sp_padding_mask_stays_present_on_every_rank_when_padding_exists():
    full_rank = SimpleNamespace(
        sp_video_has_padding=True,
        sp_video_valid_token_count=8,
    )
    tail_rank = SimpleNamespace(
        sp_video_has_padding=True,
        sp_video_valid_token_count=5,
    )

    full_mask = LTX2DenoisingStage._build_ltx2_sp_padding_mask(
        full_rank,
        seq_len=8,
        batch_size=1,
        key="sp_video_valid_token_count",
        padding_key="sp_video_has_padding",
        device=torch.device("cpu"),
    )
    tail_mask = LTX2DenoisingStage._build_ltx2_sp_padding_mask(
        tail_rank,
        seq_len=8,
        batch_size=1,
        key="sp_video_valid_token_count",
        padding_key="sp_video_has_padding",
        device=torch.device("cpu"),
    )

    assert torch.equal(full_mask, torch.ones(1, 8, dtype=torch.bool))
    assert torch.equal(
        tail_mask,
        torch.tensor([[True, True, True, True, True, False, False, False]]),
    )


def test_sp_tail_attention_meta_uses_global_video_token_bounds():
    batch = SimpleNamespace(
        sp_video_has_padding=True,
        sp_video_orig_token_count=29,
        sp_video_padded_token_count=32,
    )

    with patch(
        "sglang.multimodal_gen.runtime.pipelines_core.stages."
        "model_specific_stages.ltx_2.denoising.get_sp_world_size",
        return_value=4,
    ):
        meta = LTX2DenoisingStage._build_ltx2_sp_tail_attention_meta(
            batch,
            seq_len=8,
            batch_size=1,
            device=torch.device("cpu"),
        )

    assert meta is not None
    assert meta["pad_start"] == 29
    assert meta["pad_end"] == 32
    assert meta["max_seqlen_tail"] == 29
    assert meta["cu_seqlens_tail"].tolist() == [0, 29, 32]


def test_cfg_batch_repeat_expands_sp_tail_attention_meta():
    attention_meta = {
        "pad_start": 29,
        "pad_end": 32,
        "cu_seqlens_tail": torch.tensor([0, 29, 32], dtype=torch.int32),
        "max_seqlen_tail": 29,
    }

    repeated = LTX2DenoisingStage._repeat_ltx2_model_kwargs_batch(
        {"video_self_attention_mask_meta": attention_meta},
        target_batch_size=2,
    )

    repeated_meta = repeated["video_self_attention_mask_meta"]
    assert repeated_meta is not attention_meta
    assert repeated_meta["cu_seqlens_tail"].tolist() == [0, 29, 32, 61, 64]
    assert attention_meta["cu_seqlens_tail"].tolist() == [0, 29, 32]


def test_split_guidance_passes_restore_per_batch_sp_tail_attention_meta():
    attention_meta = {
        "pad_start": 29,
        "pad_end": 32,
        "cu_seqlens_tail": torch.tensor([0, 29, 32, 61, 64], dtype=torch.int32),
        "max_seqlen_tail": 29,
    }

    chunks = LTX2DenoisingStage._split_ltx2_model_kwargs(
        {"video_self_attention_mask_meta": attention_meta},
        split_sizes=[1, 1],
    )

    assert len(chunks) == 2
    assert chunks[0]["video_self_attention_mask_meta"]["cu_seqlens_tail"].tolist() == [
        0,
        29,
        32,
    ]
    assert chunks[1]["video_self_attention_mask_meta"]["cu_seqlens_tail"].tolist() == [
        0,
        29,
        32,
    ]


def test_usp_tail_padding_keeps_sage_backend():
    class FakeSageImpl:
        def __init__(self):
            self.call = None

        def forward_varlen(self, query, key, value, **kwargs):
            self.call = (query.shape, key.shape, value.shape, kwargs)
            output = torch.ones_like(query)
            output[6:].zero_()
            return output

    attention = object.__new__(USPAttention)
    torch.nn.Module.__init__(attention)
    fake_impl = FakeSageImpl()
    attention.attn_impl = fake_impl
    attention.backend = AttentionBackendEnum.SAGE_ATTN
    attention.skip_sequence_parallel = False
    attention.sp_attention_mode = "ulysses"
    attention.sp_attention_mode_is_auto = False
    attention.allow_cudnn_sdp = False
    attention.softmax_scale = 0.125
    attention.enable_packed_qkv_input_a2a = False

    local = torch.zeros(1, 2, 4, 8)
    gathered = torch.zeros(1, 8, 1, 8)
    mask = torch.ones(1, 2, dtype=torch.bool)
    meta = {
        "pad_start": 6,
        "pad_end": 8,
        "cu_seqlens_tail": torch.tensor([0, 6, 8], dtype=torch.int32),
    }

    with (
        patch.object(
            attention_layer,
            "get_forward_context",
            return_value=SimpleNamespace(attn_metadata=None),
        ),
        patch.object(
            attention_layer, "get_sequence_parallel_world_size", return_value=4
        ),
        patch.object(attention_layer, "get_ring_parallel_world_size", return_value=1),
        patch.object(
            attention_layer, "get_ulysses_parallel_world_size", return_value=4
        ),
        patch.object(
            attention_layer,
            "_ipc_input_a2a_qkv",
            return_value=(gathered, gathered, gathered),
        ),
        patch.object(
            attention_layer, "_usp_output_all_to_all", side_effect=lambda x, **_: x
        ),
    ):
        output = attention(local, local, local, attn_mask=mask, attn_mask_meta=meta)

    assert fake_impl.call is not None
    assert fake_impl.call[0] == torch.Size([8, 1, 8])
    assert fake_impl.call[3]["cu_seqlens_host"] == (0, 6, 8)
    assert torch.count_nonzero(output[:, 6:]) == 0


def test_usp_tail_padding_keeps_sage_backend_for_cfg_batch():
    class FakeSageImpl:
        def __init__(self):
            self.call = None

        def forward(self, query, key, value, _attention_metadata):
            self.call = (query.shape, key.shape, value.shape)
            return torch.ones_like(query)

    attention = object.__new__(USPAttention)
    torch.nn.Module.__init__(attention)
    fake_impl = FakeSageImpl()
    attention.attn_impl = fake_impl
    attention.backend = AttentionBackendEnum.SAGE_ATTN
    attention.skip_sequence_parallel = False
    attention.sp_attention_mode = "ulysses"
    attention.sp_attention_mode_is_auto = False
    attention.allow_cudnn_sdp = False
    attention.softmax_scale = 0.125
    attention.enable_packed_qkv_input_a2a = False

    local = torch.zeros(2, 2, 4, 8)
    gathered = torch.zeros(2, 8, 1, 8)
    mask = torch.ones(2, 2, dtype=torch.bool)
    meta = {
        "pad_start": 6,
        "pad_end": 8,
        "cu_seqlens_tail": torch.tensor([0, 6, 8, 14, 16], dtype=torch.int32),
    }

    with (
        patch.object(
            attention_layer,
            "get_forward_context",
            return_value=SimpleNamespace(attn_metadata=None),
        ),
        patch.object(
            attention_layer, "get_sequence_parallel_world_size", return_value=4
        ),
        patch.object(attention_layer, "get_ring_parallel_world_size", return_value=1),
        patch.object(
            attention_layer,
            "get_ulysses_parallel_world_size",
            return_value=4,
        ),
        patch.object(
            attention_layer,
            "_ipc_input_a2a_qkv",
            return_value=(gathered, gathered, gathered),
        ),
        patch.object(
            attention_layer, "_usp_output_all_to_all", side_effect=lambda x, **_: x
        ),
    ):
        output = attention(local, local, local, attn_mask=mask, attn_mask_meta=meta)

    assert fake_impl.call is not None
    assert fake_impl.call[0] == torch.Size([2, 6, 1, 8])
    assert torch.count_nonzero(output[:, :6]) == 12 * 8
    assert torch.count_nonzero(output[:, 6:]) == 0


def test_ti2v_clean_tokens_and_mask_are_preserved_for_cache_steps():
    batch = SimpleNamespace(did_sp_shard_latents=False)
    latents = torch.arange(16, dtype=torch.float32).reshape(1, 8, 2)
    condition = torch.tensor([[[101.0, 102.0], [103.0, 104.0]]])

    conditioned, denoise_mask, clean_latent = (
        LTX2DenoisingStage._prepare_ltx2_ti2v_clean_state(
            batch=batch,
            latents=latents,
            image_latent=condition,
            num_img_tokens=2,
            zero_clean_latent=False,
        )
    )

    torch.testing.assert_close(conditioned[:, :2], condition)
    torch.testing.assert_close(clean_latent[:, :2], condition)
    torch.testing.assert_close(denoise_mask[:, :2], torch.zeros(1, 2, 1))
    torch.testing.assert_close(denoise_mask[:, 2:], torch.ones(1, 6, 1))
    torch.testing.assert_close(
        latents, torch.arange(16, dtype=torch.float32).reshape(1, 8, 2)
    )


def test_ulysses_only_conditions_rank_that_owns_first_frame():
    condition = torch.tensor([[[101.0, 102.0], [103.0, 104.0]]])
    first_rank = SimpleNamespace(
        did_sp_shard_latents=True,
        sp_video_tokens_per_frame=2,
        raw_latent_shape=(1, 8, 2),
        sp_video_start_frame=0,
        sp_video_latent_num_frames=2,
    )
    later_rank = SimpleNamespace(
        did_sp_shard_latents=True,
        sp_video_tokens_per_frame=2,
        raw_latent_shape=(1, 8, 2),
        sp_video_start_frame=2,
        sp_video_latent_num_frames=2,
    )

    first_spans = LTX2DenoisingStage._get_ltx2_condition_spans(
        first_rank, torch.zeros(1, 4, 2), condition, 2
    )
    later_spans = LTX2DenoisingStage._get_ltx2_condition_spans(
        later_rank, torch.zeros(1, 4, 2), condition, 2
    )

    assert len(first_spans) == 1
    assert first_spans[0][0] == 0
    assert first_spans[0][1] is condition
    assert later_spans == []


def test_ulysses_slices_temporal_prefix_across_owning_ranks():
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


def test_clean_state_uses_actual_local_prefix_span_length():
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
