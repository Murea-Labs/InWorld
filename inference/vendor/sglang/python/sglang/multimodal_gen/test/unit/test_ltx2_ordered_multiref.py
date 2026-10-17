import os
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import pytest
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
from sglang.multimodal_gen.runtime.layers.attention.layer import USPAttention
from sglang.multimodal_gen.runtime.models.dits.ltx_2 import (
    apply_ltx2_reference_id_embedding,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.image_encoding import (
    LTX2ImageEncodingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.ltx_2.denoising import (
    LTX2DenoisingStage,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum


def _reference_batch(
    target: torch.Tensor, references: list[torch.Tensor]
) -> SimpleNamespace:
    return SimpleNamespace(
        image_path=None,
        reference_image_paths=[f"ref_{index}.png" for index in range(len(references))],
        latents=target,
        ltx2_reference_latents=references,
        ltx2_reference_latent_shapes=[
            {
                "num_frames": 1,
                "height": 1,
                "width": int(reference.shape[1]),
                "token_count": int(reference.shape[1]),
            }
            for reference in references
        ],
        num_frames=9,
    )


def test_sampling_params_keep_ordered_references_separate_from_ti2v():
    params = LTX25SamplingParams(reference_image_paths=["a.png", "b.png", "c.png"])
    params._validate_with_pipeline_config(LTX25PipelineConfig())
    assert params.reference_image_paths == ["a.png", "b.png", "c.png"]
    assert params.image_path is None

    request = VideoGenerationsRequest(
        prompt="test", reference_image_paths=["a.png", "b.png"]
    )
    assert request.reference_image_paths == ["a.png", "b.png"]


def test_video_api_lowers_ordered_references_into_ltx25_sampling_params():
    request = VideoGenerationsRequest(
        prompt="test",
        reference_image_paths=["first.png", "second.png", "third.png"],
        width=1280,
        height=704,
        num_frames=121,
        fps=24,
        seed=844,
    )
    server_args = SimpleNamespace(
        backend="auto",
        model_id=None,
        model_path=("/path/to/LTX-2.5-Diffusers"),
        output_path="outputs",
        comfyui_mode=False,
        num_gpus=1,
        pipeline_class_name="LTX2Pipeline",
        pipeline_config=LTX25PipelineConfig(),
    )

    with (
        patch(
            "sglang.multimodal_gen.runtime.entrypoints.openai.video_api."
            "get_global_server_args",
            return_value=server_args,
        ),
        patch(
            "sglang.multimodal_gen.runtime.entrypoints.openai.utils."
            "get_global_server_args",
            return_value=server_args,
        ),
    ):
        params = _build_video_sampling_params("ordered-reference-request", request)

    assert isinstance(params, LTX25SamplingParams)
    assert params.reference_image_paths == [
        "first.png",
        "second.png",
        "third.png",
    ]
    assert params.image_path is None
    assert (
        params.width,
        params.height,
        params.num_frames,
        params.fps,
        params.seed,
    ) == (
        1280,
        704,
        121,
        24,
        844,
    )


def test_sampling_params_reject_ti2v_and_ordered_references_together():
    with pytest.raises(ValueError, match="mutually exclusive"):
        LTX25SamplingParams(
            image_path="first.png", reference_image_paths=["identity.png"]
        )


def test_sampling_params_accept_temporal_continuation_video():
    params = LTX25SamplingParams(
        video_path="parent.mp4",
        continuation_prefix_frames=121,
        num_frames=241,
    )

    assert params.video_path == "parent.mp4"
    assert params.continuation_prefix_frames == 121


def test_video_api_lowers_temporal_continuation_prefix_frames():
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
        params = _build_video_sampling_params("continuation-test", request)

    assert params.video_path == "parent.mp4"
    assert params.num_frames == 241
    assert params.continuation_prefix_frames == 121


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"video_path": ["a.mp4", "b.mp4"]},
            "exactly one",
        ),
        (
            {"video_path": "a.mp4", "continuation_prefix_frames": 120},
            "frames % 8 == 1",
        ),
        (
            {
                "video_path": "a.mp4",
                "continuation_prefix_frames": 121,
                "num_frames": 121,
            },
            "smaller than num_frames",
        ),
        (
            {"video_path": "a.mp4", "reference_image_paths": ["ref.png"]},
            "mutually exclusive",
        ),
        # A float or a string with the right value is still not an integer.
        ({"video_path": "a.mp4", "continuation_prefix_frames": 17.0}, "integer"),
        ({"video_path": "a.mp4", "continuation_prefix_frames": "17"}, "integer"),
        ({"video_path": "a.mp4", "continuation_prefix_frames": True}, "integer"),
    ],
)
def test_sampling_params_reject_invalid_temporal_continuation(kwargs, message):
    with pytest.raises(ValueError, match=message):
        LTX25SamplingParams(**kwargs)


def test_prefix_frame_count_uses_longest_valid_aligned_prefix():
    assert (
        LTX2ImageEncodingStage._resolve_ltx2_prefix_frame_count(
            decoded_frames=125,
            target_frames=241,
            requested_frames=None,
        )
        == 121
    )


@pytest.mark.parametrize("count", [0, 6])
def test_sampling_params_reject_invalid_reference_count(count: int):
    with pytest.raises(ValueError, match="between 1 and 5"):
        LTX25SamplingParams(
            reference_image_paths=[f"ref_{index}.png" for index in range(count)]
        )


def test_stride_minimal_resize_preserves_aligned_training_shapes():
    assert LTX2ImageEncodingStage._stride_minimal_reference_size(
        1536, 2048, stride=32
    ) == (2048, 1536)
    assert LTX2ImageEncodingStage._stride_minimal_reference_size(
        2848, 1600, stride=32
    ) == (1600, 2848)


def test_ordered_reference_latent_dump_is_opt_in_and_atomic(tmp_path, monkeypatch):
    monkeypatch.setenv("SGLANG_LTX2_REFERENCE_LATENT_DUMP_DIR", str(tmp_path))
    entries = [
        {
            "latents": torch.arange(8, dtype=torch.bfloat16).reshape(2, 1, 2, 2),
            "source_ref_path": "first.png",
        }
    ]

    LTX2ImageEncodingStage._dump_ordered_reference_latents(
        SimpleNamespace(request_id="request-1"), entries
    )

    dump = torch.load(tmp_path / "request-1.pt", map_location="cpu", weights_only=True)
    assert dump["ref_count"] == 1
    assert dump["ref_resize_policy"] == "stride_minimal_resize"
    torch.testing.assert_close(dump["refs"][0]["latents"], entries[0]["latents"])
    assert list(tmp_path.glob("*.tmp.*")) == []


def test_ordered_references_append_clean_tokens_in_input_order():
    target = torch.arange(10, dtype=torch.float32).reshape(1, 5, 2)
    first = torch.full((1, 2, 2), 11.0)
    second = torch.full((1, 3, 2), 22.0)
    batch = _reference_batch(target, [first, second])

    with patch(
        "sglang.multimodal_gen.runtime.pipelines_core.stages."
        "model_specific_stages.ltx_2.denoising.get_sp_world_size",
        return_value=1,
    ):
        LTX2DenoisingStage._prepare_ltx2_ordered_reference_state(batch, None)

    assert batch.latents.shape == (1, 10, 2)
    torch.testing.assert_close(batch.latents[:, :5], target)
    torch.testing.assert_close(batch.latents[:, 5:], torch.zeros(1, 5, 2))
    torch.testing.assert_close(
        batch.ltx2_ordered_ref_clean_latent[:, :5], torch.zeros_like(target)
    )
    torch.testing.assert_close(batch.ltx2_ordered_ref_clean_latent[:, 5:7], first)
    torch.testing.assert_close(batch.ltx2_ordered_ref_clean_latent[:, 7:10], second)
    assert batch.ltx2_ordered_ref_denoise_mask.flatten().tolist() == [
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    assert batch.ltx2_ordered_ref_target_mask.tolist() == [
        [True, True, True, True, True, False, False, False, False, False]
    ]
    assert batch.ltx2_ordered_ref_ranges == [(5, 7), (7, 10)]
    assert batch.ltx2_ordered_ref_reference_ids.tolist() == [
        [0, 0, 0, 0, 0, 1, 1, 2, 2, 2]
    ]


def test_arbitrary_token_sp_sharding_and_padding_use_total_sequence():
    target = torch.zeros(1, 5, 2)
    references = [torch.ones(1, 2, 2), torch.full((1, 4, 2), 2.0)]
    batch = _reference_batch(target, references)

    with (
        patch(
            "sglang.multimodal_gen.runtime.pipelines_core.stages."
            "model_specific_stages.ltx_2.denoising.get_sp_world_size",
            return_value=4,
        ),
        patch(
            "sglang.multimodal_gen.runtime.pipelines_core.stages."
            "model_specific_stages.ltx_2.denoising.get_sp_parallel_rank",
            return_value=3,
        ),
    ):
        LTX2DenoisingStage._prepare_ltx2_ordered_reference_state(batch, None)

    assert batch.did_sp_shard_latents
    assert batch.sp_video_orig_token_count == 11
    assert batch.sp_video_padded_token_count == 12
    assert batch.sp_video_valid_token_count == 2
    assert batch.latents.shape == (1, 3, 2)
    assert batch.ltx2_ordered_ref_local_start == 9
    assert batch.ltx2_ordered_ref_local_end == 12
    assert batch.ltx2_ordered_ref_target_mask.tolist() == [[False, False, False]]
    assert batch.ltx2_ordered_ref_denoise_mask.flatten().tolist() == [0.0, 0.0, 0.0]
    assert batch.ltx2_ordered_ref_reference_ids.tolist() == [[2, 2, 0]]


def test_ordered_reference_attention_meta_matches_validation_segments():
    batch = SimpleNamespace(
        ltx2_ordered_ref_target_token_count=5,
        ltx2_ordered_ref_orig_token_count=10,
        sp_video_padded_token_count=12,
        ltx2_ordered_ref_ranges=[(5, 7), (7, 10)],
    )
    meta = LTX2DenoisingStage._build_ltx2_ordered_reference_attention_meta(
        batch=batch,
        batch_size=1,
        device=torch.device("cpu"),
    )["ordered_ref"]

    assert meta["target_indices"].tolist() == [0, 1, 2, 3, 4, 10, 11]
    assert meta["reference_indices"].tolist() == [5, 6, 7, 8, 9]
    assert meta["target_cu_seqlens"].tolist() == [0, 5, 7]
    assert meta["sample_cu_seqlens"].tolist() == [0, 10, 12]
    assert meta["reference_cu_seqlens"].tolist() == [0, 2, 5]


def test_ordered_reference_attention_matches_validation_routing_on_cpu():
    batch = SimpleNamespace(
        ltx2_ordered_ref_target_token_count=3,
        ltx2_ordered_ref_orig_token_count=7,
        sp_video_padded_token_count=7,
        ltx2_ordered_ref_ranges=[(3, 5), (5, 7)],
    )
    meta = LTX2DenoisingStage._build_ltx2_ordered_reference_attention_meta(
        batch=batch,
        batch_size=1,
        device=torch.device("cpu"),
    )["ordered_ref"]
    generator = torch.Generator().manual_seed(17)
    q = torch.randn(1, 7, 2, 4, generator=generator)
    k = torch.randn(1, 7, 2, 4, generator=generator)
    v = torch.randn(1, 7, 2, 4, generator=generator)
    fake_attention = SimpleNamespace(
        backend=AttentionBackendEnum.FA,
        allow_cudnn_sdp=False,
        softmax_scale=0.5,
    )

    with patch(
        "sglang.multimodal_gen.runtime.layers.attention.layer."
        "get_ulysses_parallel_world_size",
        return_value=1,
    ):
        actual = USPAttention._forward_ordered_references(
            fake_attention,
            q,
            k,
            v,
            meta,
            effective_skip_sp=True,
        )

    expected = torch.zeros_like(q)
    expected[:, :3] = torch.nn.functional.scaled_dot_product_attention(
        q[:, :3].transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        scale=0.5,
    ).transpose(1, 2)
    for start, end in [(3, 5), (5, 7)]:
        expected[:, start:end] = torch.nn.functional.scaled_dot_product_attention(
            q[:, start:end].transpose(1, 2),
            k[:, start:end].transpose(1, 2),
            v[:, start:end].transpose(1, 2),
            scale=0.5,
        ).transpose(1, 2)
    torch.testing.assert_close(actual, expected)


def test_reference_id_embedding_is_zero_for_target_and_distinct_per_ref():
    embedding = torch.nn.Embedding(4, 3, padding_idx=0)
    with torch.no_grad():
        embedding.weight.copy_(
            torch.tensor(
                [
                    [9.0, 9.0, 9.0],
                    [1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0],
                    [7.0, 8.0, 9.0],
                ]
            )
        )
    hidden = torch.zeros(1, 4, 3)
    ids = torch.tensor([[0, 1, 2, 0]], dtype=torch.int64)
    actual = apply_ltx2_reference_id_embedding(hidden, ids, embedding)
    torch.testing.assert_close(
        actual,
        torch.tensor(
            [[[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [0.0, 0.0, 0.0]]]
        ),
    )


class _FakeVideoRope:
    @staticmethod
    def prepare_video_coords(
        *, batch_size, num_frames, height, width, device, fps, **_kwargs
    ):
        del fps
        tokens = int(num_frames) * int(height) * int(width)
        coords = torch.zeros(batch_size, 3, tokens, 2, device=device)
        coords[:, 0, :, 1] = 1.0
        return coords


def test_ordered_reference_rope_centers_are_negative_and_ordered():
    batch = SimpleNamespace(
        fps=24,
        ltx2_reference_latent_shapes=[
            {"num_frames": 1, "height": 1, "width": 2, "token_count": 2},
            {"num_frames": 1, "height": 1, "width": 3, "token_count": 3},
        ],
        ltx2_ordered_ref_orig_token_count=9,
        sp_video_padded_token_count=9,
        ltx2_ordered_ref_local_start=0,
        ltx2_ordered_ref_local_end=9,
    )
    coords = LTX2DenoisingStage._build_ltx2_ordered_reference_coords(
        model=SimpleNamespace(rope=_FakeVideoRope()),
        batch=batch,
        latent_model_input=torch.zeros(1, 9, 2),
        target_num_frames=2,
        target_height=1,
        target_width=2,
    )

    centers = coords[:, 0].mean(dim=-1)
    torch.testing.assert_close(centers[:, 4:6], torch.full((1, 2), -2.0))
    torch.testing.assert_close(centers[:, 6:9], torch.full((1, 3), -1.0))


def test_decode_domain_truncates_appended_references():
    stage = object.__new__(LTX2DenoisingStage)
    batch = SimpleNamespace(
        did_sp_shard_latents=False,
        ltx2_ordered_ref_target_token_count=5,
    )
    latents = torch.arange(20, dtype=torch.float32).reshape(1, 10, 2)
    output, trajectory = stage._postprocess_sp_latents(batch, latents, None)
    assert trajectory is None
    torch.testing.assert_close(output, latents[:, :5])


# --- Request fields Murea-Fast refuses (request_policy) --------------------------------

_SERVER_PATHS_ENV = "MUREA_FAST_ALLOW_SERVER_PATHS"
_PNG = "data:image/png;base64,iVBORw0KGgo="


def _server_args(tmp_path):
    return SimpleNamespace(
        backend="auto",
        model_id=None,
        model_path="/path/to/LTX-2.5-Diffusers",
        output_path=str(tmp_path / "outputs"),
        input_save_path=str(tmp_path / "uploads"),
        comfyui_mode=False,
        num_gpus=1,
        pipeline_class_name="LTX2Pipeline",
        pipeline_config=LTX25PipelineConfig(),
        served_model_name="Murea-Fast",
    )


@pytest.fixture
def video_client(tmp_path, monkeypatch):
    """/v1/videos with the scheduler cut off: a request the policy lets through
    reaches _build_video_sampling_params, which records it and answers 400."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from sglang.multimodal_gen.runtime.entrypoints.openai import utils, video_api

    server_args = _server_args(tmp_path)
    monkeypatch.setattr(video_api, "get_global_server_args", lambda: server_args)
    monkeypatch.setattr(utils, "get_global_server_args", lambda: server_args)
    monkeypatch.delenv(_SERVER_PATHS_ENV, raising=False)
    accepted = []

    def build(request_id, req):
        accepted.append(req)
        raise ValueError("accepted by the request policy")

    monkeypatch.setattr(video_api, "_build_video_sampling_params", build)
    app = FastAPI()
    app.include_router(video_api.router)
    with TestClient(app) as client:
        client.accepted = accepted
        client.uploads = tmp_path / "uploads"
        yield client


def _accepted(response):
    return (
        response.status_code == 400
        and response.json()["detail"] == "accepted by the request policy"
    )


_UNSUPPORTED = {
    "enable_frame_interpolation": True,
    "frame_interpolation_exp": 2,
    "frame_interpolation_scale": 0.5,
    "frame_interpolation_model_path": "org/interpolator",
    "enable_upscaling": True,
    "upscaling_model_path": "org/upscaler",
    "upscaling_scale": 2,
    "diffusers_kwargs": {"flow_shift": 3.0},
}


@pytest.mark.parametrize("allow", [False, True])
@pytest.mark.parametrize(("name", "value"), list(_UNSUPPORTED.items()))
def test_video_api_always_refuses_post_processing_fields(
    video_client, monkeypatch, allow, name, value
):
    if allow:
        monkeypatch.setenv(_SERVER_PATHS_ENV, "true")
    json_response = video_client.post(
        "/v1/videos", json={"prompt": "test", "reference_image_paths": [_PNG], name: value}
    )
    nested = video_client.post(
        "/v1/videos", json={"prompt": "test", "extra_body": {name: value}}
    )
    form = {"prompt": "test"}
    if name == "diffusers_kwargs":
        form["extra_body"] = '{"diffusers_kwargs": {"flow_shift": 3.0}}'
    else:
        form[name] = str(value).lower() if isinstance(value, bool) else str(value)
    multipart = video_client.post(
        "/v1/videos", data=form, files={"video_reference": ("p.mp4", b"0", "video/mp4")}
    )
    for response in (json_response, nested, multipart):
        assert response.status_code == 400, response.text
        assert name in response.json()["detail"]
    assert video_client.accepted == []
    # The refused multipart request leaves no upload behind.
    assert list(video_client.uploads.glob("*")) == []


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("output_path", "/srv/elsewhere"),
        ("perf_dump_path", "/srv/perf.json"),
        ("video_path", "/srv/parent.mp4"),
        ("video_url", "parent.mp4"),
        ("input_reference", "/srv/person.png"),
        ("reference_url", "/srv/parent.mp4"),
        ("reference_image_paths", [_PNG, "/srv/cup.png"]),
    ],
)
def test_video_api_refuses_server_paths_by_default(video_client, name, value):
    response = video_client.post("/v1/videos", json={"prompt": "test", name: value})
    assert response.status_code == 400, response.text
    # A video given as video_url or reference_url is also copied to video_path.
    assert response.json()["detail"].startswith((name, "video_path"))
    assert "allow_server_paths" in response.json()["detail"]
    assert video_client.accepted == []


@pytest.mark.parametrize(
    "form",
    [
        {"output_path": "/srv/elsewhere"},
        {"video_path": "/srv/parent.mp4"},
        {"video_url": "/srv/parent.mp4"},
        {"extra_body": '{"video_path": "/srv/parent.mp4"}'},
    ],
)
def test_multipart_refuses_server_paths_by_default(video_client, form):
    # (None, value) parts force multipart/form-data even without a file.
    fields = {key: (None, value) for key, value in {"prompt": "test", **form}.items()}
    response = video_client.post("/v1/videos", files=fields)
    assert response.status_code == 400, response.text
    assert "allow_server_paths" in response.json()["detail"]
    assert video_client.accepted == []


def test_urls_and_uploads_still_work(video_client):
    references = [_PNG, "https://example.com/cup.png"]
    response = video_client.post(
        "/v1/videos", json={"prompt": "test", "reference_image_paths": references}
    )
    assert _accepted(response), response.text
    assert video_client.accepted[-1].reference_image_paths == references

    # The continuation prefix as a multipart upload: the server's own saved copy is a
    # server path, but not one the client chose.
    response = video_client.post(
        "/v1/videos",
        data={"prompt": "test", "size": "1280x704"},
        files={"video_reference": ("prefix.mp4", b"prefix", "video/mp4")},
    )
    assert _accepted(response), response.text
    saved = video_client.accepted[-1].video_path
    assert saved.startswith(str(video_client.uploads))
    assert open(saved, "rb").read() == b"prefix"


def test_upload_file_names_cannot_leave_the_upload_directory(video_client, tmp_path):
    for name in ("../../escape.mp4", "nested/dir/escape.mp4", "..\\escape.mp4"):
        response = video_client.post(
            "/v1/videos",
            data={"prompt": "test", "size": "1280x704"},
            files={"video_reference": (name, b"prefix", "video/mp4")},
        )
        assert _accepted(response), response.text
        saved = video_client.accepted[-1].video_path
        assert os.path.dirname(os.path.realpath(saved)) == os.path.realpath(
            video_client.uploads
        )
        assert saved.endswith("_escape.mp4")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["uploads"]


def test_server_paths_work_when_the_launcher_allows_them(video_client, monkeypatch):
    monkeypatch.setenv(_SERVER_PATHS_ENV, "true")
    response = video_client.post(
        "/v1/videos",
        json={
            "prompt": "test",
            "video_path": "/srv/parent.mp4",
            "continuation_prefix_frames": 17,
            "perf_dump_path": "/srv/perf.json",
            "output_path": "/srv/outputs",
        },
    )
    assert _accepted(response), response.text
    assert video_client.accepted[-1].video_path == "/srv/parent.mp4"


def test_sampling_params_never_receive_post_processing_fields(tmp_path):
    from sglang.multimodal_gen.runtime.entrypoints.openai import video_api

    request = VideoGenerationsRequest(prompt="test", **_UNSUPPORTED)
    server_args = _server_args(tmp_path)
    with (
        patch.object(video_api, "get_global_server_args", return_value=server_args),
        patch(
            "sglang.multimodal_gen.runtime.entrypoints.openai.utils.get_global_server_args",
            return_value=server_args,
        ),
    ):
        params = _build_video_sampling_params("dropped-fields", request)
    assert not params.enable_frame_interpolation
    assert params.frame_interpolation_model_path is None
    assert not params.enable_upscaling
    assert params.upscaling_model_path is None
    assert params.diffusers_kwargs is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"enable_frame_interpolation": True},
        {"frame_interpolation_model_path": "org/interpolator"},
        {"enable_upscaling": True},
        {"upscaling_model_path": "org/upscaler"},
    ],
)
def test_sampling_params_refuse_post_processing_on_every_endpoint(kwargs):
    with pytest.raises(ValueError, match="not supported by Murea-Fast"):
        LTX25SamplingParams(**kwargs)


def test_sampling_params_refuse_perf_dump_path_unless_allowed(monkeypatch):
    monkeypatch.delenv(_SERVER_PATHS_ENV, raising=False)
    with pytest.raises(ValueError, match="allow_server_paths"):
        LTX25SamplingParams(perf_dump_path="/srv/perf.json")
    monkeypatch.setenv(_SERVER_PATHS_ENV, "true")
    assert LTX25SamplingParams(perf_dump_path="/srv/perf.json").perf_dump_path


def test_vertex_route_refuses_server_image_paths(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from sglang.multimodal_gen.runtime.entrypoints import http_server

    monkeypatch.delenv(_SERVER_PATHS_ENV, raising=False)
    built = []

    def build(request_id, **kwargs):
        built.append(kwargs)
        raise ValueError("accepted by the request policy")

    monkeypatch.setattr(http_server, "build_sampling_params", build)
    monkeypatch.setattr(http_server, "get_global_server_args", lambda: None)
    app = FastAPI()
    app.include_router(http_server.vertex_router)
    with TestClient(app, raise_server_exceptions=False) as client:
        for key in ("image", "image_url"):
            response = client.post(
                http_server.VERTEX_ROUTE,
                json={"instances": [{"prompt": "test", key: "/srv/person.png"}]},
            )
            assert response.status_code == 400, response.text
            assert "allow_server_paths" in response.json()["detail"]
        assert built == []
        client.post(
            http_server.VERTEX_ROUTE,
            json={"instances": [{"prompt": "test", "image": _PNG}]},
        )
        assert built[-1]["image_path"] == _PNG


# --- Input limits: prefix decoding and remote downloads -------------------------------


def test_prefix_decode_stops_at_the_frames_the_prefix_can_use():
    limit = LTX2ImageEncodingStage._ltx2_prefix_decode_limit
    assert limit(target_frames=137, requested_frames=17) == 17
    # Without an explicit prefix the longest valid one is shorter than the target.
    assert limit(target_frames=241, requested_frames=None) == 240
    resolved = LTX2ImageEncodingStage._resolve_ltx2_prefix_frame_count(
        decoded_frames=limit(target_frames=241, requested_frames=None),
        target_frames=241,
        requested_frames=None,
    )
    assert resolved == LTX2ImageEncodingStage._resolve_ltx2_prefix_frame_count(
        decoded_frames=1000, target_frames=241, requested_frames=None
    )


def _write_video(path, frames: int) -> None:
    import imageio
    import numpy as np

    with imageio.get_writer(str(path), fps=24, macro_block_size=1) as writer:
        for index in range(frames):
            writer.append_data(np.full((32, 32, 3), index * 8 % 256, dtype=np.uint8))


def test_load_video_decodes_only_the_requested_frames(tmp_path):
    from sglang.multimodal_gen.runtime.utils.vision import load_video

    path = tmp_path / "parent.mp4"
    _write_video(path, 30)
    assert len(load_video(str(path))) == 30
    assert len(load_video(str(path), max_frames=17)) == 17
    assert len(load_video(str(path), max_frames=40)) == 30
    for bad in (0, 17.0, "17"):
        with pytest.raises(ValueError, match="max_frames"):
            load_video(str(path), max_frames=bad)


class _FakeVideoResponse:
    status_code = 200

    def __init__(self, chunks, headers=None):
        self.chunks = chunks
        self.headers = headers or {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def iter_content(self, chunk_size):
        yield from self.chunks


def test_remote_video_download_has_a_total_deadline(tmp_path, monkeypatch):
    from sglang.multimodal_gen.runtime.utils import vision

    created = []
    real_named_temporary_file = vision.tempfile.NamedTemporaryFile

    def named_temporary_file(*args, **kwargs):
        handle = real_named_temporary_file(*args, dir=tmp_path, **kwargs)
        created.append(handle.name)
        return handle

    clock = iter([0.0, 1.0, 2.0, vision.REMOTE_VIDEO_DEADLINE_SECONDS + 1])
    monkeypatch.setattr(vision.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(vision.tempfile, "NamedTemporaryFile", named_temporary_file)
    monkeypatch.setattr(
        vision.requests,
        "get",
        lambda *a, **k: _FakeVideoResponse([b"x"] * 10),
    )
    with pytest.raises(TimeoutError, match="not downloaded within"):
        vision.load_video("https://example.com/parent.mp4")
    assert created and not any(os.path.exists(name) for name in created)


def test_remote_video_download_has_a_size_limit(tmp_path, monkeypatch):
    from sglang.multimodal_gen.runtime.utils import vision

    monkeypatch.setattr(vision, "REMOTE_VIDEO_MAX_BYTES", 4)
    real_named_temporary_file = tempfile.NamedTemporaryFile
    monkeypatch.setattr(
        vision.tempfile,
        "NamedTemporaryFile",
        lambda *a, **k: real_named_temporary_file(*a, dir=tmp_path, **k),
    )
    for response in (
        _FakeVideoResponse([b"x"], headers={"content-length": "5"}),
        _FakeVideoResponse([b"xxx", b"xxx"]),
    ):
        monkeypatch.setattr(vision.requests, "get", lambda *a, r=response, **k: r)
        with pytest.raises(ValueError, match="byte input limit"):
            vision.load_video("https://example.com/parent.mp4")
    assert list(tmp_path.iterdir()) == []


def _image_client(monkeypatch, handler):
    import httpx

    from sglang.multimodal_gen.runtime.entrypoints.openai import utils

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        utils.httpx,
        "AsyncClient",
        lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs),
    )
    return utils


def test_image_url_download_has_a_size_limit(tmp_path, monkeypatch):
    import asyncio

    import httpx

    utils = _image_client(
        monkeypatch,
        lambda request: httpx.Response(
            200, content=b"x" * 64, headers={"content-type": "image/png"}
        ),
    )
    monkeypatch.setattr(utils, "REMOTE_IMAGE_MAX_BYTES", 32)
    with pytest.raises(Exception, match="byte download limit"):
        asyncio.run(
            utils._save_url_image_to_path(
                "https://example.com/a.png", str(tmp_path / "a")
            )
        )
    assert list(tmp_path.iterdir()) == []

    monkeypatch.setattr(utils, "REMOTE_IMAGE_MAX_BYTES", 64)
    saved = asyncio.run(
        utils._save_url_image_to_path("https://example.com/a.png", str(tmp_path / "a"))
    )
    assert open(saved, "rb").read() == b"x" * 64


def test_image_url_download_has_a_total_deadline(tmp_path, monkeypatch):
    import asyncio

    import httpx

    async def slow(request):
        await asyncio.sleep(5)
        return httpx.Response(200, content=b"x")

    utils = _image_client(monkeypatch, slow)
    monkeypatch.setattr(utils, "REMOTE_IMAGE_DEADLINE_SECONDS", 0.05)
    with pytest.raises(Exception, match="no complete response within"):
        asyncio.run(
            utils._save_url_image_to_path(
                "https://example.com/a.png", str(tmp_path / "a")
            )
        )


def test_weight_lora_and_rollout_endpoints_are_not_mounted():
    from sglang.multimodal_gen.runtime.entrypoints import http_server

    app = http_server.create_app(SimpleNamespace(pipeline_config=LTX25PipelineConfig()))
    paths = set(app.openapi()["paths"])
    for path in (
        "/update_weights_from_disk",
        "/update_weights_from_tensor",
        "/update_weights_from_tensor_checker",
        "/get_weights_checksum",
        "/release_memory_occupation",
        "/resume_memory_occupation",
        "/v1/set_lora",
        "/v1/merge_lora_weights",
        "/v1/unmerge_lora_weights",
    ):
        assert path not in paths
    assert not any(str(path).startswith("/rollout") for path in paths)
    for path in ("/health", "/v1/models", "/v1/videos", "/v1/videos/{video_id}/content"):
        assert path in paths
