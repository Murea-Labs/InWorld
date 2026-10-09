# Modified by the Murea authors from SGLang; see inference/docs/provenance.md.
import dataclasses
from dataclasses import field
from typing import Any

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.entrypoints.openai.request_policy import (
    SERVER_PATHS_ENV,
    server_paths_allowed,
)


@dataclasses.dataclass
class LTX2SamplingParams(SamplingParams):
    """Sampling parameters for LTX-2."""

    # Ordered, independent reference images used by the Murea-Fast checkpoint.
    # This is intentionally separate from ``image_path``, whose one/two-image
    # contract remains first-frame/last-frame TI2V conditioning.
    reference_image_paths: list[str] | None = None

    # Temporal continuation input. ``video_path`` is inherited from
    # ``SamplingParams``; when it is set for LTX-2, the decoded prefix is VAE
    # encoded into the target timeline and kept clean during every denoise step.
    # The explicit frame override is useful when the input container has more
    # frames than should be retained.  ``None`` keeps the longest valid prefix.
    continuation_prefix_frames: int | None = None

    # Match the reference defaults used by ltx-pipelines (one-stage).
    # See: LTX-2/packages/ltx-pipelines/src/ltx_pipelines/utils/constants.py
    seed: int = 10
    generator_device: str = "cpu"

    # Video parameters
    height: int = 512
    width: int = 768
    num_frames: int = 121
    fps: int = 24

    # LTX time-shards packed video latents and removes SP-only padding after
    # all-gather, so multi-GPU execution must preserve the requested frames.
    adjust_frames: bool = False

    # Audio specific
    generate_audio: bool = True

    # Denoising parameters
    guidance_scale: float = 4.0
    num_inference_steps: int = 40

    # Match ltx-pipelines default negative prompt (covers video + audio artifacts).
    negative_prompt: str = (
        "blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, "
        "grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, "
        "deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, "
        "wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of "
        "field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent "
        "lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny "
        "valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, "
        "mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, "
        "off-sync audio, incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward "
        "pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, "
        "inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts."
    )

    @classmethod
    def lower_video_request_kwargs(
        cls,
        request: Any,
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        continuation_prefix_frames = (request.model_extra or {}).get(
            "continuation_prefix_frames"
        )
        if continuation_prefix_frames is not None:
            kwargs["continuation_prefix_frames"] = continuation_prefix_frames
        return kwargs

    def _validate(self) -> None:
        super()._validate()
        # Murea-Fast never uses frame interpolation or upscaling. Their model paths
        # name repositories the server would download and load, so they are refused
        # on every endpoint, not only /v1/videos (see request_policy).
        if (
            self.enable_frame_interpolation
            or self.frame_interpolation_model_path
            or self.enable_upscaling
            or self.upscaling_model_path
        ):
            raise ValueError(
                "Frame interpolation and upscaling are not supported by Murea-Fast."
            )
        if self.perf_dump_path and not server_paths_allowed():
            raise ValueError(
                "perf_dump_path writes a file on the server and is refused unless "
                f"{SERVER_PATHS_ENV} is enabled (\"allow_server_paths\": true in the "
                "launcher configuration)."
            )
        if self.continuation_prefix_frames is not None:
            if self.video_path is None:
                raise ValueError(
                    "continuation_prefix_frames requires video_path for LTX-2."
                )
            # type() rather than isinstance(): bool is an int, and 17.0 or "17"
            # must not slip through to the frame arithmetic.
            if (
                type(self.continuation_prefix_frames) is not int
                or self.continuation_prefix_frames < 1
                or self.continuation_prefix_frames % 8 != 1
            ):
                raise ValueError(
                    "continuation_prefix_frames must be a positive integer satisfying "
                    "frames % 8 == 1."
                )
            if self.continuation_prefix_frames >= self.num_frames:
                raise ValueError(
                    "continuation_prefix_frames must be smaller than num_frames."
                )
        if self.video_path is not None:
            video_paths = (
                self.video_path
                if isinstance(self.video_path, list)
                else [self.video_path]
            )
            if (
                len(video_paths) != 1
                or not isinstance(video_paths[0], str)
                or not video_paths[0].strip()
            ):
                raise ValueError(
                    "LTX-2 temporal continuation accepts exactly one non-empty video_path."
                )
            if self.image_path is not None or self.reference_image_paths is not None:
                raise ValueError(
                    "video_path is mutually exclusive with image_path and "
                    "reference_image_paths for LTX-2."
                )
        if self.reference_image_paths is None:
            return
        if self.image_path is not None:
            raise ValueError(
                "reference_image_paths and image_path are mutually exclusive for LTX-2."
            )
        if not isinstance(self.reference_image_paths, list) or not (
            1 <= len(self.reference_image_paths) <= 5
        ):
            raise ValueError(
                "reference_image_paths must contain between 1 and 5 paths."
            )
        if any(
            not isinstance(path, str) or not path.strip()
            for path in self.reference_image_paths
        ):
            raise ValueError("reference_image_paths must contain non-empty strings.")

    def _validate_with_pipeline_config(self, pipeline_config) -> None:
        if self.reference_image_paths is None:
            super()._validate_with_pipeline_config(pipeline_config)
            return
        if not pipeline_config.task_type.accepts_image_input():
            raise ValueError(
                f"reference_image_paths is not supported for "
                f"{pipeline_config.task_type.name} models."
            )


@dataclasses.dataclass
class LTX23SamplingParams(LTX2SamplingParams):
    """Sampling parameters matching official LTX-2.3 one-stage defaults."""

    seed: int = 42
    generator_device: str = "cuda"
    guidance_scale: float = 3.0
    num_inference_steps: int = 30

    video_cfg_scale: float = 3.0
    video_stg_scale: float = 1.0
    video_rescale_scale: float = 0.7
    video_modality_scale: float = 3.0
    video_skip_step: int = 0
    video_stg_blocks: list[int] = field(default_factory=lambda: [28])

    audio_cfg_scale: float = 7.0
    audio_stg_scale: float = 1.0
    audio_rescale_scale: float = 0.7
    audio_modality_scale: float = 3.0
    audio_skip_step: int = 0
    audio_stg_blocks: list[int] = field(default_factory=lambda: [28])
    skip_v2a_cross_attn_for_video_gt: bool = False

    def build_request_extra(self) -> dict[str, Any]:
        extra = super().build_request_extra()
        # RL rollout uses the official CFG path (guidance_scale=1, no guider).
        if self.rollout:
            return extra
        extra["ltx2_stage1_guider_params"] = {
            "video_cfg_scale": self.video_cfg_scale,
            "video_stg_scale": self.video_stg_scale,
            "video_rescale_scale": self.video_rescale_scale,
            "video_modality_scale": self.video_modality_scale,
            "video_skip_step": self.video_skip_step,
            "video_stg_blocks": self.video_stg_blocks,
            "audio_cfg_scale": self.audio_cfg_scale,
            "audio_stg_scale": self.audio_stg_scale,
            "audio_rescale_scale": self.audio_rescale_scale,
            "audio_modality_scale": self.audio_modality_scale,
            "audio_skip_step": self.audio_skip_step,
            "audio_stg_blocks": self.audio_stg_blocks,
        }
        if self.skip_v2a_cross_attn_for_video_gt:
            extra["ltx2_skip_v2a_cross_attn_for_video_gt"] = True
        return extra


@dataclasses.dataclass
class LTX23HQSamplingParams(LTX23SamplingParams):
    """Sampling parameters matching official LTX-2.3 HQ two-stage defaults."""

    height: int = 1088
    width: int = 1920
    num_inference_steps: int = 15
    distilled_lora_strength_stage_1: float = 0.25
    distilled_lora_strength_stage_2: float = 0.5

    video_cfg_scale: float = 3.0
    video_stg_scale: float = 0.0
    video_rescale_scale: float = 0.45
    video_modality_scale: float = 3.0
    video_skip_step: int = 0
    video_stg_blocks: list[int] = field(default_factory=list)

    audio_cfg_scale: float = 7.0
    audio_stg_scale: float = 0.0
    audio_rescale_scale: float = 1.0
    audio_modality_scale: float = 3.0
    audio_skip_step: int = 0
    audio_stg_blocks: list[int] = field(default_factory=list)

    def build_request_extra(self) -> dict[str, Any]:
        extra = super().build_request_extra()
        extra["ltx2_distilled_lora_strength_stage_1"] = float(
            self.distilled_lora_strength_stage_1
        )
        extra["ltx2_distilled_lora_strength_stage_2"] = float(
            self.distilled_lora_strength_stage_2
        )
        return extra
