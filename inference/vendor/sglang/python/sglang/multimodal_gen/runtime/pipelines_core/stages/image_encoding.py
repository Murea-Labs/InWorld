# Modified by the InWorld authors from SGLang; see inference/docs/provenance.md.
# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
"""
Image encoding stages for I2V diffusion pipelines.

This module contains implementations of image encoding stages for diffusion pipelines.
"""

import inspect
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import PIL
import PIL.Image
import torch
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from sglang.multimodal_gen.configs.pipeline_configs.base import TextConditioningOutput
from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import (
    ComponentUse,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
    configure_layerwise_offload_modules,
)
from sglang.multimodal_gen.runtime.models.vaes.common import ParallelTiledVAE
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    StageValidators as V,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    VerificationResult,
)
from sglang.multimodal_gen.runtime.server_args import (
    ServerArgs,
    is_ltx2_two_stage_pipeline_name,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.precision import (
    align_tensor_to_module_dtype,
    autocast_context,
    autocast_enabled,
    resolve_precision,
    temporary_module_dtype,
)
from sglang.multimodal_gen.runtime.utils.vision import (
    normalize,
    numpy_to_pt,
    pil_to_numpy,
)

logger = init_logger(__name__)


@dataclass(frozen=True)
class ImageEncodingFingerprint:
    image_source: Any
    prompt: Any
    negative_prompt: Any
    do_classifier_free_guidance: bool
    height: int | None
    width: int | None
    num_frames: int | None


@dataclass(frozen=True)
class LTX2ImageEncodingFingerprint:
    image_source: Any
    height: int | None
    width: int | None
    num_frames: int | None
    latent_dtype: str
    condition_encoder_subdir: str
    encode_sample_mode: str


@dataclass(frozen=True)
class ImageVAEEncodingFingerprint:
    image_source: Any
    height: int | None
    width: int | None
    num_frames: int | None
    encode_sample_mode: str
    vae_precision: Any
    vae_tiling: bool


def _freeze_image_source_value(value):
    """Build a hashable identity fragment for image inputs.

    Image inputs are often PIL/numpy/tensor objects. For file paths we can use
    the path value; for in-memory objects we only dedup when the exact same
    object instance is shared by multiple requests. This avoids expensive image
    hashing and avoids treating two mutable image objects as equivalent just
    because they currently have the same shape.
    """
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_image_source_value(item) for item in value)
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    return ("object", id(value))


def _build_image_source_fingerprint(batch: Req, *, prefer_vae_image: bool = False):
    """Return the image input fragment used by image encoding fingerprints."""
    if batch.image_path is not None:
        return ("path", PipelineStage.freeze_for_dedup(batch.image_path))
    image = (
        batch.vae_image if prefer_vae_image and batch.vae_image is not None else None
    )
    if image is None:
        image = batch.condition_image
    return ("image", _freeze_image_source_value(image))


class ImageEncodingStage(PipelineStage):
    """
    Stage for encoding image prompts into embeddings for diffusion models.

    This stage handles the encoding of image prompts into the embedding space
    expected by the diffusion model.
    """

    deduplicated_output_fields = (
        "image_embeds",
        "prompt_embeds",
        "negative_prompt_embeds",
        "prompt_embeds_mask",
        "negative_prompt_embeds_mask",
        "prompt_seq_lens",
        "negative_prompt_seq_lens",
    )

    def __init__(
        self,
        image_processor,
        image_encoder=None,
        text_encoder=None,
    ) -> None:
        """
        Initialize the prompt encoding stage.

        Args:
            text_encoder: An encoder to encode input_ids and pixel values
        """
        super().__init__()
        self.image_processor = image_processor
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder

    def component_uses(
        self, server_args: ServerArgs, stage_name: str | None = None
    ) -> list[ComponentUse]:
        stage_name = self._component_stage_name(stage_name)
        uses = []
        if self.image_encoder is not None:
            uses.append(ComponentUse(stage_name, "image_encoder"))
        if self.text_encoder is not None:
            uses.append(ComponentUse(stage_name, "text_encoder"))
        return uses

    def encoding_image_edit(self, outputs, image_inputs, pipeline_config):
        """Encode image-edit text features via pipeline-configured postprocess hook."""
        postprocess_funcs = getattr(pipeline_config, "postprocess_text_funcs", ())
        if not postprocess_funcs or not callable(postprocess_funcs[0]):
            raise ValueError(
                "Image-edit pipeline requires a callable postprocess_text_funcs[0]."
            )

        return postprocess_funcs[0](outputs, image_inputs)

    @staticmethod
    def _split_text_conditioning_output(output):
        if isinstance(output, TextConditioningOutput):
            return (
                output.prompt_embeds,
                output.prompt_embeds_mask,
                output.prompt_seq_lens,
            )
        return output, None, None

    @staticmethod
    def _full_text_seq_lens(prompt_embeds: torch.Tensor) -> list[int]:
        if prompt_embeds.ndim == 2:
            return [int(prompt_embeds.shape[0])]
        return [int(prompt_embeds.shape[1])] * int(prompt_embeds.shape[0])

    @staticmethod
    def _default_text_mask(prompt_embeds: torch.Tensor) -> torch.Tensor:
        if prompt_embeds.ndim == 2:
            shape = (1, prompt_embeds.shape[0])
        else:
            shape = prompt_embeds.shape[:2]
        return torch.ones(shape, dtype=torch.bool, device=prompt_embeds.device)

    @torch.no_grad()
    def forward(
        self,
        batch: Req,
        server_args: ServerArgs,
    ) -> Req:
        """
        Encode the prompt into image encoder hidden states.
        """

        if batch.condition_image is None:
            return batch
        cuda_device = get_local_torch_device()

        image_processor_kwargs = (
            server_args.pipeline_config.prepare_image_processor_kwargs(batch)
        )
        per_prompt_images = image_processor_kwargs.pop("per_prompt_images", None)
        texts = image_processor_kwargs.pop("text", None)

        if per_prompt_images is None:
            per_prompt_images = [batch.condition_image]
            texts = [None] if texts is None else texts

        all_prompt_embeds = []
        all_neg_prompt_embeds = []
        all_prompt_embeds_masks = []
        all_neg_prompt_embeds_masks = []
        all_prompt_seq_lens = []
        all_neg_prompt_seq_lens = []

        image_processor_call_params = inspect.signature(
            self.image_processor.__call__
        ).parameters
        image_processor_kwargs = {
            k: v
            for k, v in image_processor_kwargs.items()
            if k in image_processor_call_params
        }

        for idx, prompt_images in enumerate(per_prompt_images):
            if not prompt_images:
                continue

            cur_kwargs = image_processor_kwargs.copy()
            if texts and idx < len(texts) and "text" in image_processor_call_params:
                cur_kwargs["text"] = [texts[idx]]

            image_inputs = self.image_processor(
                images=prompt_images, return_tensors="pt", **cur_kwargs
            ).to(cuda_device)

            if self.image_encoder:
                # if an image encoder is provided
                with self.use_declared_component(
                    component_name="image_encoder",
                    module=self.image_encoder,
                ) as image_encoder:
                    assert image_encoder is not None
                    self.image_encoder = image_encoder
                    if hasattr(image_inputs, "pixel_values") and isinstance(
                        image_inputs.pixel_values, torch.Tensor
                    ):
                        image_inputs["pixel_values"] = align_tensor_to_module_dtype(
                            image_inputs.pixel_values,
                            self.image_encoder,
                            device=cuda_device,
                        )
                    with set_forward_context(current_timestep=0, attn_metadata=None):
                        outputs = self.image_encoder(
                            **image_inputs,
                            **server_args.pipeline_config.image_encoder_extra_args,
                        )
                        image_embeds = server_args.pipeline_config.postprocess_image(
                            outputs
                        )
                batch.image_embeds.append(image_embeds)
            elif self.text_encoder:
                # if a text encoder is provided, e.g. Qwen-Image-Edit
                # 1. neg prompt embeds
                if batch.do_classifier_free_guidance:
                    neg_image_processor_kwargs = (
                        server_args.pipeline_config.prepare_image_processor_kwargs(
                            batch, neg=True
                        )
                    )
                    neg_image_processor_kwargs.pop("per_prompt_images", None)
                    neg_texts = neg_image_processor_kwargs.pop("text", None)
                    if neg_texts and idx < len(neg_texts):
                        neg_image_processor_kwargs["text"] = [neg_texts[idx]]
                    neg_image_inputs = self.image_processor(
                        images=prompt_images,
                        return_tensors="pt",
                        **neg_image_processor_kwargs,
                    ).to(cuda_device)

                with self.use_declared_component(
                    component_name="text_encoder",
                    module=self.text_encoder,
                ) as text_encoder:
                    assert text_encoder is not None
                    self.text_encoder = text_encoder
                    if hasattr(image_inputs, "pixel_values") and isinstance(
                        image_inputs.pixel_values, torch.Tensor
                    ):
                        image_inputs["pixel_values"] = align_tensor_to_module_dtype(
                            image_inputs.pixel_values,
                            self.text_encoder,
                            device=cuda_device,
                        )
                    if (
                        batch.do_classifier_free_guidance
                        and hasattr(neg_image_inputs, "pixel_values")
                        and isinstance(neg_image_inputs.pixel_values, torch.Tensor)
                    ):
                        neg_image_inputs["pixel_values"] = align_tensor_to_module_dtype(
                            neg_image_inputs.pixel_values,
                            self.text_encoder,
                            device=cuda_device,
                        )
                    with set_forward_context(current_timestep=0, attn_metadata=None):
                        outputs = self.text_encoder(
                            input_ids=image_inputs.input_ids,
                            attention_mask=image_inputs.attention_mask,
                            pixel_values=image_inputs.pixel_values,
                            image_grid_thw=image_inputs.image_grid_thw,
                            output_hidden_states=True,
                            use_cache=False,
                        )
                        if batch.do_classifier_free_guidance:
                            neg_outputs = self.text_encoder(
                                input_ids=neg_image_inputs.input_ids,
                                attention_mask=neg_image_inputs.attention_mask,
                                pixel_values=neg_image_inputs.pixel_values,
                                image_grid_thw=neg_image_inputs.image_grid_thw,
                                output_hidden_states=True,
                                use_cache=False,
                            )

                prompt_embeds, prompt_embeds_mask, prompt_seq_lens = (
                    self._split_text_conditioning_output(
                        self.encoding_image_edit(
                            outputs, image_inputs, server_args.pipeline_config
                        )
                    )
                )
                all_prompt_embeds.append(prompt_embeds)
                all_prompt_embeds_masks.append(prompt_embeds_mask)
                all_prompt_seq_lens.extend(
                    prompt_seq_lens
                    if prompt_seq_lens is not None
                    else self._full_text_seq_lens(prompt_embeds)
                )
                if batch.do_classifier_free_guidance:
                    neg_prompt_embeds, neg_prompt_embeds_mask, neg_prompt_seq_lens = (
                        self._split_text_conditioning_output(
                            self.encoding_image_edit(
                                neg_outputs,
                                neg_image_inputs,
                                server_args.pipeline_config,
                            )
                        )
                    )
                    all_neg_prompt_embeds.append(neg_prompt_embeds)
                    all_neg_prompt_embeds_masks.append(neg_prompt_embeds_mask)
                    all_neg_prompt_seq_lens.extend(
                        neg_prompt_seq_lens
                        if neg_prompt_seq_lens is not None
                        else self._full_text_seq_lens(neg_prompt_embeds)
                    )

        if all_prompt_embeds:
            batch.prompt_embeds.append(torch.cat(all_prompt_embeds, dim=0))
            if batch.prompt_embeds_mask is None:
                batch.prompt_embeds_mask = []
            batch.prompt_embeds_mask.append(
                torch.cat(
                    [
                        (
                            mask
                            if mask is not None
                            else self._default_text_mask(prompt_embeds)
                        )
                        for prompt_embeds, mask in zip(
                            all_prompt_embeds, all_prompt_embeds_masks, strict=True
                        )
                    ],
                    dim=0,
                )
            )
            if batch.prompt_seq_lens is None:
                batch.prompt_seq_lens = []
            batch.prompt_seq_lens.append(all_prompt_seq_lens)
        if all_neg_prompt_embeds:
            batch.negative_prompt_embeds.append(torch.cat(all_neg_prompt_embeds, dim=0))
            if batch.negative_prompt_embeds_mask is None:
                batch.negative_prompt_embeds_mask = []
            batch.negative_prompt_embeds_mask.append(
                torch.cat(
                    [
                        (
                            mask
                            if mask is not None
                            else self._default_text_mask(neg_prompt_embeds)
                        )
                        for neg_prompt_embeds, mask in zip(
                            all_neg_prompt_embeds,
                            all_neg_prompt_embeds_masks,
                            strict=True,
                        )
                    ],
                    dim=0,
                )
            )
            if batch.negative_prompt_seq_lens is None:
                batch.negative_prompt_seq_lens = []
            batch.negative_prompt_seq_lens.append(all_neg_prompt_seq_lens)

        return batch

    def build_dedup_fingerprint(
        self, batch: Req, server_args: ServerArgs
    ) -> ImageEncodingFingerprint:
        return ImageEncodingFingerprint(
            image_source=_build_image_source_fingerprint(batch),
            prompt=self.freeze_for_dedup(batch.prompt),
            negative_prompt=self.freeze_for_dedup(batch.negative_prompt),
            do_classifier_free_guidance=bool(batch.do_classifier_free_guidance),
            height=batch.height,
            width=batch.width,
            num_frames=batch.num_frames,
        )

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        """Verify image encoding stage inputs."""
        result = VerificationResult()
        if batch.debug:
            logger.debug(f"{batch.condition_image=}")
            logger.debug(f"{batch.image_embeds=}")
        result.add_check("pil_image", batch.condition_image, V.not_none)
        result.add_check("image_embeds", batch.image_embeds, V.is_list)
        return result

    def verify_output(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        """Verify image encoding stage outputs."""
        result = VerificationResult()
        # result.add_check("image_embeds", batch.image_embeds, V.list_of_tensors_dims(3))
        return result


class LTX2ImageEncodingStage(PipelineStage):
    """Encode ``batch.image_path`` into packed token latents for LTX-2 TI2V.

    Runs before denoising. Populates:
      - ``batch.condition_image`` (resized PIL image)
      - ``batch.image_latent``    (packed [B, S0, D] token latents)
      - ``batch.ltx2_num_image_tokens``
    """

    deduplicated_output_fields = (
        "condition_image",
        "image_latent",
        "ltx2_num_image_tokens",
        "ltx2_conditioning_mode",
        "ltx2_prefix_num_frames",
        "ltx2_prefix_num_latent_frames",
        "ltx2_reference_latents",
        "ltx2_reference_latent_shapes",
    )

    def __init__(self, vae=None, **kwargs) -> None:
        super().__init__()
        self.vae = vae
        self._condition_image_encoder = None
        self._condition_image_encoder_dir = None

    def component_uses(
        self, server_args: ServerArgs, stage_name: str | None = None
    ) -> list[ComponentUse]:
        arch_config = server_args.pipeline_config.vae_config.arch_config
        encoder_subdir = str(getattr(arch_config, "condition_encoder_subdir", ""))
        stage_name = self._component_stage_name(stage_name)
        if encoder_subdir:
            return [
                ComponentUse(
                    stage_name,
                    "condition_image_encoder",
                )
            ]
        if self.vae is None:
            return []
        return [ComponentUse(stage_name, "vae")]

    # -- lazy condition encoder (LTX-2.3) --------------------------------

    def _ensure_condition_image_encoder(self, server_args: ServerArgs) -> bool:
        """Load LTX-2.3 condition-encoder weights on first call. Returns True if available."""
        import json
        import os

        arch_config = server_args.pipeline_config.vae_config.arch_config
        encoder_subdir = str(getattr(arch_config, "condition_encoder_subdir", ""))
        if not encoder_subdir:
            return False

        vae_model_path = server_args.model_paths["vae"]
        encoder_dir = os.path.join(vae_model_path, encoder_subdir)
        if (
            self._condition_image_encoder is not None
            and self._condition_image_encoder_dir == encoder_dir
        ):
            return True

        config_path = os.path.join(encoder_dir, "config.json")
        weights_path = os.path.join(encoder_dir, "model.safetensors")
        if not os.path.exists(config_path) or not os.path.exists(weights_path):
            raise ValueError(
                f"LTX-2 condition encoder files not found under {encoder_dir}"
            )

        from safetensors.torch import load_file as safetensors_load_file
        from sglang.multimodal_gen.runtime.models.vaes.ltx_2_3_condition_encoder import (
            LTX23VideoConditionEncoder,
        )

        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
        self._condition_image_encoder = LTX23VideoConditionEncoder(config)
        self._condition_image_encoder.load_state_dict(
            safetensors_load_file(weights_path), strict=True
        )
        self._condition_image_encoder_dir = encoder_dir
        if server_args.should_configure_layerwise_offload_for_lazy_component(
            "condition_image_encoder"
        ):
            modules = {"condition_image_encoder": self._condition_image_encoder}
            configure_layerwise_offload_modules(
                modules,
                server_args,
                component_names=(
                    None
                    if server_args.component_residency is not None
                    else server_args.layerwise_offload_components
                ),
                warn_missing=False,
            )
        return True

    # -- image preprocessing ---------------------------------------------

    # Conditioning images are re-compressed to match training: CRF 33 for
    # LTX-2 / 2.3, 18 for LTX-2.5. Like upstream, keyed off the text-encoder
    # generation -- the only signal that separates them.
    _DEFAULT_IMAGE_CRF = 33
    _LTX_2_5_IMAGE_CRF = 18
    _GEMMA_4_MODEL_TYPES = ("gemma4_unified", "gemma4")

    @classmethod
    def _resolve_image_conditioning_crf(cls, server_args: ServerArgs) -> int:
        text_encoder_configs = server_args.pipeline_config.text_encoder_configs
        for encoder_config in text_encoder_configs:
            if encoder_config.prefix in ("gemma_4_unified", "gemma_4"):
                return cls._LTX_2_5_IMAGE_CRF
        return cls._DEFAULT_IMAGE_CRF

    @staticmethod
    def _apply_video_codec_compression(
        img_array: np.ndarray, crf: int = 33
    ) -> np.ndarray:
        """Single H.264 frame round-trip to simulate compression artifacts."""
        from io import BytesIO

        import av

        if crf == 0:
            return img_array
        h, w = img_array.shape[0] // 2 * 2, img_array.shape[1] // 2 * 2
        img_array = img_array[:h, :w]
        buf = BytesIO()
        container = av.open(buf, mode="w", format="mp4")
        stream = container.add_stream(
            "libx264", rate=1, options={"crf": str(crf), "preset": "veryfast"}
        )
        stream.height, stream.width = h, w
        frame = av.VideoFrame.from_ndarray(img_array, format="rgb24").reformat(
            format="yuv420p"
        )
        container.mux(stream.encode(frame))
        container.mux(stream.encode())
        container.close()
        buf.seek(0)
        container = av.open(buf)
        decoded = next(container.decode(container.streams.video[0]))
        container.close()
        return decoded.to_ndarray(format="rgb24")

    @staticmethod
    def _pil_to_video_tensor(
        img: PIL.Image.Image,
        *,
        width: int,
        height: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Scale-to-cover, center-crop, normalize to [1, C, 1, H, W] in [-1, 1]."""
        import math

        arr = np.array(img).astype(np.uint8)[..., :3]
        t = (
            torch.from_numpy(arr.astype(np.float32))
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(device=device)
        )
        src_h, src_w = t.shape[2], t.shape[3]
        scale = max(height / src_h, width / src_w)
        new_h, new_w = math.ceil(src_h * scale), math.ceil(src_w * scale)
        t = torch.nn.functional.interpolate(
            t, size=(new_h, new_w), mode="bilinear", align_corners=False
        )
        top, left = (new_h - height) // 2, (new_w - width) // 2
        t = t[:, :, top : top + height, left : left + width]
        return ((t / 127.5 - 1.0).to(dtype=dtype)).unsqueeze(2)

    @staticmethod
    def _align_up(value: float, stride: int) -> int:
        return max(stride, int(math.ceil(float(value) / stride)) * stride)

    @staticmethod
    def _align_down(value: float, stride: int) -> int:
        return max(stride, int(math.floor(float(value) / stride)) * stride)

    @classmethod
    def _stride_minimal_reference_size(
        cls, raw_width: int, raw_height: int, *, stride: int
    ) -> tuple[int, int]:
        """Match the training-time ``ref_image_size_for_stride`` exactly."""
        if raw_width <= 0 or raw_height <= 0 or stride <= 0:
            raise ValueError(
                "Invalid ordered-reference resize request: "
                f"raw_width={raw_width}, raw_height={raw_height}, stride={stride}."
            )
        target_area = raw_width * raw_height
        aspect = raw_width / raw_height
        nominal_h = math.sqrt(target_area / aspect)
        nominal_w = nominal_h * aspect
        candidates: set[tuple[int, int]] = set()
        max_dim = max(cls._align_up(max(nominal_h, nominal_w) * 4, stride), stride * 4)
        for height in range(stride, max_dim + stride, stride):
            width_float = height * aspect
            candidates.add((height, cls._align_down(width_float, stride)))
            candidates.add((height, cls._align_up(width_float, stride)))
        for width in range(stride, max_dim + stride, stride):
            height_float = width / aspect
            candidates.add((cls._align_down(height_float, stride), width))
            candidates.add((cls._align_up(height_float, stride), width))
        for height in (
            cls._align_down(nominal_h, stride),
            cls._align_up(nominal_h, stride),
        ):
            for width in (
                cls._align_down(nominal_w, stride),
                cls._align_up(nominal_w, stride),
            ):
                candidates.add((height, width))

        def score(size: tuple[int, int]) -> tuple[float, float, float, float, int]:
            height, width = size
            area = height * width
            aspect_error = abs(math.log((width / height) / aspect))
            area_error = abs(math.log(area / target_area))
            dimension_error = abs(math.log(width / raw_width)) + abs(
                math.log(height / raw_height)
            )
            return (
                aspect_error + area_error,
                area_error,
                aspect_error,
                dimension_error,
                area,
            )

        return min(candidates, key=score)

    @classmethod
    def _pil_to_ordered_reference_tensor(
        cls,
        img: PIL.Image.Image,
        *,
        stride: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms.functional import resize, to_tensor

        img = img.convert("RGB")
        raw_width, raw_height = img.size
        height, width = cls._stride_minimal_reference_size(
            raw_width, raw_height, stride=stride
        )
        tensor = to_tensor(img).unsqueeze(0)
        if (height, width) != (raw_height, raw_width):
            tensor = resize(
                tensor,
                size=[height, width],
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            )
        tensor = tensor.clamp_(0.0, 1.0)
        tensor = tensor.sub_(0.5).div_(0.5)
        return tensor.to(device=device, dtype=dtype).unsqueeze(2), (height, width)

    # -- encode paths ----------------------------------------------------

    def _vae_encode(
        self,
        video_condition: torch.Tensor,
        server_args: ServerArgs,
        generator: torch.Generator | None,
        *,
        force_direct: bool = False,
    ) -> torch.Tensor:
        """VAE encode → sample → per-channel normalize (LTX-2 convention)."""
        vae_dtype = resolve_precision(
            server_args, "vae", precision_attr="vae_precision"
        )
        vae_autocast_enabled = autocast_enabled(vae_dtype, server_args.disable_autocast)
        restore_tiling = False

        if force_direct:
            if not hasattr(self.vae, "disable_tiling"):
                raise ValueError(
                    "Direct LTX-2 reference encoding requires VAE tiling controls."
                )
            restore_tiling = bool(getattr(self.vae, "use_tiling", False))
            self.vae.disable_tiling()

        try:
            with autocast_context(vae_dtype, server_args.disable_autocast):
                try:
                    if server_args.pipeline_config.vae_tiling and not force_direct:
                        self.vae.enable_tiling()
                except Exception:
                    pass
                should_cast_vae = not vae_autocast_enabled
                if not vae_autocast_enabled:
                    video_condition = video_condition.to(vae_dtype)
                with temporary_module_dtype(
                    self.vae, vae_dtype, enabled=should_cast_vae
                ) as vae:
                    latent_dist = vae.encode(video_condition)
                if isinstance(latent_dist, AutoencoderKLOutput):
                    latent_dist = latent_dist.latent_dist
        finally:
            if restore_tiling:
                self.vae.enable_tiling()

        mode = server_args.pipeline_config.vae_config.encode_sample_mode()
        if mode == "argmax":
            latent = latent_dist.mode()
        elif mode == "sample":
            if generator is None:
                raise ValueError("Generator must be provided for VAE sampling.")
            latent = latent_dist.sample(generator)
        else:
            raise ValueError(f"Unsupported encode_sample_mode: {mode}")

        mean = self.vae.latents_mean.view(1, -1, 1, 1, 1).to(latent)
        std = self.vae.latents_std.view(1, -1, 1, 1, 1).to(latent)
        return (latent - mean) / std

    def _condition_encode(
        self, video_condition: torch.Tensor, server_args: ServerArgs
    ) -> torch.Tensor:
        """LTX-2.3 condition-image encoder path (bypasses VAE)."""
        vae_dtype = resolve_precision(
            server_args, "vae", precision_attr="vae_precision"
        )

        with autocast_context(vae_dtype, server_args.disable_autocast):
            return self._condition_image_encoder(video_condition)

    @staticmethod
    def _normalize_ltx2_image_paths(image_path: str | list[str]) -> list[str]:
        image_paths = image_path if isinstance(image_path, list) else [image_path]
        if len(image_paths) > 2:
            raise ValueError(
                "LTX-2 TI2V currently supports at most two conditioning images "
                "([first_frame, last_frame])."
            )
        return image_paths

    @staticmethod
    def _normalize_ltx2_image_latents(
        image_latent: torch.Tensor | list[torch.Tensor] | None,
    ) -> list[torch.Tensor]:
        if image_latent is None:
            return []
        return image_latent if isinstance(image_latent, list) else [image_latent]

    @staticmethod
    def _normalize_ltx2_video_path(video_path: str | list[str]) -> str:
        video_paths = video_path if isinstance(video_path, list) else [video_path]
        if len(video_paths) != 1 or not isinstance(video_paths[0], str):
            raise ValueError(
                "LTX-2 temporal continuation accepts exactly one video_path."
            )
        path = video_paths[0].strip()
        if not path:
            raise ValueError("LTX-2 temporal continuation video_path is empty.")
        return path

    @staticmethod
    def _resolve_ltx2_prefix_frame_count(
        *, decoded_frames: int, target_frames: int, requested_frames: int | None
    ) -> int:
        if decoded_frames <= 0:
            raise ValueError("LTX-2 temporal continuation decoded no prefix frames.")
        if target_frames <= 1:
            raise ValueError(
                "LTX-2 temporal continuation requires a multi-frame target."
            )
        if requested_frames is not None:
            if requested_frames % 8 != 1:
                raise ValueError(
                    "continuation_prefix_frames must satisfy frames % 8 == 1."
                )
            if requested_frames > decoded_frames:
                raise ValueError(
                    "continuation_prefix_frames exceeds the decoded input video length: "
                    f"requested={requested_frames}, decoded={decoded_frames}."
                )
            prefix_frames = int(requested_frames)
        else:
            usable = min(int(decoded_frames), int(target_frames) - 1)
            prefix_frames = ((usable - 1) // 8) * 8 + 1
        if prefix_frames >= int(target_frames):
            raise ValueError(
                "LTX-2 temporal continuation prefix must be shorter than the target: "
                f"prefix={prefix_frames}, target={target_frames}."
            )
        return prefix_frames

    @staticmethod
    def _ltx2_prefix_decode_limit(
        *, target_frames: int, requested_frames: int | None
    ) -> int:
        """How many frames of the input video the prefix can use at most.

        Decoding stops there, so a long parent video is not decoded whole. One frame
        more than requested is never needed: _resolve_ltx2_prefix_frame_count only
        checks that the video has at least the requested frames.
        """
        if requested_frames is not None:
            return max(1, int(requested_frames))
        return max(1, int(target_frames) - 1)

    @classmethod
    def _video_frames_to_tensor(
        cls,
        frames: list[PIL.Image.Image],
        *,
        width: int,
        height: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        encoded_frames = [
            cls._pil_to_video_tensor(
                frame.convert("RGB"),
                width=width,
                height=height,
                device=device,
                dtype=dtype,
            )
            for frame in frames
        ]
        return torch.cat(encoded_frames, dim=2)

    @torch.no_grad()
    def _encode_video_prefix(self, batch: Req, server_args: ServerArgs) -> Req:
        if is_ltx2_two_stage_pipeline_name(server_args.pipeline_class_name):
            raise ValueError(
                "LTX-2 temporal continuation currently supports one-stage inference only."
            )
        if (
            batch.image_path is not None
            or getattr(batch, "reference_image_paths", None) is not None
        ):
            raise ValueError(
                "video_path is mutually exclusive with image_path and "
                "reference_image_paths for LTX-2."
            )
        if self.vae is None:
            raise ValueError("VAE must be provided for LTX-2 temporal continuation.")
        if self._ensure_condition_image_encoder(server_args):
            raise ValueError(
                "LTX-2 temporal continuation requires the LTX-2.5 video VAE path."
            )

        from sglang.multimodal_gen.runtime.utils.vision import load_video

        video_path = self._normalize_ltx2_video_path(batch.video_path)
        requested_frames = getattr(batch, "continuation_prefix_frames", None)
        frames = load_video(
            video_path,
            max_frames=self._ltx2_prefix_decode_limit(
                target_frames=int(batch.num_frames), requested_frames=requested_frames
            ),
        )
        prefix_frames = self._resolve_ltx2_prefix_frame_count(
            decoded_frames=len(frames),
            target_frames=int(batch.num_frames),
            requested_frames=requested_frames,
        )
        frames = frames[:prefix_frames]

        device = get_local_torch_device()
        encode_dtype = batch.latents.dtype
        video_condition = self._video_frames_to_tensor(
            frames,
            width=int(batch.width),
            height=int(batch.height),
            device=device,
            dtype=encode_dtype,
        )
        batch.condition_image = frames[0].resize(
            (int(batch.width), int(batch.height)),
            resample=PIL.Image.Resampling.BILINEAR,
        )

        with self.use_declared_component(
            component_name="vae",
            module=self.vae,
            target_dtype=encode_dtype,
        ) as active_vae:
            assert active_vae is not None
            self.vae = active_vae
            latent = self._vae_encode(video_condition, server_args, batch.generator)

        packed = server_args.pipeline_config.maybe_pack_latents(
            latent, latent.shape[0], batch
        )
        if not (isinstance(packed, torch.Tensor) and packed.ndim == 3):
            raise ValueError("Expected packed LTX-2 continuation latents [B, S, D].")
        target_tokens = int(batch.latents.shape[1])
        if int(packed.shape[1]) >= target_tokens:
            raise ValueError(
                "LTX-2 temporal continuation prefix consumes the full target timeline: "
                f"prefix_tokens={packed.shape[1]}, target_tokens={target_tokens}."
            )

        batch.image_latent = packed
        batch.ltx2_num_image_tokens = int(packed.shape[1])
        batch.ltx2_conditioning_mode = "video_prefix"
        batch.ltx2_prefix_num_frames = int(prefix_frames)
        batch.ltx2_prefix_num_latent_frames = int(latent.shape[2])
        logger.info(
            "LTX2 temporal continuation: source=%s pixel_frames=%d "
            "latent_frames=%d prefix_tokens=%d target_tokens=%d",
            video_path,
            prefix_frames,
            int(latent.shape[2]),
            int(packed.shape[1]),
            target_tokens,
        )
        return batch

    @staticmethod
    def _dump_ordered_reference_latents(
        batch: Req,
        reference_entries: list[dict[str, Any]],
    ) -> None:
        dump_dir = os.environ.get("SGLANG_LTX2_REFERENCE_LATENT_DUMP_DIR")
        if not dump_dir:
            return
        if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
            return

        request_id = os.path.basename(str(batch.request_id or "request"))
        output_dir = Path(dump_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{request_id}.pt"
        temp_path = output_dir / f".{request_id}.pt.tmp.{os.getpid()}"
        torch.save(
            {
                "refs": reference_entries,
                "ref_count": len(reference_entries),
                "ref_resize_policy": "stride_minimal_resize",
            },
            temp_path,
        )
        os.replace(temp_path, output_path)

    @torch.no_grad()
    def _encode_ordered_references(
        self,
        batch: Req,
        server_args: ServerArgs,
        reference_paths: list[str],
    ) -> Req:
        if is_ltx2_two_stage_pipeline_name(server_args.pipeline_class_name):
            raise ValueError(
                "Ordered LTX-2 reference_image_paths currently support one-stage "
                "inference only."
            )
        if self.vae is None:
            raise ValueError("VAE must be provided for ordered LTX-2 references.")
        if self._ensure_condition_image_encoder(server_args):
            raise ValueError(
                "Ordered reference_image_paths require the LTX-2.5 VAE path; "
                "the LTX-2.3 condition encoder is not supported."
            )
        if server_args.pipeline_config.vae_config.encode_sample_mode() != "argmax":
            raise ValueError(
                "Ordered reference_image_paths require deterministic VAE argmax encoding."
            )

        from sglang.multimodal_gen.runtime.utils.vision import load_image

        device = get_local_torch_device()
        encode_dtype = batch.latents.dtype
        stride = int(server_args.pipeline_config.vae_scale_factor)
        packed_latents: list[torch.Tensor] = []
        latent_shapes: list[dict[str, int]] = []
        reference_entries: list[dict[str, Any]] = []

        with self.use_declared_component(
            component_name="vae",
            module=self.vae,
            target_dtype=encode_dtype,
        ) as active_vae:
            assert active_vae is not None
            self.vae = active_vae
            for path in reference_paths:
                image = load_image(path)
                raw_width, raw_height = image.size
                video_condition, resized_shape = self._pil_to_ordered_reference_tensor(
                    image,
                    stride=stride,
                    device=device,
                    dtype=encode_dtype,
                )
                latent = self._vae_encode(
                    video_condition,
                    server_args,
                    batch.generator,
                    force_direct=True,
                )
                if int(latent.shape[2]) != 1:
                    raise ValueError(
                        "Ordered references must encode to one latent frame, got "
                        f"shape={tuple(latent.shape)} for {path!r}."
                    )
                packed = server_args.pipeline_config.maybe_pack_latents(
                    latent, latent.shape[0], batch
                )
                if not (isinstance(packed, torch.Tensor) and packed.ndim == 3):
                    raise ValueError(
                        "Expected packed ordered-reference latents [B, S, D]."
                    )
                packed_latents.append(packed)
                latent_shapes.append(
                    {
                        "num_frames": int(latent.shape[2]),
                        "height": int(latent.shape[3]),
                        "width": int(latent.shape[4]),
                        "token_count": int(packed.shape[1]),
                        "resized_height": int(resized_shape[0]),
                        "resized_width": int(resized_shape[1]),
                    }
                )
                reference_entries.append(
                    {
                        "latents": latent.detach().squeeze(0).to(device="cpu"),
                        "num_frames": int(latent.shape[2]),
                        "height": int(latent.shape[3]),
                        "width": int(latent.shape[4]),
                        "fps": 1.0,
                        "source_ref_path": str(path),
                        "raw_height": int(raw_height),
                        "raw_width": int(raw_width),
                        "resized_height": int(resized_shape[0]),
                        "resized_width": int(resized_shape[1]),
                        "ref_resize_policy": "stride_minimal_resize",
                    }
                )

        batch.ltx2_reference_latents = packed_latents
        batch.ltx2_reference_latent_shapes = latent_shapes
        self._dump_ordered_reference_latents(batch, reference_entries)
        if batch.debug:
            logger.info(
                "LTX2 ordered references: count=%d token_counts=%s latent_shapes=%s",
                len(packed_latents),
                [int(latent.shape[1]) for latent in packed_latents],
                latent_shapes,
            )
        return batch

    # -- forward ---------------------------------------------------------

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        if batch.video_path is not None:
            if (
                getattr(batch, "ltx2_conditioning_mode", None) == "video_prefix"
                and batch.image_latent is not None
            ):
                return batch
            return self._encode_video_prefix(batch, server_args)
        reference_paths = getattr(batch, "reference_image_paths", None)
        if reference_paths is not None:
            if batch.image_path is not None:
                raise ValueError(
                    "reference_image_paths and image_path are mutually exclusive for LTX-2."
                )
            if getattr(batch, "ltx2_reference_latents", None) is not None:
                return batch
            return self._encode_ordered_references(
                batch, server_args, list(reference_paths)
            )
        if batch.image_path is None:
            return batch
        image_paths = self._normalize_ltx2_image_paths(batch.image_path)

        vae_sf = int(server_args.pipeline_config.vae_scale_factor)
        patch = int(server_args.pipeline_config.patch_size)
        expected_tokens = (int(batch.height) // vae_sf // patch) * (
            int(batch.width) // vae_sf // patch
        )
        if (
            batch.image_latent is not None
            and int(getattr(batch, "ltx2_num_image_tokens", 0)) > 0
        ):
            # Re-encode if resolution changed (e.g. two-stage upsample between stages)
            existing_latents = self._normalize_ltx2_image_latents(batch.image_latent)
            if len(existing_latents) == len(image_paths) and all(
                int(latent.shape[1]) == expected_tokens for latent in existing_latents
            ):
                return batch
            # Resolution or reference-count mismatch — clear and re-encode below
            batch.image_latent = None
            batch.ltx2_num_image_tokens = 0

        batch.ltx2_num_image_tokens = 0
        batch.image_latent = None

        if self.vae is None:
            raise ValueError("VAE must be provided for LTX-2 TI2V.")

        from sglang.multimodal_gen.runtime.utils.vision import load_image

        # 1. Load images, apply codec compression, resize for condition_image
        crf = self._resolve_image_conditioning_crf(server_args)
        conditioned_imgs = []
        for image_path in image_paths:
            img = load_image(image_path)
            arr = np.array(img).astype(np.uint8)[..., :3]
            arr = self._apply_video_codec_compression(arr, crf=crf)
            conditioned_img = PIL.Image.fromarray(arr)
            conditioned_imgs.append(conditioned_img)
        batch.condition_image = [
            img.resize(
                (int(batch.width), int(batch.height)),
                resample=PIL.Image.Resampling.BILINEAR,
            )
            for img in conditioned_imgs
        ]
        if len(batch.condition_image) == 1:
            batch.condition_image = batch.condition_image[0]

        # 2. Select encoder(s); residency manager moves it to device and dtype.
        use_condition_encoder = self._ensure_condition_image_encoder(server_args)

        device = get_local_torch_device()
        encode_dtype = batch.latents.dtype

        if use_condition_encoder:
            component_name = "condition_image_encoder"
            encoder = self._condition_image_encoder
        else:
            component_name = "vae"
            encoder = self.vae

        packed_latents = []
        with self.use_declared_component(
            component_name=component_name,
            module=encoder,
            target_dtype=encode_dtype,
        ) as active_encoder:
            assert active_encoder is not None
            if use_condition_encoder:
                self._condition_image_encoder = active_encoder
            else:
                self.vae = active_encoder

            for conditioned_img in conditioned_imgs:
                video_condition = self._pil_to_video_tensor(
                    conditioned_img,
                    width=int(batch.width),
                    height=int(batch.height),
                    device=device,
                    dtype=encode_dtype,
                )

                # 3. Encode
                if use_condition_encoder:
                    latent = self._condition_encode(video_condition, server_args).to(
                        dtype=encode_dtype
                    )
                else:
                    latent = self._vae_encode(
                        video_condition, server_args, batch.generator
                    )

                packed = server_args.pipeline_config.maybe_pack_latents(
                    latent, latent.shape[0], batch
                )
                if not (isinstance(packed, torch.Tensor) and packed.ndim == 3):
                    raise ValueError("Expected packed image latents [B, S0, D].")
                if int(packed.shape[1]) != expected_tokens:
                    raise ValueError(
                        f"LTX-2 conditioning token count mismatch: "
                        f"{packed.shape[1]=} {expected_tokens=}."
                    )
                packed_latents.append(packed)

        batch.image_latent = (
            packed_latents[0] if len(packed_latents) == 1 else packed_latents
        )
        batch.ltx2_num_image_tokens = int(packed_latents[0].shape[1])

        if batch.debug:
            logger.info(
                "LTX2 TI2V: %d tokens (shape=%s) for %sx%s",
                batch.ltx2_num_image_tokens,
                tuple(packed_latents[0].shape),
                batch.width,
                batch.height,
            )

        return batch

    def build_dedup_fingerprint(
        self, batch: Req, server_args: ServerArgs
    ) -> LTX2ImageEncodingFingerprint | int:
        if batch.video_path is not None:
            # Video decoding and long-prefix VAE encoding are request scoped.
            return id(batch)
        reference_paths = getattr(batch, "reference_image_paths", None)
        if reference_paths is not None:
            if getattr(batch, "ltx2_reference_latents", None) is not None:
                return id(batch)
            image_source = (
                "ordered_reference_paths",
                PipelineStage.freeze_for_dedup(reference_paths),
            )
        elif batch.image_path is None or batch.image_latent is not None:
            return id(batch)
        else:
            image_source = _build_image_source_fingerprint(batch)

        sample_mode = server_args.pipeline_config.vae_config.encode_sample_mode()
        arch_config = server_args.pipeline_config.vae_config.arch_config
        encoder_subdir = str(getattr(arch_config, "condition_encoder_subdir", ""))
        if not encoder_subdir and sample_mode == "sample":
            return id(batch)

        latent_dtype = batch.latents.dtype if batch.latents is not None else None
        return LTX2ImageEncodingFingerprint(
            image_source=image_source,
            height=batch.height,
            width=batch.width,
            num_frames=batch.num_frames,
            latent_dtype=str(latent_dtype),
            condition_encoder_subdir=encoder_subdir,
            encode_sample_mode=sample_mode,
        )


class ImageVAEEncodingStage(PipelineStage):
    """
    Stage for encoding pixel representations into latent space.

    This stage handles the encoding of pixel representations into the final
    input format (e.g., image_latents).
    """

    deduplicated_output_fields = (
        "image_latent",
        "condition_image_latent_ids",
        "vae_image_sizes",
    )

    def __init__(
        self,
        vae: ParallelTiledVAE,
        component_name: str = "vae",
        **kwargs,
    ) -> None:
        super().__init__()
        self.vae: ParallelTiledVAE = vae
        self.component_name = component_name

    def component_uses(
        self, server_args: ServerArgs, stage_name: str | None = None
    ) -> list[ComponentUse]:
        vae_dtype = resolve_precision(
            server_args, self.component_name, precision_attr="vae_precision"
        )
        stage_name = self._component_stage_name(stage_name)
        return [
            ComponentUse(
                stage_name,
                self.component_name,
                target_dtype=vae_dtype,
            )
        ]

    def forward(
        self,
        batch: Req,
        server_args: ServerArgs,
    ) -> Req:
        """
        Encode pixel representations into latent space.
        """

        if batch.condition_image is None:
            return batch

        num_frames = batch.num_frames

        images = (
            batch.vae_image if batch.vae_image is not None else batch.condition_image
        )
        if not isinstance(images, list):
            images = [images]

        all_image_latents = []
        prepare_condition_image_latent_ids = getattr(
            server_args.pipeline_config, "prepare_condition_image_latent_ids", None
        )
        condition_latents = [] if callable(prepare_condition_image_latent_ids) else None
        # Setup VAE precision from user policy.
        vae_dtype = resolve_precision(
            server_args, self.component_name, precision_attr="vae_precision"
        )
        vae_autocast_enabled = autocast_enabled(vae_dtype, server_args.disable_autocast)

        with self.use_declared_component(
            component_name=self.component_name,
            module=self.vae,
        ) as vae:
            assert vae is not None
            self.vae = vae

            for image in images:
                image = self.preprocess(
                    image,
                ).to(get_local_torch_device(), dtype=torch.float32)

                # (B, C, H, W) -> (B, C, 1, H, W)
                image = image.unsqueeze(2)

                if num_frames == 1:
                    video_condition = image
                else:
                    video_condition = torch.cat(
                        [
                            image,
                            image.new_zeros(
                                image.shape[0],
                                image.shape[1],
                                num_frames - 1,
                                image.shape[3],
                                image.shape[4],
                            ),
                        ],
                        dim=2,
                    )
                video_condition = video_condition.to(
                    device=get_local_torch_device(), dtype=torch.float32
                )

                # Encode Image
                with autocast_context(vae_dtype, server_args.disable_autocast):
                    if server_args.pipeline_config.vae_tiling:
                        self.vae.enable_tiling()
                    # if server_args.vae_sp:
                    #     self.vae.enable_parallel()
                    should_cast_vae = not vae_autocast_enabled
                    if not vae_autocast_enabled:
                        video_condition = video_condition.to(vae_dtype)
                    video_condition = server_args.pipeline_config.preprocess_vae_encode(
                        video_condition, self.vae
                    )
                    with temporary_module_dtype(
                        self.vae, vae_dtype, enabled=should_cast_vae
                    ) as vae:
                        latent_dist: DiagonalGaussianDistribution = vae.encode(
                            video_condition
                        )
                    # for auto_encoder from diffusers
                    if isinstance(latent_dist, AutoencoderKLOutput):
                        latent_dist = latent_dist.latent_dist

                generator = batch.generator
                if generator is None:
                    raise ValueError("Generator must be provided")

                sample_mode = (
                    server_args.pipeline_config.vae_config.encode_sample_mode()
                )

                latent_condition = self.retrieve_latents(
                    latent_dist, generator, sample_mode=sample_mode
                )
                latent_condition = server_args.pipeline_config.postprocess_vae_encode(
                    latent_condition, self.vae
                )
                normalized_latent_condition = (
                    server_args.pipeline_config.normalize_vae_encode(
                        latent_condition, self.vae
                    )
                )
                if normalized_latent_condition is None:
                    scaling_factor, shift_factor = (
                        server_args.pipeline_config.get_decode_scale_and_shift(
                            device=latent_condition.device,
                            dtype=latent_condition.dtype,
                            vae=self.vae,
                        )
                    )

                    latent_condition = self.scale_and_shift_encode_latents(
                        latent_condition, scaling_factor, shift_factor
                    )
                else:
                    latent_condition = normalized_latent_condition

                if condition_latents is not None:
                    condition_latents.append(latent_condition)

                image_latent = server_args.pipeline_config.postprocess_image_latent(
                    latent_condition, batch
                )
                all_image_latents.append(image_latent)

        batch.image_latent = torch.cat(all_image_latents, dim=1)
        if condition_latents is not None:
            prepare_condition_image_latent_ids(condition_latents, batch)

        return batch

    @staticmethod
    def scale_and_shift_encode_latents(
        latents: torch.Tensor, scaling_factor, shift_factor
    ) -> torch.Tensor:
        if shift_factor is not None:
            if isinstance(shift_factor, torch.Tensor):
                shift_factor = shift_factor.to(latents.device)
            latents -= shift_factor

        if isinstance(scaling_factor, torch.Tensor):
            scaling_factor = scaling_factor.to(latents.device)
        return latents * scaling_factor

    def build_dedup_fingerprint(
        self, batch: Req, server_args: ServerArgs
    ) -> ImageVAEEncodingFingerprint | int:
        if batch.condition_image is None:
            return id(batch)

        sample_mode = server_args.pipeline_config.vae_config.encode_sample_mode()
        if sample_mode == "sample":
            return id(batch)

        return ImageVAEEncodingFingerprint(
            image_source=_build_image_source_fingerprint(batch, prefer_vae_image=True),
            height=batch.height,
            width=batch.width,
            num_frames=batch.num_frames,
            encode_sample_mode=sample_mode,
            vae_precision=server_args.pipeline_config.vae_precision,
            vae_tiling=bool(server_args.pipeline_config.vae_tiling),
        )

    def retrieve_latents(
        self,
        encoder_output: DiagonalGaussianDistribution,
        generator: torch.Generator | None = None,
        sample_mode: str = "sample",
    ):
        if sample_mode == "sample":
            if hasattr(encoder_output, "latent_dist"):
                return encoder_output.latent_dist.sample(generator)
            if hasattr(encoder_output, "latent"):
                return encoder_output.latent
            if hasattr(encoder_output, "latents"):
                return encoder_output.latents
            return encoder_output.sample(generator)
        elif sample_mode == "argmax":
            if hasattr(encoder_output, "latent_dist"):
                return encoder_output.latent_dist.mode()
            if hasattr(encoder_output, "latent"):
                return encoder_output.latent
            if hasattr(encoder_output, "latents"):
                return encoder_output.latents
            return encoder_output.mode()
        else:
            raise AttributeError("Could not access latents of provided encoder_output")

    def preprocess(
        self,
        image: torch.Tensor | PIL.Image.Image,
    ) -> torch.Tensor:
        if isinstance(image, PIL.Image.Image):
            image = pil_to_numpy(image)  # to np
            image = numpy_to_pt(image)  # to pt

        do_normalize = True
        if image.min() < 0:
            do_normalize = False
        if do_normalize:
            image = normalize(image)

        return image

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        """Verify encoding stage inputs."""
        result = VerificationResult()

        assert batch.condition_image is None or (
            isinstance(batch.condition_image, PIL.Image.Image)
            or isinstance(batch.condition_image, torch.Tensor)
            or isinstance(batch.condition_image, list)
        )
        assert batch.height is not None and isinstance(batch.height, int)
        assert batch.width is not None and isinstance(batch.width, int)
        assert batch.num_frames is not None and isinstance(batch.num_frames, int)

        result.add_check("generator", batch.generator, V.generator_or_list_generators)
        result.add_check("height", batch.height, V.positive_int)
        result.add_check("width", batch.width, V.positive_int)
        result.add_check("num_frames", batch.num_frames, V.positive_int)
        return result

    def verify_output(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        """Verify encoding stage outputs."""
        result = VerificationResult()
        # result.add_check(
        #     "image_latent", batch.image_latent, [V.is_tensor, V.with_dims(5)]
        # )
        return result
