"""Image generation client (Gemini REST)."""

from __future__ import annotations

import base64
import itertools
import time
from typing import Any
from collections.abc import Sequence

import httpx

from murea_agent import config
from murea_agent.util import _say
from murea_agent import reqlog
from murea_agent.providers.llm import _post
from murea_agent.director.prompt_language import ensure_english_text

SUFFIX = {
    # Scene refs must be empty: people come from the character refs, and a scene
    # image with people in it fights them.
    "place": " No people, no text. Wide establishing view, 16:9.",
    # Character reference images are later composited into scenes by the video model. Keep the
    # appearance details of the prompt, but do not let its action or place wording become a
    # background competing with the place image; this applies to people, creatures and characters
    # discovered at run time alike.
    "subject": (
        " Show one complete full-body character or creature, centered and fully visible "
        "in a neutral standing pose. Use the description for its stable appearance. "
        "The composition consists of the entity against a uniform pure white (#FFFFFF) background."
    ),
    "object": (
        " Show one complete object, fully visible in three-quarter view, "
        "on a uniform pure white (#FFFFFF) background. Preserve its shape, "
        "materials, colors and functional parts."
    ),
}


def _reference_instruction(kind: str) -> str:
    """Keep a character's story setting out of its identity reference.

    A trailing white-background suffix is not enough when the description starts
    with e.g. "standing in an abandoned classroom". The resulting classroom then
    competes with the actual scene reference in the video model. Give the asset contract its
    own instruction role; scene generation must retain its environment.
    """
    if kind not in ("subject", "object"):
        return ""
    entity = "one complete inanimate object" if kind == "object" else "one complete character or creature"
    return (
        "Create isolated reference assets. Extract ONLY stable entity appearance "
        "from the target description and its identity references. Follow the requested "
        "role of attached images: identity references preserve that individual, but "
        "images identified as teammate comparisons are NOT identity references. "
        "For teammate comparisons, create the distinct new individual described "
        "in the request; retain only the shared art style, not the teammate identity. "
        "Discard all locations, scenery, "
        "architecture, story actions and environmental lighting. Preserve the "
        "entity's identity, intrinsic colors, materials and artistic medium. "
        f"The output contains {entity}, fully visible, against a uniform pure "
        "white (#FFFFFF) background, never a story scene. Use even neutral lighting."
    )


# Image models often return **JPEG** (`ff d8 ff`) even when PNG is expected. Saving and
# uploading it as image/png once caused a confusing invalid_image error. Detect from the bytes.
_MAGIC = (
    (b"\x89PNG\r\n\x1a\n", ".png", "image/png"),
    (b"\xff\xd8\xff", ".jpg", "image/jpeg"),
    (b"RIFF", ".webp", "image/webp"),  # RIFF....WEBP
    (b"GIF8", ".gif", "image/gif"),
)


def sniff(raw: bytes) -> tuple[str, str]:
    """(extension, content-type). Unknown bytes are treated as png, as before."""
    for magic, ext, mime in _MAGIC:
        if raw.startswith(magic):
            return ext, mime
    return ".png", "image/png"


class ImageRefused(RuntimeError):
    """The model returned no image. Usually safety filtering, not a server fault."""


# finishReason / blockReason -> a message a user can understand
_REASON = {
    "NO_IMAGE": (
        "模型这次没出图，多半是提示词触发了内容安全过滤",
        "The model returned no image, most likely a safety filter on the prompt",
    ),
    "SAFETY": ("提示词触发了内容安全过滤", "The prompt tripped the safety filter"),
    "PROHIBITED_CONTENT": ("提示词涉及被禁止的内容", "The prompt touches prohibited content"),
    "IMAGE_SAFETY": ("生成的图片没通过安全检查", "The generated image failed the safety check"),
    "RECITATION": ("内容与受版权保护的素材过于接近", "Too close to copyrighted material"),
    "MAX_TOKENS": ("提示词太长了", "The prompt is too long"),
    "BLOCKLIST": ("提示词命中了屏蔽词", "The prompt hit a blocked word"),
}


def _why(resp: dict[str, Any]) -> str:
    cands = resp.get("candidates") or [{}]
    reason = str(cands[0].get("finishReason") or "")
    block = str((resp.get("promptFeedback") or {}).get("blockReason") or "")
    why = _say(
        *(_REASON.get(reason) or _REASON.get(block) or ("模型没有返回图片", "The model returned no image"))
    )
    code = reason or block or "unknown"
    return _say(
        f"{why}（{code}）。换个说法再试 —— 避开暴力、血腥、真人姓名这类词。",
        f"{why} ({code}). Try other wording; avoid violence, gore and real people's names.",
    )


def _with_visual_style(prompt: str, visual_style: str, *, kind: str = "place") -> str:
    """The world style stays consistent; entity reference images use clean lighting independent of the scene's photography."""
    style = " ".join(
        ensure_english_text(
            visual_style,
            fallback="cinematic, coherent character and environment design, detailed lighting",
            model=config.REWRITER_MODEL,
            thinking=config.REWRITER_THINKING,
            tag="image_visual_style_english_guard",
            translate=False,
        ).split()
    )
    if kind in ("subject", "object"):
        return (
            f"{prompt.rstrip()}\n\n"
            + (f"WORLD ART DIRECTION (appearance only):\n{style}\n" if style else "")
            + "Use the world style for artistic medium, design language, intrinsic colors "
            "and materials. Present this asset with even neutral lighting, sharp focus across "
            "the entire entity from near to far, and a clean, well-defined silhouette."
        )
    if not style:
        return prompt.rstrip()
    return (
        f"{prompt.rstrip()}\n\n"
        "MANDATORY WORLD VISUAL STYLE — keep this exact art direction consistent "
        "with every other image and video in the world:\n"
        f"{style}\n"
        "Do not switch medium, rendering technique, palette, material language, "
        "lighting style, or camera treatment."
    )


# Reference image size. **Match the cloud model's aspect**: it draws 16:9 (1376×768 places,
# 1408×768 characters); character images are wide too, not square. A square image would be a
# different kind of reference and change the video model's composition habits.
#
# Time scales with size (measured, 4 steps):
#     1024×576  1.71s      768×432  1.12s      640×360  0.88s      512×288  0.70s
# 1024×576 keeps 16:9 one step smaller than the cloud; going lower trades face detail for
# time, and a blurry reference makes the people blurry in **every** later beat.
_LOCAL_SIZE = {"place": (1024, 576), "subject": (1024, 576), "object": (1024, 576)}

_rr = itertools.count()


def _local_bases() -> list[str]:
    """Several replicas, comma separated. When a beat draws two images they run in parallel."""
    return [x.strip().rstrip("/") for x in config.IMAGE_BASE.split(",") if x.strip()]


def _local_draw(prompt: str, kind: str) -> bytes:
    """Call a self-hosted Qwen-Image service.

    Failures do **not silently fall back** to the cloud model: that would hide whether the local
    service is in use at all, and latency is the only reason this path exists. Report errors.
    """
    bases = _local_bases()
    if not bases:
        raise ImageRefused("image.service_urls is set but contains no URL")
    base = bases[next(_rr) % len(bases)]
    w, h = _LOCAL_SIZE.get(kind, _LOCAL_SIZE["place"])
    r = httpx.post(f"{base}/generate", json={"prompt": prompt, "width": w, "height": h}, timeout=180.0)
    if r.status_code >= 400:
        raise ImageRefused(f"Image service {r.status_code}: {r.text[:200]}")
    if not r.content:
        raise ImageRefused("Image service returned an empty response")
    return r.content


def _openai_draw(prompt: str, images: Sequence[tuple[str, bytes]] = ()) -> bytes:
    """An OpenAI-compatible image endpoint (`/images/generations`, or
    `/images/edits` when reference images are given)."""
    if not config.OPENAI_KEY:
        raise ImageRefused("openai.api_key is empty -- set it in config.yaml")
    base = config.IMAGE_OPENAI_BASE or config.OPENAI_BASE
    headers = {"Authorization": f"Bearer {config.OPENAI_KEY}"}
    if images:
        files = [("image[]", (f"ref{i}{sniff(raw)[0]}", raw, mime)) for i, (mime, raw) in enumerate(images)]
        r = httpx.post(
            f"{base}/images/edits",
            headers=headers,
            timeout=300.0,
            files=files,
            data={"model": config.IMAGE_MODEL, "prompt": prompt, "size": "1536x1024"},
        )
    else:
        r = httpx.post(
            f"{base}/images/generations",
            headers=headers,
            timeout=300.0,
            json={"model": config.IMAGE_MODEL, "prompt": prompt, "size": "1536x1024", "n": 1},
        )
    if r.status_code >= 400:
        raise ImageRefused(f"image API {r.status_code}: {r.text[:300]}")
    item = (r.json().get("data") or [{}])[0]
    if item.get("b64_json"):
        return base64.b64decode(item["b64_json"])
    if item.get("url"):
        return httpx.get(item["url"], timeout=120.0, follow_redirects=True).content
    raise ImageRefused(f"image API returned no image: {r.text[:200]}")


# Output sizes of DashScope image models, by asset kind (width*height).
_DASHSCOPE_SIZE = {"place": "1664*928"}


def _dashscope_draw(prompt: str, kind: str, images: Sequence[tuple[str, bytes]] = ()) -> bytes:
    """Alibaba Cloud Model Studio (DashScope) image generation, e.g. qwen-image-2.0.

    Reference images are sent along when given; a model that rejects them is
    retried with the text alone.
    """
    if not config.IMAGE_DASHSCOPE_KEY:
        raise ImageRefused("image.api_key is empty -- set it in config.yaml")
    text = [{"text": prompt}]
    refs = [{"image": f"data:{mime};base64,{base64.b64encode(raw).decode()}"} for mime, raw in images]
    for content in ([*refs, *text], text) if refs else (text,):
        r = httpx.post(
            f"{config.IMAGE_DASHSCOPE_BASE}/services/aigc/multimodal-generation/generation",
            headers={"Authorization": f"Bearer {config.IMAGE_DASHSCOPE_KEY}"},
            timeout=180.0,
            json={
                "model": config.IMAGE_MODEL,
                "input": {"messages": [{"role": "user", "content": content}]},
                "parameters": {"size": _DASHSCOPE_SIZE.get(kind, "1328*1328"), "watermark": False},
            },
        )
        if r.status_code < 400:
            break
        if content is text:
            raise ImageRefused(f"DashScope image API {r.status_code}: {r.text[:300]}")
    out = r.json().get("output") or {}
    for choice in out.get("choices") or []:
        for part in (choice.get("message") or {}).get("content") or []:
            if part.get("image"):
                return httpx.get(part["image"], timeout=120.0, follow_redirects=True).content
    raise ImageRefused(f"DashScope image API returned no image: {r.text[:200]}")


def generate(
    prompt: str,
    *,
    kind: str = "place",
    model: str | None = None,
    visual_style: str = "",
    force_gemini: bool = False,
    images: Sequence[tuple[str, bytes]] = (),
) -> bytes:
    """Text/image -> one reference image.

    ``images`` are optional visual references (Gemini and OpenAI-compatible
    providers). Self-hosted image services are text-only.
    """
    # Character, scene and object descriptions can originate from Chinese UI
    # fields.  Translate only when needed, then append the English asset contract
    # suffix.  This keeps every image backend on the same language boundary.
    prompt = ensure_english_text(
        prompt,
        fallback=(
            "Generate the requested isolated reference asset with clear stable "
            "appearance, neutral lighting, and no text."
        ),
        allow_dialogue=False,
        model=config.REWRITER_MODEL,
        thinking=config.REWRITER_THINKING,
        tag="image_prompt_english_guard",
    )
    prompt = _with_visual_style(prompt, visual_style, kind=kind)
    if kind == "place" and not visual_style.strip():
        prompt += " Cinematic, natural lighting."
    prompt += SUFFIX.get(kind, SUFFIX["place"])
    body = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    *(
                        {"inlineData": {"mimeType": mime, "data": base64.b64encode(raw).decode()}}
                        for mime, raw in images
                    ),
                    {"text": prompt},
                ],
            }
        ],
        "generationConfig": {"responseModalities": ["IMAGE"]},
    }
    reference_instruction = _reference_instruction(kind)
    if reference_instruction:
        body["systemInstruction"] = {"parts": [{"text": reference_instruction}]}
    _t0 = time.time()
    if config.IMAGE_BASE and not force_gemini:
        # The local endpoint has no system-message field.
        local_prompt = (
            f"{reference_instruction}\n\nAPPEARANCE SOURCE:\n{prompt}" if reference_instruction else prompt
        )
        raw = _local_draw(local_prompt, kind)
        reqlog.record(
            "image",
            asset_kind=kind,
            model=f"local:{config.IMAGE_BASE}",
            seconds=round(time.time() - _t0, 3),
            prompt=local_prompt,
        )
        return raw
    if config.IMAGE_PROVIDER == "dashscope" and not force_gemini:
        full = f"{reference_instruction}\n\n{prompt}" if reference_instruction else prompt
        raw = _dashscope_draw(full, kind, images)
        reqlog.record(
            "image",
            asset_kind=kind,
            model=config.IMAGE_MODEL,
            seconds=round(time.time() - _t0, 3),
            prompt=full,
        )
        return raw
    if config.IMAGE_PROVIDER == "openai" and not force_gemini:
        full = f"{reference_instruction}\n\n{prompt}" if reference_instruction else prompt
        raw = _openai_draw(full, images)
        reqlog.record(
            "image",
            asset_kind=kind,
            model=config.IMAGE_MODEL,
            seconds=round(time.time() - _t0, 3),
            prompt=full,
        )
        return raw
    resp = _post(model or config.IMAGE_MODEL, body, timeout=180.0)
    # record's first argument is already the log event kind; the asset type uses its own field, so
    # the positional "image" and the keyword kind are not bound twice before reaching the logger.
    reqlog.record(
        "image",
        asset_kind=kind,
        model=model or config.IMAGE_MODEL,
        seconds=round(time.time() - _t0, 3),
        prompt=prompt,
        reference_instruction=reference_instruction,
    )
    for cand in resp.get("candidates", []):
        for part in cand.get("content", {}).get("parts", []):
            inline = part.get("inlineData") or part.get("inline_data")
            if inline and inline.get("data"):
                return base64.b64decode(inline["data"])
    raise ImageRefused(_why(resp))
