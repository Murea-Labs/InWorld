"""Video generation over the configured provider.

Murea (`murea.py`) implements `base.VideoProvider`; the rest of the code calls the
functions here and asks the provider's capability flags (`supports_continuation()`,
`voice_references`, `references_on_continuation`) rather than which backend it is.
"""

from __future__ import annotations

import threading
import time
from typing import Any

from murea_agent import config, reqlog
from murea_agent.providers.video import murea
from murea_agent.providers.video.base import Generation, PromptTooLongError, VideoError, VideoProvider

__all__ = [
    "PROVIDERS",
    "Generation",
    "PromptTooLongError",
    "VideoError",
    "VideoProvider",
    "cancel",
    "configured",
    "continuation_status",
    "describe",
    "generate",
    "health",
    "model",
    "supports_continuation",
]

PROVIDERS: dict[str, VideoProvider] = {murea.PROVIDER.id: murea.PROVIDER}


def model(model_id: str = "") -> VideoProvider:
    """The provider for `model_id`; worlds saved with a service that is no longer built in use Murea."""
    return PROVIDERS.get(model_id) or PROVIDERS[config.VIDEO_MODEL]


def configured() -> list[str]:
    """Video models this deployment can actually call."""
    return [p.id for p in PROVIDERS.values() if p.is_configured()] or [model().id]


def describe(model_id: str = "") -> dict[str, Any]:
    m = model(model_id)
    return {
        "id": m.id,
        "label": m.label,
        "note": m.note,
        "choices": [{"id": k, "label": PROVIDERS[k].label, "note": PROVIDERS[k].note} for k in configured()],
    }


def supports_continuation(model_id: str = "") -> bool:
    """Whether the next beat can continue from the previous clip's latent."""
    return model(model_id).supports_continuation()


def continuation_status(model_id: str = "") -> str:
    return model(model_id).continuation_note()


def health() -> bool:
    return model().healthy()


def cancel(job_id: str, model_id: str = "") -> bool:
    """Withdraw a queued or running job; False when the provider cannot."""
    return model(model_id).cancel(job_id)


# ⚠️ The video service runs one job at a time, so **submit until result** is serialized.
# It used to be serialized in sse.stream around the whole beat (agent + video), so another user's
# LLM seconds blocked you too. Now only this part is covered and agents run concurrently.
_gpu_lock = threading.Lock()


def generate(*args: Any, model_id: str = "", **kw: Any) -> Generation:
    """Run one generation on `model_id` (default: the configured one), serialized.

    ⚠️ Only the video part queues here. The agent's seconds are pure LLM and unrelated to the video
    service. `on_queue` is called when this request has to wait; `cancelled` is checked once the
    slot is free, so a withdrawn speculative request never takes it from an accepted action.
    """
    provider = model(model_id)
    queued_at = time.monotonic()
    on_queue = kw.pop("on_queue", None)
    cancelled = kw.pop("cancelled", None)
    if on_queue and _gpu_lock.locked():
        on_queue()
    with _gpu_lock:
        if cancelled is not None and cancelled():
            raise VideoError("Video generation was cancelled")
        queue_seconds = time.monotonic() - queued_at
        reqlog.record("video_slot_acquired", queue_seconds=queue_seconds)
        result = provider.generate(*args, **kw)
        result.params["local_queue_seconds"] = round(queue_seconds, 3)
        return result
