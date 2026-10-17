"""Read the director's line aloud.

Why: `reply` arrives in a second while the video takes much longer; reading it aloud turns
the wait from watching a progress bar into someone telling you a story.

The backend is a self-hosted OmniVoice (https://github.com/k2-fsa/OmniVoice), about 0.6 s per
sentence on one GPU regardless of length.
Without `TTS_URL`, or when the service is down, `available()` is False and the frontend
disables the switch; one optional feature less, never an app error.
"""

from __future__ import annotations

import hashlib
import threading

import time

import httpx

from murea_agent import config

# TTS is a service on a GPU box as well; reuse connections.
_client = httpx.Client(
    # Narration is only useful while the video is still pending. A dead TTS
    # backend must fail quickly instead of occupying a worker for a full minute.
    timeout=httpx.Timeout(10.0, connect=3.0),
    limits=httpx.Limits(max_keepalive_connections=4, max_connections=8, keepalive_expiry=600.0),
)


def _headers() -> dict[str, str]:
    # For a TTS server behind an authenticating proxy; leave tts.key empty for a direct connection.
    return {"Authorization": f"Bearer {config.TTS_KEY}"} if config.TTS_KEY else {}


# Do not synthesize the same line twice: replaying clips or auto-continue resending a line
# both hit the cache. A line's wav is usually 100-200 KB; keep 64 and drop the oldest.
_cache: dict[str, bytes] = {}
_order: list[str] = []
_lock = threading.Lock()
_CACHE_MAX = 64


class TTSError(RuntimeError):
    pass


def available() -> bool:
    return bool(config.TTS_URL)


# Cache the probe briefly: the frontend polls /api/health, and probing every time would load
# test the TTS service. 15 s is enough for up / down to show in the UI in time.
_health_cache: tuple[float, bool] = (0.0, False)


def health() -> bool:
    """Whether the TTS service answers **right now**.

    ⚠️ Not the same as `available()`, which only checks that TTS_URL is set. Reporting only
    `available()` once told the UI "tts": true while the service refused connections, so the
    switch was lit and turning it on did nothing.
    """
    global _health_cache
    if not config.TTS_URL:
        return False
    now = time.time()
    when, value = _health_cache
    if now - when < 15.0:
        return value
    try:
        response = _client.get(f"{config.TTS_URL}/health", headers=_headers(), timeout=4.0)
        ok = response.status_code == 200
        if ok:
            try:
                payload = response.json()
            except (TypeError, ValueError):
                payload = {}
            # Both the retired Qwen service and OmniVoice expose `loaded`.
            # A live HTTP process with no model cannot synthesize anything.
            if isinstance(payload, dict) and payload.get("loaded") is False:
                ok = False
    except Exception:  # noqa: BLE001
        ok = False
    _health_cache = (now, ok)
    return ok


def speak(text: str, speaker: str = "") -> bytes:
    """A line -> wav bytes."""
    text = (text or "").strip()[:600]
    if not text:
        raise TTSError("Empty text")
    if not config.TTS_URL:
        raise TTSError("Voice is not configured (tts.url)")
    key = hashlib.sha1(f"{speaker}|{text}".encode()).hexdigest()
    with _lock:
        hit = _cache.get(key)
    if hit is not None:
        return hit

    body = {"text": text}
    if speaker:
        body["speaker"] = speaker
    try:
        r = _client.post(f"{config.TTS_URL}/tts", json=body, headers=_headers())
    except Exception as exc:  # noqa: BLE001
        raise TTSError(f"Cannot reach the TTS service: {exc}") from exc
    if r.status_code >= 400:
        raise TTSError(f"TTS {r.status_code}: {r.text[:200]}")
    wav = r.content

    with _lock:
        _cache[key] = wav
        _order.append(key)
        while len(_order) > _CACHE_MAX:
            _cache.pop(_order.pop(0), None)
    return wav
