"""Text LLM client (Gemini REST)."""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import random
import threading
import time
from functools import lru_cache
from typing import Any
from collections.abc import Callable, Sequence

import httpx

from murea_agent import config
from murea_agent import reqlog

log = logging.getLogger(__name__)

# Import during service startup, never make the first user's turn pay for loading
# the optional compiler and its dependencies.
if config.OPENAI_JSON_MAX_WHITESPACE:
    try:
        from xgrammar import Grammar as _BoundedGrammar
    except ImportError as exc:
        raise RuntimeError("llm.json_max_whitespace needs xgrammar: pip install '.[grammar]'") from exc
else:
    _BoundedGrammar = None


class GeminiError(RuntimeError):
    pass


def _url(model: str) -> str:
    return f"{config.GEMINI_BASE}/models/{model}:generateContent"


# Reuse connections. Each httpx.post would do a new TLS handshake, twice per turn, and the round
# trip to the provider is not short.
#
# ⚠️ keepalive_expiry must be set long explicitly. httpx keeps connections only 5 s by default, while beats
# are separated by long video renders; by the next beat the connection had expired and every call
# handshook again. Measured: a second call right away 1.3 s, after 30 s idle 2.2 s.
_client = httpx.Client(
    timeout=120.0,
    limits=httpx.Limits(max_keepalive_connections=8, max_connections=16, keepalive_expiry=600.0),
    headers={"accept-encoding": "gzip"},
)


# Shared Gemini endpoints return 503/429 mid-run; retry those, not our own 4xx.
_RETRY_STATUS = {408, 429, 500, 502, 503, 504}

# Models that reject thinkingConfig: some return 400 when it is sent, and they do not think anyway.
# A hard-coded list would go stale, so remember a model after the first rejection and stop sending
# it. Switching to another provider must not break the app over an optional field.
_no_thinking: set[str] = set()
# OpenAI-compatible models whose endpoint rejects `json_schema` response
# formats (e.g. DeepSeek on DashScope). They get `json_object` plus the schema
# written into the system prompt; learnt from the first 400 per model.
_no_json_schema: set[str] = set()


def _schema_rejected(exc: Exception) -> bool:
    text = str(exc)
    return " 400" in f" {text}" and "response_format" in text


def _post(
    model: str,
    body: dict[str, Any],
    timeout: float = 120.0,
    *,
    attempts: int = 4,
) -> dict[str, Any]:
    if not config.GEMINI_API_KEY:
        raise GeminiError("GEMINI_API_KEY is empty -- set it in .env")
    last = ""
    # The director / rewriter / image paths all go through here (imagegen references this function
    # object), so one hook is enough. See murea_agent/reqlog.py.
    started = time.time()
    for i in range(attempts):
        try:
            r = _client.post(
                _url(model),
                params={"key": config.GEMINI_API_KEY},
                json=body,
                timeout=timeout,
            )
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            last = f"{type(exc).__name__}: {exc}"
        else:
            if r.status_code >= 400 and _uncache_body(body):
                # The cache expired or was deleted. Restore the inline system prompt and try again;
                # this does not count as a retry: it is not the model's fault, our cost-saving path
                # failed.
                reqlog.record(
                    "cache",
                    model=model,
                    ok=False,
                    error=f"{r.status_code}; resent with systemInstruction restored",
                )
                r = _client.post(
                    _url(model),
                    params={"key": config.GEMINI_API_KEY},
                    json=body,
                    timeout=timeout,
                )
            if r.status_code < 400:
                resp = r.json()
                reqlog.record(
                    "gemini",
                    model=model,
                    ok=True,
                    retries=i,
                    seconds=round(time.time() - started, 3),
                    usage=_usage(resp),
                    request=_log_body(body),
                    response=_parts_text(resp) or "(no text, possibly an image)",
                )
                return resp
            last = f"{r.status_code}: {r.text[:400]}"
            if r.status_code not in _RETRY_STATUS:
                raise GeminiError(f"{model} -> {last}")
        if i < attempts - 1:
            # The backoff used to be 2/4/8 s, 14 s of sleeping across four attempts, and this path
            # is **inside the user's wait** (director and rewriter both use it).
            #
            # Now 0.5/1/2 s (3.5 s in total). Most 429/503 from shared endpoints are momentary and
            # pass after half a second; when really rate limited, waiting 14 s does not help either,
            # it only adds to the user's wait.
            #
            # Jitter is required: a beat now prefetches several options concurrently, and a fixed
            # backoff would make them retry at the same moment and trigger another 429.
            time.sleep((0.5 * 2**i) * random.uniform(0.7, 1.3))
    reqlog.record(
        "gemini",
        model=model,
        ok=False,
        retries=attempts,
        seconds=round(time.time() - started, 3),
        request=_log_body(body),
        error=last,
    )
    raise GeminiError(f"{model} -> {last} (failed after {attempts} attempts)")


@reqlog.complete_text_request
def _log_body(body: dict[str, Any]) -> dict[str, Any]:
    """The parts of a request body that matter for debugging.

    Only the length of system is kept: it is a template, identical each time, and storing it would
    make the log bigger than the world; "what was sent this time" is in contents.
    """
    if isinstance(body.get("messages"), list):

        def text_of(message: dict[str, Any]) -> str:
            content = message.get("content") or ""
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                # Never write image data URLs or credentials into request logs.
                return " ".join(
                    str(part.get("text") or "")
                    for part in content
                    if isinstance(part, dict) and part.get("type") == "text"
                )
            return ""

        messages = [m for m in body["messages"] if isinstance(m, dict)]
        turns = [m for m in messages if m.get("role") != "system"]
        return {
            "system_chars": sum(len(text_of(m)) for m in messages if m.get("role") == "system"),
            "turns": [{"role": m.get("role"), "text": text_of(m)[: reqlog.MAX_TEXT]} for m in turns[-6:]],
            "schema": bool((body.get("response_format") or {}).get("json_schema") or body.get("ebnf")),
            "thinking": (body.get("chat_template_kwargs") or {}).get("enable_thinking"),
        }
    sys_parts = (body.get("systemInstruction") or {}).get("parts") or []
    contents = body.get("contents") or []
    turns = []
    for c in contents[-6:]:  # keep only the last few turns; the history rolls
        text = " ".join(str(p.get("text") or "") for p in (c.get("parts") or []))
        turns.append({"role": c.get("role"), "text": text})
    cfg = body.get("generationConfig") or {}
    return {
        "system_chars": sum(len(str(p.get("text") or "")) for p in sys_parts),
        "turns": turns,
        "schema": bool(cfg.get("responseSchema") or cfg.get("responseJsonSchema")),
        "thinking": (cfg.get("thinkingConfig") or {}).get("thinkingBudget"),
    }


def _usage(resp: dict[str, Any]) -> dict[str, int]:
    u = resp.get("usageMetadata") or {}
    return {
        "in": int(u.get("promptTokenCount") or 0),
        "out": int(u.get("candidatesTokenCount") or 0),
        "think": int(u.get("thoughtsTokenCount") or 0),
        # The part of the input served from the implicit cache. The system prompt is large and sent
        # several times per beat (director + prefetches); hits mean cost and first-token latency,
        # and unmeasured means unmanaged.
        "cached": int(u.get("cachedContentTokenCount") or 0),
    }


def _parts_text(resp: dict[str, Any]) -> str:
    out = []
    for cand in resp.get("candidates", []):
        for part in cand.get("content", {}).get("parts", []):
            if "text" in part:
                out.append(part["text"])
    return "".join(out).strip()


def _empty_response_reason(resp: dict[str, Any]) -> str:
    """Explain a successful HTTP response that contains no model text."""
    candidates = resp.get("candidates") or [{}]
    candidate = candidates[0] if isinstance(candidates[0], dict) else {}
    reason = candidate.get("finishReason") or "unknown"
    feedback = resp.get("promptFeedback") or {}
    block = feedback.get("blockReason") or ""
    message = candidate.get("finishMessage") or ""
    details = ", ".join(
        str(x)
        for x in (
            f"finishReason={reason}",
            f"blockReason={block}" if block else "",
            f"finishMessage={message}" if message else "",
        )
        if x
    )
    return f"Gemini returned no text ({details})"


# ─────────── Explicit cache for the system prompt ───────────
#
# A beat calls the model several times (director + option prefetches), each uploading the same large
# system prompt; most of the input tokens are this prompt.
#
# ⚠️ **This is a cost optimization, not a latency one.** Measured: cache hits 1.01–1.20 s, misses
# 0.78–0.96 s; prefill is fast and hides in the fixed overhead. It saves money: cached tokens are
# billed at about a quarter, and input dominates output.
#
# Implicit caching cannot be relied on: four calls with the same prefix all reported
# `cachedContentTokenCount` 0.
#
# System prompts are module constants (AGENT_SYSTEM / REWRITER_SYSTEM / GM_SYSTEM / …), stable
# within the process, so there are only a few caches and no churn.
_CACHE_TTL = 1800
_CACHE_MIN_CHARS = 8000  # too short is not worth it and would not pass the model's minimum token threshold
_cache_lock = threading.Lock()
# (model, sha1(system)) -> {"name": str, "until": float}
_caches: dict[tuple[str, str], dict[str, Any]] = {}
# name -> (model, system), used to restore systemInstruction on errors
_cache_systems: dict[str, tuple[str, str]] = {}
# After one failed creation do not try again: a key without the feature or an unsupported model
# would only waste a call every beat
_cache_off: set[tuple[str, str]] = set()


def _cache_key(model: str, system: str) -> tuple[str, str]:
    return (model, hashlib.sha1(system.encode()).hexdigest())


def _cache_for(model: str, system: str) -> str | None:
    """Get a usable cache name. Any problem returns None: the cache saves money, it is not required."""
    if len(system) < _CACHE_MIN_CHARS or not config.GEMINI_API_KEY:
        return None
    key = _cache_key(model, system)
    now = time.time()
    with _cache_lock:
        if key in _cache_off:
            return None
        hit = _caches.get(key)
        if hit and hit["until"] - now > 120:
            return hit["name"]
    try:
        r = _client.post(
            f"{config.GEMINI_BASE}/cachedContents",
            params={"key": config.GEMINI_API_KEY},
            json={
                "model": f"models/{model}",
                "systemInstruction": {"parts": [{"text": system}]},
                "ttl": f"{_CACHE_TTL}s",
            },
            timeout=60.0,
        )
        if r.status_code >= 400:
            with _cache_lock:
                _cache_off.add(key)
            reqlog.record(
                "cache",
                model=model,
                ok=False,
                error=f"{r.status_code}: {r.text[:200]} (context caching disabled for this model/key)",
            )
            return None
        name = str(r.json()["name"])
    except Exception as exc:  # noqa: BLE001 -- a failed saving must never affect the main flow
        with _cache_lock:
            _cache_off.add(key)
        reqlog.record("cache", model=model, ok=False, error=f"{type(exc).__name__}: {exc}")
        return None
    with _cache_lock:
        _caches[key] = {"name": name, "until": now + _CACHE_TTL}
        _cache_systems[name] = (model, system)
    reqlog.record("cache", model=model, ok=True, name=name, chars=len(system))
    return name


def _drop_cache(name: str) -> None:
    """This cache can no longer be used (expired / deleted). Forget it; it is rebuilt next time."""
    with _cache_lock:
        for key, val in list(_caches.items()):
            if val.get("name") == name:
                _caches.pop(key, None)
        _cache_systems.pop(name, None)


def _uncache_body(body: dict[str, Any]) -> bool:
    """Switch body from "use the cache" back to "inline systemInstruction". Returns True if changed."""
    name = body.get("cachedContent")
    if not name:
        return False
    pair = _cache_systems.get(str(name))
    _drop_cache(str(name))
    body.pop("cachedContent", None)
    if pair is None:
        return False  # the original is unknown; let the caller report the error as usual
    body["systemInstruction"] = {"parts": [{"text": pair[1]}]}
    return True


def _body(
    system: str,
    history: Sequence[dict[str, str]],
    user: str,
    model: str,
    schema: dict[str, Any] | None,
    temperature: float,
    thinking: int | None,
    images: Sequence[tuple[str, bytes]] = (),
    *,
    tools: Sequence[dict[str, Any]] = (),
    final_round: bool = False,
    extra: Sequence[dict[str, Any]] = (),
) -> dict[str, Any]:
    contents = [{"role": h["role"], "parts": [{"text": h["text"]}]} for h in history]
    # Images go before the text. Each image costs a fixed number of input tokens (the provider
    # downsamples), adding about 0.27 s to the first token only for the first image; total time
    # barely changes.
    parts: list[dict[str, Any]] = [
        {"inlineData": {"mimeType": mime, "data": base64.b64encode(raw).decode()}} for mime, raw in images
    ]
    parts.append({"text": user})
    contents.append({"role": "user", "parts": parts})
    # Earlier tool rounds of this same request: the model's calls and our results.
    contents.extend(extra)
    body: dict[str, Any] = {
        "contents": contents,
        "generationConfig": {"temperature": temperature},
    }
    if tools:
        body["tools"] = [{"functionDeclarations": list(tools)}]
        if final_round:
            # Out of tool rounds: the model must answer now.
            body["toolConfig"] = {"functionCallingConfig": {"mode": "NONE"}}
    # On a hit, replace system with the cache reference; otherwise send it inline (unchanged
    # behavior) A cached system prompt cannot be combined with tools in one request.
    cached = None if tools else _cache_for(model, system)
    if cached:
        body["cachedContent"] = cached
    else:
        body["systemInstruction"] = {"parts": [{"text": system}]}
    if schema is not None:
        body["generationConfig"]["responseMimeType"] = "application/json"
        # Compact Agent schemas contain conditional scene/physics contracts.
        # Use the JSON Schema endpoint field so const/additionalProperties keep
        # their meaning instead of weakening them for the legacy OpenAPI subset.
        compact = "world_delta" in schema.get("properties", {})
        body["generationConfig"]["responseJsonSchema" if compact else "responseSchema"] = schema
    if thinking is not None and model not in _no_thinking:
        body["generationConfig"]["thinkingConfig"] = {"thinkingBudget": thinking}
    return body


def partial_json(buf: str) -> dict[str, Any] | None:
    """Cut unfinished JSON at **the last completed top-level field**, add `}` and parse.

    The model emits field by field, so the first field completes long before the end. Only two cut
    points count: a top-level value just closed (depth back to 1), or a top-level comma.
    Half-written fields are dropped: better to report it with the next chunk than to treat half a
    sentence as finished.
    """
    depth, cut = 0, -1
    in_str = esc = False
    for i, ch in enumerate(buf):
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = in_str
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
            if depth == 1:
                cut = i + 1
        elif ch == "," and depth == 1:
            cut = i
    if cut < 0:
        return None
    try:
        out = json.loads(buf[:cut].rstrip(", ") + "}")
    except json.JSONDecodeError:
        return None
    return out if isinstance(out, dict) else None


# ─────────── OpenAI-protocol backends: self-hosted SGLang / vLLM, hosted APIs ───────────
#
# Self-hosting is about **the tail**, not the median: a hosted small model had a median first token
# of 1.3 s and p90 5.5 s, while a self-hosted server with prefix caching answered in 0.16 s / p90
# 0.25 s. Decoding can be slower, so this is a choice to measure, not an upgrade; see
# config.LLM_BACKEND.
#
# The contract matches the Gemini path exactly: the same parsed JSON, the same streamed fields, the
# same `llm` entry in reqlog, the same fallback to non-streaming. Callers cannot tell the
# difference.


def _openai_messages(
    system: str,
    history: Sequence[dict[str, str]],
    user: str,
    images: Sequence[tuple[str, bytes]],
) -> list[dict[str, Any]]:
    """Translate the system / history / user / images shape into messages.

    Images go on **the last user message**, the same place as in `_body`: the image is "where the
    world stands now", and attached elsewhere the model would read it as an old image from the
    history.
    """
    msgs: list[dict[str, Any]] = [{"role": "system", "content": system}]
    for turn in history:
        text = str(turn.get("text") or "")
        if not text.strip():
            continue
        msgs.append({"role": "assistant" if turn.get("role") == "model" else "user", "content": text})
    if not images:
        msgs.append({"role": "user", "content": user})
        return msgs
    parts: list[dict[str, Any]] = [{"type": "text", "text": user}]
    for mime, raw in images:
        b64 = base64.b64encode(raw).decode()
        parts.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})
    msgs.append({"role": "user", "content": parts})
    return msgs


@lru_cache(maxsize=128)
def _bounded_json_grammar(schema_text: str, maximum: int) -> str:
    compiler = _BoundedGrammar
    if compiler is None:
        from xgrammar import Grammar as compiler
    return str(
        compiler.from_json_schema(
            schema_text, any_whitespace=True, max_whitespace_cnt=maximum, strict_mode=True
        )
    )


def _openai_body(
    system: str,
    history: Sequence[dict[str, str]],
    user: str,
    model: str,
    schema: dict[str, Any],
    temperature: float,
    thinking: int | None,
    images: Sequence[tuple[str, bytes]],
    *,
    stream: bool,
) -> dict[str, Any]:
    if schema and model in _no_json_schema:
        system = (
            system + "\n\nReply with exactly one JSON object that conforms to this "
            "JSON Schema (every required key present, no other text):\n"
            + json.dumps(schema, ensure_ascii=False)
        )
    body: dict[str, Any] = {
        "model": model,
        "messages": _openai_messages(system, history, user, images),
        "temperature": temperature,
        "max_tokens": 4096,
    }
    if schema and model in _no_json_schema:
        body["response_format"] = {"type": "json_object"}
    elif schema:
        if config.OPENAI_JSON_MAX_WHITESPACE:
            body["ebnf"] = _bounded_json_grammar(
                json.dumps(schema, ensure_ascii=False), config.OPENAI_JSON_MAX_WHITESPACE
            )
        else:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "agent", "strict": True, "schema": schema},
            }
    if thinking == 0:
        # Qwen3-family models disable thinking through the chat template, not sampling parameters.
        # Without it they write thousands of thinking tokens first and the first token arrives after
        # 10 s.
        body["chat_template_kwargs"] = {"enable_thinking": False}
    if stream:
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}
    return body


def _openai_usage(raw: dict[str, Any] | None) -> dict[str, int]:
    u = raw or {}
    out = {"in": int(u.get("prompt_tokens") or 0), "out": int(u.get("completion_tokens") or 0)}
    think = int(u.get("reasoning_tokens") or 0)
    if think:
        out["think"] = think
    return out


def _openai_post(body: dict[str, Any], base: str = "") -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if config.OPENAI_KEY:
        headers["Authorization"] = f"Bearer {config.OPENAI_KEY}"
    r = _client.post(
        f"{base or config.OPENAI_BASE}/chat/completions", json=body, headers=headers, timeout=180.0
    )
    if r.status_code >= 400:
        raise GeminiError(f"{body['model']} -> {r.status_code}: {r.text[:300]}")
    return r.json()


def _openai_chat(
    *,
    system: str,
    history: Sequence[dict[str, str]],
    user: str,
    model: str,
    schema: dict[str, Any],
    temperature: float,
    thinking: int | None,
    tag: str,
    images: Sequence[tuple[str, bytes]],
    base: str = "",
) -> Any:
    """⚠️ The non-streaming path can get **truncated JSON** too: past max_tokens both paths fail the same way.

    This used to call `json.loads(text)` and raise, after "streaming truncated → fall back to
    non-streaming → truncated again → raise"; the whole beat failed with no visible cause. Now it
    first checks whether the fields already completed are enough.
    """
    body = _openai_body(system, history, user, model, schema, temperature, thinking, images, stream=False)
    t0 = time.time()
    try:
        data = _openai_post(body, base)
    except GeminiError as exc:
        if not (schema and model not in _no_json_schema and _schema_rejected(exc)):
            raise
        _no_json_schema.add(model)
        reqlog.record(
            "llm",
            tag=tag or model,
            model=model,
            ok=False,
            error="json_schema not supported; using json_object",
        )
        return _openai_chat(
            system=system,
            history=history,
            user=user,
            model=model,
            schema=schema,
            temperature=temperature,
            thinking=thinking,
            tag=tag,
            images=images,
            base=base,
        )
    text = ((data.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    usage = _openai_usage(data.get("usage"))
    try:
        out = json.loads(text)
    except json.JSONDecodeError:
        salvage = partial_json(text)
        if not (salvage and len(salvage) >= 3):
            reqlog.record(
                "llm",
                tag=tag or model,
                model=model,
                ok=False,
                stream=False,
                seconds=round(time.time() - t0, 3),
                chars=len(text),
                usage=usage,
                error="JSON truncated and the completed fields are not enough",
                request=_log_body(body),
                response=text,
            )
            raise
        out = salvage
        reqlog.record(
            "llm",
            tag=tag or model,
            model=model,
            ok=True,
            stream=False,
            seconds=round(time.time() - t0, 3),
            chars=len(text),
            usage=usage,
            salvaged=True,
            request=_log_body(body),
            response=text,
        )
        return out
    reqlog.record(
        "llm",
        tag=tag or model,
        model=model,
        ok=True,
        stream=False,
        seconds=round(time.time() - t0, 3),
        chars=len(text),
        usage=usage,
        request=_log_body(body),
        response=text,
    )
    return out


class _Stream:
    """A streamed reply: accumulates text and reports JSON fields to `on_partial` as they complete."""

    def __init__(self, on_partial: Callable[[dict[str, Any]], None] | None) -> None:
        self.t0 = time.time()
        self.buf = ""
        # The first-token time is kept apart: "text starts appearing" and "everything is written"
        # are different experiences; one number would hide whether streaming helps.
        self.first_at = 0.0
        self.usage: dict[str, int] = {}
        self.finish_reason: str | None = None
        self._seen: set[str] = set()
        self._on_partial = on_partial

    def add(self, text: str) -> None:
        if text and not self.first_at:
            self.first_at = time.time() - self.t0
        self.buf += text

    def report(self) -> None:
        if self._on_partial is None:
            return
        got = partial_json(self.buf)
        if got and set(got) - self._seen:
            self._seen |= set(got)
            try:
                self._on_partial(got)
            except Exception:
                # The callback is the UI's business; if it fails it must not take this call down,
                # or we would fall back to non-streaming and pay twice for the same beat.
                pass

    def record(self, tag: str, model: str, body: dict[str, Any], *, ok: bool, **extra: Any) -> None:
        reqlog.record(
            "llm",
            tag=tag or model,
            model=model,
            ok=ok,
            stream=True,
            seconds=round(time.time() - self.t0, 3),
            first_token_seconds=round(self.first_at, 3),
            chars=len(self.buf),
            usage=self.usage,
            request=_log_body(body),
            response=self.buf,
            **extra,
        )


def _read_openai_events(r: httpx.Response, stream: _Stream, calls: dict[int, dict[str, str]]) -> None:
    """Consume an OpenAI-compatible SSE response: text into `stream`, tool calls into `calls`."""
    for line in r.iter_lines():
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        try:
            chunk = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if chunk.get("usage"):
            stream.usage = _openai_usage(chunk["usage"])
        for choice in chunk.get("choices") or []:
            if choice.get("finish_reason"):
                stream.finish_reason = choice["finish_reason"]
            delta = choice.get("delta") or {}
            for tc in delta.get("tool_calls") or []:
                call = calls.setdefault(int(tc.get("index") or 0), {"id": "", "name": "", "arguments": ""})
                call["id"] = tc.get("id") or call["id"]
                fn = tc.get("function") or {}
                call["name"] += fn.get("name") or ""
                call["arguments"] += fn.get("arguments") or ""
            stream.add(delta.get("content") or "")
        stream.report()


def _openai_tool_messages(
    calls: dict[int, dict[str, str]],
    content: str,
    call_tool: Callable[[str, dict[str, Any]], dict[str, Any]],
    round_: int,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Run the requested tools; returns the assistant + tool messages to append, and the calls."""
    ordered = [calls[i] for i in sorted(calls)]
    for n, call in enumerate(ordered):
        call["id"] = call["id"] or f"call_{round_}_{n}"
    messages: list[dict[str, Any]] = [
        {
            "role": "assistant",
            "content": content or None,
            "tool_calls": [
                {
                    "id": c["id"],
                    "type": "function",
                    "function": {"name": c["name"], "arguments": c["arguments"] or "{}"},
                }
                for c in ordered
            ],
        }
    ]
    for c in ordered:
        try:
            args = json.loads(c["arguments"] or "{}")
        except json.JSONDecodeError:
            args = {}
        result = call_tool(c["name"], args if isinstance(args, dict) else {})
        messages.append(
            {"role": "tool", "tool_call_id": c["id"], "content": json.dumps(result, ensure_ascii=False)}
        )
    return messages, ordered


def _openai_stream(
    *,
    system: str,
    history: Sequence[dict[str, str]],
    user: str,
    model: str,
    schema: dict[str, Any],
    temperature: float,
    thinking: int | None,
    tag: str,
    images: Sequence[tuple[str, bytes]],
    base: str = "",
    on_partial: Callable[[dict[str, Any]], None] | None,
    tools: Sequence[dict[str, Any]] = (),
    call_tool: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
    _extra: Sequence[dict[str, Any]] = (),
    _round: int = 0,
) -> Any:
    """`_chat_stream` for OpenAI-compatible servers."""
    request = dict(
        system=system,
        history=history,
        user=user,
        model=model,
        schema=schema,
        temperature=temperature,
        thinking=thinking,
        tag=tag,
        images=images,
        base=base,
    )
    tools = tools if call_tool is not None else ()
    again = dict(
        request, on_partial=on_partial, tools=tools, call_tool=call_tool, _extra=_extra, _round=_round
    )
    body = _openai_body(system, history, user, model, schema, temperature, thinking, images, stream=True)
    body["messages"].extend(_extra)
    if tools:
        body["tools"] = [{"type": "function", "function": t} for t in tools]
        if _round >= MAX_TOOL_ROUNDS:
            body["tool_choice"] = "none"
    headers = {"Content-Type": "application/json"}
    if config.OPENAI_KEY:
        headers["Authorization"] = f"Bearer {config.OPENAI_KEY}"
    stream = _Stream(on_partial)
    calls: dict[int, dict[str, str]] = {}
    try:
        with _client.stream(
            "POST",
            f"{base or config.OPENAI_BASE}/chat/completions",
            json=body,
            headers=headers,
            timeout=180.0,
        ) as r:
            if r.status_code >= 400:
                r.read()
                error = GeminiError(f"{model} -> {r.status_code}: {r.text[:300]}")
                if not (schema and model not in _no_json_schema and _schema_rejected(error)):
                    raise error
                # Remember servers without json_schema and ask again with json_object.
                _no_json_schema.add(model)
                reqlog.record(
                    "llm",
                    tag=tag or model,
                    model=model,
                    ok=False,
                    error="json_schema not supported; using json_object",
                )
                return _openai_stream(**again)
            _read_openai_events(r, stream, calls)
        if calls and call_tool is not None:
            messages, ordered = _openai_tool_messages(calls, stream.buf, call_tool, _round)
            reqlog.record(
                "llm_tools",
                tag=tag or model,
                model=model,
                round=_round + 1,
                calls=[{"name": c["name"], "args": c["arguments"]} for c in ordered],
            )
            return _openai_stream(**dict(again, _extra=[*_extra, *messages], _round=_round + 1))
        out = json.loads(stream.buf)
    except Exception as exc:  # like the Gemini path: if streaming collapses, use non-streaming
        # ⚠️ Before falling back to non-streaming, check whether **the completed fields are enough**.
        #
        # The most common "streaming collapsed" is not the network but the model exceeding
        # max_tokens and cutting the JSON halfway, while the earlier fields are fine. Falling back
        # would make the same model write the same overlong output again: twice as slow and likely
        # truncated again. Seen in 2 of 14 beats.
        salvage = partial_json(stream.buf)
        if salvage and len(salvage) >= 3:
            stream.record(
                tag,
                model,
                body,
                ok=True,
                salvaged=True,
                error=f"{type(exc).__name__}: {exc} (salvaged from the completed fields)",
            )
            return salvage
        stream.record(
            tag,
            model,
            body,
            ok=False,
            finish_reason=stream.finish_reason,
            error=f"{type(exc).__name__}: {exc} (retried without streaming)",
        )
        log.warning(
            "%s: streaming failed, retrying without streaming: %s: %s", tag or model, type(exc).__name__, exc
        )
        return _openai_chat(**request)
    stream.record(tag, model, body, ok=True)
    if config.LLM_TRACE:
        log.info(
            "%-10s %5.2fs  self-hosted stream  chars=%5d  first token %.2fs",
            tag or model,
            time.time() - stream.t0,
            len(stream.buf),
            stream.first_at,
        )
    return out


# Tool rounds per request before the model must answer.
MAX_TOOL_ROUNDS = 2


def _use_openai(base: str = "") -> bool:
    """The self-hosted path is used only when **explicitly configured**: silently using Gemini because one
    setting is missing is the worst case, since both arms of a comparison would come from the same
    backend.

    ⚠️ `base` is the endpoint passed per call (one per replica when hedging) and must take part in the
    decision: checking only `config.OPENAI_BASE`, a config with only a replica list and no default
    endpoint would **silently fall back to Gemini**, which is exactly how hedging is usually
    configured. On a network without Gemini access that showed up as the whole call hanging with no
    error.
    """
    return config.LLM_BACKEND == "openai" and bool(base or config.OPENAI_BASE)


def _chat_stream(
    *,
    system: str,
    history: Sequence[dict[str, str]] = (),
    user: str,
    model: str | None = None,
    schema: dict[str, Any],
    temperature: float = 0.8,
    thinking: int | None = None,
    tag: str = "",
    images: Sequence[tuple[str, bytes]] = (),
    on_partial: Callable[[dict[str, Any]], None] | None = None,
    base: str = "",
    tools: Sequence[dict[str, Any]] = (),
    call_tool: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
    _extra: Sequence[dict[str, Any]] = (),
    _round: int = 0,
) -> Any:
    """Like `chat`, returns the whole parsed JSON, but reports fields as soon as they are complete.

    With `tools` and `call_tool`, the model may call tools before it answers:
    each call is run, its result appended, and the request streamed again (at
    most `MAX_TOOL_ROUNDS` times). Without calls this is one streamed request.

    Why: the first token takes about 0.85 s (round trip + prefill), then each output token only ~3.6
    ms. Waiting for the whole output makes the user wait for fields they cannot use yet (options,
    state_updates…). The first field is `reply`, the line the world says, complete much earlier.

    Streaming is only a speed-up and must not become a new failure source: connection errors, early
    errors or unparsable output all fall back to non-streaming `chat`.
    """
    m = model or config.TEXT_MODEL
    tools = tools if call_tool is not None else ()
    again = dict(
        system=system,
        history=history,
        user=user,
        model=m,
        schema=schema,
        temperature=temperature,
        thinking=thinking,
        tag=tag,
        images=images,
        on_partial=on_partial,
        base=base,
        tools=tools,
        call_tool=call_tool,
    )
    if _use_openai(base):
        return _openai_stream(**again)
    body = _body(
        system,
        history,
        user,
        m,
        schema,
        temperature,
        thinking,
        images,
        tools=tools,
        final_round=_round >= MAX_TOOL_ROUNDS,
        extra=_extra,
    )
    stream = _Stream(on_partial)
    model_parts: list[dict[str, Any]] = []
    try:
        with _client.stream(
            "POST",
            f"{config.GEMINI_BASE}/models/{m}:streamGenerateContent",
            params={"key": config.GEMINI_API_KEY, "alt": "sse"},
            json=body,
            timeout=120.0,
        ) as r:
            if r.status_code >= 400:
                r.read()
                # An invalid cache shows up here as a 4xx. Drop it; the non-streaming fallback below
                # sends the inline systemInstruction and does not fail a second time.
                _uncache_body(body)
                # Like chat: remember models that reject thinkingConfig and retry without the field.
                # Otherwise the first beat after every restart would hit a 400 and fall back to
                # non-streaming, wasting about 1.5 s with the reply arriving last.
                if r.status_code == 400 and "thinkingConfig" in body["generationConfig"]:
                    _no_thinking.add(m)
                    return _chat_stream(**dict(again, thinking=None, _extra=_extra, _round=_round))
                raise GeminiError(f"{m} -> {r.status_code}")
            for line in r.iter_lines():
                if not line.startswith("data:"):
                    continue
                chunk = json.loads(line[5:])
                if chunk.get("usageMetadata"):
                    stream.usage = _usage(chunk)
                for cand in chunk.get("candidates", []):
                    for part in cand.get("content", {}).get("parts", []):
                        if tools:
                            model_parts.append(part)  # incl. thought signatures
                        stream.add(part.get("text", ""))
                stream.report()
        calls = [p["functionCall"] for p in model_parts if p.get("functionCall")]
        if calls and call_tool is not None:
            results = [
                {
                    "functionResponse": {
                        "name": c.get("name", ""),
                        "response": call_tool(str(c.get("name") or ""), dict(c.get("args") or {})),
                    }
                }
                for c in calls
            ]
            reqlog.record(
                "llm_tools",
                tag=tag or m,
                model=m,
                round=_round + 1,
                calls=[{"name": c.get("name"), "args": c.get("args")} for c in calls],
            )
            extra = [*_extra, {"role": "model", "parts": model_parts}, {"role": "user", "parts": results}]
            return _chat_stream(**dict(again, _extra=extra, _round=_round + 1))
        out = json.loads(stream.buf)
    except Exception as exc:  # if streaming fails, pretend it never happened and use the old path
        stream.record(
            tag, m, body, ok=False, error=f"{type(exc).__name__}: {exc} (retried without streaming)"
        )
        log.warning(
            "%s: streaming failed, retrying without streaming: %s: %s", tag or m, type(exc).__name__, exc
        )
        return chat(
            system=system,
            history=history,
            user=user,
            model=model,
            schema=schema,
            temperature=temperature,
            thinking=thinking,
            tag=tag,
            images=images,
        )
    stream.record(tag, m, body, ok=True)
    if config.LLM_TRACE:
        log.info("%-10s %5.2fs  stream  chars=%5d", tag or m, time.time() - stream.t0, len(stream.buf))
    return out


def chat(
    *,
    system: str,
    history: Sequence[dict[str, str]] = (),
    user: str,
    model: str | None = None,
    schema: dict[str, Any] | None = None,
    temperature: float = 0.8,
    thinking: int | None = None,
    tag: str = "",
    images: Sequence[tuple[str, bytes]] = (),
    on_partial: Callable[[dict[str, Any]], None] | None = None,
    base: str = "",
    tools: Sequence[dict[str, Any]] = (),
    call_tool: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
) -> Any:
    """One turn. With `schema`, the reply is forced to JSON and returned parsed.

    `tools` (function declarations) and `call_tool(name, args) -> result` let the
    model call tools before answering; they apply to streamed calls.

    `thinking` is the thinking-token budget: 0 turns it off, None leaves the model's
    default alone. Both of our roles pass 0 -- see the note in `config.py`.

    With `on_partial` the call streams and calls back each time another top-level field is complete.
    The return value is unchanged, the whole parsed result; streaming only lets the caller use the
    first fields early.
    """
    m = model or config.TEXT_MODEL
    if on_partial is not None and schema is not None:
        return _chat_stream(
            system=system,
            history=history,
            user=user,
            model=m,
            schema=schema,
            temperature=temperature,
            thinking=thinking,
            tag=tag,
            images=images,
            on_partial=on_partial,
            base=base,
            tools=tools,
            call_tool=call_tool,
        )
    if _use_openai(base):
        return _openai_chat(
            system=system,
            history=history,
            user=user,
            model=m,
            schema=schema or {},
            temperature=temperature,
            thinking=thinking,
            tag=tag,
            images=images,
            base=base,
        )
    body = _body(system, history, user, m, schema, temperature, thinking, images)

    t0 = time.time()
    try:
        resp = _post(m, body)
    except GeminiError as exc:
        if "thinkingConfig" not in body["generationConfig"] or " 400:" not in f" {exc}":
            raise
        _no_thinking.add(m)
        body["generationConfig"].pop("thinkingConfig")
        resp = _post(m, body)
    if config.LLM_TRACE:
        # Latency is almost entirely output tokens; to see why something is slow, start here.
        u = _usage(resp)
        log.info(
            "%-10s %5.2fs  in=%5s out=%4s think=%4s",
            tag or m,
            time.time() - t0,
            u["in"],
            u["out"],
            u["think"],
        )
    text = _parts_text(resp)
    if schema is None:
        return text
    if not text:
        raise GeminiError(_empty_response_reason(resp))
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise GeminiError(f"model did not return valid JSON: {text[:400]}") from exc
