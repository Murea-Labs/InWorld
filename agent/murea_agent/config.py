"""Runtime configuration, loaded once from ``config.yaml``.

Everything a deployment can change lives in that one file (see
``config.example.yaml``). Secrets stay out of it: write ``${ENV_NAME}`` and
put the value in the environment or in ``.env``.

The rest of the code reads the module-level constants below, so this module is
the only place that knows the file layout.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = Path(os.environ.get("MUREA_AGENT_CONFIG") or ROOT / "config.yaml")


def _dotenv_value(raw: str) -> str:
    """A quoted value is taken as is; an unquoted one ends at a " #" comment, as in
    Docker Compose and python-dotenv."""
    v = raw.strip()
    if v[:1] in ('"', "'") and v[0] in v[1:]:
        return v[1 : v.index(v[0], 1)]
    return re.split(r"\s#", v, maxsplit=1)[0].rstrip()


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), _dotenv_value(v))


_ENV_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        return _ENV_REF.sub(lambda m: os.environ.get(m.group(1), ""), value)
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


def _read() -> dict[str, Any]:
    # The .env next to the config file (the repo root by default).
    _load_dotenv(CONFIG_PATH.parent / ".env")
    path = CONFIG_PATH
    if not path.exists():
        example = ROOT / "config.example.yaml"
        if "MUREA_AGENT_CONFIG" in os.environ or not example.exists():
            raise RuntimeError(
                f"{CONFIG_PATH} not found. Copy config.example.yaml to config.yaml and fill in your API keys."
            )
        # No config.yaml: the defaults, with keys from .env. That is the normal setup.
        path = example
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise RuntimeError(f"{path} must be a YAML mapping")
    return _expand(raw)


def _get(cfg: dict[str, Any], dotted: str, default: Any = None) -> Any:
    node: Any = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return default if node is None else node


def _str(cfg: dict[str, Any], dotted: str, default: str = "") -> str:
    return str(_get(cfg, dotted, default)).strip()


def _int(cfg: dict[str, Any], dotted: str, default: int) -> int:
    return int(_get(cfg, dotted, default))


_cfg = _read()

# ── server ────────────────────────────────────────────────────────────────
HOST = _str(_cfg, "server.host", "127.0.0.1")
PORT = _int(_cfg, "server.port", 8080)
DATA = (ROOT / _str(_cfg, "server.data_dir", "data")).resolve()
# Import the demo world for users who have no world yet, if one is bundled (see app/demo.py).
DEMO_WORLD = bool(_get(_cfg, "server.demo_world", True))
REFS = DATA / "refs"
CLIPS = DATA / "clips"
SESSIONS = DATA / "sessions"
# A few frames of every clip, shown to the agent on the next beat so it plans
# from what was actually filmed rather than from its own script.
FRAMES = DATA / "frames"
# One locked voice sample per character.
VOICES = DATA / "voices"


# ── credentials ───────────────────────────────────────────────────────────
# Keys may come from config.yaml or straight from the environment / .env, so
# filling in .env alone is enough.
def _key(dotted: str, env: str) -> str:
    return _str(_cfg, dotted) or os.environ.get(env, "").strip()


GEMINI_API_KEY = _key("gemini.api_key", "GEMINI_API_KEY")
GEMINI_BASE = _str(_cfg, "gemini.base_url", "https://generativelanguage.googleapis.com/v1beta").rstrip("/")
OPENAI_KEY = _key("openai.api_key", "OPENAI_API_KEY")
OPENAI_BASE = _str(_cfg, "openai.base_url", "https://api.openai.com/v1").rstrip("/")
# Alibaba Cloud Model Studio: one key for the language model (DeepSeek, Qwen) and images.
DASHSCOPE_API_KEY = _key("dashscope.api_key", "DASHSCOPE_API_KEY")
DASHSCOPE_COMPATIBLE_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# ── language model ────────────────────────────────────────────────────────
# auto picks by which key is set: DashScope, then Gemini, then OpenAI.
LLM_PROVIDER = _str(_cfg, "llm.provider", "auto").lower()
if LLM_PROVIDER == "auto":
    LLM_PROVIDER = (
        "dashscope"
        if DASHSCOPE_API_KEY
        else "gemini"
        if GEMINI_API_KEY
        else "openai"
        if OPENAI_KEY
        else "gemini"
    )
if LLM_PROVIDER not in {"gemini", "openai", "dashscope"}:
    raise RuntimeError(f"llm.provider must be auto, dashscope, gemini or openai, got {LLM_PROVIDER!r}")
if LLM_PROVIDER == "dashscope":
    # DashScope speaks the OpenAI protocol.
    OPENAI_KEY = DASHSCOPE_API_KEY
    OPENAI_BASE = DASHSCOPE_COMPATIBLE_BASE
# The backend that actually serves the calls.
LLM_BACKEND = "gemini" if LLM_PROVIDER == "gemini" else "openai"
_DEFAULT_MODELS = {
    "gemini": ("gemini-3.5-flash-lite", "gemini-3.5-flash"),
    "dashscope": ("deepseek-v4.1-flash", "deepseek-v4.1-flash"),
    "openai": ("", ""),
}[LLM_PROVIDER]
GM_MODEL = _str(_cfg, "llm.models.agent", _DEFAULT_MODELS[0])
if not GM_MODEL:
    raise RuntimeError(
        "llm.provider is openai: set llm.models.agent (and optionally rewriter, text) in config.yaml"
    )
REWRITER_MODEL = _str(_cfg, "llm.models.rewriter", GM_MODEL)
TEXT_MODEL = _str(_cfg, "llm.models.text", _DEFAULT_MODELS[1] or GM_MODEL)
ACTION_AUDIT_MODEL = TEXT_MODEL
_thinking = _get(_cfg, "llm.thinking", 0)
GM_THINKING: int | None = None if _thinking == "" else int(_thinking)
REWRITER_THINKING = GM_THINKING
# SGLang only: bound the whitespace between JSON tokens of structured output.
OPENAI_JSON_MAX_WHITESPACE = max(0, _int(_cfg, "llm.json_max_whitespace", 0))
# A second OpenAI-compatible replica for background calls (option prefetch,
# action audits) so they never queue in front of the player's own request.
OPENAI_BASE_BOOK = _str(_cfg, "llm.background_base_url").rstrip("/")


def _background_base() -> str:
    return OPENAI_BASE_BOOK or OPENAI_BASE


# ── image model ───────────────────────────────────────────────────────────
# auto follows the language model's provider (OpenAI-compatible LLM servers often have no
# images API, so a plain openai LLM falls back to whichever image key is set).
IMAGE_PROVIDER = _str(_cfg, "image.provider", "auto").lower()
if IMAGE_PROVIDER == "auto":
    IMAGE_PROVIDER = (
        LLM_PROVIDER
        if LLM_PROVIDER in ("dashscope", "gemini")
        else "dashscope"
        if DASHSCOPE_API_KEY
        else "gemini"
        if GEMINI_API_KEY
        else "openai"
    )
if IMAGE_PROVIDER not in {"gemini", "openai", "dashscope"}:
    raise RuntimeError(f"image.provider must be auto, gemini, openai or dashscope, got {IMAGE_PROVIDER!r}")
IMAGE_MODEL = _str(
    _cfg,
    "image.model",
    {"gemini": "gemini-3.1-flash-lite-image", "dashscope": "qwen-image-2.0", "openai": "gpt-image-1"}[
        IMAGE_PROVIDER
    ],
)
# OpenAI-compatible image endpoint; defaults to openai.base_url.
IMAGE_OPENAI_BASE = _str(_cfg, "image.base_url").rstrip("/")
# Optional self-hosted image services (comma separated); empty = Gemini.
IMAGE_BASE = _str(_cfg, "image.service_urls").rstrip("/")
# Alibaba Cloud Model Studio (DashScope) image API, e.g. qwen-image-2.0.
IMAGE_DASHSCOPE_KEY = _str(_cfg, "image.api_key") or DASHSCOPE_API_KEY or OPENAI_KEY
IMAGE_DASHSCOPE_BASE = (_str(_cfg, "image.base_url") or "https://dashscope.aliyuncs.com/api/v1").rstrip("/")

# ── video model ───────────────────────────────────────────────────────────
# Your Murea service (inference/ in this repository): its address, and a key
# only if a reverse proxy in front of it asks for one.
MUREA_URL = (_str(_cfg, "video.url") or os.environ.get("MUREA_URL", "")).strip().rstrip("/")
MUREA_API_KEY = _key("video.api_key", "MUREA_API_KEY")
VIDEO_MODEL = "murea"
# Hard limit on any clip: Murea-Fast's positional range is 20 s. Each video model may allow less
# (`VideoProvider.max_seconds`; the full Murea model takes 4-15 s).
VIDEO_HARD_MAX_SECONDS = 20
VIDEO_MIN_SECONDS = max(4, min(VIDEO_HARD_MAX_SECONDS, _int(_cfg, "video.min_seconds", 6)))
VIDEO_MAX_SECONDS = max(VIDEO_MIN_SECONDS, min(VIDEO_HARD_MAX_SECONDS, _int(_cfg, "video.max_seconds", 10)))
VIDEO_SECONDS = VIDEO_MIN_SECONDS
VIDEO_MAX_PROMPT_CHARS = _int(_cfg, "video.max_prompt_chars", 6000)


# One model edit to fit an over-long prompt, then sentence trimming. Measured: a retry only shaved
# tens more characters and never fitted (0 of 9), at 5-8 s each. A timing plan the agent got
# slightly wrong is cheaper to repair than to drop.
VIDEO_PROMPT_BUDGET_TRIES = 1
GM_PLAN_REPAIR_TRIES = 3

# ── voice ─────────────────────────────────────────────────────────────────
TTS_URL = _str(_cfg, "tts.url").rstrip("/")
TTS_KEY = _str(_cfg, "tts.key")

# ── agent ─────────────────────────────────────────────────────────────────
# The language the story is told in: narration, choices, names and dialogue.
# Video prompts are always English; only spoken lines use this language.
LANGUAGE = _str(_cfg, "agent.language", "en").lower()
if LANGUAGE not in {"en", "zh"}:
    raise RuntimeError(f"agent.language must be en or zh, got {LANGUAGE!r}")
LANGUAGE_NAME = {"en": "English", "zh": "Chinese"}[LANGUAGE]
SERVER_OPTION_PREFETCH = bool(_get(_cfg, "agent.prefetch_options", False))
LLM_TRACE = bool(_get(_cfg, "agent.trace_llm", False))
# The director sees the most recent `history_window` story messages (two per
# beat). Older beats stay reachable through its recall tools.
HISTORY_WINDOW = max(4, _int(_cfg, "agent.history_window", 24))
# Director tools: `auto` offers them once the story is longer than the window.
AGENT_TOOLS = _str(_cfg, "agent.tools", "auto").lower()
if AGENT_TOOLS not in {"auto", "on", "off"}:
    raise RuntimeError(f"agent.tools must be auto, on or off, got {AGENT_TOOLS!r}")


# ── co-op (two players) ───────────────────────────────────────────────────
# `single` serves the single-player world; `coop` serves two-player rooms.
WORLD_MODE = _str(_cfg, "mode", "single").lower()
if WORLD_MODE not in {"single", "coop"}:
    raise RuntimeError(f"mode must be single or coop, got {WORLD_MODE!r}")
_COOP_KEYS = {
    "COOP_PUBLIC_URL": "public_url",
    "COOP_OPENING_SECONDS": "opening_seconds",
    "COOP_ACTION_SECONDS": "max_seconds",
    "COOP_ACTION_MIN_SECONDS": "min_seconds",
    "COOP_AI_MODE": "ai_mode",
    "COOP_AI_MODEL": "ai_model",
    "COOP_SPECULATIVE_VIDEO": "speculative_video",
}


def coop_env(name: str, default: str = "") -> str:
    """A co-op setting: environment variable first, then `coop:` in config.yaml.

    Environment overrides exist so tests and one-off runs can flip a switch
    without editing the config file.
    """
    if name in os.environ:
        return os.environ[name]
    value = _get(_cfg, f"coop.{_COOP_KEYS.get(name, name.lower())}")
    if value is None:
        return default
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


COOP_OPENING_SECONDS = max(4, min(VIDEO_HARD_MAX_SECONDS, int(coop_env("COOP_OPENING_SECONDS", "6"))))
COOP_ACTION_SECONDS = max(4, min(VIDEO_HARD_MAX_SECONDS, int(coop_env("COOP_ACTION_SECONDS", "15"))))
COOP_ACTION_MIN_SECONDS = max(4, min(COOP_ACTION_SECONDS, int(coop_env("COOP_ACTION_MIN_SECONDS", "6"))))


def git_sha() -> str:
    """The commit of the running checkout, shown in the health endpoint."""
    import subprocess

    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=3
        ).stdout.strip()
    except Exception:
        return ""


# Captured at start-up: the health endpoint compares it with the checkout so
# a process still running old code is easy to spot.
VERSION = git_sha()

for d in (REFS, CLIPS, SESSIONS, FRAMES, VOICES):
    d.mkdir(parents=True, exist_ok=True)
