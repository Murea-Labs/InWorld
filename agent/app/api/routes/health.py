from __future__ import annotations

from fastapi import APIRouter

from murea_agent import config
from murea_agent.providers import tts, video
from murea_agent.memory import voices

router = APIRouter(tags=["health"])


@router.get("/api/health")
def health() -> dict:
    head = config.git_sha()
    return {
        "version": config.VERSION,
        # The commit at process start differs from the working tree = old code is running.
        # Python does not hot-reload; a backend change needs a restart.
        "stale": bool(head and config.VERSION and head != config.VERSION),
        "head": head,
        "video_online": video.health(),
        # The frontend uses this to decide whether the voice switch exists at all. Not configured
        # means no feature, not an error.
        "tts": tts.available(),
        # ⚠️ Configured ≠ alive; report both. Reporting only `available()` once lit the switch while
        # the service was unreachable, and turning it on silently did nothing.
        "tts_up": tts.health(),
        # Three voice library counts to compare: `library` is everything on disk, `pickable` is what
        # the director may pick, `in_prompt` is how many were rendered into the prompt **at process
        # start**. `in_prompt != pickable` means the library was synced after start and the prompt
        # is stale; restart. Otherwise this failure is silent: the director is told there is no
        # library, sets no voices, and the video model invents a voice every clip.
        "voices": {
            "library": len(voices.library()),
            "pickable": len(voices.gm_pool()),
            "in_prompt": voices.prompt_pool_size(),
        },
        # The video service's address is not sent: health needs no login, and the address is
        # often an internal host or a tunnel.
        "video": video.describe(),
        "continuation": video.supports_continuation(),
        "continuation_note": video.continuation_status(),
        # ⚠️ This is what the header pill shows. It used to show `TEXT_MODEL` while the agent ran on
        # `GM_MODEL`, so after switching to another model the UI still named the old one. Show the
        # one actually in use.
        "text_model": config.GM_MODEL,
        "gm_model": config.GM_MODEL,
        "backend": config.LLM_BACKEND,
        # `config._background_base()` says background work must never compete with the visible path:
        # during playback the server runs option previews and a critic call, and on the same replica
        # a player's request would queue behind them. Report it so it is visible without reading
        # .env.
        "background_shares_visual_replica": bool(
            config.OPENAI_BASE and config._background_base().rstrip("/") == config.OPENAI_BASE.rstrip("/")
        ),
        "rewriter_model": config.REWRITER_MODEL,
        "image_model": config.IMAGE_MODEL,
        "gemini_key": bool(config.GEMINI_API_KEY),
        "language": config.LANGUAGE,
        # Which providers have credentials. With any of them missing the UI says how to
        # set them up instead of failing on the first beat; the demo world still plays.
        "ready": _ready(),
        "mode": config.WORLD_MODE,
    }


def _ready() -> dict[str, bool]:
    llm = bool(config.OPENAI_KEY if config.LLM_BACKEND == "openai" else config.GEMINI_API_KEY)
    image = {
        "gemini": bool(config.GEMINI_API_KEY),
        "openai": bool(config.OPENAI_KEY),
        "dashscope": bool(config.IMAGE_DASHSCOPE_KEY),
    }.get(config.IMAGE_PROVIDER, False)
    return {"llm": llm, "image": image, "video": video.model().ready()}
