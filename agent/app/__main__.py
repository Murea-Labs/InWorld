"""`python -m app` starts the server on the host/port from config.yaml."""

import uvicorn

from app import demo
from app.api.routes.health import _ready
from murea_agent import config


def _setup_line() -> str:
    ready = _ready()
    parts = [
        f"language model: {config.LLM_PROVIDER} ({config.GM_MODEL})",
        f"images: {config.IMAGE_PROVIDER}",
        f"video: {config.VIDEO_MODEL}",
    ]
    missing = [f"{name} (MUREA_URL)" if name == "video" else name for name, ok in ready.items() if not ok]
    if missing:
        note = " (the demo world works without them)" if demo.DEMO.exists() else ""
        return (
            "  ".join(parts)
            + f"\n  not configured: {', '.join(missing)} -- add them to .env and restart{note}"
        )
    return "  ".join(parts)


if __name__ == "__main__":
    print(f"Murea Agent on http://{config.HOST}:{config.PORT}\n  {_setup_line()}", flush=True)
    uvicorn.run("app.main:app", host=config.HOST, port=config.PORT, log_level="warning")
