"""FastAPI app assembly."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import (
    assets,
    health,
    media,
    transfer,
    tts,
    turns,
    users,
    voices,
    worlds,
)

DIST = Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"


def create_app() -> FastAPI:
    from murea_agent import config

    if not logging.getLogger().handlers:
        # uvicorn configures only its own loggers; give the murea_agent.* ones a handler too.
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
        # httpx logs every request at INFO; uvicorn's access log already covers ours.
        logging.getLogger("httpx").setLevel(logging.WARNING)
    app = FastAPI(title="Murea-Mind")
    if config.WORLD_MODE == "coop":
        from app.api.routes import coop

        app.include_router(coop.router)

    @app.middleware("http")
    async def private_responses(request, call_next):
        response = await call_next(request)
        path = request.url.path
        root = request.scope.get("root_path", "").rstrip("/")
        if root and path.startswith(root + "/"):
            path = path[len(root) :]
        if path.startswith(("/api/users", "/api/worlds", "/api/coop/")):
            response.headers["Cache-Control"] = "no-store"
            response.headers.append("Vary", "Cookie")
        return response

    for r in (
        health.router,
        users.router,
        worlds.router,
        transfer.router,
        assets.router,
        turns.router,
        media.router,
        tts.router,
        voices.router,
    ):
        app.include_router(r)

    if DIST.is_dir():
        # Asset file names carry a content hash, so any change gives a new name; browsers
        # may cache them forever and never get a stale one.
        app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

        @app.get("/")
        def index() -> FileResponse:
            # ⚠️ index.html must be no-store. It is the only file whose name never changes, and it
            # names this build's hashed assets. Without Cache-Control the browser caches it
            # heuristically, and a deploy looks like nothing changed after a refresh.
            return FileResponse(
                DIST / "index.html",
                headers={"Cache-Control": "no-store, must-revalidate"},
            )

    return app
