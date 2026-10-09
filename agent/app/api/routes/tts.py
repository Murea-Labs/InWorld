"""Read the narration line aloud."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response
from starlette.concurrency import run_in_threadpool

from murea_agent.providers import tts

router = APIRouter(prefix="/api", tags=["tts"])


@router.post("/tts")
async def speak(req: Request) -> Response:
    body = await req.json()
    text = (body.get("text") or "").strip()
    if not text:
        raise HTTPException(400, "empty text")
    if not tts.available():
        raise HTTPException(503, "Voice is not configured (tts.url)")
    try:
        wav = await run_in_threadpool(
            tts.speak,
            text,
            body.get("speaker") or "",
        )
    except tts.TTSError as exc:
        raise HTTPException(502, str(exc)) from exc
    # The same line always sounds the same; let the browser cache it.
    return Response(content=wav, media_type="audio/wav", headers={"Cache-Control": "public, max-age=86400"})
