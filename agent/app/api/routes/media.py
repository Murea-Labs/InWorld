"""Serving generated clips and reference images."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from murea_agent import config
from murea_agent.memory import voices

router = APIRouter(tags=["media"])


# Clip and reference file names are unique per content (job id / token_hex) and never
# change. Without cache headers the browser may refetch a whole 2-6 MB file on replay,
# on switching clips or on remount; marked immutable they are all local hits.
# nosniff: an uploaded or imported file is only ever rendered as the type set here.
_IMMUTABLE = {"Cache-Control": "public, max-age=31536000, immutable", "X-Content-Type-Options": "nosniff"}

# Reference images are written with these extensions only (imagegen.sniff). Anything else,
# e.g. an .html file smuggled in through a world import, is never served from this origin.
IMAGE_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
}


@router.get("/clips/{name}")
def clip(name: str) -> FileResponse:
    p = config.CLIPS / Path(name).name
    if p.suffix.lower() != ".mp4" or not p.exists():
        raise HTTPException(404, "clip not found")
    return FileResponse(p, media_type="video/mp4", headers=_IMMUTABLE)


@router.get("/refs/{name}")
def ref(name: str) -> FileResponse:
    p = config.REFS / Path(name).name
    media_type = IMAGE_TYPES.get(p.suffix.lower())
    if media_type is None or not p.exists():
        raise HTTPException(404, "ref not found")
    return FileResponse(p, media_type=media_type, headers=_IMMUTABLE)


# Library voices are library/xxx.wav and uploads sit at the root, so the path must
# accept a slash. Path joining and traversal checks live in voices.path; here we
# only return 404 when it is missing.
@router.get("/voices/{name:path}")
def voice(name: str) -> FileResponse:
    p = voices.path(name)
    if p is None:
        raise HTTPException(404, "voice not found")
    # A voice sample never changes; let the browser cache it so preview clicks do not refetch.
    return FileResponse(p, media_type="audio/wav", headers={"Cache-Control": "public, max-age=86400"})
