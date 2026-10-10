from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from murea_agent import config
from app import demo, option_prefetch
from murea_agent import reqlog
from murea_agent.providers import video
from murea_agent.director import worldgen
from murea_agent.memory.world import World
from murea_agent.memory.world_media import delete_worlds
from app.api.auth import CurrentUser, load_owned_world
from app.api.sse import GenerationBusy, mutation_slot

router = APIRouter(prefix="/api/worlds", tags=["worlds"])


@router.get("")
def list_worlds(user: CurrentUser) -> list[dict]:
    worlds = World.list_all(owner_id=user["id"])
    if config.DEMO_WORLD and config.WORLD_MODE == "single" and demo.seed_demo(user["id"], bool(worlds)):
        worlds = World.list_all(owner_id=user["id"])
    return worlds


@router.post("")
async def create_world(req: Request, user: CurrentUser) -> dict:
    body = await req.json()
    chosen = (body.get("video_model") or "").strip() or config.VIDEO_MODEL
    if chosen not in video.configured():
        raise HTTPException(400, f"Video model {chosen!r} is not configured")
    w = World(
        owner_id=user["id"],
        title=(body.get("title") or ("未命名世界" if config.LANGUAGE == "zh" else "Untitled world")).strip(),
        setting=(body.get("setting") or "").strip(),
        # Optional. A mainline written by the user is followed; empty lets the director pick
        # one in the first beat.
        mainline=(body.get("mainline") or "").strip()[:200],
        visual_style=(body.get("visual_style") or "").strip()[:2000],
        video_model=chosen,
    )
    w.save()
    return asdict(w)


@router.post("/random")
async def random_world(req: Request) -> dict:
    """Roll a world setting. Nothing is saved; the caller decides whether to create it.

    ⚠️ Must be registered before `/{wid}`. FastAPI matches in order, otherwise "random"
    is taken as a world id and returns 404.
    """
    try:
        body = await req.json()
    except Exception:
        body = {}
    styles = (body or {}).get("styles") if isinstance(body, dict) else None
    idea = body.get("idea", "") if isinstance(body, dict) else ""
    if not isinstance(idea, str) or len(idea) > 2000:
        raise HTTPException(400, "World idea must be text of at most 2000 characters")
    try:
        return await run_in_threadpool(worldgen.spin, styles if isinstance(styles, list) else None, idea)
    except Exception as exc:
        raise HTTPException(502, str(exc)) from exc


@router.get("/{wid}")
def get_world(wid: str, user: CurrentUser) -> dict:
    world = load_owned_world(wid, user)
    option_prefetch.warm(world, source="world_load")
    return asdict(world)


@router.patch("/{wid}")
async def update_world(wid: str, req: Request, user: CurrentUser) -> dict:
    """Update the small set of world metadata that remains editable."""
    body = await req.json()
    if not isinstance(body, dict) or not isinstance(body.get("title"), str):
        raise HTTPException(400, "title required")
    title = body["title"].strip()
    if not title:
        raise HTTPException(400, "The world name cannot be empty")
    if len(title) > 80:
        raise HTTPException(400, "The world name cannot be longer than 80 characters")

    try:
        with mutation_slot(wid):
            w = load_owned_world(wid, user)
            w.title = title
            w.save()
            return asdict(w)
    except GenerationBusy:
        raise HTTPException(409, "A clip is being generated; rename the world after it finishes") from None


@router.delete("/{wid}")
def delete_world(wid: str, user: CurrentUser) -> dict:
    """Delete a world and the media only it referenced."""
    load_owned_world(wid, user)
    return {"ok": True, "removed_files": delete_worlds([wid])}


@router.post("/{wid}/rollback")
def rollback(wid: str, user: CurrentUser) -> dict:
    w = load_owned_world(wid, user)
    ok = w.rollback()
    w.save()
    return {"ok": ok, "head": w.head, "here": w.here}


@router.get("/{wid}/movie")
def movie(wid: str, user: CurrentUser) -> JSONResponse:
    """Lineage root -> head: the film as currently assembled."""
    w = load_owned_world(wid, user)
    return JSONResponse([asdict(n) for n in w.lineage()])


@router.get("/{wid}/log")
def world_log(wid: str, user: CurrentUser, limit: int = 200) -> list[dict]:
    """This world's outbound call log, newest first.

    `load_owned_world` is required: the log contains raw prompts, so access must match the
    world itself. Other users' worlds are always 404; not even their existence shows.
    """
    load_owned_world(wid, user)
    return reqlog.read(wid, limit=max(1, min(int(limit or 200), 500)))


@router.delete("/{wid}/log")
def clear_world_log(wid: str, user: CurrentUser) -> dict:
    load_owned_world(wid, user)
    reqlog.clear(wid)
    return {"ok": True}
