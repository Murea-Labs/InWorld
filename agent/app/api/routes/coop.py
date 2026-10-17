from __future__ import annotations

import asyncio
from fastapi.responses import StreamingResponse
from typing import Annotated, Literal
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile
from pydantic import BaseModel, Field, StrictInt

from app.api.auth import selected_user
from murea_agent.coop import (
    room as coop,
    ai as coop_ai,
    portrait as coop_portrait,
    intro as coop_intro,
    events as coop_events,
    assets as coop_assets,
)
from murea_agent import reqlog
from murea_agent.coop.messages import msg
from murea_agent.coop import (
    lifecycle as coop_lifecycle,
    immersive as coop_immersive,
    story_display as coop_story_display,
)

router = APIRouter(prefix="/api/coop/rooms", tags=["coop"])


def require_player(request: Request):
    user = selected_user(request)
    if user is None:
        raise HTTPException(401, msg("请先选择或创建你的用户，再加入队伍"))
    return user


Player = Annotated[dict, Depends(require_player)]


class RoomInput(BaseModel):
    title: str = Field(min_length=1, max_length=80)
    setting: str = Field(min_length=1, max_length=2000)
    visual_style: str = Field(default="", max_length=500)
    capacity: StrictInt = Field(default=4, ge=2, le=4)
    mode: Literal["local", "online"] = "online"


class ActionInput(BaseModel):
    target_uid: str | None = None
    action: str = Field(max_length=1200)
    round: StrictInt = Field(ge=1)
    ready: bool = True


class GenerateInput(BaseModel):
    round: StrictInt = Field(ge=1)
    revision: StrictInt = Field(ge=1)
    generation_signature: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class ImmersiveInput(BaseModel):
    session_id: str = Field(min_length=1, max_length=80)


class RollbackInput(BaseModel):
    node_id: str = Field(min_length=1, max_length=100)


@router.post("/{code}/rollback")
def rollback(code: str, body: RollbackInput, user: Player):
    from murea_agent.coop import rollback as coop_rollback

    return coop_rollback.rollback(code, user["id"], body.node_id)


class ImmersiveAdvanceInput(ImmersiveInput):
    round: StrictInt = Field(ge=1)


@router.post("/{code}/immersive/start")
def start_immersive(code: str, body: ImmersiveInput, user: Player):
    return coop_immersive.start(code, user["id"], body.session_id)


@router.post("/{code}/immersive/advance")
def advance_immersive(code: str, body: ImmersiveAdvanceInput, user: Player):
    return coop_immersive.advance(code, user["id"], body.session_id, body.round)


@router.post("/{code}/immersive/stop")
def stop_immersive(code: str, body: ImmersiveInput, user: Player):
    return coop_immersive.stop(code, user["id"], body.session_id)


class AIInput(BaseModel):
    profile: str = "axe"


class PortraitInput(BaseModel):
    target_uid: str | None = None
    name: str = Field(default="", max_length=60)
    description: str = Field(default="", max_length=500)
    randomize: bool = False


class UsePortraitInput(BaseModel):
    request_id: str
    target_uid: str | None = None


AssetKind = Literal["scene", "monster"]


class AssetInput(BaseModel):
    name: str = Field(min_length=1, max_length=60)
    description: str = Field(min_length=1, max_length=500)


class RecommendInput(BaseModel):
    refresh: bool = False


class UseAssetInput(BaseModel):
    request_id: str = Field(min_length=1, max_length=80)


@router.post("/{code}/assets/recommendations")
def recommend_assets(code: str, body: RecommendInput, user: Player):
    return coop_assets.recommend(code, user["id"], body.refresh)


@router.post("/{code}/assets/{kind}/generate", status_code=202)
def generate_asset(code: str, kind: AssetKind, body: AssetInput, user: Player):
    return coop_assets.start(code, user["id"], kind, **body.model_dump())


@router.post("/{code}/assets/{kind}/use")
def use_asset(code: str, kind: AssetKind, body: UseAssetInput, user: Player):
    return coop_assets.use(code, user["id"], kind, body.request_id)


@router.post("/{code}/assets/{kind}/upload")
async def upload_asset(
    code: str,
    kind: AssetKind,
    user: Player,
    name: Annotated[str, Form()],
    description: Annotated[str, Form()],
    file: UploadFile = File(...),
):
    # Check membership before reading an upload or performing image work.
    with coop._edit(code) as room:
        coop_assets._authorize(room, user["id"], kind)
    raw = await file.read(coop.MAX_IMAGE_BYTES + 1)
    return coop_assets.upload(code, user["id"], kind, name, description, raw)


@router.delete("/{code}/assets/{kind}")
def remove_asset(code: str, kind: AssetKind, user: Player):
    return coop_assets.remove(code, user["id"], kind)


@router.get("")
def list_rooms(user: Player):
    return coop.list_rooms(user["id"])


@router.post("", status_code=201)
def create_room(body: RoomInput, user: Player):
    return coop.create_room(user, body.title, body.setting, body.visual_style, body.capacity, body.mode)


@router.get("/{code}")
def get_room(code: str, user: Player):
    return coop.get_room(code, user["id"])


@router.post("/{code}/story/display-zh")
def story_display(code: str, user: Player):
    return coop_story_display.localized_display(code, user["id"])


@router.delete("/{code}")
def delete_room(code: str, user: Player):
    return coop_lifecycle.delete_room(code, user["id"])


@router.get("/{code}/events")
async def events(code: str, user: Player):
    room = await asyncio.to_thread(coop.get_room, code, user["id"])
    return StreamingResponse(
        coop_events.stream(room["code"], user["id"]),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
    )


@router.post("/{code}/join")
def join(code: str, user: Player):
    return coop.join_room(code, user)


@router.get("/{code}/log")
def log(code: str, user: Player, limit: int = Query(default=500, ge=1, le=500)):
    room = coop.get_room(code, user["id"], prepare_intro=False)
    return reqlog.read(room["world_id"], limit=limit)


@router.post("/{code}/intro/retry")
def retry_intro(code: str, user: Player):
    with coop._edit(code) as room:
        coop._host(room, user["id"])
        coop._waiting(room)
        if not coop_intro.eligible(room) and (room.get("chapter_transition") or {}).get("status") != "failed":
            raise HTTPException(409, msg("全员设置形象后才能编排开场"))
    # After the opening, retries use the history-only planner as well; the
    # visual keyframe path is reserved for the initial opening guidance.
    coop_intro.queue(
        code,
        force=True,
        history_only=bool(room.get("opening_node") or room.get("rounds")),
    )
    return coop.get_room(code, user["id"])


@router.post("/{code}/opening/start", status_code=202)
def start_opening(code: str, user: Player):
    return coop.start_opening(code, user["id"])


@router.post("/{code}/leave")
def leave(code: str, user: Player):
    coop.leave_room(code, user["id"])
    return {"ok": True}


@router.post("/{code}/character")
async def character(
    code: str,
    user: Player,
    name: Annotated[str, Form()],
    description: Annotated[str, Form()] = "",
    file: UploadFile | None = File(default=None),
    target_uid: Annotated[str | None, Form()] = None,
):
    coop.get_room(code, user["id"])
    raw = await file.read(coop.MAX_IMAGE_BYTES + 1) if file else None
    return coop.set_reference(code, user["id"], name, description, raw, target_uid=target_uid)


@router.post("/{code}/scene")
async def scene(
    code: str,
    user: Player,
    name: Annotated[str, Form()],
    description: Annotated[str, Form()] = "",
    file: UploadFile | None = File(default=None),
):
    coop.get_room(code, user["id"])
    raw = await file.read(coop.MAX_IMAGE_BYTES + 1) if file else None
    return coop.set_reference(code, user["id"], name, description, raw, scene=True)


@router.put("/{code}/action")
def action(code: str, body: ActionInput, user: Player):
    return coop.submit_action(code, user["id"], body.action, body.round, body.ready, body.target_uid)


@router.post("/{code}/generate", status_code=202)
def generate(code: str, body: GenerateInput, user: Player):
    return coop.generate(
        code, user["id"], body.round, body.revision, expected_signature=body.generation_signature
    )


@router.post("/{code}/ai")
def add_ai(code: str, body: AIInput, user: Player):
    return coop_ai.add(code, user["id"], body.profile)


@router.delete("/{code}/ai/{bot_id}")
def remove_ai(code: str, bot_id: str, user: Player):
    return coop_ai.remove(code, user["id"], bot_id)


@router.post("/{code}/ai/respond")
def respond_ai(code: str, user: Player):
    return coop_ai.retry(code, user["id"])


@router.post("/{code}/portrait", status_code=202)
def portrait(code: str, body: PortraitInput, user: Player):
    return coop_portrait.start(code, user["id"], **body.model_dump())


@router.post("/{code}/portrait/use")
def use_portrait(code: str, body: UsePortraitInput, user: Player):
    return coop_portrait.use(code, user["id"], **body.model_dump())
