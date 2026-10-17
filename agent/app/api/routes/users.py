"""Pick a user to enter. Users with a password must type it; others enter with one click (see app/users.py)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request, Response

from murea_agent import config
from app import users
from murea_agent.memory.world import World
from murea_agent.memory.world_media import delete_worlds
from app.api.auth import selected_user, user_cookie_name

router = APIRouter(prefix="/api/users", tags=["users"])


def _set_cookie(request: Request, response: Response, user_id: str) -> None:
    # Store the signed value, not the bare id; anyone can edit a bare id into someone else's.
    response.set_cookie(
        user_cookie_name(),
        users.sign(user_id),
        httponly=True,
        samesite="lax",
        path="/",
        secure=request.url.scheme == "https",
        max_age=60 * 60 * 24 * 365,
    )


@router.get("")
def list_users() -> list[dict[str, Any]]:
    return users.list_users()


@router.post("")
async def create_user(request: Request, response: Response) -> dict[str, Any]:
    body = await request.json()
    if not isinstance(body, dict):
        raise HTTPException(400, "body must be an object")
    try:
        user = users.create_user(body.get("name"), body.get("password") or "")
    except ValueError as exc:
        raise HTTPException(409 if "already exists" in str(exc) else 400, str(exc)) from exc
    _set_cookie(request, response, user["id"])
    return user


@router.get("/me")
def me(request: Request) -> dict[str, Any]:
    user = selected_user(request)
    if user is None:
        raise HTTPException(401, "No user selected")
    return user


@router.post("/{user_id}/select")
async def select(user_id: str, request: Request, response: Response) -> dict[str, Any]:
    # The frontend sends no body for users without a password, so missing JSON is normal here.
    try:
        body = await request.json()
    except Exception:
        body = {}
    password = (body or {}).get("password") or "" if isinstance(body, dict) else ""
    try:
        user = users.authenticate(user_id, password)
    except LookupError:
        raise HTTPException(404, "user not found") from None
    except ValueError as exc:
        raise HTTPException(401, str(exc)) from exc
    _set_cookie(request, response, user_id)
    return user


@router.post("/logout")
def logout(response: Response) -> dict[str, bool]:
    response.delete_cookie(user_cookie_name(), path="/")
    return {"ok": True}


@router.delete("/{user_id}")
async def delete_user(user_id: str, request: Request, response: Response) -> dict[str, Any]:
    if user_id == users.DEFAULT_USER_ID:
        raise HTTPException(409, "The default user cannot be deleted")
    try:
        body = await request.json()
    except Exception:
        body = {}
    password = (body or {}).get("password") or "" if isinstance(body, dict) else ""
    try:
        candidate = users.authenticate(user_id, password)
    except LookupError:
        raise HTTPException(404, "user not found") from None
    except ValueError as exc:
        raise HTTPException(401, str(exc)) from exc

    if config.WORLD_MODE == "coop":
        # Leave co-op rooms first; a room with no other human player is removed
        # together with its world.
        from murea_agent.coop import lifecycle as coop_lifecycle

        coop_lifecycle.leave_user_rooms(user_id)
    # Worlds are private to this profile, so remove them together with the
    # user instead of leaving orphaned data behind.
    delete_worlds([w["id"] for w in World.list_all(owner_id=user_id)])

    current = selected_user(request)
    try:
        users.delete_user(user_id, password)
    except LookupError:
        raise HTTPException(404, "user not found") from None
    except ValueError as exc:
        status = 409 if user_id == users.DEFAULT_USER_ID else 401
        raise HTTPException(status, str(exc)) from exc
    if current and current["id"] == user_id:
        response.delete_cookie(user_cookie_name(), path="/")
    return {"ok": True, "user": candidate}
