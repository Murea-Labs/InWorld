"""Current-user dependency and owned-world lookup helpers."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import Depends, HTTPException, Request

from app.users import DEFAULT_USER_ID, get_user, unsign
from murea_agent.memory.world import World


def user_cookie_name() -> str:
    return "murea_agent_user"


def selected_user(request: Request) -> dict[str, str] | None:
    uid, signed = unsign((request.cookies.get(user_cookie_name()) or "").strip())
    if not uid:
        return None
    user = get_user(uid)
    # A bad signature is only acceptable for a user without a password (nothing to protect). Users
    # with a password must present a valid signature, otherwise editing a cookie would bypass it.
    if user is None or (not signed and user.get("locked")):
        return None
    return user


def current_user(request: Request) -> dict[str, str]:
    """Use the selected user; no cookie falls back to legacy/default API access.

    The browser gate uses /api/users/me, which deliberately does not apply this
    fallback. Keeping it here preserves CLI clients and old route tests.
    """
    return selected_user(request) or get_user(DEFAULT_USER_ID)  # type: ignore[return-value]


CurrentUser = Annotated[dict[str, Any], Depends(current_user)]


def load_owned_world(world_id: str, user: dict[str, Any]) -> World:
    try:
        world = World.load(world_id)
    except FileNotFoundError:
        raise HTTPException(404, "world not found") from None
    if world.owner_id != user["id"]:
        # Do not reveal whether another user's world exists.
        raise HTTPException(404, "world not found")
    return world
