"""Local user list. Worlds are separated per user.

⚠️ Passwords are **optional** on purpose: users created before passwords existed have none,
   and requiring one would lock them out. The rule is "whoever set one must type it": a user
   with a password must give the right one; others enter with one click.

   Only a PBKDF2 hash with a random per-user salt is kept; the plain text is never written or logged.
"""

from __future__ import annotations

from datetime import datetime, UTC
import hashlib
import hmac
import json
import secrets
import threading
import uuid
from typing import Any

from murea_agent import config

from murea_agent.memory.world import DEFAULT_USER_ID

DEFAULT_USER_NAME = "默认用户" if config.LANGUAGE == "zh" else "Default"
MAX_NAME_LENGTH = 40
MIN_PASSWORD_LENGTH = 6
# PBKDF2 rounds. About 60 ms per check; login is not a hot path, slow is fine.
_ROUNDS = 200_000

_lock = threading.Lock()


def _secret() -> bytes:
    """The cookie signing key. Generated on first use and kept.

    Without it passwords would be pointless: the cookie would hold a plain user id that anyone
    could edit. Replacing this file sends everyone back to the user picker.
    """
    path = config.DATA / "user_secret"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        # write-then-chmod is briefly 644; O_EXCL creates it as 600 directly
        import os

        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(secrets.token_hex(32))
    return path.read_text().strip().encode()


def sign(user_id: str) -> str:
    return f"{user_id}.{hmac.new(_secret(), user_id.encode(), hashlib.sha256).hexdigest()}"


def unsign(token: str) -> tuple[str, bool]:
    """cookie -> (user_id, signature valid).

    Old cookies have no signature part; they return `(id, False)` and the caller decides:
    allowed for users without a password, refused for users with one.
    """
    uid, _, sig = (token or "").partition(".")
    if not sig:
        return uid, False
    want = hmac.new(_secret(), uid.encode(), hashlib.sha256).hexdigest()
    return uid, hmac.compare_digest(want, sig)


def _digest(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), _ROUNDS).hex()


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _default_user() -> dict[str, Any]:
    return {"id": DEFAULT_USER_ID, "name": DEFAULT_USER_NAME, "created": ""}


def _path():
    # Tests replace config.DATA at runtime, so resolve the path on every call.
    return config.DATA / "users.json"


def normalize_name(value: object) -> str:
    name = " ".join(str(value or "").strip().split())
    if not name:
        raise ValueError("The user name cannot be empty")
    if len(name) > MAX_NAME_LENGTH:
        raise ValueError(f"The user name cannot be longer than {MAX_NAME_LENGTH} characters")
    return name


def _read_unlocked() -> list[dict[str, Any]]:
    path = _path()
    try:
        raw: Any = json.loads(path.read_text())
    except FileNotFoundError:
        raw = []
    except (json.JSONDecodeError, OSError):
        raw = []

    users: list[dict[str, Any]] = [_default_user()]
    seen = {DEFAULT_USER_ID}
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            uid = str(item.get("id") or "").strip()
            name = str(item.get("name") or "").strip()
            if not uid or not name or uid in seen:
                continue
            seen.add(uid)
            users.append(
                {
                    "id": uid,
                    "name": name,
                    "created": str(item.get("created") or ""),
                    # Users without a password have both fields empty; empty = one click to enter.
                    "salt": str(item.get("salt") or ""),
                    "hash": str(item.get("hash") or ""),
                }
            )
    return users


def _public(user: dict[str, str]) -> dict[str, Any]:
    """The public view. The hash and the salt never leave this module.

    `locked` is all the UI needs to know: whether this user needs a password.
    """
    return {
        "id": user["id"],
        "name": user["name"],
        "created": user.get("created", ""),
        "locked": bool(user.get("hash")),
    }


def list_users() -> list[dict[str, Any]]:
    with _lock:
        return [_public(user) for user in _read_unlocked()]


def get_user(user_id: str) -> dict[str, Any] | None:
    with _lock:
        return next((_public(user) for user in _read_unlocked() if user["id"] == user_id), None)


def authenticate(user_id: str, password: str) -> dict[str, Any]:
    """Select this user. A user with a password must give the right one; others enter directly."""
    with _lock:
        user = next((u for u in _read_unlocked() if u["id"] == user_id), None)
    if user is None:
        raise LookupError("user not found")
    if not user.get("hash"):
        return _public(user)
    if not password:
        raise ValueError("This user has a password; enter it")
    if not hmac.compare_digest(_digest(password, user["salt"]), user["hash"]):
        raise ValueError("Wrong password")
    return _public(user)


def create_user(value: object, password: str = "") -> dict[str, Any]:
    name = normalize_name(value)
    password = str(password or "")
    if password and len(password) < MIN_PASSWORD_LENGTH:
        raise ValueError(f"The password must be at least {MIN_PASSWORD_LENGTH} characters")
    with _lock:
        users = _read_unlocked()
        if any(user["name"].casefold() == name.casefold() for user in users):
            raise ValueError("This user name already exists")
        user = {"id": f"u-{uuid.uuid4().hex[:12]}", "name": name, "created": _now()}
        if password:
            user["salt"] = secrets.token_hex(16)
            user["hash"] = _digest(password, user["salt"])
        users.append(user)
        path = _path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(
                [{k: v for k, v in u.items() if v or k in ("id", "name", "created")} for u in users],
                ensure_ascii=False,
                indent=2,
            )
        )
        tmp.replace(path)
        return _public(user)


def delete_user(user_id: str, password: str = "") -> dict[str, Any]:
    """Remove a non-default profile after checking its optional password."""
    if user_id == DEFAULT_USER_ID:
        raise ValueError("The default user cannot be deleted")
    with _lock:
        users = _read_unlocked()
        user = next((u for u in users if u["id"] == user_id), None)
        if user is None:
            raise LookupError("user not found")
        if user.get("hash"):
            if not password:
                raise ValueError("This user has a password; enter it")
            if not hmac.compare_digest(_digest(password, user["salt"]), user["hash"]):
                raise ValueError("Wrong password")
        path = _path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(
                [
                    {k: v for k, v in existing.items() if v or k in ("id", "name", "created")}
                    for existing in users
                    if existing["id"] != user_id
                ],
                ensure_ascii=False,
                indent=2,
            )
        )
        tmp.replace(path)
        return _public(user)
