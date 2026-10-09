"""Delete a host's co-op world while excluding active workers and shared media."""

from contextlib import ExitStack
import fcntl
import json

from fastapi import HTTPException

from murea_agent import config
from app.api.sse import GenerationBusy, mutation_slot
from murea_agent.coop import room as coop, events as coop_events
from murea_agent import reqlog
from murea_agent.memory.world import World
from murea_agent.memory.world_media import media_of, media_path, room_media
from murea_agent.coop.messages import msg


def leave_user_rooms(uid: str) -> None:
    """Remove a user from every active room before their profile is deleted.

    A room with no other human players has no useful owner after this user
    leaves, so delete the room (and its co-op world) in one locked operation.
    Rooms with other humans use the normal leave path, which promotes a new
    host when necessary.
    """
    for path in coop._directory().glob("*.json"):
        try:
            room = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if room.get("status") == "closed":
            continue
        members = room.get("members")
        if not isinstance(members, list) or not any(
            isinstance(member, dict) and member.get("user_id") == uid for member in members
        ):
            continue

        # Ignore/remove an orphaned room record left by an interrupted older
        # version.  It has no world to preserve and must not make deletion of a
        # user fail with an internal error.
        if not isinstance(room.get("world_id"), str) or not room["world_id"]:
            path.unlink(missing_ok=True)
            coop_events.notify(path.stem)
            continue

        humans = [
            member
            for member in members
            if isinstance(member, dict) and member.get("kind", "human") == "human"
        ]
        if len(humans) <= 1:
            # The host is normally the only human.  Using the persisted host
            # when available also repairs old rooms whose membership metadata
            # is slightly stale.
            actor = room.get("host_id") or uid
            try:
                delete_room(path.stem, actor)
            except HTTPException as exc:
                # A malformed legacy room cannot be deleted through the normal
                # lifecycle path; leave it alone rather than blocking profile
                # deletion for an unrelated broken file.
                if exc.status_code == 404:
                    continue
                raise
        else:
            try:
                coop.leave_room(path.stem, uid)
            except KeyError:
                # Fill the same membership change for a partially migrated
                # room whose old member records lack newer optional fields.
                # Real rooms take the normal path above; this keeps cleanup
                # idempotent for legacy data instead of surfacing a 500.
                with coop._edit(path.stem) as current:
                    current_members = current.get("members", [])
                    current["members"] = [
                        member
                        for member in current_members
                        if isinstance(member, dict) and member.get("user_id") != uid
                    ]
                    remaining = [
                        member
                        for member in current["members"]
                        if isinstance(member, dict) and member.get("kind", "human") == "human"
                    ]
                    if not remaining:
                        current["status"] = "closed"
                    elif current.get("host_id") == uid:
                        new_host = remaining[0].get("user_id")
                        current["host_id"] = new_host
                        world_id = current.get("world_id")
                        if isinstance(world_id, str) and new_host:
                            try:
                                world = World.load(world_id)
                                world.owner_id = new_host
                                world.save()
                            except (FileNotFoundError, ValueError, OSError):
                                pass
                    current.setdefault("revision", 1)
                    current.setdefault("state_revision", current["revision"])


def delete_room(code, uid):
    code = coop._code(code)
    # Keep the room lock file: unlinking a lock can split queued requests across
    # different inodes. persist=False prevents recreating the deleted JSON.
    with coop._edit(code, persist=False) as room, ExitStack() as stack:
        coop._host(room, uid)
        locks = [f"{code}.{job}.lock" for job in ("run", "intro", "ai", "recommend", "prefetch")]
        locks.extend(
            (
                f"{code}.opening-assets.lock",
                f"{code}.round-boss.lock",
                f"{code}.observation.lock",
                f"{code}.story.lock",
            )
        )
        locks.extend(path.name for path in coop._directory().glob(f"{code}.action-*.lock"))
        # Previews can be removed while their worker is still running, so their
        # status fields alone are insufficient to establish that drawing stopped.
        locks.append("portrait-render.lock")
        for name in locks:
            guard = stack.enter_context((coop._directory() / name).open("a"))
            try:
                fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise HTTPException(409, msg("生成任务仍在进行，请完成后再删除世界")) from None
        try:
            stack.enter_context(mutation_slot(room["world_id"]))
        except GenerationBusy:
            raise HTTPException(409, msg("世界正在生成，请完成后再删除")) from None

        session = config.SESSIONS / f"{room['world_id']}.json"
        mine = room_media(room)
        if session.exists():
            mine.update(media_of(World.load(room["world_id"])))
        others = set()
        for path in config.SESSIONS.glob("*.json"):
            if path != session:
                others.update(media_of(World.load(path.stem)))
        for path in coop._directory().glob("*.json"):
            if path.stem != code:
                others.update(room_media(json.loads(path.read_text())))

        removed = 0
        for relative in mine - others:
            path = media_path(*relative.split("/", 1))  # media_of and room_media keep valid names only
            if path.is_file() or path.is_symlink():
                path.unlink()
                removed += 1
        session.unlink(missing_ok=True)
        reqlog.clear(room["world_id"])
        (coop._directory() / f"{code}.json").unlink()
        coop_events.notify(code)
    return {"ok": True, "removed_files": removed}
