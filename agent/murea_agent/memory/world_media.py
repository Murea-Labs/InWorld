"""Media references shared by export, import and world deletion.

Worlds and co-op rooms store media as bare file names in data/refs, data/clips and
data/frames. Those names come from JSON files that a world import can supply, so every
name is checked here before it is joined onto a media directory.
"""

import json
from pathlib import Path

from murea_agent import config
from murea_agent import reqlog

from murea_agent.memory.world import World

# The file types each media directory holds. refs/ is served to the browser, so a name
# with any other extension (an .html page, say) is never read or written there.
MEDIA_TYPES = {
    "refs": {".png", ".jpg", ".jpeg", ".webp", ".gif"},
    "clips": {".mp4"},
    "frames": {".jpg", ".jpeg", ".png"},
}


def _roots() -> dict[str, Path]:
    return {"refs": config.REFS, "clips": config.CLIPS, "frames": config.FRAMES}


def media_name(kind: str, name: object) -> str | None:
    """`name` if it is a bare file name of a type the `kind` directory holds, else None.

    A bare name has no directory part (`/` or `\\`), is not `.`/`..` and does not start
    with a dot."""
    if (
        not isinstance(name, str)
        or not name
        or name.startswith(".")
        or any(c in name for c in "/\\\0")
        or Path(name).suffix.lower() not in MEDIA_TYPES[kind]
    ):
        return None
    return name


def media_path(kind: str, name: object) -> Path:
    """The file `name` in the `kind` media directory ("refs", "clips" or "frames").

    Raises ValueError for anything but a bare media file name, so a stored name such as
    `../../.env` never reaches the file system."""
    clean = media_name(kind, name)
    if clean is None:
        raise ValueError(f"invalid {kind} file name: {name!r}")
    return _roots()[kind] / clean


def has_media(kind: str, name: object) -> bool:
    """Whether `name` is a bare media file name and that file exists."""
    return media_name(kind, name) is not None and media_path(kind, name).is_file()


def named_media(subjects: dict, places: dict, nodes: dict):
    """(kind, field, value) for every media name a world stores. Values are not checked."""
    for sid, s in subjects.items():
        for key in ("ref", "source_ref"):
            # Items registered by the critic first use the last frame as a temporary ref,
            # replaced once the background image is drawn; source_ref is the real evidence
            # that must still be exported / deleted with the world.
            if isinstance(s, dict) and s.get(key):
                yield "refs", f"subjects[{sid!r}].{key}", s[key]
    for name, p in places.items():
        if isinstance(p, dict) and p.get("ref"):
            yield "refs", f"places[{name!r}].ref", p["ref"]
    for nid, n in nodes.items():
        if not isinstance(n, dict):
            continue
        if n.get("clip"):
            yield "clips", f"nodes[{nid!r}].clip", n["clip"]
        # Extracted frames are named by node id and belong to this world only; they are
        # included for a complete export / delete.
        for key in ("refs", "frames"):
            values = n.get(key) or []
            for value in values if isinstance(values, list) else [values]:
                yield key, f"nodes[{nid!r}].{key}", value


def media_of(w: World) -> set[str]:
    """Every refs/, clips/ and frames/ path this world points at."""
    return _media(w.subjects, w.places, w.nodes)


def _media(subjects: dict, places: dict, nodes: dict) -> set[str]:
    # Names that are not bare media file names are left out: they cannot belong to a file
    # this app wrote, and joining them would reach outside the media directories.
    return {
        f"{kind}/{value}"
        for kind, _, value in named_media(subjects, places, nodes)
        if media_name(kind, value)
    }


def room_media(value):
    """Media a co-op room file points at, including unapplied previews and
    historical inputs absent from its World."""
    found = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key in ("ref", "scene_ref", "source_ref", "clip", "preview_clip") and isinstance(item, str):
                kind = "clips" if key in ("clip", "preview_clip") else "refs"
                if media_name(kind, item):
                    found.add(f"{kind}/{item}")
            elif key in ("refs", "frames") and isinstance(item, list):
                found.update(f"{key}/{name}" for name in item if media_name(key, name))
            else:
                found.update(room_media(item))
    elif isinstance(value, list):
        for item in value:
            found.update(room_media(item))
    return found


def delete_worlds(world_ids: list[str]) -> int:
    """Delete worlds and the media files no other world still references.

    Reference images and clips are shared by filename, so a file is only
    removed once no remaining world or co-op room points at it. If any other
    world or room cannot be read, what it references is unknown, so no media
    is removed at all; the session files are still deleted.
    """
    targets = set(world_ids)
    if not targets:
        return 0
    mine: set[str] = set()
    others: set[str] = set()
    unknown = False
    for path in config.SESSIONS.glob("*.json"):
        try:
            media = media_of(World.load(path.stem))
        except FileNotFoundError:
            continue  # deleted meanwhile
        except Exception:  # noqa: BLE001 -- unreadable: its media are unknown
            unknown |= path.stem not in targets
            continue
        (mine if path.stem in targets else others).update(media)
    for path in (config.DATA / "coop").glob("*.json"):
        try:
            others.update(room_media(json.loads(path.read_text())))
        except FileNotFoundError:
            continue
        except Exception:  # noqa: BLE001
            unknown = True

    removed = 0
    for relative in set() if unknown else mine - others:
        path = media_path(*relative.split("/", 1))  # _media and room_media keep valid names only
        if path.is_file() or path.is_symlink():
            path.unlink()
            removed += 1
    for world_id in targets:
        (config.SESSIONS / f"{world_id}.json").unlink(missing_ok=True)
        reqlog.clear(world_id)
    return removed
