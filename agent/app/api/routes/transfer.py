"""Exporting a world to a zip and importing one back.

A world is a session JSON plus the media files it points at, so moving one between
installations means moving all three. The zip carries them
together; import rewrites nothing except the world id.
"""

from __future__ import annotations

import io
import json
import os
import tempfile
import uuid
import zipfile
from dataclasses import asdict, fields
from urllib.parse import quote
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from murea_agent.memory.world import World
from murea_agent.memory.world_media import MEDIA_TYPES, media_name, media_of, media_path, named_media
from murea_agent.util import _say
from app.api.auth import CurrentUser, load_owned_world

router = APIRouter(prefix="/api/worlds", tags=["transfer"])

MAX_ZIP = 512 * 1024 * 1024
# Total bytes an archive may unpack to; a zip bomb is refused instead of filling the disk.
MAX_UNPACKED = 4 * MAX_ZIP


@router.get("/{wid}/export")
def export_world(wid: str, user: CurrentUser) -> StreamingResponse:
    w = load_owned_world(wid, user)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("world.json", json.dumps(asdict(w), ensure_ascii=False, indent=2))
        for rel in sorted(media_of(w)):
            f = media_path(*rel.split("/", 1))
            # Only regular files that really sit in the media directory are packed; a
            # symlink pointing elsewhere is left out.
            real = f.resolve()
            if real.parent == f.parent.resolve() and real.is_file():
                z.write(real, rel)
    buf.seek(0)

    # HTTP headers are latin-1, and titles are usually Chinese, so the ASCII
    # `filename` falls back to the id and the real name goes in `filename*`.
    pretty = quote(f"{w.title or 'world'}-{wid}.zip")
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=\"{wid}.zip\"; filename*=UTF-8''{pretty}"},
    )


@router.post("/import")
async def import_world(user: CurrentUser, file: UploadFile = File(...)) -> dict:
    raw = await file.read(MAX_ZIP + 1)
    if len(raw) > MAX_ZIP:
        raise HTTPException(413, f"Archive too large (>{MAX_ZIP // 1024 // 1024} MB)")
    try:
        w, written = import_archive(raw, user["id"])
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    # Count clips on the chain, not all nodes: undone clips stay on disk, and saying
    # "5 clips" for a 4-clip film looks like the import lost something. Same rule as
    # the segments count in the world list.
    return {
        "id": w.id,
        "title": w.title,
        "segments": len(w.lineage()),
        "places": len(w.places),
        "subjects": len(w.subjects),
        "files_written": written,
    }


def import_archive(raw: bytes, owner_id: str) -> tuple[World, int]:
    """Unpack an exported world zip into the data directory as a new world of
    `owner_id`. Returns the world and how many media files were written."""
    try:
        z = zipfile.ZipFile(io.BytesIO(raw))
    except zipfile.BadZipFile:
        raise ValueError("Not a valid zip file") from None
    try:
        data = json.loads(z.read("world.json"))
    except KeyError:
        raise ValueError("No world.json in the archive; this is not an exported world") from None

    if not isinstance(data, dict):
        raise ValueError("world.json must contain a JSON object")

    # New id so importing into the environment it came from does not overwrite.
    data["id"] = f"w{uuid.uuid4().hex[:8]}"
    data["title"] = data.get("title") or _say("导入的世界", "Imported world")
    # The archive may have come from another user or another installation. It
    # always becomes the importing user's world in this environment.
    data["owner_id"] = owner_id
    # Same rule as World.load: fields from a newer checkout are dropped, not fatal.
    known = {f.name for f in fields(World)}
    try:
        w = World(**{key: value for key, value in data.items() if key in known})
        # Imports can come from a pre-dynamic-state checkout.  Normalize the asset
        # records before writing so the first subsequent turn sees the same shape.
        w._migrate_dynamic_state()
        _check_media_names(w)
    except (TypeError, ValueError, AttributeError, KeyError) as exc:
        raise ValueError(f"world.json is not a valid world: {exc}") from None

    entries = []
    for info in z.infolist():
        if info.filename == "world.json" or info.is_dir():
            continue
        parts = Path(info.filename).parts
        if len(parts) != 2 or parts[0] not in MEDIA_TYPES:
            continue  # ignore anything outside the media directories
        if media_name(parts[0], parts[1]) is None:
            continue  # not a media file this app writes
        entries.append((info, media_path(parts[0], parts[1])))
    if sum(info.file_size for info, _ in entries) > MAX_UNPACKED:
        raise ValueError(f"Archive unpacks to more than {MAX_UNPACKED // 1024 // 1024} MB")

    written = 0
    budget = MAX_UNPACKED
    for info, dest in entries:
        if dest.exists():  # same filename means same content
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=dest.parent, prefix=".import-", suffix=".tmp")
        try:
            # Copied in chunks and counted, since the sizes in the zip header can lie.
            with z.open(info) as src, os.fdopen(fd, "wb") as out:
                while chunk := src.read(1 << 20):
                    budget -= len(chunk)
                    if budget < 0:
                        raise ValueError(f"Archive unpacks to more than {MAX_UNPACKED // 1024 // 1024} MB")
                    out.write(chunk)
            os.replace(tmp, dest)
        except zipfile.BadZipFile as exc:
            raise ValueError(f"Damaged archive: {exc}") from None
        finally:
            Path(tmp).unlink(missing_ok=True)
        written += 1

    w.save()
    return w, written


def _check_media_names(w: World) -> None:
    """Refuse a world whose media names are not bare file names of an allowed type.

    Export packs the files these names point at, and turns and co-op rooms read them,
    so a name with a directory part, `..` or a leading dot must never get in."""
    for kind, where, value in named_media(w.subjects, w.places, w.nodes):
        if media_name(kind, value) is None:
            raise ValueError(f"{where} is not a plain {kind} file name: {value!r}")
