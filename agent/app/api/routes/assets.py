"""Reference images: characters, places, and standalone previews."""

from __future__ import annotations

import io
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from starlette.concurrency import run_in_threadpool

from murea_agent import config
from murea_agent.providers import imagegen
from murea_agent.memory import voices
from murea_agent.memory.world_media import has_media
from murea_agent.providers.imagegen import ImageRefused
from app.api.auth import CurrentUser, load_owned_world
from app.api.sse import GenerationBusy, mutation_slot

router = APIRouter(tags=["assets"])

# An uploaded reference image is kept as it is (Murea-Fast gets a downscaled copy); a 48 MP
# phone photo is about 20 MB.
MAX_IMAGE_BYTES = 25 * 1024 * 1024


def _voice_form(voice_id: str, voice_ref: str) -> dict | None:
    """The two voice fields of the form -> the record stored on the subject. Both empty means no voice."""
    try:
        return voices.choose(voice_id, voice_ref)
    except voices.VoiceError as exc:
        raise HTTPException(404, str(exc)) from exc


def _readable_image(raw: bytes) -> bool:
    """A PNG, JPEG, WebP or GIF that PIL can read (the types refs/ is served with)."""
    try:
        from PIL import Image

        with Image.open(io.BytesIO(raw)) as im:
            kind = im.format
            im.verify()
        return kind in ("PNG", "JPEG", "WEBP", "GIF")
    except Exception:  # noqa: BLE001 -- anything PIL cannot read is not a reference image
        return False


async def _resolve_ref(
    *,
    ref: str,
    image: UploadFile | None,
    image_prompt: str,
    kind: str,
    prefix: str,
    visual_style: str = "",
) -> str:
    """Return the filename to use, drawing or storing the image if needed."""
    if ref and has_media("refs", Path(ref).name):
        return Path(ref).name  # already drawn via /api/imagine
    if image is not None:
        raw = await image.read(MAX_IMAGE_BYTES + 1)
        if len(raw) > MAX_IMAGE_BYTES:
            raise HTTPException(413, f"Image too large (>{MAX_IMAGE_BYTES // 1024 // 1024} MB)")
        if not _readable_image(raw):
            raise HTTPException(400, "Upload a JPG, PNG, WebP or GIF image")
    elif image_prompt.strip():
        try:
            raw = await run_in_threadpool(
                imagegen.generate,
                image_prompt.strip(),
                kind=kind,
                visual_style=visual_style,
            )
        except ImageRefused as exc:
            raise HTTPException(422, str(exc)) from exc
    else:
        raise HTTPException(400, "need either image, image_prompt or ref")
    # Detect the type from the bytes, not the file name: image models often return JPEG
    fn = f"{prefix}_{uuid.uuid4().hex[:8]}{imagegen.sniff(raw)[0]}"
    (config.REFS / fn).write_bytes(raw)
    return fn


@router.post("/api/imagine")
async def imagine(req: Request) -> dict:
    """Draw a reference image without attaching it, so the UI can preview and retry."""
    body = await req.json()
    prompt = (body.get("prompt") or "").strip()
    if not prompt:
        raise HTTPException(400, "prompt required")
    kind = body.get("kind") if body.get("kind") in imagegen.SUFFIX else "place"
    visual_style = (body.get("visual_style") or "").strip()[:2000]
    started = time.time()
    try:
        raw = await run_in_threadpool(imagegen.generate, prompt, kind=kind, visual_style=visual_style)
    except ImageRefused as exc:
        # Not a server fault -- tell the user what to change instead of a bare 500.
        raise HTTPException(422, str(exc)) from exc
    prefix = {"subject": "subj", "object": "item"}.get(kind, "place")
    fn = f"{prefix}_{uuid.uuid4().hex[:8]}{imagegen.sniff(raw)[0]}"
    (config.REFS / fn).write_bytes(raw)
    return {
        "ref": fn,
        "url": f"refs/{fn}",
        "bytes": len(raw),
        "elapsed": round(time.time() - started, 1),
        "model": config.IMAGE_MODEL,
    }


@router.post("/api/worlds/{wid}/subjects")
async def add_subject(
    wid: str,
    user: CurrentUser,
    name: str = Form(...),
    desc: str = Form(""),
    image: UploadFile | None = File(None),
    image_prompt: str = Form(""),
    ref: str = Form(""),
    control: str = Form("ai"),  # "user" = the one the player plays
    kind: str = Form(
        "人物"
    ),  # 人物 / 生物 / 物件 (character / creature / object); objects reuse the subject reference-image pipeline
    # Voice: a library entry (voice_id) or a file just uploaded via POST /api/voices/upload
    # (voice_ref). **With neither, one is picked from the name and description** (`voices.suggest`).
    # The video model generates audio with the video; without a reference voice it invents a new
    # voice every clip and the character never sounds like the same person. A wrong pick is at
    # least stable, visible in the inspector and replaceable in one click.
    voice_id: str = Form(""),
    voice_ref: str = Form(""),
) -> dict:
    try:
        with mutation_slot(wid):
            w = load_owned_world(wid, user)
            entity_kind = kind.strip() if kind.strip() in {"人物", "生物", "物件"} else "人物"
            is_item = entity_kind == "物件"
            fn = await _resolve_ref(
                ref=ref,
                image=image,
                image_prompt=image_prompt,
                kind="object" if is_item else "subject",
                prefix="item" if is_item else "subj",
                visual_style=w.visual_style,
            )
            sid = w.add_subject(
                name.strip(),
                fn,
                desc.strip() or image_prompt.strip(),
                control="ai" if is_item else ("user" if control == "user" else "ai"),
                kind=entity_kind,
                voice=None
                if is_item
                else (
                    _voice_form(voice_id, voice_ref)
                    or voices.suggest(name.strip(), desc.strip() or image_prompt.strip())
                ),
            )
            if is_item:
                # Uploaded or drawn items already have their final reference image; keep them apart
                # from the critic's asynchronous ``pending`` items so a background job cannot
                # overwrite them.
                w.subjects[sid]["ref_status"] = "ready"
                w.subjects[sid]["ref_source"] = "user" if image is not None else "generated"
            w.save()
            return {"id": sid, "ref": fn, **w.subjects[sid]}
    except GenerationBusy:
        raise HTTPException(409, "A clip is being generated; edit assets after it finishes") from None


@router.post("/api/worlds/{wid}/subjects/{sid}/control")
async def set_subject_control(
    wid: str,
    sid: str,
    req: Request,
    user: CurrentUser,
) -> dict:
    """Change who drives this character. The one the user plays is up to the user; the world plays the rest."""
    body = await req.json()
    control = "user" if (body.get("control") == "user") else "ai"
    try:
        with mutation_slot(wid):
            w = load_owned_world(wid, user)
            if sid not in w.subjects:
                raise HTTPException(404, f"No character {sid}")
            w.subjects[sid]["control"] = control
            w.save()
            return {"id": sid, **w.subjects[sid]}
    except GenerationBusy:
        raise HTTPException(409, "A clip is being generated; edit assets after it finishes") from None


@router.put("/api/worlds/{wid}/subjects/{sid}")
async def replace_subject(
    wid: str,
    sid: str,
    user: CurrentUser,
    desc: str = Form(""),
    image: UploadFile | None = File(None),
    image_prompt: str = Form(""),
    ref: str = Form(""),
) -> dict:
    """Swap one character's reference without changing its stable subject id."""
    try:
        with mutation_slot(wid):
            w = load_owned_world(wid, user)
            subject = w.subjects.get(sid)
            if subject is None:
                raise HTTPException(404, "subject not found")
            is_item = str(subject.get("kind") or "").strip() in {"物件", "物品", "道具"}
            fn = await _resolve_ref(
                ref=ref,
                image=image,
                image_prompt=image_prompt,
                kind="object" if is_item else "subject",
                prefix="item" if is_item else "subj",
                visual_style=w.visual_style,
            )
            subject["ref"] = fn
            # Never pair a new picture with wording that described the old one.
            subject["desc"] = desc.strip() or image_prompt.strip()
            # A new image needs a new visual definition on the next generated segment.
            subject.pop("definition", None)
            # Cancel any last-frame image job still running. Background jobs check these fields as a
            # version, so an old result cannot overwrite the image the user just picked.
            if is_item:
                subject["ref_status"] = "ready"
                subject["ref_source"] = "user" if image is not None else "generated"
                subject.pop("ref_prompt", None)
                subject.pop("source_ref", None)
                subject.pop("ref_error", None)
            w.save()
            return {"id": sid, **subject}
    except GenerationBusy:
        raise HTTPException(409, "A clip is being generated; edit assets after it finishes") from None


@router.delete("/api/worlds/{wid}/subjects/{sid}")
def drop_item(wid: str, sid: str, user: CurrentUser) -> dict:
    """Delete an item from the live asset catalogue; historical shots stay intact."""
    try:
        with mutation_slot(wid):
            w = load_owned_world(wid, user)
            try:
                w.drop_item(sid)
            except KeyError:
                raise HTTPException(404, "item not found") from None
            except ValueError as exc:
                raise HTTPException(400, str(exc)) from exc
            w.save()
            return {"ok": True}
    except GenerationBusy:
        raise HTTPException(409, "A clip is being generated; edit assets after it finishes") from None


@router.post("/api/worlds/{wid}/places")
async def add_place(
    wid: str,
    user: CurrentUser,
    name: str = Form(...),
    desc: str = Form(""),
    image: UploadFile | None = File(None),
    image_prompt: str = Form(""),
    ref: str = Form(""),
    make_current: bool = Form(True),
) -> dict:
    """Add or replace a place's reference image.

    Re-posting an existing name swaps the image; later segments pick it up, earlier
    ones keep the clip they already rendered.
    """
    try:
        with mutation_slot(wid):
            w = load_owned_world(wid, user)
            name = name.strip()
            if not name:
                raise HTTPException(400, "place name required")
            fn = await _resolve_ref(
                ref=ref,
                image=image,
                image_prompt=image_prompt,
                kind="place",
                prefix="place",
                visual_style=w.visual_style,
            )
            prev = w.places.get(name)
            description = desc.strip() or image_prompt.strip()
            if prev is None:
                w.add_place(name, fn, description)
            else:
                # Keep identity metadata, but never carry old visual wording to a new image.
                w.places[name] = {**prev, "ref": fn, "desc": description}
            w.places[name].pop("definition", None)
            if make_current or not w.here:
                w.here = name
            w.save()
            return {"name": name, **w.places[name]}
    except GenerationBusy:
        raise HTTPException(409, "A clip is being generated; edit assets after it finishes") from None


@router.put("/api/worlds/{wid}/places/{name}/main-scene")
async def update_place_main_scene(
    wid: str,
    name: str,
    req: Request,
    user: CurrentUser,
) -> dict:
    """Move a scene into a main-scene group used by the world map."""
    try:
        with mutation_slot(wid):
            w = load_owned_world(wid, user)
            if name not in w.places:
                raise HTTPException(404, "place not found")
            body = await req.json()
            if not isinstance(body, dict):
                raise HTTPException(400, "body must be an object")
            try:
                main_scene = w.set_place_main_scene(
                    name,
                    str(body.get("main_scene") or ""),
                )
            except ValueError as exc:
                raise HTTPException(400, str(exc)) from exc
            w.save()
            return {"name": name, "main_scene": main_scene}
    except GenerationBusy:
        raise HTTPException(409, "A clip is being generated; edit the map after it finishes") from None


@router.delete("/api/worlds/{wid}/places/{name}")
def drop_place(wid: str, name: str, user: CurrentUser) -> dict:
    try:
        with mutation_slot(wid):
            w = load_owned_world(wid, user)
            if name not in w.places:
                raise HTTPException(404, "place not found")
            if any(n.get("place") == name for n in w.nodes.values()):
                raise HTTPException(400, "This place already has clips and cannot be deleted")
            for child_name, place in w.places.items():
                if child_name != name and w.main_scene_for(child_name) == name:
                    place["main_scene"] = child_name
            w.places.pop(name)
            if w.here == name:
                w.here = next(iter(w.places), "")
            w.save()
            return {"ok": True}
    except GenerationBusy:
        raise HTTPException(409, "A clip is being generated; edit assets after it finishes") from None
