"""Voices: the library, previews, your own uploads, binding to a character.

Character voices and the narrator's voice are different things:
  · The narrator (the director's `reply`) goes through `/api/tts`; its voice is the TTS server's
    `voice.wav`, always the same, unrelated to who is in the world.
  · A character voice is a wav managed here, used for one thing: sent to the video model as
    `reference_audios` so the character speaks with that voice in the video.
"""

from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from murea_agent.memory import voices
from app.api.auth import CurrentUser, load_owned_world
from app.api.sse import GenerationBusy, mutation_slot

router = APIRouter(tags=["voices"])


@router.get("/api/voices")
def list_voices() -> dict:
    """The whole voice library. The frontend searches and filters it, and every entry can be previewed.

    Sent in one piece (~100 KB) instead of a search API: the frontend fetches it once and does
    filtering, paging and previews locally. The library is static, so it never goes stale. `tags`
    and `url` are not sent: the frontend builds search strings itself and uses `voiceUrl(ref)`.

    `gm` marks the curated tier the director picks from; the UI lists it first, but the user can
    pick any entry.
    """
    return {
        "voices": [
            {
                "id": e["id"],
                "name": e["name"],
                "desc": e["desc"],
                "gender": e["gender"],
                "age": e["age"],
                "accent": e["accent"],
                "group": e["group"],
                "gm": e["gm"],
                "seconds": e["seconds"],
                "ref": e["ref"],
            }
            for e in voices.library()
        ],
    }


@router.post("/api/voices/upload")
async def upload_voice(audio: UploadFile = File(...)) -> dict:
    """Receive an audio clip, convert it to wav in data/voices and return the file name.

    Upload first and bind later so the user can preview their clip before saving, the same flow
    as reference images: generate a candidate, look, then adopt.
    """
    try:
        # One byte past the limit is enough for store_upload to refuse it.
        raw = await audio.read(voices.MAX_UPLOAD_BYTES + 1)
        ref, name = voices.store_upload(raw, audio.filename or "")
    except voices.VoiceError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"ref": ref, "name": name, "url": f"voices/{ref}"}


@router.post("/api/worlds/{wid}/subjects/{sid}/voice")
async def set_subject_voice(
    wid: str,
    sid: str,
    req: Request,
    user: CurrentUser,
) -> dict:
    """Change this character's voice. `voice_id` is a library entry, `ref` a fresh upload,
    `clear` removes the voice.
    """
    body = await req.json()
    if not isinstance(body, dict):
        raise HTTPException(400, "body must be an object")
    try:
        with mutation_slot(wid):
            w = load_owned_world(wid, user)
            if sid not in w.subjects:
                raise HTTPException(404, f"No character {sid}")
            w.subjects[sid]["voice"] = _voice_from(body)
            w.save()
            return {"id": sid, **w.subjects[sid]}
    except GenerationBusy:
        raise HTTPException(409, "A clip is being generated; edit assets after it finishes") from None


def _voice_from(body: dict) -> dict:
    """Request body -> the voice record stored on the subject."""
    if body.get("clear"):
        return dict(voices.NONE)  # return a copy, never the module-level constant
    try:
        picked = voices.choose(
            str(body.get("voice_id") or ""), str(body.get("ref") or ""), str(body.get("name") or "")
        )
    except voices.VoiceError as exc:
        raise HTTPException(404, str(exc)) from exc
    if picked is None:
        raise HTTPException(400, "Give one of voice_id, ref or clear")
    return picked
