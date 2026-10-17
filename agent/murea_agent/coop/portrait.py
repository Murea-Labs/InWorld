"""Preview-first co-op portraits from an explicitly configured local image API."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import fcntl
import secrets
import threading
import time
from collections.abc import Sequence

from fastapi import HTTPException

from murea_agent import config
from app.api.sse import GenerationBusy, mutation_slot
from murea_agent.coop import room as coop, ai as coop_ai, trace as coop_trace
from murea_agent.providers import imagegen
from murea_agent import reqlog
from murea_agent.memory.world import World
from murea_agent.memory.world_media import has_media, media_path
from murea_agent.coop.messages import msg

_POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="coop-portrait")

# Drawings (portraits and shared references) run one at a time, under `_guard`. A request that
# found the renderer busy used to be refused with a "busy" message, so two players drawing at
# once, or anyone drawing while the default portraits were still rendering, had to retry by
# hand. Requests now wait in line. These are the ones this process has accepted and not finished:
# `recover` must not fail a request that is only waiting.
_ACCEPTED: set[str] = set()
_ACCEPTED_LOCK = threading.Lock()


def accept(request_id: str) -> None:
    with _ACCEPTED_LOCK:
        _ACCEPTED.add(request_id)


def accepted(request_id: str) -> bool:
    with _ACCEPTED_LOCK:
        return request_id in _ACCEPTED


def when_free(run, snapshot) -> None:
    """Run `run(snapshot, guard)` once the renderer is free; `run` closes the guard."""
    try:
        guard = _guard()
        try:
            fcntl.flock(guard, fcntl.LOCK_EX)
        except BaseException:
            guard.close()
            raise
        run(snapshot, guard)
    finally:
        with _ACCEPTED_LOCK:
            _ACCEPTED.discard(snapshot["request_id"])


# The Chinese descriptions are shown in the room UI.  Default image requests use
# a separate English visual description so imagegen's language guard can return
# immediately instead of making a translation-model round trip for every portrait.
_DEFAULT_ROLE_SPECS = {
    "space": (
        (
            "星野 · 星际游侠",
            "成年角色，轻型航天装甲，腰间配有多功能工具。",
            "Adult spacefaring ranger in lightweight space armor, carrying a multifunction tool at the waist.",
        ),
        (
            "岚音 · 星舰工程师",
            "成年角色，带工具腰带的航天工作服，携带便携维修设备。",
            "Adult spacecraft engineer in practical workwear with a tool belt, carrying a portable repair device.",
        ),
    ),
    "harbor": (
        (
            "沈舟 · 雾港侦探",
            "成年角色，长风衣，腰间挂着小皮包，擅长观察线索。",
            "Adult harbor detective in a long trench coat, carrying a small leather satchel and an observant demeanor.",
        ),
        (
            "洛宁 · 航海家",
            "成年角色，实用航海短外套，佩戴罗盘，熟悉港口路线。",
            "Adult navigator in a practical nautical jacket, wearing a compass and ready for harbor travel.",
        ),
    ),
    "fantasy": (
        (
            "洛伊 · 森林守卫",
            "成年角色，精致轻甲，背着长弓，行动敏捷。",
            "Adult forest guardian in refined light armor, carrying a longbow and an agile athletic build.",
        ),
        (
            "浅月 · 持剑游侠",
            "成年角色，轻便皮甲，腰间佩一把长剑，擅长近身护卫。",
            "Adult sword-bearing ranger in light leather armor, carrying a longsword at the hip and trained for close protection.",
        ),
    ),
}


# English display names and descriptions of the same roles, used when agent.language is en.
_DEFAULT_ROLE_NAMES_EN = {
    "space": (
        ("Hoshi · Star Ranger", "An adult in light space armour with a multitool at the waist."),
        (
            "Ran · Starship Engineer",
            "An adult in spacecraft workwear with a tool belt and a portable repair kit.",
        ),
    ),
    "harbor": (
        (
            "Shen · Harbour Detective",
            "An adult in a long trench coat with a small leather satchel, sharp-eyed for clues.",
        ),
        (
            "Lorne · Navigator",
            "An adult in a practical sailing jacket wearing a compass, at home on harbour routes.",
        ),
    ),
    "fantasy": (
        ("Rowan · Forest Guardian", "An adult in refined light armour with a longbow, quick on their feet."),
        (
            "Mira · Sword Ranger",
            "An adult in light leather armour with a longsword at the hip, a close-quarters protector.",
        ),
    ),
}

_SPACE_WORDS = (
    "星际",
    "空间站",
    "飞船",
    "太空",
    "机甲",
    "space",
    "starship",
    "spaceship",
    "station",
    "orbit",
    "mech",
)
_HARBOR_WORDS = (
    "港口",
    "码头",
    "调查",
    "侦探",
    "harbor",
    "harbour",
    "port",
    "dock",
    "detective",
    "investigat",
)


def _theme(setting: str) -> str:
    text = setting.lower()
    if any(word in text for word in _SPACE_WORDS):
        return "space"
    if any(word in text for word in _HARBOR_WORDS):
        return "harbor"
    return "fantasy"


def _default_specs(setting: str):
    """(display name, display description, English visual description) of the two default roles."""
    theme = _theme(setting)
    specs = _DEFAULT_ROLE_SPECS[theme]
    if config.LANGUAGE == "zh":
        return specs
    return tuple(
        (name, desc, media) for (name, desc), (_, _, media) in zip(_DEFAULT_ROLE_NAMES_EN[theme], specs)
    )


def _default_media_description(name: str, setting: str) -> str:
    """Return the English-only visual description for a built-in role."""
    for candidate_name, _, media_description in _default_specs(setting):
        if candidate_name == name:
            return media_description
    return ""


def default_members(room):
    """Fill empty seats with editable story-appropriate defaults before rendering."""
    specs = _default_specs(room.get("setting", ""))
    for index, member in enumerate(room.get("members", [])):
        if not member.get("character"):
            name, description, _ = specs[index % len(specs)]
            member.update(character=name, description=description)


def queue_defaults(code, uid):
    """Render and bind default portraits without blocking room creation."""
    # COOP_AUTO_GENERATE=0 is the explicit offline/test switch used by local
    # runs; keep the editable default metadata while skipping network work.
    if enabled() and coop.automatic():
        guard = _guard()
        try:
            _POOL.submit(_run_defaults, code, guard)
        except Exception:
            guard.close()


def _run_defaults(code, guard):
    try:
        try:
            fcntl.flock(guard, fcntl.LOCK_EX)
        except BlockingIOError:
            return
        with coop._edit(code) as room:
            targets = [
                (m["user_id"], m["character"], m["description"])
                for m in room["members"]
                if not m.get("ref") and m.get("character")
            ]
            style = room["visual_style"]
            setting = room.get("setting", "")
        for target_uid, name, description in targets:
            request_id = secrets.token_hex(16)
            with coop._edit(code) as room:
                member = next((m for m in room["members"] if m["user_id"] == target_uid), None)
                if (
                    not member
                    or member.get("ref")
                    or member.get("portrait", {}).get("status") == "generating"
                ):
                    continue
                member["portrait"] = {
                    "request_id": request_id,
                    "status": "generating",
                    "name": name,
                    "description": description,
                    "ref": "",
                    "error": "",
                    "source": "local",
                }
                # Only untouched default characters use a teammate's portrait as a design reference;
                # custom looks keep the normal flow.
                is_default = any(name == n and description == d for n, d, _ in _default_specs(setting))
                peer_refs = (
                    [m["ref"] for m in room["members"] if m["user_id"] != target_uid and m.get("ref")]
                    if is_default
                    else []
                )
            try:
                peer_images = []
                for peer_ref in peer_refs:
                    raw_peer = media_path("refs", peer_ref).read_bytes()
                    peer_images.append((imagegen.sniff(raw_peer)[1], raw_peer))
                raw = draw(
                    _prompt(
                        name,
                        description,
                        style,
                        media_description=_default_media_description(name, setting) if is_default else "",
                        distinguish_from_peers=bool(peer_images),
                    ),
                    images=peer_images,
                )
                ref = "coop-portrait-" + secrets.token_hex(12) + ".jpg"
                (config.REFS / ref).write_bytes(raw)
                with coop._edit(code) as room:
                    member = next((m for m in room["members"] if m["user_id"] == target_uid), None)
                    if not member or member.get("portrait", {}).get("request_id") != request_id:
                        (config.REFS / ref).unlink(missing_ok=True)
                        continue
                    member["portrait"].update(status="ready", ref=ref, elapsed=0)
                    # Automatic defaults belong to the receiving human player.
                    # Only local seats and AI teammates are host-controlled.
                    actor_uid = (
                        room["host_id"]
                        if room.get("mode") == "local" or member.get("kind") == "ai"
                        else target_uid
                    )
                # A player who uploaded during the ~6 s draw used to have their image replaced
                # by this default.
                use(code, actor_uid, request_id, target_uid=target_uid, expected_ref="")
            except Exception:
                with coop._edit(code) as room:
                    member = next((m for m in room["members"] if m["user_id"] == target_uid), None)
                    if member and member.get("portrait", {}).get("request_id") == request_id:
                        member["portrait"].update(
                            status="failed", error=msg("默认角色形象生成未完成，可重新生成或上传。")
                        )
    finally:
        guard.close()


def enabled():
    """Whether co-op can draw references with the configured image model."""
    return bool(
        (config.IMAGE_PROVIDER == "gemini" and config.GEMINI_API_KEY and not config.IMAGE_BASE)
        or (config.IMAGE_PROVIDER == "openai" and config.OPENAI_KEY)
        or (config.IMAGE_PROVIDER == "dashscope" and config.IMAGE_DASHSCOPE_KEY)
        or config.coop_env("COOP_IMAGE_SOCKET", "").strip()
        or config.coop_env("COOP_IMAGE_BASE", "").strip()
    )


def _guard():
    return (coop._directory() / "portrait-render.lock").open("a")


def target(room, uid, target_uid=None):
    if room.get("mode") == "local":
        return coop._controlled(room, uid, target_uid)
    coop._member(room, uid)
    member = coop._member(room, target_uid or uid)
    if member["user_id"] != uid:
        coop._host(room, uid)
        if member.get("kind") != "ai":
            raise HTTPException(403, msg("只能为自己或 AI 队友生成形象"))
    return member


def recover(room):
    pending = [
        m
        for m in room["members"]
        if m.get("portrait", {}).get("status") == "generating"
        and not accepted(m["portrait"].get("request_id", ""))
    ]
    if not pending:
        return
    with _guard() as guard:
        try:
            fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        for member in pending:
            member["portrait"].update(
                status="failed", error=msg("服务重启中断了生图，请重新生成；原形象已保留。")
            )


_RANDOM_ZH = {
    "space": [
        ("星际游侠", "轻型航天装甲，腰间配有多功能工具"),
        ("星舰工程师", "带工具腰带的航天工作服"),
        ("轨道侦察员", "贴身防护服与便携扫描仪"),
    ],
    "harbor": [
        ("雾港侦探", "长风衣，腰间挂着小皮包"),
        ("航海家", "实用航海短外套，佩戴罗盘"),
        ("寻宝猎人", "皮革护肩与旅行靴"),
    ],
    "fantasy": [
        ("持剑游侠", "轻便皮甲，腰间佩一把长剑"),
        ("森林守卫", "精致轻甲，背着长弓"),
        ("旅行法师", "有刺绣细节的法袍，手持木质法杖"),
    ],
}
_RANDOM_EN = {
    "space": [
        ("Star Ranger", "light space armour with a multitool at the waist"),
        ("Starship Engineer", "spacecraft workwear with a tool belt"),
        ("Orbital Scout", "a fitted protective suit and a handheld scanner"),
    ],
    "harbor": [
        ("Harbour Detective", "a long trench coat with a small leather satchel"),
        ("Navigator", "a practical sailing jacket and a compass"),
        ("Treasure Hunter", "leather pauldrons and travel boots"),
    ],
    "fantasy": [
        ("Sword Ranger", "light leather armour with a longsword at the hip"),
        ("Forest Guardian", "refined light armour with a longbow"),
        ("Wandering Mage", "an embroidered robe and a wooden staff"),
    ],
}
_LOOKS = {
    "zh": (
        ("成年男性", "成年女性"),
        ("黑色短发", "银白色短发", "棕色卷发", "深蓝色长发"),
        ("琥珀色眼睛", "深棕色眼睛", "灰蓝色眼睛"),
        ("墨绿色与金色配色", "深蓝色与银色配色", "酒红色与铜色配色", "米白色与深棕色配色"),
        ("林川", "晴岚", "洛伊", "南星", "伊诺", "浅月"),
    ),
    "en": (
        ("adult man", "adult woman"),
        ("short black hair", "short silver hair", "curly brown hair", "long dark blue hair"),
        ("amber eyes", "dark brown eyes", "grey-blue eyes"),
        (
            "forest green and gold colours",
            "navy and silver colours",
            "wine red and copper colours",
            "ivory and dark brown colours",
        ),
        ("Rowan", "Mira", "Ash", "Nell", "Ivo", "Sable"),
    ),
}


def _random_character(room, member):
    choose = secrets.choice
    zh = config.LANGUAGE == "zh"
    role, gear = choose((_RANDOM_ZH if zh else _RANDOM_EN)[_theme(room["setting"])])
    sexes, hair, eyes, colours, names = _LOOKS["zh" if zh else "en"]
    if member.get("kind") == "ai":
        name = member["character"]
        gear = coop_ai.PROFILES[member["ai_profile"]]["description"]
    else:
        taken = {m["character"].casefold() for m in room["members"] if m["user_id"] != member["user_id"]}
        name = choose([n + " · " + role for n in names if (n + " · " + role).casefold() not in taken])
    details = ", ".join((choose(sexes), choose(hair), choose(eyes), choose(colours), gear))
    return name, details


def _prompt(name, description, style, *, media_description="", distinguish_from_peers=False):
    if media_description:
        # Do not include the Chinese display name or description in the default
        # media request.  They remain editable UI metadata, while this prompt is
        # already in the English contract expected by the image model.
        prompt = (
            f"Character reference image. {media_description}\n"
            "Show one adult character in a complete full-body, slightly turned frontal pose, "
            "with a clear face, complete limbs, and crisp clothing and equipment details. "
            "Preserve the stated weapon or tool. Only this character appears in the image. "
            "Use a pure white background and even lighting. No scenery, no other people, "
            "no storyboard, no multi-view layout, no text, watermark, or logo."
        )
    elif config.LANGUAGE != "zh":
        prompt = (
            f"Character reference image: {name}. {description}.\n"
            "Show one adult character in a complete full-body, slightly turned frontal pose, "
            "with a clear face, complete limbs, and crisp clothing and equipment details. "
            "Preserve the stated weapon or tool. Only this character appears in the image. "
            "Use a pure white background and even lighting. No scenery, no other people, "
            "no storyboard, no multi-view layout, no text, watermark, or logo."
        )
    else:
        # User-entered names/descriptions intentionally stay on the existing path;
        # imagegen will translate and validate them when they contain Chinese.
        prompt = (
            f"角色设定参考图：{name}。{description}。\n"
            "一位成年角色，完整全身，正面略微侧身站姿，五官清楚、四肢完整、服装与装备细节清晰。"
            "保留描述中的随身武器或工具。画面中只有这一位角色，纯白背景、均匀光照。"
            "不要场景、不要其他人物、不要分镜或多视图、不要文字、水印或标志。"
        )
    if distinguish_from_peers:
        prompt += (
            "\nThe attached images show existing teammates for design comparison. "
            "Create the new individual described above, with a distinct facial structure, "
            "hairstyle, garment silhouette and intrinsic color palette within the same world art style. "
            "Keep the requested role and equipment. The output is a single portrait of the new individual."
        )
    return imagegen._with_visual_style(prompt, style)


def draw(prompt, kind="subject", images: Sequence[tuple[str, bytes]] = ()):
    """Render a co-op reference with the configured image model.

    The text-only self-hosted services (`image.service_urls`) are skipped:
    co-op references are drawn with the other players' images attached.
    """
    return imagegen.generate(prompt, kind=kind, images=images, force_gemini=config.IMAGE_PROVIDER == "gemini")


def start(code, uid, *, target_uid=None, name="", description="", randomize=False):
    # Authorize before reserving the shared renderer or doing any model work.
    with coop._edit(code) as room:
        member = target(room, uid, target_uid)
        coop._recover(room)
        coop._waiting(room)
        if not enabled():
            raise HTTPException(503, msg("当前实例尚未配置本地生图服务"))
        if randomize:
            name, description = _random_character(room, member)
        name = coop._text(name, "角色名称", 60)
        description = coop._text(description, "角色特点", 500)
        if room.get("monster", {}).get("name", "").casefold() == name.casefold():
            raise HTTPException(409, msg("角色名称不能与共享怪兽重复"))
        if any(
            m["user_id"] != member["user_id"] and m["character"].casefold() == name.casefold()
            for m in room["members"]
        ):
            raise HTTPException(409, msg("队友已使用这个角色名，请换一个"))
    with coop._edit(code) as room:
        member = target(room, uid, target_uid)
        coop._waiting(room)
        request_id = secrets.token_hex(16)
        member["portrait"] = {
            "request_id": request_id,
            "status": "generating",
            "name": name,
            "description": description,
            "ref": "",
            "error": "",
            "source": "local",
        }
        snapshot = {
            "code": code,
            "world_id": room["world_id"],
            "target_uid": member["user_id"],
            "round": room["round"] if room["rounds"] else 0,
            "request_id": request_id,
            "prompt": _prompt(name, description, room["visual_style"]),
        }
        accept(request_id)
    try:
        _POOL.submit(when_free, _run, snapshot)
    except Exception:
        with _ACCEPTED_LOCK:
            _ACCEPTED.discard(request_id)
        with coop._edit(code) as room:
            member = target(room, uid, target_uid)
            member["portrait"].update(status="failed", error=msg("生图服务繁忙，请重试"))
        raise
    return coop.get_room(code, uid)


def queue_ai(code, bot_id):
    """Draw a portrait for a new AI teammate in the background, then bind it.

    Its built-in look (`coop_ai.portrait`) is a flat illustration the video model does not take
    for a person: in one test the teammate was missing from every clip until a portrait was
    drawn by hand. Without an image service the built-in look stays.
    """
    if not (enabled() and coop.automatic()):
        return
    with coop._edit(code) as room:
        member = next((m for m in coop_ai.bots(room) if m["user_id"] == bot_id), None)
        if member is None or member.get("portrait", {}).get("status") == "generating":
            return
        request_id = secrets.token_hex(16)
        member["portrait"] = {
            "request_id": request_id,
            "status": "generating",
            "name": member["character"],
            "description": member["description"],
            "ref": "",
            "error": "",
            "source": "local",
        }
        snapshot = {
            "code": code,
            "world_id": room["world_id"],
            "target_uid": bot_id,
            "round": room["round"] if room["rounds"] else 0,
            "request_id": request_id,
            "prompt": _prompt(member["character"], member["description"], room["visual_style"]),
            "host_id": room["host_id"],
            "builtin_ref": member["ref"],
        }
        accept(request_id)
    _POOL.submit(when_free, _run_ai, snapshot)


# A drawing that finishes while a round is filming is bound once the round ends.
_AI_BIND_RETRY_SECONDS = 5.0
_AI_BIND_ATTEMPTS = 120


def _run_ai(snapshot, guard):
    _run(snapshot, guard)
    _bind_ai(snapshot, _AI_BIND_ATTEMPTS)


def _bind_ai(snapshot, attempts):
    """Bind the AI teammate's drawn portrait to its seat.

    The drawing waits behind the default portraits, so a quick party has often started a round
    by then. Giving up would leave the built-in look, which the video model does not film, so a
    busy room or world is retried on a timer (not on the single drawing thread). Any other
    refusal leaves the drawing as a preview the host can adopt, and is logged.
    """
    busy, error = False, "the room stayed busy"
    try:
        with coop._edit(snapshot["code"], persist=False) as room:
            host_id, busy = room["host_id"], room["status"] == "generating"
        if not busy:
            use(
                snapshot["code"],
                host_id,
                snapshot["request_id"],
                target_uid=snapshot["target_uid"],
                expected_ref=snapshot["builtin_ref"],
            )
            return
    except HTTPException as exc:
        busy = exc.status_code == 409 and exc.detail in (
            msg("这一轮正在生成，请等待完成"),
            msg("世界正在生成，请稍后使用这张形象"),
        )
        error = str(exc.detail)
    except Exception as exc:  # noqa: BLE001 -- the drawn portrait stays a preview the host can adopt
        error = f"{type(exc).__name__}: {exc}"
    if busy and attempts > 1:
        timer = threading.Timer(_AI_BIND_RETRY_SECONDS, _bind_ai, (snapshot, attempts - 1))
        timer.daemon = True
        timer.start()
        return
    token = reqlog.bind(snapshot["world_id"])
    try:
        reqlog.record("coop_ai_portrait", ok=False, request_id=snapshot["request_id"], error=error)
    finally:
        reqlog.unbind(token)


def _run(snapshot, guard):
    with coop_trace.operation(
        snapshot["world_id"], snapshot.get("round", 0), snapshot["request_id"], "generation"
    ):
        _run_logged(snapshot, guard)


def _run_logged(snapshot, guard):
    ref = ""
    retained = False
    started = time.monotonic()
    try:
        reqlog.record(
            "portrait_request",
            target_uid=snapshot["target_uid"],
            prompt=snapshot["prompt"],
            width=1024,
            height=576,
        )
        raw = draw(snapshot["prompt"])
        ref = "coop-portrait-" + secrets.token_hex(12) + ".jpg"
        (config.REFS / ref).write_bytes(raw)
        with coop._edit(snapshot["code"]) as room:
            member = next((m for m in room["members"] if m["user_id"] == snapshot["target_uid"]), None)
            if (
                member
                and room["status"] != "closed"
                and member.get("portrait", {}).get("request_id") == snapshot["request_id"]
            ):
                member["portrait"].update(
                    status="ready", ref=ref, elapsed=round(time.monotonic() - started, 1)
                )
                retained = True
        reqlog.record(
            "coop_portrait",
            ok=True,
            source="local",
            ref=ref,
            target_uid=snapshot["target_uid"],
            seconds=round(time.monotonic() - started, 1),
        )
    except Exception as exc:
        reqlog.record("coop_portrait", ok=False, error=str(exc)[:300])
        with coop._edit(snapshot["code"]) as room:
            member = next((m for m in room["members"] if m["user_id"] == snapshot["target_uid"]), None)
            if member and member.get("portrait", {}).get("request_id") == snapshot["request_id"]:
                member["portrait"].update(
                    status="failed", error=msg("本地生图未完成，请重试；原形象已保留。")
                )
    finally:
        if ref and not retained:
            (config.REFS / ref).unlink(missing_ok=True)
        guard.close()


def use(code, uid, request_id, target_uid=None, *, expected_ref=None):
    """Bind a ready portrait to the seat. `expected_ref` (automatic portraits): bind only if the
    seat still shows that image (none for a default, the built-in look for an AI teammate), so
    an image set while the portrait was drawing is never replaced. Checked under the same lock
    that binds it; the portrait stays a ready preview the player can still adopt."""
    with coop._edit(code) as room:
        member = target(room, uid, target_uid)
        coop._recover(room)
        coop._waiting(room)
        portrait = member.get("portrait", {})
        if portrait.get("request_id") != request_id or portrait.get("status") != "ready":
            raise HTTPException(409, msg("这张预览已更新或尚未生成完成，请重新查看"))
        if expected_ref is not None and member.get("ref", "") != expected_ref:
            return None
        ref, name, description = (portrait[k] for k in ("ref", "name", "description"))
        if room.get("monster", {}).get("name", "").casefold() == name.casefold():
            raise HTTPException(409, msg("角色名称不能与共享怪兽重复"))
        if not has_media("refs", ref):
            raise HTTPException(409, msg("预览图片已失效，请重新生成"))
        if any(
            m["user_id"] != member["user_id"] and m["character"].casefold() == name.casefold()
            for m in room["members"]
        ):
            raise HTTPException(409, msg("队友已使用这个角色名，请重新生成"))
        try:
            with mutation_slot(room["world_id"]):
                world = World.load(room["world_id"])
                sid = member["subject_id"]
                if sid in world.subjects:
                    world.subjects[sid].update(
                        name=name, ref=ref, desc=description, definition="", control="user"
                    )
                else:
                    sid = world.add_subject(name, ref, desc=description, control="user")
                world.save()
                member.update(character=name, description=description, subject_id=sid, ref=ref)
        except GenerationBusy:
            raise HTTPException(409, msg("世界正在生成，请稍后使用这张形象")) from None
        portrait["status"] = "applied"
        coop._mark_asset_edit(room)
        coop._unready(room)
    return coop.get_room(code, uid)
