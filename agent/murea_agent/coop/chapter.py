"""Chapter hand-off: reuses the chapter task's lock and retries; candidate assets are adopted together once ready."""

import copy
import hashlib
from murea_agent.coop import boss as coop_boss, prompts_en
from murea_agent.util import _say

from app.api.sse import mutation_slot
from murea_agent.memory.world_media import has_media, media_path


SYSTEM = """
本次同时规划关卡交接 handoff。scene 和 monster 各选择 keep、reuse、new、none；
场景必须存在，不能选 none。reuse 使用 asset_catalog 中精确 id；new 填写 name 和 description。
keep 表示当前参考仍适合下一关首镜头，none 表示原生物已离场；新生物建立新身份。
boss_life.status=dead 的旧 BOSS 不再作为下一关生物，选择 none 或不同身份的新生物。不得复用已死亡身份。
仅为下一关首镜头在场的生物选择资产，未来敌人留在大纲中，出现时再建档。
entry 描述下一关首镜头的自然起始局势和与已发生动作的承接，不写服务端指令。
上一段已经到达的地方就是起点，完成的出门、脱险等动作不再作为下一关任务。
新场景的描述提供布局、可通行区域、地标和光线，人物与怪兽由独立参考承担。
若仍需穿过通道或移动到远处，首镜头地点选择当前已到达的位置，分阶段推进；
不要用远方目的地图替代尚未完成的移动过程。新一关阶段应服务于新的目标。
场景布局与本关挑战配套：明确目标涉及的地标、路线或交互对象如何出现在场景中，
让玩家看到要处理什么及行动的方向；保持环境描述和 overall_goal 一致。
reconcile_chapter 是旧存档资产交接：保留当前章节目标和已发生结果，仅校准当前地点与在场生物。
"""
SYSTEM = _say(SYSTEM, prompts_en.CHAPTER_HANDOFF)

_ASSET = {
    "type": "object",
    "properties": {
        "mode": {"type": "string", "enum": ["keep", "reuse", "new", "none"]},
        "id": {"type": "string"},
        "name": {"type": "string"},
        "description": {"type": "string"},
    },
    "required": ["mode", "id", "name", "description"],
}
SCHEMA = {
    "type": "object",
    "properties": {"scene": _ASSET, "monster": _ASSET, "entry": {"type": "string"}},
    "required": ["scene", "monster", "entry"],
}


def needed(room):
    from murea_agent.coop import story as coop_story

    outline = room.get("story_outline") or {}
    if not coop_story.enabled() or not outline.get("stages"):
        return False
    return bool(
        coop_story.context(room).get("chapter_complete")
        or (outline.get("chapter", 1) > 1 and not outline.get("assets_ready"))
    )


def blocked(room):
    return needed(room) or (room.get("chapter_transition") or {}).get("status") in (
        "planning",
        "assets",
        "failed",
    )


def entry(room):
    transition = room.get("chapter_transition") or {}
    head = ((room.get("rounds") or [{}])[-1].get("node") or room.get("opening_node") or {}).get("id")
    return transition.get("entry", "") if transition.get("from_node") == head else ""


def retry(code):
    from murea_agent.coop import room as coop, story as coop_story

    with coop._edit(code) as room:
        if (room.get("chapter_transition") or {}).get("status") != "failed":
            return False
        job = room.get("story_outline_job") or {}
        job.update(attempts=0, retry_at=0)
    return coop_story.queue(code)


def catalog(room):
    from murea_agent.memory.world import World

    world = World.load(room["world_id"])
    players = {m.get("subject_id") for m in room.get("members", [])}
    return {
        "scene": [
            {"id": key, "name": key, "description": p.get("desc", ""), "ref": p.get("ref", "")}
            for key, p in world.places.items()
            if p.get("ref")
        ],
        "monster": [
            {"id": key, "name": p.get("name", ""), "description": p.get("desc", ""), "ref": p.get("ref", "")}
            for key, p in world.subjects.items()
            if key not in players
            and p.get("kind") == "生物"
            and p.get("ref")
            and ((room.get("story_state") or {}).get("boss_lifecycle") or {}).get(key, {}).get("status")
            != "dead"
            and not (
                key == coop_boss.identity(room.get("monster") or {}) and coop_boss.status(room) == "dead"
            )
        ],
    }


def validate(raw, room, assets):
    if not isinstance(raw, dict) or not isinstance(raw.get("entry"), str) or not raw["entry"].strip():
        raise ValueError("chapter handoff has no starting situation")
    result = {"entry": raw["entry"].strip()[:1200]}
    for kind in ("scene", "monster"):
        item = raw.get(kind) or {}
        mode = item.get("mode")
        if kind == "monster" and mode == "keep" and coop_boss.status(room) == "dead":
            mode = "none"
        if mode not in ("keep", "reuse", "new", "none") or (kind == "scene" and mode == "none"):
            raise ValueError("invalid chapter asset choice")
        if mode == "keep":
            current = (
                {
                    "name": room.get("scene_name"),
                    "description": room.get("scene_description"),
                    "ref": room.get("scene_ref"),
                }
                if kind == "scene"
                else room.get("monster") or {}
            )
            if not current.get("ref"):
                raise ValueError("no chapter asset to reuse")
            item = {**current, "mode": mode}
        elif mode == "reuse":
            found = next((a for a in assets[kind] if a["id"] == item.get("id")), None)
            if not found:
                raise ValueError("the chapter references an asset that does not exist")
            if (
                kind == "monster"
                and ((room.get("story_state") or {}).get("boss_lifecycle") or {})
                .get(item["id"], {})
                .get("status")
                == "dead"
            ):
                raise ValueError("a dead boss cannot return as the next chapter's creature")
            item = {**found, "mode": mode}
        elif mode == "new":
            if any(
                not isinstance(item.get(k), str) or not item[k].strip() or len(item[k]) > 1500
                for k in ("name", "description")
            ):
                raise ValueError("the new chapter asset has no valid description")
            if kind == "scene" and any(a["name"] == item["name"] for a in assets[kind]):
                raise ValueError(
                    "a place with this name exists; use reuse or give the new place its own name"
                )
            item = {"mode": mode, "name": item["name"].strip(), "description": item["description"].strip()}
        else:
            item = {"mode": "none"}
        result[kind] = copy.deepcopy(item)
    if len(room.get("members", [])) + 1 + int(result["monster"]["mode"] != "none") > 5:
        raise ValueError("not enough reference slots for every player, the place and the creature")
    return result


def prepare(snapshot, request_id, plan):
    from murea_agent.coop import room as coop, assets as coop_assets, portrait as coop_portrait
    from murea_agent.providers import imagegen

    handoff = copy.deepcopy(plan)
    for kind in ("scene", "monster"):
        item = handoff[kind]
        if item["mode"] != "new":
            continue
        # Stable candidate file names let retries after a crash or partial failure reuse finished
        # images.
        key = hashlib.sha256(
            (
                snapshot["world_id"] + str(snapshot["story_outline"].get("version")) + kind + repr(item)
            ).encode()
        ).hexdigest()[:24]
        ref = f"coop-chapter-{key}.jpg"
        path = media_path("refs", ref)
        if not path.exists():
            prompt = coop_assets.prompt(snapshot, kind, item)
            raw = coop_portrait.draw(
                imagegen._with_visual_style(prompt, snapshot.get("visual_style", "")),
                kind="place" if kind == "scene" else "subject",
            )
            raw = coop.image_bytes(raw)
            temporary = path.with_suffix(".tmp")
            temporary.write_bytes(raw)
            temporary.replace(path)
        item["ref"] = ref
        with coop._edit(snapshot["code"]) as room:
            if (room.get("story_outline_job") or {}).get("request_id") != request_id:
                raise ValueError("the chapter preparation job was replaced")
            room["chapter_transition"].update(status="assets", prepared=kind)
    return handoff


def apply(room, plan):
    """The caller holds the room lock; the world is saved first, and retries find the same asset by ref instead of creating a new one."""
    from murea_agent.coop import room as coop, prefetch as coop_prefetch
    from murea_agent.memory.world import World

    with mutation_slot(room["world_id"]):
        world = World.load(room["world_id"])
        scene, monster = plan["scene"], plan["monster"]
        if monster["mode"] in ("keep", "reuse"):
            candidate = dict(monster, subject_id=monster.get("subject_id") or monster.get("id"))
            if coop_boss.status(room, candidate) == "dead":
                monster = {"mode": "none"}
        if not has_media("refs", scene["ref"]):
            raise ValueError("the next chapter's place reference is missing")
        if monster["mode"] != "none" and not has_media("refs", monster["ref"]):
            raise ValueError("the next chapter's creature reference is missing")
        coop_boss.remember(room)
        room.setdefault("story_state", {}).setdefault(
            "boss_identity", coop_boss.identity(room.get("monster") or {})
        )
        if scene["name"] not in world.places:
            world.add_place(scene["name"], scene["ref"], desc=scene.get("description", ""))
        world.here = scene["name"]
        active = {}
        if monster["mode"] != "none":
            sid = monster.get("subject_id") or monster.get("id")
            if monster["mode"] == "new":
                sid = next((k for k, s in world.subjects.items() if s.get("ref") == monster["ref"]), None)
                if not sid:
                    sid = world.add_subject(
                        monster["name"],
                        monster["ref"],
                        desc=monster["description"],
                        control="ai",
                        kind="生物",
                    )
            active = {k: v for k, v in monster.items() if k not in ("mode", "id")}
            active.update(subject_id=sid, source="chapter")
        world.save()
        changed = room.get("scene_ref") != scene["ref"] or (room.get("monster") or {}).get(
            "ref"
        ) != active.get("ref")
        death_only = room.get("scene_ref") == scene["ref"] and not active and coop_boss.status(room) == "dead"
        room.update(
            scene_ref=scene["ref"],
            scene_name=scene["name"],
            scene_description=scene.get("description", ""),
            monster=active,
        )
        coop_boss.bind(room)
        if changed and not death_only:
            room.update(scene_asset_source="chapter", scene_asset_from_node="")
            coop._mark_asset_edit(room)
        room["state_revision"] = int(room.get("state_revision", 0)) + 1
        coop_prefetch.invalidate(room, "下一关资产已就绪", clear_actions=False)
        room.pop("pending_guidance", None)
        room.pop("intro", None)
        head = ((room.get("rounds") or [{}])[-1].get("node") or room.get("opening_node") or {}).get("id")
        room["chapter_transition"] = {
            "status": "ready",
            "entry": plan["entry"],
            "scene": scene["name"],
            "from_node": head,
        }
        # A new chapter keeps only environment facts; the old enemy's visible state is not carried
        # into new shots.
        state = room.setdefault("story_state", {})
        if monster["mode"] != "keep":
            state.update(boss_visible=False, boss_known=bool(active), boss="")
        state["environment"] = scene.get("description", "")
