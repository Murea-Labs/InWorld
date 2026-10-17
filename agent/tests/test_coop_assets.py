"""Shared creature/scene references, isolated storage and mocked model calls."""

import json
import os
import unittest
from copy import deepcopy
from dataclasses import asdict
from unittest.mock import patch

from murea_agent import config
from murea_agent.coop import (
    room as coop,
    assets,
    intro as coop_intro,
    portrait as coop_portrait,
    prefetch as coop_prefetch,
)
from murea_agent.memory.world import World
from murea_agent.memory.world import Node
from tests import test_coop


class SharedAssetsTest(unittest.TestCase):
    create = test_coop.CoopTest.create
    role = test_coop.CoopTest.role

    def setUp(self):
        test_coop.CoopTest.setUp(self)
        self.jobs = []
        p = patch.object(assets._POOL, "submit", side_effect=lambda *args: self.jobs.append(args))
        p.start()
        self.patches.append(p)
        self.room = self.create(capacity=4)
        self.base = self.url + "/assets"
        self.plan = {
            k: {"name": name, "description": desc}
            for k, name, desc in [
                ("scene", "山洞宝库", "巨龙守护的岩石山洞，金币堆积在洞穴两侧，中间留出通道。"),
                ("monster", "岩甲巨龙", "与山洞岩石同色的灰色鳞甲、金色眼睛、短角和强健四肢。"),
            ]
        }

    def tearDown(self):
        for job in self.jobs:
            # Extractions hold their lock; a requested drawing only waits for the renderer.
            getattr(job[-1], "close", lambda: None)()
        test_coop.CoopTest.tearDown(self)

    def start(self, kind="monster", client=None):
        return (client or self.a).post(self.base + f"/{kind}/generate", json=self.plan[kind])

    def complete(self):
        fn, *args = self.jobs.pop(0)
        with patch.object(coop_portrait, "draw", return_value=test_coop.picture()) as draw:
            fn(*args)
        return draw

    def upload(self, kind, client=None):
        return (client or self.a).post(
            self.base + f"/{kind}/upload",
            data=self.plan[kind],
            files={"file": ("ref.png", test_coop.picture(), "image/png")},
        )

    def test_recommendations_use_background_and_style_and_cache_without_generating(self):
        repaired_scene = {
            "description": "岩石山洞顶部延伸出天然拱顶，金币沿两侧台阶堆积，中间保留宽阔通道，远处可见通往山外的狭长出口。"
        }
        with patch.object(assets.coop_trace, "chat", side_effect=[self.plan, repaired_scene]) as chat:
            response = self.a.post(self.base + "/recommendations", json={})
            self.assertEqual(200, response.status_code, response.text)
            payload = json.loads(chat.call_args.kwargs["user"])
            self.assertEqual(self.room["setting"], payload["setting"])
            self.assertIn("visual_style", payload)
            again = self.a.post(self.base + "/recommendations", json={})
            self.assertEqual(response.json()["asset_recommendations"], again.json()["asset_recommendations"])
            self.assertEqual(2, chat.call_count)
            self.assertNotRegex(
                response.json()["asset_recommendations"]["scene"]["description"], assets._SCENE_FORBIDDEN
            )
        self.assertEqual([], self.jobs)
        self.assertEqual("", response.json()["scene_ref"])
        self.assertNotIn("monster", response.json())

    def test_recommendation_failure_keeps_previous_and_allows_manual_specs(self):
        with patch.object(assets.coop_trace, "chat", return_value=self.plan):
            first = self.a.post(self.base + "/recommendations", json={}).json()
        with patch.object(assets.coop_trace, "chat", side_effect=RuntimeError("model unavailable")):
            self.assertEqual(
                502, self.a.post(self.base + "/recommendations", json={"refresh": True}).status_code
            )
        self.assertEqual(first["asset_recommendations"], self.a.get(self.url).json()["asset_recommendations"])
        self.assertEqual(202, self.start().status_code)

    def test_room_creation_does_not_queue_or_bind_a_default_scene(self):
        self.assertEqual([], self.jobs)
        with patch.dict(os.environ, {"COOP_AUTO_GENERATE": "1"}):
            created = self.create()
        self.assertEqual([], self.jobs)
        self.assertEqual("", created["scene_ref"])
        self.assertEqual("", created["scene_name"])
        self.assertNotIn("scene", created.get("asset_previews", {}))

    def test_opening_extraction_only_queues_assets_missing_from_user_setup(self):
        manual_scene = self.upload("scene").json()
        frames = self.root / "frames"
        frames.mkdir()
        frame_name = "opening-manual-scene.jpg"
        (frames / frame_name).write_bytes(test_coop.picture())
        with coop._edit(self.code) as room:
            room["opening_node"] = {"id": "opening-manual-scene", "frames": [frame_name]}
        with patch.object(config, "FRAMES", frames):
            self.assertTrue(assets.queue_opening_extraction(self.code))
            self.assertEqual(1, len(self.jobs))
            fn, snapshot, guard = self.jobs.pop()
            self.assertEqual(["monster"], snapshot["missing"])
            self.assertEqual(manual_scene["scene_ref"], snapshot["baseline"]["scene"]["ref"])
            detected = {
                "scene": {
                    "present": True,
                    "name": "错误覆盖场景",
                    "description": "宽阔岩洞内有连续石阶与拱形出口，顶部裂隙投下冷色光线。",
                    "frame_index": 0,
                },
                "monster": {
                    "present": True,
                    "name": "岩甲巨龙",
                    "description": "庞大四足躯体覆盖灰黑色岩甲，金色双眼、弯曲短角与宽阔翼膜构成清晰轮廓。",
                    "frame_index": 0,
                },
            }
            with (
                patch.object(assets.coop_trace, "chat", return_value=detected),
                patch.object(assets.coop_portrait, "draw", return_value=test_coop.picture()) as draw,
            ):
                fn(snapshot, guard)
        room = self.a.get(self.url).json()
        self.assertEqual(manual_scene["scene_ref"], room["scene_ref"])
        self.assertEqual("岩甲巨龙", room["monster"]["name"])
        self.assertEqual("opening_video", room["monster"]["source"])
        self.assertEqual("opening-manual-scene", room["monster"]["from_node"])
        self.assertEqual(manual_scene.get("asset_edit_revision"), room.get("asset_edit_revision"))
        self.assertEqual(1, draw.call_count)
        self.assertEqual("subject", draw.call_args.kwargs["kind"])

    def test_failed_opening_asset_can_retry_without_replacing_manual_boss(self):
        manual_boss = self.upload("monster").json()
        frames = self.root / "frames"
        frames.mkdir()
        (frames / "retry.jpg").write_bytes(test_coop.picture())
        with coop._edit(self.code) as room:
            room["opening_node"] = {"id": "retry", "frames": ["retry.jpg"]}
        detected = {
            "scene": {
                "present": True,
                "name": "地下宝库",
                "description": "宽阔岩洞内连续石阶通向远处拱形出口，顶部裂隙投下冷色光线。",
                "frame_index": 0,
            },
            "monster": {"present": False, "name": "", "description": "", "frame_index": 0},
        }
        with (
            patch.object(config, "FRAMES", frames),
            patch.object(assets.coop_trace, "chat", return_value=detected),
            patch.object(coop_intro, "queue"),
        ):
            assets.queue_opening_extraction(self.code)
            fn, snapshot, guard = self.jobs.pop()
            with patch.object(coop_portrait, "draw", side_effect=ValueError("schema rejected")):
                fn(snapshot, guard)
            with coop._edit(self.code) as room:
                self.assertEqual("failed", room["asset_extraction"]["status"])
                self.assertIn("schema rejected", room["asset_extraction"]["error"])
                self.assertEqual(manual_boss["monster"]["ref"], room["monster"]["ref"])
            self.assertTrue(assets.queue_opening_extraction(self.code))
            fn, snapshot, guard = self.jobs.pop()
            self.assertEqual(["scene"], snapshot["missing"])
            with patch.object(coop_portrait, "draw", return_value=test_coop.picture()):
                fn(snapshot, guard)
            with coop._edit(self.code) as room:
                self.assertEqual("ready", room["asset_extraction"]["status"])
                self.assertEqual("applied", room["asset_extraction"]["scene"]["status"])
                self.assertEqual(manual_boss["monster"]["ref"], room["monster"]["ref"])

    def test_opening_extraction_keeps_user_boss_and_queues_only_scene(self):
        manual_boss = self.upload("monster").json()
        frames = self.root / "frames"
        frames.mkdir()
        frame_name = "opening-manual-boss.jpg"
        (frames / frame_name).write_bytes(test_coop.picture())
        with coop._edit(self.code) as room:
            room["opening_node"] = {"id": "opening-manual-boss", "frames": [frame_name]}
        with patch.object(config, "FRAMES", frames):
            self.assertTrue(assets.queue_opening_extraction(self.code))
            self.assertEqual(1, len(self.jobs))
            fn, snapshot, guard = self.jobs.pop()
            self.assertEqual(["scene"], snapshot["missing"])
            self.assertEqual(manual_boss["monster"]["ref"], snapshot["baseline"]["monster"]["ref"])
            detected = {
                "scene": {
                    "present": True,
                    "name": "地下宝库",
                    "description": "岩石拱顶覆盖宽阔地下空间，两侧石阶围绕中央平台，深处出口透入微弱冷光。",
                    "frame_index": 0,
                },
                "monster": {
                    "present": True,
                    "name": "错误覆盖怪兽",
                    "description": "庞大四足生物覆盖深色鳞甲，双角与翼膜构成清晰轮廓。",
                    "frame_index": 0,
                },
            }
            with (
                patch.object(assets.coop_trace, "chat", return_value=detected),
                patch.object(assets.coop_portrait, "draw", return_value=test_coop.picture()) as draw,
            ):
                fn(snapshot, guard)
        room = self.a.get(self.url).json()
        self.assertEqual(manual_boss["monster"]["ref"], room["monster"]["ref"])
        self.assertEqual("地下宝库", room["scene_name"])
        self.assertEqual("opening_video", room["scene_asset_source"])
        self.assertEqual("opening-manual-boss", room["scene_asset_from_node"])
        self.assertEqual(manual_boss.get("asset_edit_revision"), room.get("asset_edit_revision"))
        self.assertEqual(1, draw.call_count)
        self.assertEqual("place", draw.call_args.kwargs["kind"])

    def test_host_only_before_model_work_and_invalid_kind_rejected(self):
        self.b.post(self.url + "/join")
        with patch.object(assets.coop_trace, "chat") as chat:
            self.assertEqual(403, self.b.post(self.base + "/recommendations", json={}).status_code)
            chat.assert_not_called()
        self.assertEqual(403, self.start(client=self.b).status_code)
        self.assertEqual(403, self.upload("scene", self.b).status_code)
        self.assertEqual(403, self.b.delete(self.base + "/monster").status_code)
        self.assertEqual(401, self.start(client=self.anon).status_code)
        self.assertEqual(422, self.a.post(self.base + "/npc/generate", json=self.plan["monster"]).status_code)
        self.assertEqual([], self.jobs)

    def test_preview_is_separate_then_adoption_binds_ai_subject_and_unreadies_players(self):
        self.b.post(self.url + "/join")
        self.role(self.a, "阿青")
        self.role(self.b, "小白")
        with coop._edit(self.code) as room:
            for member in room["members"]:
                member.update(ready=True, action="打招呼")
        world_before = (config.SESSIONS / f"{self.room['world_id']}.json").read_bytes()
        changed_intro = coop_intro.signature(self.a.get(self.url).json())
        pending = self.start().json()["asset_previews"]["monster"]
        self.assertEqual("generating", pending["status"])
        self.assertEqual(
            409,
            self.a.post(self.base + "/monster/use", json={"request_id": pending["request_id"]}).status_code,
        )
        draw = self.complete()
        self.assertIn("岩甲巨龙", draw.call_args.args[0])
        self.assertNotIn("故事背景", draw.call_args.args[0])
        self.assertIn("不要把地点画进怪兽图", draw.call_args.args[0])
        self.assertEqual(world_before, (config.SESSIONS / f"{self.room['world_id']}.json").read_bytes())
        result = self.a.post(self.base + "/monster/use", json={"request_id": pending["request_id"]})
        self.assertEqual(200, result.status_code, result.text)
        room = result.json()
        monster = room["monster"]
        world = World.load(room["world_id"])
        self.assertEqual("ai", world.subjects[monster["subject_id"]]["control"])
        self.assertNotIn(monster["subject_id"], [m["subject_id"] for m in room["members"]])
        self.assertTrue(all(not m["ready"] and m["action"] == "打招呼" for m in room["members"]))
        self.assertNotEqual(changed_intro, coop_intro.signature(room))
        self.assertEqual(
            409,
            self.a.post(self.base + "/monster/use", json={"request_id": pending["request_id"]}).status_code,
        )
        self.assertEqual(3, room["capacity"])

    def test_scene_adoption_keeps_description_and_old_reference_until_used(self):
        first = self.upload("scene").json()
        self.plan["scene"]["description"] = "岩洞中增加一道通往山外的透光拱门"
        pending = self.start("scene").json()["asset_previews"]["scene"]
        self.complete()
        self.assertEqual(first["scene_ref"], self.a.get(self.url).json()["scene_ref"])
        self.assertEqual(first["asset_edit_revision"], self.a.get(self.url).json()["asset_edit_revision"])
        room = self.a.post(self.base + "/scene/use", json={"request_id": pending["request_id"]}).json()
        self.assertEqual(self.plan["scene"]["description"], room["scene_description"])
        world = World.load(room["world_id"])
        self.assertEqual(room["scene_ref"], world.places[room["scene_name"]]["ref"])
        self.assertNotEqual(first["scene_ref"], room["scene_ref"])
        self.assertEqual(first["asset_edit_revision"] + 1, room["asset_edit_revision"])
        self.assertEqual("manual", room["scene_asset_source"])
        removed = self.a.delete(self.base + "/scene").json()
        self.assertEqual(room["asset_edit_revision"] + 1, removed["asset_edit_revision"])
        self.assertTrue((config.REFS / first["scene_ref"]).is_file())

    def test_manual_description_edit_reuses_image_but_invalidates_continuation(self):
        before = self.upload("scene").json()
        after = coop.set_reference(self.code, self.ua["id"], "Updated cave", "New lighting.", scene=True)
        self.assertEqual(before["scene_ref"], after["scene_ref"])
        self.assertEqual(before["asset_edit_revision"] + 1, after["asset_edit_revision"])
        self.assertEqual("manual", after["scene_asset_source"])
        self.assertEqual("", after["scene_asset_from_node"])

    def test_shared_renderer_lock_failure_recovery_and_stale_result_cleanup(self):
        old = self.upload("monster").json()["monster"]
        self.start()
        # The place waits for the boss drawing instead of being refused (409 before).
        self.assertEqual(202, self.start("scene").status_code)
        self.assertEqual("generating", self.a.get(self.url).json()["asset_previews"]["scene"]["status"])
        fn, *args = self.jobs.pop(0)
        with patch.object(coop_portrait, "draw", side_effect=RuntimeError("offline")):
            fn(*args)
        room = self.a.get(self.url).json()
        self.assertEqual("failed", room["asset_previews"]["monster"]["status"])
        self.assertEqual(old, room["monster"])
        self.complete()
        self.assertEqual("ready", self.a.get(self.url).json()["asset_previews"]["scene"]["status"])
        self.start()
        self.jobs.pop(0)
        coop_portrait._ACCEPTED.clear()  # the process stopped before the drawing ran
        self.assertEqual("failed", self.a.get(self.url).json()["asset_previews"]["monster"]["status"])
        self.start()
        self.a.delete(self.base + "/monster")
        refs_before = set(config.REFS.iterdir())
        self.complete()
        self.assertEqual(refs_before, set(config.REFS.iterdir()))
        self.assertNotIn("monster", self.a.get(self.url).json())

    def test_limits_apply_to_upload_adoption_join_and_ai(self):
        self.b.post(self.url + "/join")
        self.c.post(self.url + "/join")
        self.assertEqual(200, self.upload("monster").status_code)
        self.assertEqual(409, self.a.post(self.url + "/ai", json={"profile": "axe"}).status_code)
        self.anon.post("/api/users", json={"name": "Fourth"}).json()
        self.assertEqual(409, self.anon.post(self.url + "/join").status_code)
        self.a.delete(self.base + "/monster")
        self.assertEqual(200, self.anon.post(self.url + "/join").status_code)
        self.assertEqual(409, self.start().status_code)
        self.assertEqual(409, self.upload("monster").status_code)

    def test_local_mode_assets_stay_host_owned_and_generation_freezes_changes(self):
        local = self.a.post(
            "/api/coop/rooms", json={"title": "本地", "setting": "山洞巨龙", "mode": "local"}
        ).json()
        url = "/api/coop/rooms/" + local["code"] + "/assets/scene/upload"
        self.assertEqual(
            403,
            self.b.post(
                url, data=self.plan["scene"], files={"file": ("ref.png", test_coop.picture(), "image/png")}
            ).status_code,
        )
        self.assertEqual(
            200,
            self.a.post(
                url, data=self.plan["scene"], files={"file": ("ref.png", test_coop.picture(), "image/png")}
            ).status_code,
        )
        # A held round lock prevents recovery from treating the fixture as a crash.
        import fcntl

        with (coop._directory() / f"{self.code}.run.lock").open("a") as guard:
            fcntl.flock(guard, fcntl.LOCK_EX)
            with coop._edit(self.code) as room:
                room["status"] = "generating"
            self.assertEqual(409, self.start().status_code)
            self.assertEqual(409, self.upload("scene").status_code)
            self.assertEqual(409, self.a.delete(self.base + "/scene").status_code)

    def test_shared_image_generation_requires_coop_local_service(self):
        with (
            patch.dict(os.environ, {"COOP_IMAGE_BASE": "", "COOP_IMAGE_SOCKET": ""}),
            patch.object(config, "GEMINI_API_KEY", "test"),
            patch.object(config, "IMAGE_BASE", "http://image"),
        ):
            self.assertEqual(503, self.start().status_code)
            self.assertEqual([], self.jobs)

    def test_shared_images_use_the_same_coop_drawer_as_portraits(self):
        self.start("scene").json()["asset_previews"]["scene"]
        fn, *args = self.jobs.pop(0)
        with (
            patch.object(coop_portrait, "draw", return_value=test_coop.picture()) as draw,
            patch.object(assets.imagegen, "generate") as gemini,
        ):
            fn(*args)
        self.assertIn("没有人物、怪兽", draw.call_args.args[0])
        self.assertNotIn("故事背景", draw.call_args.args[0])
        gemini.assert_not_called()
        self.assertEqual("ready", self.a.get(self.url).json()["asset_previews"]["scene"]["status"])

    def test_round_video_discovers_new_boss_and_binds_fixed_asset(self):
        self.b.post(self.url + "/join")
        self.role(self.a, "阿青")
        self.role(self.b, "小白")
        world = World.load(self.room["world_id"])
        frame_name = "round-boss-frame.jpg"
        (config.FRAMES / frame_name).write_bytes(test_coop.picture())
        node = Node(
            id="n-round-boss",
            parent=world.head,
            video_id="v-round-boss",
            clip="round-boss.mp4",
            place="宝库",
            beat="巨大的身影出现",
            mode="cut",
            frames=[frame_name],
            subjects=[m["subject_id"] for m in self.room["members"]],
        )
        world.nodes[node.id] = asdict(node)
        world.head = node.id
        world.save()
        with coop._edit(self.code) as room:
            room["rounds"] = [
                {"round": 1, "merged_prompt": "", "inputs": room["members"], "node": asdict(node)}
            ]
            room["round"] = 2
            room["status"] = "waiting"
        self.assertTrue(assets.queue_round_boss_extraction(self.code))
        fn, snapshot, guard = self.jobs.pop(0)
        detected = {
            "present": True,
            "new": True,
            "name": "岩甲巨龙",
            "description": "体型庞大，灰色岩甲覆盖四肢与躯干，金色眼睛、短角和宽阔翼膜构成稳定轮廓。",
            "frame_index": 0,
        }
        with (
            patch.object(assets.coop_trace, "chat", return_value=detected),
            patch.object(assets.coop_portrait, "draw", return_value=test_coop.picture()) as draw,
        ):
            fn(snapshot, guard)
        room = self.a.get(self.url).json()
        self.assertEqual("岩甲巨龙", room["monster"]["name"])
        self.assertEqual("round_video", room["monster"]["source"])
        self.assertEqual("ready", room["boss_detection"]["status"])
        self.assertEqual("applied", room["boss_detection"]["result"])
        monster = World.load(room["world_id"]).subjects[room["monster"]["subject_id"]]
        self.assertEqual("生物", monster["kind"])
        self.assertEqual("frame", monster["ref_source"])
        self.assertEqual(node.id, monster["from_node"])
        self.assertTrue((config.REFS / monster["ref"]).is_file())
        self.assertTrue((config.REFS / monster["source_ref"]).is_file())
        self.assertIn("纯白背景", draw.call_args.args[0])

    def test_round_video_does_not_create_boss_when_model_says_not_visible(self):
        world = World.load(self.room["world_id"])
        frame_name = "round-empty-frame.jpg"
        (config.FRAMES / frame_name).write_bytes(test_coop.picture())
        node = Node(
            id="n-round-empty",
            parent=world.head,
            video_id="v-round-empty",
            clip="round-empty.mp4",
            place="宝库",
            beat="角色走动",
            mode="cut",
            frames=[frame_name],
        )
        world.nodes[node.id] = asdict(node)
        world.head = node.id
        world.save()
        with coop._edit(self.code) as room:
            room["rounds"] = [
                {"round": 1, "merged_prompt": "", "inputs": room["members"], "node": asdict(node)}
            ]
            room["round"] = 2
        self.assertTrue(assets.queue_round_boss_extraction(self.code))
        fn, snapshot, guard = self.jobs.pop(0)
        with (
            patch.object(
                assets.coop_trace,
                "chat",
                return_value={
                    "present": False,
                    "new": False,
                    "name": "",
                    "description": "",
                    "frame_index": 0,
                },
            ),
            patch.object(assets.coop_portrait, "draw"),
        ):
            fn(snapshot, guard)
        room = self.a.get(self.url).json()
        self.assertNotIn("monster", room)

    def test_boss_detection_does_not_write_story_observations(self):
        self.b.post(self.url + "/join")
        self.role(self.a, "阿青")
        self.role(self.b, "小白")
        world = World.load(self.room["world_id"])
        frame_name = "round-analysis-frame.jpg"
        (config.FRAMES / frame_name).write_bytes(test_coop.picture())
        node = Node(
            id="n-round-analysis",
            parent=world.head,
            video_id="v-analysis",
            clip="analysis.mp4",
            place="宝库",
            beat="两人站稳",
            mode="cut",
            frames=[frame_name],
            subjects=[m["subject_id"] for m in self.room["members"]],
        )
        world.nodes[node.id] = asdict(node)
        world.head = node.id
        world.save()
        with coop._edit(self.code) as room:
            room["rounds"] = [
                {"round": 1, "merged_prompt": "", "inputs": room["members"], "node": asdict(node)}
            ]
            room["round"] = 2
            room["status"] = "waiting"
            members = deepcopy(room["members"])
        audit = {
            "present": False,
            "new": False,
            "name": "",
            "description": "",
            "frame_index": 0,
            "observation": {
                "event_outcome": "unknown",
                "summary": "两人站在宝库中央，画面里尚未出现怪兽。",
                "environment_state": "宝库中央通道保持畅通，深处拱门可见。",
                "boss_state": "",
                "player_states": [{"user_id": m["user_id"], "state": "站在中央通道一侧。"} for m in members],
            },
            "story_phase": "threat",
        }
        with (
            patch.dict(os.environ, {"COOP_INTRO_ENABLED": "0"}),
            patch.object(assets.coop_trace, "chat", return_value=audit) as chat,
            patch.object(assets.coop_portrait, "draw") as draw,
            patch.object(coop_prefetch, "queue_recommendation") as queue,
        ):
            self.assertTrue(assets.queue_round_boss_extraction(self.code))
            fn, snapshot, guard = self.jobs.pop(0)
            fn(snapshot, guard)
        room = self.a.get(self.url).json()
        self.assertEqual(1, chat.call_count)
        self.assertEqual("coop_round_boss_detection", chat.call_args.kwargs["tag"])
        self.assertEqual("ready", room["boss_detection"]["status"])
        self.assertNotIn("observation", chat.call_args.kwargs["schema"]["properties"])
        self.assertEqual("opening", room["story_state"]["phase"])
        self.assertEqual("", room["story_state"]["last_node_id"])
        self.assertNotIn("observation", room["rounds"][-1])
        self.assertNotIn("intro", room)
        queue.assert_not_called()
        self.assertEqual("none", room["boss_detection"]["result"])
        draw.assert_not_called()

    def test_player_boss_edit_during_visual_analysis_requeues_current_guidance(self):
        self.b.post(self.url + "/join")
        self.role(self.a, "阿青")
        self.role(self.b, "小白")
        world = World.load(self.room["world_id"])
        frame_name = "round-analysis-race.jpg"
        (config.FRAMES / frame_name).write_bytes(test_coop.picture())
        node = Node(
            id="n-round-race",
            parent=world.head,
            video_id="v-race",
            clip="race.mp4",
            place="宝库",
            beat="陌生巨兽出现",
            mode="cut",
            frames=[frame_name],
        )
        world.nodes[node.id] = asdict(node)
        world.head = node.id
        world.save()
        with coop._edit(self.code) as room:
            room["rounds"] = [
                {"round": 1, "merged_prompt": "", "inputs": room["members"], "node": asdict(node)}
            ]
            room["round"] = 2
        detected = {
            "present": True,
            "new": True,
            "name": "画面巨兽",
            "description": "灰黑色鳞甲覆盖高大躯体，金色眼睛，头顶双角，背部有宽阔骨翼。",
            "frame_index": 0,
        }

        def player_replaces_boss(*args, **kwargs):
            with coop._edit(self.code) as room:
                room["monster"] = {
                    "name": "玩家选择的首领",
                    "description": "玩家设置的外观",
                    "ref": "manual-boss.jpg",
                    "subject_id": "manual-boss",
                }
                assets._invalidate_round_boss(room)
            return test_coop.picture()

        with (
            patch.dict(os.environ, {"COOP_INTRO_ENABLED": "1"}),
            patch.object(assets.coop_trace, "chat", return_value=detected),
            patch.object(assets.coop_portrait, "draw", side_effect=player_replaces_boss),
            patch.object(coop_prefetch, "queue_recommendation") as queue,
        ):
            self.assertTrue(assets.queue_round_boss_extraction(self.code))
            fn, snapshot, guard = self.jobs.pop(0)
            old_request = snapshot["request_id"]
            fn(snapshot, guard)
        room = self.a.get(self.url).json()
        self.assertEqual("superseded", room["boss_detection"]["status"])
        self.assertEqual("玩家选择的首领", room["monster"]["name"])
        self.assertEqual("generating", room["intro"]["status"])
        self.assertNotEqual(old_request, room["intro"]["request_id"])
        self.assertEqual(1, len(self.intro_pending))
        queue.assert_not_called()


if __name__ == "__main__":
    unittest.main()
