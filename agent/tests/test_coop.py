"""Co-op contracts: distinct users, real persistence, no model/network calls."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import io
import json
import os
import asyncio
import fcntl
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
    os.environ.pop(key, None)

from fastapi.testclient import TestClient
from PIL import Image
from murea_agent import config
from murea_agent.coop import guidance_checks
from app.api import create_app
from murea_agent.providers import llm
from murea_agent.coop import (
    room as coop,
    ai as coop_ai,
    portrait as coop_portrait,
    intro as coop_intro,
    observation as coop_observation,
    events as coop_events,
    prefetch as coop_prefetch,
)
from murea_agent import reqlog
from murea_agent.memory.world import Node, World


def picture():
    out = io.BytesIO()
    Image.new("RGB", (16, 16), "teal").save(out, format="PNG")
    return out.getvalue()


class CoopTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.patches = []
        for key, path in [
            ("DATA", self.root),
            ("REFS", self.root / "refs"),
            ("SESSIONS", self.root / "sessions"),
            ("CLIPS", self.root / "clips"),
        ]:
            path.mkdir(exist_ok=True)
            p = patch.object(config, key, path)
            p.start()
            self.patches.append(p)
        p = patch.dict(
            os.environ,
            {
                "COOP_GENERATION_ENABLED": "1",
                "COOP_AI_MODE": "offline",
                "COOP_AUTO_GENERATE": "0",
                "COOP_INTRO_ENABLED": "0",
                "COOP_OPENING_VIDEO": "0",
                "COOP_ACTION_PREFETCH": "0",
                "COOP_RECOMMENDATION_PREFETCH": "0",
                "COOP_SPECULATIVE_VIDEO": "0",
                "COOP_IMAGE_BASE": "http://127.0.0.1:1",
                "COOP_IMAGE_SOCKET": "",
            },
        )
        p.start()
        self.patches.append(p)
        app = create_app()
        self.a, self.b, self.c, self.anon = [TestClient(app) for _ in range(4)]
        self.ua = self.a.post("/api/users", json={"name": "Alice"}).json()
        self.ub = self.b.post("/api/users", json={"name": "Bob"}).json()
        self.uc = self.c.post("/api/users", json={"name": "Carol"}).json()
        self.pending = []
        p = patch.object(coop._POOL, "submit", side_effect=lambda *args: self.pending.append(args))
        p.start()
        self.patches.append(p)

        self.observation_pending = []
        p = patch.object(
            coop_observation._POOL, "submit", side_effect=lambda *args: self.observation_pending.append(args)
        )
        p.start()
        self.patches.append(p)

        self.intro_pending = []
        p = patch.object(
            coop_intro._POOL, "submit", side_effect=lambda *args: self.intro_pending.append(args)
        )
        p.start()
        self.patches.append(p)
        self.ai_pending = []
        p = patch.object(coop_ai._POOL, "submit", side_effect=lambda *args: self.ai_pending.append(args))
        p.start()
        self.patches.append(p)
        self.portrait_pending = []
        p = patch.object(
            coop_portrait._POOL, "submit", side_effect=lambda *args: self.portrait_pending.append(args)
        )
        p.start()
        self.patches.append(p)

    def tearDown(self):
        for _, _, guard in self.observation_pending:
            guard.close()
        for _, _, guard in self.intro_pending:
            guard.close()
        for _, _, _, guard in self.pending:
            guard.close()
        for _, _, guard in self.ai_pending:
            guard.close()
        for job in self.portrait_pending:
            # Default portraits hold the renderer; a player's drawing only waits for it.
            getattr(job[-1], "close", lambda: None)()
        coop_portrait._ACCEPTED.clear()
        for p in reversed(self.patches):
            p.stop()
        for c in (self.a, self.b, self.c, self.anon):
            c.close()
        self.tmp.cleanup()

    def create(self, capacity=4):
        r = self.a.post(
            "/api/coop/rooms", json={"title": "巨龙的宝库", "setting": "共同对抗巨龙", "capacity": capacity}
        )
        self.assertEqual(201, r.status_code, r.text)
        self.code = r.json()["code"]
        self.url = "/api/coop/rooms/" + self.code
        return r.json()

    def role(self, client, name):
        r = client.post(
            self.url + "/character", data={"name": name}, files={"file": ("ref.png", picture(), "image/png")}
        )
        self.assertEqual(200, r.status_code, r.text)
        return r.json()

    def ready(self):
        self.create()
        self.assertEqual(200, self.b.post(self.url + "/join").status_code)
        self.role(self.a, "角色A")
        self.role(self.b, "角色B")
        for client, action in [(self.a, "用剑砍怪兽"), (self.b, "用斧子劈他")]:
            r = client.put(self.url + "/action", json={"action": action, "round": 1})
            self.assertEqual(200, r.status_code, r.text)
        return self.a.get(self.url).json()

    def launch(self, room=None, client=None):
        room = room or self.a.get(self.url).json()
        return (client or self.a).post(
            self.url + "/generate", json={"round": room["round"], "revision": room["revision"]}
        )

    @staticmethod
    def video(world, text, **kwargs):
        node = Node(
            id="n-coop",
            parent=world.head,
            video_id="video-one",
            clip="one.mp4",
            place=world.here,
            beat="双方协作",
            mode="cut",
            user_text=text,
            reply="游侠挥剑牵制巨龙，战士挥斧配合。",
            subjects=kwargs["mentions"],
        )
        world.nodes[node.id] = asdict(node)
        world.head = node.id
        world.save()
        return node

    def local_room(self):
        r = self.a.post(
            "/api/coop/rooms", json={"title": "本地双人", "setting": "协力对抗怪兽", "mode": "local"}
        )
        self.assertEqual(201, r.status_code, r.text)
        self.code = r.json()["code"]
        self.url = "/api/coop/rooms/" + self.code
        return r.json()

    def local_role(self, uid, name):
        r = self.a.post(
            self.url + "/character",
            data={"name": name, "target_uid": uid},
            files={"file": ("ref.png", picture(), "image/png")},
        )
        self.assertEqual(200, r.status_code, r.text)
        return r.json()

    def test_running_round_shows_its_agent_steps_to_every_player(self):
        self.create()
        self.assertEqual(200, self.b.post(self.url + "/join").status_code)
        snapshot = {"request_id": "r-live"}
        with coop._edit(self.code) as room:
            room["inflight"] = dict(snapshot)
        events = coop._live_trace(self.code, snapshot)
        with events.step("plan", "director", "plans") as step:
            step.data["cached"] = False
            live = self.b.get(self.url).json()["agent_trace"]
            self.assertEqual([("director", "running")], [(s["name"], s["status"]) for s in live])
        # A step that ends while the round holds the room lock is published later.
        with coop._edit(self.code):
            with events.step("act", "stage_assets"):
                pass
        with events.step("commit", "update_world"):
            pass
        live = self.b.get(self.url).json()["agent_trace"]
        self.assertEqual(["director", "stage_assets", "update_world"], [s["name"] for s in live])
        self.assertTrue(all(s["status"] == "done" for s in live))
        # Another round's emitter never writes into this one.
        with coop._edit(self.code) as room:
            room["inflight"] = {"request_id": "r-next"}
            room["agent_trace"] = []
        with events.step("act", "render_clip"):
            pass
        self.assertEqual([], self.b.get(self.url).json()["agent_trace"])

    def test_last_local_send_automatically_starts_exactly_one_shared_video(self):
        with patch.dict(os.environ, {"COOP_AUTO_GENERATE": "1"}):
            room = self.local_room()
            ids = [m["user_id"] for m in room["members"]]
            for uid, name in zip(ids, ["剑士", "斧手"]):
                self.local_role(uid, name)
            with coop._edit(self.code) as stored:
                stored["intro"] = {
                    "status": "ready",
                    "round": 1,
                    "narration": "两人守在洞口。",
                    "world_event": "下一刻洞顶碎石将落向中央通道。",
                    "world_event_type": "environment",
                    "world_event_key": "falling_rocks",
                    "boss_candidate": {"name": "", "description": ""},
                    "objective": "避开落石。",
                    "suggestions": [],
                    "error": "",
                }
                stored["intro"]["signature"] = coop_intro.signature(stored)
            self.a.put(self.url + "/action", json={"round": 1, "action": "挥剑牵制", "target_uid": ids[0]})
            self.assertEqual([], self.pending)
            response = self.a.put(
                self.url + "/action", json={"round": 1, "action": "挥斧协作", "target_uid": ids[1]}
            )
            self.assertEqual("generating", response.json()["status"])
            self.assertEqual(1, len(self.pending))
            for _ in range(3):
                self.a.get(self.url)
            self.assertEqual(1, len(self.pending))
            self.assertEqual(409, self.launch().status_code)
            with patch.object(coop.coop_loop, "play_round", side_effect=self.video):
                fn, *args = self.pending[0]
                fn(*args)
            result = self.a.get(self.url).json()
            self.assertEqual(2, result["round"])
            self.assertEqual(1, len(result["rounds"]))
            self.assertEqual("falling_rocks", result["rounds"][0]["guidance"]["world_event_key"])
            self.assertEqual("挥剑牵制", result["rounds"][0]["inputs"][0]["action"])
            coop.maybe_generate(self.code, 1)
            self.assertEqual(1, len(self.pending))

    def test_actions_submitted_during_guidance_resume_after_it_is_ready(self):
        with patch.dict(os.environ, {"COOP_AUTO_GENERATE": "1"}):
            room = self.local_room()
            ids = [member["user_id"] for member in room["members"]]
            for uid, name in zip(ids, ["剑士", "斧手"]):
                self.local_role(uid, name)
            with coop._edit(self.code) as stored:
                stored["opening_node"] = {"id": "opening-ready"}
                stored["intro"] = {
                    "status": "generating",
                    "round": 1,
                    "request_id": "guide-next",
                    "narration": "",
                    "world_event": "",
                    "world_event_type": "none",
                    "world_event_key": "",
                    "boss_candidate": {},
                    "objective": "",
                    "suggestions": [],
                    "error": "",
                }
                stored["intro"]["signature"] = coop_intro.signature(stored)
            # A live guidance worker owns intro.lock. Without the lock this state
            # is an orphan from a stopped process and room recovery must fail it.
            with (coop._directory() / f"{self.code}.intro.lock").open("a") as intro_guard:
                fcntl.flock(intro_guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
                for uid, action in zip(ids, ["举剑掩护", "挥斧开路"]):
                    response = self.a.put(
                        self.url + "/action", json={"round": 1, "action": action, "target_uid": uid}
                    )
                    self.assertEqual(200, response.status_code, response.text)
            # The last submission is held until its matching forecast publishes.
            self.assertEqual([], self.pending)
            with coop._edit(self.code) as stored:
                stored["intro"].update(
                    status="ready",
                    narration="两人站在洞穴出口前。",
                    world_event="下一刻出口上方会落下碎石。",
                    world_event_type="environment",
                    world_event_key="falling_rocks",
                    objective="避开落石并穿过出口。",
                    suggestions=[
                        {"user_id": uid, "action": action}
                        for uid, action in zip(ids, ["向左闪避", "向右掩护"])
                    ],
                    error="",
                )
                stored["intro"]["signature"] = coop_intro.signature(stored)
            coop_intro._resume_waiting_round(self.code)
            self.assertEqual(1, len(self.pending))
            self.assertIs(self.pending[0][0], coop._run_round)
            self.assertEqual("generating", self.a.get(self.url).json()["status"])

    def test_concurrent_online_sends_start_one_video_without_the_host_browser(self):
        self.create()
        self.b.post(self.url + "/join")
        self.role(self.a, "剑士")
        self.role(self.b, "斧手")
        with patch.dict(os.environ, {"COOP_AUTO_GENERATE": "1"}):
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [
                    pool.submit(c.put, self.url + "/action", json={"round": 1, "action": a})
                    for c, a in [(self.a, "挥剑"), (self.b, "挥斧")]
                ]
                self.assertEqual([200, 200], [f.result().status_code for f in futures])
            self.assertEqual(1, len(self.pending))
            self.assertEqual("generating", self.b.get(self.url).json()["status"])

    def test_ai_completion_is_the_last_send_and_failed_video_does_not_retry_on_poll(self):
        with patch.dict(os.environ, {"COOP_AUTO_GENERATE": "1"}):
            self.ai_room()
            self.assertEqual([], self.pending)
            self.run_ai()
            self.assertEqual(1, len(self.pending))
            with patch.object(coop.coop_loop, "play_round", side_effect=RuntimeError("model unavailable")):
                fn, *args = self.pending[0]
                fn(*args)
            for _ in range(3):
                self.assertEqual("failed", self.a.get(self.url).json()["status"])
            self.assertEqual(1, len(self.pending))
            self.assertEqual(202, self.launch().status_code)
            self.assertEqual(2, len(self.pending))

    def test_opening_setup_needs_no_agent_and_refreshes_when_cast_changes(self):
        with patch.dict(os.environ, {"COOP_INTRO_ENABLED": "1"}):
            room = self.local_room()
            ids = [m["user_id"] for m in room["members"]]
            self.local_role(ids[0], "剑士")
            self.assertEqual([], self.intro_pending)
            self.local_role(ids[1], "斧手")
            self.assertEqual(0, len(self.intro_pending))
            for _ in range(3):
                self.a.get(self.url)
            room = self.a.get(self.url).json()
            self.assertEqual("ready", room["intro"]["status"])
            self.assertEqual("opening_setup", room["intro"]["source"])
            self.assertEqual([], room["intro"]["suggestions"])
            self.assertTrue(all(not m["action"] and not m["ready"] for m in room["members"]))
            self.assertEqual([], self.pending)
            first_signature = room["intro"]["signature"]
            self.local_role(ids[1], "法师")
            self.assertEqual(0, len(self.intro_pending))
            room = self.a.get(self.url).json()
            self.assertEqual("ready", room["intro"]["status"])
            self.assertEqual("opening_setup", room["intro"]["source"])
            self.assertNotEqual(first_signature, room["intro"]["signature"])

    def test_opening_video_once_without_consuming_round_then_refresh_guidance(self):
        with patch.dict(os.environ, {"COOP_INTRO_ENABLED": "1", "COOP_OPENING_VIDEO": "1"}):
            room = self.local_room()
            ids = [m["user_id"] for m in room["members"]]
            for uid, name in zip(ids, ["剑士", "斧手"]):
                self.local_role(uid, name)
            plan = {
                "narration": "巨龙苏醒，挡住出口。",
                "objective": "寻找出口。",
                "suggestions": [{"user_id": uid, "action": "沿岩壁前进"} for uid in ids],
            }
            self.assertEqual([], self.intro_pending)
            self.assertEqual("opening_setup", self.a.get(self.url).json()["intro"]["source"])
            for _ in range(3):
                self.a.get(self.url)
            self.assertEqual(202, self.a.post(self.url + "/opening/start").status_code)
            self.assertEqual(1, len(self.pending))
            snapshot = self.pending[0][2]
            self.assertTrue(snapshot["opening"])
            self.assertEqual(0, snapshot["round"])
            with patch.object(coop.coop_loop, "play_round", side_effect=self.video):
                fn, *args = self.pending[0]
                fn(*args)
            room = self.a.get(self.url).json()
            self.assertTrue(room["opening_node"]["clip"])
            self.assertEqual(1, room["round"])
            self.assertEqual([], room["rounds"])
            self.assertTrue(all(not m["ready"] and not m["action"] for m in room["members"]))
            self.assertEqual(1, len(self.intro_pending))
            reactions = {
                "world_event_key": "",
                "suggestions": [
                    {**item, "forecast_cue": "寻找出口", "response_reason": "沿岩壁探索出口方向。"}
                    for item in plan["suggestions"]
                ],
            }
            with patch.object(coop_intro.coop_trace.llm, "chat", side_effect=[plan, reactions]):
                fn, *args = self.intro_pending[0]
                fn(*args)
            self.assertEqual(1, len(self.pending))
            self.assertEqual("ready", self.a.get(self.url).json()["intro"]["status"])
            for uid in ids:
                self.a.put(self.url + "/action", json={"round": 1, "action": "前进", "target_uid": uid})
            self.assertEqual(202, self.launch().status_code)
            with patch.object(coop.coop_loop, "play_round", side_effect=self.video):
                fn, *args = self.pending[1]
                fn(*args)
            room = self.a.get(self.url).json()
            self.assertEqual(2, room["round"])
            self.assertEqual(2, len(self.intro_pending))
            self.assertEqual("generating", room["intro"]["status"])
            self.assertEqual(2, room["intro"]["round"])
            worker, payload, _ = self.intro_pending[1]
            self.assertEqual("_run_history_guidance", worker.__name__)
            self.assertEqual(1, payload["round"])
            self.assertEqual([], payload["rounds"])
            self.assertEqual("history_actions", room["intro"]["source"])
            self.assertEqual("generating", room["intro"]["status"])

    def test_guidance_history_forces_boss_reveal_and_rejects_repeated_events(self):
        room = self.local_room()
        opening_payload, _ = coop_intro.guidance_payload(
            {**room, "setting": "两人共同对抗沉睡的巨龙", "rounds": []},
            {"id": "opening-node", "reply": "两人进入宝库。"},
        )
        self.assertTrue(opening_payload["story_context"]["boss_introduction_due"])
        history = []
        for number, (event, event_key) in enumerate(
            (
                ("下一刻左侧石柱将坍塌并堵住侧路。", "pillar_collapse"),
                ("与此同时宝库地面的裂缝将向中央延伸。", "floor_crack_spreads"),
            ),
            1,
        ):
            history.append(
                {
                    "round": number,
                    "inputs": [{**member, "action": "向前移动"} for member in room["members"]],
                    "guidance": {
                        "world_event": event,
                        "world_event_type": "environment",
                        "world_event_key": event_key,
                        "boss_candidate": {"name": "", "description": ""},
                        "objective": "继续深入",
                    },
                    "observation": {"event_outcome": "observed", "summary": event},
                    "node": {"id": f"n{number}", "reply": "两人继续前进", "beat": "前进"},
                }
            )
        snapshot = {
            **room,
            "rounds": history,
            "story_state": {
                "phase": "threat",
                "environment": "石柱倒塌，地面出现裂缝",
                "boss": "",
                "players": {},
                "last_node_id": "n2",
            },
        }
        payload, _ = coop_intro.guidance_payload(snapshot, history[-1]["node"])
        context = payload["story_context"]
        self.assertTrue(context["boss_introduction_allowed"])
        self.assertTrue(context["boss_introduction_due"])
        self.assertEqual(
            ["向前移动", "向前移动"],
            [entry["action"] for entry in context["recent_rounds"][-1]["requested_actions"]],
        )
        plan = {
            "narration": "下一步可以尝试：守住中央通道。",
            "world_event": "下一刻岩甲巨龙将从裂缝下方攀上宝库平台。",
            "world_event_type": "boss_introduction",
            "world_event_key": "boss_enters_vault",
            "boss_candidate": {
                "name": "岩甲巨龙",
                "description": "灰色岩甲覆盖庞大四足躯体，金色眼睛、短角与宽翼形成清晰轮廓。",
            },
            "objective": "分守两侧迎接新威胁。",
            "suggestions": [
                {"user_id": member["user_id"], "action": "移动到通道一侧保持警戒"}
                for member in room["members"]
            ],
        }
        result, _ = guidance_checks.validate_response(plan, room["members"], payload)
        self.assertEqual("boss_introduction", result["world_event_type"])
        delayed = dict(
            plan,
            world_event="下一刻另一排碎石将从洞顶落下。",
            world_event_type="environment",
            world_event_key="more_falling_rocks",
            boss_candidate={"name": "", "description": ""},
        )
        with self.assertRaisesRegex(ValueError, "must forecast the boss"):
            guidance_checks.validate_response(delayed, room["members"], payload)

        repeated_context = {
            "story_context": {
                "boss_introduction_allowed": True,
                "boss_introduction_due": False,
                "recent_rounds": context["recent_rounds"][-1:],
            }
        }
        repeated = dict(
            plan,
            world_event="下一刻中央地缝会继续向队伍脚下扩张。",
            world_event_type="environment",
            world_event_key="floor_crack_spreads",
            boss_candidate={"name": "", "description": ""},
        )
        result, _ = guidance_checks.validate_response(repeated, room["members"], repeated_context)
        self.assertEqual("floor_crack_spreads", result["world_event_key"])

    def test_visual_observation_updates_story_state_and_consumed_event_history(self):
        room = {
            "rounds": [
                {
                    "round": 3,
                    "guidance": {
                        "world_event": "下一刻巨龙将撞开右侧石门。",
                        "world_event_type": "boss_action",
                        "world_event_key": "boss_breaks_gate",
                    },
                    "node": {"id": "n-story"},
                }
            ]
        }
        observation = {
            "node_id": "",
            "event_outcome": "observed",
            "summary": "巨龙撞开石门，两人退至左侧。",
            "environment_state": "右侧石门已经破裂并形成新出口。",
            "boss_state": "巨龙停在破裂石门前。",
            "player_states": [
                {"user_id": "u1", "state": "位于左侧石柱后。"},
                {"user_id": "u2", "state": "位于左侧出口旁。"},
            ],
            "story_phase": "escalation",
        }
        coop_intro.publish_observation(room, observation, "n-story")
        self.assertEqual("escalation", room["story_state"]["phase"])
        self.assertEqual("右侧石门已经破裂并形成新出口。", room["story_state"]["environment"])
        self.assertEqual("observed", room["story_state"]["recent_events"][-1]["outcome"])
        self.assertEqual(observation["summary"], room["rounds"][0]["observation"]["summary"])

    def test_failed_opening_retries_without_player_actions_and_recovers_saved_clip(self):
        with patch.dict(os.environ, {"COOP_INTRO_ENABLED": "1", "COOP_OPENING_VIDEO": "1"}):
            room = self.local_room()
            ids = [m["user_id"] for m in room["members"]]
            for uid, name in zip(ids, ["剑士", "斧手"]):
                self.local_role(uid, name)
            self.assertEqual([], self.intro_pending)
            self.assertEqual(202, self.a.post(self.url + "/opening/start").status_code)
            with patch.object(
                coop.coop_loop, "play_round", side_effect=ConnectionRefusedError(111, "Connection refused")
            ):
                fn, *args = self.pending[0]
                fn(*args)
            for _ in range(2):
                failed = self.a.get(self.url).json()
                self.assertEqual("failed", failed["status"])
                self.assertTrue(failed["opening_generation"])
                self.assertIn("[Errno 111] Connection refused", failed["error"])
            retry = self.a.post(self.url + "/opening/start")
            self.assertEqual(202, retry.status_code, retry.text)
            self.assertEqual("generating", retry.json()["status"])
            self.assertEqual(2, len(self.pending))
            _, _, snapshot, guard = self.pending[1]
            world = World.load(snapshot["world_id"])
            self.video(
                world, coop.merged_prompt(snapshot), mentions=[m["subject_id"] for m in snapshot["members"]]
            )
            guard.close()
            result = self.a.get(self.url).json()
            self.assertEqual("waiting", result["status"])
            self.assertEqual(1, result["round"])
            self.assertTrue(result["opening_node"]["clip"])
            self.assertEqual([], result["rounds"])
            self.assertEqual(2, len(self.pending))

    def test_opening_setup_is_not_replaced_by_a_stale_worker(self):
        with patch.dict(os.environ, {"COOP_INTRO_ENABLED": "1"}):
            room = self.local_room()
            ids = [m["user_id"] for m in room["members"]]
            for uid, name in zip(ids, ["剑士", "斧手"]):
                self.local_role(uid, name)
            room = self.a.get(self.url).json()
            self.assertEqual("opening_setup", room["intro"]["source"])
            first_signature = room["intro"]["signature"]
            self.local_role(ids[1], "法师")
            room = self.a.get(self.url).json()
            self.assertEqual("ready", room["intro"]["status"])
            self.assertEqual("opening_setup", room["intro"]["source"])
            self.assertNotEqual(first_signature, room["intro"]["signature"])
            self.assertEqual(0, len(self.intro_pending))
            self.assertEqual([], self.pending)

    def test_room_events_push_updates_reconnect_and_revoke_membership(self):
        self.local_room()
        self.assertEqual(403, self.b.get(self.url + "/events").status_code)
        self.assertEqual(401, self.anon.get(self.url + "/events").status_code)

        async def check():
            stream = coop_events.stream(self.code, self.ua["id"])
            initial = await anext(stream)
            self.assertIn("event: room", initial)
            pending = asyncio.create_task(anext(stream))
            await asyncio.sleep(0.02)

            def edit():
                with coop._edit(self.code) as r:
                    r.update(preview_clip="ready.mp4", progress="video ready")

            await asyncio.to_thread(edit)
            update = await asyncio.wait_for(pending, 0.5)
            self.assertIn("ready.mp4", update)
            await stream.aclose()
            self.assertNotIn(self.code, coop_events._watchers)
            reconnect = coop_events.stream(self.code, self.ua["id"])
            self.assertIn("ready.mp4", await anext(reconnect))

            def revoke():
                with coop._edit(self.code) as r:
                    r["members"] = []

            await asyncio.to_thread(revoke)
            self.assertIn("event: revoked", await asyncio.wait_for(anext(reconnect), 0.5))
            await reconnect.aclose()
            self.assertNotIn(self.code, coop_events._watchers)

        asyncio.run(check())

    def test_generation_narration_is_public_only_while_video_is_pending(self):
        room = self.ready()
        self.assertEqual(202, self.launch(room).status_code)
        narration = "巨龙正朝出口逼近，角色A挥剑牵制，角色B同时举斧从侧面配合。"

        def narrated(world, text, **kwargs):
            kwargs["on_narration_ready"](narration)
            live = self.a.get(self.url).json()
            self.assertEqual("generating", live["status"])
            self.assertEqual(narration, live["generation_narration"])
            return self.video(world, text, **kwargs)

        with patch.object(coop.coop_loop, "play_round", side_effect=narrated):
            fn, *args = self.pending[0]
            fn(*args)
        completed = self.a.get(self.url).json()
        self.assertEqual("waiting", completed["status"])
        self.assertNotIn("generation_narration", completed)

    def test_coop_log_is_shared_only_with_members_and_keeps_full_text(self):
        room = self.ready()
        long = "完整文本" * 1800
        with coop.coop_trace.operation(room["world_id"], 1, "test-log"):
            reqlog.record("agent_request", request={"system": long, "user": "双方动作"})
            with reqlog.scope(stage="generation"):
                reqlog.record("video_request", prompt=long)
        a = self.a.get(self.url + "/log")
        b = self.b.get(self.url + "/log")
        self.assertEqual(200, a.status_code)
        self.assertEqual(a.json(), b.json())
        self.assertEqual(long, a.json()[0]["prompt"])
        self.assertEqual("generation", a.json()[0]["stage"])
        self.assertEqual("agent", a.json()[1]["stage"])
        self.assertEqual(1, a.json()[0]["round"])
        self.assertEqual(403, self.c.get(self.url + "/log").status_code)
        self.assertEqual(401, self.anon.get(self.url + "/log").status_code)
        self.assertEqual(404, self.b.get("/api/worlds/" + room["world_id"] + "/log").status_code)

    def test_local_owner_controls_two_distinct_seats_and_one_generation(self):
        room = self.local_room()
        self.assertEqual("local", room["mode"])
        self.assertEqual(2, room["capacity"])
        ids = [p["user_id"] for p in room["members"]]
        self.assertEqual(self.ua["id"], ids[0])
        self.assertNotEqual(*ids)
        for uid, name in zip(ids, ["持剑者", "持斧者"]):
            self.local_role(uid, name)
        for uid, action in zip(ids, ["用剑砍怪兽", "用斧子劈他"]):
            r = self.a.put(self.url + "/action", json={"round": 1, "action": action, "target_uid": uid})
            self.assertEqual(200, r.status_code, r.text)
        room = self.a.get(self.url).json()
        self.assertTrue(all(p["ready"] for p in room["members"]))
        self.assertEqual(202, self.launch(room).status_code)
        self.assertEqual(409, self.launch(room).status_code)
        with patch.object(coop.coop_loop, "play_round", side_effect=self.video) as render:
            fn, *args = self.pending.pop()
            fn(*args)
        self.assertEqual(1, render.call_count)
        self.assertEqual([m["subject_id"] for m in room["members"]], render.call_args.kwargs["mentions"])
        result = self.a.get(self.url).json()
        self.assertEqual(2, result["round"])
        self.assertEqual(1, len(result["rounds"]))
        self.assertEqual(["用剑砍怪兽", "用斧子劈他"], [p["action"] for p in result["rounds"][0]["inputs"]])
        self.assertTrue(all(not m["ready"] and not m["action"] for m in result["members"]))

    def test_local_room_cannot_be_joined_or_modified_by_other_user(self):
        room = self.local_room()
        target = room["members"][1]["user_id"]
        self.assertEqual("", room["invite_url"])
        self.assertEqual(403, self.b.post(self.url + "/join").status_code)
        self.assertEqual(403, self.b.get(self.url).status_code)
        self.assertEqual([], self.b.get("/api/coop/rooms").json())
        self.assertEqual(
            403,
            self.b.put(
                self.url + "/action", json={"round": 1, "action": "攻击", "target_uid": target}
            ).status_code,
        )
        self.assertEqual(
            403,
            self.b.post(self.url + "/portrait", json={"randomize": True, "target_uid": target}).status_code,
        )
        self.assertEqual(409, self.a.post(self.url + "/ai", json={"profile": "axe"}).status_code)
        self.assertEqual(200, self.a.post(self.url + "/leave").status_code)
        self.assertEqual([], self.a.get("/api/coop/rooms").json())
        self.assertEqual(
            "closed", json.loads((config.DATA / "coop" / f"{self.code}.json").read_text())["status"]
        )

    def test_online_host_cannot_take_over_another_human(self):
        self.ready()
        target = self.ub["id"]
        self.assertEqual(
            403,
            self.a.put(
                self.url + "/action", json={"round": 1, "action": "冒充", "target_uid": target}
            ).status_code,
        )
        self.assertEqual(
            403,
            self.a.post(
                self.url + "/character",
                data={"name": "冒充", "target_uid": target},
                files={"file": ("x.png", picture(), "image/png")},
            ).status_code,
        )
        self.assertEqual(
            403,
            self.a.post(self.url + "/portrait", json={"randomize": True, "target_uid": target}).status_code,
        )
        room = self.b.get(self.url).json()
        self.assertEqual("用斧子劈他", room["members"][1]["action"])
        self.assertEqual("online", room["mode"])

    def test_local_reference_and_action_changes_affect_only_selected_seat(self):
        room = self.local_room()
        ids = [p["user_id"] for p in room["members"]]
        for uid, name in zip(ids, ["角色A", "角色B"]):
            self.local_role(uid, name)
        for uid, action in zip(ids, ["挥剑", "挥斧"]):
            self.a.put(self.url + "/action", json={"round": 1, "action": action, "target_uid": uid})
        r = self.a.put(
            self.url + "/action", json={"round": 1, "action": "挥斧", "ready": False, "target_uid": ids[1]}
        )
        self.assertTrue(r.json()["members"][0]["ready"])
        self.assertFalse(r.json()["members"][1]["ready"])
        self.assertEqual(
            409,
            self.a.put(
                self.url + "/action", json={"round": 2, "action": "旧回合", "target_uid": ids[1]}
            ).status_code,
        )
        self.assertEqual(
            409,
            self.a.post(self.url + "/character", data={"name": "角色A", "target_uid": ids[1]}).status_code,
        )
        self.assertEqual(
            202,
            self.a.post(self.url + "/portrait", json={"randomize": True, "target_uid": ids[1]}).status_code,
        )
        fn, run, snapshot = self.portrait_pending.pop()
        self.assertEqual(ids[1], snapshot["target_uid"])
        with patch.object(coop_portrait, "draw", return_value=picture()):
            fn(run, snapshot)
        r = self.a.get(self.url).json()
        before = r["members"][0].copy()
        ref = r["members"][1]["portrait"]["request_id"]
        r = self.a.post(self.url + "/portrait/use", json={"target_uid": ids[1], "request_id": ref})
        self.assertEqual(200, r.status_code, r.text)
        self.assertEqual(before["ref"], r.json()["members"][0]["ref"])
        self.assertEqual(before["character"], r.json()["members"][0]["character"])
        self.assertNotEqual(r.json()["members"][0]["ref"], r.json()["members"][1]["ref"])
        self.assertFalse(any(p["ready"] for p in r.json()["members"]))

    def test_old_rooms_default_to_online_mode(self):
        self.create()
        path = config.DATA / "coop" / f"{self.code}.json"
        raw = json.loads(path.read_text())
        raw.pop("mode")
        path.write_text(json.dumps(raw))
        self.assertEqual("online", self.a.get(self.url).json()["mode"])
        self.assertEqual("online", self.a.get("/api/coop/rooms").json()[0]["mode"])
        self.assertEqual(200, self.b.post(self.url + "/join").status_code)

    def test_login_membership_and_personal_world_permissions(self):
        room = self.create()
        self.assertEqual(401, self.anon.get("/api/coop/rooms").status_code)
        self.assertEqual(403, self.b.get(self.url).status_code)
        self.assertEqual([], self.b.get("/api/coop/rooms").json())
        self.b.post(self.url + "/join")
        self.assertEqual(200, self.b.get(self.url).status_code)
        # Joining a room does not grant access to its owner's personal APIs.
        self.assertEqual(404, self.b.get("/api/worlds/" + room["world_id"]).status_code)
        personal = self.a.post("/api/worlds", json={"title": "私人世界"}).json()
        self.assertEqual(404, self.b.get("/api/worlds/" + personal["id"]).status_code)

    def test_join_is_idempotent_and_capacity_is_enforced(self):
        self.create(2)
        self.b.post(self.url + "/join")
        self.b.post(self.url + "/join")
        self.assertEqual(2, len(self.a.get(self.url).json()["members"]))
        self.assertEqual(409, self.c.post(self.url + "/join").status_code)

    def test_input_validation_and_path_traversal(self):
        self.create()
        self.assertEqual(404, self.a.post("/api/coop/rooms/NOTACODE/join").status_code)
        self.assertEqual(
            422,
            self.a.post("/api/coop/rooms", json={"title": "x", "setting": "x", "capacity": 5}).status_code,
        )
        self.assertEqual(
            400,
            self.a.post(
                self.url + "/character", data={"name": "x"}, files={"file": ("x.png", b"bad", "image/png")}
            ).status_code,
        )
        self.assertEqual(
            403,
            self.b.post(
                self.url + "/character", data={"name": "x"}, files={"file": ("x.png", picture(), "image/png")}
            ).status_code,
        )

    def test_roles_are_unique_and_upload_is_scoped_to_actor(self):
        self.create()
        self.b.post(self.url + "/join")
        a = self.role(self.a, "游侠")["members"][0]
        self.assertEqual(
            409,
            self.b.post(
                self.url + "/character",
                data={"name": "游侠"},
                files={"file": ("x.png", picture(), "image/png")},
            ).status_code,
        )
        room = self.role(self.b, "战士")
        self.assertEqual(a, room["members"][0])
        self.assertNotEqual(room["members"][0]["subject_id"], room["members"][1]["subject_id"])
        world = World.load(room["world_id"])
        for m in room["members"]:
            self.assertEqual(m["ref"], world.subjects[m["subject_id"]]["ref"])
            self.assertTrue((config.REFS / m["ref"]).is_file())

    def test_only_host_updates_shared_scene(self):
        self.create()
        self.b.post(self.url + "/join")
        kwargs = {"data": {"name": "巨龙山洞"}, "files": {"file": ("scene.png", picture(), "image/png")}}
        self.assertEqual(403, self.b.post(self.url + "/scene", **kwargs).status_code)
        r = self.a.post(self.url + "/scene", **kwargs)
        self.assertEqual(200, r.status_code)
        self.assertEqual("巨龙山洞", World.load(r.json()["world_id"]).here)

    def test_concurrent_actions_preserve_both_players_and_round(self):
        self.create()
        self.b.post(self.url + "/join")
        self.role(self.a, "角色A")
        self.role(self.b, "角色B")
        with ThreadPoolExecutor(2) as pool:
            futures = [
                pool.submit(coop.submit_action, self.code, u["id"], action, 1)
                for u, action in [(self.ua, "用剑砍怪兽"), (self.ub, "用斧子劈他")]
            ]
            for f in futures:
                f.result()
        room = self.a.get(self.url).json()
        self.assertEqual(["用剑砍怪兽", "用斧子劈他"], [m["action"] for m in room["members"]])
        self.assertTrue(all(m["ready"] for m in room["members"]))
        for text in ("角色A", "角色B", "用剑砍怪兽", "用斧子劈他"):
            self.assertIn(text, room["merged_prompt"])

    def test_action_history_keeps_server_acceptance_order(self):
        self.create()
        self.b.post(self.url + "/join")
        self.role(self.a, "角色A")
        self.role(self.b, "角色B")
        self.b.put(self.url + "/action", json={"round": 1, "action": "角色B先观察出口"})
        self.a.put(self.url + "/action", json={"round": 1, "action": "角色A随后举剑掩护"})
        room = self.a.get(self.url).json()
        self.assertEqual([self.ub["id"], self.ua["id"]], room["action_order"])
        self.assertEqual(202, self.launch(room).status_code)
        with patch.object(coop.coop_loop, "play_round", side_effect=self.video):
            fn, *args = self.pending.pop()
            fn(*args)
        result = self.a.get(self.url).json()
        self.assertEqual(
            ["角色B先观察出口", "角色A随后举剑掩护"],
            [entry["action"] for entry in result["rounds"][0]["inputs"]],
        )

    def test_rollback_restores_shared_state_inputs_and_preserves_media(self):
        room = self.ready()
        world = World.load(room["world_id"])
        parent = Node(
            id="parent",
            parent="",
            video_id="parent-video",
            place="",
            beat="opening",
            mode="cut",
            clip="parent.mp4",
            artifact_id="parent-latent",
        )
        world.nodes[parent.id] = asdict(parent)
        world.head = parent.id
        world.save()
        with coop._edit(self.code) as saved:
            saved["opening_node"] = asdict(parent)
            saved["asset_extraction"] = {"status": "ready", "request_id": "opening-assets"}
            saved["story_state"] = {"environment": "before"}
            saved["story_progress"] = {"stages": {}, "version": 1}
        self.assertEqual(202, self.launch().status_code)
        with (
            patch.object(coop.coop_loop, "play_round", side_effect=self.video),
            patch.object(coop, "_after_round"),
        ):
            fn, *args = self.pending.pop()
            fn(*args)
        with coop._edit(self.code) as saved:
            saved["story_state"] = {"environment": "after"}
            saved["story_progress"] = {"stages": {"goal": {"status": "completed"}}}
            saved["intro"] = {"status": "ready", "narration": "stale"}
        media = config.CLIPS / "one.mp4"
        media.write_bytes(b"keep")
        body = {"node_id": "n-coop"}
        self.assertEqual(403, self.b.post(self.url + "/rollback", json=body).status_code)
        response = self.a.post(self.url + "/rollback", json=body)
        self.assertEqual(200, response.status_code, response.text)
        current = response.json()
        self.assertEqual([], current["rounds"])
        self.assertEqual(1, current["round"])
        self.assertEqual("before", current["story_state"]["environment"])
        self.assertEqual({}, current["story_progress"]["stages"])
        self.assertEqual(["用剑砍怪兽", "用斧子劈他"], [m["action"] for m in current["members"]])
        self.assertTrue(all(not m["ready"] for m in current["members"]))
        self.assertNotIn("_rollback_states", current)
        world = World.load(room["world_id"])
        self.assertEqual("parent", world.head)
        self.assertEqual("parent-latent", world.head_node.artifact_id)
        self.assertEqual("ready", current["asset_extraction"]["status"])
        self.assertIn("n-coop", world.nodes)
        self.assertTrue(media.is_file())
        self.assertEqual(409, self.a.post(self.url + "/rollback", json=body).status_code)
        for client, action in [(self.a, "换一个动作"), (self.b, "重新尝试")]:
            self.assertEqual(
                200, client.put(self.url + "/action", json={"action": action, "round": 1}).status_code
            )
        self.assertEqual(202, self.launch().status_code)

    def test_rollback_rejects_generation_and_stale_head(self):
        self.ready()
        self.assertEqual(202, self.launch().status_code)
        self.assertEqual(409, self.a.post(self.url + "/rollback", json={"node_id": "old"}).status_code)
        with patch.object(coop.coop_loop, "play_round", side_effect=self.video):
            fn, *args = self.pending.pop()
            fn(*args)
        self.assertEqual(409, self.a.post(self.url + "/rollback", json={"node_id": "old"}).status_code)

    def test_rollback_rejects_legacy_opening_and_keeps_world_and_characters(self):
        room = self.ready()
        world = World.load(room["world_id"])
        node = self.video(world, "opening", mentions=[])
        with coop._edit(self.code) as saved:
            saved["opening_node"] = asdict(node)
        before = asdict(World.load(world.id))
        response = self.a.post(self.url + "/rollback", json={"node_id": node.id})
        self.assertEqual(409, response.status_code, response.text)
        current = self.a.get(self.url).json()
        self.assertEqual(node.id, current["opening_node"]["id"])
        self.assertEqual(2, len(current["members"]))
        self.assertEqual(before, asdict(World.load(world.id)))

    def test_all_players_ready_and_host_only(self):
        self.create()
        self.assertEqual(409, self.launch().status_code)
        self.b.post(self.url + "/join")
        self.assertEqual(403, self.launch(client=self.b).status_code)

    def test_stale_preview_cannot_generate_different_actions(self):
        room = self.ready()
        self.b.put(self.url + "/action", json={"action": "改为举盾保护队友", "round": 1})
        stale = self.a.post(
            self.url + "/generate",
            json={
                "round": room["round"],
                "revision": room["revision"],
                "generation_signature": room["generation_signature"],
            },
        )
        self.assertEqual(409, stale.status_code)
        self.assertEqual("generation_inputs_changed", stale.json()["detail"]["code"])
        self.assertEqual([], self.pending)

    def test_background_room_update_does_not_make_generation_preview_stale(self):
        room = self.ready()
        # Prefetch, observation and guidance workers may persist diagnostic
        # fields while the host is reviewing the same locked actions. These
        # writes must not invalidate the generation request.
        with coop._edit(self.code) as stored:
            stored["progress"] = "后台预生成状态更新"
        response = self.a.post(
            self.url + "/generate",
            json={
                "round": room["round"],
                "revision": room["revision"],
                "generation_signature": room["generation_signature"],
            },
        )
        self.assertEqual(202, response.status_code, response.text)
        self.assertEqual(1, len(self.pending))

    def test_one_merged_generation_with_all_references_shared_by_clients(self):
        room = self.ready()
        self.assertEqual(202, self.launch(room).status_code)
        self.assertEqual(409, self.launch(room).status_code)
        self.assertEqual(
            409, self.b.put(self.url + "/action", json={"action": "改动作", "round": 1}).status_code
        )
        self.assertEqual(409, self.b.post(self.url + "/leave").status_code)
        with patch.object(coop.coop_loop, "play_round", side_effect=self.video) as generate:
            fn, *args = self.pending[0]
            fn(*args)
            generate.assert_called_once()
            self.assertEqual(
                [m["subject_id"] for m in room["members"]], generate.call_args.kwargs["mentions"]
            )
            self.assertIn("用斧子劈他", generate.call_args.args[1])
        a, b = self.a.get(self.url).json(), self.b.get(self.url).json()
        self.assertEqual(a["rounds"], b["rounds"])
        self.assertEqual(2, a["round"])
        self.assertEqual("one.mp4", a["rounds"][0]["node"]["clip"])
        self.assertTrue(all(not m["ready"] and not m["action"] for m in a["members"]))
        self.assertEqual(
            409, self.b.put(self.url + "/action", json={"action": "旧一轮", "round": 1}).status_code
        )

    def test_generation_failure_preserves_actions_and_allows_retry(self):
        self.ready()
        self.launch()
        with patch.object(coop.coop_loop, "play_round", side_effect=RuntimeError("model offline")):
            fn, *args = self.pending[0]
            fn(*args)
        room = self.a.get(self.url).json()
        self.assertEqual("failed", room["status"])
        self.assertTrue(all(m["ready"] for m in room["members"]))
        self.assertEqual(1, room["round"])
        self.assertEqual(202, self.launch().status_code)

    def test_restart_recovers_abandoned_round_without_automatic_resubmit(self):
        self.ready()
        self.launch()
        self.pending[0][-1].close()  # OS releases this lock when a worker exits.
        room = self.a.get(self.url).json()
        self.assertEqual("failed", room["status"])
        self.assertEqual("用剑砍怪兽", room["members"][0]["action"])
        self.assertEqual(1, len(self.pending))

    def test_reference_change_invalidates_ready_preview(self):
        self.ready()
        self.role(self.a, "新版游侠")
        self.assertFalse(any(m["ready"] for m in self.a.get(self.url).json()["members"]))

    def test_restart_after_clip_saved_publishes_it_without_rendering_again(self):
        self.ready()
        self.launch()
        _, _, snapshot, guard = self.pending[0]
        world = World.load(snapshot["world_id"])
        self.video(
            world, coop.merged_prompt(snapshot), mentions=[m["subject_id"] for m in snapshot["members"]]
        )
        guard.close()
        room = self.b.get(self.url).json()
        self.assertEqual(2, room["round"])
        self.assertEqual("one.mp4", room["rounds"][0]["node"]["clip"])
        self.assertEqual("waiting", room["status"])
        self.assertEqual(1, len(self.pending))

    def test_external_reference_change_rejected_before_generation(self):
        room = self.ready()
        world = World.load(room["world_id"])
        world.subjects[room["members"][0]["subject_id"]]["ref"] = "unexpected.jpg"
        world.save()
        self.assertEqual(409, self.launch().status_code)
        self.assertEqual([], self.pending)

    def test_four_players_share_one_pipeline_call(self):
        self.ready()
        self.c.post(self.url + "/join")
        d = TestClient(create_app())
        d.post("/api/users", json={"name": "Dave"})
        d.post(self.url + "/join")
        self.role(self.c, "角色C")
        self.role(d, "角色D")
        for client, action in [(self.a, "挥剑"), (self.b, "劈斧"), (self.c, "施法"), (d, "举盾")]:
            self.assertEqual(
                200, client.put(self.url + "/action", json={"action": action, "round": 1}).status_code
            )
        self.assertEqual(202, self.launch().status_code)
        with patch.object(coop.coop_loop, "play_round", side_effect=self.video) as call:
            fn, *args = self.pending[0]
            fn(*args)
        self.assertEqual(1, call.call_count)
        self.assertEqual(4, len(call.call_args.kwargs["mentions"]))
        self.assertEqual(self.a.get(self.url).json()["rounds"], d.get(self.url).json()["rounds"])
        d.close()

    def test_reference_changed_while_queued_never_reaches_pipeline(self):
        room = self.ready()
        self.launch()
        world = World.load(room["world_id"])
        world.subjects[room["members"][0]["subject_id"]]["ref"] = "changed-while-queued.jpg"
        world.save()
        with patch.object(coop.coop_loop, "play_round") as call:
            fn, *args = self.pending[0]
            fn(*args)
        call.assert_not_called()
        failed = self.a.get(self.url).json()
        self.assertEqual("failed", failed["status"])
        self.assertEqual(1, failed["round"])
        self.assertEqual("用剑砍怪兽", failed["members"][0]["action"])

    def test_host_transfer_and_closed_room(self):
        room = self.create()
        self.b.post(self.url + "/join")
        self.a.post(self.url + "/leave")
        self.assertEqual(self.ub["id"], self.b.get(self.url).json()["host_id"])
        self.assertEqual(self.ub["id"], World.load(room["world_id"]).owner_id)
        self.assertEqual(403, self.a.get(self.url).status_code)
        self.b.post(self.url + "/leave")
        self.assertEqual(410, self.a.post(self.url + "/join").status_code)

    def test_isolated_environment_blocks_generation_but_keeps_preview(self):
        self.ready()
        with patch.dict(os.environ, {"COOP_GENERATION_ENABLED": "0"}):
            room = self.a.get(self.url).json()
            self.assertFalse(room["generation_enabled"])
            self.assertIn("角色B", room["merged_prompt"])
            self.assertEqual(503, self.launch().status_code)
            self.assertEqual([], self.pending)

    def ai_room(self):
        self.create()
        self.role(self.a, "游侠")
        r = self.a.post(self.url + "/ai", json={"profile": "axe"})
        self.assertEqual(200, r.status_code, r.text)
        self.a.put(self.url + "/action", json={"action": "用剑砍巨龙", "round": 1})
        return self.a.get(self.url).json()

    def run_ai(self, index=-1):
        fn, *args = self.ai_pending[index]
        fn(*args)
        return self.a.get(self.url).json()

    def suggest(self, actions):
        """A ready forecast whose suggestions are `actions` (by user id)."""
        with coop._edit(self.code) as stored:
            stored["intro"] = {
                "status": "ready",
                "round": stored["round"],
                "narration": "",
                "world_event": "",
                "world_event_type": "",
                "world_event_key": "",
                "boss_candidate": {"name": "", "description": ""},
                "objective": "",
                "suggestions": [{"user_id": uid, "action": text} for uid, text in actions.items()],
                "error": "",
            }
            stored["intro"]["signature"] = coop_intro.signature(stored)

    def test_ai_follows_its_suggestion_when_every_human_followed_theirs(self):
        # The clip of exactly the suggested actions may already be rendering; an AI writing its
        # own action made it unusable every time (0 of 5 used in one test).
        self.create()
        self.role(self.a, "游侠")
        bot = self.a.post(self.url + "/ai", json={"profile": "axe"}).json()["members"][1]["user_id"]
        self.suggest({self.ua["id"]: "举剑牵制巨龙", bot: "挥斧砍向巨龙侧翼"})
        with (
            patch.object(llm, "chat") as model,
            patch.object(coop, "maybe_generate") as start,
        ):
            self.a.put(self.url + "/action", json={"action": "举剑牵制巨龙", "round": 1})
        model.assert_not_called()
        self.assertEqual([], self.ai_pending)
        start.assert_called_with(
            self.code, 1
        )  # the submit starts the round as well; a second start is a no-op
        member = next(m for m in self.a.get(self.url).json()["members"] if m["user_id"] == bot)
        self.assertEqual(
            ("挥斧砍向巨龙侧翼", True, "suggestion"), (member["action"], member["ready"], member["ai_source"])
        )

    def test_ai_seated_between_players_keeps_the_prefetched_order(self):
        # host, AI, friend: appended after the humans, the AI's action changed the order the
        # suggestions were rendered in, and the prefetched clip was discarded as out of order.
        self.create()
        self.role(self.a, "游侠")
        bot = self.a.post(self.url + "/ai", json={"profile": "axe"}).json()["members"][1]["user_id"]
        self.assertEqual(200, self.b.post(self.url + "/join").status_code)
        self.role(self.b, "法师")
        self.suggest({self.ua["id"]: "举剑牵制巨龙", bot: "挥斧砍向巨龙侧翼", self.ub["id"]: "念咒冻住龙爪"})
        with patch.object(llm, "chat"), patch.object(coop, "maybe_generate"):
            self.a.put(self.url + "/action", json={"action": "举剑牵制巨龙", "round": 1})
            self.b.put(self.url + "/action", json={"action": "念咒冻住龙爪", "round": 1})
        with coop._edit(self.code) as stored:
            self.assertEqual([self.ua["id"], bot, self.ub["id"]], stored["action_order"])
            self.assertEqual(
                stored["action_order"], coop_prefetch._action_order({**stored, "action_order": []})
            )

    def test_ai_writes_its_own_action_when_a_human_changed_theirs(self):
        self.create()
        self.role(self.a, "游侠")
        bot = self.a.post(self.url + "/ai", json={"profile": "axe"}).json()["members"][1]["user_id"]
        self.suggest({self.ua["id"]: "举剑牵制巨龙", bot: "挥斧砍向巨龙侧翼"})
        self.a.put(self.url + "/action", json={"action": "转身撤退，寻找出口", "round": 1})
        self.assertEqual(1, len(self.ai_pending))
        self.assertEqual("thinking", self.a.get(self.url).json()["ai_status"])

    def test_an_ai_portrait_drawn_during_a_round_is_bound_after_it(self):
        # The drawing waits behind the default portraits; if a round was filming by then the bind
        # was refused and dropped, leaving the built-in look the video model does not film.
        self.create()
        with patch.dict(os.environ, {"COOP_AUTO_GENERATE": "1"}):
            builtin = self.a.post(self.url + "/ai", json={"profile": "axe"}).json()["members"][1]["ref"]
        with coop._edit(self.code) as stored:
            status, stored["status"] = stored["status"], "generating"
        with (
            patch.object(coop_portrait, "draw", return_value=picture()),
            patch.object(coop_portrait, "_AI_BIND_RETRY_SECONDS", 0.05),
        ):
            fn, *args = self.portrait_pending.pop()
            fn(*args)
            self.assertEqual(builtin, self.a.get(self.url).json()["members"][1]["ref"])
            with coop._edit(self.code) as stored:
                stored["status"] = status
            for _ in range(100):
                ref = self.a.get(self.url).json()["members"][1]["ref"]
                if ref != builtin:
                    break
                time.sleep(0.05)
        self.assertTrue(ref.startswith("coop-portrait-"), ref)

    def test_a_new_ai_teammate_gets_a_drawn_portrait(self):
        # Its built-in look is a flat illustration: the teammate was missing from every clip
        # until a portrait was drawn by hand.
        self.create()
        with patch.dict(os.environ, {"COOP_AUTO_GENERATE": "1"}):
            room = self.a.post(self.url + "/ai", json={"profile": "axe"}).json()
        bot = room["members"][1]
        self.assertEqual("generating", bot["portrait"]["status"])
        self.assertTrue(bot["ref"].startswith("coop-ai-"))
        with patch.object(coop_portrait, "draw", return_value=picture()):
            fn, *args = self.portrait_pending.pop()
            fn(*args)
        bot = self.a.get(self.url).json()["members"][1]
        self.assertTrue(bot["ref"].startswith("coop-portrait-"))
        self.assertEqual("applied", bot["portrait"]["status"])
        self.assertEqual(bot["ref"], World.load(room["world_id"]).subjects[bot["subject_id"]]["ref"])

    def test_the_drawn_ai_portrait_never_replaces_a_look_the_host_set(self):
        self.create()
        with patch.dict(os.environ, {"COOP_AUTO_GENERATE": "1"}):
            bot = self.a.post(self.url + "/ai", json={"profile": "axe"}).json()["members"][1]
        automatic = self.portrait_pending.pop()
        # Meanwhile the host draws and adopts a look of their own for the teammate.
        self.start_portrait(target_uid=bot["user_id"])
        with patch.object(coop_portrait, "draw", return_value=picture()):
            fn, *args = self.portrait_pending.pop()
            fn(*args)
            request_id = self.a.get(self.url).json()["members"][1]["portrait"]["request_id"]
            chosen = self.a.post(
                self.url + "/portrait/use", json={"request_id": request_id, "target_uid": bot["user_id"]}
            ).json()["members"][1]["ref"]
            automatic[0](*automatic[1:])
        bot = self.a.get(self.url).json()["members"][1]
        self.assertTrue(chosen.startswith("coop-portrait-"))
        self.assertEqual(chosen, bot["ref"])

    def test_ai_has_own_reference_and_host_controls_seats(self):
        self.create(2)
        self.assertEqual(403, self.b.post(self.url + "/ai", json={"profile": "axe"}).status_code)
        self.assertEqual(400, self.a.post(self.url + "/ai", json={"profile": "invalid"}).status_code)
        room = self.a.post(self.url + "/ai", json={"profile": "axe"}).json()
        bot = room["members"][1]
        self.assertEqual("ai", bot["kind"])
        self.assertTrue((config.REFS / bot["ref"]).is_file())
        self.assertEqual(bot["ref"], World.load(room["world_id"]).subjects[bot["subject_id"]]["ref"])
        self.assertFalse(any(u["id"] == bot["user_id"] for u in self.a.get("/api/users").json()))
        self.assertEqual(2, len(self.a.post(self.url + "/ai", json={"profile": "axe"}).json()["members"]))
        self.assertEqual(409, self.a.post(self.url + "/ai", json={"profile": "mage"}).status_code)
        self.assertEqual(403, self.b.delete(self.url + "/ai/" + bot["user_id"]).status_code)
        self.assertEqual(404, self.a.delete(self.url + "/ai/" + self.ua["id"]).status_code)

    def test_ai_offline_pair_makes_one_joint_video_request(self):
        self.ai_room()
        self.assertEqual(1, len(self.ai_pending))
        with patch.object(llm, "chat") as llm_call:
            room = self.run_ai()
        llm_call.assert_not_called()
        bot = room["members"][1]
        self.assertEqual("offline", bot["ai_source"])
        self.assertIn("战斧", bot["action"])
        self.assertIn("巨龙", bot["action"])
        self.assertTrue(all(m["ready"] for m in room["members"]))
        self.assertEqual(202, self.launch(room).status_code)
        self.assertEqual(409, self.a.post(self.url + "/ai", json={"profile": "shield"}).status_code)
        self.assertEqual(409, self.a.delete(self.url + "/ai/" + bot["user_id"]).status_code)
        with patch.object(coop.coop_loop, "play_round", side_effect=self.video) as pipeline_call:
            fn, *args = self.pending[0]
            fn(*args)
        pipeline_call.assert_called_once()
        self.assertEqual(2, len(pipeline_call.call_args.kwargs["mentions"]))
        finished = self.a.get(self.url).json()
        self.assertEqual(2, finished["round"])
        self.assertEqual("waiting", finished["ai_status"])
        self.assertFalse(any(m["ready"] for m in finished["members"]))

    def test_ai_waits_for_every_human(self):
        self.create()
        self.a.post(self.url + "/ai", json={"profile": "axe"})
        self.b.post(self.url + "/join")
        self.role(self.a, "游侠")
        self.role(self.b, "猎人")
        self.a.put(self.url + "/action", json={"action": "用剑砍巨龙", "round": 1})
        self.assertEqual([], self.ai_pending)
        self.b.put(self.url + "/action", json={"action": "射箭牵制巨龙", "round": 1})
        self.assertEqual(1, len(self.ai_pending))
        room = self.run_ai()
        self.assertIn("游侠、猎人", coop_ai.bots(room)[0]["action"])
        self.assertEqual(403, self.b.post(self.url + "/ai/respond").status_code)

    def test_ai_discards_stale_response_and_follows_changed_action(self):
        self.ai_room()
        self.a.put(self.url + "/action", json={"action": "转身撤退，寻找出口", "round": 1})
        self.assertEqual(1, len(self.ai_pending))
        room = self.run_ai(0)
        self.assertFalse(room["members"][1]["ready"])
        self.assertEqual(2, len(self.ai_pending))
        room = self.run_ai(1)
        self.assertIn("出口", room["members"][1]["action"])
        self.assertNotIn("战斧劈", room["members"][1]["action"])

    def test_ai_withdrawal_and_removal_discard_inflight_response(self):
        self.ai_room()
        self.a.put(self.url + "/action", json={"action": "还没决定", "round": 1, "ready": False})
        room = self.run_ai()
        self.assertEqual("waiting", room["ai_status"])
        self.assertFalse(room["members"][1]["ready"])
        self.assertEqual("", room["members"][1]["action"])
        self.assertEqual(1, len(self.ai_pending))
        self.a.put(self.url + "/action", json={"action": "调查线索", "round": 1})
        bot_id = room["members"][1]["user_id"]
        self.a.delete(self.url + "/ai/" + bot_id)
        room = self.run_ai()
        self.assertEqual(1, len(room["members"]))
        self.assertEqual("waiting", room["ai_status"])

    def test_ai_restart_needs_explicit_retry(self):
        self.ai_room()
        self.ai_pending[0][-1].close()
        room = self.a.get(self.url).json()
        self.assertEqual("failed", room["ai_status"])
        self.assertEqual("用剑砍巨龙", room["members"][0]["action"])
        self.assertEqual(1, len(self.ai_pending))
        self.assertEqual(200, self.a.post(self.url + "/ai/respond").status_code)
        self.assertEqual(2, len(self.ai_pending))
        self.assertEqual(409, self.a.post(self.url + "/ai/respond").status_code)
        self.assertEqual("ready", self.run_ai()["ai_status"])

    def test_ai_llm_uses_real_human_context_and_explicit_model(self):
        with patch.dict(os.environ, {"COOP_AI_MODE": "llm", "COOP_AI_MODEL": "test-companion"}):
            room = self.ai_room()
            bot = room["members"][1]
            with patch.object(
                llm,
                "chat",
                return_value={
                    "actions": [{"user_id": bot["user_id"], "action": "举起战斧，配合游侠从巨龙侧翼进攻。"}]
                },
            ) as call:
                room = self.run_ai()
            call.assert_called_once()
            self.assertEqual("test-companion", call.call_args.kwargs["model"])
            self.assertIn("用剑砍巨龙", call.call_args.kwargs["user"])
            self.assertIn("共同对抗巨龙", call.call_args.kwargs["user"])
            self.assertEqual("llm", room["members"][1]["ai_source"])
            self.assertTrue(room["members"][1]["ready"])

    def test_ai_llm_failure_does_not_silently_invent_ready_action(self):
        with patch.dict(os.environ, {"COOP_AI_MODE": "llm"}):
            self.ai_room()
            with patch.object(
                llm,
                "chat",
                return_value={"actions": [{"user_id": self.ua["id"], "action": "不允许改真人动作"}]},
            ):
                room = self.run_ai()
            self.assertEqual("failed", room["ai_status"])
            self.assertFalse(room["members"][1]["ready"])
            self.assertEqual("用剑砍巨龙", room["members"][0]["action"])
            self.assertEqual(409, self.launch().status_code)

    def test_ai_never_becomes_host_and_last_human_closes_room(self):
        self.create()
        self.a.post(self.url + "/ai", json={"profile": "axe"})
        self.b.post(self.url + "/join")
        self.a.post(self.url + "/leave")
        room = self.b.get(self.url).json()
        self.assertEqual(self.ub["id"], room["host_id"])
        self.assertEqual(self.ub["id"], World.load(room["world_id"]).owner_id)
        self.b.post(self.url + "/leave")
        self.assertEqual(410, self.a.post(self.url + "/join").status_code)

    def test_ai_multiple_companions_keep_unique_references_and_actions(self):
        self.create()
        self.role(self.a, "游侠")
        for profile in coop_ai.PROFILES:
            self.a.post(self.url + "/ai", json={"profile": profile})
        self.a.put(self.url + "/action", json={"action": "用剑砍怪兽", "round": 1})
        room = self.run_ai()
        self.assertEqual(4, len(room["members"]))
        self.assertEqual(4, len({m["ref"] for m in room["members"]}))
        self.assertTrue(all(m["ready"] for m in room["members"]))
        actions = [m["action"] for m in coop_ai.bots(room)]
        for word in ("战斧", "圆盾", "法杖"):
            self.assertTrue(any(word in action for action in actions))

    def start_portrait(self, **kwargs):
        return self.a.post(self.url + "/portrait", json={"randomize": True, **kwargs})

    def run_portrait(self):
        with patch.object(coop_portrait, "draw", return_value=picture()):
            fn, *args = self.portrait_pending[-1]
            fn(*args)
        return self.a.get(self.url).json()

    def test_portrait_is_preview_first_and_binding_keeps_identity(self):
        self.create()
        original = self.role(self.a, "原游侠")["members"][0]
        self.a.put(self.url + "/action", json={"action": "举剑", "round": 1})
        r = self.start_portrait()
        self.assertEqual(202, r.status_code, r.text)
        room = self.run_portrait()
        member = room["members"][0]
        self.assertEqual(original["ref"], member["ref"])
        self.assertEqual("ready", member["portrait"]["status"])
        request_id = member["portrait"]["request_id"]
        room = self.a.post(self.url + "/portrait/use", json={"request_id": request_id}).json()
        member = room["members"][0]
        self.assertNotEqual(original["ref"], member["ref"])
        self.assertEqual(original["subject_id"], member["subject_id"])
        self.assertFalse(member["ready"])
        subject = World.load(room["world_id"]).subjects[member["subject_id"]]
        self.assertEqual(member["ref"], subject["ref"])
        self.assertEqual("", subject["definition"])
        self.assertEqual(
            409, self.a.post(self.url + "/portrait/use", json={"request_id": request_id}).status_code
        )

    def test_portrait_can_start_without_any_uploaded_reference(self):
        self.create()
        self.start_portrait()
        room = self.run_portrait()
        self.assertFalse(room["members"][0]["ref"])
        room = self.a.post(
            self.url + "/portrait/use", json={"request_id": room["members"][0]["portrait"]["request_id"]}
        ).json()
        self.assertTrue(room["members"][0]["ref"])
        self.assertTrue(room["members"][0]["subject_id"])
        self.assertEqual("applied", room["members"][0]["portrait"]["status"])

    def test_portrait_authorization_prevents_drawing_for_other_humans(self):
        self.create()
        self.b.post(self.url + "/join")
        self.assertEqual(401, self.anon.post(self.url + "/portrait", json={"randomize": True}).status_code)
        self.assertEqual(403, self.c.post(self.url + "/portrait", json={"randomize": True}).status_code)
        self.assertEqual(403, self.start_portrait(target_uid=self.ub["id"]).status_code)
        self.assertEqual(
            403,
            self.b.post(
                self.url + "/portrait", json={"randomize": True, "target_uid": self.ua["id"]}
            ).status_code,
        )
        self.assertEqual([], self.portrait_pending)

    def test_portrait_ai_appearance_retains_its_role_and_gear(self):
        self.create()
        room = self.a.post(self.url + "/ai", json={"profile": "axe"}).json()
        bot = room["members"][1]
        self.assertEqual(202, self.start_portrait(target_uid=bot["user_id"]).status_code)
        room = self.run_portrait()
        candidate = room["members"][1]["portrait"]
        self.assertEqual(bot["character"], candidate["name"])
        self.assertIn("战斧", candidate["description"])
        room = self.a.post(
            self.url + "/portrait/use",
            json={"request_id": candidate["request_id"], "target_uid": bot["user_id"]},
        ).json()
        self.assertEqual(bot["subject_id"], room["members"][1]["subject_id"])
        self.assertEqual("ai", room["members"][1]["kind"])
        self.assertEqual("axe", room["members"][1]["ai_profile"])

    def test_a_drawing_requested_while_another_runs_waits_instead_of_being_refused(self):
        # Before, the second request got a 409 "busy" error and had to be retried by hand.
        self.create()
        self.start_portrait()
        with coop_portrait._guard() as busy:
            # Another drawing is running. Non-blocking: a waiting request must not hold the
            # renderer (the old code held it from the request on, and refused the next one).
            fcntl.flock(busy, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.assertEqual(202, self.start_portrait().status_code)
            room = self.a.get(self.url).json()  # recovery runs here: the waiting request survives
            portrait = room["members"][0]["portrait"]
            self.assertEqual("generating", portrait["status"])
        self.assertEqual(2, len(self.portrait_pending))
        # The newer request supersedes the first; the stale one cannot be adopted.
        first, latest = self.portrait_pending
        with patch.object(coop_portrait, "draw", return_value=picture()):
            first[0](*first[1:])
            self.assertEqual(
                409, self.a.post(self.url + "/portrait/use", json={"request_id": "stale"}).status_code
            )
            latest[0](*latest[1:])
        room = self.a.get(self.url).json()
        self.assertEqual("ready", room["members"][0]["portrait"]["status"])
        self.assertEqual(portrait["request_id"], room["members"][0]["portrait"]["request_id"])
        self.assertFalse(coop_portrait.accepted(portrait["request_id"]))

    def test_portrait_failure_and_restart_keep_existing_reference(self):
        self.create()
        old = self.role(self.a, "原角色")["members"][0]["ref"]
        self.start_portrait()
        with patch.object(coop_portrait, "draw", side_effect=RuntimeError("local image unavailable")):
            fn, *args = self.portrait_pending[-1]
            fn(*args)
        room = self.a.get(self.url).json()
        self.assertEqual(old, room["members"][0]["ref"])
        self.assertEqual("failed", room["members"][0]["portrait"]["status"])
        self.start_portrait()
        coop_portrait._ACCEPTED.clear()  # the process stopped before the drawing ran
        room = self.a.get(self.url).json()
        self.assertEqual("failed", room["members"][0]["portrait"]["status"])
        self.assertEqual(old, room["members"][0]["ref"])

    def test_portrait_unconfigured_service_does_not_reserve_work(self):
        self.create()
        with (
            patch.object(config, "GEMINI_API_KEY", ""),
            patch.dict(os.environ, {"COOP_IMAGE_BASE": "", "COOP_IMAGE_SOCKET": ""}),
        ):
            self.assertFalse(self.a.get(self.url).json()["image_generation_enabled"])
            self.assertEqual(503, self.start_portrait().status_code)
        self.assertEqual([], self.portrait_pending)

    def test_portrait_custom_prompt_and_round_generation_freeze(self):
        room = self.ready()
        r = self.start_portrait(randomize=False, name="银发游侠", description="银发蓝眼，轻甲长剑")
        self.assertEqual(202, r.status_code)
        prompt = self.portrait_pending[-1][-1]["prompt"]
        self.assertIn("银发蓝眼", prompt)
        self.assertIn("纯白背景", prompt)
        room = self.run_portrait()
        request_id = room["members"][0]["portrait"]["request_id"]
        self.assertEqual(202, self.launch().status_code)
        self.assertEqual(409, self.start_portrait().status_code)
        self.assertEqual(
            409, self.a.post(self.url + "/portrait/use", json={"request_id": request_id}).status_code
        )


class SuggestionIdRecoveryTest(unittest.TestCase):
    """Recover a user_id the model truncated instead of failing the whole opening, but only when unambiguous.

    `coop_generation` already had this defense (`_normalize_action_ids`), `coop_intro` did not:
    suggestions have only `user_id` and `action`, no second anchor, so one wrong id raised
    "suggestions do not match the party" and failed the whole opening.
    """

    def setUp(self):
        self.members = [{"user_id": "u-abc123"}, {"user_id": "u-def456"}]

    def _ids(self, plan):
        return [s["user_id"] for s in plan["suggestions"]]

    def test_video_observation_ids_get_the_same_recovery(self):
        """user_ids in picture observations get miswritten too; recover them with the same rules.

        One day saw 21 "observed characters do not match the room" across 10 worlds, roughly every
        2–3 beats. Room ids look like `local-4120da325e7931ab00fa3e22`; one wrong character voided
        that beat's picture confirmation, and the world lost its "trust the picture when it
        disagrees with your beat" correction. Suggestions had recovery; observations did not.
        """
        members = [{"user_id": "u-default"}, {"user_id": "local-4120da325e7931ab00fa3e22"}]
        # truncated
        states = [{"user_id": "u-default", "state": "a"}, {"user_id": "local-4120da325e", "state": "b"}]
        guidance_checks._recover_entry_ids(states, members, event="observation_user_id_recovered")
        self.assertEqual(["u-default", "local-4120da325e7931ab00fa3e22"], [x["user_id"] for x in states])
        # exactly one unknown, exactly one missing
        states = [{"user_id": "u-default", "state": "a"}, {"user_id": "完全写错了", "state": "b"}]
        guidance_checks._recover_entry_ids(states, members, event="observation_user_id_recovered")
        self.assertEqual(["u-default", "local-4120da325e7931ab00fa3e22"], [x["user_id"] for x in states])
        # Two unknowns: do not guess; attaching state to the wrong person is worse than failing
        states = [{"user_id": "错1", "state": "a"}, {"user_id": "错2", "state": "b"}]
        guidance_checks._recover_entry_ids(states, members, event="observation_user_id_recovered")
        self.assertEqual(["错1", "错2"], [x["user_id"] for x in states])

    def test_a_character_name_used_as_user_id_is_claimed(self):
        """With only character names the model writes them into user_id; accept a unique mapping.

        The observation path used to send **only** character names (player_characters in
        `coop_assets`) while the schema required user_id, so the model could only write:

            "player_states": [{"user_id": "Character 1", ...}, {"user_id": "Character 2", ...}]

        and every beat was judged "observed characters do not match the room". The payload now
        includes user_id; this is the fallback, also for old caches. Names that are not unique in
        the room are not accepted, so state never lands on the wrong person.
        """
        members = [
            {"user_id": "u-default", "character": "角色1"},
            {"user_id": "local-4120da325e7931ab00fa3e22", "character": "角色2"},
        ]
        states = [{"user_id": "角色1", "state": "a"}, {"user_id": "角色2", "state": "b"}]
        guidance_checks._recover_entry_ids(states, members, event="observation_user_id_recovered")
        self.assertEqual(["u-default", "local-4120da325e7931ab00fa3e22"], [x["user_id"] for x in states])
        # duplicate names are not accepted
        dup = [{"user_id": "u1", "character": "同名"}, {"user_id": "u2", "character": "同名"}]
        states = [{"user_id": "同名", "state": "a"}, {"user_id": "同名", "state": "b"}]
        guidance_checks._recover_entry_ids(states, dup, event="observation_user_id_recovered")
        self.assertEqual(["同名", "同名"], [x["user_id"] for x in states])

    def test_a_missing_story_phase_does_not_discard_the_whole_observation(self):
        """Fields the schema does not require and the prompt calls optional must not be enforced by validation.

        The `coop_assets` call required only the boss fields, and the system prompt said observation
        and story_phase are optional. The model returned only observation, validation raised
        "observation lacks a valid story phase", and the whole picture confirmation was dropped.
        Same class of bug as requiring a user_id the payload never sent.

        If missing, leave it: the phase stays, and character state, environment and boss facts from
        the picture are written as usual. When a phase is given, "never go back to an earlier phase"
        still applies.
        """
        members = [{"user_id": "u1", "character": "甲"}, {"user_id": "u2", "character": "乙"}]
        base = {
            "event_outcome": "not_observed",
            "summary": "两人站在通道两侧。",
            "environment_state": "stable",
            "boss_visible": False,
            "boss_state": "",
            "player_states": [{"user_id": "u1", "state": "站立"}, {"user_id": "u2", "state": "站立"}],
        }
        ctx = {"story_context": {"visual_evidence_available": True}}
        # No story_phase: the observation must survive
        out = guidance_checks._validate_observation({"observation": dict(base)}, members, ctx)
        self.assertIsNotNone(out, "缺阶段不该整条丢掉")
        self.assertEqual("", out["story_phase"])
        self.assertEqual({"u1", "u2"}, {x["user_id"] for x in out["player_states"]})
        # A valid phase: carried through as usual
        phase = list(guidance_checks._STORY_PHASES)[0]
        out = guidance_checks._validate_observation(
            {"observation": dict(base), "story_phase": phase}, members, ctx
        )
        self.assertEqual(phase, out["story_phase"])

    def test_enum_style_boss_states_are_not_read_as_present(self):
        """`not_present` literally means absent; it must not be read as "stated present".

        `_BOSS_STATE_NEGATIVE` was written for Chinese prose and missed the enum-like phrases models
        actually output:

            boss_visible=False  boss_state='not_present'  → whole observation rejected
            boss_visible=False  boss_state='no_change'    → whole observation rejected

        Real presence claims ("looming over the gold") must still count as present, or the check is
        pointless: it blocks writing the boss into persistent world facts when it is not in the
        picture.
        """
        for value in (
            "not_present",
            "no_change",
            "none",
            "未出现",
            "absent",
            "unknown",
            "N/A",
            "not visible",
        ):
            with self.subTest(value=value):
                self.assertFalse(guidance_checks._positive_boss_state(value))
        for value in ("looming over the gold pile", "present and attacking", "巨龙盘踞在金币堆上"):
            with self.subTest(value=value):
                self.assertTrue(guidance_checks._positive_boss_state(value))

    def test_a_neutral_state_is_valid_while_the_boss_is_visible(self):
        """Neutral ≠ negative: the boss is in the picture and unchanged this beat, a normal observation.

        The previous version counted no_change / unknown as invalid, so `boss_visible=True,
        boss_state='no_change'` was rejected right after shipping. Only an **explicit denial** of
        presence contradicts boss_visible=True.
        """
        for value in ("no_change", "unknown", "n/a", "没有变化"):
            with self.subTest(value=value):
                self.assertFalse(guidance_checks._denies_boss_presence(value), "中性取值不是否认")
        for value in ("not_present", "absent", "not visible", "未出现", "none"):
            with self.subTest(value=value):
                self.assertTrue(guidance_checks._denies_boss_presence(value))

        members = [{"user_id": "u1", "character": "甲"}, {"user_id": "u2", "character": "乙"}]
        ctx = {"story_context": {"visual_evidence_available": True}}
        base = {
            "event_outcome": "not_observed",
            "summary": "巨龙仍盘踞。",
            "environment_state": "stable",
            "boss_visible": True,
            "player_states": [{"user_id": "u1", "state": "站立"}, {"user_id": "u2", "state": "站立"}],
        }
        out = guidance_checks._validate_observation(
            {"observation": dict(base, boss_state="no_change")}, members, ctx
        )
        self.assertIsNotNone(out, "BOSS 可见 + 中性状态是合法观察")
        # Saw the boss, but the state says it is absent: self-contradictory, still rejected
        with self.assertRaises(ValueError):
            guidance_checks._validate_observation(
                {"observation": dict(base, boss_state="not_present")}, members, ctx
            )

    def test_one_missing_player_does_not_discard_the_whole_observation(self):
        """One unreported person only loses that person; environment and boss facts from the picture are kept.

        In two-player rooms the model sometimes reported only one person (always missing the long
        `local-…` id), and the whole observation was rejected for "missing character state", losing
        all picture feedback for the beat. Reporting nobody is what really means nothing was read.
        """
        members = [{"user_id": "u1", "character": "甲"}, {"user_id": "u2", "character": "乙"}]
        base = {
            "event_outcome": "not_observed",
            "summary": "甲站在通道左侧。",
            "environment_state": "stable",
            "boss_visible": False,
            "boss_state": "",
            "player_states": [{"user_id": "u1", "state": "站立"}],
        }
        ctx = {"story_context": {"visual_evidence_available": True}}
        out = guidance_checks._validate_observation({"observation": dict(base)}, members, ctx)
        self.assertIsNotNone(out, "少报一个人不该整条丢掉")
        self.assertEqual(["u1"], [x["user_id"] for x in out["player_states"]])
        self.assertEqual("stable", out["environment_state"])
        # nobody reported -> still rejected
        empty = dict(base, player_states=[])
        with self.assertRaises(ValueError):
            guidance_checks._validate_observation({"observation": empty}, members, ctx)

    def test_publishing_a_partial_observation_keeps_the_other_player_state(self):
        """Unreported people keep their previous state; it must not be cleared."""
        room = {
            "story_state": {
                "players": {"u1": "旧甲", "u2": "旧乙"},
                "phase": "",
                "environment": "",
                "boss": "",
                "persistent_history": [],
                "environment_history": [],
                "boss_history": [],
            }
        }
        obs = {
            "event_outcome": "not_observed",
            "summary": "甲往前走。",
            "environment_state": "stable",
            "boss_visible": False,
            "boss_state": "",
            "player_states": [{"user_id": "u1", "state": "新甲"}],
            "story_phase": "",
        }
        coop_intro.publish_observation(room, obs, "n1")
        self.assertEqual({"u1": "新甲", "u2": "旧乙"}, room["story_state"]["players"])

    def test_boss_detection_payload_carries_the_user_id(self):
        """Member info sent for picture observation must include user_id, or the model cannot fill it in."""
        import inspect
        from murea_agent.coop import assets as coop_assets

        src = inspect.getsource(coop_assets)
        src = src.replace("'", '"')
        i = src.index('"player_characters"')
        block = src[i : i + 260]
        self.assertIn('"user_id"', block, "only character names: the model cannot fill in real ids")
        self.assertIn('"character"', block)

    def test_a_truncated_id_is_restored(self):
        plan = {"suggestions": [{"user_id": "u-abc", "action": "a"}, {"user_id": "u-def456", "action": "b"}]}
        guidance_checks._recover_suggestion_ids(plan, self.members)
        self.assertEqual(["u-abc123", "u-def456"], self._ids(plan))

    def test_one_unknown_against_one_missing_slot_is_unambiguous(self):
        plan = {
            "suggestions": [{"user_id": "garbage", "action": "a"}, {"user_id": "u-def456", "action": "b"}]
        }
        guidance_checks._recover_suggestion_ids(plan, self.members)
        self.assertEqual(["u-abc123", "u-def456"], self._ids(plan))

    def test_two_unknowns_are_never_guessed(self):
        """Guessing wrong attaches a suggestion to someone else: worse than failing; better let validation stop it."""
        plan = {"suggestions": [{"user_id": "x", "action": "a"}, {"user_id": "y", "action": "b"}]}
        guidance_checks._recover_suggestion_ids(plan, self.members)
        self.assertEqual(["x", "y"], self._ids(plan))
        with self.assertRaisesRegex(ValueError, "suggestions do not match the party"):
            guidance_checks._validate(plan, self.members)

    def test_a_prefix_matching_two_free_members_is_not_guessed(self):
        """With neither claimed yet the prefix matches two: ambiguous, so not accepted.

        (If one was already claimed by another suggestion, only one candidate remains and accepting
        it is right: that is what "only when unique" means, not "never accept prefixes".)
        """
        members = [{"user_id": "u-a1"}, {"user_id": "u-a2"}]
        plan = {"suggestions": [{"user_id": "u-a", "action": "a"}, {"user_id": "u-zzz", "action": "b"}]}
        guidance_checks._recover_suggestion_ids(plan, members)
        self.assertEqual("u-a", plan["suggestions"][0]["user_id"], "前缀撞两个就不该认")

    def test_a_prefix_becomes_unambiguous_once_the_other_slot_is_taken(self):
        members = [{"user_id": "u-a1"}, {"user_id": "u-a2"}]
        plan = {"suggestions": [{"user_id": "u-a", "action": "a"}, {"user_id": "u-a2", "action": "b"}]}
        guidance_checks._recover_suggestion_ids(plan, members)
        self.assertEqual(["u-a1", "u-a2"], self._ids(plan))

    def test_correct_ids_are_left_alone(self):
        plan = {
            "suggestions": [{"user_id": "u-abc123", "action": "a"}, {"user_id": "u-def456", "action": "b"}]
        }
        guidance_checks._recover_suggestion_ids(plan, self.members)
        self.assertEqual(["u-abc123", "u-def456"], self._ids(plan))


class WaitingNarrationTest(unittest.TestCase):
    """Chinese waiting narration: no repeated names or connectives."""

    def test_names_and_connectives_are_not_doubled(self):
        from murea_agent.coop.action_text import _generation_narration

        members = [
            {"character": "沈舟 · 雾港侦探", "action": "沈舟走向船首"},
            {"character": "洛宁 · 航海家", "action": "挥手示意"},
        ]
        for event in ("与此同时，船体一晃", "接下来，船体一晃", "船体一晃"):
            with self.subTest(event=event):
                self.assertEqual(
                    "本轮行动：沈舟走向船首；洛宁挥手示意。与此同时，船体一晃。",
                    _generation_narration(members, event),
                )

    def test_a_long_action_is_cut_at_a_clause_not_mid_phrase(self):
        # Real round: "...pulls open three drawers one after." was read aloud and kept as text.
        from murea_agent.coop.action_text import _generation_narration

        action = (
            "Courier shoves the loose plank fully aside, ducks through the gap into the dark shopfront, "
            "sweeps a flashlight across the dusty counter, pulls open three drawers one after another, "
            "finds a rusted tin box under the floorboards, pries it open with a knife"
        )
        members = [
            {"character": "Courier", "action": action},
            {"character": "Mara", "action": "Mara keeps watch."},
        ]
        with patch.object(config, "LANGUAGE", "en"):
            narration = _generation_narration(members, "")
        self.assertTrue(
            narration.startswith(
                "Courier shoves the loose plank fully aside, ducks through the gap into the dark shopfront, "
                "sweeps a flashlight across the dusty counter. Mara keeps watch."
            ),
            narration,
        )
        self.assertNotIn("one after", narration)
        long_zh = "，".join(f"信使在第{i}个抽屉里翻找" for i in range(1, 12))
        with patch.object(config, "LANGUAGE", "zh"):
            narration = _generation_narration([{"character": "信使", "action": long_zh}], "")
        self.assertTrue(narration.endswith("翻找。"), narration)


if __name__ == "__main__":
    unittest.main()
