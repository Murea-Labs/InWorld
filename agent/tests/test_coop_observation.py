"""Independent story memory: durable jobs, fact boundaries and playback races."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from murea_agent import config
from murea_agent.coop import guidance_checks
from murea_agent.coop import (
    room as coop,
    assets as coop_assets,
    intro as coop_intro,
    observation as coop_observation,
    prefetch as coop_prefetch,
    story as coop_story,
)


class CoopObservationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.jobs = []
        self.members = [
            {
                "user_id": f"u{i}",
                "name": f"玩家{i}",
                "subject_id": f"s{i}",
                "character": f"角色{i}",
                "description": "轻甲",
                "ref": f"{i}.jpg",
                "action": "向桥头跑",
                "ready": False,
            }
            for i in range(1, 3)
        ]
        self.forecast = {
            "world_event": "接下来巨龙将击碎桥面。",
            "world_event_type": "boss_action",
            "world_event_key": "bridge_breaks",
            "objective": "两人共同撤离石桥",
            "boss_candidate": {"name": "", "description": ""},
        }
        self.node = {"id": "n1", "frames": ["frame.jpg"]}
        self.room = {
            "code": "ABCDEF123456",
            "world_id": "w-observation",
            "round": 2,
            "revision": 1,
            "state_revision": 3,
            "status": "waiting",
            "capacity": 4,
            "setting": "古桥冒险",
            "visual_style": "cinematic",
            "scene_name": "石桥",
            "scene_description": "桥下有河流",
            "scene_ref": "scene.jpg",
            "members": self.members,
            "story_state": {"phase": "confrontation", "players": {}},
            "rounds": [
                {
                    "round": 1,
                    "node": self.node,
                    "inputs": copy.deepcopy(self.members),
                    "guidance": self.forecast,
                }
            ],
            "intro": {
                "status": "ready",
                "source": "history_actions",
                "world_event": "接下来碎石将滑入河水。",
                "world_event_type": "environment",
                "world_event_key": "rubble_falls",
                "objective": "寻找安全落脚点",
                "suggestions": [{"user_id": m["user_id"], "action": "退到护栏外侧"} for m in self.members],
            },
        }
        self.room["intro"]["signature"] = coop_intro.signature(self.room)
        for mock in (
            patch.object(config, "DATA", Path(self.tmp.name)),
            patch.object(coop_story, "queue"),
            patch.object(coop_observation._POOL, "submit", side_effect=lambda *args: self.jobs.append(args)),
            patch.object(
                coop_observation.coop_generation, "frame_images", return_value=[("image/jpeg", b"frame")]
            ),
        ):
            mock.start()
            self.addCleanup(mock.stop)
        self.addCleanup(lambda: [job[-1].close() for job in self.jobs])
        self.save()

    def save(self):
        coop._write(coop._directory() / f"{self.room['code']}.json", self.room)

    def read(self):
        with coop._edit(self.room["code"], persist=False) as room:
            return room

    def enqueue(self, *, node=None, round_number=1, opening=False):
        with coop._edit(self.room["code"]) as room:
            snapshot = copy.deepcopy(room)
            snapshot.update(round=round_number, intro=copy.deepcopy(self.forecast), opening=opening)
            coop_observation.enqueue(room, snapshot, node or self.node)

    def run_job(self, response=None, side_effect=None):
        fn, code, guard = self.jobs.pop(0)
        with patch.object(
            coop_observation.coop_trace,
            "chat",
            return_value=response or self.observation(),
            side_effect=side_effect,
        ) as chat:
            fn(code, guard)
        return chat

    def observation(self, *, summary="石桥中央已断裂", players=None, boss_visible=True):
        return {
            "observation": {
                "event_outcome": "observed",
                "summary": summary,
                "environment_state": summary,
                "boss_visible": boss_visible,
                "boss_state": "巨龙站在桥头" if boss_visible else "",
                "player_states": (
                    [{"user_id": m["user_id"], "state": "站在护栏旁"} for m in self.members]
                    if players is None
                    else players
                ),
            },
            "story_phase": "escalation",
        }

    def test_existing_boss_four_players_and_disabled_image_backend_all_observe(self):
        for mode in ("existing_boss", "four_players", "no_image_backend", "boss_preview"):
            with self.subTest(mode=mode):
                self.room.pop("_story_observation_pending", None)
                self.room.pop("story_observation", None)
                self.room["story_state"] = {"phase": "confrontation", "players": {}}
                self.room["rounds"][0].pop("observation", None)
                self.room["members"] = copy.deepcopy(self.members)
                self.room["monster"] = {"name": "岩龙", "description": "灰色鳞片", "ref": "boss.jpg"}
                self.room["asset_previews"] = {}
                if mode == "four_players":
                    self.room.pop("monster")
                    self.room["members"] += [
                        dict(m, user_id=f"u{i}", subject_id=f"s{i}") for i, m in enumerate(self.members, 3)
                    ]
                elif mode == "boss_preview":
                    self.room.pop("monster")
                    self.room["asset_previews"] = {"monster": {"status": "generating"}}
                self.room["rounds"][0]["inputs"] = copy.deepcopy(self.room["members"])
                self.room["intro"]["signature"] = coop_intro.signature(self.room)
                self.save()
                with patch.object(
                    coop_assets.coop_portrait, "enabled", return_value=mode != "no_image_backend"
                ):
                    self.assertFalse(coop_assets.queue_round_boss_extraction(self.room["code"]))
                    self.enqueue()
                    self.assertTrue(coop_observation.queue(self.room["code"]))
                    response = self.observation(
                        players=[{"user_id": m["user_id"], "state": "站在桥侧"} for m in self.room["members"]]
                    )
                    chat = self.run_job(response)
                room = self.read()
                self.assertEqual("ready", room["story_observation"]["status"])
                self.assertEqual("n1", room["story_state"]["last_node_id"])
                self.assertEqual(len(room["members"]), len(room["story_state"]["players"]))
                self.assertEqual("observed", room["rounds"][0]["observation"]["event_outcome"])
                sent = json.loads(chat.call_args.kwargs["user"])
                self.assertEqual(self.forecast, sent["current_forecast"])
                self.assertEqual("向桥头跑", sent["requested_actions"][0]["action"])
                self.assertIn("observation", chat.call_args.kwargs["schema"]["required"])
                self.assertNotIn("suggestions", chat.call_args.kwargs["schema"]["properties"])

    def test_observation_keeps_guidance_prefetch_and_playback_state(self):
        self.room["recommendation_prefetch"] = {"status": "rendering", "job_id": "video-job"}
        self.room["progress"] = "正在生成下一段视频"
        self.room["status"] = "generating"
        self.save()
        before = self.read()
        self.enqueue()
        coop_observation.queue(self.room["code"])
        self.run_job()
        room = self.read()
        self.assertEqual(before["intro"], room["intro"])
        self.assertEqual(before["recommendation_prefetch"], room["recommendation_prefetch"])
        self.assertEqual(before["progress"], room["progress"])
        self.assertEqual("generating", room["status"])
        self.assertEqual(coop_intro.signature(before), coop_intro.signature(room))
        self.assertEqual(coop_prefetch.state_signature(before), coop_prefetch.state_signature(room))
        payload, _ = coop_intro.guidance_payload(room, self.node, include_images=False, history_only=True)
        self.assertEqual("石桥中央已断裂", payload["story_context"]["current_state"]["environment"])
        self.assertEqual("n1", payload["story_context"]["persistent_history"][0]["node_id"])
        self.assertNotIn("_story_observation_pending", coop._public(room))

    def test_concurrent_reads_do_not_duplicate_and_restart_resumes_pending_job(self):
        self.enqueue()
        self.enqueue()
        self.assertEqual(1, len(self.read()["_story_observation_pending"]))
        coop_observation.queue(self.room["code"])
        coop_observation.queue(self.room["code"])
        self.assertEqual(1, len(self.jobs))
        self.jobs.pop()[2].close()  # The process stopped before the worker ran.
        coop_observation.queue(self.room["code"])
        self.run_job()
        self.enqueue()
        self.assertFalse(coop_observation.queue(self.room["code"]))
        self.assertEqual(1, len(self.read()["story_state"]["persistent_history"]))

    def test_commit_and_read_recover_the_observation_handoff(self):
        with coop._edit(self.room["code"]) as room:
            snapshot = copy.deepcopy(room)
            snapshot["intro"] = self.forecast
            node = {"id": "n2", "frames": ["frame2.jpg"]}
            coop._finish(room, snapshot, node)
        self.assertEqual("n2", self.read()["_story_observation_pending"][0]["node"]["id"])
        with (
            patch.object(coop_assets, "queue_round_boss_extraction", return_value=False),
            patch.object(coop_intro, "queue"),
        ):
            coop.get_room(self.room["code"], "u1", prepare_intro=False)
        self.assertEqual(1, len(self.jobs))
        self.run_job()
        self.assertEqual("n2", self.read()["story_state"]["last_node_id"])

    def test_post_video_callback_queues_observation_when_boss_check_skips(self):
        self.enqueue()
        with (
            patch.object(coop_assets, "queue_round_boss_extraction", return_value=False),
            patch.object(coop_intro, "queue"),
            patch.object(coop_intro, "retry_failed"),
            patch.object(coop_prefetch, "queue_recommendation"),
        ):
            coop._after_round(self.room["code"], {"state_revision": 2})
        self.assertEqual(1, len(self.jobs))
        self.run_job()

    def test_opening_is_observed_without_a_requested_action_or_forecast(self):
        self.room["rounds"] = []
        self.room["opening_node"] = self.node
        self.save()
        self.enqueue(opening=True)
        coop_observation.queue(self.room["code"])
        chat = self.run_job()
        payload = json.loads(chat.call_args.kwargs["user"])
        self.assertEqual({}, payload["current_forecast"])
        self.assertEqual([], payload["requested_actions"])
        record = self.read()["story_state"]["persistent_history"][0]
        self.assertEqual(0, record["round"])
        self.assertEqual("unknown", record["observation"]["event_outcome"])

    def test_no_visible_players_preserves_previous_player_and_boss_facts(self):
        self.room["story_state"].update(
            players={"u1": "左侧护栏", "u2": "右侧护栏"}, boss="岩龙", boss_known=True
        )
        self.save()
        self.enqueue()
        coop_observation.queue(self.room["code"])
        self.run_job(self.observation(players=[], boss_visible=False))
        state = self.read()["story_state"]
        self.assertEqual(self.room["story_state"]["players"], state["players"])
        self.assertEqual("岩龙", state["boss"])
        self.assertEqual("石桥中央已断裂", state["environment"])

    def test_slow_observations_are_processed_in_order_with_previous_facts(self):
        self.enqueue()
        coop_observation.queue(self.room["code"])
        payloads = []

        def respond(**kwargs):
            payloads.append(json.loads(kwargs["user"]))
            if len(payloads) == 1:
                node = {"id": "n2", "frames": ["frame2.jpg"]}
                with coop._edit(self.room["code"]) as room:
                    room["rounds"].append(
                        {"round": 2, "node": node, "inputs": self.members, "guidance": self.forecast}
                    )
                    room["round"] = 3
                self.enqueue(node=node, round_number=2)
                coop_observation.queue(self.room["code"])
                return self.observation(summary="石桥中央已断裂")
            return self.observation(summary="桥下河水淹没碎石")

        self.run_job(side_effect=respond)
        self.assertEqual(2, len(payloads))
        self.assertEqual("石桥中央已断裂", payloads[1]["story_context"]["current_state"]["environment"])
        state = self.read()["story_state"]
        self.assertEqual(["n1", "n2"], [item["node_id"] for item in state["persistent_history"]])
        self.assertEqual("n2", state["last_node_id"])
        self.assertEqual([], self.jobs)

    def test_late_observation_after_manual_edit_only_records_historical_evidence(self):
        for edit in ("asset", "character"):
            with self.subTest(edit=edit):
                self.save()
                self.enqueue()
                coop_observation.queue(self.room["code"])
                with coop._edit(self.room["code"]) as room:
                    if edit == "asset":
                        room["asset_edit_revision"] = 1
                    else:
                        room["members"][0]["description"] = "新的角色设定"
                    room["story_state"].update(boss="玩家新选择的冰龙", environment="新的冰原")
                self.run_job()
                state = self.read()["story_state"]
                self.assertEqual("玩家新选择的冰龙", state["boss"])
                self.assertEqual("新的冰原", state["environment"])
                self.assertEqual(
                    "石桥中央已断裂", state["persistent_history"][0]["observation"]["environment_state"]
                )

    def test_older_observation_never_rewinds_latest_state_or_adds_duplicate_history(self):
        room = copy.deepcopy(self.room)
        room["rounds"].append({"round": 2, "node": {"id": "n2"}, "inputs": self.members})
        newer = guidance_checks._validate_observation(
            self.observation(summary="最新状态"),
            self.members,
            {"story_context": {"visual_evidence_available": True}},
        )
        older = dict(newer, summary="旧状态", environment_state="旧状态", story_phase="opening")
        coop_intro.publish_observation(room, newer, "n2")
        coop_intro.publish_observation(room, older, "n1")
        coop_intro.publish_observation(room, newer, "n2")
        self.assertEqual("n2", room["story_state"]["last_node_id"])
        self.assertEqual("最新状态", room["story_state"]["environment"])
        self.assertEqual("escalation", room["story_state"]["phase"])
        self.assertEqual(2, len(room["story_state"]["persistent_history"]))
        self.assertEqual(
            ["n1", "n2"], [item["node_id"] for item in room["story_state"]["persistent_history"]]
        )

    def test_missing_frames_and_failed_observer_never_publish_plans_as_facts(self):
        for missing_frames in (True, False):
            with self.subTest(missing_frames=missing_frames):
                self.save()
                before = self.read()["intro"]
                self.enqueue()
                coop_observation.queue(self.room["code"])
                with patch.object(
                    coop_observation.coop_generation,
                    "frame_images",
                    return_value=[] if missing_frames else [("image/jpeg", b"frame")],
                ):
                    chat = self.run_job(side_effect=RuntimeError("observer unavailable"))
                room = self.read()
                self.assertEqual(
                    "skipped" if missing_frames else "failed", room["story_observation"]["status"]
                )
                self.assertEqual(0 if missing_frames else 2, chat.call_count)
                self.assertEqual(before, room["intro"])
                self.assertEqual(self.room["story_state"], room["story_state"])
                self.assertFalse(coop_observation.queue(self.room["code"]))

    def test_missing_observation_retries_and_schema_failure_does_not_stall_queue(self):
        self.enqueue()
        coop_observation.queue(self.room["code"])
        chat = self.run_job(side_effect=[{}, self.observation()])
        self.assertEqual(2, chat.call_count)
        self.assertEqual("ready", self.read()["story_observation"]["status"])

    def test_executor_failure_preserves_job_for_later_retry(self):
        self.enqueue()
        with patch.object(coop_observation._POOL, "submit", side_effect=RuntimeError("busy")):
            self.assertFalse(coop_observation.queue(self.room["code"]))
        self.assertEqual("queued", self.read()["story_observation"]["status"])
        self.assertTrue(coop_observation.queue(self.room["code"]))
        self.run_job()
        self.assertEqual("ready", self.read()["story_observation"]["status"])


if __name__ == "__main__":
    unittest.main()
