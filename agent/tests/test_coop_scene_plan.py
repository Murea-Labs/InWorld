"""Contracts for the co-op fact boundary and history-only forecast scheduler."""

from concurrent import futures
import copy
import json
import os
from pathlib import Path
import tempfile
import dataclasses
import unittest
from types import MappingProxyType
from unittest.mock import patch

from murea_agent import config
from murea_agent.coop import guidance_checks
from murea_agent.coop import (
    room as coop,
    generation as coop_generation,
    intro as coop_intro,
    prefetch as coop_prefetch,
)
from murea_agent.coop import scene_plan as plan_mod
from murea_agent.coop.scene_plan import AssetResolver, ScenePlan, ShotCompiler, WorldSnapshot


def immediate(fn, *args):
    future = futures.Future()
    try:
        future.set_result(fn(*args))
    except BaseException as exc:
        future.set_exception(exc)
    return future


def reactions_for(plan):
    return {
        "world_event_key": plan.get("world_event_key", ""),
        "suggestions": [
            {
                **item,
                "forecast_cue": plan.get("world_event") or plan["objective"],
                "response_reason": "按照预告调整路线，协作通过危险区域。",
            }
            for item in plan["suggestions"]
        ],
    }


class CoopScenePlanTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.patches = [patch.object(config, "DATA", self.root)]
        for item in self.patches:
            item.start()
        self.code = "ABCDEF123456"
        self.members = [
            {
                "user_id": "u1",
                "name": "player A",
                "subject_id": "s1",
                "character": "A",
                "description": "green armor",
                "ref": "a.jpg",
                "action": "向左跑",
            },
            {
                "user_id": "u2",
                "name": "player B",
                "subject_id": "s2",
                "character": "B",
                "description": "brown coat",
                "ref": "b.jpg",
                "action": "举盾前进",
            },
        ]
        self.room = {
            "code": self.code,
            "world_id": "w-test",
            "revision": 1,
            "capacity": 2,
            "mode": "local",
            "host_id": "u1",
            "title": "test",
            "state_revision": 7,
            "round": 2,
            "status": "generating",
            "setting": "古桥冒险",
            "visual_style": "cinematic",
            "scene_name": "石桥",
            "scene_description": "古老石桥",
            "scene_ref": "scene.jpg",
            "members": self.members,
            "rounds": [],
            "story_state": {"phase": "exploration", "persistent_history": []},
            "intro": {
                "status": "ready",
                "world_event": "",
                "world_event_type": "none",
                "world_event_key": "",
                "boss_candidate": {"name": "", "description": ""},
            },
            "inflight": {"request_id": "turn-2"},
        }
        coop._write(coop._directory() / f"{self.code}.json", self.room)

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()
        self.tmp.cleanup()

    def test_snapshot_is_deeply_immutable_and_assets_have_stable_order(self):
        snapshot = WorldSnapshot.from_room(self.room, node_id="n1")
        assets = AssetResolver.resolve(self.members, self.room)
        self.assertEqual(["s1", "s2", "scene:石桥"], [asset.asset_id for asset in assets])
        self.room["members"][0]["character"] = "changed"
        self.room["story_state"]["phase"] = "changed"
        self.assertEqual("A", snapshot.players[0]["character"])
        self.assertEqual("exploration", snapshot.story_state["phase"])
        with self.assertRaises(TypeError):
            snapshot.players[0]["character"] = "mutate"

    def test_snapshot_serialization_removes_nested_mapping_proxies_before_node_commit(self):
        self.room["story_state"] = {
            "phase": "threat",
            "players": {"u1": {"position": "left"}, "u2": {"position": "right"}},
            "persistent_history": [
                {
                    "node_id": "n1",
                    "observation": {"environment_state": "石桥中央已经断裂。"},
                }
            ],
        }
        value = WorldSnapshot.from_room(self.room, node_id="n1").as_dict()
        # Node persistence uses dataclasses.asdict(), whose recursive deepcopy
        # reproduces the original failure if any nested mappingproxy remains.
        copied = copy.deepcopy(value)
        self.assertEqual("left", copied["story_state"]["players"]["u1"]["position"])
        self.assertEqual(
            "石桥中央已经断裂。",
            copied["story_state"]["persistent_history"][0]["observation"]["environment_state"],
        )
        json.dumps(copied, ensure_ascii=False)

    def test_compiler_uses_each_action_once_and_ignores_model_timeline(self):
        raw = {
            "scene": "An old stone bridge.",
            "sound": "Wind over stone.",
            "timeline": "MODEL MAY REASSIGN OR DROP AN ACTION",
            "actions": [
                {
                    "user_id": "u1",
                    "source_action": "向左跑",
                    "appearance": "green armor",
                    "action": "runs left along the parapet",
                    "beat": "runs left",
                },
                {
                    "user_id": "u2",
                    "source_action": "举盾前进",
                    "appearance": "brown coat",
                    "action": "advances behind a raised shield",
                    "beat": "advances",
                },
            ],
        }
        plan = ScenePlan.from_legacy(
            raw, self.members, base_revision=7, duration=10, submission_order=("u2", "u1")
        )
        timeline = ShotCompiler.timeline(plan)
        self.assertNotIn("MODEL MAY", timeline)
        self.assertEqual(1, timeline.count("runs left along the parapet"))
        self.assertEqual(1, timeline.count("advances behind a raised shield"))
        caption = ShotCompiler.timeline(plan, caption=True)
        self.assertIn("<Subject 1> runs left along the parapet.", caption)
        self.assertNotIn("0.00s", caption)
        self.assertEqual(("u1", "u2"), plan.subject_order)
        self.assertEqual(("u2", "u1"), plan.submission_order)
        self.assertIn("subject_order", plan.as_dict())

    def test_caption_keeps_metadata_internal(self):
        """The video model's text describes only the visible performance; state and route fields stay internal."""
        plan = ScenePlan.from_legacy(
            {
                "actions": [
                    {
                        "user_id": "u1",
                        "action": "runs left",
                        "path": "an arc around the creature",
                        "target": "the creature",
                        "end_state": "its rear flank",
                    },
                ]
            },
            self.members,
            duration=10,
        )
        caption = ShotCompiler.compile_timeline(
            plan,
            camera="A continuous wide shot follows the players.",
            environment="The action unfolds inside <Subject 3>.",
            style=" Visual style: cinematic.",
        )
        self.assertTrue(caption.startswith("[Shot 1]"))
        self.assertIn("<Subject 1> runs left.", caption)
        self.assertNotIn("As the movement unfolds,", caption)
        self.assertNotIn("state is", caption)
        self.assertNotIn("its rear flank", caption)
        self.assertEqual("its rear flank", plan.actions[0].end_state)
        self.assertNotRegex(caption, r"\d+\.\d+s")

    def test_submission_order_never_changes_subject_numbers(self):
        raw = {
            "actions": [
                {"user_id": "u2", "action": "raises the shield"},
                {"user_id": "u1", "action": "draws the bow"},
            ]
        }
        plan = ScenePlan.from_legacy(raw, self.members, submission_order=("u2", "u1"))
        timeline = ShotCompiler.timeline(plan)
        self.assertLess(timeline.index("<Subject 2> raises"), timeline.index("<Subject 1> draws"))

    def test_compiler_emits_timed_path_and_end_state_for_each_actor(self):
        plan = ScenePlan.from_legacy(
            {
                "actions": [
                    {
                        "user_id": "u1",
                        "action": "runs left",
                        "path": "an arc around the creature",
                        "target": "the creature",
                        "end_state": "its rear flank",
                    },
                ]
            },
            self.members,
            duration=10,
        )
        timeline = ShotCompiler.timeline(plan)
        self.assertIn("From 0.00s to 10.00s", timeline)
        self.assertIn("By 10.00s", timeline)
        self.assertIn("follows an arc around the creature", timeline)
        self.assertIn("relative to the creature", timeline)
        self.assertIn("final state is: its rear flank", timeline)
        self.assertNotIn("holds that spatial relationship", timeline)

    def test_joint_timing_survives_compilation_without_simultaneous_prefix(self):
        action = "passes along the creature's right flank after it turns toward the archer"
        event = "Before the runner passes, the dragon turns left; afterward it remains behind the runner"
        plan = ScenePlan.from_legacy(
            {
                "actions": [
                    {"user_id": "u1", "action": action, "path": ""},
                ],
                "event_action": event,
            },
            self.members,
            duration=8,
        )
        caption = ShotCompiler.timeline(plan, caption=True)
        self.assertIn(action, caption)
        self.assertIn(event, caption)
        self.assertNotIn("At the same time", caption)
        self.assertNotIn("Meanwhile", caption)
        self.assertNotIn("As the movement unfolds", caption)

    def test_history_forecast_never_reads_frames_and_stays_private_until_commit(self):
        snapshot = json.loads(json.dumps(self.room))
        snapshot["request_id"] = "turn-2"
        snapshot["intro"].update(
            world_event="接下来岩龙将从桥底探出头部。",
            world_event_type="boss_introduction",
            world_event_key="dragon_emerges",
            objective="两人共同寻找安全通路",
            boss_candidate={"name": "岩龙", "description": "灰色岩甲和金色眼睛"},
        )
        plan = {
            "narration": "根据既有局势与本轮请求，下一轮可继续协作推进。",
            "world_event": "下一刻桥侧将落下碎石，迫使队伍调整路线。",
            "world_event_type": "environment",
            "world_event_key": "falling_rocks",
            "boss_candidate": {"name": "", "description": ""},
            "objective": "穿过石桥抵达对岸",
            "suggestions": [
                {"user_id": "u1", "action": "沿左侧护栏快速前进并观察落石间隙"},
                {"user_id": "u2", "action": "举盾贴近队友，为两人挡住桥侧碎石"},
            ],
        }
        with (
            patch.dict(os.environ, {"COOP_INTRO_ENABLED": "1"}),
            patch.object(coop_intro._POOL, "submit", side_effect=immediate),
            patch.object(
                coop_generation, "frame_images", side_effect=AssertionError("history forecast read frames")
            ),
            patch.object(coop_intro.coop_trace, "chat", side_effect=[plan, reactions_for(plan)]) as chat,
        ):
            self.assertTrue(coop_intro.queue_history_guidance(snapshot))
        self.assertEqual(2, chat.call_count)
        self.assertNotIn("images", chat.call_args.kwargs)
        sent = json.loads(chat.call_args.kwargs["user"])
        self.assertNotIn("latest_video", sent)
        self.assertNotIn("video_summary", json.dumps(sent, ensure_ascii=False))
        self.assertNotIn("prompt", json.dumps(sent, ensure_ascii=False))
        self.assertFalse(sent["story_context"]["visual_evidence_available"])
        self.assertEqual(coop_intro.round_forecast(snapshot), sent["current_forecast"])
        self.assertEqual("两人共同寻找安全通路", sent["current_forecast"]["objective"])
        self.assertEqual("岩龙", sent["current_forecast"]["boss_candidate"]["name"])
        self.assertEqual([], sent["story_context"]["persistent_history"])
        self.assertEqual([], sent["story_context"]["recent_rounds"])
        self.assertEqual(plan["world_event"], sent["target_forecast"]["world_event"])
        with coop._edit(self.code, persist=False) as room:
            self.assertEqual("ready", room["pending_guidance"]["status"])
            public = coop._public(room)
            self.assertNotIn("result", public["pending_guidance"])
        with coop._edit(self.code) as room:
            room["round"] = 3
            room["state_revision"] = 8
            room.pop("inflight", None)
            self.assertTrue(coop_intro.promote_history_guidance(room, snapshot))
            self.assertEqual("history_actions", room["intro"]["source"])
            self.assertNotIn("pending_guidance", room)

    def test_current_forecast_changes_retry_cache_and_allows_continuation(self):
        snapshot = copy.deepcopy(self.room)
        old_signature = coop_intro.history_guidance_signature(snapshot)
        snapshot["intro"].update(
            world_event="接下来桥面将崩塌。",
            world_event_type="environment",
            world_event_key="bridge_collapse",
        )
        self.assertNotEqual(old_signature, coop_intro.history_guidance_signature(snapshot))
        payload, _ = coop_intro.guidance_payload(
            snapshot,
            {},
            include_images=False,
            history_only=True,
            current_forecast=coop_intro.round_forecast(snapshot),
        )
        repeated = {
            **coop_intro.round_forecast(snapshot),
            "narration": "两人准备继续前进。",
            "objective": "寻找安全通路",
            "suggestions": [{"user_id": m["user_id"], "action": m["action"]} for m in self.members],
        }
        result, _ = guidance_checks.validate_response(repeated, self.members, payload)
        self.assertEqual("bridge_collapse", result["world_event_key"])
        changed_goal = copy.deepcopy(snapshot)
        changed_goal["intro"]["objective"] = "保护队友离开古桥"
        self.assertNotEqual(
            coop_intro.history_guidance_signature(snapshot),
            coop_intro.history_guidance_signature(changed_goal),
        )

    def test_current_boss_introduction_is_counted_without_becoming_a_fact(self):
        snapshot = copy.deepcopy(self.room)
        snapshot["setting"] = "寻找巨龙"
        snapshot["intro"].update(
            world_event="接下来巨龙将从洞口现身。",
            world_event_type="boss_introduction",
            world_event_key="dragon_enters",
        )
        payload, _ = coop_intro.guidance_payload(
            snapshot,
            {"id": "previous"},
            include_images=False,
            history_only=True,
            current_forecast=coop_intro.round_forecast(snapshot),
        )
        self.assertFalse(payload["story_context"]["boss_introduction_due"])
        self.assertNotIn("boss_known", payload["story_context"]["current_state"])
        recovery, _ = coop_intro.guidance_payload(snapshot, {}, include_images=False, history_only=True)
        self.assertNotIn("current_forecast", recovery)

    def test_skipped_boss_audit_starts_recommendation_prefetch(self):
        snapshot = json.loads(json.dumps(self.room))
        snapshot["request_id"] = "turn-2"
        with coop._edit(self.code) as room:
            room.update(status="waiting", state_revision=8, round=3)
            room["rounds"] = [{"round": 2, "node": {"id": "n2"}}]
            room["intro"].update(source="history_actions", base_revision=7)
            room.pop("inflight", None)
        with (
            patch.object(coop.coop_assets, "queue_round_boss_extraction", return_value=False),
            patch.object(coop_prefetch, "queue_recommendation") as queue,
        ):
            coop._after_round(self.code, snapshot)
        queue.assert_called_once_with(self.code)

    def test_exact_video_retry_rebinds_ready_private_forecast_to_new_request(self):
        first = json.loads(json.dumps(self.room))
        first["request_id"] = "turn-failed"
        with coop._edit(self.code) as room:
            room["inflight"] = {"request_id": "turn-failed"}
        plan = {
            "narration": "下一轮继续沿石桥推进。",
            "world_event": "下一刻桥面右侧将有碎石滑落。",
            "world_event_type": "environment",
            "world_event_key": "right_side_gravel",
            "boss_candidate": {"name": "", "description": ""},
            "objective": "避开碎石并抵达对岸",
            "suggestions": [
                {"user_id": "u1", "action": "沿左侧护栏快速前进"},
                {"user_id": "u2", "action": "举盾掩护队友通过"},
            ],
        }
        with (
            patch.dict(os.environ, {"COOP_INTRO_ENABLED": "1"}),
            patch.object(coop_intro._POOL, "submit", side_effect=immediate),
            patch.object(coop_intro.coop_trace, "chat", side_effect=[plan, reactions_for(plan)]) as chat,
        ):
            self.assertTrue(coop_intro.queue_history_guidance(first))
            with coop._edit(self.code) as room:
                self.assertEqual("ready", room["pending_guidance"]["status"])
                room["inflight"] = {"request_id": "turn-retry"}
            retry = json.loads(json.dumps(first))
            retry["request_id"] = "turn-retry"
            self.assertTrue(coop_intro.queue_history_guidance(retry))
        self.assertEqual(2, chat.call_count)
        with coop._edit(self.code) as room:
            self.assertEqual("turn-retry", room["pending_guidance"]["turn_request_id"])
            room["round"] = 3
            room["state_revision"] = 8
            room.pop("inflight", None)
            self.assertTrue(coop_intro.promote_history_guidance(room, retry))
            self.assertEqual("history_actions", room["intro"]["source"])

    def test_failed_private_forecast_releases_visible_loading_state(self):
        snapshot = json.loads(json.dumps(self.room))
        snapshot["request_id"] = "turn-fails-after-video"
        queued = []
        with coop._edit(self.code) as room:
            room["inflight"] = {"request_id": snapshot["request_id"]}
        with (
            patch.dict(os.environ, {"COOP_INTRO_ENABLED": "1"}),
            patch.object(coop_intro._POOL, "submit", side_effect=lambda *args: queued.append(args)),
        ):
            self.assertTrue(coop_intro.queue_history_guidance(snapshot))
        request_id = "forecast-" + snapshot["request_id"]
        with coop._edit(self.code) as room:
            room.update(status="waiting", round=3, state_revision=8)
            room["rounds"] = [{"round": 2, "node": {"id": "n2"}}]
            room.pop("inflight", None)
            room["intro"] = {
                "status": "generating",
                "request_id": request_id,
                "source": "history_actions",
                "signature": coop_intro.signature(room),
                "round": 3,
                "error": "",
            }
        worker, worker_snapshot, ownership = queued.pop()
        with patch.object(coop_intro.coop_trace, "chat", side_effect=RuntimeError("planner unavailable")):
            worker(worker_snapshot, ownership)
        with coop._edit(self.code, persist=False) as room:
            self.assertEqual("failed", room["pending_guidance"]["status"])
            self.assertEqual("failed", room["intro"]["status"])
            self.assertIn("可以重试", room["intro"]["error"])

    def test_orphaned_guidance_is_recovered_after_process_restart(self):
        with coop._edit(self.code) as room:
            room.update(status="waiting", round=3, state_revision=8)
            room.pop("inflight", None)
            room["pending_guidance"] = {
                "status": "planning",
                "request_id": "forecast-orphan",
                "turn_request_id": "turn-2",
                "error": "",
            }
            room["intro"] = {
                "status": "generating",
                "request_id": "forecast-orphan",
                "source": "history_actions",
                "signature": "stale",
                "error": "",
            }
        with coop._edit(self.code) as room:
            coop_intro.recover(room)
        with coop._edit(self.code, persist=False) as room:
            self.assertEqual("failed", room["pending_guidance"]["status"])
            self.assertEqual("failed", room["intro"]["status"])
            self.assertEqual("", room["intro"]["signature"])
            self.assertIn("服务重启", room["pending_guidance"]["error"])

    def test_orphaned_matching_private_forecast_is_requeued(self):
        snapshot = json.loads(json.dumps(self.room))
        snapshot["request_id"] = "turn-orphan-requeue"
        signature_value = coop_intro.history_guidance_signature(snapshot)
        with coop._edit(self.code) as room:
            room["inflight"] = {"request_id": snapshot["request_id"]}
            room["pending_guidance"] = {
                "status": "planning",
                "request_id": "old-forecast",
                "turn_request_id": snapshot["request_id"],
                "signature": signature_value,
                "base_revision": snapshot["state_revision"],
                "error": "",
            }
        queued = []
        with (
            patch.dict(os.environ, {"COOP_INTRO_ENABLED": "1"}),
            patch.object(coop_intro._POOL, "submit", side_effect=lambda *args: queued.append(args)),
        ):
            self.assertTrue(coop_intro.queue_history_guidance(snapshot))
        self.assertEqual(1, len(queued))
        with coop._edit(self.code, persist=False) as room:
            self.assertEqual("planning", room["pending_guidance"]["status"])
            self.assertEqual("forecast-" + snapshot["request_id"], room["pending_guidance"]["request_id"])
        queued[0][-1].close()

    def test_no_as_dict_output_leaks_a_frozen_proxy_into_the_node(self):
        """Every as_dict() must survive a deep copy by asdict(node); a frozen proxy fails at commit time.

        The failure: the video was already rendered and video_ready sent, then committing the
        Node deep-copied the directive with `asdict(node)` and hit a MappingProxyType in the
        story_state patch:

            round_failed  cannot pickle 'mappingproxy' object

        The users waited a whole round for nothing, and retries failed deterministically.
        WorldSnapshot.as_dict already unfroze with _copy_json; WorldDelta.as_dict did not, and
        values is the only field in this family holding arbitrary JSON. This test scans every
        as_dict(), not only that one.
        """
        frozen_state = plan_mod._freeze_json(
            {"env": {"light": "dim", "props": ["灯", "桌"]}, "boss": {"hp": 3}}
        )
        room = {
            "world_id": "w1",
            "code": "AB",
            "state_revision": 3,
            "round": 0,
            "setting": "s",
            "visual_style": "v",
            "members": [
                {
                    "user_id": "u1",
                    "subject_id": "S1",
                    "character": "甲",
                    "description": "d",
                    "action": "a",
                    "ref": "r.jpg",
                }
            ],
            "story_state": {"env": {"light": "dim"}},
        }
        snapshot = plan_mod.WorldSnapshot.from_room(room, node_id="n1")
        delta = plan_mod.WorldDelta(
            values=(("story_state", frozen_state),), source="planner", base_revision=1
        )
        scene_plan = plan_mod.ScenePlan(
            base_revision=1,
            duration=10,
            continuation=True,
            actions=(
                plan_mod.PlannedAction(user_id="u1", source_action="走", appearance="a", action="walks"),
            ),
            scene="a room",
            sound="hum",
            world_delta=delta,
            subject_order=("u1",),
            submission_order=("u1",),
        )

        def proxies(value, path="root"):
            found = []
            if isinstance(value, MappingProxyType):
                found.append(path)
            if isinstance(value, dict):
                for k, v in value.items():
                    found += proxies(v, f"{path}.{k}")
            elif isinstance(value, (list, tuple)):
                for i, v in enumerate(value):
                    found += proxies(v, f"{path}[{i}]")
            return found

        @dataclasses.dataclass
        class _Node:
            directive: dict = dataclasses.field(default_factory=dict)

        for name, produced in (
            ("WorldDelta", delta.as_dict()),
            ("WorldSnapshot", snapshot.as_dict()),
            ("ScenePlan", scene_plan.as_dict()),
        ):
            with self.subTest(source=name):
                self.assertEqual([], proxies(produced), f"{name}.as_dict() 漏了冻结代理")
                # Equivalent to what coop.py does on commit.
                dataclasses.asdict(_Node(directive={"scene_plan": produced}))

    def test_timeline_never_repeats_the_subject_it_already_bound(self):
        """`<Subject N>` must not be followed by another subject; the number exists to disambiguate.

        The server prepends `<Subject N>` to each timeline action, so action must be a bare
        predicate. Models often write a full sentence, which becomes:

            <Subject 2> The character uses a weapon to block the arm.

        In a sample of 110 timeline entries with a capitalized word after `<Subject N>`, 63 (57%)
        repeated the subject. Worse than ugly: real text also used role names for **another**
        subject ("The swordswoman follows closely behind the archer"), and the video model cannot
        tell who is who.

        Verb anchor: actions are present tense, third person, so the verb ends in s. All 34
        recorded samples were stripped correctly, none wrongly.
        """
        cases = [
            ("The character stands at the left edge.", "stands at the left edge."),
            ("The female ranger rolls quickly to her left.", "rolls quickly to her left."),
            ("The Forest Guard stands at the entrance.", "stands at the entrance."),
            ("The silver-haired female stands firm.", "stands firm."),
            ("The player is violently pulled backward.", "is violently pulled backward."),
            ("The swordswoman follows closely behind the archer.", "follows closely behind the archer."),
            ("She raises the bow and draws the string.", "raises the bow and draws the string."),
            # Already bare predicates stay; new objects introduced with an indefinite article stay
            # too.
            (
                "swings a massive stone fist toward the players.",
                "swings a massive stone fist toward the players.",
            ),
            (
                "A massive stone fist swings toward the players.",
                "A massive stone fist swings toward the players.",
            ),
        ]
        for source, expected in cases:
            with self.subTest(source=source):
                self.assertEqual(expected, plan_mod._strip_leading_subject(source)[0])

        actions = tuple(
            plan_mod.PlannedAction(user_id=uid, source_action="x", appearance="a", action=act)
            for uid, act in (
                ("u1", "The character stands on the left."),
                ("u2", "The swordswoman follows behind."),
            )
        )
        compiled = plan_mod.ShotCompiler.timeline(
            plan_mod.ScenePlan(
                base_revision=1,
                duration=10,
                continuation=False,
                actions=actions,
                scene="a bridge",
                sound="wind",
                subject_order=("u1", "u2"),
                event_action="The stone bridge trembles under the impact.",
            )
        )
        self.assertIn("<Subject 1> stands on the left", compiled)
        self.assertIn("<Subject 2> follows behind", compiled)
        self.assertNotIn("<Subject 1> The", compiled)
        self.assertNotIn("<Subject 2> The", compiled)
        # Environment events carry no number and must not be stripped.
        self.assertIn("The stone bridge trembles under the impact", compiled)


if __name__ == "__main__":
    unittest.main(verbosity=2)
