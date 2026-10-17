"""Forecast-first guidance, immutable reaction inputs and atomic publication."""

import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from murea_agent import config
from murea_agent.coop import guidance_checks
from murea_agent.coop import room as coop, intro as coop_intro, prefetch as coop_prefetch


class CoopGuidanceTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        data = patch.object(config, "DATA", Path(self.tmp.name))
        data.start()
        self.addCleanup(data.stop)
        self.members = [
            {
                "user_id": "player-archer",
                "name": "玩家 A",
                "character": "弓手",
                "description": "手持长弓",
                "subject_id": "s1",
                "ref": "a.jpg",
                "action": "向左跑",
                "ready": False,
            },
            {
                "user_id": "player-swordsman",
                "name": "玩家 B",
                "character": "剑士",
                "description": "手持长剑",
                "subject_id": "s2",
                "ref": "b.jpg",
                "action": "拔剑前进",
                "ready": False,
            },
        ]
        self.previous = {
            "world_event": "接下来河水将拍打桥墩，激起大片水花。",
            "world_event_type": "environment",
            "world_event_key": "river_splash",
            "boss_candidate": {"name": "", "description": ""},
            "objective": "保持平衡",
        }
        self.payload = {
            "setting": "两人渡过石桥",
            "visual_style": "cinematic",
            "shared_scene": {"scene_name": "石桥", "scene_description": "桥边有护栏"},
            "shared_creature": {},
            "players": copy.deepcopy(self.members),
            "story_context": {
                "current_state": {},
                "recent_rounds": [],
                "persistent_history": [],
                "visual_evidence_available": False,
            },
            "current_forecast": self.previous,
            "requested_actions": [{"user_id": m["user_id"], "action": m["action"]} for m in self.members],
        }
        self.forecast = {
            "narration": "两人接下来需要选择能通过石桥的路线。",
            "world_event": "下一刻桥面中央将塌陷，右侧护栏旁仍留出一条窄道。",
            "world_event_type": "environment",
            "world_event_key": "bridge_center_collapses",
            "boss_candidate": {"name": "", "description": ""},
            "objective": "沿右侧窄道共同通过石桥",
        }
        self.reactions = {
            "world_event_key": self.forecast["world_event_key"],
            "suggestions": [
                {
                    "user_id": "player-archer",
                    "forecast_cue": "桥面中央将塌陷",
                    "response_reason": "离开中央塌陷区，将右侧窄道留给同伴同行。",
                    "action": "收弓侧身贴近右侧护栏，沿窄道向对岸小步前进",
                },
                {
                    "user_id": "player-swordsman",
                    "forecast_cue": "右侧护栏旁仍留出一条窄道",
                    "response_reason": "沿同一条安全通道跟随，避免两人争抢窄道。",
                    "action": "持剑跟在弓手身后，沿右侧窄道避开中央裂口",
                },
            ],
        }

    def plan(self, responses, *, payload=None, latest=False, images=()):
        with patch.object(coop_intro.coop_trace, "chat", side_effect=responses) as chat:
            result, observation = coop_intro._plan_guidance(
                payload or self.payload,
                self.members,
                system=coop_intro._HISTORY_ONLY_SYSTEM,
                tag="coop_next_actions_history",
                temperature=0.6,
                latest=latest,
                images=images,
            )
        return result, observation, chat

    def test_reactions_use_new_forecast_and_cannot_replace_it(self):
        first = {
            **self.forecast,
            "suggestions": [{"user_id": m["user_id"], "action": "躲在掩体后寻找出口"} for m in self.members],
        }
        second = {
            **self.reactions,
            "world_event": "被第二步替换的事件",
            "objective": "第二步擅自更换的目标",
            "boss_candidate": {"name": "另一只怪兽"},
        }
        result, _, chat = self.plan([first, second])
        self.assertEqual(2, chat.call_count)
        forecast_call, reaction_call = [c.kwargs for c in chat.call_args_list]
        self.assertNotIn("suggestions", forecast_call["schema"]["properties"])
        self.assertEqual({"world_event_key", "suggestions"}, set(reaction_call["schema"]["properties"]))
        sent = json.loads(reaction_call["user"])
        self.assertEqual(coop_intro.round_forecast({"intro": self.forecast}), sent["target_forecast"])
        self.assertEqual(self.previous, sent["current_forecast"])
        self.assertEqual(self.payload["requested_actions"], sent["requested_actions"])
        self.assertNotIn("target_forecast", self.payload)
        for field, value in self.forecast.items():
            self.assertEqual(value, result[field])
        self.assertEqual(self.reactions["suggestions"][0]["action"], result["suggestions"][0]["action"])
        self.assertEqual({"user_id", "action"}, set(result["suggestions"][0]))
        self.assertNotIn("images", forecast_call)
        self.assertNotIn("images", reaction_call)

    def test_invalid_reactions_retry_with_same_forecast(self):
        for error in ("key", "missing_player", "missing_reason"):
            with self.subTest(error=error):
                stale = copy.deepcopy(self.reactions)
                if error == "key":
                    stale["world_event_key"] = self.previous["world_event_key"]
                elif error == "missing_player":
                    stale["suggestions"].pop()
                else:
                    stale["suggestions"][0].pop("response_reason")
                result, _, chat = self.plan([self.forecast, stale, self.reactions])
                self.assertEqual(self.forecast["world_event"], result["world_event"])
                self.assertEqual(
                    ["coop_next_actions_history", "coop_forecast_reactions", "coop_forecast_reactions"],
                    [c.kwargs["tag"] for c in chat.call_args_list],
                )
                initial, retry = [json.loads(c.kwargs["user"]) for c in chat.call_args_list[1:]]
                self.assertEqual(initial["target_forecast"], retry["target_forecast"])
                self.assertIn("correction_required", retry)
                self.assertIn("rejected_suggestions", retry)

    def test_same_event_can_continue_without_forced_replanning(self):
        repeated = {**self.forecast, **self.previous}
        reactions = {**self.reactions, "world_event_key": repeated["world_event_key"]}
        result, _, chat = self.plan([repeated, reactions])
        self.assertEqual(
            ["coop_next_actions_history", "coop_forecast_reactions"],
            [c.kwargs["tag"] for c in chat.call_args_list],
        )
        reaction_input = json.loads(chat.call_args_list[-1].kwargs["user"])
        self.assertNotIn("correction_required", reaction_input)
        self.assertEqual(repeated["world_event"], result["world_event"])

    def test_paraphrased_goal_cue_needs_no_retry(self):
        reactions = copy.deepcopy(self.reactions)
        reactions["suggestions"][0]["forecast_cue"] = "共同安全渡桥"
        result, _, chat = self.plan([self.forecast, reactions])
        self.assertEqual(2, chat.call_count)
        self.assertEqual(reactions["suggestions"][0]["action"], result["suggestions"][0]["action"])

    def test_failure_messages_distinguish_validation_and_provider(self):
        from murea_agent.providers.llm import GeminiError

        for cause, label in (
            (ValueError("invalid user"), "建议校验失败"),
            (coop_intro.coop_director.CompletedPlanError("completed"), "规划重复"),
        ):
            error = guidance_checks._GuidanceValidationError("建议校验失败", cause)
            self.assertTrue(guidance_checks._guidance_error(error).startswith(label))
        self.assertIn("接口不可用", guidance_checks._guidance_error(GeminiError("503 private details")))
        self.assertNotIn("private", guidance_checks._guidance_error(GeminiError("503 private details")))

    def test_no_event_uses_objective_as_reaction_basis(self):
        forecast = {**self.forecast, "world_event": "", "world_event_type": "none", "world_event_key": ""}
        reactions = copy.deepcopy(self.reactions)
        reactions["world_event_key"] = ""
        for item in reactions["suggestions"]:
            item["forecast_cue"] = "沿右侧窄道"
        result, _, _ = self.plan([forecast, reactions])
        self.assertEqual("none", result["world_event_type"])
        self.assertEqual(2, len(result["suggestions"]))

    def test_boss_introduction_passes_candidate_as_plan_without_confirming_it(self):
        payload = copy.deepcopy(self.payload)
        payload["story_context"]["boss_introduction_allowed"] = True
        forecast = {
            **self.forecast,
            "world_event": "接下来岩龙将从桥底攀上中央桥面。",
            "world_event_type": "boss_introduction",
            "world_event_key": "rock_dragon_emerges",
            "boss_candidate": {"name": "岩龙", "description": "覆盖灰色岩甲的四足巨兽"},
        }
        reactions = copy.deepcopy(self.reactions)
        reactions["world_event_key"] = forecast["world_event_key"]
        for item in reactions["suggestions"]:
            item["forecast_cue"] = "攀上中央桥面"
        _, observation, chat = self.plan([forecast, reactions], payload=payload)
        sent = json.loads(chat.call_args.kwargs["user"])
        self.assertEqual("岩龙", sent["target_forecast"]["boss_candidate"]["name"])
        self.assertEqual({}, sent["shared_creature"])
        self.assertEqual({}, sent["story_context"]["current_state"])
        self.assertIsNone(observation)

    def test_three_and_four_players_all_receive_reactions_in_member_order(self):
        for count in (3, 4):
            with self.subTest(count=count):
                self.members = self.members[:2] + [
                    {**self.members[0], "user_id": f"player-extra-{i}"} for i in range(2, count)
                ]
                reactions = copy.deepcopy(self.reactions)
                reactions["suggestions"] += [
                    {**reactions["suggestions"][1], "user_id": m["user_id"]} for m in self.members[2:]
                ]
                reactions["suggestions"].reverse()
                result, _, _ = self.plan([self.forecast, reactions])
                self.assertEqual(
                    [m["user_id"] for m in self.members], [s["user_id"] for s in result["suggestions"]]
                )

    def test_visual_recovery_preserves_observation_but_reactions_do_not_read_frames(self):
        payload = copy.deepcopy(self.payload)
        payload["story_context"]["visual_evidence_available"] = True
        forecast = {
            **self.forecast,
            "observation": {
                "event_outcome": "observed",
                "summary": "河水拍打桥墩。",
                "environment_state": "石桥上溅有水花",
                "boss_visible": False,
                "boss_state": "",
                "player_states": [{"user_id": m["user_id"], "state": "站在桥边"} for m in self.members],
            },
            "story_phase": "exploration",
        }
        _, observation, chat = self.plan(
            [forecast, self.reactions], payload=payload, latest=True, images=[("image/jpeg", b"frame")]
        )
        self.assertTrue(chat.call_args_list[0].kwargs["images"])
        self.assertNotIn("images", chat.call_args_list[1].kwargs)
        sent = json.loads(chat.call_args.kwargs["user"])
        self.assertEqual("石桥上溅有水花", sent["latest_observation"]["environment_state"])
        self.assertNotIn("_visual_evidence_available", sent["latest_observation"])
        self.assertEqual("observed", observation["event_outcome"])

    def stored_room(self):
        room = {
            "code": "ABCDEF123456",
            "world_id": "w-guidance",
            "round": 2,
            "revision": 7,
            "state_revision": 7,
            "status": "generating",
            "capacity": 2,
            "setting": self.payload["setting"],
            "visual_style": "cinematic",
            "members": self.members,
            "rounds": [],
            "story_state": {},
            "intro": {**self.previous, "status": "ready", "suggestions": []},
            "inflight": {"request_id": "turn-2"},
        }
        room["intro"]["signature"] = coop_intro.signature(room)
        coop._write(coop._directory() / f"{room['code']}.json", room)
        return room

    def test_video_commit_between_stages_publishes_complete_pair_only(self):
        room = self.stored_room()
        snapshot = {**copy.deepcopy(room), "request_id": "turn-2"}

        def chat(**kwargs):
            if kwargs["tag"] == "coop_next_actions_history":
                return self.forecast
            with coop._edit(room["code"]) as stored:
                self.assertEqual("planning", stored["pending_guidance"]["status"])
                self.assertIsNone(stored["pending_guidance"]["result"])
                self.assertEqual(room["intro"], stored["intro"])
                self.assertNotIn("result", coop._public(stored)["pending_guidance"])
                stored.update(round=3, state_revision=8, status="waiting")
                stored["rounds"] = [{"round": 2, "node": {"id": "n2"}}]
                stored.pop("inflight")
            return self.reactions

        with (
            patch.dict(os.environ, {"COOP_INTRO_ENABLED": "1"}),
            patch.object(coop_intro._POOL, "submit", side_effect=lambda fn, *args: fn(*args)),
            patch.object(coop_intro.coop_trace, "chat", side_effect=chat),
        ):
            self.assertTrue(coop_intro.queue_history_guidance(snapshot))
        with coop._edit(room["code"], persist=False) as stored:
            self.assertNotIn("pending_guidance", stored)
            self.assertEqual("ready", stored["intro"]["status"])
            self.assertEqual(self.forecast["world_event"], stored["intro"]["world_event"])
            self.assertEqual(
                self.reactions["suggestions"][0]["action"], stored["intro"]["suggestions"][0]["action"]
            )

    def test_failed_reactions_never_publish_or_prefetch_partial_forecast(self):
        room = self.stored_room()
        snapshot = {**copy.deepcopy(room), "request_id": "turn-2"}
        stale = {**self.reactions, "world_event_key": "river_splash"}
        with (
            patch.dict(os.environ, {"COOP_INTRO_ENABLED": "1"}),
            patch.object(coop_intro._POOL, "submit", side_effect=lambda fn, *args: fn(*args)),
            patch.object(coop_prefetch, "queue_recommendation") as prefetch,
            patch.object(coop_intro.coop_trace, "chat", side_effect=[self.forecast, stale, stale]),
        ):
            coop_intro.queue_history_guidance(snapshot)
        with coop._edit(room["code"], persist=False) as stored:
            self.assertEqual("failed", stored["pending_guidance"]["status"])
            self.assertIsNone(stored["pending_guidance"]["result"])
            self.assertEqual(room["intro"], stored["intro"])
            self.assertNotIn("persistent_history", stored["story_state"])
        prefetch.assert_not_called()

    def test_asset_edit_during_reactions_discards_complete_stale_result(self):
        room = self.stored_room()
        snapshot = {**copy.deepcopy(room), "request_id": "turn-2"}

        def chat(**kwargs):
            if kwargs["tag"] == "coop_next_actions_history":
                return self.forecast
            with coop._edit(room["code"]) as stored:
                stored["state_revision"] += 1
                stored["scene_name"] = "新场景"
            return self.reactions

        with (
            patch.dict(os.environ, {"COOP_INTRO_ENABLED": "1"}),
            patch.object(coop_intro._POOL, "submit", side_effect=lambda fn, *args: fn(*args)),
            patch.object(coop_intro.coop_trace, "chat", side_effect=chat),
        ):
            coop_intro.queue_history_guidance(snapshot)
        with coop._edit(room["code"], persist=False) as stored:
            self.assertNotIn("pending_guidance", stored)
            self.assertEqual(room["intro"], stored["intro"])

    def test_post_commit_asset_update_hands_off_after_guidance_unlocks(self):
        for superseded in (None, "new_round", "new_request"):
            with self.subTest(superseded=superseded):
                room = self.stored_room()
                snapshot = {**copy.deepcopy(room), "request_id": "turn-2"}
                queued = []

                def submit(fn, *args):
                    if fn is coop_intro._run_history_guidance:
                        fn(*args)
                    else:
                        queued.append((fn, args))

                def chat(**kwargs):
                    if kwargs["tag"] == "coop_next_actions_history":
                        return self.forecast
                    with coop._edit(room["code"]) as stored:
                        # Video submission + new boss saved, each bumps the state version once.
                        stored.update(round=3, state_revision=9, status="waiting")
                        stored["rounds"] = [{"round": 2, "node": {"id": "n2"}}]
                        stored.pop("inflight")
                        stored["monster"] = {"name": "岩龙", "ref": "dragon.jpg"}
                        stored["intro"] = {
                            "status": "generating",
                            "request_id": "forecast-turn-2",
                            "suggestions": [],
                        }
                        if superseded == "new_round":
                            stored["round"] = 4
                            stored["rounds"].append({"round": 3, "node": {"id": "n3"}})
                        elif superseded == "new_request":
                            stored["intro"]["request_id"] = "new-owner"
                    # Reproduce asset extraction retries blocked by the lock held by an old
                    # suggestion.
                    self.assertFalse(coop_intro.queue(room["code"], force=True, history_only=True))
                    return self.reactions

                try:
                    with (
                        patch.dict(os.environ, {"COOP_INTRO_ENABLED": "1"}),
                        patch.object(coop_intro._POOL, "submit", side_effect=submit),
                        patch.object(coop_intro.coop_trace, "chat", side_effect=chat),
                    ):
                        self.assertTrue(coop_intro.queue_history_guidance(snapshot))
                    with coop._edit(room["code"], persist=False) as stored:
                        self.assertNotIn("pending_guidance", stored)
                        self.assertEqual([], stored["intro"]["suggestions"])
                    if superseded:
                        self.assertEqual([], queued)
                    else:
                        self.assertEqual(1, len(queued))
                        fn, (fresh, guard) = queued[0]
                        self.assertIs(fn, coop_intro._run)
                        self.assertEqual(9, fresh["state_revision"])
                        self.assertEqual("dragon.jpg", fresh["monster"]["ref"])
                        self.assertTrue(fresh["history_only"])
                        self.assertNotEqual("forecast-turn-2", fresh["intro"]["request_id"])
                finally:
                    for _, (_, guard) in queued:
                        guard.close()

    def test_version_invalidates_old_guidance_and_retry_cache(self):
        room = self.stored_room()
        before = (coop_intro.signature(room), coop_intro.history_guidance_signature(room))
        with patch.object(coop_intro, "_GUIDANCE_VERSION", coop_intro._GUIDANCE_VERSION - 1):
            old = (coop_intro.signature(room), coop_intro.history_guidance_signature(room))
        self.assertNotEqual(before[0], old[0])
        self.assertNotEqual(before[1], old[1])


if __name__ == "__main__":
    unittest.main()
