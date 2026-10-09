"""Evidence feedback, conditional goals and corrections during speculative work."""

import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from murea_agent import config
from murea_agent.coop import (
    room as coop,
    director as coop_director,
    intro as coop_intro,
    memory as coop_memory,
    prefetch as coop_prefetch,
    story as coop_story,
)


def short_plan():
    return {
        "purpose": "breakthrough",
        "change_reason": "石门仍关闭，先开门再通过",
        "steps": [
            {
                "id": "open_gate",
                "goal": "打开石门",
                "prerequisite": "门边的把手清晰可见",
                "cooperation": "弓手掩护，法师拉门",
                "completion_evidence": "门扇打开留出通路",
                "if_blocked": "检查把手附近是否有可见的阻挡物",
            },
            {
                "id": "cross_gate",
                "goal": "穿过门口",
                "prerequisite": "已确认石门打开",
                "cooperation": "一人先通过，另一人跟随",
                "completion_evidence": "两人站在门外",
                "if_blocked": "沿门边检查可见的通行空间",
            },
        ],
    }


class DirectorTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.jobs = []
        for p in (
            patch.object(config, "DATA", Path(self.tmp.name)),
            patch.dict(os.environ, {"COOP_STORY_ENABLED": "1", "COOP_INTRO_ENABLED": "1"}),
            patch.object(
                coop_intro._REFRESH_POOL, "submit", side_effect=lambda *args: self.jobs.append(args)
            ),
            patch.object(coop_intro.threading, "Timer"),
            patch.object(coop_prefetch, "queue_recommendation"),
        ):
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(lambda: [j[-1].close() for j in self.jobs])
        self.code = "ABCDEF123456"
        members = [
            {
                "user_id": uid,
                "character": name,
                "description": "轻甲",
                "ref": uid + ".jpg",
                "subject_id": uid,
                "action": "",
                "ready": False,
            }
            for uid, name in (("a", "弓手"), ("b", "法师"))
        ]
        self.room = {
            "code": self.code,
            "world_id": "w-director",
            "round": 2,
            "revision": 1,
            "state_revision": 1,
            "status": "waiting",
            "setting": "离开洞穴",
            "visual_style": "cinematic",
            "members": members,
            "scene_name": "洞穴",
            "scene_description": "石门通向洞外",
            "rounds": [{"round": 1, "node": {"id": "n1"}, "inputs": copy.deepcopy(members)}],
            "story_state": {"environment": "洞穴中有一道石门", "players": {"a": "门左侧", "b": "门旁"}},
            "story_outline": {
                "version": 1,
                "chapter": 1,
                "overall_goal": "离开洞穴",
                "active_stage_id": "leave",
                "stages": [
                    {
                        "id": "leave",
                        "title": "离开洞穴",
                        "goal": "共同离开",
                        "completion_evidence": "两人站在洞外",
                        "alternative": "换路",
                    }
                ],
            },
            "story_progress": {
                "version": 1,
                "last_round": 0,
                "active_stage_id": "leave",
                "assessment": "advanced",
                "stages": {"leave": {"status": "active"}},
                "next_gap": "打开石门",
                "player_outcomes": [],
            },
        }
        coop_memory.apply_updates(self.room["story_state"], [self.fact("关闭")], "n0", 0)
        self.forecast = {
            "narration": "先开门再通过",
            "world_event": "接下来门外的薄雾将飘进洞内。",
            "world_event_type": "environment",
            "world_event_key": "mist_enters",
            "boss_candidate": {"name": "", "description": ""},
            "objective": "打开石门",
            "stage_id": "leave",
            "expected_progress": "石门打开",
            "short_term_plan": short_plan(),
        }
        self.room["intro"] = {
            **self.forecast,
            "short_term_plan": {**short_plan(), "stage_id": "leave"},
            "status": "ready",
            "request_id": "guide",
            "suggestions": [{"user_id": m["user_id"], "action": "拉开石门"} for m in members],
            "story_basis": {**coop_story.basis(coop_story.context(self.room)), "stage_id": "leave"},
            "state_basis": coop_memory.fingerprint(self.room["story_state"]),
        }
        self.room["intro"]["signature"] = coop_intro.signature(self.room)
        self.save()

    def fact(self, value):
        return {
            "entity": "object:stone_gate",
            "attribute": "passage",
            "value": value,
            "evidence": "关键帧中的门扇位置",
            "frame_indices": [1],
        }

    def save(self):
        coop._write(coop._directory() / (self.code + ".json"), self.room)

    def read(self):
        with coop._edit(self.code, persist=False) as room:
            return copy.deepcopy(room)

    def gate_opens(self):
        with coop._edit(self.code) as room:
            coop_memory.apply_updates(room["story_state"], [self.fact("打开")], "n1", 1)
            room["story_progress"].update(version=2, last_round=1, next_gap="穿过门口")

    def reactions(self, action="沿门边通过石门"):
        return {
            "world_event_key": "mist_enters",
            "suggestions": [
                {
                    "user_id": uid,
                    "action": action,
                    "forecast_cue": "薄雾将飘进洞内",
                    "response_reason": "沿门边前进",
                    "tactic": {
                        "technique": "move",
                        "target": "洞外",
                        "route": "门边",
                        "role": "先行" if uid == "a" else "跟随",
                    },
                }
                for uid in ("a", "b")
            ],
        }

    def run_job(self, responses):
        fn, *args = self.jobs.pop(0)
        with patch.object(coop_intro.coop_trace, "chat", side_effect=responses) as chat:
            fn(*args)
        return chat

    def test_same_stage_gate_change_marks_private_guidance_stale(self):
        before = self.read()
        self.gate_opens()
        room = self.read()
        self.assertEqual(coop.generation_signature(before), coop.generation_signature(room))
        self.assertTrue(coop_story.stale_guidance(room, before["intro"]))
        self.assertTrue(coop_intro._guidance_stale(room, before["intro"]))

    def test_slots_replace_conflicts_and_preserve_unseen_objects(self):
        state = {}
        gate = self.fact("关闭")
        bridge = dict(gate, entity="object:bridge", value="断裂")
        coop_memory.apply_updates(state, [gate, bridge], "n1", 1)
        coop_memory.apply_updates(state, [self.fact("打开")], "n2", 2)
        self.assertEqual("打开", state["facts"]["object:stone_gate.passage"]["value"])
        self.assertEqual("断裂", state["facts"]["object:bridge.passage"]["value"])
        coop_memory.apply_updates(state, [gate], "n0", 0)
        self.assertEqual("打开", state["facts"]["object:stone_gate.passage"]["value"])
        before = coop_memory.fingerprint(state)
        coop_memory.apply_updates(state, [dict(self.fact("打开"), evidence="门敞开")], "n3", 3)
        self.assertEqual(before, coop_memory.fingerprint(state))

    def test_fact_updates_require_visual_evidence_and_known_player_ids(self):
        for update in (
            dict(self.fact("打开"), frame_indices=[2]),
            dict(self.fact("打开"), entity="invented-player"),
            dict(self.fact("打开"), evidence=""),
        ):
            with self.assertRaises(ValueError):
                coop_memory.validate_updates([update], self.room["members"], 1)

    def test_current_context_does_not_duplicate_fifty_round_ledger(self):
        self.room["story_state"]["persistent_history"] = [
            {"round": n, "observation": {"summary": str(n)}} for n in range(50)
        ]
        payload, _ = coop_intro.guidance_payload(self.room, {}, include_images=False)
        self.assertNotIn("persistent_history", payload["story_context"]["current_state"])
        self.assertEqual(8, len(payload["story_context"]["persistent_history"]))

    def test_refresh_changes_actions_only_and_keeps_forecast(self):
        self.gate_opens()
        self.assertTrue(coop_intro.queue_state_refresh(self.code))
        self.assertFalse(coop_intro.queue_state_refresh(self.code))
        self.assertEqual("拉开石门", self.read()["intro"]["suggestions"][0]["action"])
        chat = self.run_job([{"scope": "actions", "reason": "门已打开"}, self.reactions()])
        self.assertEqual(
            ["coop_guidance_review", "coop_forecast_reactions"],
            [c.kwargs["tag"] for c in chat.call_args_list],
        )
        room = self.read()
        self.assertEqual(self.forecast["world_event"], room["intro"]["world_event"])
        self.assertEqual("沿门边通过石门", room["intro"]["suggestions"][0]["action"])
        self.assertFalse(coop_intro._guidance_stale(room, room["intro"]))

    def test_keep_review_preserves_prepared_cache(self):
        self.gate_opens()
        with coop._edit(self.code) as room:
            room["recommendation_prefetch"] = {
                "status": "prompt_ready",
                "signature": "cached",
                "_prepared": {"prompt": "cached"},
            }
        coop_intro.queue_state_refresh(self.code)
        self.run_job([{"scope": "keep", "reason": "这些动作仍适用"}])
        self.assertEqual("cached", self.read()["recommendation_prefetch"]["_prepared"]["prompt"])
        self.assertFalse(coop_intro.queue_state_refresh(self.code))

    def test_refresh_failure_keeps_playable_guidance_and_does_not_retry_same_evidence(self):
        self.gate_opens()
        before = self.read()["intro"]
        coop_intro.queue_state_refresh(self.code)
        self.run_job([RuntimeError("service unavailable")])
        self.assertEqual(before, self.read()["intro"])
        self.assertFalse(coop_intro.queue_state_refresh(self.code))

    def test_player_edit_during_review_wins(self):
        self.gate_opens()
        before = self.read()["intro"]
        coop_intro.queue_state_refresh(self.code)

        def review(**kwargs):
            with coop._edit(self.code) as room:
                room["members"][0]["action"] = "检查身后的墙壁"
            return {"scope": "keep", "reason": "仍适用"}

        self.run_job(review)
        self.assertEqual(before, self.read()["intro"])
        self.assertEqual("superseded", self.read()["guidance_refresh"]["status"])

    def test_slow_correction_times_out_and_late_result_cannot_replace_guidance(self):
        self.gate_opens()
        before = self.read()["intro"]
        coop_intro.queue_state_refresh(self.code)
        key = self.read()["guidance_refresh"]["key"]
        coop_intro._expire_state_refresh(self.code, key)
        self.assertEqual("timed_out", self.read()["guidance_refresh"]["status"])
        self.run_job([{"scope": "actions", "reason": "门已打开"}, self.reactions()])
        self.assertEqual(before, self.read()["intro"])
        self.assertFalse(coop_intro.queue_state_refresh(self.code))

    def test_asset_edit_removes_old_visual_slots_but_keeps_history(self):
        self.room["story_state"]["persistent_history"] = [{"round": 1, "observation": {"summary": "旧石门"}}]
        coop._mark_asset_edit(self.room)
        self.assertNotIn("facts", self.room["story_state"])
        self.assertEqual(1, len(self.room["story_state"]["persistent_history"]))

    def test_submitted_video_and_accepted_actions_are_never_replaced(self):
        for status in ("rendering", "consuming", "ready"):
            self.save()
            self.gate_opens()
            with coop._edit(self.code) as room:
                room["recommendation_prefetch"] = {"status": status}
            self.assertFalse(coop_intro.queue_state_refresh(self.code))
        self.save()
        self.gate_opens()
        with coop._edit(self.code) as room:
            room["members"][0].update(action="向后跑", ready=True)
        self.assertFalse(coop_intro.queue_state_refresh(self.code))

    def test_short_plan_and_tactics_reach_the_same_round_and_history(self):
        payload, _ = coop_intro.guidance_payload(self.room, {}, include_images=False)
        schema = coop_intro.guidance_schema(
            latest=False, forecast_only=True, story_plan=payload["story_plan"]
        )
        self.assertEqual(
            ["stage_id", "short_term_plan", "expected_progress", "objective"], list(schema["properties"])[:4]
        )
        with patch.object(
            coop_intro.coop_trace, "chat", side_effect=[self.forecast, self.reactions()]
        ) as chat:
            result, _ = coop_intro._plan_guidance(
                payload, self.room["members"], system=coop_intro._SYSTEM, tag="test", temperature=0.6
            )
        target = json.loads(chat.call_args_list[-1].kwargs["user"])["target_forecast"]
        self.assertEqual("open_gate", target["short_term_plan"]["steps"][0]["id"])
        adopted = copy.deepcopy(self.room)
        adopted["intro"] = result
        adopted["members"][0]["action"] = result["suggestions"][0]["action"]
        adopted["members"][1]["action"] = "玩家自定义动作"
        stored = coop_intro.round_forecast(adopted)
        self.assertEqual(["a"], [t["user_id"] for t in stored["tactics"]])
        self.assertEqual(2, len(stored["short_term_plan"]["steps"]))

    def test_blocked_and_unknown_have_distinct_recovery_behaviors(self):
        self.room["story_progress"]["stages"]["leave"]["status"] = "blocked"
        self.assertEqual("blocked", coop_director.feedback(self.room)["mode"])
        self.assertTrue(coop_story.context(self.room)["needs_new_approach"])
        self.room["story_progress"]["stages"]["leave"]["status"] = "active"
        self.room["story_progress"].update(assessment="unknown", stagnation_rounds=0)
        self.assertEqual("uncertain", coop_director.feedback(self.room)["mode"])
        self.assertFalse(coop_story.context(self.room)["needs_new_approach"])

    def test_stagnation_replans_now_without_waiting_for_outline_checkpoint(self):
        self.gate_opens()
        with coop._edit(self.code) as room:
            room["story_progress"].update(assessment="stalled", stagnation_rounds=2)
        self.assertIsNone(coop_story._due(self.read()))
        coop_intro.queue_state_refresh(self.code)
        chat = self.run_job([self.forecast, self.reactions()])
        self.assertEqual("coop_guidance_refresh", chat.call_args_list[0].kwargs["tag"])
        self.assertEqual("ready", self.read()["guidance_refresh"]["status"])

    def test_semantic_review_can_repair_reworded_stalling_without_rewriting_event(self):
        payload, _ = coop_intro.guidance_payload(self.room, {}, include_images=False)
        payload["director_feedback"] = {"mode": "stalled"}
        payload["recent_tactics"] = [{"round": n, "assessment": "stalled", "tactics": []} for n in (1, 2)]
        with patch.object(
            coop_intro.coop_trace,
            "chat",
            side_effect=[
                self.forecast,
                self.reactions("换个姿势继续守在门口"),
                {"scope": "actions", "reason": "应向已打开的门外前进"},
                self.reactions(),
            ],
        ) as chat:
            result, _ = coop_intro._plan_guidance(
                payload, self.room["members"], system=coop_intro._SYSTEM, tag="test", temperature=0.6
            )
        self.assertEqual(4, chat.call_count)
        self.assertEqual(self.forecast["world_event"], result["world_event"])
        self.assertEqual("沿门边通过石门", result["suggestions"][0]["action"])

    def test_same_tactic_can_continue_when_it_makes_progress(self):
        tactic = coop_director.tactics(self.reactions())[0]
        payload = {
            "director_feedback": {"mode": "stalled"},
            "recent_tactics": [{"assessment": "stalled", "tactics": [tactic]}],
        }
        self.assertEqual(["a"], coop_director.repeated_tactics([tactic], payload))
        payload["recent_tactics"].append({"assessment": "advanced", "tactics": [tactic]})
        self.assertEqual([], coop_director.repeated_tactics([tactic], payload))

    def test_only_observed_first_step_can_complete_and_cannot_be_replanned(self):
        plan = coop_story.context(self.room)
        forecast = copy.deepcopy(self.room["intro"])
        raw = {
            "assessment": "advanced",
            "summary": "石门打开",
            "next_gap": "穿过门口",
            "stage_updates": [],
            "player_outcomes": [],
            "step_outcome": {
                "step_id": "open_gate",
                "completion_condition": short_plan()["steps"][0]["completion_evidence"],
                "verdict": "satisfied",
                "evidence": "门扇移开形成通路",
                "frame_indices": [1],
            },
        }
        checked = coop_story.validate_progress(raw, plan, 1, self.room["members"], forecast)
        self.assertEqual("satisfied", checked["step_outcome"]["verdict"])
        with self.assertRaisesRegex(ValueError, "already done on screen"):
            coop_director.validate_plan(short_plan(), "leave", {"last_step": checked["step_outcome"]})
        raw["step_outcome"]["step_id"] = "cross_gate"
        with self.assertRaisesRegex(ValueError, "step actually used"):
            coop_story.validate_progress(raw, plan, 1, self.room["members"], forecast)

    def test_quality_review_unavailability_does_not_fail_guidance(self):
        payload, _ = coop_intro.guidance_payload(self.room, {}, include_images=False)
        payload["director_feedback"] = {"mode": "stalled"}
        payload["recent_tactics"] = [{"round": n, "assessment": "stalled", "tactics": []} for n in (1, 2)]
        with patch.object(
            coop_intro.coop_trace, "chat", side_effect=[self.forecast, self.reactions(), RuntimeError("busy")]
        ):
            result, _ = coop_intro._plan_guidance(
                payload, self.room["members"], system=coop_intro._SYSTEM, tag="test", temperature=0.6
            )
        self.assertEqual(2, len(result["suggestions"]))


if __name__ == "__main__":
    unittest.main()
