import unittest
from unittest.mock import patch
from murea_agent import config
from murea_agent.providers.video import murea
from murea_agent.guards import density_audit
from murea_agent.guards import action_sequence
from murea_agent.director import director  # noqa: E402
from murea_agent.director import prefetch as prefetch_mod  # noqa: E402
from murea_agent.guards import replan as replan_mod  # noqa: E402
from murea_agent.director import agent_schema
from murea_agent import util, loop
from murea_agent.memory.world import World


class AutoDurationTest(unittest.TestCase):
    def setUp(self):
        maximum = patch.object(config, "VIDEO_MAX_SECONDS", 15)
        maximum.start()
        self.addCleanup(maximum.stop)
        minimum = patch.object(config, "VIDEO_MIN_SECONDS", 10)
        minimum.start()
        self.addCleanup(minimum.stop)
        # The full Murea model: its clip length comes from the config tested here (Murea-Fast
        # clamps to 5 s, see test_murea_fast_clips_cap_the_budget).
        full = patch.object(murea, "backend", return_value="murea")
        full.start()
        self.addCleanup(full.stop)
        self.world = World(video_model="murea")

    def test_auto_budget_and_fixed_request_have_distinct_cache_keys(self):
        self.assertEqual(15, director._segment_seconds(self.world))
        self.assertNotEqual(
            prefetch_mod._prefetch_key(self.world, "过桥"), prefetch_mod._prefetch_key(self.world, "过桥", 15)
        )

    def test_model_duration_and_explicit_override(self):
        d = {"duration_seconds": 11}
        self.assertEqual(11, director._generation_seconds(self.world, d, None))
        self.assertEqual(4, director._generation_seconds(self.world, d, 4))
        self.assertIn("duration_seconds", agent_schema._agent_schema()["required"])

    def test_event_estimate_cannot_be_shortened_by_model_duration(self):
        d = {
            "duration_seconds": 4,
            "action_sequence": {
                "events": [{"id": 1, "estimated_seconds": 9.2}],
                "steps": [{"event_ids": [1]}],
            },
        }
        self.assertEqual(10, director._generation_seconds(self.world, d, None))

    def test_limits_and_missing_field(self):
        self.assertEqual(15, director._generation_seconds(self.world, {"duration_seconds": 99}, None))
        self.assertEqual(10, director._generation_seconds(self.world, {}, None))
        self.assertEqual(10, director._generation_seconds(self.world, {"duration_seconds": "bad"}, None))

    def test_gm_picks_inside_the_floor_and_ceiling(self):
        """The director picks, but never below 10 seconds, so a beat is not cut into fragments."""
        self.assertEqual(10, director._segment_floor(self.world))
        self.assertEqual(10, director._generation_seconds(self.world, {"duration_seconds": 4}, None))
        self.assertEqual(13, director._generation_seconds(self.world, {"duration_seconds": 13}, None))
        self.assertIn("Choose 10-15 seconds", director._duration_instruction(self.world, None))
        # An explicit length is not bound by the minimum; the caller decides.
        self.assertEqual(4, director._segment_floor(self.world, 4))
        self.assertEqual(4, director._generation_seconds(self.world, {}, 4))

    def test_short_budget_caps_output_but_never_truncates_a_long_action(self):
        with patch.object(config, "VIDEO_MAX_SECONDS", 6), patch.object(config, "VIDEO_MIN_SECONDS", 4):
            self.assertEqual(6, director._segment_seconds(self.world))
            self.assertEqual(6, director._generation_seconds(self.world, {"duration_seconds": 15}, None))
            self.assertEqual(12, director._generation_seconds(self.world, {}, 12))
            self.assertIn("Choose 4-6 seconds", director._duration_instruction(self.world, None))
            long_action = {
                "action_sequence": {
                    "events": [{"id": 1, "estimated_seconds": 10}],
                    "steps": [{"event_ids": [1]}],
                }
            }
            self.assertTrue(
                action_sequence._action_sequence_semantic_errors(long_action["action_sequence"], 6)
            )
            with self.assertRaisesRegex(ValueError, "must be split"):
                director._generation_seconds(self.world, long_action, None)
            short_key = prefetch_mod._prefetch_key(self.world, "过桥")
            self.assertNotEqual(short_key, prefetch_mod._prefetch_key(self.world, "过桥", 6))
        self.assertNotEqual(short_key, prefetch_mod._prefetch_key(self.world, "过桥"))

    def test_short_motion_still_gets_goal_coverage_review(self):
        plan = {
            "events": [
                {
                    "id": 1,
                    "description": "扶起同伴往前走",
                    "visible_result": "向前移动一段距离",
                    "estimated_seconds": 10,
                }
            ],
            "steps": [{"kind": "motion", "event_ids": [1]}],
        }
        self.assertTrue(density_audit._single_event_needs_density_audit(plan, 15))

    def test_continuous_events_use_duration_instead_of_an_extra_agent_turn(self):
        plan = {
            "events": [
                {
                    "id": 1,
                    "description": "上前扶起同伴",
                    "visible_result": "同伴已站起",
                    "estimated_seconds": 6,
                }
            ],
            "steps": [{"kind": "motion", "event_ids": [1]}],
        }
        audit_response = {
            "continuous_shot": True,
            "goal_covered": True,
            "events": [
                {"description": "走近", "visible_result": "抓住手臂", "estimated_seconds": 3},
                {"description": "扶起", "visible_result": "同伴站起", "estimated_seconds": 4},
            ],
        }
        with patch.object(loop.llm, "chat", return_value=audit_response):
            audit = density_audit._audit_single_event_density(plan, "上前扶起同伴", 15)
            fixed = density_audit._audit_single_event_density(plan, "上前扶起同伴", 4)
        self.assertEqual(1, audit["required_segments"])
        self.assertEqual(2, fixed["required_segments"])
        with (
            patch.object(replan_mod, "_audit_single_event_density", return_value=audit),
            patch.object(director, "_decide") as replan,
        ):
            decision, repaired = replan_mod._prepare_action_plan(
                self.world,
                "上前扶起同伴",
                {"action_sequence": plan, "duration_seconds": 6},
                lambda _: None,
                duration=None,
                asset_mentions=None,
                clock=util._Clock(),
            )
        replan.assert_not_called()
        self.assertFalse(repaired)
        self.assertEqual(10, director._generation_seconds(self.world, decision, None))

    def test_separate_scenes_are_not_merged_even_when_duration_fits(self):
        plan = {
            "events": [
                {
                    "id": 1,
                    "description": "离开并到达另一场景",
                    "visible_result": "抵达",
                    "estimated_seconds": 6,
                }
            ],
            "steps": [{"kind": "motion", "event_ids": [1]}],
        }
        with patch.object(
            loop.llm,
            "chat",
            return_value={
                "continuous_shot": False,
                "goal_covered": True,
                "events": [{"estimated_seconds": 3}, {"estimated_seconds": 4}],
            },
        ):
            audit = density_audit._audit_single_event_density(plan, "换场景", 15)
        self.assertEqual(2, audit["required_segments"])

    def test_goal_repair_can_keep_multiple_events_in_one_continuous_step(self):
        audit = {
            "required_segments": 1,
            "continuous_shot": True,
            "goal_covered": False,
            "events": [
                {"description": "走近", "visible_result": "抓住手臂", "estimated_seconds": 3},
                {"description": "扶起", "visible_result": "站起", "estimated_seconds": 4},
            ],
        }
        schema = density_audit._density_replan_schema(self.world, audit)
        steps = schema["properties"]["action_sequence"]["properties"]["steps"]
        self.assertEqual(1, steps["maxItems"])
        self.assertEqual({"const": [1, 2]}, steps["prefixItems"][0]["properties"]["event_ids"])
        self.assertIn("全部 event_ids", density_audit._density_audit_repair_hint(audit, 15))


if __name__ == "__main__":
    unittest.main()
