"""Timing covers concurrent work, causal waits, travel and nonblocking fallback."""

import copy
import unittest
from unittest.mock import patch, PropertyMock

from murea_agent import config
from murea_agent.coop import duration
from murea_agent.providers.video import murea as murea_provider


def step(sid, actors, seconds, after=()):
    return {
        "id": sid,
        "actors": actors,
        "seconds": seconds,
        "after": list(after),
        "action": "perform requested motion",
    }


def setUpModule():
    # These tests check timing against the configured 6-10 s lengths (the full Murea model).
    # The unreachable test server reads as Murea-Fast, whose 5 s cap is tested in
    # test_coop_duration.py.
    patcher = patch.object(murea_provider.Murea, "max_seconds", new_callable=PropertyMock, return_value=15)
    patcher.start()
    unittest.addModuleCleanup(patcher.stop)


class DurationTest(unittest.TestCase):
    def setUp(self):
        for key, value in (
            ("COOP_OPENING_SECONDS", 6),
            ("COOP_ACTION_MIN_SECONDS", 6),
            ("COOP_ACTION_SECONDS", 10),
        ):
            p = patch.object(config, key, value)
            p.start()
            self.addCleanup(p.stop)
        self.room = {
            "members": [{"user_id": "a", "action": "打开石门"}, {"user_id": "b", "action": "穿过门口"}]
        }

    def decide(self, steps):
        return duration.decide(self.room, {"steps": steps})

    def test_the_video_models_limit_caps_every_length(self):
        def limit(seconds):
            return patch.object(
                murea_provider.Murea, "max_seconds", new_callable=PropertyMock, return_value=seconds
            )

        # Default 15 s rounds on Murea-Fast (20 s limit); the full model (15 s) caps a 20 s setting.
        with patch.object(config, "COOP_ACTION_SECONDS", 15), limit(20):
            self.assertEqual(15, duration.planning_seconds({}))
        with patch.object(config, "COOP_ACTION_SECONDS", 20), limit(15):
            self.assertEqual(15, duration.planning_seconds({}))
        with limit(5):
            self.assertEqual([5], duration.candidate_seconds())
            self.assertEqual(5, duration.planning_seconds({"opening": True}))
            self.assertEqual(5, duration.planning_seconds({}))
            self.assertEqual(5, duration.decide({"opening": True}).seconds)
            self.assertEqual(5, duration.policy_signature()["action_min_seconds"])

    def test_parallel_actors_use_longest_path_but_waiting_for_gate_adds_time(self):
        opening = step("gate", ["a"], 3)
        crossing = step("cross", ["b"], 4)
        parallel = self.decide([opening, crossing])
        serial = self.decide([opening, dict(crossing, after=["gate"])])
        self.assertEqual((6, 8), (parallel.seconds, serial.seconds))
        self.assertEqual((4.75, 7.75), (parallel.estimated_seconds, serial.estimated_seconds))
        self.assertEqual(3, serial.schedule[1]["start"])
        self.assertEqual("structured", serial.source)

    def test_each_actor_successive_actions_are_sequential_even_if_after_omitted(self):
        result = self.decide(
            [step("draw", ["a"], 2), step("run", ["a"], 3), step("slash", ["a"], 3), step("cover", ["b"], 4)]
        )
        self.assertEqual(10, result.seconds)
        self.assertEqual([0, 2, 5, 0], [s["start"] for s in result.schedule])

    def test_four_actors_and_minor_event_do_not_automatically_extend_shot(self):
        self.room["members"] += [{"user_id": uid, "action": "站稳"} for uid in ("c", "d")]
        self.room["intro"] = {"status": "ready", "world_event": "接下来微风吹动旗帜"}
        result = self.decide([step("hold", ["a", "b", "c", "d"], 2), step("breeze", ["world_event"], 2)])
        self.assertEqual(6, result.seconds)
        self.assertEqual(2, max(s["end"] for s in result.schedule))

    def test_world_event_can_be_a_real_prerequisite(self):
        self.room["intro"] = {"status": "ready", "world_event": "接下来石门升起"}
        result = self.decide([step("gate", ["world_event"], 3), step("cross", ["a", "b"], 4, ["gate"])])
        self.assertEqual(8, result.seconds)

    def test_invalid_estimates_fall_back_without_aborting_generation(self):
        valid = [step("gate", ["a"], 3), step("cross", ["b"], 4, ["gate"])]
        bad = [None, {}, {"steps": []}, {"steps": valid[:1]}, {"steps": valid * 2}]
        for field, values in {
            "seconds": [True, -1, 0, float("inf"), float("nan"), "5", 31],
            "after": [["missing"], ["cross"], [{}], None],
            "actors": [[], ["unknown"], ["a", "a"], [{}]],
            "id": ["", [], "contains spaces"],
            "action": [None, "", "x" * 121],
        }.items():
            for value in values:
                steps = copy.deepcopy(valid)
                steps[0][field] = value
                bad.append({"steps": steps})
        before = copy.deepcopy(self.room)
        for timing in bad:
            with self.subTest(timing=timing):
                result = duration.decide(self.room, timing)
                self.assertEqual("fallback", result.source)
                self.assertIn(result.seconds, (6, 8, 10))
        self.assertEqual(before, self.room)

    def test_missing_event_timing_is_not_silently_ignored(self):
        self.room["intro"] = {"status": "ready", "world_event": "接下来石门升起"}
        result = self.decide([step("both", ["a", "b"], 2)])
        self.assertEqual("fallback", result.source)
        self.assertIn("incomplete_actors", result.reasons)

    def test_configured_ceiling_is_honest_about_overrun_and_preserves_schedule(self):
        result = self.decide([step("travel", ["a", "b"], 14)])
        self.assertEqual(10, result.seconds)
        self.assertTrue(result.limited)
        self.assertEqual(14, result.schedule[0]["end"])
        with patch.object(config, "COOP_ACTION_SECONDS", 15):
            result = self.decide([step("travel", ["a", "b"], 14)])
            self.assertEqual(15, result.seconds)
            self.assertFalse(result.limited)

    def test_opening_stays_fixed_and_needs_no_timing(self):
        self.room["opening"] = True
        result = duration.decide(self.room)
        self.assertEqual((6, "opening"), (result.seconds, result.source))
        self.assertEqual(0, result.as_dict()["end_hold_seconds"])

    def test_fallback_distinguishes_long_route_from_three_steps(self):
        for action, expected in (
            ("向前跑三步", 6),
            ("从大殿入口跑到尽头的王座前", 8),
            ("向前跑30米", 10),
            ("拔剑、冲锋、劈砍、翻滚、绕后、再次刺击", 10),
        ):
            self.room["members"][0]["action"] = action
            self.assertEqual(expected, duration.decide(self.room).seconds, action)

    def test_fallback_does_not_penalize_prose_cast_size_or_magic_keywords(self):
        self.room["members"] = [
            {"user_id": str(i), "action": "身穿厚重铠甲的角色" * 20 + "站稳"} for i in range(4)
        ]
        self.room["intro"] = {"status": "ready", "world_event": "接下来微风吹动旗帜"}
        self.assertEqual(6, duration.decide(self.room).seconds)
        self.room["members"][0]["action"] = "吟唱咒语，举起法杖释放光束"
        self.assertEqual(6, duration.decide(self.room).seconds)

    def test_short_speech_versus_actual_long_dialogue(self):
        self.room["members"][0]["action"] = "大喊“向左撤退”，随后举剑"
        self.assertEqual(6, duration.decide(self.room).seconds)
        self.room["members"][0]["action"] = "开口说“" + "大家小心" * 12 + "”"
        self.assertEqual(10, duration.decide(self.room).seconds)


if __name__ == "__main__":
    unittest.main()
