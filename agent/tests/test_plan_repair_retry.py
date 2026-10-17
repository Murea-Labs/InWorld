#!/usr/bin/env python3
"""A misordered action ledger gets a few attempts instead of failing the beat on the first miss."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from murea_agent import config
from murea_agent.providers.video import murea  # noqa: E402
from murea_agent.director import director  # noqa: E402
from murea_agent.guards import replan  # noqa: E402
from murea_agent import util
from murea_agent.memory.world import World

# The same event split into two steps: the semantic check rejects it.
BROKEN = {
    "events": [
        {"id": 1, "description": "架起同伴往前走", "visible_result": "两人一起向前", "estimated_seconds": 8}
    ],
    "steps": [{"kind": "motion", "event_ids": [1]}, {"kind": "motion", "event_ids": [1]}],
}
GOOD = {
    "events": [
        {"id": 1, "description": "架起同伴往前走", "visible_result": "两人一起向前", "estimated_seconds": 8}
    ],
    "steps": [{"kind": "motion", "event_ids": [1]}],
}


class PlanRepairRetryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.world = World(video_model="murea")
        for name, value in (
            ("VIDEO_MAX_SECONDS", 15),
            ("VIDEO_MIN_SECONDS", 10),
            ("GM_PLAN_REPAIR_TRIES", 3),
        ):
            patcher = patch.object(config, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        # The full Murea model, whose clip length is the config above.
        full = patch.object(murea, "backend", return_value="murea")
        full.start()
        self.addCleanup(full.stop)

    def _run(self, replies):
        with (
            patch.object(director, "_decide", side_effect=replies) as decide,
            patch.object(replan, "_prepared_motion_errors", return_value=[]),
        ):
            plan, repaired = replan._prepare_action_plan(
                self.world,
                "架着他往前走",
                {"action_sequence": BROKEN, "duration_seconds": 12},
                lambda _: None,
                duration=None,
                asset_mentions=None,
                clock=util._Clock(),
            )
        return plan, repaired, decide

    def _plan(self, sequence, replies=()):
        with (
            patch.object(director, "_decide", side_effect=list(replies)) as decide,
            patch.object(replan, "_prepared_motion_errors", return_value=[]),
            patch.object(replan, "_single_event_needs_density_audit", return_value=False),
        ):
            plan, _ = replan._prepare_action_plan(
                self.world,
                "push the boards apart and duck through",
                {"action_sequence": sequence},
                lambda _: None,
                duration=None,
                asset_mentions=None,
                clock=util._Clock(),
            )
        return plan, decide

    @staticmethod
    def _real(seconds: tuple[float, float], *, both_in_step: bool = False) -> dict:
        # Shape of a real DeepSeek reply: two events, one step that films both, which references
        # only the first (or both, though a step may hold one id).
        return {
            "events": [
                {"id": 1, "description": "You push the boards apart", "visible_result": "A gap opens",
                 "estimated_seconds": seconds[0]},
                {"id": 2, "description": "You duck through the gap", "visible_result": "You crouch inside",
                 "estimated_seconds": seconds[1]},
            ],
            "steps": [
                {"kind": "motion", "instruction": "Push the boards apart and duck through the gap.",
                 "event_ids": [1, 2] if both_in_step else [1]},
            ],
        }  # fmt: skip

    def test_an_unreferenced_event_is_merged_without_the_model(self):
        for both in (False, True):
            with self.subTest(both_in_step=both):
                plan, decide = self._plan(self._real((3, 4), both_in_step=both))
                decide.assert_not_called()
                sequence = plan["action_sequence"]
                self.assertEqual([], replan._action_sequence_semantic_errors(sequence, 15))
                self.assertEqual(7, sequence["events"][0]["estimated_seconds"])
                self.assertEqual("You crouch inside", sequence["events"][0]["visible_result"])
                # What is filmed is untouched.
                self.assertEqual(
                    "Push the boards apart and duck through the gap.", sequence["steps"][0]["instruction"]
                )

    def test_a_merged_step_that_is_over_budget_is_replanned_with_its_real_cause(self):
        fixed = {
            "events": [
                {"id": 1, "description": "push", "visible_result": "gap", "estimated_seconds": 8},
                {"id": 2, "description": "duck", "visible_result": "inside", "estimated_seconds": 8},
            ],
            "steps": [{"kind": "motion", "event_ids": [1]}, {"kind": "motion", "event_ids": [2]}],
        }
        plan, decide = self._plan(self._real((8, 8)), [{"action_sequence": fixed}])
        self.assertEqual(1, decide.call_count)
        hint = decide.call_args.args[1]
        self.assertIn("needs about 16s", hint)
        self.assertNotIn("may share a step", hint)
        self.assertEqual(2, len(plan["action_sequence"]["steps"]))

    def test_a_replan_with_an_incomplete_ledger_is_not_replanned_again(self):
        # Real beat: the density re-plan fixed the action but left an event unreferenced, which
        # cost one more full director call.
        over = self._real((8, 8))
        plan, decide = self._plan(over, [{"action_sequence": self._real((3, 4))}, {"action_sequence": GOOD}])
        self.assertEqual(1, decide.call_count)
        self.assertEqual(7, plan["action_sequence"]["events"][0]["estimated_seconds"])

    def test_merge_leaves_a_misordered_ledger_to_the_model(self):
        sequence = self._real((3, 4))
        sequence["steps"] = [{"kind": "motion", "event_ids": [2]}, {"kind": "motion", "event_ids": [1]}]
        self.assertIsNone(replan._merge_unassigned_events(sequence))
        self.assertIsNone(replan._merge_unassigned_events(BROKEN))

    def test_the_first_retry_is_not_the_only_one(self):
        """The first reorder is still wrong; the second is right, so it must continue."""
        plan, repaired, decide = self._run([{"action_sequence": BROKEN}, {"action_sequence": GOOD}])
        self.assertEqual(2, decide.call_count)
        self.assertTrue(repaired)
        self.assertEqual(GOOD["steps"], plan["action_sequence"]["steps"])

    def test_a_ledger_that_stays_wrong_is_rebuilt_from_the_steps(self):
        """Three failed re-plans no longer fail the beat when every step itself is complete."""
        plan, repaired, decide = self._run([{"action_sequence": BROKEN}] * 3)
        self.assertEqual(3, decide.call_count)
        sequence = plan["action_sequence"]
        self.assertEqual([], replan._action_sequence_semantic_errors(sequence, 15))
        self.assertEqual(
            [[1], [2]][: len(sequence["steps"])], [step["event_ids"] for step in sequence["steps"]]
        )

    def test_it_still_gives_up_when_there_is_nothing_to_film(self):
        empty = {"events": [], "steps": [{"kind": "motion", "event_ids": [5]}]}
        with self.assertRaisesRegex(ValueError, "after 3 attempts"):
            self._run([{"action_sequence": empty}] * 3)

    def test_a_plan_that_is_already_valid_is_never_replanned(self):
        with (
            patch.object(director, "_decide") as decide,
            patch.object(replan, "_prepared_motion_errors", return_value=[]),
        ):
            plan, repaired = replan._prepare_action_plan(
                self.world,
                "架着他往前走",
                {"action_sequence": GOOD, "duration_seconds": 12},
                lambda _: None,
                duration=None,
                asset_mentions=None,
                clock=util._Clock(),
            )
        decide.assert_not_called()
        self.assertFalse(repaired)
        self.assertEqual(GOOD["steps"], plan["action_sequence"]["steps"])


if __name__ == "__main__":
    unittest.main()
