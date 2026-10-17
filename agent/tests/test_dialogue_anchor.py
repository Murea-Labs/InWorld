#!/usr/bin/env python3
"""Speaker anchors must not contradict themselves."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from murea_agent.shot import dialogue
from murea_agent.memory.world import World


class AnchorAlignTest(unittest.TestCase):
    def test_a_self_contradicting_anchor_is_pulled_onto_the_tag(self):
        """Real failure: <Subject 3> speaks on-screen as S1: (S3)<d>, one line naming two people."""
        six = {
            "detailed_description": "The hall is quiet. <Subject 3> speaks on-screen as S1: (S3)<d>[Chinese] 伤得很重。</d>"
        }
        out = dialogue._align_dialogue_anchor(six)
        self.assertIn("<Subject 3> speaks on-screen as S3: (S3)<d>", out["detailed_description"])
        self.assertNotIn("as S1", out["detailed_description"])

    def test_the_line_itself_is_untouched(self):
        six = {
            "detailed_description": "<Subject 2> speaks on-screen as S1: (S2)<d>[Chinese] 抓紧我，我们走！</d>"
        }
        out = dialogue._align_dialogue_anchor(six)
        self.assertIn("[Chinese] 抓紧我，我们走！", out["detailed_description"])

    def test_off_screen_anchors_align_too(self):
        six = {
            "detailed_description": "an off-screen voice identified as S1 says: (S2)<d>[Chinese] 快走。</d>"
        }
        out = dialogue._align_dialogue_anchor(six)
        self.assertIn("identified as S2 says: (S2)<d>", out["detailed_description"])

    def test_unresolvable_speaker_no_longer_leaves_the_prompt_as_is(self):
        """When the director's dialogue speaker cannot be resolved, a contradictory anchor must not be sent as is either."""
        six = {"detailed_description": "<Subject 3> speaks on-screen as S1: (S3)<d>[Chinese] 让我来。</d>"}
        out = dialogue._lock_dialogue_speakers(six, "", World(video_model="murea"), [])
        self.assertIn("as S3", out["detailed_description"])


if __name__ == "__main__":
    unittest.main()
