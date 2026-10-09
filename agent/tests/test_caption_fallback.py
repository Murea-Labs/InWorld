#!/usr/bin/env python3
"""Narration fallback: prefer the director's own text over the relation template."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from murea_agent.director import narration


def _sequence(instruction: str) -> dict:
    return {"steps": [{"instruction": instruction, "place": "北门祭庭", "relation_change": "configuration"}]}


MOTION = {
    "actors": ["莱拉"],
    "relation_change": "configuration",
    "target": "月纹印",
    "interaction_type": "manipulation",
}


class CaptionFallbackTest(unittest.TestCase):
    def test_an_english_instruction_falls_back_to_the_gm_prose_not_a_template(self):
        """Real failure: the narration of the beat that shattered the moon seal became "fully open the movable part, the body still outside"."""
        out = narration._current_segment_reply(
            _sequence("Leyla releases the bowstring into the amber crystal."),
            MOTION,
            directive_reply="莱拉松开弓弦，利箭击碎了月纹印，黑根应声枯萎脱落。",
            player_actor="莱拉",
        )
        self.assertIn("月纹印", out)
        self.assertNotIn("可动部件", out)

    def test_the_template_is_still_there_when_nothing_chinese_is(self):
        out = narration._current_segment_reply(
            _sequence("Leyla releases the bowstring."),
            MOTION,
            directive_reply="",
            player_actor="莱拉",
        )
        self.assertTrue(out)
        self.assertNotIn("可动部件", out)  # the template itself is plain language now

    def test_gm_prose_that_leaks_planning_language_is_still_refused(self):
        out = narration._current_segment_reply(
            _sequence("Leyla releases the bowstring."),
            MOTION,
            directive_reply="本段先放箭，随后分成两段完成撤离。",
            player_actor="莱拉",
        )
        self.assertNotIn("分成", out)
        self.assertNotIn("本段", out)


if __name__ == "__main__":
    unittest.main()
