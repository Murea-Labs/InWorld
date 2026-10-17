#!/usr/bin/env python3
"""Everyone who speaks needs their own reference voice, or the video model reads with whichever voice is present."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from murea_agent.shot import beat_assets
from murea_agent import loop
from murea_agent.memory.world import World


class SixSectionSpeakersTest(unittest.TestCase):
    def test_reads_every_marked_speaker_in_order(self):
        six = {
            "detailed_description": "S1 enters. (S1)<d>[Chinese] 长老！</d> The elder rises. (S3)<d>[Chinese] 伤得很重。</d>"
        }
        self.assertEqual(["S1", "S3"], beat_assets._six_section_speakers(six))

    def test_ignores_numbers_that_are_not_speaking(self):
        six = {
            "detailed_description": "(S2) stands near <Subject 3>. (S1)<d>[Chinese] 走。</d>",
            "summary": "(S4) glows.",
        }
        self.assertEqual(["S1"], beat_assets._six_section_speakers(six))

    def test_dedupes(self):
        six = {"detailed_description": "(S1)<d>a</d> (S1)<d>b</d>"}
        self.assertEqual(["S1"], beat_assets._six_section_speakers(six))


class AudioRefsTest(unittest.TestCase):
    def setUp(self):
        self.world = World(video_model="murea")
        self.ids = ["S1", "S2", "S3"]
        for sid, name in zip(self.ids, ("莱拉", "艾拉尔", "长老")):
            self.world.subjects[sid] = {
                "name": name,
                "kind": "人物",
                "voice": {"id": f"v-{sid}", "name": name, "ref": f"library/{sid}.wav"},
            }

    def _refs(self, dialogue, also):
        with (
            patch.object(loop.voices, "heal", side_effect=lambda v: v),
            patch.object(loop.voices, "path", side_effect=lambda r: Path("/tmp") / r),
            patch.object(loop.voices, "label", side_effect=lambda v: v.get("name", "")),
        ):
            return beat_assets._audio_refs(self.world, self.ids, dialogue, also_speaking=also)

    def test_a_speaker_only_the_six_section_knows_about_still_gets_a_voice(self):
        """Real failure: the elder spoke, but only another character's reference voice was sent."""
        _, bind = self._refs("艾拉尔：撑不住了", ["S2", "S3"])
        self.assertEqual(["S2", "S3"], [b["sid"] for b in bind])

    def test_nobody_speaks_means_nothing_is_sent(self):
        self.assertEqual(([], []), self._refs("", []))

    def test_unknown_ids_are_ignored(self):
        _, bind = self._refs("莱拉：走", ["S9"])
        self.assertEqual(["S1"], [b["sid"] for b in bind])


if __name__ == "__main__":
    unittest.main()
