"""Out-of-range reference numbers are repaired in place instead of sending the whole caption back to the rewriter.

23 of 57 merged-path fallbacks (40%) failed on "references an image number that was not uploaded".
That is not a semantic error but a number the merged call invents without the final numbering. A
full rewrite cost a p90 of 2.36 s, and the rewriter cannot see the numbers either, so a retry is no
guarantee.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from murea_agent.shot import video_prompt

PICTURES = [
    {"n": 1, "kind": "character", "name": "林远山", "id": "s1"},
    {"n": 2, "kind": "scene", "name": "木栈道", "id": "p1"},
]


def _six(detail: str) -> dict[str, str]:
    return {
        "subject_definitions": "<Subject 1> is the lone traveller.",
        "summary": "one shot",
        "retention_analysis": "",
        "detailed_description": detail,
        "overall_soundscape": "wind over water; no dialogue",
        "non_diegetic_music": "N/A",
    }


class ReferenceRepair(unittest.TestCase):
    def test_out_of_range_becomes_a_plain_noun(self) -> None:
        six, notes = video_prompt._repair_reference_tokens(
            _six("<Subject 1> lifts the lantern while <Picture 3> glows."),
            PICTURES,
        )
        detail = six["detailed_description"]
        self.assertNotIn("<Picture 3>", detail)
        self.assertIn("the current visual state", detail)
        self.assertTrue(notes, "修补要说出来，否则日志里看不出发生过什么")
        self.assertIn("Picture 3", notes[0])

    def test_valid_numbers_survive_untouched(self) -> None:
        """Changing a valid number would undo the reference binding entirely; this matters more than the repair itself."""
        six, notes = video_prompt._repair_reference_tokens(
            _six("<Subject 1> walks past <Picture 2>."),
            PICTURES,
        )
        self.assertEqual(six["detailed_description"], "<Subject 1> walks past <Picture 2>.")
        self.assertEqual(notes, [])
        self.assertIn("<Subject 1>", six["subject_definitions"])

    def test_repaired_prompt_then_passes_the_audit(self) -> None:
        """The result must pass the original gate, otherwise the repair is wasted and it falls back anyway."""
        bad = _six("<Subject 1> reaches for <Subject 4>.")
        self.assertTrue(video_prompt._reference_token_error(bad, PICTURES))
        fixed, notes = video_prompt._repair_reference_tokens(bad, PICTURES)
        self.assertTrue(notes)
        self.assertEqual(video_prompt._reference_token_error(fixed, PICTURES), "")

    def test_nothing_to_fix_returns_the_same_object(self) -> None:
        """Nothing to repair, nothing to copy: this path runs every beat."""
        src = _six("<Subject 1> waits.")
        fixed, notes = video_prompt._repair_reference_tokens(src, PICTURES)
        self.assertIs(fixed, src)
        self.assertEqual(notes, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
