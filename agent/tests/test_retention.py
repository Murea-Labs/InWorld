"""`retention_analysis` is built by the server and no longer asked from the model.

Not "dropping a field": the field **never held any judgement by the model**:

    <Subject 1> (appears in [Shot 1] (full body)): fully_preserved - <description>

which Subjects, which image each  → `pictures` is ordered by the server visibility
→ a synonym of `motion_plan.framing` retention_level                   → always fully_preserved in
this design (references exist to be kept) the closing description           → the locked English
definition

Rewriting it cost ~126 tokens ≈ 0.4 s per beat, for rephrasings of the same sentence ("character
attributes match reference image" / "consistent character appearance").
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from murea_agent.director import agent_schema
from murea_agent.shot import definitions
from murea_agent.shot import video_prompt
from murea_agent.memory.world import World

# ⚠️ This is the **real format**: the type phrase first, `shown in <Picture N> as <view>` in the middle, and the real
# appearance **after the semicolon**. A first fake definition without a semicolon made the front
# half "pass"; only a real beat showed it produced "the single character sheet".
DEF = (
    "<Subject 1> is the single character sheet shown in <Picture 1> as a single "
    "full-body view; a young man in his 20s with short dark hair, wearing a "
    "damp grey overcoat."
)
PLACE_DEF = (
    "<Subject 2> is the single wide scene view shown in <Picture 2> as a "
    "single wide scene view; a rainy old railway platform with warm "
    "tungsten lamps and wet concrete."
)


def _world() -> tuple[World, str, str]:
    world = World(title="t")
    sid = world.add_subject("林默", "a.png", brief="三十岁男性，深色旧夹克")
    world.subjects[sid]["definition"] = DEF
    iid = world.add_subject("旧车票", "t.png", kind="物件")
    world.subjects[iid]["definition"] = "<Subject 3> is a yellowed paper train ticket"
    world.places["火车站站台"] = {"ref": "p.png", "desc": "", "first_seen": "", "definition": PLACE_DEF}
    return world, sid, iid


class RetentionTest(unittest.TestCase):
    def _build(self, framing: str) -> str:
        world, sid, iid = _world()
        pics = [
            {"n": 1, "kind": "character", "id": sid, "name": "林默"},
            {"n": 2, "kind": "scene", "id": "火车站站台", "name": "火车站站台"},
            {"n": 3, "kind": "character", "id": iid, "name": "旧车票"},
        ]
        return video_prompt._build_retention(pics, world, {"framing": framing})

    def test_one_line_per_picture_in_order(self) -> None:
        lines = self._build("wide").splitlines()
        self.assertEqual(3, len(lines))
        for i, line in enumerate(lines, start=1):
            self.assertTrue(line.startswith(f"<Subject {i}> (appears in [Shot 1]"))
            self.assertIn("fully_preserved", line)

    def test_framing_maps_to_visibility(self) -> None:
        self.assertIn("(full body)", self._build("wide"))
        self.assertIn("(most of the body)", self._build("medium-wide"))
        self.assertIn("(waist-up)", self._build("medium"))

    def test_the_scene_row_is_always_a_wide_scene_view(self) -> None:
        """A rule fixed in the prompt: the place line's visibility must be wide scene view."""
        self.assertIn("(wide scene view)", self._build("medium"))

    def test_foreground_background_is_excluded_in_retention_only(self) -> None:
        """Per-shot retention / differences go into retention, without polluting subject definitions and main fields."""
        out = self._build("wide")
        character, scene, item = out.splitlines()
        for line in (character, item):
            self.assertIn("reference backdrop", line)
            self.assertIn("replaced by the current story scene", line)
        self.assertNotIn("reference backdrop", scene)

    def test_long_visible_description_is_not_cut_at_ninety_characters(self) -> None:
        """The stable appearance enters retention complete, never leaving fragments like ``with a.``."""
        long_definition = (
            "<Subject 1> is the young male traveler in a heavy oiled leather "
            "trench coat with brass fasteners, charcoal trousers, weathered boots, "
            "short black hair, a narrow scar above his eyebrow, and a stern expression "
            "shown in <Picture 1> as a single full-body view."
        )
        got = definitions._short_definition({"definition": long_definition})
        self.assertGreater(len(got), 90)
        self.assertIn("narrow scar above his eyebrow", got)
        self.assertIn("stern expression", got)

    def test_items_are_whole_object(self) -> None:
        self.assertIn("(whole object)", self._build("wide"))

    def test_it_is_english_only(self) -> None:
        """⚠️ The six sections are English (the video model was trained on English captions). Chinese would skew the format,
        so the description can only use the locked English definition, never the Chinese brief.
        """
        out = self._build("wide")
        self.assertFalse(any("一" <= ch <= "鿿" for ch in out), out)

    def test_all_three_real_definition_shapes_yield_the_appearance(self) -> None:
        """⚠️ Definitions come in at least **three** formats, each seen in real runs:

            A  is <full appearance> shown in <Picture N> as <view>; <pose>
            B  is the single character sheet shown in <Picture N> as <view>; <full appearance>
            C  is <phrase> shown in <Picture N> as <view>, <full appearance>

        Taking the back half gave "the single character sheet"; taking the front half gave "standing
        upright"; taking the longer half gave "the young man" on C. Formats cannot all be guessed,
        so the middle clause is cut out entirely.
        """
        cases = {
            "<Subject 1> is the young Asian male traveler in his 20s, short dark "
            "hair shown in <Picture 1> as a full-body view; standing upright.": "young Asian male traveler",
            "<Subject 1> is the single character sheet shown in <Picture 1> as a "
            "full-body view; a young man with a damp grey overcoat.": "young man with a damp grey overcoat",
            "<Subject 1> is the young man shown in <Picture 1> as a full-body "
            "view, wearing a dark waterproof trench coat.": "dark waterproof trench coat",
            "<Subject 3> is a yellowed paper train ticket with faded printing": "yellowed paper train ticket",
        }
        for definition, want in cases.items():
            with self.subTest(shape=definition[:40]):
                got = definitions._short_definition({"definition": definition})
                self.assertIn(want, got)
                self.assertNotIn("<Picture", got)
                self.assertNotIn("character sheet", got)

    def test_both_real_definition_shapes_yield_the_appearance(self) -> None:
        """⚠️ Definitions have **two** formats; fixing on either half breaks the other.

            A  is <full appearance> shown in <Picture N> as <view>; <pose>
            B  is the single character sheet shown in <Picture N> as <view>; <full appearance>

        Two real runs hit each once: back half → "the single character sheet"; front half →
        "standing upright". The real appearance is always the longer half.
        """
        a = (
            "<Subject 1> is the young Asian male traveler in his 20s, sharp "
            "resolute eyes, short dark hair shown in <Picture 1> as a single "
            "full-body view; standing upright on a rainy platform."
        )
        b = (
            "<Subject 1> is the single character sheet shown in <Picture 1> as a "
            "single full-body view; a young man in his 20s with short dark hair."
        )
        self.assertIn("young Asian male traveler", definitions._short_definition({"definition": a}))
        self.assertNotIn("standing upright", definitions._short_definition({"definition": a}))
        self.assertIn("young man in his 20s", definitions._short_definition({"definition": b}))
        self.assertNotIn("character sheet", definitions._short_definition({"definition": b}))

    def test_the_real_description_comes_from_after_the_semicolon(self) -> None:
        """The format is `<type phrase> shown in <Picture N> as <view>; <real description>`.

        The front half would produce "the single character sheet" or "the scene reference", telling
        the video model to preserve the character sheet instead of what this person looks like.
        """
        out = self._build("wide")
        self.assertIn("a young man in his 20s", out)
        self.assertIn("rainy old railway platform", out)
        self.assertNotIn("character sheet", out)
        self.assertNotIn("wide scene view): fully_preserved - the single", out)

    def test_words_are_not_cut_in_half(self) -> None:
        out = self._build("wide")
        for line in out.splitlines():
            self.assertFalse(line.rstrip(".").endswith(" "), line)

    def test_the_model_is_no_longer_asked_for_it(self) -> None:
        self.assertNotIn("retention_analysis", agent_schema._agent_schema()["properties"])
        self.assertNotIn("retention_analysis", agent_schema.REWRITER_SCHEMA["properties"])
        # But it must still be in the prompt order, or the video model receives only five sections
        self.assertIn("retention_analysis", agent_schema.SECTION_ORDER)


class MergedStillHitsTest(unittest.TestCase):
    """After moving a section to the server, the merged path must still hit.

    ⚠️ Real failure: `_six_from_merged` checked `all(k in d for k in SECTION_ORDER)`, and SECTION_ORDER still had
    retention_analysis, which the model no longer writes, so **every beat** looked incomplete, the
    merged path was void, and each beat paid 2–3 s for a separate rewriter call: losing 2–3 s to
    save 0.4 s.

    The signal was visible: three real runs had rewrite times of 3.26 / 3.08 / 3.29 s, never a hit.
    This test makes it impossible to miss next time.
    """

    def test_a_model_output_without_server_sections_is_accepted(self) -> None:
        d = {k: "x" for k in agent_schema._MODEL_SECTIONS}
        self.assertIsNotNone(video_prompt._six_from_merged(d))

    def test_an_empty_shot_is_still_rejected(self) -> None:
        """detailed_description is the one section that cannot be missing: assemble sends only it when nothing else exists."""
        d = {k: "x" for k in agent_schema._MODEL_SECTIONS}
        d["detailed_description"] = "   "
        self.assertIsNone(video_prompt._six_from_merged(d))

    def test_every_server_section_is_filled_in_before_assemble(self) -> None:
        """Sections built by the server must not be missing, or the video model receives five sections."""
        for key in agent_schema._SERVER_SECTIONS:
            self.assertIn(key, agent_schema.SECTION_ORDER)
            self.assertNotIn(key, agent_schema._MODEL_SECTIONS)


if __name__ == "__main__":
    unittest.main()
