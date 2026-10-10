"""Once entity wording is locked, do not ask the merged call for subject_definitions; the server overwrites the whole section.

The mechanism existed (`REWRITER_SCHEMA_NO_DEFS`, about 30% fewer output tokens) but only on the
separate rewriter path; the merged path still wrote it every beat.

Measured on the same world and inputs: one model wrote 2 characters (it learned to skip it), a local
model wrote 238 characters ≈ 133 tokens ≈ 0.31 s wasted per beat.
"""

from __future__ import annotations

import inspect
import copy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from murea_agent import config
from murea_agent.shot import soundscape
from murea_agent.director import director  # noqa: E402
from murea_agent.director import agent_schema
from murea_agent.shot import video_prompt
from murea_agent import loop
from murea_agent.memory.world import World


class Schema(unittest.TestCase):
    def test_dropping_removes_it_from_both_lists(self) -> None:
        """⚠️ `required` decides generation order and properties decide which fields exist; remove it from both, or the
        model still writes it.
        """
        s = agent_schema._agent_schema(True)
        self.assertNotIn("subject_definitions", s["required"])
        self.assertNotIn("subject_definitions", s["properties"])

    def test_default_keeps_it(self) -> None:
        """On a new world's first beat nothing is locked, so the model must write it."""
        s = agent_schema._agent_schema()
        self.assertIn("subject_definitions", s["required"])

    def test_the_rest_of_the_six_section_survives(self) -> None:
        """Removing the wrong thing would make `_six_from_merged` miss the six sections every beat and always fall back to the rewriter."""
        s = agent_schema._agent_schema(True)
        for key in agent_schema._MODEL_SECTIONS:
            if key == "subject_definitions":
                continue
            with self.subTest(key=key):
                self.assertIn(key, s["required"])

    def test_field_order_is_still_consistent(self) -> None:
        """required and properties must have the same order; this guards the generation-order optimization."""
        s = agent_schema._agent_schema(True)
        props = [k for k in s["properties"] if k in set(s["required"])]
        self.assertEqual(props, s["required"])


class Wiring(unittest.TestCase):
    def test_turn_reuses_shot_only_when_selected_assets_are_all_locked(self) -> None:
        class Submitted(BaseException):
            pass

        with tempfile.TemporaryDirectory() as tmp:
            world = World(title="test", here="会议室")
            sid = world.add_subject("阿明", "man.png")
            world.add_place("会议室", "room.png")
            world.subjects[sid]["definition"] = "the man in a blue shirt shown in <Picture 1>."
            world.places["会议室"]["definition"] = "the room with a wooden table shown in <Picture 2>."
            shot = {
                "subject_definitions": "<Subject 1> is someone else shown in <Picture 1>.",
                "summary": "[reference_generation] <Subject 1> looks around inside <Subject 2>.",
                "detailed_description": "[Shot 1] In a wide shot, <Subject 1> looks at the wooden table inside <Subject 2>.",
                "overall_soundscape": "A quiet room tone.",
                "non_diegetic_music": "None.",
            }
            decision = {
                **shot,
                "reply": "你环顾了一下会议室。",
                "mode": "continue",
                "transition": False,
                "place": "会议室",
                "subjects": [sid],
                "beat": "阿明看向会议室的桌子。",
                "dialogue": "",
                "carry": "",
                "new_subjects": [],
                "new_places": [],
                "action_sequence": None,
                "state_updates": {"subjects": [], "places": []},
            }
            for locked in (True, False):
                current = copy.deepcopy(world)
                if not locked:
                    current.places["会议室"].pop("definition")
                with (
                    self.subTest(all_locked=locked),
                    patch.object(config, "REFS", Path(tmp)),
                    patch.object(
                        director,
                        "_decide_early",
                        return_value=(
                            copy.deepcopy(
                                {k: v for k, v in decision.items() if locked or k != "subject_definitions"}
                            ),
                            None,
                        ),
                    ),
                    patch.object(
                        loop,
                        "_rewrite",
                        return_value={
                            **shot,
                            "subject_definitions": shot["subject_definitions"]
                            + " <Subject 2> is "
                            + world.places["会议室"]["definition"],
                        },
                    ) as rewrite,
                    patch.object(loop.video, "generate", side_effect=Submitted) as generate,
                    patch.object(World, "save", side_effect=AssertionError("No world writes")),
                ):
                    with self.assertRaises(Submitted):
                        loop.turn(current, "观察会议室")
                    self.assertEqual(0 if locked else 1, rewrite.call_count)
                    prompt = generate.call_args.args[0]
                    self.assertIn("blue shirt", prompt)
                    self.assertIn("wooden table", prompt)
                    self.assertIn("<Picture 1>", prompt)
                    self.assertIn("<Picture 2>", prompt)
                    self.assertNotIn("someone else", prompt)
                    self.assertEqual(1, prompt.count("<Subject 1> is"))
                    self.assertEqual(1, prompt.count("<Subject 2> is"))

    def test_locked_assets_reuse_merged_shot_without_second_model_call(self) -> None:
        d = {key: "shot content" for key in agent_schema._MODEL_SECTIONS if key != "subject_definitions"}
        self.assertIsNone(video_prompt._six_from_merged(d))
        six = video_prompt._six_from_merged(d, definitions_locked=True)
        self.assertIsNotNone(six)
        self.assertEqual("", six["subject_definitions"])
        self.assertEqual(d["detailed_description"], six["detailed_description"])

    def test_missing_action_or_lock_placeholder_still_requires_rewrite(self) -> None:
        d = {key: "shot content" for key in agent_schema._MODEL_SECTIONS}
        for content in ("", soundscape.SOUNDSCAPE_LOCKED):
            d["detailed_description"] = content
            self.assertIsNone(video_prompt._six_from_merged(d, definitions_locked=True))
        d.pop("detailed_description")
        self.assertIsNone(video_prompt._six_from_merged(d, definitions_locked=True))

    def test_agent_step_decides_by_whether_anything_is_locked(self) -> None:
        src = inspect.getsource(director._agent_step)
        self.assertIn("_agent_schema(bool(locked))", src)


class ComposeMissingDefinitionTest(unittest.TestCase):
    """Entities without settled wording get a definition from their own reference-image prompt, not another rewriter call.

    The criteria differed: `_agent_schema(bool(_locked_definitions(world)))` stops asking for
    `subject_definitions` once **the world** has any locked description, while the server's need is
    computed from **this beat's references**. If one image had no settled wording, the section
    counted as missing and a separate rewriter call followed, about 1 s each time.
    """

    def test_merge_keeps_missing_definitions_and_prefers_locked_identity(self):
        locked = "<Subject 1> is the traveler shown in <Picture 1>."
        generated = (
            "<Subject 1> is someone else shown in <Picture 1>. "
            "<Subject 2> is a room shown in <Picture 2>. "
            "<Subject 2> is a duplicate room shown in <Picture 2>."
        )
        merged = video_prompt._merge_definitions(locked, generated)
        self.assertIn(locked, merged)
        self.assertIn("<Subject 2> is a room shown in <Picture 2>", merged)
        self.assertNotIn("someone else", merged)
        self.assertNotIn("duplicate room", merged)
        self.assertEqual(1, merged.count("<Subject 2> is"))

    def test_merge_leaves_invalid_references_for_existing_validation(self):
        invalid = "<Subject 99> is a room shown in <Picture 99>."
        self.assertIn(invalid, video_prompt._merge_definitions("", invalid))

    def test_english_desc_becomes_a_definition(self):
        got = video_prompt._definition_from_desc(
            {"desc": "single full-body view of a young male wanderer, dark hooded jacket"},
            {"n": 1, "kind": "subject"},
        )
        self.assertIn("<Picture 1>", got)
        self.assertIn("a young male wanderer", got)
        self.assertEqual(1, got.lower().count("full-body view"), "取景说明不能重复两遍")

    def test_scene_gets_the_scene_wording(self):
        got = video_prompt._definition_from_desc(
            {"desc": "establishing wide scene view of an empty alley at night"}, {"n": 2, "kind": "scene"}
        )
        self.assertIn("wide scene view", got)
        self.assertEqual(1, got.lower().count("wide scene view"))
        self.assertIn("an empty alley at night", got)

    def test_new_object_is_not_described_as_a_character(self):
        got = video_prompt._definition_from_desc(
            {"desc": "A white delivery van with blue stripes and a roof rack"}, {"n": 3, "kind": "object"}
        )
        self.assertIn("inanimate object", got)
        self.assertIn("<Picture 3>", got)
        self.assertIn("white delivery van", got)
        self.assertNotIn("character", got)
        self.assertNotIn("full-body", got)

    def test_chinese_desc_is_refused(self):
        """Definitions go verbatim into the English prompt. Chinese ones take the old path rather than being inserted."""
        self.assertEqual(
            "",
            video_prompt._definition_from_desc(
                {"desc": "成年女游侠，白色上衣和棕色轻甲，手持长剑"}, {"n": 3, "kind": "subject"}
            ),
        )
        self.assertEqual("", video_prompt._definition_from_desc({}, {"n": 4, "kind": "subject"}))

    def test_compose_no_longer_reports_missing_when_desc_can_fill_it(self):
        world = World(title="t")
        world.subjects["S1"] = {"name": "旅人", "desc": "single full-body view of a traveler in a cyan robe"}
        world.subjects["S2"] = {"name": "队友", "desc": "成年男战士，深蓝色长发"}
        pics = [
            {"id": "S1", "n": 1, "kind": "subject", "name": "旅人"},
            {"id": "S2", "n": 2, "kind": "subject", "name": "队友"},
        ]
        lines, missing = video_prompt._compose_definitions(world, pics)
        self.assertIn("<Subject 1> is", lines, "英文 desc 的那个应该被拼出来")
        self.assertEqual(["S2"], [p["id"] for p in missing], "中文 desc 的仍然要问模型")

    def test_a_locked_definition_still_wins(self):
        """Settled wording must not be overwritten by desc: that is the key against drift."""
        world = World(title="t")
        world.subjects["S1"] = {
            "name": "旅人",
            "desc": "single full-body view of someone else",
            "definition": "the cyan-robed traveler shown in <Picture 9> as a single full-body view",
        }
        lines, missing = video_prompt._compose_definitions(
            world, [{"id": "S1", "n": 1, "kind": "subject", "name": "旅人"}]
        )
        self.assertIn("cyan-robed traveler", lines)
        self.assertNotIn("someone else", lines)
        self.assertEqual([], missing)


if __name__ == "__main__":
    unittest.main(verbosity=2)
