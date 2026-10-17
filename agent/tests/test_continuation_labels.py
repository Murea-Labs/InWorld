#!/usr/bin/env python3
"""Who is who in a continuation prompt, which carries no reference images."""

from __future__ import annotations

import unittest

from murea_agent.shot.video_prompt import _plain_continuation_from_merged


def _continuation(*names: str) -> str:
    pictures = [{"n": i + 1, "kind": "subject", "name": name, "id": f"s{i}"} for i, name in enumerate(names)]
    pictures.append({"n": len(names) + 1, "kind": "scene", "name": "雨夜旧城窄巷", "id": "p"})
    six = {
        "detailed_description": " ".join(f"<Subject {i + 1}> moves." for i in range(len(names))),
        "subject_definitions": " ".join(
            f"<Subject {i + 1}> is person number {i + 1}." for i in range(len(names))
        ),
        "overall_soundscape": "rain",
        "non_diegetic_music": "",
    }
    return _plain_continuation_from_merged(six, pictures)["detailed_description"]


class ContinuationLabelTest(unittest.TestCase):
    def test_characters_without_an_english_name_are_told_apart(self) -> None:
        # Every built-in co-op name and every Chinese name used to become "the established
        # character", so a two-person round read as one person doing everything.
        text = _continuation("星野 · 星际游侠", "洛伊 · 森林守卫")
        self.assertNotIn("established character", text)
        self.assertIn("the first character moves", text)
        self.assertIn("The second character: person number 2.", text)

    def test_the_given_name_of_a_built_in_english_name_is_kept(self) -> None:
        text = _continuation("Hoshi · Star Ranger", "Aren · Axe Warrior · AI")
        self.assertIn("Hoshi moves", text)
        self.assertIn("Aren: person number 2.", text)
        self.assertNotIn("·", text)

    def test_a_single_unnamed_character_keeps_the_plain_label(self) -> None:
        text = _continuation("Mira", "星野")
        self.assertIn("Mira moves", text)
        self.assertIn("the established character moves", text)


if __name__ == "__main__":
    unittest.main()
