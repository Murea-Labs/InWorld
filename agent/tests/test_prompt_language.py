"""Regression tests for the English media-prompt boundary."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from murea_agent.director import agent_schema, prompt_language
from murea_agent.shot import video_prompt
from murea_agent.providers import llm


class PromptLanguageTest(unittest.TestCase):
    def test_dialogue_is_allowed_but_surrounding_caption_is_repaired(self):
        fields = {
            "detailed_description": "<Subject 1> raises a hand. (S1) <d>[Chinese] 你好</d>",
            "overall_soundscape": "安静的房间环境音",
        }
        with patch.object(
            llm,
            "chat",
            return_value={
                "detailed_description": "<Subject 1> raises a hand. (S1) <d>[Chinese] 你好</d>",
                "overall_soundscape": "Quiet room ambience.",
            },
        ) as chat:
            got = prompt_language.ensure_english_fields(fields, allow_dialogue=True)
        self.assertEqual(1, chat.call_count)
        self.assertIn("你好", got["detailed_description"])
        self.assertNotRegex(got["overall_soundscape"], r"[\u3400-\u9fff]")

    def test_failed_repair_uses_english_fallback(self):
        with patch.object(llm, "chat", side_effect=RuntimeError("offline")):
            got = prompt_language.ensure_english_fields(
                {"shot": "中文动作描述"}, fallbacks={"shot": "A safe English shot."}
            )
        self.assertEqual("A safe English shot.", got["shot"])

    def test_failed_repair_preserves_chinese_only_inside_dialogue(self):
        original = "中文动作 <Subject 1> speaks: <d>[Chinese] 你好</d>"
        with patch.object(llm, "chat", side_effect=RuntimeError("offline")):
            got = prompt_language.ensure_english_fields(
                {"shot": original}, allow_dialogue=True, fallbacks={"shot": "A safe English shot."}
            )
        self.assertIn("<Subject 1>", got["shot"])
        self.assertIn("<d>[Chinese] 你好</d>", got["shot"])
        self.assertFalse(prompt_language.contains_cjk(got["shot"], allow_dialogue=True))

    def test_repair_may_not_drop_subject_or_dialogue_tokens(self):
        original = "中文动作 <Subject 1> speaks: <d>[Chinese] 你好</d>"
        with patch.object(llm, "chat", return_value={"shot": "The character speaks."}):
            got = prompt_language.ensure_english_fields(
                {"shot": original}, allow_dialogue=True, fallbacks={"shot": "A safe English shot."}
            )
        self.assertIn("<Subject 1>", got["shot"])
        self.assertIn("<d>[Chinese] 你好</d>", got["shot"])

    def test_video_guard_keeps_all_sections_free_of_cjk(self):
        six = {key: "中文污染" for key in agent_schema.SECTION_ORDER}
        six["detailed_description"] = "<Subject 1> acts."
        with patch.object(
            llm, "chat", return_value={key: ("English " + key) for key in agent_schema.SECTION_ORDER}
        ):
            got = video_prompt._ensure_prompt_english(six)
        for key, value in got.items():
            self.assertNotRegex(value, r"[\u3400-\u9fff]", key)


if __name__ == "__main__":
    unittest.main()
