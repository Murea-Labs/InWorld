"""Story language: English mode must not run the Chinese-only narration fixes."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from murea_agent import config
from murea_agent.shot import dialogue
from murea_agent.director import director  # noqa: E402
from murea_agent import util
from murea_agent.shot import video_prompt
from murea_agent.director import narration
from murea_agent.guards import action_sequence, density_audit

PROMPTS = ROOT / "murea_agent" / "director" / "prompts"


class PromptTemplates(unittest.TestCase):
    def test_language_placeholder_is_the_only_language_switch(self) -> None:
        for name in ("agent.md", "rewriter.md", "video_prompt.md"):
            text = (PROMPTS / name).read_text()
            with self.subTest(prompt=name):
                self.assertNotIn("Chinese", text.replace("{{LANGUAGE}}", ""))
                rendered = text.replace("{{LANGUAGE}}", "English")
                self.assertNotIn(
                    "{{", rendered.replace("{{VOICE_LIBRARY}}", "").replace("{{VIDEO_PROMPT_RULES}}", "")
                )

    def test_loaded_prompt_uses_the_configured_language(self) -> None:
        self.assertIn(config.LANGUAGE_NAME, director.AGENT_SYSTEM)
        self.assertNotIn("{{LANGUAGE}}", director.AGENT_SYSTEM)
        for system in (director.AGENT_SYSTEM, video_prompt.REWRITER_SYSTEM):
            self.assertIn("## Physical action follows motion_plan", system)
            self.assertNotIn("{{", system)


class EnglishMode(unittest.TestCase):
    def setUp(self) -> None:
        for name, value in (("LANGUAGE", "en"), ("LANGUAGE_NAME", "English")):
            p = patch.object(config, name, value)
            p.start()
            self.addCleanup(p.stop)

    def test_english_text_is_presentable_and_chinese_is_not(self) -> None:
        self.assertTrue(util._in_story_language("You push the door open."))
        self.assertFalse(util._in_story_language("你推开了门。"))

    def test_player_narration_is_left_as_written(self) -> None:
        text = "Mira pushes the door open."
        self.assertEqual(text, narration._as_player_narration(text, "Mira", "Mira"))

    def test_segment_reply_is_the_current_step(self) -> None:
        sequence = {"steps": [{"instruction": "You cross the hall to the stairs"}]}
        self.assertEqual(
            "You cross the hall to the stairs.", narration._current_segment_reply(sequence, {}, "", "Mira")
        )

    def test_dialogue_is_tagged_with_the_story_language(self) -> None:
        out = dialogue._lock_dialogue_language({"d": "(S1) <d>Run!</d>"})
        self.assertEqual("(S1) <d>[English] Run!</d>", out["d"])

    def test_progress_messages_follow_the_language(self) -> None:
        self.assertEqual("Planning", util._say("推演", "Planning"))

    def test_padded_options_are_english(self) -> None:
        options = narration._three_options([])
        self.assertEqual(3, len(options))
        self.assertTrue(all(util._in_story_language(o) for o in options))

    def test_continuation_steps_are_not_prefixed_in_chinese(self) -> None:
        steps = narration._sequence_steps_text(
            {
                "steps": [
                    {"instruction": "You walk to the gate", "relation_change": "distance"},
                    {"instruction": "You climb onto the wall", "relation_change": "support"},
                ]
            }
        )
        self.assertEqual(["You climb onto the wall"], steps)

    def test_fallback_step_instruction_is_english(self) -> None:
        self.assertEqual("Step inside", action_sequence._goal_relation_instruction("containment"))

    def test_repair_hints_sent_to_the_director_are_english(self) -> None:
        hint = action_sequence._action_sequence_repair_hint(["step 1 has no events"], 6)
        audit = density_audit._density_audit_repair_hint(
            {
                "events": [
                    {
                        "description": "Mira opens the gate",
                        "visible_result": "the gate is open",
                        "estimated_seconds": 3,
                    }
                ],
                "required_segments": 1,
                "goal_covered": False,
                "coverage_reason": "stops halfway",
            },
            6,
        )
        for text in (hint, audit, density_audit._audit_system()):
            self.assertTrue(util._in_story_language(text), text[:80])

    def test_history_beat_marker_parses_old_and_new_worlds(self) -> None:
        for marker in ("[beat 3 · Harbour · Mira waves]", "[已生成第 3 段 · Harbour · Mira waves]"):
            hit = narration._BEAT_TAIL.search("She waves.\n" + marker)
            self.assertEqual(("3", "Harbour"), hit.groups()[:2])


if __name__ == "__main__":
    unittest.main()
