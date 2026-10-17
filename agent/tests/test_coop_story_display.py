"""Display translation stays outside adopted plans and generation state."""

import copy
from contextlib import contextmanager
import unittest
from unittest.mock import patch

from fastapi import HTTPException

from murea_agent.coop import story_display as display


class StoryDisplayTest(unittest.TestCase):
    def setUp(self):
        display._translate.cache_clear()
        self.room = {
            "code": "ABCDEF123456",
            "world_id": "w-display",
            "round": 5,
            "members": [{"user_id": "u1"}],
            "revision": 4,
            "story_outline": {
                "overall_goal": "共同击败龙王",
                "stages": [
                    {
                        "id": "break_armor",
                        "title": "协同破防",
                        "goal": "让龙鳞出现裂痕",
                        "completion_evidence": "鳞甲出现裂痕",
                    }
                ],
            },
            "story_progress": {
                "stages": {"break_armor": {"status": "active", "evidence": "Aren raises his sword."}}
            },
            "intro": {
                "short_term_plan": {
                    "stage_id": "break_armor",
                    "purpose": "breakthrough",
                    "change_reason": "Aren raises his sword.",
                    "steps": [
                        {
                            "id": "magic_cover",
                            "goal": "Mira distracts the dragon.",
                            "cooperation": "Aren从石柱后冲出",
                            "completion_evidence": "Dragon turns toward Mira.",
                        }
                    ],
                }
            },
            "recommendation_prefetch": {"status": "ready", "signature": "prepared-video"},
        }

        @contextmanager
        def edit(code, *, persist):
            self.assertEqual(self.room["code"], code)
            self.assertFalse(persist)
            yield self.room

        self.editor = patch.object(display.coop, "_edit", side_effect=edit)
        self.editor.start()
        self.addCleanup(self.editor.stop)
        self.addCleanup(display._translate.cache_clear)

    def test_collects_display_prose_only_and_deduplicates(self):
        self.assertEqual(
            ("Aren raises his sword.", "Mira distracts the dragon.", "Dragon turns toward Mira."),
            display.display_texts(self.room),
        )
        self.assertFalse(display.needs_chinese("Aren向左移动"))

    def test_translates_once_preserving_adopted_conditions_and_ready_video(self):
        before = copy.deepcopy(self.room)
        with patch.object(
            display.coop_trace,
            "chat",
            return_value={"0": "Aren举起长剑。", "1": "Mira牵制龙王。", "2": "龙王转向Mira。"},
        ) as chat:
            result = display.localized_display(self.room["code"], "u1")
            self.assertEqual(result, display.localized_display(self.room["code"], "u1"))
            self.assertEqual(1, chat.call_count)
            self.assertEqual("Mira牵制龙王。", result["translations"]["Mira distracts the dragon."])
        self.assertEqual(before, self.room)

    def test_authorization_runs_even_when_translation_cached(self):
        with patch.object(
            display.coop_trace,
            "chat",
            return_value={"0": "Aren举起剑。", "1": "Mira牵制龙王。", "2": "龙王转向Mira。"},
        ) as chat:
            display.localized_display(self.room["code"], "u1")
            with self.assertRaises(HTTPException) as error:
                display.localized_display(self.room["code"], "outsider")
            self.assertEqual(403, error.exception.status_code)
            self.assertEqual(1, chat.call_count)

    def test_chinese_needs_no_model_request(self):
        with patch.object(display.coop_trace, "chat") as chat:
            self.assertEqual({}, display._translate((), "model"))
            chat.assert_not_called()

    def test_translation_failure_is_retryable_without_modifying_world(self):
        before = copy.deepcopy(self.room)
        for response in ({"0": "Still English"}, {"0": "中文"}, RuntimeError("unavailable")):
            with patch.object(
                display.coop_trace,
                "chat",
                side_effect=response if isinstance(response, Exception) else None,
                return_value=response,
            ):
                with self.assertRaises(HTTPException) as error:
                    display.localized_display(self.room["code"], "u1")
                self.assertEqual(503, error.exception.status_code)
        self.assertEqual(before, self.room)
        self.assertEqual(0, display._translate.cache_info().currsize)

    def test_busy_translation_does_not_queue_more_inference(self):
        with display._LOCK, patch.object(display.coop_trace, "chat") as chat:
            with self.assertRaises(HTTPException):
                display.localized_display(self.room["code"], "u1")
            chat.assert_not_called()


if __name__ == "__main__":
    unittest.main()
