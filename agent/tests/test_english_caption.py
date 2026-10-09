"""English captions address the player as "you", like the Chinese ones."""

import unittest
from unittest.mock import patch

from murea_agent import config
from murea_agent.director.narration import _current_segment_reply, _english_second_person

SEQUENCE = {
    "steps": [
        {"instruction": "The player pushes the shop door open and steps in from the fog."},
        {"instruction": "Mara Quill looks up from the ledger."},
    ]
}


class EnglishCaptionTest(unittest.TestCase):
    def test_the_directors_narration_wins_over_the_step_instruction(self):
        reply = "You lean into the bookshop door and it gives; lamplight spills out."
        with patch.object(config, "LANGUAGE", "en"):
            self.assertEqual(reply, _current_segment_reply(SEQUENCE, {}, reply, "Ivo Marsh"))

    def test_the_instruction_fallback_is_turned_into_second_person(self):
        with patch.object(config, "LANGUAGE", "en"):
            out = _current_segment_reply(SEQUENCE, {}, "", "Ivo Marsh")
        self.assertEqual("You push the shop door open and step in from the fog.", out)

    def test_only_the_player_is_rewritten(self):
        self.assertEqual(
            "You cross to the counter and ask Mara about the lighthouse.",
            _english_second_person(
                "Ivo Marsh crosses to the counter and asks Mara about the lighthouse.", "Ivo Marsh"
            ),
        )
        self.assertEqual(
            "Mara Quill looks up at you.",
            _english_second_person("Mara Quill looks up at the player.", "Ivo Marsh"),
        )
        self.assertEqual(
            "You are soaked; your coat drips.",
            _english_second_person("The player is soaked; the player's coat drips.", ""),
        )
        # Nouns after "and" are not verbs.
        self.assertEqual(
            "You watch the floor and boots.",
            _english_second_person("The player watches the floor and boots.", ""),
        )


if __name__ == "__main__":
    unittest.main()
