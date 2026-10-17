"""Length limits and carry text follow the story language."""

import unittest
from unittest.mock import patch

from murea_agent import config
from murea_agent.util import cap, carrying, drop_carried


class LanguageLimitTest(unittest.TestCase):
    def test_english_options_are_not_cut_mid_sentence(self):
        option = "Walk up to the bar and ask the innkeeper about the boats that did not come back"
        with patch.object(config, "LANGUAGE", "en"):
            self.assertEqual(option, cap(option, 40))
            short = cap("word " * 40, 10)
            self.assertLessEqual(len(short), 25)
            self.assertFalse(short.endswith(" "))
            self.assertTrue(all(w == "word" for w in short.split()))

    def test_chinese_keeps_the_character_budget(self):
        with patch.object(config, "LANGUAGE", "zh"):
            self.assertEqual("一" * 40, cap("一" * 50, 40))

    def test_carry_is_written_in_the_story_language(self):
        with patch.object(config, "LANGUAGE", "en"):
            carry = carrying("", "lantern")
            carry = carrying(carry, "brass key")
            self.assertEqual("holding lantern; holding brass key", carry)
            self.assertEqual("holding brass key", drop_carried(carry, "lantern"))
        with patch.object(config, "LANGUAGE", "zh"):
            self.assertEqual("手里拿着灯笼；手里拿着钥匙", carrying("手里拿着灯笼", "钥匙"))
            self.assertEqual("手里拿着钥匙", drop_carried("手里拿着灯笼；手里拿着钥匙", "灯笼"))


if __name__ == "__main__":
    unittest.main()
