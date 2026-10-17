"""Regression tests for deterministic dialogue-speaker binding."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from murea_agent.shot import beat_assets
from murea_agent.shot import dialogue
from murea_agent import loop
from murea_agent.memory.world import World


def _world() -> World:
    world = World(title="未命名世界")
    world.subjects = {
        "S1": {
            "name": "小明",
            "kind": "人物",
            "brief": "二十岁瘦削青年",
        },
        # This bad legacy asset is deliberately present. It shares the location's
        # name but is an object and must never be selected as a dialogue source.
        "S2": {
            "name": "地下酒吧",
            "kind": "物件",
            "brief": "地下酒吧内部",
        },
        # Some generated NPCs have a stable-id-looking display name; their role is
        # only available in brief. This is the shape of the original bug.
        "S3": {
            "name": "S3",
            "kind": "人物",
            "brief": "地下酒吧的酒保，眼神锐利带着刀疤",
        },
    }
    return world


PICTURES = [
    {"n": 1, "kind": "character", "id": "S1", "name": "小明"},
    {"n": 2, "kind": "character", "id": "S3", "name": "S3"},
    {"n": 3, "kind": "scene", "id": "地下酒吧", "name": "地下酒吧"},
]


def _six(detail: str) -> dict[str, str]:
    return {
        "subject_definitions": "",
        "summary": "",
        "retention_analysis": "",
        "detailed_description": detail,
        "overall_soundscape": "N/A",
        "non_diegetic_music": "N/A",
    }


class DialogueSpeakerTest(unittest.TestCase):
    def test_npc_role_in_brief_overrides_wrong_player_token(self) -> None:
        detail = (
            "Early in the shot, <Subject 2> speaks on-screen as S1: "
            "(S1) <d>[Chinese] 黑市的门票可不便宜，你想找什么路子？</d>."
        )

        out = dialogue._lock_dialogue_speakers(
            _six(detail),
            "酒保说：黑市的门票可不便宜，你想找什么路子？",
            _world(),
            PICTURES,
        )["detailed_description"]

        self.assertIn(
            "<Subject 2> speaks on-screen as S2: (S2) <d>[Chinese]",
            out,
        )
        self.assertNotIn("as S1", out)

    def test_offscreen_anchor_and_player_name_are_both_corrected(self) -> None:
        detail = "an off-screen voice identified as S2 says: (S2) <d>[Chinese] 我来问他。</d>"
        out = dialogue._lock_dialogue_speakers(
            _six(detail),
            "小明说：我来问他。",
            _world(),
            PICTURES,
        )["detailed_description"]

        self.assertEqual(
            "an off-screen voice identified as S1 says: (S1) <d>[Chinese] 我来问他。</d>",
            out,
        )

    def test_unknown_or_ambiguous_speaker_is_not_guessed(self) -> None:
        original = "<Subject 1> speaks on-screen as S1: (S1) <d>[Chinese] 你好。</d>"
        world = _world()
        world.subjects["S4"] = {
            "name": "S4",
            "kind": "人物",
            "brief": "另一个酒保",
        }
        pictures = PICTURES[:-1] + [
            {"n": 3, "kind": "character", "id": "S4", "name": "S4"},
            PICTURES[-1],
        ]

        unknown = dialogue._lock_dialogue_speakers(
            _six(original),
            "陌生人说：你好。",
            world,
            pictures,
        )["detailed_description"]
        ambiguous = dialogue._lock_dialogue_speakers(
            _six(original),
            "酒保说：你好。",
            world,
            pictures,
        )["detailed_description"]

        self.assertEqual(original, unknown)
        self.assertEqual(original, ambiguous)


class AudioRefTest(unittest.TestCase):
    """A clip without dialogue sends no voice reference at all.

    A voice reference is a real audio file labelled "voice and speaking style of S1"; sending it
    means writing "nobody talks in this shot" in overall_soundscape while handing the model "this is
    how this person talks". The model follows the latter.
    """

    def setUp(self) -> None:
        # The voice library's wavs are large and not in the repository, so the lookup is replaced
        # entirely: what is tested is the decision to send or not, not whether the file exists.
        fake = {"id": "v-test", "ref": "library/test.wav", "instruct": "沉稳男声"}
        self._patches = [
            mock.patch.object(loop.voices, "heal", return_value=fake),
            mock.patch.object(loop.voices, "path", side_effect=lambda ref: Path("/tmp") / str(ref or "x")),
            mock.patch.object(loop.voices, "label", return_value="测试音色"),
        ]
        for p in self._patches:
            p.start()
            self.addCleanup(p.stop)

    def test_no_dialogue_sends_no_voice_reference(self) -> None:
        """Real failure: 22 of a world's 24 clips without dialogue carried voices."""
        paths, bind = beat_assets._audio_refs(_world(), ["S1", "S3"], "")
        self.assertEqual([], paths)
        self.assertEqual([], bind)

    def test_blank_dialogue_counts_as_no_dialogue(self) -> None:
        for blank in ("", "   ", "\n"):
            with self.subTest(blank=repr(blank)):
                self.assertEqual(([], []), beat_assets._audio_refs(_world(), ["S1"], blank))

    def test_dialogue_sends_only_the_speakers_voice(self) -> None:
        """A silent character's speaking style must not interfere with the current line's performance."""
        paths, bind = beat_assets._audio_refs(_world(), ["S1", "S3"], "S3说：你好。")
        self.assertEqual(1, len(paths))
        self.assertEqual(["S3"], [b["sid"] for b in bind])
        self.assertEqual([1], [b["n"] for b in bind])

    def test_role_in_brief_resolves_the_speaking_voice(self) -> None:
        paths, bind = beat_assets._audio_refs(_world(), ["S1", "S3"], "酒保急促地喊：快走！")
        self.assertEqual(1, len(paths))
        self.assertEqual("S3", bind[0]["sid"])


class DialoguePerformanceTest(unittest.TestCase):
    def test_dialogue_gets_an_expressive_performance_direction(self) -> None:
        six = _six("Early in the shot, <Subject 2> speaks on-screen as S2: (S2) <d>[Chinese] 快走！</d>.")
        out = dialogue._lock_dialogue_performance(six, "酒保喊：快走！")
        detail = out["detailed_description"]
        self.assertIn("Match vocal emotion, breath, volume, pace, and effort", detail)
        self.assertIn("<d>[Chinese] 快走！</d>", detail)

    def test_silent_shot_does_not_get_a_voice_direction(self) -> None:
        six = _six("A silent empty room.")
        self.assertEqual(six, dialogue._lock_dialogue_performance(six, ""))


if __name__ == "__main__":
    unittest.main()
