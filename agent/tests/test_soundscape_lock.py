"""The same place reuses its ambience verbatim.

⚠️ Not only to save tokens; it fixes an audible problem. Three consecutive beats in the same room:

    beat 1  low hum of a high-speed train engine + hissing vents + distant metallic footsteps
    beat 2  train engine hum + flickering fluorescent lights + creaking metal locker doors
    beat 3  low rumble of wheels on rails + fluorescent buzz + medical equipment

The same room sounded different every few seconds. Over 74 consecutive pairs, `overall_soundscape`
in the same place was only 71% similar: it describes the **place**, yet it was rewritten every beat.
Same class of problem as visual style drift, solved the same way: by locking.

Why action sounds can be dropped: the video model **sees the picture** and generates footsteps,
doors and scraping itself; it cannot infer only the room tone and whether anyone speaks. So the room
tone is locked and the speech instruction recomputed each beat.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from murea_agent.shot import soundscape
from murea_agent.memory.world import World

BED = "Low hum of the train engine, hissing vents, distant metallic footsteps."
NO_SPEECH = "No speech, no voice-over, no narration -- nobody talks in this shot."


def _world(sound: str | None = None) -> World:
    world = World(title="t")
    world.places["药剂室"] = {"ref": "r.png", "desc": "", "first_seen": ""}
    if sound:
        world.places["药剂室"]["soundscape"] = sound
    return world


class SoundscapeLockTest(unittest.TestCase):
    def test_first_visit_learns_the_bed_without_the_speech_line(self) -> None:
        world = _world()
        six = {"overall_soundscape": f"{BED} {NO_SPEECH}"}
        soundscape._ensure_soundscape(six, "")
        soundscape._learn_soundscape(world, "药剂室", six)
        self.assertEqual(
            BED, world.places["药剂室"]["soundscape"], "禁声硬约束属于当前片段，不能被锁进地点底噪"
        )

    def test_locked_bed_is_reused_verbatim(self) -> None:
        world = _world(BED)
        out = soundscape._lock_soundscape(
            {"overall_soundscape": "模型这一拍新编的完全不同的环境音"}, world, "药剂室", ""
        )
        self.assertTrue(out["overall_soundscape"].startswith(BED))

    def test_the_speech_directive_still_follows_this_beat(self) -> None:
        """⚠️ The crucial one: with no audio direction the video model invents a soundtrack, including people talking who
        have no lines.
        """
        world = _world(BED)
        silent = soundscape._lock_soundscape({"overall_soundscape": "x"}, world, "药剂室", "")
        talking = soundscape._lock_soundscape(
            {"overall_soundscape": "x"}, world, "药剂室", "林远山：你是谁？"
        )
        self.assertIn("nobody talks in this shot", silent["overall_soundscape"])
        self.assertNotIn("ABSOLUTE NO-VOICE RULE", silent["overall_soundscape"])
        self.assertIn("speech written in the shot", talking["overall_soundscape"])
        self.assertNotIn("nobody talks", talking["overall_soundscape"])
        self.assertNotIn("ABSOLUTE NO-VOICE RULE", talking["overall_soundscape"])
        self.assertNotIn("No speaking", talking["overall_soundscape"])

    def test_a_truncated_historical_no_voice_lock_is_migrated(self) -> None:
        dirty = f"{BED} ABSOLUTE NO-VOICE RULE: stale implementation details"[:180]
        self.assertIn("ABSOLUTE NO-VOICE RULE", dirty)
        world = _world(dirty)

        talking = soundscape._lock_soundscape(
            {"overall_soundscape": "x"}, world, "药剂室", "林远山：你是谁？"
        )

        self.assertEqual(BED, world.places["药剂室"]["soundscape"])
        self.assertEqual(
            f"{BED} Plus the speech written in the shot.",
            talking["overall_soundscape"],
        )

    def test_a_historical_directive_without_a_bed_is_removed(self) -> None:
        world = _world("ABSOLUTE NO-VOICE RULE: stale implementation details")
        self.assertEqual("", soundscape._place_soundscape(world, "药剂室"))
        self.assertNotIn("soundscape", world.places["药剂室"])

    def test_an_unlocked_place_is_left_alone(self) -> None:
        """First visit, or an old world without the field: behavior unchanged."""
        world = _world()
        six = {"overall_soundscape": "原样"}
        self.assertEqual(six, soundscape._lock_soundscape(six, world, "药剂室", ""))

    def test_the_bed_is_learned_only_once(self) -> None:
        world = _world(BED)
        soundscape._learn_soundscape(
            world, "药剂室", {"overall_soundscape": "后来某一拍写的完全不同的一段环境音描述文字"}
        )
        self.assertEqual(BED, world.places["药剂室"]["soundscape"])

    def test_a_too_short_bed_is_not_learned(self) -> None:
        """The short fallback sentence must not be locked as this place's room tone."""
        world = _world()
        soundscape._learn_soundscape(world, "药剂室", {"overall_soundscape": "N/A"})
        self.assertNotIn("soundscape", world.places["药剂室"])


if __name__ == "__main__":
    unittest.main()
