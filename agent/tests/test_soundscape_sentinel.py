"""With the place ambience locked the model writes one word; that text was thrown away anyway.

`_lock_soundscape` replaces the model's overall_soundscape with the place's locked room tone (the
fix for the same room sounding different every few seconds). In 192 consecutive beat pairs **90%
were in the same place**, so the model's text was almost always wasted and on the critical path.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from murea_agent.shot import soundscape
from murea_agent.director import director  # noqa: E402
from murea_agent.memory.world import World


class Sentinel(unittest.TestCase):
    def test_sentinel_never_reaches_the_prompt(self) -> None:
        """A leaked sentinel would tell the video model "the soundscape is LOCKED", and it would act on that."""
        six = {"overall_soundscape": soundscape.SOUNDSCAPE_LOCKED}
        soundscape._ensure_soundscape(six, "")
        self.assertNotIn(soundscape.SOUNDSCAPE_LOCKED, six["overall_soundscape"].upper())
        self.assertTrue(six["overall_soundscape"].strip())

    def test_sentinel_is_case_insensitive(self) -> None:
        six = {"overall_soundscape": " locked "}
        soundscape._ensure_soundscape(six, "")
        self.assertNotIn("LOCKED", six["overall_soundscape"].upper())

    def test_locked_place_fills_the_bed_over_the_sentinel(self) -> None:
        w = World(title="t")
        w.here = "木栈道"
        w.places["木栈道"] = {"name": "木栈道", "soundscape": "水面拍岸声与远处的雾号"}
        six = soundscape._lock_soundscape(
            {"overall_soundscape": soundscape.SOUNDSCAPE_LOCKED}, w, "木栈道", ""
        )
        self.assertIn("水面拍岸声与远处的雾号", six["overall_soundscape"])
        self.assertNotIn("LOCKED", six["overall_soundscape"].upper())

    def test_unlocked_place_still_falls_back_deterministically(self) -> None:
        """The model misjudged the place and wrote the sentinel where nothing is locked: an empty soundscape must not be sent.
        With no audio direction the video model invents a soundtrack, including people talking who
        have no lines.
        """
        six = {"overall_soundscape": soundscape.SOUNDSCAPE_LOCKED}
        soundscape._ensure_soundscape(six, "")
        got = six["overall_soundscape"]
        self.assertTrue(got.strip())
        self.assertNotEqual(got.strip().upper(), "N/A")

    def test_hint_only_appears_when_the_place_is_locked(self) -> None:
        """An unlocked place must get the model's full text: it is locked onto the new place, and a bad one stays bad forever."""
        import inspect

        src = inspect.getsource(director._agent_step)
        self.assertIn("if locked_bed:", src)
        self.assertIn("sound_hint", src)
        # the hint must really be sent, not only computed
        self.assertIn("{sound_hint}", src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
