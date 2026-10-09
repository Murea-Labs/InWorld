#!/usr/bin/env python3
"""A beat that crosses into another space keeps the latent instead of teleporting."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from murea_agent import loop
from murea_agent.memory.world import World


def _route() -> World:
    """Altar → bridge → hall: three adjacent sub-spaces on one route."""
    w = World(video_model="murea")
    w.add_place("祭庭", "a.jpg")
    w.add_place("桥", "b.jpg", main_scene="祭庭")
    w.add_place("厅", "c.jpg", main_scene="祭庭")
    w.add_place("远处山谷", "d.jpg")
    w.place_links = [{"from": "祭庭", "to": "桥"}, {"from": "桥", "to": "厅"}]
    return w


def walked(w: World, src: str, dst: str, **kw) -> bool:
    args = {"mode": "cut", "transition": False, "new_destination": False, **kw}
    return loop._walked_over(w, src, dst, **args)


class WalkedOverTest(unittest.TestCase):
    def test_authored_neighbour_continues_even_without_the_gm_flag(self):
        """The director leaving out transition is common and must not teleport the person."""
        w = _route()
        self.assertTrue(walked(w, "祭庭", "桥"))
        self.assertTrue(walked(w, "桥", "厅"))
        self.assertTrue(walked(w, "祭庭", "桥", transition=True))

    def test_a_different_main_scene_still_cuts(self):
        w = _route()
        self.assertFalse(walked(w, "桥", "远处山谷", transition=True))

    def test_non_adjacent_subspace_still_cuts(self):
        w = _route()
        self.assertFalse(walked(w, "祭庭", "厅", transition=True))

    def test_a_brand_new_place_needs_the_gm_to_say_it_walked(self):
        """A new place without a reference image must not receive the person out of nowhere.

        The new place was attached by the director in this beat (the pipeline calls add_place before
        the check), so it has no adjacency yet and belongs to the route only through main_scene.
        """
        w = _route()
        w.add_place("新空地", "", main_scene="祭庭")
        self.assertFalse(walked(w, "祭庭", "新空地", new_destination=True))
        self.assertTrue(walked(w, "祭庭", "新空地", new_destination=True, transition=True))

    def test_same_place_or_a_plain_continuation_is_not_a_crossing(self):
        w = _route()
        self.assertFalse(walked(w, "桥", "桥", transition=True))
        self.assertFalse(walked(w, "祭庭", "桥", mode="continue"))
        self.assertFalse(walked(w, "", "桥", transition=True))


if __name__ == "__main__":
    unittest.main()
