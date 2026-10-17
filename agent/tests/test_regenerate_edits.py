"""A prompt edited by hand in the inspector must be sent unchanged on a rerun.

Real bug (user report): edit the prompt in the inspector, rerun, and the same clip came out. Not
because the model was called (`regenerate` never calls it) but because the `_lock_*` steps overwrote
the edit, silently.

Those locks exist because **models drift**: style gets rewritten, dynamic state dropped, speakers
mixed up. But the inspector is a debugging tool for **people**; locking hand-written text back makes
it pointless.
"""

from __future__ import annotations

import sys
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from murea_agent import loop
from murea_agent.memory.world import Node, World

BED = "Low hum of the train engine, hissing vents."
MINE = "MY OWN SOUNDSCAPE, nothing like the locked one."


def _world() -> World:
    world = World(title="t", visual_style="retro steampunk, brass and oiled leather")
    world.places["药剂室"] = {"ref": "r.png", "desc": "", "first_seen": "", "soundscape": BED}
    node = Node(id="n1", parent=None, video_id="v", clip="c.mp4", place="药剂室", beat="b", mode="continue")
    node.six_section = {
        "subject_definitions": "<Subject 1> is the man.",
        "summary": "[reference_generation] 原来的 summary",
        "retention_analysis": "<Subject 1>: fully_preserved",
        "detailed_description": "[Shot 1] 原来的分镜",
        "overall_soundscape": f"{BED} No speech, no voice-over, no narration -- nobody talks in this shot.",
        "non_diegetic_music": "N/A",
    }
    node.directive = {"dialogue": ""}
    world.nodes[node.id] = asdict(node)
    world.head = node.id
    return world


class RegenerateEditsTest(unittest.TestCase):
    def _run(self, six: dict | None) -> str:
        sent: dict = {}

        def fake_generate(prompt, refs, **kw):
            sent["prompt"] = prompt
            return SimpleNamespace(
                video_id="v2", video_path=Path("x.mp4"), params={}, continued=False, artifact_id="a2"
            )

        with (
            patch.object(loop.video, "generate", side_effect=fake_generate),
            patch.object(loop, "_grab_frames", return_value=[]),
            patch.object(World, "save", lambda self: None),
        ):
            loop.regenerate(_world(), six_section=six)
        return sent["prompt"]

    def test_a_hand_edited_soundscape_is_sent_verbatim(self) -> None:
        """`_lock_soundscape` is the one that overwrote it."""
        edited = dict(_world().head_node.six_section)
        edited["overall_soundscape"] = MINE
        prompt = self._run(edited)
        self.assertIn(MINE, prompt)
        self.assertNotIn(BED, prompt, "地点锁定的底噪不该把人手写的顶掉")

    def test_a_hand_edited_shot_survives_the_augmenting_locks(self) -> None:
        """The style / state / language locks **add** things and must not be switched off too.

        They never destroy a written sentence and keep a hand-edited prompt within the world's
        style, so the hand-written text stays and so does the world's style prefix.
        """
        edited = dict(_world().head_node.six_section)
        edited["detailed_description"] = "[Shot 1] 我自己写的分镜，别动它"
        prompt = self._run(edited)
        self.assertIn("我自己写的分镜，别动它", prompt)
        self.assertNotIn("原来的分镜", prompt)
        self.assertIn("retro steampunk", prompt, "世界的画风仍然要锁住")

    def test_an_empty_soundscape_is_still_backfilled(self) -> None:
        """The only fallback kept: with no audio direction the video model invents a soundtrack, including people talking
        who have no lines. Not a lock but a guard.
        """
        edited = dict(_world().head_node.six_section)
        edited["overall_soundscape"] = "N/A"
        prompt = self._run(edited)
        self.assertNotIn("N/A\n\nN/A", prompt)
        self.assertIn("nobody talks in this shot", prompt)

    def test_without_an_edit_the_locks_still_apply(self) -> None:
        """Without manual edits nothing changes: the locks are for the model, and that path is untouched."""
        prompt = self._run(None)
        self.assertIn(BED, prompt)


if __name__ == "__main__":
    unittest.main()
