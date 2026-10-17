"""The turn loop reports its steps in order and saves them with the clip."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from murea_agent import config, loop  # noqa: E402
from murea_agent.director import director  # noqa: E402
from murea_agent.events import Cancelled, Emitter  # noqa: E402
from murea_agent.memory.world import World  # noqa: E402

DIRECTIVE = {
    "reply": "阿明打开文件。",
    "mode": "continue",
    "place": "会议室",
    "place_image_prompt": "",
    "main_scene": "会议室",
    "beat": "Amin opens the file",
    "dialogue": "",
    "carry": "",
    "options": ["看文件", "去走廊", "问小红"],
    "continue_with": "",
    "subjects": ["S1"],
    "new_subjects": [],
    "subject_definitions": "",
    "summary": "[reference_generation] <Subject 1> opens the file.",
    "detailed_description": "[Shot 1] <Subject 1> opens a file inside <Subject 2>.",
    "overall_soundscape": "Paper rustles.",
    "non_diegetic_music": "N/A",
}


class TurnTraceTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = Path(tempfile.mkdtemp())
        self.refs, self.sessions = tmp / "refs", tmp / "sessions"
        self.refs.mkdir()
        self.sessions.mkdir()
        (self.refs / "s1.png").write_bytes(b"s1")
        (self.refs / "p1.png").write_bytes(b"p1")
        self.world = World(id="wtrace", title="trace", setting="office")
        self.world.add_subject("阿明", "s1.png", "a man in a blue shirt")
        self.world.add_place("会议室", "p1.png", "long table")
        self.world.here = "会议室"

    def _patches(self, generate):
        return (
            patch.object(config, "REFS", self.refs),
            patch.object(config, "SESSIONS", self.sessions),
            patch.object(director, "_decide_early", return_value=(dict(DIRECTIVE), None)),
            patch.object(loop.video, "generate", side_effect=generate),
        )

    def test_steps_follow_the_loop_and_are_saved_on_the_clip(self) -> None:
        gen = SimpleNamespace(
            video_id="v1", video_path=Path("clip.mp4"), params={}, continued=False, artifact_id="a1"
        )
        events: list[dict] = []
        p = self._patches(lambda *a, **k: gen)
        with p[0], p[1], p[2], p[3]:
            node = loop.turn(self.world, "阿明打开文件", notify=events.append)
        phases = [s["phase"] for s in node.trace]
        self.assertEqual(
            ["perceive", "plan", "check", "check", "check", "act", "act", "act", "act", "commit"], phases
        )
        names = [s["name"] for s in node.trace]
        self.assertIn("director", names)
        self.assertIn("render_clip", names)
        self.assertTrue(all(s["status"] == "done" for s in node.trace))
        # Live events: each step is announced when it starts and when it ends.
        live = [e for e in events if e["type"] == "step"]
        self.assertEqual(2 * len(node.trace), len(live))
        self.assertEqual("running", live[0]["status"])
        self.assertIn({"type": "say", "text": "阿明打开文件。"}, events)
        # The trace survives a save/load round trip.
        saved = self.world.nodes[node.id]
        self.assertEqual(node.trace, saved["trace"])

    def test_a_failing_tool_marks_its_step(self) -> None:
        events: list[dict] = []

        def fail(*_a, **_k):
            raise RuntimeError("video service down")

        p = self._patches(fail)
        with p[0], p[1], p[2], p[3], self.assertRaises(RuntimeError):
            loop.turn(self.world, "阿明打开文件", notify=events.append)
        last = [e for e in events if e["type"] == "step"][-1]
        self.assertEqual(("render_clip", "error"), (last["name"], last["status"]))
        self.assertIn("video service down", last["data"]["error"])

    def test_cancel_before_render_stops_without_a_clip(self) -> None:
        calls: list[int] = []
        p = self._patches(lambda *a, **k: calls.append(1))
        with p[0], p[1], p[2], p[3], self.assertRaises(Cancelled):
            loop.turn(self.world, "阿明打开文件", cancelled=lambda: True)
        self.assertEqual([], calls)
        self.assertEqual({}, self.world.nodes)

    def test_emitter_isolates_a_broken_listener(self) -> None:
        def broken(_event):
            raise ValueError("ui crashed")

        events = Emitter(broken)
        with events.step("plan", "director") as step:
            step.data["x"] = 1
        self.assertEqual("done", events.trace()[0]["status"])


if __name__ == "__main__":
    unittest.main()
