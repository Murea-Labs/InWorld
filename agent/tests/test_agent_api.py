"""The public API (`murea_agent.MureaMind`) and the terminal CLI, with fake models."""

from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import murea_agent  # noqa: E402
from murea_agent import __main__ as cli  # noqa: E402
from murea_agent import config, loop  # noqa: E402
from murea_agent.director import director  # noqa: E402

DIRECTIVE = {
    "reply": "You step into the inn.",
    "mode": "cut",
    "place": "Inn",
    "place_image_prompt": "",
    "place_brief": "",
    "main_scene": "Inn",
    "beat": "The traveler steps into the inn",
    "dialogue": "",
    "carry": "",
    "options": ["Order a drink", "Go upstairs", "Ask the keeper"],
    "continue_with": "",
    "subjects": [],
    "new_subjects": [],
    "subject_definitions": "",
    "summary": "[reference_generation] the inn <Subject 1>.",
    "detailed_description": "[Shot 1] A wide shot inside <Subject 1>.",
    "overall_soundscape": "Rain on the windows.",
    "non_diegetic_music": "N/A",
}


def _fake_collect(world, pending, jobs) -> float:
    """Point each pending asset at a file name, as writing the drawn image would."""
    for p in pending:
        (world.places if p.kind == "place" else world.subjects)[p.key]["ref"] = f"{p.kind}_{p.key}.png"
    return 0.0


class AgentApiTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = Path(tempfile.mkdtemp())
        for name in ("refs", "sessions", "clips"):
            (tmp / name).mkdir()
        self.tmp = tmp
        gen = SimpleNamespace(
            video_id="v1", video_path=tmp / "clips" / "c.mp4", params={}, continued=False, artifact_id="a1"
        )
        self.patches = [
            patch.object(config, "REFS", tmp / "refs"),
            patch.object(config, "SESSIONS", tmp / "sessions"),
            patch.object(config, "CLIPS", tmp / "clips"),
            patch.object(director, "_decide_early", side_effect=lambda *a, **k: (dict(DIRECTIVE), None)),
            patch.object(loop.video, "generate", return_value=gen),
            # The new place's reference image is drawn by a fake painter.
            patch.object(loop, "_draw_pending", return_value={}),
            patch.object(loop, "_collect_pending", side_effect=_fake_collect),
        ]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)

    def test_stream_reports_steps_and_ends_with_the_clip(self) -> None:
        agent = murea_agent.MureaMind()
        world = agent.create_world(title="Harbour", setting="a harbour town")
        events = list(agent.stream(world, "I step into the inn"))
        kinds = [e["type"] for e in events]
        self.assertEqual("done", kinds[-1])
        self.assertIn("say", kinds)
        done = [e for e in events if e["type"] == "step" and e["status"] == "done"]
        self.assertEqual("perceive", done[0]["phase"])
        node = events[-1]["node"]
        self.assertEqual("Inn", node.place)
        self.assertEqual(node.id, agent.load_world(world.id).head)
        self.assertIsNone(agent.reflect(world))  # no frames to look at
        self.assertTrue(agent.undo(world))
        self.assertIsNone(world.head)

    def test_stream_turns_failures_into_an_error_event(self) -> None:
        agent = murea_agent.MureaMind()
        world = agent.create_world(title="Harbour")
        with patch.object(loop.video, "generate", side_effect=RuntimeError("video service down")):
            events = list(agent.stream(world, "I step into the inn"))
        self.assertEqual("error", events[-1]["type"])
        self.assertIn("video service down", events[-1]["message"])

    def test_cli_plays_a_beat_and_prints_the_steps(self) -> None:
        out = io.StringIO()
        with contextlib.redirect_stdout(out), patch("builtins.input", side_effect=EOFError):
            code = cli.main(["I step into the inn"])
        self.assertEqual(0, code)
        text = out.getvalue()
        for word in ("perceive", "director", "render_clip", "You step into the inn.", "Order a drink"):
            self.assertIn(word, text)

    def test_configure_after_use_is_refused(self) -> None:
        with self.assertRaises(RuntimeError):
            murea_agent.configure(self.tmp / "other.yaml")


if __name__ == "__main__":
    unittest.main()
