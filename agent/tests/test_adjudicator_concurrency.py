"""A slow background judge must neither delay a turn nor overwrite its state."""

from __future__ import annotations

import asyncio
import sys
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from murea_agent import config
from app.api import sse
from app.api.routes import turns
from murea_agent import critic, loop
from murea_agent.memory.world import Node, World

GOOD = {
    "observed": "人物在门前站定",
    "options": ["询问", "等待", "观察"],
    "recommend": 0,
    "objective": "查明来意",
    "dropped": [],
    "appeared": {},
    "picked_up": {},
}


class JudgeConcurrencyTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        for key, folder in (
            ("DATA", "data"),
            ("SESSIONS", "sessions"),
            ("REFS", "refs"),
            ("FRAMES", "frames"),
        ):
            path = root / folder
            path.mkdir()
            guard = patch.object(config, key, path)
            guard.start()
            self.addCleanup(guard.stop)
        self.world = World(title="并发裁判")
        node = Node(
            id="n1",
            parent=None,
            video_id="v",
            clip="c.mp4",
            place="门前",
            beat="站定",
            mode="continue",
            frames=["frame.jpg"],
        )
        self.world.nodes[node.id] = asdict(node)
        self.world.head = node.id
        self.world.save()
        (config.FRAMES / "frame.jpg").write_bytes(b"FRAME")
        frame_guard = patch.object(critic, "_frames_for_gm", return_value=[("image/jpeg", b"x")])
        frame_guard.start()
        self.addCleanup(frame_guard.stop)

    def judge(self):
        class Request:
            async def json(self):
                return {"node": "n1"}

        return asyncio.run(turns.adjudicate(self.world.id, Request(), {"id": self.world.owner_id}))

    def run_with_intervening_change(self, mutate, verdict=GOOD):
        entered, release = threading.Event(), threading.Event()

        def slow_model(**kwargs):
            entered.set()
            if not release.wait(3):
                raise TimeoutError("test did not release the judge")
            return verdict

        with patch.object(loop.llm, "chat", side_effect=slow_model):
            with ThreadPoolExecutor(max_workers=1) as pool:
                task = pool.submit(self.judge)
                try:
                    self.assertTrue(entered.wait(2))
                    # Would raise GenerationBusy before the fix: the model call
                    # held this very lock for its entire generation duration.
                    with sse.mutation_slot(self.world.id):
                        current = World.load(self.world.id)
                        mutate(current)
                        current.save()
                finally:
                    release.set()
                result = task.result(timeout=3)
        self.assertFalse(result["ok"])
        return World.load(self.world.id)

    def test_next_turn_proceeds_during_inference_and_is_not_overwritten(self):
        def advance(world):
            node = Node(
                id="n2",
                parent="n1",
                video_id="next",
                clip="next.mp4",
                place="屋内",
                beat="下一拍",
                mode="continue",
            )
            world.nodes[node.id] = asdict(node)
            world.head = node.id

        current = self.run_with_intervening_change(advance)
        self.assertEqual("n2", current.head)
        self.assertIn("n2", current.nodes)
        self.assertNotEqual(GOOD["options"], current.nodes["n1"]["options"])

    def test_asset_edit_with_unchanged_head_is_preserved(self):
        current = self.run_with_intervening_change(
            lambda world: world.add_subject("用户新角色", "existing.png")
        )
        self.assertEqual("n1", current.head)
        self.assertTrue(any(s["name"] == "用户新角色" for s in current.subjects.values()))

    def test_unchanged_snapshot_commits_verdict(self):
        with patch.object(loop.llm, "chat", return_value=GOOD):
            result = self.judge()
        self.assertTrue(result["ok"])
        self.assertEqual(GOOD["options"], World.load(self.world.id).nodes["n1"]["options"])

    def test_busy_commit_is_discarded_without_waiting(self):
        # Simulate a turn owning the write slot by the time inference completes.
        with sse.mutation_slot(self.world.id):
            with patch.object(loop.llm, "chat", return_value=GOOD):
                result = self.judge()
        self.assertFalse(result["ok"])

    def test_new_character_drawing_does_not_hold_the_world_lock(self):
        def draw(*args):
            with sse.mutation_slot(self.world.id):
                pass
            return b"\x89PNG\r\n\x1a\nTEST", 0.01

        with (
            patch.object(
                loop.llm,
                "chat",
                return_value={
                    **GOOD,
                    "appeared": {"name": "访客", "look": "A visitor wearing a brown coat"},
                },
            ),
            patch.object(critic, "_timed_draw", side_effect=draw),
        ):
            result = self.judge()
        self.assertTrue(result["ok"])
        self.assertEqual("访客", result["appeared"])

    def test_discarded_snapshot_cleans_only_its_own_generated_refs(self):
        (config.REFS / "existing.png").write_bytes(b"existing")
        with patch.object(critic, "_timed_draw", return_value=(b"\x89PNG\r\n\x1a\nTEST", 0.01)):
            self.run_with_intervening_change(
                lambda world: setattr(world, "title", "用户改名"),
                {
                    **GOOD,
                    "appeared": {"name": "访客", "look": "A visitor"},
                    "picked_up": {"name": "提灯", "look": "A brass lantern"},
                },
            )
        self.assertEqual(["existing.png"], sorted(p.name for p in config.REFS.iterdir()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
