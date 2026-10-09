"""Background item reference jobs: never block the turn and never overwrite the user's new choice."""

from __future__ import annotations

import sys
import tempfile
import unittest
import asyncio
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from murea_agent import config
from app.api.routes import turns
from murea_agent.memory.world import Node, World


class ItemReferenceWorkerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.original = {
            "DATA": config.DATA,
            "REFS": config.REFS,
            "SESSIONS": config.SESSIONS,
        }
        config.DATA = root
        config.REFS = root / "refs"
        config.SESSIONS = root / "sessions"
        config.REFS.mkdir()
        config.SESSIONS.mkdir()

    def tearDown(self) -> None:
        for name, value in self.original.items():
            setattr(config, name, value)
        self.tmp.cleanup()

    def _world(self) -> tuple[World, str, str]:
        world = World(owner_id="default", title="t", visual_style="soft cinematic")
        sid = world.add_subject(
            "铜灯笼", "item_frame.jpg", "a dented brass lantern with warm oil flame", kind="物件"
        )
        prompt = "a dented brass lantern with warm oil flame"
        world.subjects[sid].update(
            {
                "source_ref": "item_frame.jpg",
                "ref_prompt": prompt,
                "ref_status": "pending",
                "ref_source": "frame",
                "from_node": "n1",
            }
        )
        (config.REFS / "item_frame.jpg").write_bytes(b"frame")
        world.save()
        return world, sid, prompt

    def test_success_replaces_ref_and_keeps_source_frame(self) -> None:
        world, sid, prompt = self._world()
        with patch.object(turns.imagegen, "generate", return_value=b"generated") as draw:
            turns._run_item_reference(
                world.id,
                world.owner_id,
                sid,
                "铜灯笼",
                prompt,
                "item_frame.jpg",
                world.visual_style,
            )
        draw.assert_called_once_with(prompt, kind="object", visual_style="soft cinematic")
        saved = World.load(world.id)
        item = saved.subjects[sid]
        self.assertEqual("ready", item["ref_status"])
        self.assertEqual("generated", item["ref_source"])
        self.assertEqual("item_frame.jpg", item["source_ref"])
        self.assertNotEqual("item_frame.jpg", item["ref"])
        self.assertEqual(b"generated", (config.REFS / item["ref"]).read_bytes())

    def test_failure_keeps_tail_frame_and_marks_failed(self) -> None:
        world, sid, prompt = self._world()
        with patch.object(turns.imagegen, "generate", side_effect=RuntimeError("offline")):
            turns._run_item_reference(
                world.id,
                world.owner_id,
                sid,
                "铜灯笼",
                prompt,
                "item_frame.jpg",
                world.visual_style,
            )
        item = World.load(world.id).subjects[sid]
        self.assertEqual("failed", item["ref_status"])
        self.assertEqual("frame", item["ref_source"])
        self.assertEqual("item_frame.jpg", item["ref"])
        self.assertIn("offline", item["ref_error"])

    def test_changed_asset_is_not_overwritten_by_old_job(self) -> None:
        world, sid, prompt = self._world()

        def draw_and_replace(*_args, **_kwargs):
            latest = World.load(world.id)
            latest.subjects[sid].update(
                {
                    "ref": "user.png",
                    "ref_status": "ready",
                    "ref_source": "user",
                    "ref_prompt": "",
                }
            )
            (config.REFS / "user.png").write_bytes(b"user")
            latest.save()
            return b"stale-generated"

        with patch.object(turns.imagegen, "generate", side_effect=draw_and_replace):
            turns._run_item_reference(
                world.id,
                world.owner_id,
                sid,
                "铜灯笼",
                prompt,
                "item_frame.jpg",
                world.visual_style,
            )
        item = World.load(world.id).subjects[sid]
        self.assertEqual("user.png", item["ref"])
        self.assertFalse(
            any(
                path.name.startswith("item_") and path.name != "item_frame.jpg"
                for path in config.REFS.iterdir()
            )
        )

    def test_enqueue_finds_only_the_pending_item_from_that_node(self) -> None:
        world, sid, prompt = self._world()
        with patch.object(turns, "_schedule_item_reference", return_value=True) as schedule:
            turns._enqueue_picked_up_reference(
                world.id,
                {"id": world.owner_id},
                "n1",
                "铜灯笼",
            )
        schedule.assert_called_once_with(
            world.id,
            world.owner_id,
            sid,
            "铜灯笼",
            prompt,
            "item_frame.jpg",
            world.visual_style,
        )

    def test_adjudicate_route_enqueues_after_successful_verdict(self) -> None:
        """The background job must be reachable on the normal (non-busy) path."""
        world, _, _ = self._world()
        node = Node(id="n1", parent=None, video_id="", clip="", place="会议室", beat="", mode="cut")
        world.nodes[node.id] = {
            **node.__dict__,
            "pictures": [],
        }
        world.head = node.id

        class Request:
            async def json(self):
                return {"node": node.id}

        async def immediate(fn):
            return fn()

        verdict = {"picked_up": "铜灯笼", "dropped": [], "appeared": ""}
        with (
            patch.object(turns, "load_owned_world", return_value=world),
            patch.object(turns.critic, "adjudicate", return_value=verdict),
            patch.object(turns, "mutation_slot", return_value=nullcontext()),
            patch.object(turns, "run_in_threadpool", side_effect=immediate),
            patch.object(turns, "_enqueue_picked_up_reference") as enqueue,
        ):
            out = asyncio.run(turns.adjudicate("w1", Request(), {"id": "default"}))

        self.assertTrue(out["ok"])
        enqueue.assert_called_once_with("w1", {"id": "default"}, node.id, "铜灯笼")


if __name__ == "__main__":
    unittest.main()
