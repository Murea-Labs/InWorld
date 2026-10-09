"""Server scheduling must precede playback and share browser prefetch futures."""

import asyncio
import copy
import threading
import unittest
from dataclasses import asdict
from unittest.mock import Mock, patch

from fastapi import HTTPException
from murea_agent import config
from murea_agent.director import director  # noqa: E402
from murea_agent.director import prefetch as prefetch_mod  # noqa: E402
from murea_agent.guards import replan  # noqa: E402
from app.api import sse
from app.api.routes import turns, worlds
from app import option_prefetch
from murea_agent import critic, reqlog
from murea_agent.memory.world import World, Node


class OptionPrefetchTest(unittest.TestCase):
    def setUp(self):
        self.world = World(title="warm test")
        self.node = Node(
            id="n1",
            parent=None,
            video_id="v1",
            clip="c.mp4",
            place="门口",
            beat="等待",
            mode="continue",
            options=["观察", "询问", "离开"],
            recommend=1,
        )
        self.world.nodes[self.node.id] = asdict(self.node)
        self.world.head = self.node.id
        guard = patch.object(config, "SERVER_OPTION_PREFETCH", True)
        guard.start()
        self.addCleanup(guard.stop)
        guard = patch.object(reqlog, "record")
        self.log = guard.start()
        self.addCleanup(guard.stop)

    def test_recommendation_first_independent_snapshots_and_deduplicated_options(self):
        self.world.nodes["n1"]["options"].append(" 询问 ")
        with patch.object(prefetch_mod, "prefetch", return_value=True) as pre:
            self.assertEqual(3, option_prefetch.warm(self.world, source="test"))
        self.assertEqual(["询问", "观察", "离开"], [c.args[1] for c in pre.call_args_list])
        snapshots = [c.args[0] for c in pre.call_args_list]
        self.assertEqual(3, len({id(w) for w in snapshots}))
        snapshots[0].nodes["n1"]["options"].clear()
        self.assertEqual(4, len(self.world.nodes["n1"]["options"]))
        self.assertEqual(4, len(snapshots[1].nodes["n1"]["options"]))

    def test_guards_skip_disabled_busy_stale_and_auto_chain(self):
        with patch.object(prefetch_mod, "prefetch") as pre:
            with patch.object(config, "SERVER_OPTION_PREFETCH", False):
                option_prefetch.warm(self.world, source="test")
            with patch.object(sse, "busy", return_value=True):
                option_prefetch.warm(self.world, source="test")
            option_prefetch.warm(self.world, source="test", expected_head="old")
            self.world.nodes["n1"]["continue_with"] = "继续"
            option_prefetch.warm(self.world, source="test")
            pre.assert_not_called()

    def test_video_done_schedules_before_sse_even_while_current_turn_is_busy(self):
        timeline = []
        events = Mock()
        events.put.side_effect = lambda event: timeline.append(event["type"])
        with (
            patch.object(turns, "load_owned_world", return_value=self.world),
            patch.object(sse, "busy", return_value=True),
            patch.object(
                prefetch_mod, "prefetch", side_effect=lambda *a, **k: timeline.append("queued") or True
            ),
        ):
            turns._done(events, self.world.id, self.node, {"id": self.world.owner_id})
        self.assertEqual(["queued", "queued", "queued", "done"], timeline)

    def test_scheduling_failure_does_not_break_done(self):
        events = Mock()
        with (
            patch.object(turns, "load_owned_world", return_value=self.world),
            patch.object(prefetch_mod, "prefetch", side_effect=RuntimeError("pool closed")),
        ):
            turns._done(events, self.world.id, self.node, {"id": self.world.owner_id})
        self.assertEqual("done", events.put.call_args.args[0]["type"])

    def test_world_load_schedules_only_after_ownership_check(self):
        with (
            patch.object(worlds, "load_owned_world", return_value=self.world),
            patch.object(prefetch_mod, "prefetch", return_value=True) as pre,
        ):
            result = worlds.get_world(self.world.id, {"id": self.world.owner_id})
            self.assertEqual(self.world.id, result["id"])
            self.assertEqual(3, pre.call_count)
        with (
            patch.object(worlds, "load_owned_world", side_effect=HTTPException(404)),
            patch.object(prefetch_mod, "prefetch") as pre,
        ):
            with self.assertRaises(HTTPException):
                worlds.get_world(self.world.id, {"id": "other"})
            pre.assert_not_called()

    def test_browser_prefetch_endpoint_runs_the_director_prefetch(self):
        class Request:
            async def json(self):
                return {"text": "观察", "kind": "draft"}

        with (
            patch.object(turns, "load_owned_world", return_value=self.world),
            patch.object(sse, "busy", return_value=False),
            patch.object(prefetch_mod, "prefetch", return_value=True) as pre,
        ):
            result = asyncio.run(turns.prefetch(self.world.id, Request(), {"id": self.world.owner_id}))
        self.assertEqual({"ok": True, "cached": True}, result)
        pre.assert_called_once_with(self.world, "观察", None, "draft")

    def test_adjudicated_options_schedule_before_response_only_on_success(self):
        class Request:
            async def json(self):
                return {"node": "n1"}

        def judge(world, node, **kwargs):
            world.nodes["n1"]["options"] = ["新推荐"]
            return {"options": ["新推荐"]}

        with (
            patch.object(turns, "load_owned_world", return_value=self.world),
            patch.object(critic, "adjudicate", side_effect=judge),
            patch.object(prefetch_mod, "prefetch", return_value=True) as pre,
        ):
            result = asyncio.run(turns.adjudicate(self.world.id, Request(), {"id": self.world.owner_id}))
            self.assertTrue(result["ok"])
            self.assertEqual(["新推荐"], [c.args[1] for c in pre.call_args_list])
        with (
            patch.object(turns, "load_owned_world", return_value=self.world),
            patch.object(critic, "adjudicate", return_value=None),
            patch.object(prefetch_mod, "prefetch") as pre,
        ):
            result = asyncio.run(turns.adjudicate(self.world.id, Request(), {"id": self.world.owner_id}))
            self.assertFalse(result["ok"])
            pre.assert_not_called()

    def test_browser_and_server_share_one_inflight_decision_per_option(self):
        prefetch_mod._prefetched.clear()
        release = threading.Event()

        def decide(*args, **kwargs):
            release.wait(2)
            return {"reply": "ok"}

        with (
            patch.object(director, "_decide", side_effect=decide) as model,
            patch.object(replan, "_prepare_action_plan", side_effect=lambda w, t, d, *a, **k: (d, False)),
        ):
            try:
                option_prefetch.warm(self.world, source="video_done")
                tasks = list(prefetch_mod._prefetched.values())
                for text in self.node.options:
                    prefetch_mod.prefetch(copy.deepcopy(self.world), text)
                self.assertEqual(tasks, list(prefetch_mod._prefetched.values()))
            finally:
                release.set()
                for task in list(prefetch_mod._prefetched.values()):
                    task.result(timeout=3)
                prefetch_mod._prefetched.clear()
            self.assertEqual(3, model.call_count)

    def test_failed_background_decision_is_logged(self):
        prefetch_mod._prefetched.clear()
        with patch.object(director, "_decide", side_effect=ValueError("invalid assets")):
            prefetch_mod.prefetch(self.world, "询问")
            task = next(iter(prefetch_mod._prefetched.values()))
            self.assertIsNone(task.result(timeout=2))
        prefetch_mod._prefetched.clear()
        failures = [c for c in self.log.call_args_list if c.args[0] == "prefetch_failed"]
        self.assertEqual(1, len(failures))
        self.assertIn("invalid assets", failures[0].kwargs["error"])


if __name__ == "__main__":
    unittest.main()
