"""Regression tests of the generation queue: queueing must be visible, and a cancel must not turn the lock into an endless wait."""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.api import sse
from app.api.routes import turns
from fastapi import HTTPException
from murea_agent import config, loop, reqlog
from murea_agent.memory.world import World
from murea_agent.providers.llm import GeminiError


async def collect(response) -> list[dict]:
    return [json.loads(item[6:]) async for item in response.body_iterator if item.startswith("data: ")]


class TurnRecoveryTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        data = patch.object(config, "DATA", Path(tmp.name))
        data.start()
        self.addCleanup(data.stop)
        self.world = World(id="w-recovery", title="Recovery")
        owned = patch.object(turns, "load_owned_world", return_value=self.world)
        owned.start()
        self.addCleanup(owned.stop)
        self.addCleanup(sse._request_results.clear)

    def test_failed_turn_recovers_original_error_and_logs_request_id(self) -> None:
        class Request:
            async def json(self):
                return {"text": "Cross the bridge", "request_id": "failed-request"}

        async def run():
            response = await turns.turn(self.world.id, Request(), {"id": "owner"})
            return await collect(response)

        with patch.object(loop, "turn", side_effect=GeminiError("Expecting value: line 1 column 1 (char 0)")):
            events = asyncio.run(run())
        result = turns.turn_status(self.world.id, "failed-request", {"id": "owner"})
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["event"], events[-1])
        self.assertIn("GeminiError: Expecting value", result["event"]["message"])
        logged = next(r for r in reqlog.read(self.world.id) if r["kind"] == "turn_terminal")
        self.assertEqual(logged["request_id"], "failed-request")
        self.assertEqual(logged["event"], result["event"])
        self.assertNotIn((self.world.id, "failed-request"), sse._active_requests)
        # The endpoint must still enforce world ownership, even for cached failures.
        with patch.object(turns, "load_owned_world", side_effect=HTTPException(404)):
            with self.assertRaises(HTTPException):
                turns.turn_status(self.world.id, "failed-request", {"id": "other"})

    def test_queued_request_is_running_but_unrelated_requests_are_unknown(self) -> None:
        lock = sse._world_lock(self.world.id)
        lock.acquire()
        try:
            response = sse.stream(lambda *_: None, world_id=self.world.id, request_id="queued")
            self.assertEqual(turns.turn_status(self.world.id, "queued", {})["status"], "running")
            self.assertEqual(turns.turn_status(self.world.id, "other-request", {})["status"], "unknown")
            self.assertEqual(sse.request_status("other-world", "queued")["status"], "unknown")
        finally:
            lock.release()
            asyncio.run(collect(response))

    def test_cancellation_is_recoverable(self) -> None:
        def cancel(*_):
            raise sse.Cancelled()

        response = sse.stream(cancel, world_id=self.world.id, request_id="cancelled-request")
        asyncio.run(collect(response))
        self.assertEqual(
            turns.turn_status(self.world.id, "cancelled-request", {}),
            {"status": "cancelled", "event": {"type": "cancelled"}},
        )

    def test_completed_node_takes_precedence_over_post_commit_failure(self) -> None:
        def fail_after_save(*_):
            self.world.nodes["n1"] = {"id": "n1", "request_id": "saved", "reply": "Arrived"}
            raise RuntimeError("post-commit notification failed")

        response = sse.stream(fail_after_save, world_id=self.world.id, request_id="saved")
        asyncio.run(collect(response))
        result = turns.turn_status(self.world.id, "saved", {})
        self.assertEqual(result["status"], "done")
        self.assertEqual(result["event"]["node"]["id"], "n1")

    def test_terminal_cache_is_bounded(self) -> None:
        def fail(*_):
            raise ValueError("invalid plan")

        with patch.object(sse, "_MAX_REQUEST_RESULTS", 2):
            for request_id in ("old", "recent", "latest"):
                response = sse.stream(fail, world_id=self.world.id, request_id=request_id)
                asyncio.run(collect(response))
        self.assertEqual(sse.request_status(self.world.id, "old")["status"], "unknown")
        self.assertEqual(sse.request_status(self.world.id, "latest")["status"], "failed")


class GenerationQueueTest(unittest.TestCase):
    @patch.object(config, "LANGUAGE", "zh")
    def test_queue_timeout_returns_an_error_event(self) -> None:
        original_limit = sse._QUEUE_WAIT_LIMIT_SECONDS
        original_poll = sse._QUEUE_POLL_SECONDS
        sse._QUEUE_WAIT_LIMIT_SECONDS = 0.03
        sse._QUEUE_POLL_SECONDS = 0.01
        # The queue waits only for the previous beat of **the same world**: with per-world locks,
        # another world running must not block here (exactly what this fix is about).
        held = sse._world_lock("w-same")
        held.acquire()
        try:
            response = sse.stream(lambda _emit, _events: None, world_id="w-same", request_id="timeout")

            async def collect() -> list[str]:
                return [item async for item in response.body_iterator]

            events = asyncio.run(collect())
        finally:
            held.release()
            sse._QUEUE_WAIT_LIMIT_SECONDS = original_limit
            sse._QUEUE_POLL_SECONDS = original_poll

        text = "".join(events)
        self.assertIn('"stage": "queue"', text)
        self.assertIn("超过 5 分钟仍未结束", text)
        self.assertIn('"type": "error"', text)
        recovered = sse.request_status("w-same", "timeout")
        self.assertEqual(recovered["status"], "failed")
        self.assertIn("GenerationBusy", recovered["event"]["message"])


class VideoWatchCancellationTest(unittest.TestCase):
    def test_status_callback_converts_cancel_to_cancelled_event(self) -> None:
        watch = loop._video_watch(lambda *_args: None, "video", lambda: True)
        with self.assertRaises(sse.Cancelled):
            watch({"status": "running"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
