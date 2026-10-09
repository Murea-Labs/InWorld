"""Regression tests of the generation queue: queueing must be visible, and a cancel must not turn the lock into an endless wait."""

from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.api import sse
from murea_agent import loop


class GenerationQueueTest(unittest.TestCase):
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
            response = sse.stream(lambda _emit, _events: None, world_id="w-same")

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


class VideoWatchCancellationTest(unittest.TestCase):
    def test_status_callback_converts_cancel_to_cancelled_event(self) -> None:
        watch = loop._video_watch(lambda *_args: None, "video", lambda: True)
        with self.assertRaises(sse.Cancelled):
            watch({"status": "running"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
