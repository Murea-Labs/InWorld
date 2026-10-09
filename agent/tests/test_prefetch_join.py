"""When a choice is clicked, a prefetch still running must be **joined**, never restarted.

Clicking a choice still meant waiting. Measured:

    clicked the recommended choice    104 beats   57% hit
    clicked another choice             87 beats   29% hit

Prefetches were **cached only after finishing**, so a running one was invisible to the server. Each
took 4–5 s and many users clicked within 10 s; `_take_prefetched` saw nothing and ran the same call
again: wasted, and the user waited 3–5 s anyway.

Now the cache stores **the call itself** (a future), leaving two outcomes: finished → take it;
running → join it.
"""

from __future__ import annotations

import sys
import time
import threading
from concurrent import futures
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from murea_agent import reqlog
from murea_agent.director import director  # noqa: E402
from murea_agent.director import prefetch as prefetch_mod  # noqa: E402
from murea_agent.guards import replan  # noqa: E402
from murea_agent.memory.world import World


class PrefetchJoinTest(unittest.TestCase):
    def setUp(self) -> None:
        prefetch_mod._prefetched.clear()
        self.addCleanup(prefetch_mod._prefetched.clear)
        self.world = World(title="t")
        self.world.head = "n1"

    def test_a_finished_prefetch_is_a_hit(self) -> None:
        with patch.object(director, "_decide", lambda *a, **k: {"reply": "ok"}):
            self.assertTrue(prefetch_mod.prefetch(self.world, "走出去"))
            time.sleep(0.2)
            self.assertEqual({"reply": "ok"}, prefetch_mod._take_prefetched(self.world, "走出去"))

    def test_inflight_prefetch_is_joined_without_a_second_decision(self):
        started, release = threading.Event(), threading.Event()

        def slow(*a, **k):
            started.set()
            release.wait(3)
            return {"reply": "same result"}

        with (
            patch.object(director, "_decide", side_effect=slow) as decide,
            futures.ThreadPoolExecutor(max_workers=1) as runner,
        ):
            prefetch_mod.prefetch(self.world, "走出去")
            self.assertTrue(started.wait(1))
            joined = runner.submit(prefetch_mod._take_prefetched, self.world, "走出去")
            try:
                # Old behavior returned None after 50 ms and restarted planning.
                with self.assertRaises(futures.TimeoutError):
                    joined.result(timeout=0.15)
                self.assertTrue(prefetch_mod.prefetch(self.world, "走出去"))
            finally:
                release.set()
            self.assertEqual({"reply": "same result"}, joined.result(timeout=1))
            decide.assert_called_once()

    def test_click_during_validation_reuses_the_same_audited_result(self):
        started, release = threading.Event(), threading.Event()

        def audit(world, text, decision, *args, **kwargs):
            started.set()
            release.wait(3)
            return decision, False

        with (
            patch.object(director, "_decide", return_value={"reply": "audited"}) as decide,
            patch.object(replan, "_prepare_action_plan", side_effect=audit) as prepare,
            futures.ThreadPoolExecutor(max_workers=1) as runner,
        ):
            prefetch_mod.prefetch(self.world, "走出去")
            self.assertTrue(started.wait(1))
            joined = runner.submit(prefetch_mod._take_prefetched, self.world, "走出去")
            try:
                with self.assertRaises(futures.TimeoutError):
                    joined.result(timeout=0.15)
            finally:
                release.set()
            result = joined.result(timeout=1)
            self.assertIsInstance(result, replan._PreparedDecision)
            self.assertEqual("audited", result["reply"])
            decide.assert_called_once()
            prepare.assert_called_once()

    def test_wait_timeout_keeps_original_task_for_retry(self):
        task = futures.Future()
        key = prefetch_mod._prefetch_key(self.world, "走出去")
        prefetch_mod._prefetched[key] = task
        with patch.object(prefetch_mod, "_PREFETCH_JOIN_TIMEOUT", 0.03):
            with self.assertRaisesRegex(TimeoutError, "still being planned"):
                prefetch_mod._take_prefetched(self.world, "走出去")
        self.assertIs(task, prefetch_mod._prefetched[key])
        task.set_result({"reply": "late success"})
        self.assertEqual({"reply": "late success"}, prefetch_mod._take_prefetched(self.world, "走出去"))
        self.assertNotIn(key, prefetch_mod._prefetched)

    def test_cancel_during_join_returns_promptly_and_preserves_task(self):
        from murea_agent.events import Cancelled

        task = futures.Future()
        key = prefetch_mod._prefetch_key(self.world, "走出去")
        prefetch_mod._prefetched[key] = task
        cancel, waiting = threading.Event(), threading.Event()
        with futures.ThreadPoolExecutor(max_workers=1) as runner:
            joined = runner.submit(
                prefetch_mod._take_prefetched,
                self.world,
                "走出去",
                cancelled=cancel.is_set,
                on_wait=waiting.set,
            )
            self.assertTrue(waiting.wait(1))
            cancel.set()
            with self.assertRaises(Cancelled):
                joined.result(timeout=0.5)
        self.assertIs(task, prefetch_mod._prefetched[key])

    def test_worker_timeout_is_failure_not_another_wait(self):
        task = futures.Future()
        task.set_exception(TimeoutError("upstream timed out"))
        key = prefetch_mod._prefetch_key(self.world, "走出去")
        prefetch_mod._prefetched[key] = task
        self.assertIsNone(prefetch_mod._take_prefetched(self.world, "走出去"))
        self.assertNotIn(key, prefetch_mod._prefetched)

    def test_a_second_prefetch_of_the_same_text_does_not_burn_a_call(self) -> None:
        calls: list[int] = []
        with patch.object(director, "_decide", lambda *a, **k: (calls.append(1), {"reply": "ok"})[1]):
            prefetch_mod.prefetch(self.world, "走出去")
            prefetch_mod.prefetch(self.world, "走出去")
            time.sleep(0.3)
        self.assertEqual(1, len(calls))

    def test_a_failed_prefetch_falls_back_to_the_normal_path(self) -> None:
        """A failed speed-up must never become a failure: fall back to normal speed, no error."""
        with patch.object(director, "_decide", side_effect=RuntimeError("boom")):
            prefetch_mod.prefetch(self.world, "x")
            time.sleep(0.2)
            self.assertIsNone(prefetch_mod._take_prefetched(self.world, "x"))

    def test_prefetch_does_not_leak_the_world_log_binding_to_the_caller(self) -> None:
        self.assertEqual("", reqlog.current())
        with patch.object(director, "_decide", lambda *a, **k: {"reply": "ok"}):
            self.assertTrue(prefetch_mod.prefetch(self.world, "走出去"))
            self.assertEqual(
                "",
                reqlog.current(),
                "后台预推演不能把世界日志绑定留在调用线程",
            )
            self.assertEqual(
                {"reply": "ok"},
                prefetch_mod._take_prefetched(self.world, "走出去"),
            )
        self.assertEqual("", reqlog.current())

    def test_a_miss_is_still_a_miss(self) -> None:
        self.assertIsNone(prefetch_mod._take_prefetched(self.world, "没推演过的话"))

    def test_no_head_means_no_prefetch(self) -> None:
        world = World(title="t")
        with patch.object(director, "_decide") as decide:
            self.assertFalse(prefetch_mod.prefetch(world, "x"))
        decide.assert_not_called()


class DraftMustNotEvictOptionsTest(unittest.TestCase):
    """Drafts must not evict choices; this was why the hit rate was only 19%.

    It was a FIFO of 4: after playback three choices were queued (the recommended one first); typing
    added a draft at every 400 ms pause, and the second draft evicted and cancelled the oldest
    entry, exactly the recommended choice. By the time the user clicked a choice its prefetch was
    gone.

    The median time from the end of a clip to the next send was 29.8 s and one prefetch took 5.88 s,
    a window five times the cost, yet the hit rate was 19%.
    """

    def setUp(self):
        prefetch_mod._prefetched.clear()
        self.addCleanup(prefetch_mod._prefetched.clear)
        self.world = World(title="t")
        self.world.head = "n1"

    def _fill(self):
        with patch.object(director, "_decide", lambda *a, **k: {"reply": "ok"}):
            for text in ("推荐选项", "选项二", "选项三"):
                prefetch_mod.prefetch(self.world, text, None, "option")
            for i in range(6):  # six pauses while typing one sentence
                prefetch_mod.prefetch(self.world, f"我要往前走{i}", None, "draft")
            time.sleep(0.2)

    def test_only_the_latest_draft_is_kept(self):
        self._fill()
        alive = [
            k
            for k, f in prefetch_mod._prefetched.items()
            if getattr(f, "murea_agent_kind", "option") == "draft"
        ]
        self.assertEqual(1, len(alive), "草稿只留最新那一条")
        self.assertEqual("我要往前走5", alive[0][2])


if __name__ == "__main__":
    unittest.main()
