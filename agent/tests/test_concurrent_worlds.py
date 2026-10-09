"""Beats of different worlds must run concurrently; only the video part queues.

Real failure (several users on one server): `sse._lock` was a **module-level global lock** around
the whole beat (agent + video). A beat took 23–39 s, so whenever someone else was playing you waited
a whole beat; one user sent the same line twice 70 s apart, and the UI said "previous clip still
generating" about **someone else's** clip.

The agent's seconds are pure LLM calls and can run concurrently. Only two things must be serial: two
beats of the same world (concurrent writes corrupt the world file) and requests to the video model
(one job at a time).
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.api import sse


class ConcurrentWorldsTest(unittest.TestCase):
    def test_two_worlds_do_not_block_each_other(self) -> None:
        """This is the whole point of the change."""
        held = sse._world_lock("wA")
        held.acquire()
        try:
            other = sse._world_lock("wB")
            self.assertTrue(other.acquire(blocking=False), "别人的世界在跑，不该挡住这个世界")
            other.release()
        finally:
            held.release()

    def test_the_same_world_still_serialises(self) -> None:
        """The same world must be serial: concurrent writes corrupt the world file."""
        lock = sse._world_lock("wSame")
        lock.acquire()
        try:
            self.assertFalse(sse._world_lock("wSame").acquire(blocking=False))
        finally:
            lock.release()

    def test_busy_is_scoped_to_one_world(self) -> None:
        """Prefetch uses busy() to decide whether to run. Another user's world running is unrelated."""
        with sse._owner_guard:
            sse._running["wA"] = "u1"
        try:
            self.assertTrue(sse.busy("wA"))
            self.assertFalse(sse.busy("wB"))
        finally:
            with sse._owner_guard:
                sse._running.pop("wA", None)

    def test_cancel_does_not_kill_someone_elses_turn(self) -> None:
        """`_cancel` used to be one global Event: one person pressing stop cancelled everyone's running beats."""
        with sse._owner_guard:
            sse._running.update({"wA": "u1", "wB": "u2"})
        try:
            self.assertTrue(sse.cancel_current("u1"))
            with sse._owner_guard:
                self.assertIn("wA", sse._cancel)
                self.assertNotIn("wB", sse._cancel, "不许波及别人的世界")
        finally:
            with sse._owner_guard:
                sse._running.clear()
                sse._cancel.clear()

    def test_the_gpu_lock_still_serialises_video(self) -> None:
        """The video service runs one job at a time; this must not be relaxed along with the rest."""
        from murea_agent.providers import video

        self.assertTrue(hasattr(video, "_gpu_lock"))
        video._gpu_lock.acquire()
        try:
            self.assertFalse(video._gpu_lock.acquire(blocking=False))
        finally:
            video._gpu_lock.release()

    def test_mutation_on_another_world_is_not_blocked(self) -> None:
        with sse.mutation_slot("wA"):
            with sse.mutation_slot("wB"):
                pass

    def test_mutation_on_the_same_world_is_blocked(self) -> None:
        with sse.mutation_slot("wA"):
            with self.assertRaises(sse.GenerationBusy):
                with sse.mutation_slot("wA"):
                    pass


if __name__ == "__main__":
    unittest.main()
