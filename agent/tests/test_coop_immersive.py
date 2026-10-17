"""Immersive cancellation races with real room/world persistence, no models."""

from concurrent import futures
import os
import threading
import unittest
from unittest.mock import patch

from murea_agent import config
from murea_agent.coop import (
    room as coop,
    generation as coop_generation,
    immersive as coop_immersive,
    prefetch as coop_prefetch,
)
from murea_agent.coop import loop as coop_loop  # noqa: E402
from murea_agent.memory.world import World
from tests.test_coop_prefetch import CoopPrefetchTest, immediate


class ImmersiveTest(CoopPrefetchTest):
    def _prepared(self, room):
        return {
            **super()._prepared(room),
            "plan": {**super()._prepared(room)["plan"], "narration": "两人沿岩壁前行。"},
        }

    def enter(self, session="immersive-one"):
        response = self.a.post(self.url + "/immersive/start", json={"session_id": session})
        self.assertEqual(200, response.status_code, response.text)
        return response.json()

    def advance(self, session="immersive-one", round_number=1):
        response = self.a.post(
            self.url + "/immersive/advance", json={"session_id": session, "round": round_number}
        )
        self.assertEqual(200, response.status_code, response.text)
        return response.json()

    def stop(self, session="immersive-one"):
        response = self.a.post(self.url + "/immersive/stop", json={"session_id": session})
        self.assertEqual(200, response.status_code, response.text)
        return response.json()

    def test_atomic_advance_survives_duplicates_and_stale_round(self):
        room, actions = self.prepared_room()
        self.enter()
        result = self.advance()
        self.assertEqual("generating", result["status"])
        self.assertEqual(actions, {m["user_id"]: m["action"] for m in result["members"]})
        self.advance()
        self.advance(round_number=2)
        self.assertEqual(1, len(self.pending))
        self.assertEqual(room["intro"], result["intro"])

    def test_exit_before_worker_starts_preserves_guidance_and_never_renders(self):
        room, _ = self.prepared_room()
        self.enter()
        self.advance()
        result = self.stop()
        self.assertTrue(result["generation_cancelling"])
        self.assertEqual(room["intro"], result["intro"])
        self.assertTrue(all(not m["ready"] for m in result["members"]))
        fn, *args = self.pending.pop()
        with patch.object(coop_loop, "play_round") as render, patch.object(coop, "_after_round") as after:
            fn(*args)
        render.assert_not_called()
        after.assert_not_called()
        final = self.a.get(self.url).json()
        self.assertEqual("waiting", final["status"])
        self.assertEqual(1, final["round"])
        self.assertEqual("", final["error"])
        self.assertEqual(room["intro"], final["intro"])
        self.assertEqual("n-prefetch-base", World.load(room["world_id"]).head)

    def test_cancel_during_main_render_stops_remote_and_rejects_late_preview_and_commit(self):
        room, _ = self.prepared_room()
        prepared, rendered = self._draft(room)
        self.enter()
        self.advance()

        def rendering(prepared, **kw):
            kw["on_status"]({"id": "active-video-job", "status": "running"})
            self.stop()
            self.assertTrue(kw["cancelled"]())
            kw["on_video_ready"]("spec.mp4")
            self.assertNotIn("preview_clip", self.a.get(self.url).json())
            return rendered  # Simulate an upstream finishing despite cancel.

        fn, *args = self.pending.pop()
        with (
            patch.object(coop_generation, "prepare", return_value=prepared),
            patch.object(coop_generation, "render_prepared", side_effect=rendering),
        ):
            fn(*args)
        self.assertTrue(any(call.args[1] == "active-video-job" for call in self.cancel.call_args_list))
        result = self.a.get(self.url).json()
        self.assertEqual("waiting", result["status"])
        self.assertEqual(room["intro"], result["intro"])
        self.assertEqual("n-prefetch-base", World.load(room["world_id"]).head)
        self.assertFalse((config.CLIPS / "spec.mp4").exists())

    def test_cancel_accepted_prefetch_unblocks_consumer_without_fallback(self):
        room, actions = self.prepared_room()
        prepared = self._prepared(room)
        self._seed_draft(room, actions, prepared, status="prompt_ready")
        future = futures.Future()
        future.set_running_or_notify_cancel()
        coop_prefetch._RECOMMENDATION_FUTURES["recommend-test"] = future
        self.enter()
        self.advance()
        fn, *args = self.pending.pop()
        started = threading.Event()
        original = coop_immersive.await_prefetch

        def waiting(*a):
            started.set()
            return original(*a)

        with (
            patch.object(coop_immersive, "await_prefetch", side_effect=waiting),
            patch.object(coop, "_execute_round") as fallback,
            patch.object(coop, "_after_round") as after,
        ):
            worker = threading.Thread(target=fn, args=args)
            worker.start()
            self.assertTrue(started.wait(2))
            self.stop()
            worker.join(2)
            self.assertFalse(worker.is_alive())
        fallback.assert_not_called()
        after.assert_not_called()
        self.assertTrue(any(call.args[1] == "spec-job" for call in self.cancel.call_args_list))
        self.assertEqual("waiting", self.a.get(self.url).json()["status"])
        self.assertTrue(coop_prefetch._is_cancelled("recommend-test"))
        future.set_result(None)

    def test_exit_does_not_restart_speculative_video_on_room_reads(self):
        room, _ = self.prepared_room()
        self.enter()
        self.stop()
        with (
            patch.dict(os.environ, {"COOP_RECOMMENDATION_PREFETCH": "1", "COOP_SPECULATIVE_VIDEO": "1"}),
            patch.object(coop_prefetch._POOL, "submit", side_effect=immediate),
            patch.object(coop_generation, "prepare", return_value=self._prepared(room)) as plan,
            patch.object(coop_generation, "render_prepared") as render,
        ):
            for _ in range(4):
                result = self.a.get(self.url).json()
        plan.assert_called_once()
        render.assert_not_called()
        self.assertEqual("prompt_ready", result["recommendation_prefetch"]["status"])
        self.assertEqual(room["intro"], result["intro"])

    def test_reenter_after_cancel_can_generate_and_publish_next_guidance(self):
        room, _ = self.prepared_room()
        self.enter()
        self.advance()
        self.stop()
        fn, *args = self.pending.pop()
        fn(*args)
        prepared, rendered = self._draft(room)
        self.enter("immersive-two")
        self.advance("immersive-two")
        fn, *args = self.pending.pop()
        with (
            patch.object(coop_generation, "prepare", return_value=prepared),
            patch.object(coop_generation, "render_prepared", return_value=rendered),
            patch.object(coop, "_after_round") as after,
        ):
            fn(*args)
        after.assert_called_once()
        self.assertEqual(2, self.a.get(self.url).json()["round"])

    def test_late_old_session_exit_and_advance_cannot_affect_new_session(self):
        self.prepared_room()
        self.enter()
        self.enter("new")
        self.advance("new")
        self.stop()
        response = self.a.post(
            self.url + "/immersive/advance", json={"session_id": "immersive-one", "round": 1}
        )
        self.assertEqual(409, response.status_code)
        result = self.a.get(self.url).json()
        self.assertFalse(result["generation_cancelling"])
        self.assertTrue(all(m["ready"] for m in result["members"]))

    def test_ordinary_turn_is_not_cancelled_by_entering_then_exiting(self):
        room, actions = self.prepared_room()
        self._send_actions(room, actions)
        self.launch()
        self.enter()
        result = self.stop()
        self.assertFalse(result["generation_cancelling"])
        self.assertTrue(all(m["ready"] for m in result["members"]))
        self.cancel.assert_not_called()

    def test_already_committed_video_wins_late_exit(self):
        room, _ = self.prepared_room()
        prepared, rendered = self._draft(room)
        self.enter()
        self.advance()
        fn, *args = self.pending.pop()
        original = coop._finish

        def finish(current, snapshot, node):
            self.assertTrue(current["inflight"]["committed"])
            original(current, snapshot, node)

        with (
            patch.object(coop_generation, "prepare", return_value=prepared),
            patch.object(coop_generation, "render_prepared", return_value=rendered),
            patch.object(coop, "_finish", side_effect=finish),
            patch.object(coop, "_after_round"),
        ):
            fn(*args)
        self.stop()
        self.assertEqual(2, self.a.get(self.url).json()["round"])
        self.assertTrue((config.CLIPS / "spec.mp4").exists())
        self.cancel.assert_not_called()

    def test_nonmember_cannot_start_advance_or_stop(self):
        self.prepared_room()
        for action in ("start", "advance", "stop"):
            r = self.b.post(self.url + "/immersive/" + action, json={"session_id": "intruder", "round": 1})
            self.assertEqual(403, r.status_code)

    def test_manual_submit_after_exit_restores_normal_video_prefetch(self):
        room, actions = self.prepared_room()
        self.enter()
        self.stop()
        member = room["members"][0]
        result = self.a.put(
            self.url + "/action",
            json={"round": 1, "action": actions[member["user_id"]], "target_uid": member["user_id"]},
        )
        self.assertEqual(200, result.status_code)
        with coop._edit(self.code, persist=False) as stored:
            self.assertFalse(stored.get("recommendation_video_paused"))

    def test_unaccepted_speculative_preview_never_replaces_current_video(self):
        room, _ = self.prepared_room()
        prepared, rendered = self._draft(room)

        def render(*args, **kwargs):
            kwargs["on_video_ready"]("spec.mp4")
            with coop._edit(self.code, persist=False) as stored:
                self.assertNotIn("preview_clip", stored)
            return rendered

        with (
            patch.dict(os.environ, {"COOP_RECOMMENDATION_PREFETCH": "1", "COOP_SPECULATIVE_VIDEO": "1"}),
            patch.object(coop_prefetch._POOL, "submit", side_effect=immediate),
            patch.object(coop_generation, "prepare", return_value=prepared),
            patch.object(coop_generation, "render_prepared", side_effect=render),
        ):
            coop_prefetch.queue_recommendation(self.code)
        with coop._edit(self.code, persist=False) as stored:
            self.assertEqual("ready", stored["recommendation_prefetch"]["status"])
            self.assertNotIn("preview_clip", stored)


def load_tests(loader, tests, pattern):
    # Reuse fixture helpers without rerunning all inherited suites twice.
    return unittest.TestSuite(
        ImmersiveTest(name) for name in ImmersiveTest.__dict__ if name.startswith("test_")
    )


if __name__ == "__main__":
    unittest.main()
