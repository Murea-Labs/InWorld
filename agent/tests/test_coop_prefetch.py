"""Co-op latency hiding must remain exact-action and exact-world gated."""

from concurrent import futures
from dataclasses import asdict
import fcntl
import hashlib
import os
import threading
import unittest
from unittest.mock import ANY, patch, PropertyMock

from murea_agent import config
from murea_agent.coop import (
    room as coop,
    generation as coop_generation,
    intro as coop_intro,
    prefetch as coop_prefetch,
    memory as coop_memory,
)
from murea_agent.providers import video
from murea_agent.memory.world import Node, World
from murea_agent.providers.video import murea as murea_provider
from tests import test_coop


ACTION = {
    "appearance": "A traveler in a blue coat.",
    "action": "runs along the left wall toward the stone arch.",
    "beat": "runs toward the stone arch",
    "reference_layout": "single full-body view",
    "visibility": "full body",
    "retention_level": "fully_preserved",
    "retention": "The face, blue coat and boots remain visible while the running pose changes.",
}


def immediate(fn, *args):
    future = futures.Future()
    try:
        future.set_result(fn(*args))
    except BaseException as exc:
        future.set_exception(exc)
    return future


def setUpModule():
    # These tests check timing against the configured 6-10 s lengths (the full Murea model).
    # The unreachable test server reads as Murea-Fast, whose 5 s cap is tested in
    # test_coop_duration.py.
    patcher = patch.object(murea_provider.Murea, "max_seconds", new_callable=PropertyMock, return_value=15)
    patcher.start()
    unittest.addModuleCleanup(patcher.stop)


class CoopPrefetchTest(test_coop.CoopTest):
    def setUp(self):
        super().setUp()
        coop_prefetch._ACTION_FUTURES.clear()
        coop_prefetch._RECOMMENDATION_FUTURES.clear()
        coop_prefetch._CANCELLED.clear()
        coop_prefetch._PENDING_SUBMISSIONS.clear()
        cancel = patch.object(coop_prefetch._CANCEL_POOL, "submit")
        self.cancel = cancel.start()
        self.patches.append(cancel)

    def tearDown(self):
        for pending in self.pending:
            pending[3].close()
        self.pending.clear()
        coop_prefetch._ACTION_FUTURES.clear()
        coop_prefetch._RECOMMENDATION_FUTURES.clear()
        coop_prefetch._CANCELLED.clear()
        coop_prefetch._PENDING_SUBMISSIONS.clear()
        super().tearDown()

    def test_speculative_video_is_enabled_by_default_and_can_be_disabled(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertTrue(coop_prefetch._enabled("COOP_SPECULATIVE_VIDEO"))
        with patch.dict(os.environ, {"COOP_SPECULATIVE_VIDEO": "0"}, clear=True):
            self.assertFalse(coop_prefetch._enabled("COOP_SPECULATIVE_VIDEO"))

    def test_recommendation_waits_for_an_active_boss_audit(self):
        room, suggestions = self.prepared_room()
        with coop._edit(self.code) as current:
            current["boss_detection"] = {"status": "generating"}
        with (
            patch.dict(os.environ, {"COOP_RECOMMENDATION_PREFETCH": "1", "COOP_SPECULATIVE_VIDEO": "1"}),
            patch.object(coop_prefetch._POOL, "submit") as submit,
        ):
            self.assertFalse(coop_prefetch.queue_recommendation(self.code))
        submit.assert_not_called()

    def test_room_poll_schedules_missing_recommendation_once(self):
        self.prepared_room()
        future = futures.Future()
        with (
            patch.dict(os.environ, {"COOP_RECOMMENDATION_PREFETCH": "1", "COOP_SPECULATIVE_VIDEO": "1"}),
            patch.object(coop_prefetch._POOL, "submit", return_value=future) as submit,
        ):
            first = self.a.get(self.url).json()
            second = self.a.get(self.url).json()
        self.assertEqual("planning", first["recommendation_prefetch"]["status"])
        self.assertEqual("planning", second["recommendation_prefetch"]["status"])
        submit.assert_called_once()
        self.assertTrue(future.cancel())

    def prepared_room(self):
        room = self.local_room()
        ids = [m["user_id"] for m in room["members"]]
        for uid, name in zip(ids, ("阿青", "小白")):
            self.local_role(uid, name)
        stored = self.a.get(self.url).json()
        world = World.load(stored["world_id"])
        refs = [m["ref"] for m in stored["members"]]
        node = Node(
            id="n-prefetch-base",
            parent=world.head,
            video_id="base-video",
            clip="base.mp4",
            place="山洞",
            beat="序幕",
            mode="cut",
            refs=refs,
            artifact_id="base-latent",
            origin="coop",
            directive={
                "prompt_version": 5,
                "players": [
                    {k: m[k] for k in ("user_id", "character", "description")} for m in stored["members"]
                ],
                "shot": {"actions": [], "scene": "A cave."},
            },
            six_section={
                "subject_definitions": "\n".join(f"<Subject {i}> is a traveler." for i in range(1, 3))
            },
        )
        world.nodes[node.id] = asdict(node)
        world.head = node.id
        world.save()
        suggestions = {uid: text for uid, text in zip(ids, ("沿左侧岩壁向出口前进", "沿右侧岩壁向出口前进"))}
        with coop._edit(self.code) as current:
            current["opening_node"] = asdict(node)
            current["intro"] = {
                "status": "ready",
                "round": 1,
                "request_id": "guide",
                "narration": "两人站在山洞中。",
                "world_event": "下一刻碎石坠落。",
                "objective": "穿过山洞。",
                "error": "",
                "suggestions": [{"user_id": uid, "action": suggestions[uid]} for uid in ids],
            }
            current["intro"]["signature"] = coop_intro.signature(current)
        return self.a.get(self.url).json(), suggestions

    def test_ready_player_is_translated_while_waiting_and_exact_cache_is_exposed(self):
        room, _ = self.prepared_room()
        uid = room["members"][0]["user_id"]
        with (
            patch.dict(os.environ, {"COOP_ACTION_PREFETCH": "1"}),
            patch.object(coop_prefetch._POOL, "submit", side_effect=immediate),
            patch.object(coop_generation, "prefetch_member_action", return_value=ACTION) as translate,
        ):
            response = self.a.put(
                self.url + "/action", json={"round": 1, "action": "沿左侧岩壁前进", "target_uid": uid}
            )
        self.assertEqual(200, response.status_code, response.text)
        with coop._edit(self.code, persist=False) as stored:
            self.assertEqual("ready", stored["action_prefetch"][uid]["status"])
            self.assertEqual({uid: ACTION}, coop_prefetch.matching_actions(stored))
        translate.assert_called_once()

    def test_duration_policy_change_invalidates_prefetch_signatures(self):
        room, suggestions = self.prepared_room()
        original_state = coop_prefetch.state_signature(room)
        original_recommendation = coop_prefetch.recommendation_signature(room, suggestions)
        with patch.object(config, "COOP_ACTION_SECONDS", 8):
            self.assertNotEqual(original_state, coop_prefetch.state_signature(room))
            self.assertNotEqual(
                original_recommendation,
                coop_prefetch.recommendation_signature(room, suggestions),
            )
        with patch.object(coop_generation.coop_duration, "POLICY_VERSION", 1):
            self.assertNotEqual(original_state, coop_prefetch.state_signature(room))
            self.assertNotEqual(
                original_recommendation, coop_prefetch.recommendation_signature(room, suggestions)
            )

    def test_asset_provenance_and_user_edit_version_invalidate_prefetch(self):
        room, _ = self.prepared_room()
        before = coop_prefetch.state_signature(room)
        for field, value in (
            ("scene_asset_source", "opening_video"),
            ("scene_asset_from_node", "n-prefetch-base"),
            ("asset_edit_revision", 1),
        ):
            self.assertNotEqual(before, coop_prefetch.state_signature({**room, field: value}))
        room["monster"] = {"name": "Dragon", "ref": "boss.jpg", "source": "manual"}
        before = coop_prefetch.state_signature(room)
        room["monster"]["source"] = "round_video"
        room["monster"]["from_node"] = "n-prefetch-base"
        self.assertNotEqual(before, coop_prefetch.state_signature(room))

    def test_superseded_queued_action_translation_releases_its_lock(self):
        room, _ = self.prepared_room()
        uid = room["members"][0]["user_id"]
        with coop._edit(self.code) as current:
            member = next(member for member in current["members"] if member["user_id"] == uid)
            member.update(action="先沿岩壁观察", ready=True)
        future = futures.Future()
        with (
            patch.dict(os.environ, {"COOP_ACTION_PREFETCH": "1"}),
            patch.object(coop_prefetch._POOL, "submit", return_value=future),
        ):
            self.assertTrue(coop_prefetch.queue_action(self.code, uid))
        with coop._edit(self.code) as current:
            entry = current["action_prefetch"][uid]
            coop_prefetch.cancel_action_prefetch(current, uid)
        self.assertTrue(future.cancelled())
        lock_name = hashlib.sha256(entry["signature"].encode()).hexdigest()[:16]
        with (coop._directory() / f"{self.code}.action-{lock_name}.lock").open("a") as guard:
            fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _prepared(self, room):
        refs = [m["ref"] for m in room["members"]]
        plan = {
            "reply": "两人同时向前。",
            "scene": "A stone cave.",
            "sound": "Falling gravel.",
            "camera": "A wide shot.",
            "actions": [],
        }
        six = {
            "subject_definitions": "defs",
            "summary": "summary",
            "retention_analysis": "retention",
            "detailed_description": "[Shot 1] action",
            "overall_soundscape": "stone",
            "non_diegetic_music": "N/A",
        }
        return {
            "base_head": "n-prefetch-base",
            "duration": 12,
            "previous": "base-latent",
            "refs": refs,
            "pictures": [],
            "players": [],
            "plan": plan,
            "six": six,
            "prompt": "complete recommended prompt",
            "merge_seconds": 0.1,
        }

    def _draft(self, room):
        clip = config.CLIPS / "spec.mp4"
        clip.write_bytes(b"video")
        return self._prepared(room), {
            "node_id": "n-spec",
            "frames": [],
            "frame_seconds": 0.1,
            "generation": video.Generation("spec-job", clip, 12, 1, {}, True, "next-latent"),
        }

    def _seed_draft(self, room, suggestions, prepared, rendered=None, *, status="ready"):
        with coop._edit(self.code) as current:
            current["recommendation_prefetch"] = {
                "status": status,
                "request_id": "recommend-test",
                "signature": coop_prefetch.recommendation_signature(current, suggestions),
                "actions": suggestions,
                "job_id": "spec-job",
                "_prepared": prepared,
            }
            if rendered:
                current["recommendation_prefetch"]["_rendered"] = coop_prefetch._serialize_rendered(rendered)

    def _send_actions(self, room, actions):
        for member in room["members"]:
            response = self.a.put(
                self.url + "/action",
                json={"round": 1, "action": actions[member["user_id"]], "target_uid": member["user_id"]},
            )
            self.assertEqual(200, response.status_code, response.text)

    def _consume_pending(self):
        fn, *args = self.pending.pop()
        try:
            self.assertIs(fn, coop._run_prefetched_round)
            with (
                patch.object(coop, "_after_round"),
                patch.object(coop_generation, "render_prepared") as render,
            ):
                fn(*args)
            render.assert_not_called()
        finally:
            args[2].close()
        result = self.a.get(self.url).json()
        self.assertEqual(2, result["round"])
        self.assertEqual("next-latent", result["rounds"][-1]["node"]["artifact_id"])
        self.assertTrue((config.CLIPS / "spec.mp4").exists())
        self.cancel.assert_not_called()

    def test_poll_during_adoption_of_saved_result_never_cancels_completed_video(self):
        room, suggestions = self.prepared_room()
        prepared, rendered = self._draft(room)
        self._seed_draft(room, suggestions, prepared, rendered)
        self._send_actions(room, suggestions)
        self.assertEqual(202, self.launch().status_code)
        for _ in range(3):
            result = self.a.get(self.url).json()
            self.assertEqual("consuming", result["recommendation_prefetch"]["status"])
        self.cancel.assert_not_called()
        self._consume_pending()

    def test_future_completion_during_adoption_does_not_look_like_a_restart(self):
        room, suggestions = self.prepared_room()
        prepared, rendered = self._draft(room)
        self._seed_draft(room, suggestions, prepared, status="rendering")
        future = futures.Future()
        coop_prefetch._register(
            coop_prefetch._RECOMMENDATION_FUTURES, "recommend-test", future, request_id="recommend-test"
        )
        # Use the real producer lock to cover polling before the claim.
        with (coop._directory() / f"{self.code}.prefetch.lock").open("a") as guard:
            fcntl.flock(guard, fcntl.LOCK_EX)
            self._send_actions(room, suggestions)
            self.assertEqual(202, self.launch().status_code)
            with coop._edit(self.code) as current:
                current["recommendation_prefetch"]["_rendered"] = coop_prefetch._serialize_rendered(rendered)
            future.set_result({"prepared": prepared, "rendered": rendered})
        self.assertNotIn("recommend-test", coop_prefetch._RECOMMENDATION_FUTURES)
        self.assertEqual("consuming", self.a.get(self.url).json()["recommendation_prefetch"]["status"])
        self._consume_pending()

    def test_restart_preserves_completed_uncommitted_result_for_retry(self):
        room, suggestions = self.prepared_room()
        prepared, rendered = self._draft(room)
        self._seed_draft(room, suggestions, prepared, rendered)
        self._send_actions(room, suggestions)
        self.assertEqual(202, self.launch().status_code)
        _, *args = self.pending.pop()
        args[2].close()  # simulate the receiving process exiting before committing to the world
        result = self.a.get(self.url).json()
        self.assertEqual("failed", result["status"])
        self.assertEqual("ready", result["recommendation_prefetch"]["status"])
        self.assertEqual(202, self.launch().status_code)
        self._consume_pending()

    def test_restart_after_world_commit_never_cancels_owned_artifact(self):
        room, suggestions = self.prepared_room()
        prepared, rendered = self._draft(room)
        self._seed_draft(room, suggestions, prepared, rendered)
        self._send_actions(room, suggestions)
        self.assertEqual(202, self.launch().status_code)
        _, code, snapshot, guard, _, _ = self.pending.pop()
        try:
            world = World.load(room["world_id"])
            coop_generation.commit_prepared(
                world,
                coop.merged_prompt(snapshot),
                room=snapshot,
                mentions=[m["subject_id"] for m in snapshot["members"]],
                prepared=prepared,
                rendered=rendered,
                origin="coop",
            )
        finally:
            guard.close()  # the world is saved, but _finish has not run yet
        result = self.a.get(self.url).json()
        self.assertEqual(2, result["round"])
        self.assertNotIn("recommendation_prefetch", result)
        self.assertTrue(rendered["generation"].video_path.exists())
        self.cancel.assert_not_called()

    def test_recovery_uses_producer_lock_before_future_registration(self):
        room, suggestions = self.prepared_room()
        self._seed_draft(room, suggestions, self._prepared(room), status="planning")
        with (coop._directory() / f"{self.code}.prefetch.lock").open("a") as guard:
            fcntl.flock(guard, fcntl.LOCK_EX)
            self.assertEqual("planning", self.a.get(self.url).json()["recommendation_prefetch"]["status"])
        self.cancel.assert_not_called()

    def test_only_abandoned_uncommitted_job_is_cancelled_after_restart(self):
        room, suggestions = self.prepared_room()
        self._seed_draft(room, suggestions, self._prepared(room), status="rendering")
        self.assertEqual("failed", self.a.get(self.url).json()["recommendation_prefetch"]["status"])
        self.cancel.assert_called_once_with(
            coop_prefetch._cancel_remote, "spec-job", room["world_id"], 1, ANY
        )
        self.assertEqual("base-latent", World.load(room["world_id"]).head_node.artifact_id)

    def test_custom_actions_discard_only_the_unselected_child(self):
        room, suggestions = self.prepared_room()
        prepared, rendered = self._draft(room)
        self._seed_draft(room, suggestions, prepared, rendered)
        custom = dict(suggestions)
        custom[room["members"][0]["user_id"]] = "留在岩壁旁检查背包"
        self._send_actions(room, custom)
        self.cancel.assert_called_once_with(
            coop_prefetch._cancel_remote, "spec-job", room["world_id"], 1, ANY
        )
        self.assertFalse(rendered["generation"].video_path.exists())
        world = World.load(room["world_id"])
        self.assertEqual("base-latent", world.head_node.artifact_id)
        self.assertEqual(202, self.launch().status_code)
        fn, code, snapshot, guard = self.pending.pop()
        guard.close()
        self.assertIs(fn, coop._run_round)
        self.assertEqual(custom, {m["user_id"]: m["action"] for m in snapshot["members"]})

    def test_producer_never_downgrades_an_accepted_recommendation(self):
        room, _ = self.prepared_room()
        prepared, rendered = self._draft(room)

        def accept(*args, **kwargs):
            with coop._edit(self.code) as current:
                current["recommendation_prefetch"]["status"] = "consuming"
            return prepared

        def render(*args, **kwargs):
            with coop._edit(self.code, persist=False) as current:
                self.assertEqual("consuming", current["recommendation_prefetch"]["status"])
            return rendered

        with (
            patch.dict(os.environ, {"COOP_RECOMMENDATION_PREFETCH": "1", "COOP_SPECULATIVE_VIDEO": "1"}),
            patch.object(coop_prefetch._POOL, "submit", side_effect=immediate),
            patch.object(coop_generation, "prepare", side_effect=accept),
            patch.object(coop_generation, "render_prepared", side_effect=render) as generate,
        ):
            self.assertTrue(coop_prefetch.queue_recommendation(self.code))
        generate.assert_called_once()
        with coop._edit(self.code, persist=False) as current:
            self.assertEqual("consuming", current["recommendation_prefetch"]["status"])

    def test_prompt_ready_after_restart_resumes_video_without_replanning(self):
        room, suggestions = self.prepared_room()
        prepared, rendered = self._draft(room)
        self._seed_draft(room, suggestions, prepared, status="prompt_ready")
        with (
            patch.dict(os.environ, {"COOP_RECOMMENDATION_PREFETCH": "1", "COOP_SPECULATIVE_VIDEO": "1"}),
            patch.object(coop_prefetch._POOL, "submit", side_effect=immediate),
            patch.object(coop_generation, "prepare") as plan,
            patch.object(coop_generation, "render_prepared", return_value=rendered) as render,
        ):
            result = self.a.get(self.url).json()
        self.assertEqual("ready", result["recommendation_prefetch"]["status"])
        plan.assert_not_called()
        render.assert_called_once()
        self.assertEqual("complete recommended prompt", render.call_args.args[0]["prompt"])
        with coop._edit(self.code, persist=False) as stored:
            self.assertNotEqual("recommend-test", stored["recommendation_prefetch"]["request_id"])

    def test_observation_wait_preserves_preparation_and_resumes_on_result_or_deadline(self):
        room, _ = self.prepared_room()
        prepared, rendered = self._draft(room)
        now = [100.0]
        with coop._edit(self.code) as stored:
            stored["intro"]["state_basis"] = coop_memory.fingerprint(stored.get("story_state") or {})
            stored["story_observation"] = {"status": "generating", "node_id": stored["opening_node"]["id"]}
        with (
            patch.dict(os.environ, {"COOP_RECOMMENDATION_PREFETCH": "1", "COOP_SPECULATIVE_VIDEO": "1"}),
            patch.object(coop_prefetch._POOL, "submit", side_effect=immediate),
            patch.object(coop_prefetch.threading, "Timer") as timer,
            patch.object(coop_prefetch.time, "time", side_effect=lambda: now[0]),
            patch.object(coop_generation, "prepare", return_value=prepared) as plan,
            patch.object(coop_generation, "render_prepared", return_value=rendered) as render,
        ):
            self.assertTrue(coop_prefetch.queue_recommendation(self.code))
            plan.assert_called_once()
            render.assert_not_called()
            timer.assert_called_once()
            with coop._edit(self.code, persist=False) as stored:
                self.assertEqual("prompt_ready", stored["recommendation_prefetch"]["status"])
            self.assertTrue(coop_prefetch.queue_recommendation(self.code))
            render.assert_not_called()
            now[0] += 7
            # The observer is still pending: reaching the deadline is sufficient.
            coop_prefetch.resume_after_observation(self.code)
            plan.assert_called_once()
            render.assert_called_once()
        with coop._edit(self.code, persist=False) as stored:
            self.assertEqual("ready", stored["recommendation_prefetch"]["status"])

    def test_guidance_correction_during_preparation_defers_only_video_submission(self):
        room, _ = self.prepared_room()
        prepared, rendered = self._draft(room)

        def prepare(*args, **kwargs):
            with coop._edit(self.code) as stored:
                stored["guidance_refresh"] = {"status": "planning"}
            return prepared

        with (
            patch.dict(os.environ, {"COOP_RECOMMENDATION_PREFETCH": "1", "COOP_SPECULATIVE_VIDEO": "1"}),
            patch.object(coop_prefetch._POOL, "submit", side_effect=immediate),
            patch.object(coop_generation, "prepare", side_effect=prepare) as plan,
            patch.object(coop_generation, "render_prepared", return_value=rendered) as render,
        ):
            self.assertTrue(coop_prefetch.queue_recommendation(self.code))
            render.assert_not_called()
            with coop._edit(self.code) as stored:
                stored["guidance_refresh"]["status"] = "ready"
            coop_prefetch.queue_recommendation(self.code)
            plan.assert_called_once()
            render.assert_called_once()

    def test_custom_action_cancels_queued_draft_and_releases_its_lock(self):
        self.prepared_room()
        future = futures.Future()
        with (
            patch.dict(os.environ, {"COOP_RECOMMENDATION_PREFETCH": "1"}),
            patch.object(coop_prefetch._POOL, "submit", return_value=future),
        ):
            self.assertTrue(coop_prefetch.queue_recommendation(self.code))
        with coop._edit(self.code) as current:
            coop_prefetch.invalidate(current, "玩家输入自定义动作")
        self.assertTrue(future.cancelled())
        with (coop._directory() / f"{self.code}.prefetch.lock").open("a") as guard:
            fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.cancel.assert_not_called()  # not submitted to the video model yet, so there is no remote job to cancel

    def test_invalidation_racing_recommendation_submit_cancels_late_future(self):
        self.prepared_room()
        future = futures.Future()

        def submit(fn, *args):
            # Simulate a player edit after the worker was submitted but before
            # its Future was inserted in the registry.
            with coop._edit(self.code) as current:
                coop_prefetch.invalidate(current, "玩家输入自定义动作")
            return future

        with (
            patch.dict(os.environ, {"COOP_RECOMMENDATION_PREFETCH": "1"}),
            patch.object(coop_prefetch._POOL, "submit", side_effect=submit),
        ):
            self.assertTrue(coop_prefetch.queue_recommendation(self.code))
        self.assertTrue(future.cancelled())
        self.assertFalse(coop_prefetch._CANCELLED)
        self.assertFalse(coop_prefetch._RECOMMENDATION_FUTURES)

    def test_old_recommendation_callback_cannot_clear_new_same_signature_cancel(self):
        old = futures.Future()
        new = futures.Future()
        coop_prefetch._register(
            coop_prefetch._RECOMMENDATION_FUTURES, "same-request", old, request_id="same-request"
        )
        coop_prefetch._register(
            coop_prefetch._RECOMMENDATION_FUTURES, "same-request", new, request_id="same-request"
        )
        with coop_prefetch._LOCK:
            coop_prefetch._CANCELLED.add("same-request")
        old.set_result(None)
        self.assertTrue(coop_prefetch._is_cancelled("same-request"))
        new.set_result(None)
        self.assertFalse(coop_prefetch._is_cancelled("same-request"))

    def test_late_submit_after_custom_action_cancels_only_the_abandoned_child(self):
        room, _ = self.prepared_room()
        prepared, rendered = self._draft(room)

        def render(*args, **kwargs):
            with coop._edit(self.code) as current:
                coop_prefetch.invalidate(current, "玩家输入自定义动作")
            # Simulate the cancel arriving before the job id; the Future is not registered yet, so
            # checking only the in-memory cancel flag is not enough.
            kwargs["on_status"]({"id": "spec-job", "status": "processing"})
            return rendered

        with (
            patch.dict(os.environ, {"COOP_RECOMMENDATION_PREFETCH": "1", "COOP_SPECULATIVE_VIDEO": "1"}),
            patch.object(coop_prefetch._POOL, "submit", side_effect=immediate),
            patch.object(coop_generation, "prepare", return_value=prepared),
            patch.object(coop_generation, "render_prepared", side_effect=render),
        ):
            self.assertTrue(coop_prefetch.queue_recommendation(self.code))
        self.cancel.assert_called_once_with(
            coop_prefetch._cancel_remote, "spec-job", room["world_id"], 1, ANY
        )
        self.assertFalse(rendered["generation"].video_path.exists())
        self.assertEqual("base-latent", World.load(room["world_id"]).head_node.artifact_id)

    def test_recommendation_prompt_and_video_are_prepared_then_exact_actions_consume_them(self):
        room, suggestions = self.prepared_room()
        prepared = self._prepared(room)
        clip = config.CLIPS / "spec.mp4"
        clip.write_bytes(b"video")
        rendered = {
            "node_id": "n-spec",
            "frames": [],
            "frame_seconds": 0.1,
            "generation": video.Generation("spec-job", clip, 12, 1, {}, True, "next-latent"),
        }
        with (
            patch.dict(
                os.environ,
                {
                    "COOP_RECOMMENDATION_PREFETCH": "1",
                    "COOP_SPECULATIVE_VIDEO": "1",
                    "COOP_ACTION_PREFETCH": "0",
                },
            ),
            patch.object(coop_prefetch._POOL, "submit", side_effect=immediate),
            patch.object(coop_generation, "prepare", return_value=prepared) as plan,
            patch.object(coop_generation, "render_prepared", return_value=rendered) as render,
        ):
            self.assertTrue(coop_prefetch.queue_recommendation(self.code))
        stored = self.a.get(self.url).json()
        self.assertEqual("ready", stored["recommendation_prefetch"]["status"])
        self.assertNotIn("clip", stored["recommendation_prefetch"])
        self.assertNotIn("video_id", stored["recommendation_prefetch"])
        plan.assert_called_once()
        render.assert_called_once()
        for member in room["members"]:
            response = self.a.put(
                self.url + "/action",
                json={"round": 1, "action": suggestions[member["user_id"]], "target_uid": member["user_id"]},
            )
            self.assertEqual(200, response.status_code, response.text)
        with coop._edit(self.code) as current:
            request_id, future = coop_prefetch.take(current)
        bundle = future.result()
        self.assertEqual("complete recommended prompt", bundle["prepared"]["prompt"])
        self.assertEqual("spec-job", bundle["rendered"]["generation"].video_id)
        coop_prefetch.complete(request_id)

    def test_editing_a_recommendation_discards_the_speculative_clip(self):
        room, _ = self.prepared_room()
        prepared = self._prepared(room)
        clip = config.CLIPS / "wasted.mp4"
        clip.write_bytes(b"video")
        rendered = {
            "node_id": "n-wasted",
            "frames": [],
            "frame_seconds": 0.1,
            "generation": video.Generation("wasted-job", clip, 12, 1, {}, True, "latent"),
        }
        with (
            patch.dict(
                os.environ,
                {
                    "COOP_RECOMMENDATION_PREFETCH": "1",
                    "COOP_SPECULATIVE_VIDEO": "1",
                    "COOP_ACTION_PREFETCH": "0",
                },
            ),
            patch.object(coop_prefetch._POOL, "submit", side_effect=immediate),
            patch.object(coop_generation, "prepare", return_value=prepared),
            patch.object(coop_generation, "render_prepared", return_value=rendered),
        ):
            coop_prefetch.queue_recommendation(self.code)
            uid = room["members"][0]["user_id"]
            self.a.put(self.url + "/action", json={"round": 1, "action": "我改成留在原地", "target_uid": uid})
        self.assertFalse(clip.exists())
        self.assertEqual("superseded", self.a.get(self.url).json()["recommendation_prefetch"]["status"])

    def test_submission_order_miss_discards_speculative_clip_and_runs_normal_round(self):
        """Same actions in B→A order must not consume an A→B draft."""
        room, suggestions = self.prepared_room()
        prepared, rendered = self._draft(room)
        clip = rendered["generation"].video_path
        self._seed_draft(room, suggestions, prepared, rendered)
        ids = [member["user_id"] for member in room["members"]]
        # Deliberately accept the second seat first; action text still matches
        # the recommendation, but the timeline order does not.
        for uid in (ids[1], ids[0]):
            response = self.a.put(
                self.url + "/action", json={"round": 1, "action": suggestions[uid], "target_uid": uid}
            )
            self.assertEqual(200, response.status_code, response.text)
        self.assertEqual(202, self.launch().status_code)
        self.assertEqual(1, len(self.pending))
        self.assertIs(self.pending[0][0], coop._run_round)
        self.assertFalse(clip.exists())
        self.assertEqual("superseded", self.a.get(self.url).json()["recommendation_prefetch"]["status"])
        self.cancel.assert_called_once_with(
            coop_prefetch._cancel_remote, "spec-job", room["world_id"], 1, ANY
        )

    def test_running_recommendation_is_joined_instead_of_restarted(self):
        room, suggestions = self.prepared_room()
        prepared, rendered = self._draft(room)
        self._seed_draft(room, suggestions, prepared, status="ready")
        future = futures.Future()
        self.assertTrue(future.set_running_or_notify_cancel())
        coop_prefetch._register(
            coop_prefetch._RECOMMENDATION_FUTURES, "recommend-test", future, request_id="recommend-test"
        )
        self._send_actions(room, suggestions)
        self.assertEqual(202, self.launch().status_code)
        fn, *args = self.pending.pop()
        self.assertIs(fn, coop._run_prefetched_round)
        try:
            with (
                patch.object(coop, "_execute_round") as fallback,
                patch.object(coop, "_after_round"),
                patch.object(coop_generation, "render_prepared") as render,
            ):
                worker = threading.Thread(target=fn, args=args)
                worker.start()
                self.assertTrue(worker.is_alive())
                future.set_result({"prepared": prepared, "rendered": rendered})
                worker.join(timeout=5)
                self.assertFalse(worker.is_alive())
                fallback.assert_not_called()
                render.assert_not_called()
        finally:
            args[2].close()
        result = self.a.get(self.url).json()
        self.assertEqual(2, result["round"])
        self.assertEqual("next-latent", result["rounds"][-1]["node"]["artifact_id"])
        self.cancel.assert_not_called()
        self.assertNotIn("recommend-test", coop_prefetch._RECOMMENDATION_FUTURES)


def load_tests(loader, tests, pattern):
    """Run this module's focused cases without repeating CoopTest's full suite."""
    suite = unittest.TestSuite()
    for name in loader.getTestCaseNames(CoopPrefetchTest):
        if name in CoopPrefetchTest.__dict__:
            suite.addTest(CoopPrefetchTest(name))
    return suite


if __name__ == "__main__":
    unittest.main()
