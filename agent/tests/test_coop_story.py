"""Actual-evidence progress, five-round outlines and asynchronous ownership."""

import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from murea_agent import config
from murea_agent.coop import (
    room as coop,
    intro as coop_intro,
    observation as coop_observation,
    prefetch as coop_prefetch,
    story as coop_story,
)


def outline():
    return {
        "overall_goal": "共同离开洞穴",
        "active_stage_id": "find_exit",
        "abandon_stage_ids": [],
        "stages": [
            {
                "id": sid,
                "title": title,
                "goal": title,
                "completion_evidence": evidence,
                "alternative": "受阻时换一条可见路线",
            }
            for sid, title, evidence in (
                ("find_exit", "发现出口", "画面清楚显示通往洞外的洞口"),
                ("reach_exit", "接近出口", "两人移动到洞口附近"),
                ("leave_cave", "离开洞穴", "两人在洞外同框"),
            )
        ],
    }


class CoopStoryTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.jobs = []
        for p in (
            patch.object(config, "DATA", Path(self.tmp.name)),
            patch.dict(os.environ, {"COOP_INTRO_ENABLED": "1", "COOP_STORY_ENABLED": "1"}),
            patch.object(coop_story._POOL, "submit", side_effect=lambda *args: self.jobs.append(args)),
        ):
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(lambda: [j[-1].close() for j in self.jobs])
        self.code = "ABCDEF123456"
        self.room = {
            "code": self.code,
            "world_id": "w-story",
            "round": 1,
            "revision": 1,
            "state_revision": 1,
            "status": "generating",
            "setting": "洞穴探险",
            "visual_style": "cinematic",
            "capacity": 2,
            "inflight": {"opening": True},
            "rounds": [],
            "story_state": {},
            "members": [
                {
                    "user_id": f"u{i}",
                    "subject_id": f"s{i}",
                    "name": f"玩家{i}",
                    "character": f"角色{i}",
                    "description": "轻甲",
                    "ref": f"{i}.jpg",
                    "action": "寻找出口",
                    "ready": False,
                }
                for i in (1, 2)
            ],
            "intro": {
                "world_event": "接下来碎石落下",
                "world_event_type": "environment",
                "world_event_key": "stones",
                "status": "ready",
                "objective": "寻找出口",
            },
        }
        self.save()

    def save(self):
        coop._write(coop._directory() / f"{self.code}.json", self.room)

    def read(self):
        with coop._edit(self.code, persist=False) as r:
            return r

    def run_job(self, response=None, side_effect=None):
        fn, *args = self.jobs.pop(0)
        with (
            patch.object(coop_story, "_OUTLINE_SYSTEM", coop_story._OUTLINE_SYSTEM),
            patch(
                "murea_agent.coop.trace.chat", return_value=response or outline(), side_effect=side_effect
            ) as chat,
        ):
            fn(*args)
        return chat

    def initial(self):
        coop_story.queue(self.code)
        self.run_job()
        self.room = self.read()

    def commit_observation(self, n, assessment="stalled", updates=None, status="ready"):
        self.room = self.read()
        node = {"id": f"n{n}", "frames": ["frame.jpg"], "clip": f"{n}.mp4"}
        self.room["rounds"].append(
            {
                "round": n,
                "node": node,
                "inputs": copy.deepcopy(self.room["members"]),
                "guidance": copy.deepcopy(self.room["intro"]),
                "observation": {"summary": f"第{n}轮可见事实"},
            }
        )
        self.room.update(round=n + 1, status="waiting")
        job = {"round": n, "node": node, "story_plan": coop_story.context(self.room)}
        delta = {
            "assessment": assessment,
            "summary": f"第{n}轮进度",
            "next_gap": "还需要抵达出口",
            "stage_updates": updates or [],
            "player_outcomes": [],
        }
        coop_story.record_progress(
            self.room, job, {"_story_progress": delta} if status == "ready" else None, status
        )
        self.save()
        return job

    def test_outline_runs_beside_opening_and_deduplicates_reads(self):
        self.assertTrue(coop_story.queue(self.code))
        self.assertFalse(coop_story.queue(self.code))
        self.assertEqual(1, len(self.jobs))
        self.assertEqual("generating", self.read()["status"])
        self.run_job()
        room = self.read()
        self.assertEqual(1, room["story_outline"]["version"])
        self.assertEqual("find_exit", coop_story.context(room)["active_stage_id"])
        self.assertEqual("generating", room["status"])
        self.assertEqual(1, room["state_revision"])
        self.assertFalse(coop_story.queue(self.code))

    def test_fifteen_rounds_revision_windows_include_missing_observations(self):
        self.initial()
        windows = []
        for n in range(1, 16):
            self.commit_observation(n, status="skipped" if n == 4 else "ready")
            queued = coop_story.queue(self.code)
            self.assertEqual(n % 5 == 0, queued)
            if queued:
                chat = self.run_job()
                payload = json.loads(chat.call_args.kwargs["user"])
                windows.append([r["round"] for r in payload["recent_rounds"]])
                if n == 5:
                    self.assertEqual("skipped", payload["recent_rounds"][3]["observation_status"])
                self.assertEqual(n, self.read()["story_outline"]["reviewed_round"])
        self.assertEqual([list(range(1, 6)), list(range(6, 11)), list(range(11, 16))], windows)
        self.assertEqual(4, self.read()["story_outline"]["version"])
        self.assertEqual(15, self.read()["story_progress"]["last_round"])

    def test_no_extra_progress_call_in_existing_observer(self):
        self.initial()
        job = {
            "node": {"id": "n1", "frames": ["frame.jpg"]},
            "members": self.room["members"],
            "setting": "洞穴",
            "shared_scene": {},
            "shared_creature": {},
            "current_forecast": {},
            "story_plan": coop_story.context(self.room),
        }
        response = {
            "observation": {
                "event_outcome": "unknown",
                "summary": "洞口显露",
                "environment_state": "洞口显露",
                "boss_visible": False,
                "boss_state": "",
                "player_states": [],
            },
            "story_phase": "exploration",
            "story_progress": {
                "assessment": "advanced",
                "summary": "发现出口",
                "next_gap": "接近出口",
                "stage_updates": [
                    {
                        "stage_id": "find_exit",
                        "status": "completed",
                        "completion_condition": outline()["stages"][0]["completion_evidence"],
                        "completion_verdict": "satisfied",
                        "evidence": "洞口可见",
                        "remaining": "",
                        "frame_indices": [1],
                    }
                ],
                "player_outcomes": [],
            },
        }
        with (
            patch.object(
                coop_observation.coop_generation, "frame_images", return_value=[("image/jpeg", b"frame")]
            ),
            patch.object(coop_observation.coop_trace, "chat", return_value=response) as chat,
        ):
            observation = coop_observation._observe(job, {})
        self.assertEqual(1, chat.call_count)
        self.assertIn("story_progress", chat.call_args.kwargs["schema"]["properties"])
        self.assertEqual("completed", observation["_story_progress"]["stage_updates"][0]["status"])
        self.assertEqual("unknown", observation["event_outcome"])
        # The stage can be visible even when this round had no forecast.
        self.assertEqual(2, len(observation["_story_progress"]["player_outcomes"]))

    def test_observer_worker_commits_progress_and_schedules_fifth_round_revision(self):
        self.initial()
        for n in range(1, 5):
            self.commit_observation(n)
        self.room = self.read()
        node = {"id": "n5", "frames": ["frame.jpg"]}
        self.room["rounds"].append({"round": 5, "node": node})
        snapshot = {**copy.deepcopy(self.room), "round": 5}
        self.room.update(round=6, status="waiting")
        coop_observation.enqueue(self.room, snapshot, node)
        self.save()
        response = {
            "observation": {
                "event_outcome": "unknown",
                "summary": "洞口显露",
                "environment_state": "洞口显露",
                "boss_visible": False,
                "boss_state": "",
                "player_states": [],
            },
            "story_phase": "exploration",
            "story_progress": {
                "assessment": "advanced",
                "summary": "发现出口",
                "next_gap": "接近出口",
                "stage_updates": [
                    {
                        "stage_id": "find_exit",
                        "status": "completed",
                        "completion_condition": outline()["stages"][0]["completion_evidence"],
                        "completion_verdict": "satisfied",
                        "evidence": "洞口可见",
                        "remaining": "",
                        "frame_indices": [1],
                    }
                ],
                "player_outcomes": [],
            },
        }
        with (
            patch.object(coop_observation._POOL, "submit", side_effect=lambda fn, *args: fn(*args)),
            patch.object(
                coop_observation.coop_generation, "frame_images", return_value=[("image/jpeg", b"frame")]
            ),
            patch.object(coop_observation.coop_trace, "chat", return_value=response) as chat,
        ):
            self.assertTrue(coop_observation.queue(self.code))
        room = self.read()
        self.assertEqual(1, chat.call_count)
        self.assertEqual(5, room["story_progress"]["last_round"])
        self.assertEqual("reach_exit", room["story_progress"]["active_stage_id"])
        self.assertEqual("洞口显露", room["rounds"][-1]["observation"]["summary"])
        self.assertEqual(room["story_progress"], room["rounds"][-1]["story_progress"])
        self.assertNotIn("_story_progress", room["rounds"][-1]["observation"])
        self.assertEqual("planning", room["story_outline_job"]["status"])
        self.assertEqual(5, self.jobs[0][3])

    def test_manual_edit_and_failed_observation_still_count_toward_review(self):
        self.initial()
        for n in range(1, 5):
            self.commit_observation(n)
        room = self.read()
        room["rounds"].append({"round": 5, "node": {"id": "n5"}})
        coop_story.record_progress(
            room,
            {"round": 5, "node": {"id": "n5"}, "story_plan": coop_story.context(room)},
            None,
            "failed",
            update_current=False,
        )
        self.assertEqual(4, room["story_progress"]["last_round"])
        self.assertEqual(("revise", 5), coop_story._due(room))
        payload = coop_story._payload(room, "revise", 5)
        self.assertEqual("failed", payload["recent_rounds"][-1]["observation_status"])
        self.assertIsNone(payload["recent_rounds"][-1]["observation"])

    def test_exhausted_review_retries_wait_for_new_five_round_window(self):
        self.initial()
        for n in range(1, 16):
            self.commit_observation(n)
        room = self.read()
        room["story_outline_job"] = {"status": "failed", "attempts": 3, "round": 5}
        self.assertEqual(("revise", 10), coop_story._due(room))
        room["story_outline_job"]["round"] = 10
        self.assertEqual(("revise", 15), coop_story._due(room))
        room["story_outline_job"]["round"] = 15
        self.assertIsNone(coop_story._due(room))

    def test_review_without_early_snapshots_does_not_use_future_progress(self):
        self.initial()
        self.commit_observation(6)
        room = self.read()
        self.assertEqual({}, coop_story._payload(room, "revise", 5)["progress"])
        self.assertEqual(6, coop_story._payload(room, "revise", 10)["progress"]["last_round"])

    def test_unknown_evidence_cannot_complete_goal_or_increment_stagnation(self):
        self.initial()
        self.commit_observation(1)
        self.commit_observation(2, status="failed")
        self.assertEqual(1, self.read()["story_progress"]["stagnation_rounds"])
        self.commit_observation(3)
        self.assertTrue(coop_story.context(self.read())["needs_new_approach"])
        raw = {
            "assessment": "unknown",
            "summary": "未看清",
            "next_gap": "",
            "player_outcomes": [],
            "stage_updates": [
                {
                    "stage_id": "find_exit",
                    "status": "completed",
                    "frame_indices": [1],
                    "evidence": "玩家要求寻找出口",
                    "remaining": "",
                }
            ],
        }
        with self.assertRaises(ValueError):
            coop_story.validate_progress(raw, coop_story.context(self.read()), 1, self.room["members"])
        raw["assessment"] = "advanced"
        raw["stage_updates"][0]["frame_indices"] = [9]
        with self.assertRaises(ValueError):
            coop_story.validate_progress(raw, coop_story.context(self.read()), 1, self.room["members"])

    def test_completed_stage_advances_and_cannot_regress(self):
        self.initial()
        change = {
            "stage_id": "find_exit",
            "status": "completed",
            "frame_indices": [1],
            "evidence": "洞口已显露",
            "remaining": "",
        }
        self.commit_observation(1, "advanced", [change])
        self.assertEqual("reach_exit", self.read()["story_progress"]["active_stage_id"])
        self.commit_observation(2, "stalled", [{**change, "status": "active"}])
        self.assertEqual("completed", self.read()["story_progress"]["stages"]["find_exit"]["status"])
        self.assertEqual("接近出口", self.read()["story_progress"]["stage_title"])

    def test_completion_rejects_unmet_or_rewritten_criteria(self):
        self.initial()
        stage = outline()["stages"][0]
        raw = {
            "assessment": "advanced",
            "summary": "已经进入后续情节",
            "next_gap": "",
            "player_outcomes": [],
            "stage_updates": [
                {
                    "stage_id": stage["id"],
                    "status": "completed",
                    "frame_indices": [1],
                    "completion_condition": stage["completion_evidence"],
                    "completion_verdict": "not_met",
                    "evidence": "画面只显示岩壁",
                    "remaining": "出口未见",
                }
            ],
        }
        plan = coop_story.context(self.room)
        with self.assertRaises(ValueError):
            coop_story.validate_progress(raw, plan, 1, self.room["members"])
        raw["stage_updates"][0].update(completion_verdict="satisfied", completion_condition="已经在战斗")
        with self.assertRaises(ValueError):
            coop_story.validate_progress(raw, plan, 1, self.room["members"])

    def test_abandoned_route_is_not_reactivated_by_a_late_observation(self):
        self.initial()
        room = self.read()
        job = {"round": 1, "node": {"id": "n1"}, "story_plan": coop_story.context(room)}
        room["rounds"] = [{"round": 1, "node": job["node"]}]
        room["story_progress"]["stages"]["find_exit"] = {"status": "abandoned"}
        delta = {
            "assessment": "stalled",
            "summary": "洞口被封住",
            "next_gap": "换路",
            "player_outcomes": [],
            "stage_updates": [
                {
                    "stage_id": "find_exit",
                    "status": "active",
                    "frame_indices": [1],
                    "evidence": "看见洞口",
                    "remaining": "开路",
                }
            ],
        }
        coop_story.record_progress(room, job, {"_story_progress": delta}, "ready")
        self.assertEqual("abandoned", room["story_progress"]["stages"]["find_exit"]["status"])
        self.assertEqual("reach_exit", room["story_progress"]["active_stage_id"])

    def stage_change(self, sid, status):
        return {
            "stage_id": sid,
            "status": status,
            "frame_indices": [1],
            "evidence": "当前画面确认的状态",
            "remaining": "尚未满足目标条件",
        }

    def test_blocked_earlier_goal_does_not_trap_focus_before_observed_next_stage(self):
        self.initial()
        self.commit_observation(
            1, updates=[self.stage_change("find_exit", "blocked"), self.stage_change("reach_exit", "active")]
        )
        room = self.read()
        for progress in (
            room["story_progress"],
            room["rounds"][-1]["story_progress"],
            coop_story.context(room)["progress"],
            coop._public(room)["story_progress"],
        ):
            self.assertEqual("reach_exit", progress["active_stage_id"])
            self.assertEqual("接近出口", progress["stage_title"])
            self.assertEqual("blocked", progress["stages"]["find_exit"]["status"])
            self.assertEqual(0, progress["completed_stages"])
        self.assertEqual("reach_exit", coop_story.context(room)["active_stage_id"])
        self.assertEqual("reach_exit", coop_story._payload(room, "revise", 5)["progress"]["active_stage_id"])

    def test_legacy_cursor_is_repaired_on_reads_without_rewriting_history_or_adopted_plan(self):
        self.initial()
        room = self.read()
        room["story_progress"].update(
            active_stage_id="find_exit",
            stage_title="发现出口",
            last_round=2,
            stages={
                "find_exit": {**self.stage_change("find_exit", "blocked"), "round": 2},
                "reach_exit": {**self.stage_change("reach_exit", "active"), "round": 2},
            },
        )
        room["story_progress"].pop("focus_round")
        room["story_progress"].pop("focus_source")
        room["rounds"] = [
            {"round": 1, "node": {"id": "n1"}, "story_progress": copy.deepcopy(room["story_progress"])}
        ]
        before = copy.deepcopy(room)
        public = coop._public(room)
        self.assertEqual("reach_exit", public["story_progress"]["active_stage_id"])
        self.assertEqual("reach_exit", coop_story.context(room)["active_stage_id"])
        self.assertEqual(before, room)
        self.assertEqual(before["rounds"], public["rounds"])

    def test_new_observed_focus_stays_current_when_blocked_or_observation_unknown(self):
        self.initial()
        self.commit_observation(1, updates=[self.stage_change("reach_exit", "active")])
        self.commit_observation(2, updates=[self.stage_change("leave_cave", "active")])
        self.assertEqual("leave_cave", self.read()["story_progress"]["active_stage_id"])
        self.commit_observation(3, updates=[self.stage_change("leave_cave", "blocked")])
        self.commit_observation(4, status="failed")
        plan = coop_story.context(self.read())
        self.assertEqual("leave_cave", plan["active_stage_id"])
        self.assertFalse(plan["chapter_complete"])
        self.assertEqual("blocked", plan["progress"]["stages"]["leave_cave"]["status"])

    def test_opening_observation_can_advance_focus_in_the_same_round_as_initial_outline(self):
        self.initial()
        self.commit_observation(
            0, updates=[self.stage_change("find_exit", "blocked"), self.stage_change("reach_exit", "active")]
        )
        self.assertEqual("reach_exit", self.read()["story_progress"]["active_stage_id"])

    def test_outline_can_choose_new_focus_without_old_active_evidence_overriding_it(self):
        self.initial()
        for n in range(1, 6):
            self.commit_observation(n, updates=[self.stage_change("reach_exit", "active")])
        coop_story.queue(self.code)
        self.run_job({**outline(), "active_stage_id": "leave_cave"})
        room = self.read()
        self.assertEqual("leave_cave", coop_story.context(room)["active_stage_id"])
        self.assertEqual("leave_cave", coop._public(room)["story_progress"]["active_stage_id"])

    def test_slow_outline_cannot_overwrite_focus_from_a_newer_observation(self):
        self.initial()
        for n in range(1, 6):
            self.commit_observation(n)
        coop_story.queue(self.code)
        self.commit_observation(6, updates=[self.stage_change("reach_exit", "active")])
        self.run_job()
        room = self.read()
        self.assertEqual("reach_exit", coop_story.context(room)["active_stage_id"])
        self.assertEqual(6, room["story_progress"]["focus_round"])

    def test_slow_revision_keeps_newer_progress_and_completed_conditions(self):
        self.initial()
        for n in range(1, 6):
            self.commit_observation(n)
        coop_story.queue(self.code)
        change = {
            "stage_id": "find_exit",
            "status": "completed",
            "frame_indices": [1],
            "evidence": "洞口已显露",
            "remaining": "",
        }
        self.commit_observation(6, "advanced", [change])
        altered = outline()
        altered["stages"][0]["completion_evidence"] = "新的更难的完成条件"
        self.run_job(altered)
        room = self.read()
        self.assertEqual(6, room["story_progress"]["last_round"])
        self.assertEqual("completed", room["story_progress"]["stages"]["find_exit"]["status"])
        self.assertEqual(outline()["stages"][0], room["story_outline"]["stages"][0])
        self.assertEqual("reach_exit", coop_story.context(room)["active_stage_id"])

    def test_progress_and_outline_do_not_invalidate_adopted_guidance_or_prefetch(self):
        self.initial()
        before = self.read()
        before["intro"]["signature"] = coop_intro.signature(before)
        before["recommendation_prefetch"] = {"status": "rendering", "job_id": "video-active"}
        self.room = before
        self.save()
        # Record only an observer update against an already committed row.
        self.room["rounds"] = [{"round": 1, "node": {"id": "n1"}}]
        self.room["round"] = 2
        self.save()
        before = self.read()
        signature = coop_prefetch.state_signature(before)
        intro_signature = coop_intro.signature(before)
        job = {"round": 1, "node": {"id": "n1"}, "story_plan": coop_story.context(before)}
        coop_story.record_progress(self.room, job, None, "skipped")
        self.save()
        after = self.read()
        self.assertEqual(signature, coop_prefetch.state_signature(after))
        self.assertEqual(intro_signature, coop_intro.signature(after))
        self.assertEqual(before["recommendation_prefetch"], after["recommendation_prefetch"])

    def test_restart_resumes_orphan_once_and_failed_submission_is_retryable(self):
        coop_story.queue(self.code)
        orphan = self.jobs.pop()
        orphan[-1].close()
        self.assertTrue(coop_story.queue(self.code))
        self.run_job()
        self.assertEqual("ready", self.read()["story_outline_job"]["status"])
        for n in range(1, 6):
            self.commit_observation(n)
        with patch.object(coop_story._POOL, "submit", side_effect=RuntimeError("executor unavailable")):
            self.assertFalse(coop_story.queue(self.code))
        self.assertEqual("failed", self.read()["story_outline_job"]["status"])
        self.assertFalse(coop_story.queue(self.code))
        with coop._edit(self.code) as room:
            room["story_outline_job"]["retry_at"] = 0
        self.assertTrue(coop_story.queue(self.code))
        self.run_job()
        self.assertEqual(2, self.read()["story_outline"]["version"])

    def test_manual_edit_or_closed_room_discards_outline(self):
        for mode in ("edit", "closed"):
            with self.subTest(mode=mode):
                self.save()
                coop_story.queue(self.code)
                with coop._edit(self.code) as room:
                    if mode == "edit":
                        room["asset_edit_revision"] = 4
                    else:
                        room["status"] = "closed"
                self.run_job()
                self.assertNotIn("story_outline", self.read())
                self.assertEqual("superseded", self.read()["story_outline_job"]["status"])

    def test_invalid_progress_keeps_visual_observation_and_snapshot(self):
        self.initial()
        job = {"round": 1, "node": {"id": "n1"}, "story_plan": coop_story.context(self.room)}
        self.room["rounds"] = [{"round": 1, "node": {"id": "n1"}}]
        coop_story.record_progress(self.room, job, {"summary": "事实保留"}, "ready")
        progress = copy.deepcopy(self.room["story_progress"])
        coop_story.record_progress(self.room, job, None, "failed")
        self.assertEqual(progress, self.room["story_progress"])
        self.assertEqual(1, len(self.room["story_progress_history"]))
        self.assertEqual("unknown", progress["assessment"])

    def test_complete_chapter_archives_progress_and_continues_new_outline(self):
        self.initial()
        changes = [
            {
                "stage_id": s["id"],
                "status": "completed",
                "frame_indices": [1],
                "evidence": s["completion_evidence"],
                "remaining": "",
            }
            for s in outline()["stages"]
        ]
        self.commit_observation(1, "advanced", changes)
        self.assertEqual(("next_chapter", 1), coop_story._due(self.read()))
        with coop._edit(self.code) as room:
            room.pop("inflight", None)
        from murea_agent.coop import chapter as coop_chapter

        handoff = {
            "scene": {"mode": "new", "id": "", "name": "山谷", "description": "开阔的山谷"},
            "monster": {"mode": "none", "id": "", "name": "", "description": ""},
            "entry": "两人已走到洞外",
        }
        coop_story.queue(self.code)
        with (
            patch.object(coop_chapter, "catalog", return_value={"scene": [], "monster": []}),
            patch.object(coop_chapter, "prepare", side_effect=lambda snapshot, request, plan: plan),
            patch.object(coop_chapter, "apply"),
            patch.object(coop_intro, "queue"),
        ):
            next_outline = outline()
            for stage in next_outline["stages"]:
                stage["id"] = "valley_" + stage["id"]
            next_outline["active_stage_id"] = next_outline["stages"][0]["id"]
            self.run_job({**next_outline, "overall_goal": "前往洞外山谷寻找营地", "handoff": handoff})
        room = self.read()
        self.assertEqual(2, room["story_outline"]["chapter"])
        self.assertEqual({}, room["story_progress"]["stages"])
        self.assertEqual(1, len(room["story_chapters"]))
        self.assertFalse(coop_story.context(room)["chapter_complete"])

    def test_next_chapter_rejects_reintroduced_old_stage_ids(self):
        payload = {"mode": "next_chapter", "old_outline": outline(), "progress": {}}
        with self.assertRaisesRegex(ValueError, "new stage ids"):
            coop_story._validate_outline(outline(), payload)
        payload["mode"] = "revise"
        self.assertEqual(outline()["stages"], coop_story._validate_outline(outline(), payload)["stages"])

    def test_guidance_rejects_completed_stage_and_carries_planning_versions(self):
        self.initial()
        self.commit_observation(
            1,
            "advanced",
            [
                {
                    "stage_id": "find_exit",
                    "status": "completed",
                    "frame_indices": [1],
                    "evidence": "看见洞口",
                    "remaining": "",
                }
            ],
        )
        room = self.read()
        payload, _ = coop_intro.guidance_payload(room, {}, include_images=False, history_only=True)
        schema = coop_intro.guidance_schema(
            latest=False, forecast_only=True, story_plan=payload["story_plan"]
        )
        self.assertNotIn("find_exit", schema["properties"]["stage_id"]["enum"])
        forecast = {
            "narration": "两人即将接近出口",
            "world_event": "接下来洞口旁的藤蔓将垂落",
            "world_event_type": "environment",
            "world_event_key": "vines_fall",
            "boss_candidate": {},
            "objective": "沿洞壁接近出口",
            "stage_id": "reach_exit",
            "expected_progress": "两人移至洞口旁",
        }
        reactions = {
            "world_event_key": "vines_fall",
            "suggestions": [
                {
                    "user_id": m["user_id"],
                    "action": "沿洞壁避开藤蔓走向洞口",
                    "forecast_cue": "藤蔓将垂落",
                    "response_reason": "避开藤蔓接近目标",
                }
                for m in room["members"]
            ],
        }
        with patch.object(coop_intro.coop_trace, "chat", side_effect=[forecast, reactions]):
            result, _ = coop_intro._plan_guidance(
                payload, room["members"], system=coop_intro._HISTORY_ONLY_SYSTEM, tag="test", temperature=0.6
            )
        self.assertEqual("reach_exit", result["story_basis"]["stage_id"])
        self.assertEqual(1, result["story_basis"]["evidence_through_round"])
        self.assertEqual(result["story_basis"], coop_intro.round_forecast({"intro": result})["story_basis"])

    def guidance_pair(self, stage="find_exit"):
        return (
            {
                "narration": "两人即将接近出口",
                "world_event": "接下来洞口旁的藤蔓将垂落",
                "world_event_type": "environment",
                "world_event_key": "vines_fall",
                "boss_candidate": {},
                "objective": "沿洞壁接近出口",
                "stage_id": stage,
                "expected_progress": "两人移至洞口旁",
            },
            {
                "world_event_key": "vines_fall",
                "suggestions": [
                    {
                        "user_id": m["user_id"],
                        "action": "沿洞壁避开藤蔓走向洞口",
                        "forecast_cue": "藤蔓将垂落",
                        "response_reason": "避开藤蔓接近目标",
                    }
                    for m in self.room["members"]
                ],
            },
        )

    def test_private_forecast_refreshes_once_when_goal_completes_between_calls(self):
        self.initial()
        payload, _ = coop_intro.guidance_payload(self.room, {}, include_images=False, history_only=True)
        self.commit_observation(
            1,
            "advanced",
            [
                {
                    "stage_id": "find_exit",
                    "status": "completed",
                    "frame_indices": [1],
                    "evidence": "看见洞口",
                    "remaining": "",
                }
            ],
        )
        fresh = coop_intro._fresh_story_payload(self.room, payload)
        with (
            patch.object(
                coop_intro.coop_trace,
                "chat",
                side_effect=[*self.guidance_pair(), *self.guidance_pair("reach_exit")],
            ) as chat,
            patch.object(coop_intro, "_fresh_story_payload", return_value=fresh) as refresh,
        ):
            result, _ = coop_intro._plan_guidance(
                payload,
                self.room["members"],
                system=coop_intro._HISTORY_ONLY_SYSTEM,
                tag="test",
                temperature=0.6,
                refresh=lambda: coop_intro._fresh_story_payload(self.room, payload),
            )
        self.assertEqual(4, chat.call_count)
        refresh.assert_called_once()
        self.assertEqual("reach_exit", result["story_basis"]["stage_id"])
        self.assertFalse(coop_story.stale_guidance(self.read(), result))
        self.assertNotIn("story_basis", self.read()["intro"])

    def test_stale_guidance_after_commit_requeues_without_polling(self):
        self.initial()
        self.room["inflight"] = {"request_id": "turn-1"}
        self.room["intro"]["signature"] = coop_intro.signature(self.room)
        self.save()
        snapshot = {**copy.deepcopy(self.room), "request_id": "turn-1"}
        forecast, reactions = self.guidance_pair()
        stale_result = {
            **forecast,
            "suggestions": reactions["suggestions"],
            "story_basis": coop_story.basis(coop_story.context(snapshot)),
        }
        guidance_jobs = []
        with (
            patch.object(coop_intro._POOL, "submit", side_effect=lambda *args: guidance_jobs.append(args)),
            patch.object(coop_prefetch, "queue_recommendation"),
        ):
            self.assertTrue(coop_intro.queue_history_guidance(snapshot))
            self.commit_observation(
                1,
                "advanced",
                [
                    {
                        "stage_id": "find_exit",
                        "status": "completed",
                        "frame_indices": [1],
                        "evidence": "看见洞口",
                        "remaining": "",
                    }
                ],
            )
            with coop._edit(self.code) as room:
                room["state_revision"] += 1
                room.pop("inflight")
            # Simulate a second material update after the bounded refresh.
            fn, *args = guidance_jobs.pop(0)
            with patch.object(coop_intro, "_plan_guidance", return_value=(stale_result, None)):
                fn(*args)
            self.assertEqual(1, len(guidance_jobs))
            self.assertNotIn("pending_guidance", self.read())
            self.assertEqual("generating", self.read()["intro"]["status"])
            fn, *args = guidance_jobs.pop(0)
            with patch.object(coop_intro.coop_trace, "chat", side_effect=self.guidance_pair("reach_exit")):
                fn(*args)
            self.assertEqual("ready", self.read()["intro"]["status"])
            self.assertEqual("reach_exit", self.read()["intro"]["story_basis"]["stage_id"])
        self.assertEqual([], guidance_jobs)


if __name__ == "__main__":
    unittest.main()
