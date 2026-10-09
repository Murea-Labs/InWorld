"""Chapter asset hand-off, identity preservation and partial retries."""

import copy
import os
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from murea_agent import config
from murea_agent.coop import (
    room as coop,
    chapter as coop_chapter,
    story as coop_story,
    portrait as coop_portrait,
)
from murea_agent.coop import generation as coop_generation
from murea_agent.memory.world import World


class ChapterTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        for name, value in (("DATA", root), ("REFS", root / "refs")):
            p = patch.object(config, name, value)
            p.start()
            self.addCleanup(p.stop)
        config.REFS.mkdir()
        self.world = World(id="w-chapter")
        self.world.add_place("旧洞穴", "old.jpg", desc="洞穴")
        sid = self.world.add_subject("巨龙", "dragon.jpg", desc="红色巨龙", kind="生物")
        self.world.here = "旧洞穴"
        self.world.save()
        self.room = {
            "code": "ABCDEF123456",
            "world_id": self.world.id,
            "round": 3,
            "status": "waiting",
            "revision": 3,
            "state_revision": 3,
            "members": [],
            "rounds": [],
            "setting": "冒险",
            "visual_style": "cinematic",
            "scene_name": "旧洞穴",
            "scene_ref": "old.jpg",
            "scene_description": "洞穴",
            "monster": {"subject_id": sid, "name": "巨龙", "ref": "dragon.jpg", "description": "红色巨龙"},
            "story_outline": {"version": 1},
            "story_outline_job": {"request_id": "job"},
            "chapter_transition": {"status": "assets"},
        }
        for ref in ("old.jpg", "dragon.jpg", "valley.jpg", "newboss.jpg"):
            (config.REFS / ref).write_bytes(b"image")
        coop._write(coop._directory() / "ABCDEF123456.json", self.room)

    def plan(self):
        return {
            "entry": "两人已在山谷入口",
            "scene": {"mode": "new", "name": "山谷", "description": "山谷小径", "ref": "valley.jpg"},
            "monster": {"mode": "new", "name": "岩兽", "description": "岩石覆盖的生物", "ref": "newboss.jpg"},
        }

    def test_new_boss_preserves_old_identity_and_apply_is_idempotent(self):
        old = copy.deepcopy(self.world.subjects)
        coop_chapter.apply(self.room, self.plan())
        first_id = self.room["monster"]["subject_id"]
        coop_chapter.apply(self.room, self.plan())
        world = World.load(self.world.id)
        self.assertEqual(old["S1"], world.subjects["S1"])
        self.assertNotEqual("S1", first_id)
        self.assertEqual(first_id, self.room["monster"]["subject_id"])
        self.assertEqual(2, len(world.subjects))
        self.assertEqual("山谷", world.here)
        self.assertIn("旧洞穴", world.places)
        self.assertEqual("chapter", self.room["scene_asset_source"])

    def test_scene_change_starts_new_composition_and_keep_preserves_latent(self):
        head = SimpleNamespace(
            origin="coop", directive={"asset_edit_revision": 0}, refs=["old.jpg", "dragon.jpg"]
        )
        world = SimpleNamespace(head_node=head)
        self.assertTrue(coop_generation._continuation_decision(world, self.room, head.refs, [])["continue"])
        coop_chapter.apply(self.room, self.plan())
        decision = coop_generation._continuation_decision(world, self.room, ["valley.jpg", "newboss.jpg"], [])
        self.assertFalse(decision["continue"])

    def test_entry_is_used_only_until_first_new_chapter_video(self):
        self.room["rounds"] = [{"node": {"id": "old"}}]
        coop_chapter.apply(self.room, self.plan())
        self.assertEqual("两人已在山谷入口", coop_chapter.entry(self.room))
        self.room["rounds"].append({"node": {"id": "new"}})
        self.assertEqual("", coop_chapter.entry(self.room))

    def test_missing_asset_does_not_switch_room_or_world(self):
        plan = self.plan()
        plan["monster"]["ref"] = "missing.jpg"
        before = copy.deepcopy(self.room)
        with self.assertRaises(ValueError):
            coop_chapter.apply(self.room, plan)
        self.assertEqual(before, self.room)
        self.assertEqual("旧洞穴", World.load(self.world.id).here)

    def test_leaving_boss_removes_only_current_binding(self):
        plan = self.plan()
        plan["monster"] = {"mode": "none"}
        coop_chapter.apply(self.room, plan)
        self.assertEqual({}, self.room["monster"])
        self.assertIn("S1", World.load(self.world.id).subjects)

    def test_partial_image_failure_reuses_finished_scene(self):
        plan = self.plan()
        for k in ("scene", "monster"):
            plan[k].pop("ref")
        with (
            patch.object(coop_portrait, "draw", side_effect=[b"scene", RuntimeError("offline")]) as draw,
            patch.object(coop, "image_bytes", side_effect=lambda raw: raw),
        ):
            with self.assertRaisesRegex(RuntimeError, "offline"):
                coop_chapter.prepare(self.room, "job", plan)
            self.assertEqual(2, draw.call_count)
        with (
            patch.object(coop_portrait, "draw", return_value=b"boss") as draw,
            patch.object(coop, "image_bytes", side_effect=lambda raw: raw),
        ):
            ready = coop_chapter.prepare(self.room, "job", plan)
            self.assertEqual(1, draw.call_count)
        self.assertEqual(b"scene", (config.REFS / ready["scene"]["ref"]).read_bytes())
        self.assertEqual("old.jpg", self.room["scene_ref"])

    def test_legacy_chapter_requires_reconciliation_not_another_chapter(self):
        self.room["story_outline"] = {
            "chapter": 2,
            "version": 2,
            "stages": [{"id": "explore", "title": "探索", "goal": "探索"}],
            "active_stage_id": "explore",
        }
        self.room["rounds"] = [{"round": 2, "node": {"id": "n2"}}]
        self.assertTrue(coop_chapter.blocked(self.room))
        self.assertEqual(("reconcile_chapter", 0), coop_story._due(self.room))

    def test_all_abandoned_does_not_complete_goal(self):
        self.room["story_outline"] = {
            "chapter": 1,
            "version": 1,
            "stages": [{"id": "escape", "title": "逃离", "goal": "逃离"}],
        }
        self.room["story_progress"] = {"stages": {"escape": {"status": "abandoned"}}}
        self.assertFalse(coop_story.context(self.room)["chapter_complete"])
        self.room["story_progress"]["goal_result"] = {
            "verdict": "satisfied",
            "evidence": "两人位于洞外",
            "frame_indices": [3],
        }
        self.assertTrue(coop_story.context(self.room)["chapter_complete"])

    def test_worker_handoff_publishes_assets_and_outline_together(self):
        from murea_agent.coop import intro as coop_intro, trace as coop_trace

        def stage(sid):
            return {
                "id": sid,
                "title": sid,
                "goal": sid,
                "completion_evidence": "画面可见结果",
                "alternative": "换路",
            }

        self.room["rounds"] = [{"round": 2, "node": {"id": "n2"}, "inputs": []}]
        self.room["story_outline"] = {
            "version": 1,
            "chapter": 1,
            "overall_goal": "逃出洞穴",
            "stages": [stage("escape"), stage("arrive")],
            "active_stage_id": "escape",
        }
        self.room["story_progress"] = {
            "last_round": 2,
            "node_id": "n2",
            "version": 1,
            "stages": {k: {"status": "completed"} for k in ("escape", "arrive")},
        }
        coop._write(coop._directory() / "ABCDEF123456.json", self.room)
        plan = self.plan()
        for kind in ("scene", "monster"):
            plan[kind].pop("ref")
            plan[kind]["id"] = ""
        answer = {
            "overall_goal": "穿越山谷",
            "stages": [stage("cross"), stage("camp")],
            "active_stage_id": "cross",
            "abandon_stage_ids": [],
            "handoff": plan,
        }
        jobs = []
        with (
            patch.dict(os.environ, {"COOP_INTRO_ENABLED": "1", "COOP_STORY_ENABLED": "1"}),
            patch.object(coop_story._POOL, "submit", side_effect=lambda *args: jobs.append(args)),
            patch.object(coop_trace, "chat", return_value=answer) as chat,
            patch.object(coop_portrait, "draw", return_value=b"generated"),
            patch.object(coop, "image_bytes", side_effect=lambda raw: raw),
            patch.object(coop_intro, "queue") as guidance,
        ):
            self.assertTrue(coop_story.queue(self.room["code"]))
            fn, *args = jobs.pop()
            with coop._edit(self.room["code"], persist=False) as before:
                self.assertEqual("旧洞穴", before["scene_name"])
                self.assertEqual(1, before["story_outline"]["chapter"])
                self.assertTrue(coop_chapter.blocked(before))
            fn(*args)
            with coop._edit(self.room["code"], persist=False) as after:
                self.assertEqual("山谷", after["scene_name"])
                self.assertEqual("岩兽", after["monster"]["name"])
                self.assertEqual(2, after["story_outline"]["chapter"])
                self.assertTrue(after["story_outline"]["assets_ready"])
                self.assertFalse(coop_chapter.blocked(after))
                self.assertEqual("逃出洞穴", after["story_chapters"][0]["outline"]["overall_goal"])
            self.assertEqual(1, chat.call_count)
            guidance.assert_called_once_with(self.room["code"], force=True, history_only=True)


if __name__ == "__main__":
    unittest.main()
