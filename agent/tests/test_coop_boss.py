"""A dead Boss retires from generation without restarting the video chain."""

import copy
import json
import unittest
from unittest.mock import patch

from murea_agent import config
from murea_agent.coop import loop as coop_loop  # noqa: E402
from murea_agent.coop import (
    boss as coop_boss,
    intro as coop_intro,
    memory as coop_memory,
    prefetch as coop_prefetch,
    chapter as coop_chapter,
    observation as coop_observation,
)
from murea_agent.coop import generation as render
from murea_agent.coop import generation_prompts
from murea_agent.coop.scene_plan import AssetResolver, WorldSnapshot
from tests import test_coop_generation, test_coop_observation, test_coop_chapter


class BossLifeTest(unittest.TestCase):
    def setUp(self):
        self.room = {"monster": {"subject_id": "S3", "ref": "dragon.jpg"}, "story_state": {}}

    def test_only_explicit_old_observations_retire_a_boss(self):
        for text in ("巨龙已被击杀，尸体留在王座前", "The dragon is dead.", "A slain dragon lies here."):
            with self.subTest(text=text):
                self.room["story_state"]["boss"] = text
                self.assertEqual("dead", coop_boss.status(self.room))
        for text in (
            "巨龙被击败后倒地不动",
            "巨龙未死亡",
            "巨龙可能已经死亡",
            "尸体尚未出现",
            "击杀失败，巨龙逃走",
            "死亡之翼：红色巨龙，正在喷火",
            "巨龙倒伏在地，双眼红光微弱",
            "The dragon is not dead.",
            "The dragon appears dead.",
            "The dragon is motionless.",
            "No corpse is visible.",
        ):
            with self.subTest(text=text):
                self.room["story_state"]["boss"] = text
                self.assertEqual("unknown", coop_boss.status(self.room))
        self.room["monster"]["description"] = "已死亡的巨龙"
        self.room["intro"] = {"world_event": "击杀巨龙"}
        self.assertEqual("unknown", coop_boss.status(self.room))

    def test_bad_life_evidence_is_optional_and_not_a_death_fact(self):
        raw = {"status": "dead", "evidence": "巨龙尸体", "frame_indices": [2]}
        self.assertEqual(raw, coop_boss.validate(raw, 3))
        for indices in ([], [0], [4], [True], "2"):
            self.assertEqual({}, coop_boss.validate(dict(raw, frame_indices=indices), 3))
        self.assertEqual({}, coop_boss.validate(raw, 3, has_boss=False))

    def test_death_survives_missing_frames_and_a_later_alive_hallucination(self):
        dead = {"boss_life": {"status": "dead", "evidence": "Visible corpse", "frame_indices": [1]}}
        coop_boss.publish(self.room, dead, "n1", 1)
        for life in ({"status": "unknown"}, {"status": "alive"}):
            coop_boss.publish(self.room, {"boss_life": life}, "n2", 2)
            self.assertEqual("dead", coop_boss.status(self.room))
        self.room["story_state"]["boss_visible"] = False
        self.assertEqual("dead", coop_memory.current(self.room["story_state"])["boss_life"]["status"])

    def test_new_boss_does_not_inherit_old_death_or_facts(self):
        self.room["story_state"].update(
            boss="巨龙已死亡", facts={"boss.condition": {"value": "dead"}, "scene.passage": {"value": "open"}}
        )
        coop_boss.remember(self.room)
        self.room["monster"] = {"subject_id": "S4", "ref": "new.jpg"}
        coop_boss.bind(self.room)
        self.assertEqual("unknown", coop_boss.status(self.room))
        self.assertNotIn("boss.condition", self.room["story_state"]["facts"])
        self.assertIn("scene.passage", self.room["story_state"]["facts"])
        self.assertEqual("dead", self.room["story_state"]["boss_lifecycle"]["S3"]["status"])
        self.room["monster"] = {"subject_id": "S3", "ref": "dragon.jpg"}
        coop_boss.bind(self.room)
        self.assertEqual("dead", coop_boss.status(self.room))
        coop_boss.bind(self.room, reset=True)
        self.assertEqual("unknown", coop_boss.status(self.room))


class BossGenerationTest(unittest.TestCase):
    tearDown = test_coop_generation.JointShotTest.tearDown

    def setUp(self):
        test_coop_generation.JointShotTest.setUp(self)
        (config.REFS / "boss.jpg").write_bytes(b"boss")
        (config.REFS / "scene.jpg").write_bytes(b"scene")
        self.boss_id = self.world.add_subject("Dragon", "boss.jpg", kind="生物")
        self.room.update(
            scene_name="Bridge",
            scene_ref="scene.jpg",
            scene_description="A bridge",
            monster={
                "subject_id": self.boss_id,
                "ref": "boss.jpg",
                "name": "Dragon",
                "description": "Red scales",
            },
        )
        self.plan["monster_appearance"] = "A red dragon with broad wings."
        for action in self.plan["actions"]:
            action["action"] = "walks toward the exit along the stone path."
            action["beat"] = "walks toward the exit"

    def run_shot(self):
        plan = copy.deepcopy(self.plan)
        plan["actions"] = plan["actions"][:2]
        plan["timeline"] = generation_prompts._SERVER_TIMELINE
        with (
            patch.object(render.llm, "chat", return_value=plan) as planner,
            patch.object(render.video, "generate", return_value=self.gen) as gen,
            patch.object(render.coop_media, "grab_frames", return_value=[]),
        ):
            node = coop_loop.play_round(
                self.world,
                "boss-test",
                room=self.room,
                progress=lambda *a: None,
                cancelled=lambda: False,
                origin="coop",
                mentions=[m["subject_id"] for m in self.room["members"]],
            )
        return node, planner, gen

    def test_dead_reference_removed_and_remaining_scene_rebound_with_same_latent(self):
        for order in ("boss_first", "boss_last"):
            with self.subTest(order=order):
                self.world.head = None
                self.room.pop("story_state", None)
                self.run_shot()
                if order == "boss_last":
                    raw = self.world.nodes[self.world.head]
                    raw["refs"][-2:] = reversed(raw["refs"][-2:])
                    raw["pictures"][-2:] = reversed(raw["pictures"][-2:])
                self.room["story_state"] = {"boss": "巨龙已死亡，尸体留在原处", "boss_visible": True}
                node, planner, gen = self.run_shot()
                self.assertEqual("latent-1", gen.call_args.kwargs["continue_from"])
                self.assertTrue(gen.call_args.kwargs["require_continuation"])
                self.assertEqual([m["ref"] for m in self.members[:2]] + ["scene.jpg"], node.refs)
                self.assertNotIn(self.boss_id, node.subjects)
                self.assertNotIn("nonhuman creature shown", node.prompt)
                self.assertNotIn("shares the environment with the players", node.prompt)
                self.assertNotIn("<Subject 4>", node.prompt)
                self.assertEqual("dead", json.loads(planner.call_args.kwargs["user"])["boss_life"]["status"])
                self.assertEqual(
                    "dead_boss_reference_retired", node.directive["continuation_decision"]["reason"]
                )
                self.assertIsNone(node.directive["world_snapshot"]["creature"])
                self.assertIn(self.boss_id, self.world.subjects)
                self.assertTrue((config.REFS / "boss.jpg").exists())
                # Subsequent rounds keep the smaller reference set and chain.
                _, _, next_gen = self.run_shot()
                self.assertEqual("latent-1", next_gen.call_args.kwargs["continue_from"])

    def test_not_visible_or_knocked_down_does_not_remove_a_living_reference(self):
        self.room["story_state"] = {"boss_visible": False, "boss": "巨龙倒地不动"}
        _, _, gen = self.run_shot()
        self.assertIn(config.REFS / "boss.jpg", gen.call_args.kwargs["refs"])

    def test_death_cannot_bypass_manual_asset_edit_or_scene_removal(self):
        self.run_shot()
        self.room["story_state"] = {"boss": "巨龙已死亡"}
        self.room["asset_edit_revision"] = 1
        node, _, gen = self.run_shot()
        self.assertIsNone(gen.call_args.kwargs["continue_from"])
        self.assertEqual("user_asset_edit", node.directive["continuation_decision"]["reason"])
        self.room["scene_ref"] = ""
        _, _, gen = self.run_shot()
        self.assertIsNone(gen.call_args.kwargs["continue_from"])

    def test_dead_boss_is_not_reinserted_when_all_shared_references_are_empty(self):
        self.room.update(scene_ref="", story_state={"boss": "巨龙已死亡"})
        self.assertEqual(2, len(AssetResolver.resolve(self.members[:2], self.room, [])))
        self.assertEqual(2, len(AssetResolver.resolve(self.members[:2], self.room)))
        self.assertIsNone(WorldSnapshot.from_room(self.room).creature)
        node, _, gen = self.run_shot()
        self.assertEqual(2, len(node.refs))
        self.assertEqual(2, len(gen.call_args.kwargs["refs"]))

    def test_old_automated_boss_attack_is_omitted_without_losing_player_actions(self):
        self.room["story_state"] = {"boss": "巨龙已死亡"}
        self.room["intro"] = {
            "status": "ready",
            "world_event": "巨龙向玩家喷火",
            "world_event_type": "boss_action",
            "world_event_key": "dragon_breath",
            "boss_candidate": {},
        }
        self.plan["event_action"] = "The dragon breathes fire at the players."
        node, planner, _ = self.run_shot()
        self.assertNotIn("breathes fire", node.prompt)
        self.assertIn("walks toward the exit", node.prompt)
        self.assertEqual("", json.loads(planner.call_args.kwargs["user"])["upcoming_world_event"])
        self.assertEqual("none", node.directive["shot"]["world_event_type"])

    def test_restored_prefetch_keeps_retired_reference_out_of_final_request(self):
        self.run_shot()
        self.room["story_state"] = {"boss": "巨龙已死亡"}
        plan = copy.deepcopy(self.plan)
        plan["actions"] = plan["actions"][:2]
        plan["timeline"] = generation_prompts._SERVER_TIMELINE
        with patch.object(render.llm, "chat", return_value=plan):
            prepared = render.prepare(
                self.world, "", room=self.room, mentions=[m["subject_id"] for m in self.room["members"]]
            )
        restored = json.loads(json.dumps(coop_prefetch._serialize_prepared(prepared)))
        with (
            patch.object(render.video, "generate", return_value=self.gen) as gen,
            patch.object(render.coop_media, "grab_frames", return_value=[]),
        ):
            render.render_prepared(restored, progress=lambda *a: None, cancelled=lambda: False)
        self.assertEqual("latent-1", gen.call_args.kwargs["continue_from"])
        self.assertNotIn(config.REFS / "boss.jpg", gen.call_args.kwargs["refs"])
        self.assertNotIn("nonhuman creature shown", gen.call_args.args[0])

    def test_death_invalidates_persisted_living_prefetch_before_adoption(self):
        actions = {m["user_id"]: m["action"] for m in self.room["members"]}
        self.room["recommendation_prefetch"] = {
            "status": "ready",
            "request_id": "living",
            "actions": actions,
            "signature": coop_prefetch.recommendation_signature(self.room, actions),
            "_prepared": {"refs": ["boss.jpg"]},
        }
        self.room["story_state"] = {"boss": "巨龙已死亡"}
        coop_prefetch.recover(self.room)
        self.assertEqual("superseded", self.room["recommendation_prefetch"]["status"])
        self.assertIsNone(coop_prefetch.take(self.room))
        self.assertEqual(actions, {m["user_id"]: m["action"] for m in self.room["members"]})


class BossObservationTest(unittest.TestCase):
    setUp = test_coop_observation.CoopObservationTest.setUp
    save = test_coop_observation.CoopObservationTest.save
    read = test_coop_observation.CoopObservationTest.read
    enqueue = test_coop_observation.CoopObservationTest.enqueue
    observation = test_coop_observation.CoopObservationTest.observation
    run_job = test_coop_observation.CoopObservationTest.run_job

    def test_invalid_death_evidence_preserves_observation_without_retiring_boss(self):
        self.room["monster"] = {"subject_id": "dragon", "name": "巨龙", "ref": "boss.jpg"}
        self.save()
        self.enqueue()
        self.assertTrue(coop_observation.queue(self.room["code"]))
        response = self.observation()
        response["observation"].update(
            boss_state="巨龙已死亡", boss_life={"status": "dead", "evidence": "尸体", "frame_indices": [99]}
        )
        self.run_job(response)
        stored = self.read()
        self.assertEqual("ready", stored["story_observation"]["status"])
        self.assertEqual("unknown", coop_boss.status(stored))
        self.assertIn("boss.jpg", [r["ref"] for r in render._shared_references(stored)])

    def test_late_death_observation_cannot_kill_manually_replaced_boss(self):
        self.room["monster"] = {"subject_id": "dragon", "name": "巨龙", "ref": "boss.jpg"}
        self.save()
        self.enqueue()
        self.room["asset_edit_revision"] = 1
        self.room["monster"] = {"subject_id": "new-boss", "name": "石兽", "ref": "stone.jpg"}
        # Preserve the enqueued job while updating the actual room.
        stored = self.read()
        stored.update(asset_edit_revision=1, monster=self.room["monster"])
        from murea_agent.coop import room as coop

        coop._write(coop._directory() / f"{stored['code']}.json", stored)
        self.assertTrue(coop_observation.queue(self.room["code"]))
        response = self.observation()
        response["observation"]["boss_life"] = {
            "status": "dead",
            "evidence": "巨龙尸体",
            "frame_indices": [1],
        }
        self.run_job(response)
        stored = self.read()
        self.assertEqual("unknown", coop_boss.status(stored))
        self.assertEqual("new-boss", stored["monster"]["subject_id"])

    def test_visible_death_invalidates_ready_speculation_without_clearing_actions(self):
        self.room["monster"] = {
            "subject_id": "dragon",
            "name": "巨龙",
            "description": "红色巨龙",
            "ref": "boss.jpg",
        }
        for member in self.room["members"]:
            member.update(action="向出口移动", ready=True)
        old_sig = coop_prefetch.state_signature(self.room)
        intro_sig = coop_intro.signature(self.room)
        self.room["recommendation_prefetch"] = {"status": "ready", "request_id": "old", "job_id": "old-video"}
        self.save()
        self.enqueue()
        self.assertTrue(coop_observation.queue(self.room["code"]))
        response = self.observation(boss_visible=False)
        response["observation"]["boss_life"] = {
            "status": "dead",
            "evidence": "前一关键帧确认巨龙尸体",
            "frame_indices": [1],
        }
        with (
            patch.object(coop_prefetch, "cancel_job") as cancel,
            patch.object(coop_intro, "queue") as guidance,
        ):
            self.run_job(response)
        stored = self.read()
        self.assertEqual("dead", coop_boss.status(stored))
        self.assertFalse(stored["story_state"]["boss_visible"])
        self.assertEqual("superseded", stored["recommendation_prefetch"]["status"])
        self.assertTrue(all(m["ready"] and m["action"] == "向出口移动" for m in stored["members"]))
        self.assertNotEqual(old_sig, coop_prefetch.state_signature(stored))
        self.assertNotEqual(intro_sig, coop_intro.signature(stored))
        cancel.assert_called_once()
        guidance.assert_called_once_with(stored["code"], history_only=True)


class BossChapterTest(unittest.TestCase):
    setUp = test_coop_chapter.ChapterTest.setUp

    def test_same_scene_handoff_retires_dead_boss_and_preserves_continuation_revision(self):
        self.room["story_state"] = {"boss": "巨龙已死亡"}
        raw = {"scene": {"mode": "keep"}, "monster": {"mode": "keep"}, "entry": "从原位置向出口前进"}
        assets = coop_chapter.catalog(self.room)
        self.assertEqual([], assets["monster"])
        plan = coop_chapter.validate(raw, self.room, assets)
        self.assertEqual({"mode": "none"}, plan["monster"])
        coop_chapter.apply(self.room, plan)
        self.assertEqual({}, self.room["monster"])
        self.assertEqual(0, self.room.get("asset_edit_revision", 0))
        self.assertEqual("dead", self.room["story_state"]["boss_lifecycle"]["S1"]["status"])
        from types import SimpleNamespace

        head = SimpleNamespace(
            origin="coop",
            refs=["old.jpg", "dragon.jpg"],
            pictures=[{"kind": "scene"}, {"kind": "creature", "id": "S1"}],
            directive={"asset_edit_revision": 0, "monster": {"subject_id": "S1", "ref": "dragon.jpg"}},
        )
        decision = render._continuation_decision(SimpleNamespace(head_node=head), self.room, ["old.jpg"], [])
        self.assertTrue(decision["continue"])

    def test_archived_dead_identity_cannot_be_reused_as_new_chapter_boss(self):
        self.room["story_state"] = {"boss_identity": "new", "boss_lifecycle": {"S1": {"status": "dead"}}}
        assets = {"scene": [], "monster": [{"id": "S1", "ref": "dragon.jpg"}]}
        raw = {"scene": {"mode": "keep"}, "monster": {"mode": "reuse", "id": "S1"}, "entry": "走向出口"}
        with self.assertRaisesRegex(ValueError, "dead boss"):
            coop_chapter.validate(raw, self.room, assets)


if __name__ == "__main__":
    unittest.main()
