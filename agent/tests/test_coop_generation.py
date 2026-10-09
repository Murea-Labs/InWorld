"""Validate the final multi-actor caption and the single Ref2VA call, offline."""

import copy
import re
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch, PropertyMock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
    os.environ.pop(k, None)
from murea_agent import config
from murea_agent.coop import generation_prompts
from murea_agent.coop import action_text
from murea_agent.coop import loop as coop_loop  # noqa: E402
from murea_agent.coop import (
    duration as coop_duration,
    generation as render,
    prefetch as coop_prefetch,
)
from murea_agent.providers import video
from murea_agent.memory.world import World
from murea_agent.providers.video import murea as murea_provider  # noqa: E402


def setUpModule():
    # These tests check timing against the configured 6-10 s lengths (the full Murea model).
    # The unreachable test server reads as Murea-Fast, whose 5 s cap is tested in
    # test_coop_duration.py.
    patcher = patch.object(murea_provider.Murea, "max_seconds", new_callable=PropertyMock, return_value=15)
    patcher.start()
    unittest.addModuleCleanup(patcher.stop)


class JointShotTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.patches = []
        # Most scenes here exercise continuations that also carry new reference images, a
        # capability flag of the provider; Murea's own behaviour is tested in MureaRoundTest.
        p = patch.object(render.video.murea.Murea, "references_on_continuation", True)
        p.start()
        self.patches.append(p)
        for key in ("DATA", "REFS", "CLIPS", "SESSIONS", "FRAMES"):
            path = self.root / key
            path.mkdir()
            p = patch.object(config, key, path)
            p.start()
            self.patches.append(p)
        for key, value in (
            ("COOP_OPENING_SECONDS", 6),
            ("COOP_ACTION_MIN_SECONDS", 6),
            ("COOP_ACTION_SECONDS", 10),
        ):
            p = patch.object(config, key, value)
            p.start()
            self.patches.append(p)
        self.world = World(title="共享场景", setting="共同对抗怪兽")
        self.members = []
        for i in range(4):
            ref = f"p{i}.jpg"
            (config.REFS / ref).write_bytes(b"reference")
            sid = self.world.add_subject(f"角色{i}", ref, desc=f"外观{i}")
            self.members.append(
                dict(
                    user_id=f"u{i}",
                    subject_id=sid,
                    character=f"角色{i}",
                    ref=ref,
                    description=f"外观{i}",
                    action=f"动作{i}",
                )
            )
        self.room = {
            "members": self.members[:2],
            "setting": "同一怪兽",
            "visual_style": "电影感",
            "scene_ref": "",
            "scene_name": "",
            "title": "冒险",
            "mode": "local",
        }
        self.plan = {
            "reply": "两人互相配合。",
            "scene": "An ancient bridge with one stone monster.",
            "camera": "One medium-wide shot with both characters visible.",
            "sound": "Metal and stone impacts.",
            "narration": "两人同时沿着古桥前进，怪兽的气息在前方逼近。",
            "actions": [
                {
                    "user_id": m["user_id"],
                    "source_action": m["action"],
                    "beat": f"performs the action {i}",
                    "reference_layout": "single full-body view",
                    "visibility": "full body",
                    "retention_level": "fully_preserved",
                    "retention": "The face, clothing colors and weapon remain visible while the body changes pose.",
                    "appearance": f"Character {i}.",
                    "action": f"performs their own action {i} against the stone monster.",
                }
                for i, m in enumerate(self.members)
            ],
        }
        self.plan["scene"] = (
            "An ancient stone bridge over a river, with weathered parapets and warm afternoon lighting."
        )
        self.plan["timeline"] = (
            "<Subject 1> performs their own action 0 against the stone monster while <Subject 2> performs their own action 1 against the stone monster."
        )
        self.gen = video.Generation(
            "joint-job", config.CLIPS / "joint.mp4", 4, 1, {"duration_seconds": 4}, artifact_id="latent-1"
        )
        self.gen.video_path.write_bytes(b"video")

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        self.tmp.cleanup()

    def run_shot(self, room=None):
        room = room or self.room
        plan = copy.deepcopy(self.plan)
        plan["actions"] = plan["actions"][: len(room["members"])]
        if len(room["members"]) > 2:
            plan["timeline"] += " " + " ".join(
                f"<Subject {i + 1}> {plan['actions'][i]['action']}" for i in range(2, len(room["members"]))
            )
        if room.get("monster", {}).get("ref"):
            plan["timeline"] += (
                f" <Subject {len(room['members']) + 1}> remains ahead of the players in the shared environment."
            )
        with (
            patch.object(render.llm, "chat", return_value=plan) as text,
            patch.object(render.video, "generate", return_value=self.gen) as gen,
            patch.object(render.coop_media, "grab_frames", return_value=[]),
        ):
            node = coop_loop.play_round(
                self.world,
                "unique-request",
                room=room,
                progress=lambda *a: None,
                cancelled=lambda: False,
                origin="coop",
                mentions=[m["subject_id"] for m in room["members"]],
            )
        return node, text, gen

    def test_without_continuation_the_same_scene_is_filmed_from_references(self):
        first, _text, _gen = self.run_shot()
        self.world.nodes[first.id]["origin"] = "coop"
        with patch.object(render.video.murea.PROVIDER, "supports_continuation", return_value=False):
            second, _text, gen = self.run_shot()
        self.assertIsNone(gen.call_args.kwargs.get("continue_from"))
        self.assertEqual("continuation_unavailable", second.directive["continuation_decision"]["reason"])

    def test_murea_continues_unchanged_rounds_with_a_reference_free_prompt(self):
        first, _text, _gen = self.run_shot()
        self.world.nodes[first.id]["origin"] = "coop"
        with patch.object(render.video.murea.Murea, "references_on_continuation", False):
            second, _text, gen = self.run_shot()
        self.assertEqual(first.artifact_id, gen.call_args.kwargs["continue_from"])
        self.assertNotIn("<Picture", gen.call_args.args[0])
        self.assertIn("<Picture 1>", gen.call_args.kwargs["continuation_fallback_prompt"])
        self.assertEqual("unchanged_references", second.directive["continuation_decision"]["reason"])

    def test_murea_films_a_new_asset_from_the_references_instead_of_continuing(self):
        opening, _, _ = self.run_shot()
        (config.REFS / "observed-scene.jpg").write_bytes(b"scene")
        room = dict(
            self.room,
            scene_ref="observed-scene.jpg",
            scene_name="Cave",
            scene_asset_source="opening_video",
            scene_asset_from_node=opening.id,
        )
        with patch.object(render.video.murea.Murea, "references_on_continuation", False):
            second, _text, gen = self.run_shot(room)
        self.assertIsNone(gen.call_args.kwargs.get("continue_from"))
        self.assertEqual("new_reference_needs_images", second.directive["continuation_decision"]["reason"])

    def test_a_round_runs_the_agent_loop_and_keeps_its_trace(self):
        node, _text, _gen = self.run_shot()
        steps = [(step["phase"], step["name"]) for step in node.trace]
        self.assertEqual(
            [
                ("perceive", "observe"),
                ("plan", "director"),
                ("check", "world_event"),
                ("act", "stage_assets"),
                ("act", "write_prompt"),
                ("act", "render_clip"),
                ("commit", "update_world"),
            ],
            steps,
        )
        self.assertTrue(all(step["status"] == "done" for step in node.trace))
        self.assertEqual(node.trace, self.world.nodes[node.id]["trace"])

    def test_both_modes_keep_every_reference_action_and_share_one_clip(self):
        for mode, count in [("local", 2), ("online", 4)]:
            room = dict(self.room, mode=mode, members=self.members[:count])
            node, text, gen = self.run_shot(room)
            text.assert_called_once()
            gen.assert_called_once()
            self.assertEqual(count, len(gen.call_args.kwargs["refs"]))
            for i in range(count):
                self.assertIn(f"<Subject {i + 1}> performs their own action {i}", node.prompt)
                self.assertIn(f"<Picture {i + 1}>", node.prompt)
            self.assertEqual("joint.mp4", node.clip)
            self.assertEqual("unique-request", node.user_text)
            self.assertTrue(node.six_section["summary"].startswith("[reference_generation]"))
            self.assertEqual(count, len(node.six_section["retention_analysis"].splitlines()))
            # The reference layout is fixed by the server, no longer asked from the model; see
            # test_the_reference_layout_is_never_described_as_a_sheet.
            self.assertIn(
                "as " + generation_prompts._REFERENCE_LAYOUT, node.six_section["subject_definitions"]
            )
            self.assertNotIn("share the shot", node.six_section["detailed_description"])
            self.assertEqual(self.plan["sound"], node.six_section["overall_soundscape"])
            self.assertNotIn("No speech or narration.", node.prompt)
            self.assertNotIn("Visual style:", node.prompt)
            self.assertEqual(count, len(node.directive["players"]))
            self.assertEqual(node.id, World.load(self.world.id).head)
            self.assertEqual(render.PROMPT_VERSION, node.directive["prompt_version"])
            self.assertIn(node.directive["shot"]["timeline"], node.six_section["detailed_description"])
            self.assertIn("timeline", text.call_args.kwargs["schema"]["required"])
            bindings = json.loads(text.call_args.kwargs["user"])["subject_bindings"]
            self.assertEqual(
                [f"<Subject {i}>" for i in range(1, count + 1)], [b["subject"] for b in bindings]
            )

    def test_cached_caption_replaces_legacy_template_clauses(self):
        plan = copy.deepcopy(self.plan)
        plan["actions"] = plan["actions"][:2]
        with patch.object(render.llm, "chat", return_value=plan):
            prepared = render.prepare(
                self.world,
                "cached-caption",
                room=self.room,
                mentions=[m["subject_id"] for m in self.room["members"]],
            )
        prepared = coop_prefetch._serialize_prepared(prepared)
        prepared["caption_context"]["cast_count"] = (
            "Exactly 2 people share the frame: <Subject 1>, <Subject 2>."
        )
        prepared["caption_context"]["style"] = "Visual style: cinematic lighting."
        prepared["caption_source"]["overall_soundscape"] = plan["sound"] + " No speech or narration."
        with patch.object(render.llm, "chat", side_effect=AssertionError("unexpected model call")):
            six = render._rebuild_prepared_caption(prepared)
        self.assertEqual(plan["sound"], six["overall_soundscape"])
        self.assertNotIn("share the shot", six["detailed_description"])
        self.assertIn("The scene's visual treatment is cinematic lighting.", six["detailed_description"])
        self.assertNotIn("Exactly 2 people", prepared["prompt"])
        self.assertNotIn("Visual style:", prepared["prompt"])
        self.assertNotIn("No speech or narration.", prepared["prompt"])

    def test_missing_duplicate_and_unknown_actor_abort_before_video(self):
        for entries in [self.plan["actions"][:1], [self.plan["actions"][0]] * 2, self.plan["actions"][2:]]:
            plan = dict(self.plan, actions=entries)
            with (
                patch.object(render.llm, "chat", return_value=plan),
                patch.object(render.video, "generate") as gen,
            ):
                with self.assertRaises(ValueError):
                    coop_loop.play_round(
                        self.world,
                        "x",
                        room=self.room,
                        progress=lambda *a: None,
                        cancelled=lambda: False,
                        origin="coop",
                        mentions=[m["subject_id"] for m in self.room["members"]],
                    )
                gen.assert_not_called()

    def test_long_four_player_turn_keeps_actions_without_generic_truncation(self):
        fixture = json.loads((ROOT / "tests/fixtures/coop_four_actor_caption.json").read_text())
        self.plan = copy.deepcopy(fixture["plan"])
        for member, action in zip(self.members, self.plan["actions"]):
            action["user_id"] = member["user_id"]
            member["action"] = action["source_action"]
        (config.REFS / "courtyard.jpg").write_bytes(b"scene")
        room = dict(
            self.room,
            members=self.members,
            mode="online",
            scene_ref="courtyard.jpg",
            scene_name="Courtyard",
            visual_style="Cinematic daylight.",
        )
        self.plan["timeline"] = generation_prompts._SERVER_TIMELINE
        with (
            patch.object(render.llm, "chat", return_value=self.plan),
            patch.object(render.video, "generate", return_value=self.gen) as gen,
            patch.object(render.coop_media, "grab_frames", return_value=[]),
        ):
            node = coop_loop.play_round(
                self.world,
                "x",
                room=room,
                progress=lambda *a: None,
                cancelled=lambda: False,
                origin="coop",
                mentions=[m["subject_id"] for m in room["members"]],
            )
        gen.assert_called_once()
        self.assertLessEqual(len(node.prompt), 6000)
        self.assertNotIn("[compressed]", node.prompt)
        for action in fixture["plan"]["actions"]:
            self.assertIn(action["action"], node.prompt)
            stored = next(a for a in node.directive["shot"]["actions"] if a["action"] == action["action"])
            self.assertEqual(action["path"], stored["path"])
            self.assertEqual(action["end_state"], stored["end_state"])

    def test_appending_a_scene_keeps_bindings_and_resets_latent(self):
        """New reference sets use a fresh shot while retaining Picture order."""
        self.run_shot()
        _, _, gen = self.run_shot()
        self.assertEqual("latent-1", gen.call_args.kwargs["continue_from"])
        self.assertTrue(gen.call_args.kwargs["require_continuation"])
        scene = "scene.jpg"
        (config.REFS / scene).write_bytes(b"scene")
        room = dict(self.room, members=self.members, scene_ref=scene, scene_name="古桥")
        node, _, gen = self.run_shot(room)
        self.assertIsNone(gen.call_args.kwargs["continue_from"], "新增参考图必须重新参考生成")
        self.assertEqual(5, len(gen.call_args.kwargs["refs"]))
        self.assertIn("<Picture 5>", node.prompt)
        self.assertEqual("scene", node.pictures[-1]["kind"])
        self.assertIn("<Subject 5>", node.six_section["detailed_description"])
        self.assertIn("<Subject 5> (appears in [Shot 1]", node.six_section["retention_analysis"])

    def test_a_creature_arriving_after_a_scene_does_not_renumber_the_scene(self):
        """New assets reset latent; the existing scene still keeps its number."""
        scene = "scene2.jpg"
        (config.REFS / scene).write_bytes(b"scene")
        beast = "beast.jpg"
        (config.REFS / beast).write_bytes(b"beast")
        two = self.members[:2]
        # round 0: only the two characters
        self.room["members"] = two
        self.run_shot()
        # round 1: the place is added
        with_scene = dict(self.room, members=two, scene_ref=scene, scene_name="山洞")
        node, _, gen = self.run_shot(with_scene)
        self.assertIsNone(gen.call_args.kwargs["continue_from"])
        self.assertEqual("scene", node.pictures[2]["kind"], "场景应在第 3 张")
        # round 2: the monster arrives later; the place keeps its position and the monster is
        # appended
        self.plan["monster_appearance"] = "A large armored dragon with golden eyes."
        with_both = dict(
            with_scene, monster={"name": "巨龙", "description": "岩甲巨龙", "subject_id": "M1", "ref": beast}
        )
        node, _, gen = self.run_shot(with_both)
        kinds = [p["kind"] for p in node.pictures]
        self.assertEqual(
            ["character", "character", "scene", "creature"], kinds, "怪兽必须追加在末尾，不能插到场景前面"
        )
        self.assertIsNone(gen.call_args.kwargs["continue_from"])
        # The environment anchor must point to the place's real number, never assume "the last image
        # is the place".
        self.assertIn("inside <Subject 3>", node.six_section["detailed_description"])
        self.assertIn("<Subject 4> is the single nonhuman creature", node.six_section["subject_definitions"])

    def test_assets_extracted_from_prior_video_continue_and_keep_picture_order(self):
        opening, _, _ = self.run_shot()
        (config.REFS / "observed-scene.jpg").write_bytes(b"scene")
        (config.REFS / "observed-boss.jpg").write_bytes(b"creature")
        room = dict(
            self.room,
            scene_ref="observed-scene.jpg",
            scene_name="Cave",
            scene_asset_source="opening_video",
            scene_asset_from_node=opening.id,
            opening_node={"id": opening.id},
        )
        scene_node, _, scene_gen = self.run_shot(room)
        self.assertEqual("latent-1", scene_gen.call_args.kwargs["continue_from"])
        self.assertTrue(scene_gen.call_args.kwargs["require_continuation"])
        self.assertEqual(
            "observed_assets_registered", scene_node.directive["continuation_decision"]["reason"]
        )
        self.plan["monster_appearance"] = "A large armored dragon with golden eyes."
        room["monster"] = {
            "name": "Dragon",
            "description": "An armored dragon.",
            "subject_id": "M1",
            "ref": "observed-boss.jpg",
            "source": "round_video",
            "from_node": scene_node.id,
        }
        boss_node, _, boss_gen = self.run_shot(room)
        self.assertEqual("latent-1", boss_gen.call_args.kwargs["continue_from"])
        self.assertEqual(
            ["character", "character", "scene", "creature"], [p["kind"] for p in boss_node.pictures]
        )
        self.assertIn("inside <Subject 3>", boss_node.prompt)
        self.assertIn("<Subject 4> is the single nonhuman creature", boss_node.prompt)
        self.assertEqual(4, len(boss_gen.call_args.kwargs["refs"]))
        self.assertTrue(boss_gen.call_args.kwargs["send_refs_on_continuation"])
        self.assertEqual("observed_assets_registered", boss_node.directive["continuation_decision"]["reason"])
        _, _, stable_gen = self.run_shot(room)
        self.assertEqual("latent-1", stable_gen.call_args.kwargs["continue_from"])

    def test_observed_replacement_keeps_creature_slot_and_continues(self):
        for ref in ("old-boss.jpg", "scene.jpg", "new-boss.jpg"):
            (config.REFS / ref).write_bytes(b"reference")
        self.plan["monster_appearance"] = "A large armored dragon with golden eyes."
        room = dict(
            self.room,
            scene_ref="scene.jpg",
            scene_name="Cave",
            monster={"name": "Dragon", "description": "A dragon.", "subject_id": "M1", "ref": "old-boss.jpg"},
        )
        first, _, _ = self.run_shot(room)
        self.assertEqual(["character", "character", "creature", "scene"], [p["kind"] for p in first.pictures])
        room["monster"] = {
            **room["monster"],
            "ref": "new-boss.jpg",
            "source": "round_video",
            "from_node": first.id,
        }
        second, _, generated = self.run_shot(room)
        self.assertEqual("latent-1", generated.call_args.kwargs["continue_from"])
        self.assertEqual(
            ["character", "character", "creature", "scene"], [p["kind"] for p in second.pictures]
        )
        self.assertIn("inside <Subject 4>", second.prompt)

    def test_manual_edit_with_same_image_breaks_chain_once(self):
        self.run_shot()
        room = dict(self.room, asset_edit_revision=1)
        node, _, generated = self.run_shot(room)
        self.assertIsNone(generated.call_args.kwargs["continue_from"])
        self.assertEqual("user_asset_edit", node.directive["continuation_decision"]["reason"])
        self.assertEqual(1, node.directive["asset_edit_revision"])
        _, _, next_gen = self.run_shot(room)
        self.assertEqual("latent-1", next_gen.call_args.kwargs["continue_from"])

    def test_extraction_from_other_branch_cannot_reuse_current_latent(self):
        self.run_shot()
        (config.REFS / "other-scene.jpg").write_bytes(b"scene")
        room = dict(
            self.room,
            scene_ref="other-scene.jpg",
            scene_name="Cave",
            scene_asset_source="opening_video",
            scene_asset_from_node="other-node",
        )
        _, _, generated = self.run_shot(room)
        self.assertIsNone(generated.call_args.kwargs["continue_from"])

    def test_old_opening_asset_uses_its_recorded_opening_node(self):
        opening, _, _ = self.run_shot()
        (config.REFS / "old-extracted-scene.jpg").write_bytes(b"scene")
        room = dict(
            self.room,
            scene_ref="old-extracted-scene.jpg",
            scene_name="Cave",
            scene_asset_source="opening_video",
            opening_node={"id": opening.id},
        )
        _, _, generated = self.run_shot(room)
        self.assertEqual("latent-1", generated.call_args.kwargs["continue_from"])

    def test_observed_asset_still_requires_parent_artifact(self):
        opening, _, _ = self.run_shot()
        self.world.nodes[opening.id]["artifact_id"] = None
        (config.REFS / "observed-scene.jpg").write_bytes(b"scene")
        room = dict(
            self.room,
            scene_ref="observed-scene.jpg",
            scene_name="Cave",
            scene_asset_source="opening_video",
            scene_asset_from_node=opening.id,
        )
        with self.assertRaisesRegex(ValueError, "latent continuation handle"):
            self.run_shot(room)

    def test_prepared_cache_keeps_latent_when_observed_reference_was_added(self):
        opening, _, _ = self.run_shot()
        (config.REFS / "observed-scene.jpg").write_bytes(b"scene")
        room = dict(
            self.room,
            scene_ref="observed-scene.jpg",
            scene_name="Cave",
            scene_asset_source="opening_video",
            scene_asset_from_node=opening.id,
        )
        plan = copy.deepcopy(self.plan)
        plan["actions"] = plan["actions"][:2]
        with patch.object(render.llm, "chat", return_value=plan):
            prepared = render.prepare(
                self.world, "x", room=room, mentions=[m["subject_id"] for m in room["members"]]
            )
        cached = coop_prefetch._serialize_prepared(prepared)
        with (
            patch.object(render.video, "generate", return_value=self.gen) as generated,
            patch.object(render.coop_media, "grab_frames", return_value=[]),
        ):
            render.render_prepared(cached, progress=lambda *a: None, cancelled=lambda: False)
        self.assertEqual("latent-1", generated.call_args.kwargs["continue_from"])
        self.assertTrue(generated.call_args.kwargs["require_continuation"])
        self.assertEqual(3, len(generated.call_args.kwargs["refs"]))
        self.assertEqual("observed_assets_registered", cached["continuation_decision"]["reason"])

    def test_bindings_follow_shared_reference_order(self):
        """Shared assets keep the previous beat's numbers: the place stays at 3, the monster is appended at 4.

        The environment anchor and monster uniqueness must read the real numbers, never assume "the
        last image is the place".
        """
        scene = "shared-scene.jpg"
        (config.REFS / scene).write_bytes(b"scene")
        beast = "shared-beast.jpg"
        (config.REFS / beast).write_bytes(b"beast")
        self.run_shot()
        with_scene = dict(self.room, scene_ref=scene, scene_name="山洞")
        self.run_shot(with_scene)
        with_both = dict(
            with_scene, monster={"name": "巨龙", "description": "岩甲巨龙", "subject_id": "M1", "ref": beast}
        )
        planned = copy.deepcopy(self.plan)
        planned["actions"] = planned["actions"][:2]
        planned["monster_appearance"] = "A large armored dragon with golden eyes."
        with (
            patch.object(render.llm, "chat", return_value=planned) as text,
            patch.object(render.video, "generate", return_value=self.gen),
            patch.object(render.coop_media, "grab_frames", return_value=[]),
        ):
            node = coop_loop.play_round(
                self.world,
                "x",
                room=with_both,
                progress=lambda *a: None,
                cancelled=lambda: False,
                origin="coop",
                mentions=[m["subject_id"] for m in with_both["members"]],
            )
        self.assertEqual(
            ["character", "character", "scene", "creature"], [item["kind"] for item in node.pictures]
        )
        self.assertIn("inside <Subject 3>", node.prompt)
        self.assertNotIn("shares the environment with the players", node.prompt)
        bindings = json.loads(text.call_args_list[0].kwargs["user"])["subject_bindings"]
        self.assertEqual("<Subject 3>", bindings[2]["subject"])
        self.assertIn("environment", bindings[2])
        self.assertEqual("<Subject 4>", bindings[3]["subject"])
        self.assertIn("creature", bindings[3])
        self.assertEqual(
            ["p0.jpg", "p1.jpg", scene, beast],
            [asset["ref"] for asset in node.directive["scene_plan"]["assets"]],
        )

    def test_a_changed_or_reordered_reference_still_breaks_the_latent(self):
        """If a number really changed, continuation must break: only appending is relaxed, not replacing images."""
        self.run_shot()
        swapped = "other.jpg"
        (config.REFS / swapped).write_bytes(b"other")
        members = copy.deepcopy(self.members)
        members[0] = dict(members[0], ref=swapped)
        room = dict(self.room, members=members)
        _, _, gen = self.run_shot(room)
        self.assertIsNone(gen.call_args.kwargs["continue_from"], "换掉一张图，<Subject 1> 已经不是同一个人")

    def test_locked_actions_and_server_camera_always_win(self):
        """A second presentation result from the model must be ignored: there is only one call now."""
        self.room["members"][0]["action"] = "绕过去"
        self.room["members"][1]["action"] = "我来挡住它"
        translated = copy.deepcopy(self.plan)
        translated["actions"] = translated["actions"][:2]
        for i, action in enumerate(
            [
                "moves in an arc around the dragon toward its rear.",
                "intercepts the dragon from the front and blocks its path.",
            ]
        ):
            translated["actions"][i].update(source_action=self.room["members"][i]["action"], action=action)
        translated["timeline"] = (
            "<Subject 1> moves in an arc around the dragon toward its rear while <Subject 2> intercepts the dragon from the front and blocks its path."
        )
        with (
            patch.object(render.llm, "chat", return_value=translated) as text,
            patch.object(render.video, "generate", return_value=self.gen),
            patch.object(render.coop_media, "grab_frames", return_value=[]),
        ):
            node = coop_loop.play_round(
                self.world,
                "x",
                room=self.room,
                progress=lambda *a: None,
                cancelled=lambda: False,
                origin="coop",
                mentions=[m["subject_id"] for m in self.room["members"]],
            )
        self.assertIn("<Subject 1> moves in an arc around the dragon", node.prompt)
        self.assertIn("<Subject 2> intercepts the dragon from the front", node.prompt)
        self.assertNotIn("back-to-back", node.prompt)
        text.assert_called_once()

    def test_actor_ownership_survives_the_shared_compiler(self):
        self.room["members"][0]["action"] = "绕到怪物左后方"
        self.room["members"][1]["action"] = "从正面挡住怪物"
        planned = copy.deepcopy(self.plan)
        planned["actions"] = planned["actions"][:2]
        planned["actions"][0].update(
            source_action=self.room["members"][0]["action"],
            action="runs in a left arc around the monster and reaches its rear flank.",
            beat="runs around the monster to its rear flank",
            target="the monster",
            path="a left arc around the monster",
            end_state="at the monster rear flank",
        )
        planned["actions"][1].update(
            source_action=self.room["members"][1]["action"],
            action="steps into the monster front path and holds a blocking stance.",
            beat="blocks the monster from the front",
            target="the monster",
            path="straight into its front path",
            end_state="between the monster and its forward route",
        )
        with (
            patch.object(render.llm, "chat", return_value=planned) as text,
            patch.object(render.video, "generate", return_value=self.gen),
            patch.object(render.coop_media, "grab_frames", return_value=[]),
        ):
            node = coop_loop.play_round(
                self.world,
                "x",
                room=self.room,
                progress=lambda *a: None,
                cancelled=lambda: False,
                origin="coop",
                mentions=[m["subject_id"] for m in self.room["members"]],
            )
        detail = node.six_section["detailed_description"]
        first = "<Subject 1> runs in a left arc around the monster and reaches its rear flank"
        second = "<Subject 2> steps into the monster front path and holds a blocking stance"
        self.assertEqual(1, detail.count(first))
        self.assertEqual(1, detail.count(second))
        self.assertNotIn("<Subject 1> steps into the monster front path", detail)
        self.assertNotIn("<Subject 2> runs in a left arc", detail)
        self.assertEqual("server:ShotCompiler", node.directive["shot"]["timeline_source"])
        text.assert_called_once()

    def test_placeholder_like_model_text_is_ignored_before_video(self):
        """The timeline returned by the model is discarded entirely; the server compiler produces the clip as usual."""
        planned = copy.deepcopy(self.plan)
        planned["actions"] = planned["actions"][:2]
        planned["timeline"] = (
            "{{ACTION:repeat}} then {{ACTION:repeat}} with {{UNKNOWN:field}} and {{BEAT:repeat}}."
        )
        with (
            patch.object(render.llm, "chat", return_value=planned) as text,
            patch.object(render.video, "generate", return_value=self.gen) as gen,
            patch.object(render.coop_media, "grab_frames", return_value=[]),
        ):
            node = coop_loop.play_round(
                self.world,
                "x",
                room=self.room,
                progress=lambda *a: None,
                cancelled=lambda: False,
                origin="coop",
                mentions=[m["subject_id"] for m in self.room["members"]],
            )
        self.assertEqual(1, text.call_count)
        gen.assert_called_once()
        self.assertEqual("server:ShotCompiler", node.directive["shot"]["timeline_source"])
        for i in (1, 2):
            self.assertIn(
                f"<Subject {i}> performs their own action", node.six_section["detailed_description"]
            )

    def test_continuation_fallback_records_the_submitted_compiler_prompt(self):
        prepared_plan = {
            "reply": "继续。",
            "timeline": "compiled body",
            "timeline_source": "server:ShotCompiler",
        }
        fallback_six = {"detailed_description": "fixed compiler body"}
        prepared = {
            "base_head": None,
            "plan": prepared_plan,
            "refs": [],
            "pictures": [],
            "players": [],
            "previous": None,
            "six": fallback_six,
            "fallback_six": fallback_six,
            "prompt": "fixed compiler body",
            "merge_seconds": 0,
            "world_snapshot": None,
            "scene_plan": None,
        }
        rendered = {
            "node_id": "nfallback",
            "generation": self.gen,
            "frames": [],
            "submitted_prompt": "fixed compiler body",
            "submitted_six": fallback_six,
            "used_prompt_fallback": True,
            "frame_seconds": 0,
        }
        room = dict(self.room, monster={})
        node = render.commit_prepared(
            self.world, "x", room=room, mentions=[], prepared=prepared, rendered=rendered, origin="coop"
        )
        self.assertEqual(
            "server:ShotCompiler:continuation_fallback", node.directive["shot"]["timeline_source"]
        )
        self.assertEqual("fixed compiler body", node.directive["shot"]["timeline"])
        self.assertNotIn("director_plan", node.directive["shot"])
        self.assertEqual("server:ShotCompiler", prepared_plan["timeline_source"], "提交时不能污染预生成缓存")

    def test_cached_action_contract_recovers_new_fields_for_old_entries(self):
        old = {
            member["user_id"]: {
                key: self.plan["actions"][index][key]
                for key in (
                    "appearance",
                    "action",
                    "beat",
                    "reference_layout",
                    "visibility",
                    "retention_level",
                )
            }
            for index, member in enumerate(self.room["members"])
        }
        recovered = render._cached_actions(self.room["members"], old)
        self.assertIsNotNone(recovered)
        for member in self.room["members"]:
            entry = recovered[member["user_id"]]
            self.assertEqual("", entry["start_state"])
            self.assertEqual("", entry["target"])
            self.assertEqual("", entry["path"])
            self.assertEqual("", entry["end_state"])
            self.assertEqual(
                render.PlannedAction.stable_id(member["user_id"], member["action"]), entry["action_id"]
            )

    def test_bound_subject_in_optional_action_metadata_is_normalized(self):
        plan = copy.deepcopy(self.plan)
        plan["actions"] = plan["actions"][:2]
        plan["actions"][0].update(
            target="<Subject 2>",
            path="moves around <Subject 2> from the left",
            end_state="beside <Subject 2>",
        )
        with (
            patch.object(render.llm, "chat", return_value=plan),
            patch.object(render.video, "generate", return_value=self.gen) as gen,
            patch.object(render.coop_media, "grab_frames", return_value=[]),
        ):
            node = coop_loop.play_round(
                self.world,
                "x",
                room=self.room,
                progress=lambda *a: None,
                cancelled=lambda: False,
                origin="coop",
                mentions=[m["subject_id"] for m in self.room["members"]],
            )
        gen.assert_called_once()
        action = node.directive["shot"]["actions"][0]
        self.assertEqual("the teammate", action["target"])
        self.assertEqual("moves around the teammate from the left", action["path"])
        self.assertEqual("beside the teammate", action["end_state"])

    def test_metadata_uses_actual_shared_reference_order(self):
        scene = "metadata-scene.jpg"
        (config.REFS / scene).write_bytes(b"scene")
        beast = "metadata-beast.jpg"
        (config.REFS / beast).write_bytes(b"beast")
        with_scene = dict(self.room, scene_ref=scene, scene_name="山洞")
        self.run_shot(with_scene)

        self.plan["monster_appearance"] = "A large armored dragon with golden eyes."
        self.plan["actions"][0]["target"] = "<Subject 4>"
        with_both = dict(
            with_scene,
            monster={
                "name": "巨龙",
                "description": "岩甲巨龙",
                "subject_id": "M1",
                "ref": beast,
            },
        )
        node, _, gen = self.run_shot(with_both)
        gen.assert_called_once()
        self.assertEqual(
            ["character", "character", "scene", "creature"], [item["kind"] for item in node.pictures]
        )
        self.assertEqual("the shared creature", node.directive["shot"]["actions"][0]["target"])

    def test_invalid_optional_metadata_reference_is_dropped_without_blocking_video(self):
        for token in ("<Subject 99>", "<Picture 1>"):
            plan = copy.deepcopy(self.plan)
            plan["actions"] = plan["actions"][:2]
            plan["actions"][0]["target"] = token
            with (
                self.subTest(token=token),
                patch.object(render.llm, "chat", return_value=plan),
                patch.object(render.video, "generate", return_value=self.gen) as gen,
                patch.object(render.coop_media, "grab_frames", return_value=[]),
            ):
                node = coop_loop.play_round(
                    self.world,
                    "x",
                    room=self.room,
                    progress=lambda *a: None,
                    cancelled=lambda: False,
                    origin="coop",
                    mentions=[m["subject_id"] for m in self.room["members"]],
                )
            gen.assert_called_once()
            self.assertEqual("", node.directive["shot"]["actions"][0]["target"])

    def test_cached_optional_metadata_subject_is_normalized(self):
        cached = {
            member["user_id"]: {
                **{
                    key: self.plan["actions"][index][key]
                    for key in (
                        "appearance",
                        "action",
                        "beat",
                        "reference_layout",
                        "visibility",
                        "retention_level",
                    )
                },
                "target": "<Subject 2>" if index == 0 else "<Subject 3>",
                "path": "",
                "end_state": "",
            }
            for index, member in enumerate(self.room["members"])
        }
        shared_refs = [{"kind": "monster", "ref": "monster.jpg"}]
        recovered = render._cached_actions(self.room["members"], cached, shared_refs=shared_refs)
        self.assertEqual("the teammate", recovered["u0"]["target"])
        self.assertEqual("the shared creature", recovered["u1"]["target"])

    def test_mistranslated_language_or_altered_source_never_reaches_video(self):
        for field, value in [("source_action", "换成格挡"), ("action", "站在一起格挡")]:
            plan = copy.deepcopy(self.plan)
            plan["actions"] = plan["actions"][:2]
            plan["actions"][0][field] = value
            with (
                patch.object(render.llm, "chat", return_value=plan),
                patch.object(render.video, "generate") as gen,
            ):
                with self.assertRaises(ValueError):
                    coop_loop.play_round(
                        self.world,
                        "x",
                        room=self.room,
                        progress=lambda *a: None,
                        cancelled=lambda: False,
                        origin="coop",
                        mentions=[m["subject_id"] for m in self.room["members"]],
                    )
                gen.assert_not_called()

    def test_actual_previous_frames_guide_motion_and_coop_durations(self):
        self.run_shot()
        name = "previous-frame.jpg"
        (config.FRAMES / name).write_bytes(b"actual last frame")
        self.world.nodes[self.world.head]["frames"] = [name]
        node, text, gen = self.run_shot()
        self.assertEqual(("image/jpeg", b"actual last frame"), text.call_args_list[0].kwargs["images"][-1])
        self.assertEqual(1, len(text.call_args_list[0].kwargs["images"]))
        self.assertNotIn(
            "appearance", text.call_args.kwargs["schema"]["properties"]["actions"]["items"]["required"]
        )
        self.assertEqual(6, gen.call_args.kwargs["duration"])
        self.assertIn("<Subject 1>", node.six_section["summary"])
        self.assertIn("<Subject 1>", node.prompt)
        self.assertTrue(gen.call_args.kwargs["send_refs_on_continuation"])
        self.assertIn("<Subject 1>", gen.call_args.kwargs["continuation_fallback_prompt"])
        self.assertIn(generation_prompts._CONTINUATION_CAMERA, node.prompt)
        self.assertTrue(gen.call_args.kwargs["require_continuation"])
        _, _, opening_gen = self.run_shot(dict(self.room, opening=True, intro={"narration": "序章"}))
        self.assertEqual(6, opening_gen.call_args.kwargs["duration"])

    def test_server_selects_coop_duration_from_complete_room_inputs(self):
        simple = coop_duration.decide(self.room)
        self.assertEqual((6, "compact"), (simple.seconds, simple.tier))

        with_event = copy.deepcopy(self.room)
        with_event["intro"] = {
            "status": "ready",
            "world_event": "与此同时，石桥边缘开始坍塌，怪兽向两人逼近。",
            "world_event_type": "environment",
            "world_event_key": "bridge_collapse",
        }
        self.plan["timing"] = {
            "steps": [
                {
                    "id": "players",
                    "actors": ["u0", "u1"],
                    "action": "move along bridge",
                    "seconds": 4,
                    "after": [],
                },
                {
                    "id": "collapse",
                    "actors": ["world_event"],
                    "action": "bridge edge collapses",
                    "seconds": 7,
                    "after": [],
                },
            ]
        }
        event = coop_duration.decide(with_event, self.plan["timing"])
        self.assertEqual((8, "standard"), (event.seconds, event.tier))
        self.assertEqual("structured", event.source)
        node, _, generated = self.run_shot(with_event)
        self.assertEqual(8, generated.call_args.kwargs["duration"])
        self.assertEqual(8, node.directive["scene_plan"]["duration"])
        self.assertEqual(8, node.directive["duration_policy"]["seconds"])
        self.assertIn("causal_action_timing", node.directive["duration_policy"]["reasons"])

        dialogue = copy.deepcopy(self.room)
        dialogue["members"][0]["action"] = "洛伊大喊“向左撤退”，随后举弓掩护队友。"
        extended = coop_duration.decide(dialogue)
        self.assertEqual((6, "compact"), (extended.seconds, extended.tier))

        no_dialogue = copy.deepcopy(self.room)
        no_dialogue["members"][0]["action"] = "洛伊靠近传说中名叫月影的石门。"
        self.assertEqual(6, coop_duration.decide(no_dialogue).seconds)

        four_players = dict(self.room, members=copy.deepcopy(self.members))
        crowded = coop_duration.decide(four_players)
        self.assertEqual(6, crowded.seconds)

        with patch.object(config, "COOP_ACTION_SECONDS", 8):
            long_action = copy.deepcopy(dialogue)
            long_action["members"][0]["action"] = "向前跑30米"
            self.assertEqual(8, coop_duration.decide(long_action).seconds)
        with patch.object(config, "COOP_ACTION_MIN_SECONDS", 4):
            self.assertEqual([4, 6, 8, 10], coop_duration.candidate_seconds())
            self.assertEqual(4, coop_duration.decide(self.room).seconds)
        with patch.object(config, "COOP_ACTION_SECONDS", 15):
            # A round that needs 9 s is filmed as 10 s, not 15 s.
            self.assertEqual([6, 8, 10, 12, 14, 15], coop_duration.candidate_seconds())

    def test_single_player_translation_uses_ceiling_before_teammate_action_arrives(self):
        member = copy.deepcopy(self.room["members"][0])
        room = copy.deepcopy(self.room)
        room["members"][1]["action"] = ""
        with patch.object(render, "_locked_actions", return_value={member["user_id"]: {}}) as locked:
            render.prefetch_member_action(self.world, room, member)
        self.assertEqual(10, locked.call_args.kwargs["duration"])

        room["intro"] = {"status": "ready", "world_event": "下一刻桥面开始坍塌。"}
        with patch.object(render, "_locked_actions", return_value={member["user_id"]: {}}) as locked:
            render.prefetch_member_action(self.world, room, member)
        self.assertEqual(10, locked.call_args.kwargs["duration"])

    def test_joint_and_cached_action_paths_estimate_once_then_freeze_duration(self):
        self.plan["actions"] = self.plan["actions"][:2]
        self.plan["timing"] = {
            "steps": [
                {
                    "id": "first",
                    "actors": ["u0"],
                    "action": "INTERNAL_TIMING_ONLY",
                    "seconds": 3,
                    "after": [],
                },
                {
                    "id": "second",
                    "actors": ["u1"],
                    "action": "follow through opening",
                    "seconds": 4,
                    "after": ["first"],
                },
            ]
        }
        args = dict(room=self.room, mentions=[m["subject_id"] for m in self.room["members"]])
        with patch.object(render.llm, "chat", return_value=self.plan) as chat:
            prepared = render.prepare(self.world, "duration-test", **args)
        chat.assert_called_once()
        self.assertEqual(10, json.loads(chat.call_args.kwargs["user"])["duration_seconds"])
        self.assertIn("timing", chat.call_args.kwargs["schema"]["required"])
        self.assertEqual(8, prepared["duration"])
        self.assertEqual(8, prepared["scene_plan"].duration)
        self.assertEqual(8, prepared["turn_context"].final_video_request["duration"])
        self.assertNotIn("INTERNAL_TIMING_ONLY", prepared["prompt"])
        cached = {a["user_id"]: a for a in prepared["plan"]["actions"]}
        with patch.object(render.llm, "chat", return_value=self.plan) as chat:
            from_cached = render.prepare(self.world, "duration-test", pretranslated=cached, **args)
        chat.assert_called_once()
        self.assertEqual("coop_environment_plan", chat.call_args.kwargs["tag"])
        self.assertEqual(prepared["duration_policy"], from_cached["duration_policy"])
        persisted = json.loads(json.dumps(coop_prefetch._serialize_prepared(from_cached)))
        with (
            patch.object(coop_duration, "decide", side_effect=AssertionError("must not re-estimate")),
            patch.object(render.llm, "chat", side_effect=AssertionError("must not re-plan")),
            patch.object(render.video, "generate", return_value=self.gen) as gen,
            patch.object(render.coop_media, "grab_frames", return_value=[]),
        ):
            render.render_prepared(persisted, progress=lambda *args: None, cancelled=lambda: False)
        gen.assert_called_once()
        self.assertEqual(8, gen.call_args.kwargs["duration"])
        self.assertEqual(prepared["duration_policy"], persisted["duration_policy"])
        self.assertNotIn("INTERNAL_TIMING_ONLY", gen.call_args.args[0])

    def test_bad_timing_does_not_retry_model_or_prevent_video_submission(self):
        self.plan["timing"] = {"steps": [{"id": "invalid", "seconds": "unknown"}]}
        node, chat, gen = self.run_shot()
        chat.assert_called_once()
        gen.assert_called_once()
        self.assertEqual("fallback", node.directive["duration_policy"]["source"])

    def test_opening_does_not_request_optional_action_timing(self):
        _, chat, gen = self.run_shot(dict(self.room, opening=True, intro={"narration": "序章"}))
        self.assertNotIn("timing", chat.call_args.kwargs["schema"]["properties"])
        self.assertEqual(6, gen.call_args.kwargs["duration"])

    def test_every_continuation_still_carries_reference_images(self):
        """A continuation must not turn reference images off: a service that requires them rejects it.

        Recorded video submissions:

            continuation + refs>=2  ->  accepted   54 times
            continuation + refs=0   ->  rejected    5 times (1-5 reference images are required)

        After the first co-op round every round is a continuation, so turning references off here
        would fail every round from the second on. The worry about frame 0 falling back to
        a white reference plate is handled by the fixed reference layout and continuity text, not by
        sending no images.
        """
        self.run_shot()
        for _ in range(3):
            node, _, gen = self.run_shot()
            kw = gen.call_args.kwargs
            self.assertEqual("latent-1", kw["continue_from"])
            self.assertTrue(kw["require_continuation"])
            self.assertTrue(kw["send_refs_on_continuation"], "续写不带参考图会被视频服务拒绝（记录中 5/5）")
            self.assertGreaterEqual(len(kw["refs"]), 1, "视频服务要求 1-5 张参考图")
            self.assertLessEqual(len(kw["refs"]), 5)
            self.assertNotIn("white or studio reference backdrop", node.prompt)

    def test_serialized_recommendation_continuation_still_carries_references(self):
        """After hitting a finished recommended prompt, restoring from disk must keep the continuation reference policy.

        A finished recommendation Future is removed from memory; when a player adopts the
        recommendation later, the normal path rebuilds it from `_prepared` in the room JSON.
        `turn_context` is deliberately not serialized, so the policy is kept separately. Requests
        saved by older versions lack the field and are restored to sending images with a
        continuation, as they did.
        """
        self.run_shot()
        plan = copy.deepcopy(self.plan)
        plan["actions"] = plan["actions"][: len(self.room["members"])]
        with patch.object(render.llm, "chat", return_value=plan):
            prepared = render.prepare(
                self.world,
                "unique-request",
                room=self.room,
                mentions=[m["subject_id"] for m in self.room["members"]],
            )
        saved = coop_prefetch._serialize_prepared(prepared)
        self.assertNotIn("turn_context", saved)
        self.assertTrue(saved["send_refs_on_continuation"])

        legacy = copy.deepcopy(saved)
        legacy.pop("send_refs_on_continuation")
        for label, restored in (("current", saved), ("legacy", legacy)):
            with (
                self.subTest(cache=label),
                patch.object(render.video, "generate", return_value=self.gen) as gen,
                patch.object(render.coop_media, "grab_frames", return_value=[]),
            ):
                render.render_prepared(
                    restored,
                    progress=lambda *a: None,
                    cancelled=lambda: False,
                )
            self.assertTrue(gen.call_args.kwargs["send_refs_on_continuation"])
            self.assertEqual("latent-1", gen.call_args.kwargs["continue_from"])
            self.assertGreaterEqual(len(gen.call_args.kwargs["refs"]), 1)

    def test_missing_latent_stops_instead_of_resetting_an_existing_scene(self):
        self.run_shot()
        self.world.nodes[self.world.head]["artifact_id"] = None
        with patch.object(render.video, "generate") as gen, patch.object(render.llm, "chat") as llm:
            with self.assertRaisesRegex(ValueError, "latent"):
                coop_loop.play_round(
                    self.world,
                    "x",
                    room=self.room,
                    progress=lambda *a: None,
                    cancelled=lambda: False,
                    origin="coop",
                    mentions=[m["subject_id"] for m in self.room["members"]],
                )
            gen.assert_not_called()
            llm.assert_not_called()

    def test_custom_action_continues_committed_latent_not_discarded_recommendation(self):
        self.run_shot()
        self.room["recommendation_prefetch"] = {
            "status": "superseded",
            "_rendered": {"generation": {"artifact_id": "unselected-latent", "video_id": "unselected-job"}},
        }
        self.room["members"][0]["action"] = "留在岩壁旁检查背包"
        self.plan["actions"][0].update(
            source_action="留在岩壁旁检查背包",
            action="examines the backpack beside the rock wall.",
            beat="examines the backpack beside the rock wall",
        )
        self.plan["timeline"] = (
            "<Subject 1> examines the backpack beside the rock wall while "
            "<Subject 2> performs their own action 1 against the stone monster."
        )
        node, _, gen = self.run_shot()
        self.assertEqual("latent-1", gen.call_args.kwargs["continue_from"])
        self.assertTrue(gen.call_args.kwargs["require_continuation"])
        self.assertTrue(gen.call_args.kwargs["allow_continuation_fallback"])
        self.assertIn("examines the backpack", node.prompt)

    def test_shared_monster_and_scene_are_reference_bound_and_continue_together(self):
        for name in ("monster.jpg", "scene.jpg"):
            (config.REFS / name).write_bytes(name.encode())
        self.room.update(
            monster={
                "name": "岩甲巨龙",
                "description": "灰鳞金眼的巨龙",
                "ref": "monster.jpg",
                "subject_id": "S5",
            },
            scene_name="山洞宝库",
            scene_description="金色光线的空旷岩洞",
            scene_ref="scene.jpg",
        )
        self.plan["monster_appearance"] = (
            "A massive dragon with gray rocky scales, golden eyes, short horns and four sturdy legs."
        )
        node, text, gen = self.run_shot()
        self.assertEqual(["p0.jpg", "p1.jpg", "monster.jpg", "scene.jpg"], node.refs)
        self.assertEqual(["character", "character", "creature", "scene"], [p["kind"] for p in node.pictures])
        self.assertIn("<Subject 3> is the single nonhuman creature shown in <Picture 3>", node.prompt)
        self.assertIn("<Subject 4> is the environment shown in <Picture 4>", node.prompt)
        self.assertNotIn("shares the environment with the players", node.prompt)
        self.assertIn("S5", node.subjects)
        self.assertEqual(2, len(node.directive["players"]))
        self.assertEqual(
            [b"reference", b"reference", b"monster.jpg", b"scene.jpg"],
            [raw for _, raw in text.call_args.kwargs["images"]],
        )
        self.assertEqual(4, len(gen.call_args.kwargs["refs"]))
        self.assertNotIn("dragon", node.six_section["subject_definitions"].splitlines()[-1].lower())
        for line in node.six_section["retention_analysis"].splitlines()[:3]:
            self.assertNotIn("white or studio reference backdrop", line)
            self.assertNotIn("reference floor, cast shadows, reference lighting", line)
        self.assertNotIn("reference backdrop", node.six_section["retention_analysis"].splitlines()[-1])
        _, _, next_gen = self.run_shot()
        self.assertEqual("latent-1", next_gen.call_args.kwargs["continue_from"])
        (config.REFS / "monster2.jpg").write_bytes(b"new monster")
        self.room["monster"]["ref"] = "monster2.jpg"
        _, _, changed = self.run_shot()
        self.assertIsNone(changed.call_args.kwargs["continue_from"])

    def test_chinese_shared_scene_and_monster_are_repaired_before_video(self):
        for name in ("monster.jpg", "scene.jpg"):
            (config.REFS / name).write_bytes(name.encode())
        self.room.update(
            monster={
                "name": "岩甲巨龙",
                "description": "灰鳞金眼的巨龙",
                "ref": "monster.jpg",
                "subject_id": "S5",
            },
            scene_name="山洞宝库",
            scene_description="金色光线的空旷岩洞",
            scene_ref="scene.jpg",
            visual_style="cinematic",
        )
        bad = copy.deepcopy(self.plan)
        bad["actions"] = bad["actions"][:2]
        bad["scene"] = "昏暗的金币山洞，中央有一块黑色岩石平台。"
        bad["monster_appearance"] = "一只覆盖黑红鳞片、长有灰白弯角和巨大翅膀的巨龙。"
        bad["timeline"] += " <Subject 3> remains ahead of the players in the shared cave."
        repaired = {
            "scene": "Dim gold-filled cave with a central dark rock platform and rough stone walls.",
            "monster_appearance": "Massive quadrupedal dragon with black and dark red scales, curved pale horns and broad wings.",
        }
        with (
            patch.object(render.llm, "chat", side_effect=[bad, repaired]) as text,
            patch.object(render.video, "generate", return_value=self.gen) as gen,
            patch.object(render.coop_media, "grab_frames", return_value=[]),
        ):
            node = coop_loop.play_round(
                self.world,
                "x",
                room=self.room,
                progress=lambda *a: None,
                cancelled=lambda: False,
                origin="coop",
                mentions=[m["subject_id"] for m in self.room["members"]],
            )
        self.assertEqual(2, text.call_count)
        self.assertIn("Massive quadrupedal dragon", node.prompt)
        self.assertNotRegex(node.prompt, r"[\u3400-\u9fff]")
        gen.assert_called_once()

    def test_chinese_timeline_is_repaired_instead_of_failing_the_round(self):
        """`timeline` was guarded by `_english()` but never in the repair list: any Chinese in it failed the whole round.

        The `fields` list decides both who gets strict validation and who gets repaired, and
        `timeline` was validated separately (limit 4000 plus subject-binding checks), so it could be
        rejected but never repaired.

        In recorded rounds, `timeline` was 3 of the rejected translation fields and never appeared in
        `action_translation_repaired`. `_repair_plan_english_fields` supported timeline all along;
        it was never given it.
        """
        bad = copy.deepcopy(self.plan)
        bad["actions"] = bad["actions"][:2]
        bad["timeline"] = "<Subject 1> 向前突进，<Subject 2> 举剑格挡在侧翼。"
        repaired = {
            "timeline": (
                "<Subject 1> lunges forward across the stone floor while "
                "<Subject 2> raises a sword and guards the left flank, both fully visible."
            )
        }
        with (
            patch.object(render.llm, "chat", side_effect=[bad, repaired]) as text,
            patch.object(render.video, "generate", return_value=self.gen) as gen,
            patch.object(render.coop_media, "grab_frames", return_value=[]),
        ):
            node = coop_loop.play_round(
                self.world,
                "x",
                room=self.room,
                progress=lambda *a: None,
                cancelled=lambda: False,
                origin="coop",
                mentions=[m["subject_id"] for m in self.room["members"]],
            )
        self.assertEqual(2, text.call_count, "应该只多一次修复调用，不是整轮重来")
        # The legacy model timeline is repaired for compatibility but discarded;
        # ShotCompiler's deterministic timeline is the only video choreography.
        self.assertEqual("server:ShotCompiler", node.directive["shot"]["timeline_source"])
        self.assertNotRegex(node.prompt, r"[\u3400-\u9fff]")
        gen.assert_called_once()

    def test_timeline_joins_the_contamination_scan(self):
        """The scan list must include timeline: the root of the bug above."""
        self.assertIn("timeline", action_text._repairable_plan_fields(["scene", "sound"]))
        self.assertIn(
            "timeline", action_text._repairable_plan_fields(["scene", "sound", "monster_appearance"])
        )

    def test_repair_accepts_a_long_timeline_the_caller_would_accept(self):
        """A field must use the same length limit in "validate" and "repair".

        Overlong `_text` **raises** instead of truncating. The caller gives timeline 4000, while the
        repair used `_english`'s default 1600, so a valid long timeline was rejected by its own
        repair, used up 3 retries and failed the round. The bug is only reachable once timeline
        enters the repair list, so both are pinned together.
        """
        long_tail = " The camera holds steady on the shared stone bridge." * 40
        self.assertGreater(len(long_tail), 1600)
        self.assertLess(len(long_tail), action_text._plan_field_limit("timeline"))
        bad = copy.deepcopy(self.plan)
        bad["actions"] = bad["actions"][:2]
        bad["timeline"] = "<Subject 1> 与 <Subject 2> 一起前进。"
        repaired = {"timeline": self.plan["timeline"] + long_tail}
        with (
            patch.object(render.llm, "chat", side_effect=[bad, repaired]) as text,
            patch.object(render.video, "generate", return_value=self.gen) as gen,
            patch.object(render.coop_media, "grab_frames", return_value=[]),
        ):
            node = coop_loop.play_round(
                self.world,
                "x",
                room=self.room,
                progress=lambda *a: None,
                cancelled=lambda: False,
                origin="coop",
                mentions=[m["subject_id"] for m in self.room["members"]],
            )
        self.assertEqual(2, text.call_count, "合法的长 timeline 不该触发重试")
        self.assertEqual("server:ShotCompiler", node.directive["shot"]["timeline_source"])
        gen.assert_called_once()

    def test_field_limit_has_a_single_source(self):
        self.assertEqual(4000, action_text._plan_field_limit("timeline"))
        self.assertEqual(1600, action_text._plan_field_limit("scene"))
        self.assertEqual(1600, action_text._plan_field_limit("monster_appearance"))

    def test_reference_overflow_or_missing_shared_ref_stops_before_models(self):
        self.room.update(
            members=self.members,
            monster={"name": "巨龙", "description": "金眼", "ref": "monster.jpg", "subject_id": "S5"},
            scene_name="山洞",
            scene_ref="scene.jpg",
        )
        for name in ("monster.jpg", "scene.jpg"):
            (config.REFS / name).write_bytes(b"ref")
        with patch.object(render.llm, "chat") as text, patch.object(render.video, "generate") as gen:
            with self.assertRaisesRegex(ValueError, "more than five"):
                coop_loop.play_round(
                    self.world,
                    "x",
                    room=self.room,
                    progress=lambda *a: None,
                    cancelled=lambda: False,
                    origin="coop",
                    mentions=[m["subject_id"] for m in self.members],
                )
            self.room["members"] = self.members[:2]
            (config.REFS / "monster.jpg").unlink()
            with self.assertRaisesRegex(ValueError, "references"):
                coop_loop.play_round(
                    self.world,
                    "x",
                    room=self.room,
                    progress=lambda *a: None,
                    cancelled=lambda: False,
                    origin="coop",
                    mentions=[m["subject_id"] for m in self.members[:2]],
                )
            text.assert_not_called()
            gen.assert_not_called()

    def test_video_is_published_before_frame_work_and_locked_traits_do_not_drift(self):
        first, _, _ = self.run_shot()
        self.plan["actions"][0]["appearance"] = "A different redesigned character."
        ready = []

        def extract(*args):
            self.assertEqual(["joint.mp4"], ready)
            return []

        plan = copy.deepcopy(self.plan)
        plan["actions"] = plan["actions"][:2]
        with (
            patch.object(render.llm, "chat", return_value=plan),
            patch.object(render.video, "generate", return_value=self.gen),
            patch.object(render.coop_media, "grab_frames", side_effect=extract),
        ):
            node = coop_loop.play_round(
                self.world,
                "x",
                room=self.room,
                progress=lambda *a: None,
                cancelled=lambda: False,
                origin="coop",
                mentions=[m["subject_id"] for m in self.room["members"]],
                on_video_ready=ready.append,
            )
        self.assertEqual(
            first.six_section["subject_definitions"], node.directive["reference_subject_definitions"]
        )
        # This continuation beat still carries definitions (the provider requires references); the
        # point is that they must be **the ones locked on the first beat**, not an appearance the
        # model redesigned this time.
        self.assertEqual(first.six_section["subject_definitions"], node.six_section["subject_definitions"])
        self.assertNotIn("redesigned", node.prompt)
        self.assertIn("frames", node.timings)

    def test_changed_character_metadata_invalidates_cached_traits(self):
        self.run_shot()
        self.room["members"][0]["description"] = "新的角色描述"
        _, text, _ = self.run_shot()
        self.assertIn(
            "appearance", text.call_args.kwargs["schema"]["properties"]["actions"]["items"]["required"]
        )
        self.assertEqual(2, len(text.call_args.kwargs["images"]))

    def test_shared_timeline_keeps_execution_and_world_event_only_once(self):
        self.room["members"][0]["action"] = "侧身移到柱后，搭弓向怪物头部射击。"
        self.room["members"][1]["action"] = "举起盾牌向怪物推进。"
        self.room["intro"] = {"world_event": "怪物抬头准备吐火。"}
        plan = copy.deepcopy(self.plan)
        plan["actions"] = plan["actions"][:2]
        plan["actions"][0].update(
            source_action=self.room["members"][0]["action"],
            action="steps behind the pillar, draws the bow, then releases an arrow toward the monster head.",
        )
        plan["actions"][1].update(
            source_action=self.room["members"][1]["action"],
            action="advances toward the monster behind a raised shield.",
        )
        plan["event_action"] = "The monster raises its head to prepare its breath."
        plan["timeline"] = (
            "As <Subject 1> sidesteps behind the pillar and draws the bow, "
            "<Subject 2> advances toward the monster behind a raised shield. "
            "The monster raises its head to prepare its breath while <Subject 1> releases an arrow; "
            "the arrow visibly travels toward its head as <Subject 2> continues advancing."
        )
        with (
            patch.object(render.llm, "chat", return_value=plan) as text,
            patch.object(render.video, "generate", return_value=self.gen) as gen,
            patch.object(render.coop_media, "grab_frames", return_value=[]),
        ):
            node = coop_loop.play_round(
                self.world,
                "x",
                room=self.room,
                progress=lambda *a: None,
                cancelled=lambda: False,
                origin="coop",
                mentions=[m["subject_id"] for m in self.room["members"]],
            )
        detail = node.six_section["detailed_description"]
        self.assertNotIn("arrow visibly travels", detail)
        self.assertIn(plan["actions"][0]["action"], detail)
        self.assertIn(plan["actions"][1]["action"], detail)
        self.assertEqual(1, detail.count("releases an arrow"))
        self.assertEqual(1, detail.count("raises its head"))
        self.assertNotIn(self.room["intro"]["world_event"], node.prompt)
        self.assertEqual(plan["actions"][0]["action"], node.directive["shot"]["actions"][0]["action"])
        self.assertIn("SERVER_COMPILED", text.call_args.kwargs["system"])
        self.assertIn(generation_prompts._CAMERA, detail)
        text.assert_called_once()
        gen.assert_called_once()

    def test_invalid_timeline_stops_before_video(self):
        """Structural errors stop at once; Chinese is first repaired by translation and stops only if that fails. Neither may
        produce a clip.

        ⚠️ The Chinese case makes **more** calls on purpose: `timeline` used to be guarded by `_english()` without being
        repairable, so Chinese failed the round outright. Now it goes through
        `_repair_plan_english_fields` like scene / sound / monster_appearance. The stub returns the
        same bad data every time, so all retries are used; in practice almost all repairs succeed
        on the first try.
        """
        # ⚠️ `'<Subject 1> moves.'` (too short / a subject missing) is **not here**; see
        # test_a_discarded_timeline_never_vetoes_a_round below. What stays here are hard errors
        # caught by `_english()`: empty values, out-of-range numbers, mixed-in Picture markers.
        structural = (
            None,
            "",
            "<Subject 1> moves with <Subject 2> toward <Subject 5>.",
            "<Subject 1> and <Subject 2> move into <Picture 1>.",
        )
        for timeline in structural + ("<Subject 1> 和 <Subject 2> 一起移动。",):
            chinese = bool(timeline) and bool(re.search(r"[\u3400-\u9fff]", timeline))
            plan = copy.deepcopy(self.plan)
            plan["actions"] = plan["actions"][:2]
            plan["timeline"] = timeline
            with (
                self.subTest(timeline=timeline),
                patch.object(render.llm, "chat", return_value=plan) as text,
                patch.object(render.video, "generate") as gen,
            ):
                with self.assertRaises(ValueError):
                    coop_loop.play_round(
                        self.world,
                        "x",
                        room=self.room,
                        progress=lambda *a: None,
                        cancelled=lambda: False,
                        origin="coop",
                        mentions=[m["subject_id"] for m in self.room["members"]],
                    )
                if chinese:
                    self.assertGreater(text.call_count, 2, "中文应该先试着修，不是直接放弃")
                else:
                    self.assertEqual(2, text.call_count, "结构错误没什么可修的，别浪费调用")
                gen.assert_not_called()
                self.assertIsNone(self.world.head)

    def test_bound_action_references_keep_actor_and_monster_without_retries(self):
        for count, shared_monster in ((2, False), (2, True), (4, True)):
            with self.subTest(players=count, monster=shared_monster):
                self.world = World(title="引用校验", setting="共同对抗怪兽")
                self.room["members"] = self.members[:count]
                if shared_monster:
                    (config.REFS / "monster.jpg").write_bytes(b"monster")
                    self.room["monster"] = {
                        "name": "巨龙",
                        "description": "岩甲巨龙",
                        "subject_id": "M1",
                        "ref": "monster.jpg",
                    }
                    self.plan["monster_appearance"] = "A large armored dragon with golden eyes."
                target = f"<Subject {count + 1}>" if shared_monster else "<Subject 2>"
                # Sentence shape from a real failure: the subject number is inside the
                # sentence and the target number must stay verbatim.
                action = (
                    f"Standing on the right platform, <Subject 1> draws the bow, "
                    f"aims at the head of {target}, releases the arrow and watches it fly toward {target}."
                )
                self.plan["actions"][0]["action"] = action
                self.plan["actions"][1]["action"] = "<Subject 2> grips the sword and takes a step forward."
                for continuing in (False, True):
                    node, text, gen = self.run_shot()
                    text.assert_called_once()
                    gen.assert_called_once()
                    self.assertEqual(action, node.directive["shot"]["actions"][0]["action"])
                    self.assertEqual(1, node.six_section["detailed_description"].count(action))
                    self.assertIn("Standing on the right platform", node.six_section["detailed_description"])
                    self.assertIn(
                        node.directive["shot"]["timeline"], node.six_section["detailed_description"]
                    )
                    # Both first beats and continuations carry numbers: the subject number is inside
                    # the sentence and the target number must stay verbatim.
                    self.assertRegex(node.prompt, r"<Subject\s+\d+>")
                    if continuing:
                        self.assertIn(action, gen.call_args.kwargs["continuation_fallback_prompt"])
                    self.assertEqual(
                        "latent-1" if continuing else None, gen.call_args.kwargs["continue_from"]
                    )

    def test_action_unknown_or_picture_reference_rejects_with_field_specific_feedback(self):
        for token in ("<Subject 0>", "<Subject 3>", "<Subject 5>", "<Picture 1>"):
            plan = copy.deepcopy(self.plan)
            plan["actions"] = plan["actions"][:2]
            plan["actions"][0]["action"] = f"<Subject 1> aims at {token}."
            with (
                self.subTest(token=token),
                patch.object(render.llm, "chat", return_value=plan) as text,
                patch.object(render.video, "generate") as gen,
            ):
                with self.assertRaisesRegex(ValueError, "action 引用无效"):
                    coop_loop.play_round(
                        self.world,
                        "x",
                        room=self.room,
                        progress=lambda *a: None,
                        cancelled=lambda: False,
                        origin="coop",
                        mentions=[m["subject_id"] for m in self.room["members"]],
                    )
                gen.assert_not_called()
                self.assertEqual(2, text.call_count)
                feedback = json.loads(text.call_args.kwargs["user"])["correction_required"]
                self.assertIn("action", feedback)
                self.assertIn("<Subject 2>", feedback)
                self.assertIsNone(self.world.head)

    def test_appearance_still_requires_plain_english(self):
        """appearance with a number is still rejected: it describes the stable look in the reference image, numbers mean nothing there."""
        plan = copy.deepcopy(self.plan)
        plan["actions"] = plan["actions"][:2]
        plan["actions"][0]["appearance"] = "<Subject 1> wears armor."
        with (
            patch.object(render.llm, "chat", return_value=plan),
            patch.object(render.video, "generate") as gen,
        ):
            with self.assertRaisesRegex(ValueError, "appearance 引用无效"):
                coop_loop.play_round(
                    self.world,
                    "x",
                    room=self.room,
                    progress=lambda *a: None,
                    cancelled=lambda: False,
                    origin="coop",
                    mentions=[m["subject_id"] for m in self.room["members"]],
                )
            gen.assert_not_called()

    def test_a_token_inside_beat_is_reworded_instead_of_failing_the_round(self):
        """Numbers in beat are replaced with natural language instead of failing the whole beat.

        The server prefixes beat with `<Subject i>` before putting it into summary, so beat itself
        should have no numbers. But the model still sometimes writes "follows closely behind
        <Subject 1>", which once failed whole beats and cost a 100 s retry.

        The meaning is clear and not worth voiding a round. Deleting the number would leave "follows
        closely behind", so it becomes a natural-language reference, worded like
        `video_prompt._plain_continuation_from_merged`.
        """
        for before, after in (
            ("follows closely behind <Subject 1>", "follows closely behind the other character"),
            (
                "stands beside <Subject 1> and holds a defensive posture",
                "stands beside the other character and holds a defensive posture",
            ),
            ("raises the bow and draws the string", "raises the bow and draws the string"),
        ):
            with self.subTest(before=before):
                self.assertEqual(after, action_text._plain_beat(before))

        plan = copy.deepcopy(self.plan)
        plan["actions"] = plan["actions"][:2]
        plan["actions"][0]["beat"] = "follows closely behind <Subject 1>"
        with (
            patch.object(render.llm, "chat", return_value=plan),
            patch.object(render.video, "generate", return_value=self.gen) as gen,
            patch.object(render.coop_media, "grab_frames", return_value=[]),
        ):
            node = coop_loop.play_round(
                self.world,
                "x",
                room=self.room,
                progress=lambda *a: None,
                cancelled=lambda: False,
                origin="coop",
                mentions=[m["subject_id"] for m in self.room["members"]],
            )
        gen.assert_called_once()
        summary = node.six_section["summary"]
        self.assertIn("follows closely behind the other character", summary)
        # The prefix number the server adds itself must stay
        self.assertIn("<Subject 1>", summary)

    def test_player_retention_preserves_individual_equipment(self):
        """Retention uses the per-shot features; numbering and sections are still bound by the server."""
        plan = copy.deepcopy(self.plan)
        plan["actions"] = plan["actions"][:2]
        plan["actions"][0]["retention"] = (
            "Her braided hair and back-mounted quiver remain visible as she lowers her bow."
        )
        plan["actions"][1]["retention"] = (
            "His short hair and leather sword belt remain visible as he raises his sword."
        )
        with (
            patch.object(render.llm, "chat", return_value=plan) as chat,
            patch.object(render.video, "generate", return_value=self.gen),
            patch.object(render.coop_media, "grab_frames", return_value=[]),
        ):
            node = coop_loop.play_round(
                self.world,
                "x",
                room=self.room,
                progress=lambda *a: None,
                cancelled=lambda: False,
                origin="coop",
                mentions=[m["subject_id"] for m in self.room["members"]],
            )
        analysis = node.six_section["retention_analysis"]
        for i, row in enumerate(plan["actions"]):
            self.assertIn(row["retention"], analysis.splitlines()[i])
            self.assertNotIn(row["appearance"].rstrip(" ."), analysis.splitlines()[i])
        self.assertNotIn("quiver", analysis.splitlines()[1])
        self.assertEqual(chat.call_count, 1)
        self.assertNotRegex(node.prompt, r"[\u3400-\u9fff]")
        # The structure stays complete: one line per player, with visibility and retention_level
        for i, _m in enumerate(self.room["members"], 1):
            self.assertIn(f"<Subject {i}> (appears in [Shot 1]", analysis)

    def test_schema_requests_visible_retention_in_existing_plan(self):
        self.assertIn("retention", generation_prompts._ACTION_FIELDS)
        self.assertIn("retention", action_text._ACTION_REPAIR_FIELDS)
        self.assertIn("retention_level", generation_prompts._ACTION_FIELDS, "level 是枚举，仍然要模型判断")

    def test_a_discarded_timeline_never_vetoes_a_round(self):
        """A malformed timeline is only logged; users must not wait a round only to see an error.

        The value is **discarded anyway**: `_validate_model_timeline`'s docstring says never use it
        as choreography, and the caller then overwrites it with `plan['timeline'] =
        compiled_timeline`. Raising for text about to be overwritten failed whole rounds ("joint
        shot timeline misses a participant").

        What matters is guarded elsewhere: each action passes `_english`, missing people are filled
        by `_recover_missing_action_entries`, numbers are normalized by `_normalize_action_ids`.
        Hard errors like out-of-range numbers and Picture markers still stop (see the previous
        case).
        """
        for timeline in (
            "<Subject 1> moves.",  # too short
            "<Subject 1> walks forward along the path to the far door.",
        ):
            with self.subTest(timeline=timeline):  # <Subject 2> missing
                plan = copy.deepcopy(self.plan)
                plan["actions"] = plan["actions"][:2]
                plan["timeline"] = timeline
                with (
                    patch.object(render.llm, "chat", return_value=plan),
                    patch.object(render.video, "generate", return_value=self.gen) as gen,
                    patch.object(render.coop_media, "grab_frames", return_value=[]),
                ):
                    node = coop_loop.play_round(
                        self.world,
                        "x",
                        room=self.room,
                        progress=lambda *a: None,
                        cancelled=lambda: False,
                        origin="coop",
                        mentions=[m["subject_id"] for m in self.room["members"]],
                    )
                gen.assert_called_once()
                body = node.six_section["detailed_description"]
                # The clip uses the compiled timeline; not a word of the model's version reaches the
                # prompt.
                self.assertNotIn(timeline.rstrip("."), body)
                for i in (1, 2):
                    self.assertIn(f"<Subject {i}>", body)

    def test_prompt_never_shows_a_token_example_that_beat_would_copy(self):
        """The prompt must not contain a "verb + <Subject N>" example: the model copies it into beat.

        Real failure: a cross-subject example `"follows behind <Subject 1>"` was added to the
        paragraph about action. beat is exactly the field where numbers are forbidden; the model
        copied the phrase into it and `_text()` rejected it, failing whole beats. The example has
        been removed.
        """
        import re as _re

        system = generation_prompts._ACTION_SYSTEM
        # The rule "no numbers in beat" must still be there
        self.assertIn("without Subject/Picture tokens", system)
        # But no "verb + <Subject N>" example may be given
        for m in _re.finditer(r'"[^"]{0,60}<Subject \d+>[^"]{0,20}"', system):
            self.fail(f"提示词里有会被 beat 抄走的编号示范：{m.group(0)}")

    def test_an_unreferenced_boss_is_told_it_has_no_subject_number(self):
        """A boss appearing for the first time has no reference image and so no number; say so explicitly.

        The `boss_introduction` beat tells the model to introduce an unreferenced boss without
        saying it is missing from subject_bindings. The model then invented a <Subject N+1>:

            <Subject 3> slowly emerges from the shadows of the gold pile...

        and `_text()` only allows <Subject 1>..<Subject N>, rejecting it and failing whole beats.
        """
        import inspect

        src = inspect.getsource(render)
        idx = [m for m in range(len(src)) if src.startswith("unreferenced BOSS", m)]
        self.assertEqual(2, len(idx), "两个调用点都要有这段说明")
        for i in idx:
            tail = src[i : i + 700]
            self.assertIn("NO Subject token", tail)
            self.assertIn("never invent a <Subject N>", tail)

    def test_the_reference_layout_is_never_described_as_a_sheet(self):
        """Never tell the video model "this reference image is a design sheet": it draws one.

        `reference_layout` used to be an enum including character sheet / left-right multi-view
        sheet / three-view panel. Every coop portrait is drawn by `coop_portrait.draw()`, and
        imagegen's subject suffix fixes "one complete character, isolated against a solid uniform
        pure white background": always a single character, known to the server.

        The model guessed, wrongly: 23 of 188 submissions (12%) answered a design sheet, and an
        image labelled `left-right multi-view sheet` was actually a single half-body portrait on
        white. The video model then received:

            <Subject 2> is the character shown in <Picture 2> as a left-right multi-view sheet

        and clips came out split screen, half white portrait, or even the whole frame was the sheet;
        2 of 20 sampled clips (10%), matching the 12%. Single-player worlds have no such enum, so
        they were fine.
        """
        self.assertNotIn("reference_layout", generation_prompts._ACTION_FIELDS, "服务端知道的东西不要问模型")
        for bad in ("sheet", "panel", "multi-view"):
            self.assertNotIn(bad, generation_prompts._REFERENCE_LAYOUT)
        # Definitions locked in old worlds are carried verbatim into new beats; clean them before
        # sending
        cleaned = action_text._unsheet_definitions(
            [
                "<Subject 1> is the character shown in <Picture 1> as a character sheet; A.",
                "<Subject 2> is the character shown in <Picture 2> as a left-right multi-view sheet; B.",
                "<Subject 3> is the character shown in <Picture 3> as a three-view panel; C.",
            ]
        )
        for line in cleaned:
            self.assertIn(generation_prompts._REFERENCE_LAYOUT, line)
            for bad in ("sheet", "panel", "multi-view"):
                self.assertNotIn(bad, line)

    def test_schema_pins_the_discarded_timeline_to_a_marker(self):
        """Once `ShotCompiler` owns the choreography, the model should not write a full timeline again.

        `_validate_model_timeline` says "never use it as choreography" and the result sets
        `result['timeline'] = ''`, but the schema stayed a bare `{'type': 'string'}` and nothing
        told the model a marker was enough, so every round it wrote a full choreography that the
        server discarded after validation.

        Measured on 25 recorded responses: timeline median ~265 tokens, 5.3 s at 50 tok/s, wasted
        every round.
        """
        self.assertEqual(
            {"type": "string", "enum": [generation_prompts._SERVER_TIMELINE]},
            generation_prompts._SCHEMA["properties"]["timeline"],
        )
        self.assertIn("timeline", generation_prompts._SCHEMA["required"], "仍然要它返回，只是内容被钉死")

    def test_marker_timeline_still_produces_a_compiled_shot(self):
        """When the model returns the marker, the final shot is still compiled by ShotCompiler from each person's action."""
        plan = copy.deepcopy(self.plan)
        plan["actions"] = plan["actions"][:2]
        plan["timeline"] = generation_prompts._SERVER_TIMELINE
        with (
            patch.object(render.llm, "chat", return_value=plan),
            patch.object(render.video, "generate", return_value=self.gen),
            patch.object(render.coop_media, "grab_frames", return_value=[]),
        ):
            node = coop_loop.play_round(
                self.world,
                "x",
                room=self.room,
                progress=lambda *a: None,
                cancelled=lambda: False,
                origin="coop",
                mentions=[m["subject_id"] for m in self.room["members"]],
            )
        body = node.six_section["detailed_description"]
        self.assertNotIn(generation_prompts._SERVER_TIMELINE, body, "标记本身不许漏进 prompt")
        for i in (1, 2):
            self.assertIn(f"<Subject {i}>", body, "每个玩家都要出现在编排里")
        self.assertIn("performs their own action", body)

    def test_action_translation_without_bindings_remains_plain_text(self):
        plan = copy.deepcopy(self.plan)
        plan["actions"] = plan["actions"][:2]
        with patch.object(render.llm, "chat", return_value=plan):
            actions = render._locked_actions(self.room["members"], self.room["setting"])
        self.assertEqual(plan["actions"][0]["action"], actions["u0"]["action"])
        plan["actions"][0]["action"] = "<Subject 1> draws the bow."
        with patch.object(render.llm, "chat", return_value=plan):
            with self.assertRaisesRegex(ValueError, "action 引用无效"):
                render._locked_actions(self.room["members"], self.room["setting"])

    def test_timeline_structure_error_is_returned_to_gm_for_one_repair(self):
        good = copy.deepcopy(self.plan)
        good["actions"] = good["actions"][:2]
        bad = copy.deepcopy(good)
        bad.pop("timeline")
        with (
            patch.object(render.llm, "chat", side_effect=[bad, good]) as text,
            patch.object(render.video, "generate", return_value=self.gen) as gen,
            patch.object(render.coop_media, "grab_frames", return_value=[]),
        ):
            coop_loop.play_round(
                self.world,
                "x",
                room=self.room,
                progress=lambda *a: None,
                cancelled=lambda: False,
                origin="coop",
                mentions=[m["subject_id"] for m in self.room["members"]],
            )
        self.assertIn("timeline", json.loads(text.call_args.kwargs["user"])["correction_required"])
        self.assertEqual(2, text.call_count)
        gen.assert_called_once()

    def test_legacy_prompt_keeps_latent_and_gains_new_retention_and_timeline(self):
        for action in self.plan["actions"]:
            action.pop("retention", None)
        self.run_shot()
        self.world.nodes[self.world.head]["directive"]["prompt_version"] = 5
        node, text, gen = self.run_shot()
        self.assertEqual("latent-1", gen.call_args.kwargs["continue_from"])
        self.assertTrue(gen.call_args.kwargs["require_continuation"])
        self.assertTrue(gen.call_args.kwargs["send_refs_on_continuation"])
        self.assertIn("Character 0", node.six_section["retention_analysis"].splitlines()[0])
        self.assertIn("Character 1", node.six_section["retention_analysis"].splitlines()[1])
        self.assertNotIn(
            "white or studio reference backdrop", gen.call_args.kwargs["continuation_fallback_prompt"]
        )
        self.assertIn("timeline", node.directive["shot"])
        self.assertNotIn(
            "appearance", text.call_args.kwargs["schema"]["properties"]["actions"]["items"]["required"]
        )

    def test_model_failure_does_not_commit_or_start_video(self):
        with (
            patch.object(render.llm, "chat", side_effect=RuntimeError("offline")),
            patch.object(render.video, "generate") as gen,
        ):
            with self.assertRaises(RuntimeError):
                coop_loop.play_round(
                    self.world,
                    "x",
                    room=self.room,
                    progress=lambda *a: None,
                    cancelled=lambda: False,
                    origin="coop",
                    mentions=[m["subject_id"] for m in self.room["members"]],
                )
            gen.assert_not_called()
            self.assertIsNone(self.world.head)

    def test_empty_structured_model_response_retries_before_video(self):
        plan = copy.deepcopy(self.plan)
        plan["actions"] = plan["actions"][:2]
        with (
            patch.object(
                render.llm, "chat", side_effect=[render.llm.GeminiError("empty response"), plan]
            ) as text,
            patch.object(render.video, "generate", return_value=self.gen),
            patch.object(render.coop_media, "grab_frames", return_value=[]),
        ):
            node = coop_loop.play_round(
                self.world,
                "x",
                room=self.room,
                progress=lambda *a: None,
                cancelled=lambda: False,
                origin="coop",
                mentions=[m["subject_id"] for m in self.room["members"]],
            )
        self.assertEqual(2, text.call_count)
        self.assertEqual("joint.mp4", node.clip)

    def test_chinese_action_is_repaired_before_video(self):
        bad = copy.deepcopy(self.plan)
        bad["actions"] = bad["actions"][:2]
        bad["actions"][0]["action"] = "洛伊沿左侧岩壁向洞穴深处潜行，警惕地观察地形。"
        repaired = {
            "players": [
                {
                    "user_id": self.members[0]["user_id"],
                    "action": "Loy sneaks deeper along the left rock wall, cautiously observing the terrain.",
                }
            ]
        }
        with (
            patch.object(render.llm, "chat", side_effect=[bad, repaired]) as text,
            patch.object(render.video, "generate", return_value=self.gen) as gen,
            patch.object(render.coop_media, "grab_frames", return_value=[]),
        ):
            node = coop_loop.play_round(
                self.world,
                "x",
                room=self.room,
                progress=lambda *a: None,
                cancelled=lambda: False,
                origin="coop",
                mentions=[m["subject_id"] for m in self.room["members"]],
            )
        self.assertEqual(2, text.call_count)
        self.assertEqual("coop_action_translation_repair", text.call_args.kwargs["tag"])
        self.assertIn("Loy sneaks deeper", node.directive["shot"]["actions"][0]["action"])
        gen.assert_called_once()

    def test_chinese_action_and_beat_are_repaired_together(self):
        bad = copy.deepcopy(self.plan)
        bad["actions"] = bad["actions"][:2]
        bad["actions"][0]["action"] = "洛伊沿左侧岩壁向洞穴深处潜行，警惕地观察地形。"
        bad["actions"][0]["beat"] = "洛伊沿左侧岩壁潜行"
        repaired = {
            "players": [
                {
                    "user_id": self.members[0]["user_id"],
                    "action": "The character sneaks deeper along the left rock wall, cautiously observing the terrain.",
                    "beat": "sneaks deeper along the left rock wall",
                }
            ]
        }
        with (
            patch.object(render.llm, "chat", side_effect=[bad, repaired]) as text,
            patch.object(render.video, "generate", return_value=self.gen) as gen,
            patch.object(render.coop_media, "grab_frames", return_value=[]),
        ):
            node = coop_loop.play_round(
                self.world,
                "x",
                room=self.room,
                progress=lambda *a: None,
                cancelled=lambda: False,
                origin="coop",
                mentions=[m["subject_id"] for m in self.room["members"]],
            )
        self.assertEqual(2, text.call_count)
        self.assertIn("sneaks deeper along the left rock wall", node.directive["shot"]["actions"][0]["beat"])
        gen.assert_called_once()

    def test_chinese_start_state_is_repaired_and_remains_in_start_phase(self):
        plan = copy.deepcopy(self.plan)
        plan["actions"] = plan["actions"][:2]
        plan["actions"][0]["start_state"] = "站在石柱旁"
        plan["actions"][0]["end_state"] = "at the gateway"
        repaired = {
            "players": [{"user_id": self.members[0]["user_id"], "start_state": "standing beside the pillar"}]
        }
        with (
            patch.object(render.llm, "chat", side_effect=[plan, repaired]) as text,
            patch.object(render.video, "generate", return_value=self.gen),
            patch.object(render.coop_media, "grab_frames", return_value=[]),
        ):
            node = coop_loop.play_round(
                self.world,
                "x",
                room=self.room,
                progress=lambda *a: None,
                cancelled=lambda: False,
                origin="coop",
                mentions=[m["subject_id"] for m in self.room["members"]],
            )
        self.assertEqual(2, text.call_count)
        body = node.six_section["detailed_description"]
        stored = node.directive["shot"]["actions"][0]
        self.assertEqual("standing beside the pillar", stored["start_state"])
        self.assertEqual("at the gateway", stored["end_state"])
        self.assertNotIn("starting state is", body)
        self.assertNotIn("final state is", body)
        self.assertNotIn("By the end,", body)

    def test_chinese_repair_retries_when_model_copies_name_again(self):
        bad = copy.deepcopy(self.plan)
        bad["actions"] = bad["actions"][:2]
        bad["actions"][0]["action"] = "洛伊 sneaks deeper along the left rock wall."
        rejected_repair = {
            "players": [
                {
                    "user_id": self.members[0]["user_id"],
                    "action": "洛伊 continues sneaking deeper along the left rock wall.",
                }
            ]
        }
        repaired = {
            "players": [
                {
                    "user_id": self.members[0]["user_id"],
                    "action": "The character continues sneaking deeper along the left rock wall.",
                }
            ]
        }
        with (
            patch.object(render.llm, "chat", side_effect=[bad, rejected_repair, repaired]) as text,
            patch.object(render.video, "generate", return_value=self.gen) as gen,
            patch.object(render.coop_media, "grab_frames", return_value=[]),
        ):
            node = coop_loop.play_round(
                self.world,
                "x",
                room=self.room,
                progress=lambda *a: None,
                cancelled=lambda: False,
                origin="coop",
                mentions=[m["subject_id"] for m in self.room["members"]],
            )
        self.assertEqual(3, text.call_count)
        self.assertEqual(
            "The character continues sneaking deeper along the left rock wall.",
            node.directive["shot"]["actions"][0]["action"],
        )
        gen.assert_called_once()

    def test_two_contaminated_players_cost_one_repair_call_not_two(self):
        """The repair used to run in `for m in members:`, one call per person, on the critical path.

        Measured, the repair ran 74 times against 28 translations, about 2.6 per round. Models
        writing Chinese here is normal, so this path runs almost every round; a four-player room
        meant 4 sequential ~1 s calls.
        """
        bad = copy.deepcopy(self.plan)
        bad["actions"] = bad["actions"][:2]
        bad["actions"][0]["action"] = "洛伊沿左侧岩壁潜行。"
        bad["actions"][1]["beat"] = "浅月举剑格挡"
        repaired = {
            "players": [
                {
                    "user_id": self.members[0]["user_id"],
                    "action": "The character sneaks along the left rock wall past the stone monster.",
                    "beat": "sneaks along the left rock wall",
                },
                {
                    "user_id": self.members[1]["user_id"],
                    "action": "The character raises a sword and guards the right flank.",
                    "beat": "raises a sword and guards the flank",
                },
            ]
        }
        with (
            patch.object(render.llm, "chat", side_effect=[bad, repaired]) as text,
            patch.object(render.video, "generate", return_value=self.gen) as gen,
            patch.object(render.coop_media, "grab_frames", return_value=[]),
        ):
            node = coop_loop.play_round(
                self.world,
                "x",
                room=self.room,
                progress=lambda *a: None,
                cancelled=lambda: False,
                origin="coop",
                mentions=[m["subject_id"] for m in self.room["members"]],
            )
        self.assertEqual(2, text.call_count, "两个人污染也只能有一次修复调用")
        shot = node.directive["shot"]["actions"]
        self.assertIn("sneaks along the left rock wall", shot[0]["action"])
        self.assertIn("guards the flank", shot[1]["beat"])
        gen.assert_called_once()

    def test_only_the_contaminated_fields_are_written_back(self):
        """Uncontaminated fields must not be rewritten along the way: the repair fills back only the fields that were dirty."""
        clean_appearance = self.plan["actions"][0]["appearance"]
        bad = copy.deepcopy(self.plan)
        bad["actions"] = bad["actions"][:2]
        bad["actions"][0]["action"] = "洛伊沿左侧岩壁潜行。"
        repaired = {
            "players": [
                {
                    "user_id": self.members[0]["user_id"],
                    "action": "The character sneaks along the left rock wall past the stone monster.",
                    "appearance": "SOMETHING THE MODEL INVENTED",
                }
            ]
        }
        with (
            patch.object(render.llm, "chat", side_effect=[bad, repaired]),
            patch.object(render.video, "generate", return_value=self.gen),
            patch.object(render.coop_media, "grab_frames", return_value=[]),
        ):
            node = coop_loop.play_round(
                self.world,
                "x",
                room=self.room,
                progress=lambda *a: None,
                cancelled=lambda: False,
                origin="coop",
                mentions=[m["subject_id"] for m in self.room["members"]],
            )
        shot = node.directive["shot"]["actions"]
        self.assertEqual(clean_appearance, shot[0]["appearance"])
        self.assertNotIn("INVENTED", json.dumps(node.directive, ensure_ascii=False))

    def test_missing_player_entry_is_recovered_before_video(self):
        bad = copy.deepcopy(self.plan)
        bad["actions"] = bad["actions"][:1]
        recovered = copy.deepcopy(self.plan["actions"][1])
        with (
            patch.object(render.llm, "chat", side_effect=[bad, {"actions": [recovered]}]) as text,
            patch.object(render.video, "generate", return_value=self.gen) as gen,
            patch.object(render.coop_media, "grab_frames", return_value=[]),
        ):
            node = coop_loop.play_round(
                self.world,
                "x",
                room=self.room,
                progress=lambda *a: None,
                cancelled=lambda: False,
                origin="coop",
                mentions=[m["subject_id"] for m in self.room["members"]],
            )
        self.assertEqual(2, text.call_count)
        self.assertEqual("coop_action_entry_recovery", text.call_args.kwargs["tag"])
        self.assertEqual(["u0", "u1"], [a["user_id"] for a in node.directive["shot"]["actions"]])
        gen.assert_called_once()

    def test_unique_source_action_recovers_truncated_user_id(self):
        plan = copy.deepcopy(self.plan)
        plan["actions"] = plan["actions"][:2]
        plan["actions"][0]["user_id"] = "u0-truncated"
        with (
            patch.object(render.llm, "chat", return_value=plan),
            patch.object(render.video, "generate", return_value=self.gen),
            patch.object(render.coop_media, "grab_frames", return_value=[]),
        ):
            node = coop_loop.play_round(
                self.world,
                "x",
                room=self.room,
                progress=lambda *a: None,
                cancelled=lambda: False,
                origin="coop",
                mentions=[m["subject_id"] for m in self.room["members"]],
            )
        self.assertEqual(["u0", "u1"], [a["user_id"] for a in node.directive["shot"]["actions"]])


if __name__ == "__main__":
    unittest.main()
