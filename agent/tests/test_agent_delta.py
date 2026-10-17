"""World reuse must not erase changes or accept incomplete streamed decisions."""

from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from murea_agent import config
from murea_agent.guards import density_audit
from murea_agent.director import director  # noqa: E402
from murea_agent.director import agent_delta, agent_schema
from murea_agent.shot import beat_assets
from murea_agent.guards import motion
from murea_agent import loop
from murea_agent.memory.world import Node, World


class WorldDeltaTest(unittest.TestCase):
    def setUp(self):
        self.world = World(
            title="test",
            here="酒馆",
            mainline="找到失踪的旅人",
            mainline_phase="查找线索",
            mainline_milestones=["查找线索", "找到旅人"],
        )
        self.world.add_node(
            Node(
                id="n1",
                parent=None,
                video_id="",
                clip="",
                place="酒馆",
                beat="举起灯笼",
                mode="continue",
                carry="The traveler holds one lamp.",
            )
        )
        self.inherited = agent_delta.defaults(self.world)
        self.canonical = agent_schema._agent_schema()

    def test_unchanged_world_restores_existing_carry_and_milestones(self):
        out = agent_delta.expand({"world_delta": {}, "reply": "你看向门外。"}, self.inherited)
        self.assertEqual(self.world.head_node.carry, out["carry"])
        self.assertEqual(self.world.here, out["place"])
        self.assertEqual(self.world.mainline_milestones, out["mainline_milestones"])
        self.assertEqual({"subjects": [], "places": []}, out["state_updates"])
        self.assertEqual(set(agent_delta.FIELDS) | {"reply"}, set(out))
        out["mainline_milestones"].append("新内容")
        self.assertEqual(2, len(self.world.mainline_milestones))

    def test_top_level_field_nested_in_the_delta_is_moved_out(self):
        """DeepSeek wrote `mainline` next to mainline_phase; that must not fail the beat."""
        out = agent_delta.expand(
            {"world_delta": {"mainline": "Find the wrecker", "mainline_phase": "Arrival"}},
            self.inherited,
        )
        self.assertEqual("Find the wrecker", out["mainline"])
        self.assertEqual("Arrival", out["mainline_phase"])

    def test_explicit_release_is_not_overwritten_with_old_carry(self):
        out = agent_delta.expand({"world_delta": {"carry": ""}}, self.inherited)
        self.assertEqual("", out["carry"])

    def test_new_assets_player_and_location_transition_are_complete(self):
        delta = {
            "mode": "cut",
            "place": "街道",
            "transition": True,
            "place_image_prompt": "a snowy street",
            "place_brief": "雪夜街道",
            "main_scene": "街道",
            "new_subjects": [
                {
                    "name": "船夫",
                    "kind": "人物",
                    "voice": "v1",
                    "image_prompt": "a tall boatman",
                    "brief": "等待的人",
                }
            ],
            "implicit_player": {
                "appears": True,
                "name": "旅人",
                "voice": "v2",
                "image_prompt": "a blue-coated traveler",
                "brief": "旅人",
            },
            "new_places": [
                {"name": "码头", "image_prompt": "wooden dock", "brief": "码头", "main_scene": "码头"}
            ],
        }
        out = agent_delta.expand(
            {
                "world_delta": {key: value for key, value in delta.items() if key in agent_delta.FIELDS},
                **{key: value for key, value in delta.items() if key not in agent_delta.FIELDS},
            },
            self.inherited,
        )
        for key, value in delta.items():
            self.assertEqual(value, out[key])
        beat_assets._inject_implicit_player(self.world, out)
        self.assertIn("旅人", out["subjects"])
        self.assertEqual("v2", out["new_subjects"][-1]["voice"])

    def test_injuries_weather_knowledge_and_quest_changes_are_not_lost(self):
        state = {
            "subjects": [{"name": "旅人", "condition": "右手受伤", "knowledge_add": ["船夫在码头"]}],
            "places": [{"name": "酒馆", "weather": "暴雪"}],
        }
        quests = {"update": [{"id": "q1", "status": "completed", "progress": "找到线索", "place": "酒馆"}]}
        out = agent_delta.expand(
            {"world_delta": {"state_updates": state, "quest_updates": quests}}, self.inherited
        )
        self.assertEqual(state, out["state_updates"])
        self.assertEqual({**quests, "create": []}, out["quest_updates"])

    def test_missing_streamed_delta_does_not_mean_no_changes(self):
        early = {
            k: "value"
            for k in self.canonical["required"]
            if k not in agent_delta.FIELDS and k not in agent_schema.AGENT_LATE_FIELDS
        }
        self.assertFalse(
            agent_schema._agent_partial_ready(agent_delta.expand(early, self.inherited), self.canonical)
        )
        full = agent_delta.expand({**early, "world_delta": {}}, self.inherited)
        self.assertTrue(agent_schema._agent_partial_ready(full, self.canonical))

    def test_invalid_delta_or_conflicting_copies_fail_closed(self):
        for raw in (
            {"world_delta": None},
            {"world_delta": {"nonsense": "x"}},
            {"world_delta": {}, "carry": "competing copy"},
            {"world_delta": {"state_updates": []}},
        ):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                agent_delta.expand(raw, self.inherited)

    def test_actions_audio_state_and_independent_audits_are_still_required(self):
        before = copy.deepcopy(self.canonical)
        wire = agent_delta.schema(self.canonical, self.inherited)
        for key in (
            "beat",
            "motion_plan",
            "action_sequence",
            "dialogue",
            "subjects",
            "new_subjects",
            "new_places",
            "implicit_player",
            "detailed_description",
            *agent_schema._MOTION_AUDIT,
        ):
            self.assertEqual(before["properties"][key], wire["properties"][key])
            self.assertIn(key, wire["required"])
        # world_delta may now be anyOf (scene change / no change); both branches must carry it.
        node = wire["properties"]["world_delta"]
        for variant in node.get("anyOf", [node]):
            self.assertIn("state_updates", variant["properties"])
        self.assertEqual(before, self.canonical)

    def test_first_world_must_still_establish_milestones(self):
        inherited = agent_delta.defaults(World(title="新世界"))
        wire = agent_delta.schema(self.canonical, inherited)
        self.assertTrue(
            {"mainline_phase", "mainline_milestones", "place", "place_image_prompt"}.issubset(
                wire["properties"]["world_delta"]["required"]
            )
        )

    def test_changed_location_cannot_default_away_transition_decision(self):
        with self.assertRaises(ValueError):
            agent_delta.expand({"world_delta": {"place": "码头"}}, self.inherited)

    def test_stream_and_final_are_expanded_before_existing_consumers(self):
        wire = {k: "value" for k in self.canonical["required"] if k not in agent_delta.FIELDS}
        wire.update(world_delta={"carry": ""}, reply="你放下灯笼。")
        wire.update(new_subjects=[], new_places=[], implicit_player={"appears": False}, subjects=[])
        received = []

        def chat(**kw):
            self.assertIn("world_delta", kw["schema"]["required"])
            kw["on_partial"](wire)
            return wire

        with patch.object(config, "LLM_BACKEND", "openai"), patch.object(loop.llm, "chat", side_effect=chat):
            out = director._agent_step(self.world, "放下灯笼", received.append)
        self.assertEqual(out, received[0])
        self.assertEqual("", out["carry"])
        self.assertNotIn("world_delta", out)

    def test_gemini_compact_stream_restores_identical_contract(self):
        wire = {k: "value" for k in self.canonical["required"] if k not in agent_delta.FIELDS}
        wire.update(world_delta={"carry": ""}, reply="你放下灯笼。")
        wire.update(new_subjects=[], new_places=[], implicit_player={"appears": False}, subjects=[])
        received = []

        def chat(**kw):
            self.assertIn("world_delta", kw["schema"]["required"])
            kw["on_partial"](wire)
            return wire

        with patch.object(config, "LLM_BACKEND", "gemini"), patch.object(loop.llm, "chat", side_effect=chat):
            out = director._agent_step(self.world, "放下灯笼", received.append)
        self.assertEqual(out, received[0])
        self.assertEqual("", out["carry"])
        self.assertNotIn("world_delta", out)

    def test_truncated_final_delta_is_rejected(self):
        with (
            patch.object(config, "LLM_BACKEND", "openai"),
            patch.object(loop.llm, "chat", return_value={"reply": "未写完"}),
            self.assertRaises(ValueError),
        ):
            director._agent_step(self.world, "放下灯笼")

    def test_unknown_subject_without_identity_declaration_is_rejected(self):
        with self.assertRaises(ValueError):
            agent_delta.validate_assets(
                {"subjects": ["S999"], "new_subjects": [], "implicit_player": {"appears": False}}, self.world
            )
        with self.assertRaises(ValueError):
            agent_delta.validate_assets(
                {"subjects": [], "action_sequence": {"steps": [{"visible_subjects": ["未建档的红衣女人"]}]}},
                self.world,
            )

    def test_the_place_this_beat_declares_counts_as_registered(self):
        agent_delta.validate_assets(
            {"place": "Great Hall", "action_sequence": {"steps": [{"visible_subjects": ["Great Hall"]}]}},
            self.world,
        )

    def test_undeclared_cast_gets_one_replan_with_the_names(self):
        wire = {k: "value" for k in self.canonical["required"] if k not in agent_delta.FIELDS}
        wire.update(world_delta={"carry": ""}, reply="你放下灯笼。")
        wire.update(new_subjects=[], new_places=[], implicit_player={"appears": False})
        prompts = []

        def chat(**kw):
            prompts.append(kw["user"])
            return {**wire, "subjects": ["Elder Marta"] if len(prompts) == 1 else []}

        with patch.object(config, "LLM_BACKEND", "openai"), patch.object(loop.llm, "chat", side_effect=chat):
            out = director._agent_step(self.world, "放下灯笼")
        self.assertEqual(2, len(prompts))
        self.assertIn("Elder Marta", prompts[1])
        self.assertEqual([], out["subjects"])

    def test_the_same_value_written_twice_is_accepted(self):
        out = agent_delta.expand({"world_delta": {"carry": "a lamp"}, "carry": "a lamp"}, self.inherited)
        self.assertEqual("a lamp", out["carry"])
        with self.assertRaises(ValueError):
            agent_delta.expand({"world_delta": {"carry": "a lamp"}, "carry": "a rope"}, self.inherited)

    def test_a_malformed_answer_gets_one_replan(self):
        wire = {k: "value" for k in self.canonical["required"] if k not in agent_delta.FIELDS}
        wire.update(world_delta={"carry": ""}, reply="你放下灯笼。")
        wire.update(new_subjects=[], new_places=[], implicit_player={"appears": False}, subjects=[])
        prompts = []

        def chat(**kw):
            prompts.append(kw["user"])
            return {"reply": "未写完"} if len(prompts) == 1 else wire

        with patch.object(config, "LLM_BACKEND", "openai"), patch.object(loop.llm, "chat", side_effect=chat):
            out = director._agent_step(self.world, "放下灯笼")
        self.assertEqual(2, len(prompts))
        self.assertIn("did not finish world_delta", prompts[1])
        self.assertEqual("", out["carry"])

    def test_complete_delta_with_missing_video_fields_is_not_success(self):
        with (
            patch.object(config, "LLM_BACKEND", "openai"),
            patch.object(loop.llm, "chat", return_value={"reply": "未写完", "world_delta": {}}),
            self.assertRaises(ValueError),
        ):
            director._agent_step(self.world, "放下灯笼")

    def test_compiled_delta_and_full_decision_submit_identical_conditioning(self):
        class Submitted(BaseException):
            pass

        with tempfile.TemporaryDirectory() as tmp:
            world = World(title="test", here="酒馆")
            sid = world.add_subject("旅人", "traveler.png")
            world.add_place("酒馆", "tavern.png")
            world.subjects[sid]["definition"] = "the traveler in a blue coat shown in <Picture 1>."
            world.places["酒馆"]["definition"] = "the tavern with wooden walls shown in <Picture 2>."
            change = {
                "carry": "",
                "state_updates": {
                    "subjects": [
                        {
                            "name": "旅人",
                            "outfit": "a wet blue coat",
                            "condition": "a bandaged right hand",
                            "knowledge_add": ["码头有人等候"],
                        }
                    ],
                    "places": [{"name": "酒馆", "weather": "heavy rain"}],
                },
            }
            shot = {
                "reply": "你站在酒馆里看向窗外。",
                "beat": "旅人看向窗外。",
                "subjects": [sid],
                "dialogue": "",
                "action_sequence": None,
                "motion_plan": {},
                "summary": "[reference_generation] <Subject 1> looks out of the window inside <Subject 2>.",
                "detailed_description": "[Shot 1] <Subject 1> wears a wet blue coat, with a bandaged right hand, standing by the window inside <Subject 2> and watching heavy rain.",
                "overall_soundscape": "Heavy rain hits the window.",
                "non_diegetic_music": "None.",
            }
            inherited = agent_delta.defaults(world)
            canonical = {**shot, **inherited, **change}
            expanded = agent_delta.expand({**shot, "world_delta": change}, inherited)
            captured = []
            for decision in (canonical, expanded):
                current = copy.deepcopy(world)

                def capture(prompt, refs, **kw):
                    captured.append((prompt, refs, current.dynamic_state_snapshot()))
                    raise Submitted()

                with (
                    patch.object(config, "REFS", Path(tmp)),
                    patch.object(director, "_decide_early", return_value=(copy.deepcopy(decision), None)),
                    patch.object(loop, "_rewrite", side_effect=AssertionError("No second model")),
                    patch.object(loop.video, "generate", side_effect=capture),
                    patch.object(World, "save", side_effect=AssertionError("No world writes")),
                ):
                    with self.assertRaises(Submitted):
                        loop.turn(current, "观察窗外")
            self.assertEqual(captured[0], captured[1])
            for fact in ("wet blue coat", "bandaged right hand", "heavy rain", "<Picture 1>", "<Picture 2>"):
                self.assertIn(fact, captured[1][0])

    def test_gemini_json_schema_preserves_conditional_physics_and_scene_rules(self):
        from murea_agent.providers import llm

        canonical = density_audit._motion_replan_schema(self.canonical)
        wire = agent_delta.schema(canonical, self.inherited)
        with patch.object(llm, "_cache_for", return_value=None):
            body = llm._body("rules", [], "action", "gemini-test", wire, 0.9, None)
        sent = body["generationConfig"]["responseJsonSchema"]
        inactive, active = sent["properties"]["motion_plan"]["anyOf"]
        self.assertEqual({"const": "none"}, inactive["properties"]["interaction_type"])
        self.assertNotIn("actors", inactive["properties"])
        for key in ("actors", "target", "path", "end_state", "visual_evidence"):
            self.assertIn(key, active["required"])
        self.assertEqual(
            self.inherited["carry"], agent_delta.expand({"world_delta": {}}, self.inherited)["carry"]
        )
        self.assertTrue(llm._log_body(body)["schema"])

    def test_sparse_inactive_physics_normalizes_to_same_full_plan(self):
        short = {"interaction_type": "none", "phase": "none", "relation_change": "none", "framing": "none"}
        self.assertEqual(
            motion._normalize_motion_plan(short),
            motion._normalize_motion_plan(dict(motion._EMPTY_MOTION_PLAN)),
        )

    def test_gemini_compact_rejects_missing_state_or_unknown_assets(self):
        for payload in ({"reply": "unfinished"}, {"world_delta": {}, "subjects": ["unknown"]}):
            with (
                self.subTest(payload=payload),
                patch.object(config, "LLM_BACKEND", "gemini"),
                patch.object(loop.llm, "chat", return_value=payload),
                self.assertRaises(ValueError),
            ):
                director._agent_step(self.world, "询问")


class SceneFieldsAreAllOrNothingTest(unittest.TestCase):
    """The six scene-change fields are all absent or all present, otherwise the beat fails.

    `expand()` always rejected "place without mode / transition", but only in a later check, not in
    the schema: `mode` / `transition` were required only for a brand-new world, so once the world
    had a place the model could write only place and the whole beat raised "a scene change must
    decide mode and transition".

    In a scan of 12 beats, 2 failed this way, one of them for "reach out and push open the door next
    to me", not even a transition.
    """

    def _delta_schema(self, here):
        world = World(title="t")
        world.here = here
        inherited = agent_delta.defaults(world)
        schema = agent_delta.schema(agent_schema._agent_schema(False), inherited)
        return schema["properties"]["world_delta"], inherited

    def test_established_world_offers_a_branch_without_any_scene_field(self):
        node, _ = self._delta_schema("控制中心")
        self.assertIn("anyOf", node)
        quiet, moving = node["anyOf"]
        for key in ("place", "mode", "transition", "place_image_prompt", "place_brief", "main_scene"):
            with self.subTest(key=key):
                self.assertNotIn(key, quiet["properties"], "不换场景那支根本不该有这些键")
                self.assertIn(key, moving["required"], "换场景那支必须六个一起要")
        self.assertFalse(quiet.get("additionalProperties", True), "关掉才真的写不出半套场景字段")

    def test_a_brand_new_world_must_still_establish_its_place(self):
        node, _ = self._delta_schema("")
        self.assertNotIn("anyOf", node, "全新世界没有「不换场景」这个选项")
        for key in ("place", "mode", "transition"):
            self.assertIn(key, node["required"])

    def test_expand_still_rejects_half_a_scene_change(self):
        """Old caches and non-OpenAI backends are not protected by the schema; the later check must stay."""
        _, inherited = self._delta_schema("控制中心")
        with self.assertRaisesRegex(ValueError, "mode and transition"):
            agent_delta.expand({"world_delta": {"place": "新地方"}}, inherited)
        out = agent_delta.expand(
            {"world_delta": {"place": "新地方", "mode": "cut", "transition": True}}, inherited
        )
        self.assertEqual("新地方", out["place"])


if __name__ == "__main__":
    unittest.main()
