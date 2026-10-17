"""Fast regression test for automatic character creation (no external APIs)."""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from murea_agent import config
from murea_agent.guards import action_sequence
from murea_agent.shot import definitions
from murea_agent.director import director  # noqa: E402
from murea_agent.director import prefetch as prefetch_mod  # noqa: E402
from app.api.routes import assets, worlds
from murea_agent.shot import beat_assets
from murea_agent.guards import motion
from murea_agent import loop
from app.users import DEFAULT_USER_ID
from murea_agent.memory.world import Node, World

STYLE = (
    "2D anime storybook illustration, clean line art, bright pastel candy palette, "
    "soft warm lighting, hand-painted backgrounds"
)


class _JsonRequest:
    def __init__(self, body: dict[str, str]):
        self.body = body

    async def json(self) -> dict[str, str]:
        return self.body


def _six() -> dict[str, str]:
    return {
        "subject_definitions": (
            "<Subject 1> is the anime-style teenage boy shown in <Picture 1> as a "
            "single full-body view; tousled dark hair and a blue hoodie.\n"
            "<Subject 2> is the candy house shown in <Picture 2> as a single wide "
            "scene view; gingerbread walls and colorful sweets."
        ),
        "summary": "[reference_generation] <Subject 1> enters <Subject 2>.",
        "retention_analysis": (
            "<Subject 1> (appears in [Shot 1] (full body)): fully_preserved - same boy\n"
            "<Subject 2> (appears in [Shot 1] (background scene)): fully_preserved - same house"
        ),
        "detailed_description": "[Shot 1] <Subject 1> enters <Subject 2>.",
        "overall_soundscape": "N/A",
        "non_diegetic_music": "N/A",
    }


class NewSubjectTest(unittest.TestCase):
    def test_completed_approach_turn_enters_destination_with_staged_priest(self) -> None:
        gm = {
            "reply": "吉尔伽美什来到神官前，稳稳停下。",
            "mode": "cut",
            "transition": True,
            "place": "乌鲁克内殿",
            "place_image_prompt": "",
            "place_brief": "",
            "main_scene": "乌鲁克内殿",
            "beat": "接近神官",
            "action_sequence": {
                "goal": "进入内殿与等候的神官交谈",
                "steps": [
                    {
                        "instruction": "移动到神官前",
                        "kind": "motion",
                        "phase": "approach",
                        "relation_change": "distance",
                        "place": "乌鲁克神庙之巅",
                    },
                    {
                        "instruction": "与神官交谈",
                        "kind": "dialogue",
                        "phase": "none",
                        "relation_change": "none",
                        "place": "乌鲁克内殿",
                        "end_state": "双方自然交谈",
                        "continuity_anchor": "相向而立",
                    },
                    {
                        "instruction": "进入内殿",
                        "kind": "motion",
                        "phase": "transform",
                        "relation_change": "containment",
                        "place": "乌鲁克内殿",
                        "end_state": "人在内殿",
                        "continuity_anchor": "入口敞开",
                    },
                ],
            },
            "motion_plan": {
                "interaction_type": "locomotion",
                "phase": "approach",
                "relation_change": "distance",
                "actors": ["吉尔伽美什", "神官"],
                "target": "神官",
                "start_state": "at the entrance",
                "path": "moves toward the priest",
                "contact_points": "",
                "end_state": "stops outside",
                "camera_side": "terrace",
                "framing": "wide",
                "visual_evidence": ["actor_full_body", "target_full_structure", "support_surface"],
                "must_show": [],
                "invariants": ["identity"],
            },
            "dialogue": "",
            "carry": "",
            "options": [],
            "recommend": 0,
            "subjects": ["S1", "神官"],
            "new_subjects": [
                {
                    "name": "神官",
                    "kind": "人物",
                    "voice": "",
                    "image_prompt": "An ancient priest",
                    "brief": "在内殿等候的神官",
                }
            ],
            "implicit_player": {"appears": False, "name": "", "voice": "", "image_prompt": "", "brief": ""},
            "state_updates": {"subjects": [], "places": []},
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            refs = root / "refs"
            sessions = root / "sessions"
            refs.mkdir()
            sessions.mkdir()
            world = World(id="wstage", title="未命名世界", here="乌鲁克神庙之巅")
            world.add_place("乌鲁克神庙之巅", "top.png")
            world.add_place("乌鲁克内殿", "hall.png")
            world.add_subject("吉尔伽美什", "hero.png")
            previous = Node(
                id="nprev",
                parent=None,
                video_id="v1",
                clip="v1.mp4",
                place="乌鲁克神庙之巅",
                beat="走到入口",
                mode="continue",
                directive={"motion_plan": gm["motion_plan"]},
            )
            world.add_node(previous)

            def fake_video(*_args, **_kwargs):
                return SimpleNamespace(
                    video_id="v2",
                    video_path=root / "clip.mp4",
                    params={},
                    continued=False,
                    artifact_id="a2",
                )

            with (
                patch.object(config, "REFS", refs),
                patch.object(config, "SESSIONS", sessions),
                patch.object(
                    director,
                    "_decide_early",
                    return_value=(
                        {
                            **gm,
                            **_six(),
                            "motion_phase": "transform",
                            "motion_relation_change": "containment",
                            "motion_target": "乌鲁克内殿",
                            "motion_framing": "wide",
                        },
                        None,
                    ),
                ),
                patch.object(loop, "_rewrite", return_value=_six()),
                patch.object(loop.imagegen, "generate", return_value=b"priest-png"),
                patch.object(loop.video, "generate", side_effect=fake_video),
            ):
                node = loop.turn(world, "进入内殿与等候的神官交谈")

        self.assertEqual("乌鲁克内殿", node.place)
        self.assertEqual(["S1", "S2"], node.subjects)
        self.assertEqual(["S1", "S2", "乌鲁克内殿"], [p["id"] for p in node.pictures])
        self.assertEqual("dialogue", node.directive["action_sequence"]["steps"][1]["kind"])

    def test_future_destination_subject_is_staged_not_shown_in_old_scene(self) -> None:
        world = World(title="阶段资产", here="神庙之巅")
        world.add_place("神庙之巅", "top.png")
        world.add_place("乌鲁克内殿", "hall.png")
        hero = world.add_subject("英雄", "hero.png")
        directive = {
            "place": "神庙之巅",
            "subjects": [hero, "神官"],
            "new_subjects": [
                {
                    "name": "神官",
                    "kind": "人物",
                    "voice": "",
                    "image_prompt": "An ancient priest",
                    "brief": "在内殿等候的神官",
                }
            ],
            "reply": "神官正在内殿等候。",
            "beat": "英雄走到内殿入口前",
        }
        sequence = {
            "goal": "进入乌鲁克内殿与等候的神官交谈",
            "steps": [
                {
                    "instruction": "走到入口前并停稳",
                    "phase": "approach",
                    "relation_change": "distance",
                    "place": "神庙之巅",
                },
                {
                    "instruction": "进入乌鲁克内殿",
                    "phase": "transform",
                    "relation_change": "containment",
                    "place": "乌鲁克内殿",
                },
            ],
        }
        plan = motion._normalize_motion_plan(
            {
                "interaction_type": "locomotion",
                "phase": "approach",
                "relation_change": "distance",
                "actors": ["英雄"],
                "target": "神官",
                "start_state": "hero stands on the temple roof",
                "path": "walks to the entrance",
                "end_state": "hero stops outside",
                "camera_side": "on the roof",
                "framing": "wide",
                "visual_evidence": ["actor_full_body", "target_full_structure", "support_surface"],
                "must_show": [],
                "invariants": ["temple"],
            }
        )
        sequence = action_sequence._normalize_action_sequence(sequence)
        deferred = action_sequence._annotate_sequence_context(
            world,
            directive,
            sequence,
            plan,
            "进入乌鲁克内殿与等候的神官交谈",
        )
        self.assertEqual({"神官"}, deferred)
        self.assertEqual([hero], directive["subjects"])
        self.assertEqual("神庙之巅", sequence["steps"][0]["place"])
        self.assertEqual("乌鲁克内殿", sequence["steps"][1]["place"])

    def test_sequence_context_keeps_gemini_place_instead_of_parsing_user_text(self) -> None:
        """ "Leave the post office to look for…" must not be cut by the server into a nonexistent place."""
        world = World(title="荒镇遗信", here="废弃邮局")
        world.add_place("废弃邮局", "post.png", main_scene="废弃邮局")
        world.add_place("边境小镇街道", "street.png", main_scene="边境小镇街道")
        directive = {
            "place": "边境小镇街道",
            "subjects": [],
            "new_subjects": [],
        }
        sequence = action_sequence._normalize_action_sequence(
            {
                "steps": [
                    {
                        "instruction": "顺着信件提到的地址离开邮局去寻找线索",
                        "phase": "transform",
                        "relation_change": "containment",
                    }
                ],
            }
        )
        action_sequence._annotate_sequence_context(
            world,
            directive,
            sequence,
            motion._normalize_motion_plan(
                {
                    "interaction_type": "boundary",
                    "phase": "transform",
                    "relation_change": "containment",
                    "actors": [],
                    "target": "边境小镇街道",
                    "start_state": "inside post office",
                    "path": "crosses the doorway into the street",
                    "contact_points": "",
                    "end_state": "outside on the street",
                    "camera_side": "side",
                    "framing": "wide",
                    "visual_evidence": [],
                    "must_show": [],
                    "invariants": [],
                }
            ),
            "顺着信件提到的地址离开邮局去寻找线索",
        )
        self.assertEqual("边境小镇街道", directive["place"])
        self.assertEqual("边境小镇街道", sequence["steps"][0]["place"])

    def _scene_repair_fixtures(self) -> tuple[dict, dict]:
        """Output where the director got only the place field wrong, and the correct one after the rerun."""
        bad = {
            "reply": "人物走进了黑暗房间。",
            "mainline": "",
            "objective": "",
            "mode": "continue",
            "transition": False,
            "place": "贝克兰德街头",
            "place_image_prompt": "",
            "place_brief": "",
            "main_scene": "贝克兰德街头",
            "beat": "the actor crosses the open doorway and enters the dark room",
            "action_sequence": {
                "steps": [
                    {
                        "instruction": "穿过敞开的门，进入黑暗房间",
                        "phase": "transform",
                        "relation_change": "containment",
                    }
                ],
            },
            "motion_plan": {
                "interaction_type": "boundary",
                "phase": "transform",
                "relation_change": "containment",
                "actors": ["黑衣旅人"],
                "target": "贝克兰德街头",
                "start_state": "The actor stands outside the doorway on the street.",
                "path": "the actor crosses the open doorway into the dark room",
                "contact_points": "both feet on the threshold and interior floor",
                "end_state": "The actor is fully inside the dark room.",
                "camera_side": "outside the doorway looking inward",
                "framing": "wide",
                "visual_evidence": ["actor_full_body"],
                "must_show": ["the open doorway"],
                "invariants": ["the door frame"],
            },
            "dialogue": "",
            "carry": "",
            "options": [],
            "recommend": 0,
            "continue_with": "",
            "subjects": [],
            "new_subjects": [],
            "implicit_player": {"appears": False, "name": "", "voice": "", "image_prompt": "", "brief": ""},
        }
        repaired = dict(bad)
        repaired.update(
            {
                "mode": "cut",
                "transition": True,
                "place": "黑暗房间",
                "place_image_prompt": "A dark room beyond an open iron doorway, dusty floor and stone walls",
                "place_brief": "门后的昏暗房间",
                "main_scene": "黑暗房间",
            }
        )
        repaired["motion_plan"] = dict(bad["motion_plan"])
        repaired["motion_plan"]["target"] = "黑暗房间"
        repaired["action_sequence"] = {
            "steps": [
                {
                    "instruction": "穿过敞开的门，进入黑暗房间",
                    "phase": "transform",
                    "relation_change": "containment",
                }
            ],
        }
        repaired["subject_definitions"] = (
            "<Subject 1> is the dark room shown in <Picture 1> as a wide interior view."
        )
        repaired.update(
            {
                "summary": "The actor enters the dark room.",
                "retention_analysis": "<Subject 1>: fully_preserved - same room",
                "detailed_description": "[Shot 1] The actor enters <Ref:黑暗房间>.",
                "overall_soundscape": "N/A",
                "non_diegetic_music": "N/A",
            }
        )

        return bad, repaired

    def test_place_cannot_be_registered_again_as_an_object(self) -> None:
        gm = {
            "reply": "木门后是一间废弃酒馆。",
            "mode": "cut",
            "transition": True,
            "place": "废弃酒馆",
            "main_scene": "内城长街",
            "place_image_prompt": "A dim abandoned gothic tavern interior",
            "place_brief": "昏暗破败的古老酒馆",
            "beat": "The camera enters an abandoned tavern.",
            "dialogue": "",
            "subjects": ["废弃酒馆"],
            "new_subjects": [
                {
                    "name": "废弃酒馆",
                    "kind": "物件",
                    "voice": "",
                    "image_prompt": "A dim abandoned gothic tavern interior",
                    "brief": "昏暗破败的古老酒馆",
                }
            ],
            "implicit_player": {
                "appears": False,
                "name": "",
                "voice": "",
                "image_prompt": "",
                "brief": "",
            },
        }
        scene_six = {
            "subject_definitions": (
                "<Subject 1> is the abandoned tavern shown in <Picture 1> as a "
                "single wide scene view; dusty tables and a stone counter."
            ),
            "summary": "[reference_generation] The camera enters <Subject 1>.",
            "retention_analysis": "<Subject 1>: fully_preserved - same tavern",
            "detailed_description": "[Shot 1] The camera enters <Subject 1>.",
            "overall_soundscape": "N/A",
            "non_diegetic_music": "N/A",
        }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            refs, sessions = root / "refs", root / "sessions"
            refs.mkdir()
            sessions.mkdir()
            world = World(title="地点去重", visual_style=STYLE)

            def fake_video(*_args, **_kwargs):
                return SimpleNamespace(
                    video_id="video-place",
                    video_path=Path("place.mp4"),
                    params={},
                    continued=False,
                    artifact_id="artifact-place",
                )

            with (
                patch.object(config, "REFS", refs),
                patch.object(config, "SESSIONS", sessions),
                patch.object(director, "_decide_early", return_value=({**gm, **scene_six}, None)),
                patch.object(loop.imagegen, "generate", return_value=b"place-png") as draw,
                patch.object(loop.video, "generate", side_effect=fake_video),
            ):
                node = loop.turn(world, "推门进入废弃酒馆")

        self.assertEqual({}, world.subjects)
        self.assertEqual(["废弃酒馆"], list(world.places))
        self.assertEqual([], node.subjects)
        self.assertEqual([], node.directive["subjects"])
        self.assertEqual([], node.directive["new_subjects"])
        self.assertEqual(["废弃酒馆"], [p["id"] for p in node.pictures])
        draw.assert_called_once_with(
            "A dim abandoned gothic tavern interior",
            kind="place",
            visual_style=STYLE,
        )

    def test_new_subject_is_not_reclassified_as_place_by_prompt_keywords(self) -> None:
        """The asset type follows the model's structured channel; the server no longer guesses places from words."""
        world = World(title="地点候选", visual_style=STYLE, here="东京天台")
        world.add_place("东京天台", "rooftop.png", "A rainy Tokyo rooftop")
        hero = world.add_subject("雨宫莲", "hero.png")
        directive = {
            "place": "东京天台",
            "subjects": [hero, "宫殿内部走廊"],
            "new_places": [],
            "new_subjects": [
                {
                    # Regression of a real failure: a space labelled as an object, with a prompt likely
                    # to produce a character image.
                    "name": "宫殿内部走廊",
                    "kind": "物件",
                    "voice": "",
                    "image_prompt": (
                        "A wide palace interior corridor, stone floor, tall walls, "
                        "repeating columns and cool overhead light"
                    ),
                    "brief": "宫殿内部的长廊",
                }
            ],
        }
        pending: list[definitions._PendingImage] = []

        staged = action_sequence._stage_declared_places(world, directive, pending)

        self.assertEqual(set(), staged)
        self.assertNotIn("宫殿内部走廊", world.places)
        self.assertEqual("宫殿内部走廊", directive["new_subjects"][0]["name"])
        self.assertEqual([], pending)

    def test_legacy_place_duplicate_is_not_sent_as_a_character(self) -> None:
        world = World(title="旧世界")
        world.add_place("内城长街", "street.png")
        sid = world.add_subject("内城长街", "bad.png", kind="物件")
        pending: list[definitions._PendingImage] = []
        directive = {"subjects": [sid], "new_subjects": []}

        visible = beat_assets._prepare_subjects(world, directive, pending)

        self.assertEqual([], visible)
        self.assertEqual([], pending)

    def test_first_visible_implicit_player_becomes_user_asset(self) -> None:
        gm = {
            "reply": "你退后两步，斗篷被冷雨打湿。",
            "mode": "continue",
            "place": "黑石大门",
            "place_image_prompt": "",
            "beat": "A cloaked traveler steps backward before the gothic gate.",
            "dialogue": "",
            "subjects": [],
            "new_subjects": [],
            # Deliberately collide with an existing world-driven character. The
            # server must create a distinct player instead of stealing control.
            "implicit_player": {
                "appears": True,
                "name": "守门人",
                "voice": "",
                "image_prompt": "A young traveler in a rain-darkened black cloak",
                "brief": "披着黑斗篷的年轻旅人",
            },
        }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            refs, sessions = root / "refs", root / "sessions"
            refs.mkdir()
            sessions.mkdir()
            world = World(title="哥特世界", visual_style=STYLE, here="黑石大门")
            world.add_subject("守门人", "keeper.png", control="ai")
            world.add_place("黑石大门", "gate.png", "A dark gothic gate")

            def fake_video(*_args, **_kwargs):
                return SimpleNamespace(
                    video_id="video-player",
                    video_path=Path("player.mp4"),
                    params={},
                    continued=False,
                    artifact_id="artifact-player",
                )

            with (
                patch.object(config, "REFS", refs),
                patch.object(config, "SESSIONS", sessions),
                patch.object(director, "_decide_early", return_value=({**gm, **_six()}, None)),
                patch.object(loop.imagegen, "generate", return_value=b"player-png"),
                patch.object(loop.video, "generate", side_effect=fake_video),
            ):
                node = loop.turn(world, "退后两步观察石墙")
                saved = World.load(world.id)
                player_ref_bytes = (refs / world.subjects["S2"]["ref"]).read_bytes()

        self.assertEqual("ai", world.subjects["S1"]["control"])
        self.assertEqual("守门人", world.subjects["S1"]["name"])
        self.assertEqual("user", world.subjects["S2"]["control"])
        self.assertEqual("守门人（玩家）", world.subjects["S2"]["name"])
        self.assertEqual(["S2"], node.subjects)
        self.assertEqual(["S2", "黑石大门"], [p["id"] for p in node.pictures])
        self.assertEqual(["S2"], node.directive["subjects"])
        self.assertEqual("user", saved.subjects["S2"]["control"])
        self.assertEqual(b"player-png", player_ref_bytes)

    def test_offscreen_implicit_player_is_not_created(self) -> None:
        world = World(title="无形玩家")
        directive = {
            "subjects": [],
            "new_subjects": [],
            "implicit_player": {
                "appears": False,
                "name": "",
                "voice": "",
                "image_prompt": "",
                "brief": "",
            },
        }

        beat_assets._inject_implicit_player(world, directive)

        self.assertEqual({}, world.subjects)
        self.assertEqual([], directive["new_subjects"])
        self.assertEqual([], directive["subjects"])

    def test_new_person_is_created_once_and_reused_by_name(self) -> None:
        first = {
            "reply": "门开了，一个陌生少年走进糖果屋。",
            "mode": "continue",
            "place": "糖果屋",
            "place_image_prompt": "",
            "beat": "A teenage boy enters the candy house.",
            "dialogue": "",
            "subjects": ["阿洛"],
            "new_subjects": [
                {
                    "name": "阿洛",
                    "image_prompt": (
                        "An anime-style teenage boy with tousled dark hair, expressive eyes, "
                        "and a blue hoodie"
                    ),
                }
            ],
        }
        # The duplicate declaration is intentional: name-based lookup must reuse S1
        # instead of creating a second asset even if the model makes this mistake.
        second = {
            "reply": "阿洛走到糖果桌旁。",
            "mode": "continue",
            "place": "糖果屋",
            "place_image_prompt": "",
            "beat": "The boy walks toward the candy table.",
            "dialogue": "",
            "subjects": ["阿洛"],
            "new_subjects": [
                {
                    "name": "阿洛",
                    "image_prompt": "A conflicting prompt that must not be used",
                }
            ],
        }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            refs, sessions = root / "refs", root / "sessions"
            refs.mkdir()
            sessions.mkdir()
            world = World(
                title="自动角色测试",
                setting="动画风糖果世界",
                visual_style=STYLE,
                here="糖果屋",
            )
            world.add_place("糖果屋", "place.png", "A whimsical candy house")
            progress: list[tuple[str, str]] = []
            video_index = 0

            def fake_video(*_args, **_kwargs):
                nonlocal video_index
                video_index += 1
                return SimpleNamespace(
                    video_id=f"video-{video_index}",
                    video_path=Path(f"clip-{video_index}.mp4"),
                    params={},
                    continued=video_index > 1,
                    artifact_id=f"artifact-{video_index}",
                )

            with (
                patch.object(config, "REFS", refs),
                patch.object(config, "SESSIONS", sessions),
                patch.object(
                    director,
                    "_decide_early",
                    side_effect=[({**first, **_six()}, None), ({**second, **_six()}, None)],
                ),
                patch.object(loop.imagegen, "generate", return_value=b"subject-png") as draw,
                patch.object(loop.video, "generate", side_effect=fake_video),
            ):
                node1 = loop.turn(
                    world,
                    "一个叫阿洛的少年走进来",
                    progress=lambda stage, detail="": progress.append((stage, detail)),
                )
                node2 = loop.turn(world, "阿洛走向桌边")
                node3 = loop.regenerate(world, six_section=_six())
                saved = World.load(world.id)

            self.assertEqual(["S1"], list(world.subjects))
            self.assertEqual("阿洛", world.subjects["S1"]["name"])
            self.assertEqual(1, draw.call_count)
            self.assertEqual("subject", draw.call_args.kwargs["kind"])
            self.assertEqual(STYLE, draw.call_args.kwargs["visual_style"])
            self.assertEqual(b"subject-png", (refs / world.subjects["S1"]["ref"]).read_bytes())
            self.assertEqual(["S1"], node1.subjects)
            self.assertEqual(["S1"], node2.subjects)
            self.assertEqual(node1.refs[0], node2.refs[0])
            self.assertEqual(["S1", "糖果屋"], [p["id"] for p in node1.pictures])
            self.assertEqual(["S1"], node1.directive["subjects"])
            # The progress must name who is being drawn; the wording may change, but the new
            # character's name must be there
            self.assertTrue(any("新角色「阿洛」" in detail for _, detail in progress))
            # The progress must name the video model **this world actually uses**, never a
            # hard-coded one.
            label = loop.video.model(world.video_model).label
            gen = [d for st, d in progress if st == "generate"]
            self.assertTrue(gen and all(label in d for d in gen), gen)
            self.assertTrue(
                node1.six_section["detailed_description"].startswith(f"[Shot 1] Visual style: {STYLE}.")
            )
            self.assertTrue(
                node3.six_section["detailed_description"].startswith(f"[Shot 1] Visual style: {STYLE}.")
            )

            self.assertEqual("阿洛", saved.subjects["S1"]["name"])
            self.assertEqual(STYLE, saved.visual_style)
            self.assertEqual(3, len(saved.nodes))

    def test_image_style_wrapper_is_stable_and_optional(self) -> None:
        from murea_agent.providers import imagegen

        prompt = imagegen._with_visual_style("A whimsical candy house", STYLE)
        self.assertIn("A whimsical candy house", prompt)
        self.assertIn("MANDATORY WORLD VISUAL STYLE", prompt)
        self.assertIn(STYLE, prompt)
        self.assertEqual(
            "A whimsical candy house",
            imagegen._with_visual_style(
                "A whimsical candy house",
                "",
            ),
        )

    def test_manual_asset_generation_receives_world_style(self) -> None:
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(config, "REFS", Path(tmp)),
            patch.object(assets.imagegen, "generate", return_value=b"styled-place") as draw,
        ):
            filename = asyncio.run(
                assets._resolve_ref(
                    ref="",
                    image=None,
                    image_prompt="A candy kitchen",
                    kind="place",
                    prefix="place",
                    visual_style=STYLE,
                )
            )
            self.assertEqual(b"styled-place", (Path(tmp) / filename).read_bytes())
            draw.assert_called_once_with(
                "A candy kitchen",
                kind="place",
                visual_style=STYLE,
            )

    def test_create_world_persists_style_and_legacy_worlds_still_load(self) -> None:
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(
                config,
                "SESSIONS",
                Path(tmp),
            ),
        ):
            created = asyncio.run(
                worlds.create_world(
                    _JsonRequest(
                        {
                            "title": "糖果世界",
                            "setting": "少年探索糖果屋",
                            "visual_style": STYLE,
                        }
                    ),
                    {"id": DEFAULT_USER_ID},
                )
            )
            self.assertEqual(STYLE, created["visual_style"])
            self.assertEqual(STYLE, World.load(created["id"]).visual_style)

            legacy = dict(created)
            legacy["id"] = "wlegacy"
            legacy.pop("visual_style")
            (Path(tmp) / "wlegacy.json").write_text(
                json.dumps(legacy, ensure_ascii=False),
                encoding="utf-8",
            )
            self.assertEqual("", World.load("wlegacy").visual_style)


class FailedTurnTest(unittest.TestCase):
    """When a turn fails, the reference images drawn in it must be removed.

    world.save() is the last step, so a failure midway never saves the world, but the images are
    already in data/refs. Nothing refers to them and delete_world cannot remove them (it only
    removes files some world refers to). Video jobs occasionally fail, so this path is not rare.
    """

    def test_reference_drawn_this_turn_is_removed(self):
        with tempfile.TemporaryDirectory() as tmp:
            refs = Path(tmp) / "refs"
            refs.mkdir()
            sessions = Path(tmp) / "sessions"
            sessions.mkdir()
            world = World(id="wfail", title="t", setting="", visual_style=STYLE, here="")
            with (
                patch.object(config, "REFS", refs),
                patch.object(config, "SESSIONS", sessions),
                patch.object(
                    director,
                    "_decide_early",
                    return_value=(
                        {
                            "reply": "好",
                            "mode": "cut",
                            "place": "会议室",
                            "place_image_prompt": "A meeting room",
                            "beat": "走进来",
                            "dialogue": "",
                            "carry": "",
                            "options": [],
                            "continue_with": "",
                            "subjects": [],
                            "new_subjects": [],
                            **_six(),
                        },
                        None,
                    ),
                ),
                patch.object(loop.imagegen, "generate", return_value=b"place-png"),
                patch.object(loop.video, "generate", side_effect=RuntimeError("job failed")),
            ):
                with self.assertRaises(RuntimeError):
                    loop.turn(world, "他走进会议室")

            self.assertEqual([], list(refs.iterdir()), "失败的那一轮把参考图留在盘上了")


class PrefetchWarmTest(unittest.TestCase):
    """Prefetch draws reference images too: the user is watching the previous clip, free time."""

    def setUp(self) -> None:
        beat_assets._warm.clear()
        prefetch_mod._prefetched.clear()
        self.addCleanup(beat_assets._warm.clear)
        self.addCleanup(prefetch_mod._prefetched.clear)

    @staticmethod
    def _decide_returning(part):
        def fake(world, text, on_partial=None, **kw):
            if on_partial:
                on_partial(part)
            return part

        return fake

    def test_prefetch_queues_the_new_reference_images(self) -> None:
        world = World(title="t")
        world.head = "n1"
        world.visual_style = "dark"
        part = {
            "reply": "走进去",
            "place": "新地点",
            "place_image_prompt": "a new place, wide shot",
            "new_subjects": [{"name": "新人物", "image_prompt": "a man"}],
        }
        with (
            patch.object(director, "_decide", self._decide_returning(part)),
            patch.object(beat_assets, "_timed_draw", lambda *a, **k: (b"x", 0.0)),
        ):
            self.assertTrue(prefetch_mod.prefetch(world, "走进去"))
            # `prefetch` is asynchronous now: it stores **the call itself** and returns at once.
            # When the user clicks, `_take_prefetched` joins it; here too, wait for it first.
            self.assertIsNotNone(prefetch_mod._take_prefetched(world, "走进去"))
        self.assertEqual(2, len(beat_assets._warm), "新主体和新地点两张都该在用户点之前就排进队列")

    def test_warm_failure_never_breaks_prefetch(self) -> None:
        """Early drawing is purely a speed-up. If it fails, prefetch must return as usual."""
        world = World(title="t")
        world.head = "n1"
        with (
            patch.object(director, "_decide", self._decide_returning({"reply": "x"})),
            patch.object(loop, "_warm_images", side_effect=RuntimeError("boom")),
        ):
            self.assertTrue(prefetch_mod.prefetch(world, "x"))
            self.assertIsNotNone(
                prefetch_mod._take_prefetched(world, "x"), "提前开画挂了，预推演本身仍然要拿得到结果"
            )


if __name__ == "__main__":
    unittest.main()
