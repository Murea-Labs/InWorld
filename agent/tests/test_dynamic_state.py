"""Pure-logic regression tests of character and place dynamic state.

No model calls; covers migration, director state patches, branch rollback and the boundaries of the
video prompt format.
"""

from __future__ import annotations

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
from murea_agent.director import director  # noqa: E402
from murea_agent import loop
from murea_agent.memory.world import Node, World


class DynamicStateTest(unittest.TestCase):
    def _world(self) -> World:
        world = World(id="wdyn", title="动态状态测试", setting="写字楼")
        world.add_subject("阿明", "s1.png", "三十岁男性，浅蓝衬衫")
        world.add_subject("小红", "s2.png", "二十多岁女性，红色外套")
        world.add_place("会议室", "p1.png", "有长桌和白板")
        world.here = "会议室"
        return world

    def test_legacy_assets_receive_empty_state_and_snapshot_exposes_it(self) -> None:
        world = self._world()
        world.subjects["S1"].pop("state")
        world.places["会议室"].pop("state")
        snap = world.snapshot()
        self.assertEqual("", snap["subjects"]["S1"]["state"]["outfit"])
        self.assertEqual([], snap["place_states"]["会议室"]["props"])
        self.assertEqual([], snap["characters_at_place"]["会议室"])

    def test_apply_updates_merges_lists_and_stamps_visible_location(self) -> None:
        world = self._world()
        changes = world.apply_dynamic_updates(
            {
                "subjects": [
                    {
                        "name": "阿明",
                        "outfit": "雨水打湿的深蓝外套",
                        "condition": "左手缠着绷带",
                        "knowledge_add": ["知道地下室入口"],
                        "relation_updates": [{"target": "小红", "relation": "开始信任她"}],
                    }
                ],
                "places": [
                    {
                        "name": "会议室",
                        "time_of_day": "傍晚",
                        "weather": "窗外下雨",
                        "lighting": "只有桌面台灯亮着",
                        "props_add": ["打开的文件"],
                    }
                ],
            },
            current_place="会议室",
            visible_subject_ids=["S1"],
        )

        state = world.subjects["S1"]["state"]
        self.assertEqual("会议室", state["location"])
        self.assertEqual("雨水打湿的深蓝外套", state["outfit"])
        self.assertEqual(["知道地下室入口"], state["knowledge"])
        self.assertEqual("开始信任她", state["relations"]["S2"])
        self.assertEqual("傍晚", world.places["会议室"]["state"]["time_of_day"])
        self.assertEqual(["打开的文件"], world.places["会议室"]["state"]["props"])
        self.assertIn("S1", changes["subjects"])
        self.assertIn("会议室", changes["places"])

    def test_rollback_restores_dynamic_state_for_branch(self) -> None:
        world = self._world()
        before = world.dynamic_state_snapshot()
        world.apply_dynamic_updates(
            {
                "subjects": [{"name": "阿明", "mood": "眉头紧皱"}],
                "places": [{"name": "会议室", "condition": "白板留下新字"}],
            },
            current_place="会议室",
            visible_subject_ids=["S1"],
        )
        after = world.dynamic_state_snapshot()
        world.add_node(
            Node(
                id="n1",
                parent=None,
                video_id="",
                clip="1.mp4",
                place="会议室",
                beat="阿明写字",
                mode="continue",
                state_before=before,
                state_after=after,
            )
        )
        self.assertEqual("眉头紧皱", world.subjects["S1"]["state"]["mood"])
        self.assertTrue(world.rollback())
        self.assertEqual(before, world.dynamic_state_snapshot())

    def test_pipeline_applies_state_without_appending_machine_clause(self) -> None:
        world = self._world()
        refs, sessions = Path(tempfile.mkdtemp()) / "refs", Path(tempfile.mkdtemp()) / "sessions"
        refs.mkdir(parents=True, exist_ok=True)
        sessions.mkdir(parents=True, exist_ok=True)
        (refs / "s1.png").write_bytes(b"s1")
        (refs / "p1.png").write_bytes(b"p1")
        fake_directive = {
            "reply": "阿明皱着眉打开文件。",
            "mode": "continue",
            "place": "会议室",
            "place_image_prompt": "",
            "main_scene": "会议室",
            "beat": "Amin opens the file",
            "dialogue": "",
            "carry": "",
            "options": [],
            "continue_with": "",
            "subjects": ["S1"],
            "new_subjects": [],
            "state_updates": {
                "subjects": [{"name": "阿明", "outfit": "湿漉漉的深蓝外套", "mood": "眉头紧皱"}],
                "places": [{"name": "会议室", "lighting": "桌面台灯亮着"}],
            },
        }
        six = {
            "subject_definitions": "",
            "summary": "[reference_generation] <Subject 1> opens the file.",
            "retention_analysis": "<Subject 1> (appears in [Shot 1] (waist-up)): fully_preserved - same",
            "detailed_description": "[Shot 1] <Subject 1> opens a file in the room.",
            "overall_soundscape": "N/A",
            "non_diegetic_music": "N/A",
        }
        generation = SimpleNamespace(
            video_id="v1",
            video_path=Path("clip.mp4"),
            params={},
            continued=False,
            artifact_id="a1",
        )
        with (
            patch.object(config, "REFS", refs),
            patch.object(config, "SESSIONS", sessions),
            patch.object(director, "_decide_early", return_value=({**fake_directive, **six}, None)),
            patch.object(loop.video, "generate", return_value=generation),
        ):
            node = loop.turn(world, "阿明打开文件")
        self.assertEqual("湿漉漉的深蓝外套", world.subjects["S1"]["state"]["outfit"])
        self.assertEqual("会议室", world.subjects["S1"]["state"]["location"])
        # Dynamic state is still read and written by the director / rewriter, but the server no
        # longer appends `Dynamic world state — ...` machine fields after the original six sections.
        self.assertNotIn("Dynamic world state", node.prompt)
        self.assertNotIn("湿漉漉的深蓝外套", node.prompt)
        self.assertNotIn("桌面台灯亮着", node.prompt)
        self.assertIsNotNone(node.state_before)
        self.assertIsNotNone(node.state_after)


if __name__ == "__main__":
    unittest.main()
