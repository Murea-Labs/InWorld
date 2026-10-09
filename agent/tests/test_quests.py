"""Pure-logic regression tests of the world quest layer; no calls to the language or video model."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from murea_agent.memory.world import Node, World


def _world() -> World:
    world = World(
        id="wquest",
        title="任务测试",
        setting="被雨困住的港城",
        mainline="在风暴抵达前关闭海底闸门",
        objective="找到旧泵房入口",
        here="码头",
    )
    world.add_place("码头", "dock.png")
    world.add_place("旧泵房", "pump.png")
    return world


class QuestTest(unittest.TestCase):
    def test_legacy_mainline_becomes_structured_main_quest(self) -> None:
        world = _world()
        snapshot = world.quest_state_snapshot()

        self.assertEqual(1, len(snapshot["quests"]))
        main = snapshot["quests"][0]
        self.assertEqual("main", main["type"])
        self.assertEqual(world.mainline, main["title"])
        self.assertEqual(world.objective, main["progress"])
        self.assertEqual("码头", main["place"])

    def test_immediate_legacy_mainline_is_migrated_to_opening_context(self) -> None:
        world = World(
            id="wlegacy",
            title="未命名世界",
            setting="西部荒漠深处废弃的现代化医院",
            mainline="爆破倒计时显示还剩十秒，而你的双腿被卡在坍塌的护士台下。",
            objective="进入隔壁病房进行搜查",
            here="护士站",
        )
        world.add_place("护士站", "nurse.png")

        snapshot = world.snapshot()

        self.assertEqual(
            "爆破倒计时显示还剩十秒，而你的双腿被卡在坍塌的护士台下。",
            snapshot["opening_situation"],
        )
        self.assertNotEqual(snapshot["opening_situation"], snapshot["mainline"])
        self.assertIn("爆破", snapshot["mainline"])
        self.assertIn("爆炸前", snapshot["mainline"])
        self.assertEqual("脱离开场危机", snapshot["mainline_phase"])
        self.assertTrue(snapshot["mainline_milestones"])

    def test_new_immediate_mainline_keeps_original_text_over_reply(self) -> None:
        world = World(
            id="wnewlegacy",
            setting="废弃医院",
            mainline="爆破倒计时还剩十秒，你被卡在护士台下。",
        )
        world.add_place("护士站", "nurse.png")
        world.set_mainline(
            "",
            phase="脱离开场危机",
            opening_situation="你抬头看见走廊尽头的红灯。",
        )

        self.assertEqual("爆破倒计时还剩十秒，你被卡在护士台下。", world.opening_situation)
        self.assertIn("爆破危机", world.mainline)

    def test_durable_mainline_is_not_rewritten(self) -> None:
        world = _world()
        world.mainline_phase = "关闭闸门"
        world.mainline_milestones = ["找到入口", "关闭闸门", "确认安全"]

        snapshot = world.snapshot()

        self.assertEqual("在风暴抵达前关闭海底闸门", snapshot["mainline"])
        self.assertEqual("关闭闸门", snapshot["mainline_phase"])
        self.assertEqual(["找到入口", "关闭闸门", "确认安全"], snapshot["mainline_milestones"])

    def test_mainline_phase_and_milestones_follow_new_turn(self) -> None:
        world = _world()
        world.set_mainline(
            "在风暴抵达前关闭海底闸门",
            phase="寻找闸门控制室",
            milestones=["找到入口", "找到控制室", "关闭闸门"],
        )
        before = world.quest_state_snapshot()
        world.objective = "进入控制室启动闸门"
        world.set_mainline("", phase="关闭闸门")
        world.apply_quest_updates({"create": [], "update": []}, current_place="旧泵房", node_id="n2")

        main = next(quest for quest in world.quests if quest["type"] == "main")
        self.assertEqual("关闭闸门", main["phase"])
        self.assertEqual("进入控制室启动闸门", main["progress"])
        self.assertEqual(["找到入口", "找到控制室", "关闭闸门"], main["milestones"])
        self.assertEqual("寻找闸门控制室", before["mainline_phase"])

    def test_create_side_quest_binds_known_place(self) -> None:
        world = _world()
        changed = world.apply_quest_updates(
            {
                "create": [
                    {
                        "type": "side",
                        "title": "找回潮汐日志",
                        "summary": "守塔人遗失了记录潮汐的日志。",
                        "progress": "日志最后出现在旧泵房",
                        "place": "旧泵房",
                    }
                ],
                "update": [],
            },
            current_place="码头",
            node_id="n1",
        )

        side = next(quest for quest in world.quests if quest["type"] == "side")
        self.assertEqual("旧泵房", side["place"])
        self.assertEqual("n1", side["created_node"])
        self.assertEqual([side["id"]], changed["created"])

    def test_unknown_place_does_not_create_ghost_map_node(self) -> None:
        world = _world()
        world.apply_quest_updates(
            {
                "create": [
                    {
                        "type": "side",
                        "title": "寻找无线电",
                        "summary": "",
                        "progress": "",
                        "place": "不存在的灯塔",
                    }
                ],
                "update": [],
            },
            current_place="",
            node_id="n1",
        )

        side = next(quest for quest in world.quests if quest["type"] == "side")
        self.assertEqual("", side["place"])
        self.assertNotIn("不存在的灯塔", world.places)

    def test_update_uses_stable_id_and_unknown_id_is_ignored(self) -> None:
        world = _world()
        world.apply_quest_updates(
            {
                "create": [
                    {
                        "type": "side",
                        "title": "找回潮汐日志",
                        "summary": "",
                        "progress": "获得守塔人的线索",
                        "place": "码头",
                    }
                ],
                "update": [],
            },
            current_place="码头",
            node_id="n1",
        )
        side = next(quest for quest in world.quests if quest["type"] == "side")
        changed = world.apply_quest_updates(
            {
                "create": [],
                "update": [
                    {"id": side["id"], "status": "completed", "progress": "日志已归还", "place": "码头"},
                    {"id": "Q404", "status": "failed", "progress": "不应出现", "place": "码头"},
                ],
            },
            current_place="码头",
            node_id="n2",
        )

        side = next(quest for quest in world.quests if quest["id"] == side["id"])
        self.assertEqual("completed", side["status"])
        self.assertEqual("日志已归还", side["progress"])
        self.assertEqual([side["id"]], changed["updated"])
        self.assertEqual(2, len(world.quests))

    def test_duplicate_title_is_not_created_twice(self) -> None:
        world = _world()
        update = {
            "create": [
                {
                    "type": "side",
                    "title": "找回潮汐日志",
                    "summary": "",
                    "progress": "",
                    "place": "码头",
                }
            ],
            "update": [],
        }
        world.apply_quest_updates(update, current_place="码头", node_id="n1")
        world.apply_quest_updates(update, current_place="码头", node_id="n2")
        self.assertEqual(1, len([q for q in world.quests if q["type"] == "side"]))

    def test_rollback_restores_branch_quest_snapshot(self) -> None:
        world = _world()
        before = world.quest_state_snapshot()
        world.objective = "进入旧泵房"
        world.apply_quest_updates(
            {
                "create": [
                    {
                        "type": "side",
                        "title": "找回潮汐日志",
                        "summary": "",
                        "progress": "发现日志线索",
                        "place": "旧泵房",
                    }
                ],
                "update": [],
            },
            current_place="旧泵房",
            node_id="n1",
        )
        after = world.quest_state_snapshot()
        world.add_node(
            Node(
                id="n1",
                parent=None,
                video_id="",
                clip="1.mp4",
                place="旧泵房",
                beat="进入泵房",
                mode="cut",
                objective=world.objective,
                quest_before=before,
                quest_after=after,
            )
        )

        self.assertEqual(2, len(world.quests))
        self.assertTrue(world.rollback())
        self.assertEqual(before, world.quest_state_snapshot())

    def test_main_progress_follows_objective(self) -> None:
        world = _world()
        world.quest_state_snapshot()
        world.objective = "启动备用发电机"
        world.apply_quest_updates({"create": [], "update": []}, current_place="旧泵房", node_id="n2")
        main = next(quest for quest in world.quests if quest["type"] == "main")
        self.assertEqual("启动备用发电机", main["progress"])
        self.assertEqual("旧泵房", main["place"])

    def test_snapshot_exposes_quest_table(self) -> None:
        snapshot = _world().snapshot()
        self.assertIn("quests", snapshot)
        self.assertEqual("main", snapshot["quests"][0]["type"])

    def test_rollback_restores_mainline_phase_and_opening(self) -> None:
        world = _world()
        world.mainline_phase = "找到入口"
        world.mainline_milestones = ["找到入口", "关闭闸门"]
        before = world.quest_state_snapshot()
        world.mainline_phase = "关闭闸门"
        world.objective = "确认闸门已经闭合"
        after = world.quest_state_snapshot()
        world.add_node(
            Node(
                id="nphase",
                parent=None,
                video_id="",
                clip="1.mp4",
                place="旧泵房",
                beat="关闭闸门",
                mode="cut",
                objective=world.objective,
                quest_before=before,
                quest_after=after,
            )
        )

        self.assertTrue(world.rollback())
        restored = world.quest_state_snapshot()
        self.assertEqual(before["mainline_phase"], restored["mainline_phase"])
        self.assertEqual(before["mainline_milestones"], restored["mainline_milestones"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
