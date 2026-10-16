"""World state layer: map, carried items, choices.

These three separate "a world" from "a text-to-video page", and all of them live in `snapshot()`:
the director only sees what this function returns each turn. Anything missing has to be guessed
again from the chat log, so these tests make sure it does not shrink.

Pure logic, no model calls, milliseconds.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from murea_agent.memory.world import Node, World


def _world() -> World:
    w = World(id="wmap", title="t", setting="写字楼", visual_style="", here="")
    for name in ("会议室", "走廊", "茶水间"):
        w.places[name] = {"ref": f"{name}.png", "desc": f"{name}的样子", "first_seen": ""}
    return w


def _walk(w: World, *places: str) -> None:
    """Walk the person through these places in order."""
    parent = None
    for i, place in enumerate(places):
        n = Node(
            id=f"n{i}",
            parent=parent,
            video_id="",
            clip=f"{i}.mp4",
            place=place,
            beat=f"到{place}",
            mode="cut" if i else "cut",
        )
        w.add_node(n)
        parent = n.id


class MapTest(unittest.TestCase):
    def test_first_segment_cannot_be_rolled_back(self):
        from dataclasses import asdict
        w = _world()
        _walk(w, "会议室")
        w.history = [{"role": "user", "content": "开始"}, {"role": "model", "content": "开场"}]
        before = asdict(w)
        self.assertFalse(w.rollback())
        self.assertEqual(before, asdict(w))

    def test_rollback_stops_at_first_segment_even_with_detached_nodes(self):
        w = _world()
        _walk(w, "会议室", "走廊", "茶水间")
        self.assertTrue(w.rollback())
        self.assertTrue(w.rollback())
        self.assertFalse(w.rollback())
        self.assertEqual(w.head, "n0")
        self.assertEqual(len(w.nodes), 3)

    def test_walking_records_the_door(self):
        w = _world()
        _walk(w, "会议室", "走廊", "茶水间")
        self.assertEqual(
            {"会议室": ["走廊"], "走廊": ["会议室", "茶水间"], "茶水间": ["走廊"]},
            w.adjacency(),
        )

    def test_never_walked_places_have_no_doors(self):
        """Nobody walked between the meeting room and the pantry; being in places does not make them adjacent."""
        w = _world()
        _walk(w, "会议室", "走廊", "茶水间")
        self.assertNotIn("茶水间", w.adjacency()["会议室"])

    def test_doors_are_two_way(self):
        w = _world()
        _walk(w, "会议室", "走廊")
        self.assertIn("会议室", w.adjacency()["走廊"])

    def test_rollback_does_not_unknow_the_door(self):
        """Undoing a clip does not make the corridor and the meeting room non-adjacent: that is knowledge about the world,
        not about the current story line.
        """
        w = _world()
        _walk(w, "会议室", "走廊")
        w.rollback()
        self.assertIn("走廊", w.adjacency()["会议室"])

    def test_old_worlds_get_a_map_without_migration(self):
        """place_links was added later and is empty in older worlds. Reading only stored links, old worlds would never get a
        map; it must be derivable from the clip tree.
        """
        w = _world()
        _walk(w, "会议室", "走廊", "茶水间")
        w.place_links = []  # simulate an old world
        self.assertEqual(["走廊"], w.adjacency()["会议室"])

    def test_snapshot_carries_the_map_and_the_descriptions(self):
        w = _world()
        _walk(w, "会议室", "走廊")
        snap = w.snapshot()
        self.assertEqual({"会议室": ["走廊"], "走廊": ["会议室"], "茶水间": []}, snap["connections"])
        # With only names the director does not know what the meeting room looks like and must guess
        # from the chat log every turn
        self.assertEqual("会议室的样子", snap["places"]["会议室"])
        self.assertEqual(
            {
                "会议室": "会议室",
                "走廊": "走廊",
                "茶水间": "茶水间",
            },
            snap["scene_groups"],
        )

    def test_subscene_shares_its_main_scene(self):
        w = _world()
        w.add_place("潮汕牛肉餐馆", "restaurant.png")
        w.add_place("餐馆包间", "room.png", main_scene="潮汕牛肉餐馆")
        self.assertEqual("潮汕牛肉餐馆", w.snapshot()["scene_groups"]["餐馆包间"])


class CarryTest(unittest.TestCase):
    def test_snapshot_reports_what_is_in_hand(self):
        w = _world()
        n = Node(
            id="n1",
            parent=None,
            video_id="",
            clip="1.mp4",
            place="会议室",
            beat="拿起杯子",
            mode="cut",
            carry="holding a white paper cup",
        )
        w.add_node(n)
        self.assertEqual("holding a white paper cup", w.snapshot()["carried"])

    def test_nothing_in_hand_is_stated_not_omitted(self):
        """A missing field and "holding nothing" are different; the director must be able to tell them apart."""
        w = _world()
        w.add_node(
            Node(id="n1", parent=None, video_id="", clip="1.mp4", place="会议室", beat="坐着", mode="cut")
        )
        self.assertEqual("(nothing in particular)", w.snapshot()["carried"])


class ActorControlTest(unittest.TestCase):
    """People in the world are not puppets: the user drives only their own character, the director plays the rest."""

    def test_snapshot_tells_the_gm_who_drives_whom(self):
        w = _world()
        w.add_subject("阿明", "s1.png", control="user")
        w.add_subject("小红", "s2.png")  # driven by the world by default
        snap = w.snapshot()["subjects"]
        self.assertEqual("user", snap["S1"]["driven_by"])
        self.assertEqual("world (you play them)", snap["S2"]["driven_by"])

    def test_gm_invented_characters_are_world_driven(self):
        """Passers-by appearing in the conversation belong to the world, not the user; the default must be ai."""
        w = _world()
        sid = w.add_subject("保洁阿姨", "s.png")
        self.assertEqual("ai", w.subjects[sid]["control"])

    def test_old_worlds_without_the_field_are_world_driven(self):
        """control was added later and old worlds lack it. Missing must not crash, and must not default to the user."""
        w = _world()
        w.subjects["S1"] = {"name": "阿明", "ref": "s.png", "desc": ""}
        self.assertEqual("world (you play them)", w.snapshot()["subjects"]["S1"]["driven_by"])


class ContinueWithTest(unittest.TestCase):
    """When one sentence spans several spaces, the director puts the unplayed part in continue_with and the frontend continues."""

    def test_round_trip(self):
        w = _world()
        w.add_node(
            Node(
                id="n1",
                parent=None,
                video_id="",
                clip="1.mp4",
                place="会议室",
                beat="走出去",
                mode="cut",
                continue_with="接着坐电梯下到一楼，穿过马路进餐厅",
            )
        )
        self.assertEqual("接着坐电梯下到一楼，穿过马路进餐厅", w.head_node.continue_with)

    def test_auto_steps_are_marked(self):
        """Beats the system continues on its own must be distinguishable from what the user typed in the UI, or the user sees
        lines they never said.
        """
        w = _world()
        w.add_node(
            Node(
                id="n1",
                parent=None,
                video_id="",
                clip="1.mp4",
                place="会议室",
                beat="走出去",
                mode="cut",
                origin="turn",
            )
        )
        w.add_node(
            Node(
                id="n2",
                parent="n1",
                video_id="",
                clip="2.mp4",
                place="走廊",
                beat="到走廊",
                mode="cut",
                origin="auto",
            )
        )
        self.assertEqual(["turn", "auto"], [n.origin for n in w.lineage()])


class OptionsTest(unittest.TestCase):
    def test_options_round_trip(self):
        w = _world()
        w.add_node(
            Node(
                id="n1",
                parent=None,
                video_id="",
                clip="1.mp4",
                place="会议室",
                beat="坐着",
                mode="cut",
                options=["推门出去", "翻一翻文件"],
            )
        )
        self.assertEqual(["推门出去", "翻一翻文件"], w.head_node.options)


if __name__ == "__main__":
    unittest.main(verbosity=2)
