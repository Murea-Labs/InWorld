"""Critic: replaces choices from real frames, and must never become a new failure source.

Why: the three choices are written by the director **before rendering** from its own plan, without
knowing what the video model filmed. The result is choices that do not match the picture: the person
is already inside the pipe and the choice still says "jump into the pipe". The critic runs during
playback (watching plus thinking, median about 25 s) and replaces them from real frames.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from murea_agent import critic, loop
from murea_agent.memory.world import Node, World

GOOD = {
    "observed": "探险家双手撑在管壁上，前方几步是滚动的金苹果。",
    "options": ["加速前扑抓住金苹果", "贴着管壁减速稳住身形", "喊一声看有没有回音"],
    "recommend": 0,
    "objective": "在管道中追上金苹果",
    "picked_up": {"name": "", "look": ""},
    "dropped": [],
    "appeared": {"name": "", "look": ""},
}


def _world_with_node() -> tuple[World, Node]:
    world = World(title="t")
    node = Node(
        id="n1",
        parent=None,
        video_id="v",
        clip="c.mp4",
        place="黄铜管道内部",
        beat="b",
        mode="continue",
        options=["导演写的一", "导演写的二", "导演写的三"],
        recommend=1,
        objective="导演写的目标",
    )
    node.frames = ["f1.jpg"]
    world.nodes[node.id] = asdict(node)
    world.head = node.id
    return world, node


class AdjudicatorTest(unittest.TestCase):
    def setUp(self) -> None:
        # The success path of adjudicate() calls world.save(). The test world must live in a
        # temporary directory, or every full test run would create a world titled `t` in the real
        # data, appearing after the next refresh as if refreshing created worlds.
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        sessions = Path(self.tmp.name) / "sessions"
        sessions.mkdir()
        sessions_patch = patch.object(loop.config, "SESSIONS", sessions)
        sessions_patch.start()
        self.addCleanup(sessions_patch.stop)

    def test_real_frame_options_replace_the_directors(self) -> None:
        world, node = _world_with_node()
        with (
            patch.object(critic, "_frames_for_gm", return_value=[("image/jpeg", b"x")]),
            patch.object(loop.llm, "chat", return_value=GOOD),
        ):
            verdict = critic.adjudicate(world, node)
        self.assertIsNotNone(verdict)
        self.assertEqual(GOOD["options"], node.options)
        self.assertEqual(0, node.recommend)
        self.assertEqual("在管道中追上金苹果", node.objective)
        self.assertEqual(
            GOOD["options"], world.nodes["n1"]["options"], "改完要落库，否则刷新页面又变回导演那版"
        )

    def test_no_frames_means_no_verdict(self) -> None:
        """No frames, no truth: better not judge, the frames are the critic's only evidence."""
        world, node = _world_with_node()
        with patch.object(critic, "_frames_for_gm", return_value=[]), patch.object(loop.llm, "chat") as chat:
            self.assertIsNone(critic.adjudicate(world, node))
        chat.assert_not_called()
        self.assertEqual(["导演写的一", "导演写的二", "导演写的三"], node.options)

    def test_a_failed_call_leaves_the_directors_options_alone(self) -> None:
        """A quality pass must never become a new failure source; same rule as prefetch."""
        world, node = _world_with_node()
        with (
            patch.object(critic, "_frames_for_gm", return_value=[("image/jpeg", b"x")]),
            patch.object(loop.llm, "chat", side_effect=RuntimeError("boom")),
        ):
            self.assertIsNone(critic.adjudicate(world, node))
        self.assertEqual(["导演写的一", "导演写的二", "导演写的三"], node.options)
        self.assertEqual(1, node.recommend)
        self.assertEqual("导演写的目标", node.objective)

    def test_fewer_than_three_options_is_rejected_wholesale(self) -> None:
        """The UI lays out three. Padding produces robotic text, so drop them all instead."""
        world, node = _world_with_node()
        with (
            patch.object(critic, "_frames_for_gm", return_value=[("image/jpeg", b"x")]),
            patch.object(loop.llm, "chat", return_value={**GOOD, "options": ["只有一条"]}),
        ):
            self.assertIsNone(critic.adjudicate(world, node))
        self.assertEqual(3, len(node.options))

    def test_a_bad_recommend_index_is_clamped(self) -> None:
        world, node = _world_with_node()
        with (
            patch.object(critic, "_frames_for_gm", return_value=[("image/jpeg", b"x")]),
            patch.object(loop.llm, "chat", return_value={**GOOD, "recommend": 9}),
        ):
            critic.adjudicate(world, node)
        self.assertIn(node.recommend, (0, 1, 2))

    def test_an_empty_objective_does_not_wipe_the_old_one(self) -> None:
        """The objective updates every beat; if the critic gives none keep the old one instead of clearing guidance."""
        world, node = _world_with_node()
        with (
            patch.object(critic, "_frames_for_gm", return_value=[("image/jpeg", b"x")]),
            patch.object(loop.llm, "chat", return_value={**GOOD, "objective": "  "}),
        ):
            critic.adjudicate(world, node)
        self.assertEqual("导演写的目标", node.objective)


class PickedUpTest(unittest.TestCase):
    """Things really picked up in the picture → item assets, first with the last frame, then a standalone image queued.

    Items used to count as soon as the director mentioned them in its plan, with a reference drawn
    fresh by the image model (seconds plus cost), and that drawing was **not the same item** the
    video model filmed. Now the critic confirms from real frames, the last frame is usable at once
    and the standalone image is drawn during playback.
    """

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        (root / "frames").mkdir()
        (root / "refs").mkdir()
        (root / "frames" / "f1.jpg").write_bytes(b"\xff\xd8fake-frame")
        for name, path in (("FRAMES", root / "frames"), ("REFS", root / "refs")):
            p = patch.object(loop.config, name, path)
            p.start()
            self.addCleanup(p.stop)

    def _run(self, picked: dict) -> tuple[World, Node, str]:
        world, node = _world_with_node()
        got = critic._register_picked_up(world, node, picked)
        return world, node, got

    def test_a_visible_pickup_becomes_an_item_asset_from_the_frame(self) -> None:
        world, node, got = self._run({"name": "铜灯笼", "look": "a dented brass lantern with warm oil flame"})
        self.assertEqual("铜灯笼", got)
        item = next(v for v in world.subjects.values() if v["name"] == "铜灯笼")
        self.assertEqual("物件", item["kind"])
        self.assertEqual("a dented brass lantern with warm oil flame", item["definition"])
        self.assertEqual("n1", item["from_node"], "卡片要看得见它是从哪一段来的")
        # Before the background image starts, the temporary ref is that frame's content
        self.assertEqual(
            (Path(self.tmp.name) / "frames" / "f1.jpg").read_bytes(),
            (Path(self.tmp.name) / "refs" / item["ref"]).read_bytes(),
        )
        self.assertEqual("pending", item["ref_status"])
        self.assertEqual(item["ref"], item["source_ref"])
        self.assertEqual("frame", item["ref_source"])

    def test_the_item_is_carried_across_scenes(self) -> None:
        """When the person leaves the room, the lantern must not vanish."""
        _, node, _ = self._run({"name": "铜灯笼", "look": "a brass lantern"})
        self.assertIn("铜灯笼", node.carry)

    def test_nothing_picked_up_registers_nothing(self) -> None:
        world, node, got = self._run({"name": "", "look": ""})
        self.assertEqual("", got)
        self.assertEqual({}, world.subjects)
        self.assertEqual("", node.carry)

    def test_an_existing_asset_is_not_registered_twice(self) -> None:
        world, node = _world_with_node()
        world.add_subject("铜灯笼", "old.png", kind="物件")
        before = len(world.subjects)
        got = critic._register_picked_up(world, node, {"name": "铜灯笼", "look": "a brass lantern"})
        self.assertEqual("", got)
        self.assertEqual(before, len(world.subjects))

    def test_no_frame_means_no_item(self) -> None:
        """No frames, nothing "really seen": fall back to the old path instead of forcing a registration."""
        world, node = _world_with_node()
        node.frames = ["缺失的帧.jpg"]
        self.assertEqual(
            "", critic._register_picked_up(world, node, {"name": "铜灯笼", "look": "a brass lantern"})
        )
        self.assertEqual({}, world.subjects)


class ItemDedupTest(unittest.TestCase):
    """The same thing appearing in several clips must not become several items.

    One lantern seen in two clips was registered twice. The critic **did not know what was
    already in the inventory**: every beat it saw the lantern "for the first time" and named it
    differently each time, and deduplication was exact name matching.

    Other narrative games put the inventory in the prompt with "Never re-grant an item already on
    the sheet" and have no code fallback. Here both are done: the same prompt rule, plus a
    normalized fuzzy match that recognizes rewordings.
    """

    def _world_with(self, *names: str) -> World:
        world = World(title="t")
        for n in names:
            world.add_subject(n, "a.png", kind="物件")
        return world

    def test_the_same_thing_under_another_name_is_not_a_new_item(self) -> None:
        world = self._world_with("铜灯笼")
        for said in ("铜灯笼", "黄铜灯笼", "灯笼", "一盏旧铜灯笼"):
            with self.subTest(said=said):
                self.assertTrue(critic._already_have(world, said))

    def test_a_genuinely_different_thing_still_registers(self) -> None:
        world = self._world_with("铜灯笼")
        self.assertFalse(critic._already_have(world, "扳手"))
        self.assertFalse(critic._already_have(world, "绝密档案"))

    def test_places_and_people_also_block_registration(self) -> None:
        """Place and character names must not be registered as items."""
        world = World(title="t")
        world.places["黄铜齿轮大殿"] = {"ref": "r.png", "desc": "", "first_seen": ""}
        world.add_subject("林远山", "p.png", kind="人物")
        self.assertTrue(critic._already_have(world, "黄铜齿轮大殿"))
        self.assertTrue(critic._already_have(world, "林远山"))

    def test_the_inventory_is_actually_sent_to_the_adjudicator(self) -> None:
        """Without seeing the inventory the critic can only guess; that is the root cause of duplicates.

        ⚠️ This asserts the inventory is **actually sent**, not merely in the prompt template. The first version only added
        the rule to system, and the inventory line never reached user; a good template is useless if
        not sent.
        """
        world = self._world_with("铜灯笼", "生锈的扳手")
        node = Node(id="n1", parent=None, video_id="v", clip="c.mp4", place="p", beat="b", mode="continue")
        node.frames = ["f.jpg"]
        world.nodes["n1"] = asdict(node)
        world.head = "n1"
        sent: dict = {}

        def capture(**kw):
            sent.update(kw)
            return {
                "observed": "x",
                "options": ["a", "b", "c"],
                "recommend": 0,
                "objective": "o",
                "picked_up": {"name": "", "look": ""},
            }

        with (
            patch.object(critic, "_frames_for_gm", return_value=[("image/jpeg", b"x")]),
            patch.object(loop.llm, "chat", side_effect=capture),
        ):
            critic.adjudicate(world, node)

        self.assertIn("铜灯笼", sent["user"])
        self.assertIn("生锈的扳手", sent["user"])
        self.assertIn("Items already in the inventory listed below: leave empty", sent["system"])


class ItemLossTest(unittest.TestCase):
    """Items visibly lost in the picture are removed from the inventory.

    The previous version only ever added: the inventory kept growing, and `carry` kept mentioning
    things long gone, which flowed into the video prompt.
    """

    def _world(self) -> tuple[World, Node]:
        world = World(title="t")
        world.add_subject("铜灯笼", "a.png", kind="物件")
        world.add_subject("生锈的扳手", "b.png", kind="物件")
        world.add_subject("林远山", "p.png", kind="人物")
        node = Node(
            id="n1",
            parent=None,
            video_id="v",
            clip="c.mp4",
            place="p",
            beat="b",
            mode="continue",
            carry="手里拿着铜灯笼；腰间别着生锈的扳手",
        )
        return world, node

    def test_a_dropped_item_leaves_the_inventory_and_carry(self) -> None:
        world, node = self._world()
        gone = critic._drop_reported_items(world, node, ["铜灯笼"])
        self.assertEqual(["铜灯笼"], gone)
        self.assertEqual(["生锈的扳手"], critic._held_items(world))
        self.assertNotIn("铜灯笼", node.carry)
        self.assertIn("扳手", node.carry, "没丢的那件不能被顺手抹掉")

    def test_a_renamed_report_still_matches(self) -> None:
        """The critic must copy names verbatim but sometimes does not; normalization catches it."""
        world, node = self._world()
        self.assertEqual(["铜灯笼"], critic._drop_reported_items(world, node, ["灯笼"]))

    def test_something_not_in_the_inventory_is_ignored(self) -> None:
        """Only names from the inventory. A made-up name must not delete anything."""
        world, node = self._world()
        self.assertEqual([], critic._drop_reported_items(world, node, ["从来没有过的东西"]))
        self.assertEqual(2, len(critic._held_items(world)))

    def test_people_are_never_dropped(self) -> None:
        """⚠️ A wrong removal makes that thing and its reference image vanish from the world. People must never be removed by this path."""
        world, node = self._world()
        self.assertEqual([], critic._drop_reported_items(world, node, ["林远山"]))
        self.assertIn("林远山", [v["name"] for v in world.subjects.values()])

    def test_a_non_list_is_ignored(self) -> None:
        world, node = self._world()
        self.assertEqual([], critic._drop_reported_items(world, node, "铜灯笼"))
        self.assertEqual(2, len(critic._held_items(world)))


class AppearedTest(unittest.TestCase):
    """The narration says "a beast arrives" but nothing is in the picture: the critic registers it during playback.

    A real beat:

        reply            a heavily armored mechanical beast bursts out of the steam!
        new_subjects     []                    ← not registered by the director
        six sections     fleeing from an unseen threat   ← so it was not drawn
        picture          someone running from thin air

    **2.5%** of 353 clips were like this. No asset means no reference image, so the six sections
    could only say unseen.

    ⚠️ The late registration is in the critic because **no latency may be added**: the critic runs during playback,
    free time. The cost is that the character appears one beat later, but from then on it exists
    with a locked reference image instead of never.
    """

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        refs = Path(self.tmp.name) / "refs"
        refs.mkdir()
        p = patch.object(loop.config, "REFS", refs)
        p.start()
        self.addCleanup(p.stop)
        self.refs = refs

    def _node(self, reply: str) -> Node:
        return Node(
            id="n1",
            parent=None,
            video_id="v",
            clip="c.mp4",
            place="蒸汽货仓",
            beat="b",
            mode="continue",
            reply=reply,
        )

    def test_a_narrated_newcomer_becomes_an_asset_with_a_reference(self) -> None:
        world = World(title="t", visual_style="retro steampunk")
        node = self._node("一头机械巨兽从蒸汽中轰然冲出！")
        with patch.object(critic, "_timed_draw", return_value=(b"\x89PNG\r\n\x1a\nfake", 3.2)):
            got = critic._register_appeared(
                world, node, {"name": "机械巨兽", "look": "a hulking riveted brass war machine"}
            )
        self.assertEqual("机械巨兽", got)
        rec = next(v for v in world.subjects.values() if v["name"] == "机械巨兽")
        self.assertEqual("a hulking riveted brass war machine", rec["definition"])
        self.assertTrue((self.refs / rec["ref"]).exists(), "必须有参考图，否则下一拍照样画不出来")
        self.assertEqual("n1", rec["from_node"])

    def test_someone_already_known_is_not_registered_twice(self) -> None:
        world = World(title="t")
        world.add_subject("机械巨兽", "old.png")
        with patch.object(critic, "_timed_draw") as draw:
            self.assertEqual(
                "",
                critic._register_appeared(
                    world, self._node("巨兽又冲了过来"), {"name": "机械巨兽", "look": "x"}
                ),
            )
        draw.assert_not_called()
        self.assertEqual(1, len(world.subjects))

    def test_nothing_appeared_registers_nothing(self) -> None:
        world = World(title="t")
        with patch.object(critic, "_timed_draw") as draw:
            self.assertEqual(
                "", critic._register_appeared(world, self._node("你继续往前走。"), {"name": "", "look": ""})
            )
        draw.assert_not_called()
        self.assertEqual({}, world.subjects)

    def test_a_failed_draw_never_breaks_the_turn(self) -> None:
        """Late registration is a free bonus; a failure must not affect anything."""
        world = World(title="t")
        with patch.object(critic, "_timed_draw", side_effect=RuntimeError("boom")):
            self.assertEqual(
                "", critic._register_appeared(world, self._node("巨兽冲出"), {"name": "巨兽", "look": "x"})
            )
        self.assertEqual({}, world.subjects)

    def test_the_adjudicator_is_shown_the_narration_and_known_cast(self) -> None:
        """The criterion is the **narration**, not the picture; without the narration there is nothing to judge."""
        world = World(title="t")
        world.add_subject("林默", "a.png")
        node = self._node("一头机械巨兽从蒸汽中轰然冲出！")
        node.frames = ["f.jpg"]
        world.nodes["n1"] = asdict(node)
        world.head = "n1"
        sent: dict = {}

        def capture(**kw):
            sent.update(kw)
            return {**GOOD, "appeared": {"name": "", "look": ""}}

        with (
            patch.object(critic, "_frames_for_gm", return_value=[("image/jpeg", b"x")]),
            patch.object(loop.llm, "chat", side_effect=capture),
        ):
            critic.adjudicate(world, node)
        self.assertIn("机械巨兽", sent["user"], "旁白必须发给裁判")
        self.assertIn("Narration of this beat", sent["user"])
        self.assertIn("林默", sent["user"])


if __name__ == "__main__":
    unittest.main()
