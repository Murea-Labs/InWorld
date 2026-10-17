"""Pure logic tests for @ asset binding (no LLM or video calls)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from murea_agent.shot import beat_assets, video_prompt
from murea_agent.memory.world import World


class AssetMentionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.world = World(id="wmentions", title="mentions", setting="test")
        self.world.add_subject("小明", "xiaoming.jpg", kind="人物")
        self.world.add_subject("钥匙", "key.jpg", kind="物件")
        self.world.add_subject("Alice Smith", "alice.jpg", kind="人物")
        self.world.add_subject("保安", "guard.jpg", kind="人物")
        self.world.add_subject("记者", "reporter.jpg", kind="人物")
        self.world.add_place("咖啡馆", "cafe.jpg")
        self.world.add_place("屋顶", "roof.jpg")

    def test_explicit_ids_and_manual_names_resolve_to_stable_ids(self) -> None:
        explicit = beat_assets._asset_mentions(self.world, "让@小明拿起@钥匙", ["S1", "S2"])
        self.assertEqual(["S1", "S2"], [item["id"] for item in explicit])

        manual = beat_assets._asset_mentions(self.world, "让@小明走向@咖啡馆", None)
        self.assertEqual(["S1", "咖啡馆"], [item["id"] for item in manual])

    def test_email_is_not_treated_as_a_mention(self) -> None:
        self.assertEqual([], beat_assets._asset_mentions(self.world, "联系 foo@小明.com", None))

    def test_mentions_force_subjects_and_scene_but_keep_gm_extras(self) -> None:
        directive = {"subjects": ["S3"], "place": "屋顶", "main_scene": "屋顶"}
        required = beat_assets._asset_mentions(self.world, "", ["S1", "咖啡馆"])
        subject_ids = beat_assets._apply_asset_mentions(self.world, directive, required)
        self.assertEqual(["S1"], subject_ids)
        self.assertEqual(["S1", "S3"], directive["subjects"])
        self.assertEqual("咖啡馆", directive["place"])
        self.assertEqual("咖啡馆", directive["asset_mentions"][1]["id"])

    def test_only_one_scene_can_be_bound_per_shot(self) -> None:
        with self.assertRaisesRegex(ValueError, "at most one place"):
            beat_assets._asset_mentions(self.world, "", ["咖啡馆", "屋顶"])

    def test_required_subject_is_never_silently_dropped_at_reference_limit(self) -> None:
        subject_ids = list(self.world.subjects)
        with self.assertRaisesRegex(ValueError, "More than 4"):
            beat_assets._pictures_and_refs(
                self.world,
                "咖啡馆",
                subject_ids,
                required_subject_ids=set(subject_ids),
                motion_plan={"target": "钥匙", "relation_change": "possession"},
            )

    def test_required_picture_tokens_are_locked_into_prompt(self) -> None:
        pictures = [
            {"n": 1, "kind": "character", "id": "S1"},
            {"n": 2, "kind": "scene", "id": "咖啡馆"},
        ]
        out = video_prompt._lock_required_ref_assets(
            {"detailed_description": "[Shot 1] A quiet room."},
            pictures,
            {"S1", "咖啡馆"},
        )
        self.assertIn("<Subject 1>", out["detailed_description"])
        self.assertIn("<Subject 2>", out["detailed_description"])


if __name__ == "__main__":
    unittest.main()
