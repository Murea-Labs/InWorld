"""A short world idea stays grounded and never falls back to unrelated dice content."""

import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.api import create_app
from murea_agent.director import worldgen


class WorldIdeaTest(unittest.TestCase):
    def setUp(self):
        self.idea = "去冰塔峰之龙的世界探险"
        self.concept = {
            "title": "冰塔峰",
            "setting": "冰塔峰常年积雪，山腰坐落着冒险者的驿站。",
            "mainline": "寻找冰塔峰巨龙并查清雪山异变的缘由。",
            "opening": "你站在雪山驿站门前，仰望云雾中的冰塔峰。",
            "style_id": "fantasy",
            "assets": [
                {"kind": "player", "name": "旅人", "prompt": "A traveller in a blue winter coat."},
                {
                    "kind": "place",
                    "name": "雪山驿站",
                    "prompt": "A stone lodge beside a snowy mountain path.",
                },
            ],
        }
        self.styles = [{"id": "fantasy", "label": "Fantasy"}]

    def test_idea_drives_planning_without_random_ingredients(self):
        with patch.object(worldgen.llm, "chat", return_value=self.concept) as chat:
            result = worldgen.spin(self.styles, self.idea)
        self.assertEqual(self.concept, result)
        prompt = chat.call_args.kwargs
        self.assertIn(self.idea, prompt["user"])
        self.assertNotIn("Something is wrong:", prompt["user"])
        self.assertIn("`setting`: ONE sentence.", prompt["system"])
        self.assertNotIn("Two or three concrete sentences", prompt["system"])
        self.assertIn("assets", prompt["schema"]["required"])
        self.assertEqual(
            ["player", "place"],
            prompt["schema"]["properties"]["assets"]["items"]["properties"]["kind"]["enum"],
        )

    def test_setting_uses_the_same_budget_as_the_original_entry(self):
        concept = {**self.concept, "setting": "冰雪山峰" * 100}
        with patch.object(worldgen.llm, "chat", return_value=concept):
            from_idea = worldgen.spin(self.styles, self.idea)
            from_dice = worldgen.spin(self.styles)
        self.assertEqual(concept["setting"][:200], from_idea["setting"])
        self.assertEqual(from_dice["setting"], from_idea["setting"])

    def test_assets_must_be_ready_to_plan_before_world_creation(self):
        extra = {"kind": "creature", "name": "冰龙", "prompt": "A frost dragon."}
        for assets in (
            None,
            [],
            [self.concept["assets"][0]],
            [self.concept["assets"][0]] * 2,
            [self.concept["assets"][0], extra],
            [*self.concept["assets"], extra],
        ):
            with self.subTest(assets=assets):
                with patch.object(worldgen.llm, "chat", return_value={**self.concept, "assets": assets}):
                    with self.assertRaises(ValueError):
                        worldgen.spin(self.styles, self.idea)

    def test_dice_does_not_require_an_opening_asset_plan(self):
        concept = {key: value for key, value in self.concept.items() if key != "assets"}
        with patch.object(worldgen.llm, "chat", return_value=concept):
            self.assertEqual(concept, worldgen.spin(self.styles))

    def test_model_failure_does_not_create_an_unrelated_world(self):
        with patch.object(worldgen.llm, "chat", side_effect=RuntimeError("offline")):
            with self.assertRaisesRegex(RuntimeError, "offline"):
                worldgen.spin(self.styles, self.idea)
            self.assertTrue(worldgen.spin(self.styles)["opening"])

    def test_incomplete_concept_is_not_silently_patched_from_dice(self):
        with patch.object(worldgen.llm, "chat", return_value={"title": "冰塔峰"}):
            with self.assertRaisesRegex(ValueError, "incomplete"):
                worldgen.spin(self.styles, self.idea)

    def test_route_forwards_idea_and_reports_errors(self):
        with TestClient(create_app()) as client:
            with patch.object(worldgen.llm, "chat", return_value=self.concept) as chat:
                result = client.post("/api/worlds/random", json={"idea": self.idea, "styles": self.styles})
                self.assertEqual(200, result.status_code)
                self.assertEqual(self.concept, result.json())
                self.assertIn(self.idea, chat.call_args.kwargs["user"])
            with patch.object(worldgen.llm, "chat", side_effect=RuntimeError("offline")):
                self.assertEqual(502, client.post("/api/worlds/random", json={"idea": self.idea}).status_code)
            for idea in ({"bad": "value"}, "x" * 2001):
                self.assertEqual(400, client.post("/api/worlds/random", json={"idea": idea}).status_code)


if __name__ == "__main__":
    unittest.main()
