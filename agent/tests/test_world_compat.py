#!/usr/bin/env python3
"""Regression checks for loading worlds across schema revisions.

These tests use an isolated temporary data directory.  They deliberately model
the session written by the spatial-map prototype (``place_links`` and no
``visual_style``) alongside a current session, then exercise the same list,
load, and delete routes used by the UI.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient

from murea_agent import config
from app.api import create_app
from murea_agent.memory.world import World


class WorldCompatibilityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.original_paths = {
            "DATA": config.DATA,
            "REFS": config.REFS,
            "CLIPS": config.CLIPS,
            "SESSIONS": config.SESSIONS,
        }
        config.DATA = root / "data"
        config.REFS = config.DATA / "refs"
        config.CLIPS = config.DATA / "clips"
        config.SESSIONS = config.DATA / "sessions"
        for directory in (config.REFS, config.CLIPS, config.SESSIONS):
            directory.mkdir(parents=True)
        self.client = TestClient(create_app(), raise_server_exceptions=False)

    def tearDown(self) -> None:
        self.client.close()
        for name, value in self.original_paths.items():
            setattr(config, name, value)
        self.tmp.cleanup()

    @staticmethod
    def _world_data(
        world_id: str,
        title: str,
        *,
        visual_style: str = "",
        place_links: list[dict[str, str]] | None = None,
        ref: str | None = None,
    ) -> dict:
        world = World(
            id=world_id,
            title=title,
            setting="现代写字楼",
            visual_style=visual_style,
            places={
                "会议室": {
                    "ref": ref or f"{world_id}.png",
                    "desc": "一间会议室",
                    "first_seen": "2026-08-29T00:00:00",
                },
            },
            place_links=place_links or [],
        )
        return asdict(world)

    def test_loads_spatial_map_world_without_visual_style(self) -> None:
        legacy = self._world_data(
            "wlegacy",
            "旧世界",
            place_links=[
                {
                    "from": "会议室",
                    "to": "走廊",
                    "type": "corridor",
                    "source": "inferred",
                }
            ],
        )
        # This is the shape produced before Style Bible was introduced.  The
        # extra key models an additive field from another checkout as well.
        legacy.pop("visual_style")
        legacy["future_extension"] = {"kept_out_of_current_model": True}
        (config.SESSIONS / "wlegacy.json").write_text(
            json.dumps(legacy, ensure_ascii=False),
            encoding="utf-8",
        )

        loaded = World.load("wlegacy")
        self.assertEqual("", loaded.visual_style)
        self.assertEqual(legacy["place_links"], loaded.place_links)
        self.assertFalse(hasattr(loaded, "future_extension"))

        # A normal save after loading must not erase the legacy topology.
        loaded.save()
        saved = json.loads((config.SESSIONS / "wlegacy.json").read_text())
        self.assertEqual(legacy["place_links"], saved["place_links"])
        self.assertEqual("", saved["visual_style"])

    def test_old_and_current_worlds_list_load_and_delete(self) -> None:
        legacy = self._world_data(
            "wlegacy",
            "旧世界",
            place_links=[{"from": "会议室", "to": "走廊"}],
            ref="shared.png",
        )
        legacy.pop("visual_style")
        (config.SESSIONS / "wlegacy.json").write_text(
            json.dumps(legacy, ensure_ascii=False),
            encoding="utf-8",
        )
        current = self._world_data(
            "wcurrent",
            "新世界",
            visual_style="2D storybook",
            ref="shared.png",
        )
        (config.SESSIONS / "wcurrent.json").write_text(
            json.dumps(current, ensure_ascii=False),
            encoding="utf-8",
        )
        (config.REFS / "shared.png").write_bytes(b"shared")

        listed = self.client.get("/api/worlds")
        self.assertEqual(200, listed.status_code, listed.text)
        self.assertEqual({"wlegacy", "wcurrent"}, {w["id"] for w in listed.json()})
        legacy_response = self.client.get("/api/worlds/wlegacy")
        self.assertEqual(200, legacy_response.status_code, legacy_response.text)
        self.assertEqual(legacy["place_links"], legacy_response.json()["place_links"])
        current_response = self.client.get("/api/worlds/wcurrent")
        self.assertEqual(200, current_response.status_code, current_response.text)
        self.assertEqual("2D storybook", current_response.json()["visual_style"])

        # Deleting the current world while the legacy world is still present
        # exercises the sibling scan that previously crashed on place_links.
        deleted = self.client.delete("/api/worlds/wcurrent")
        self.assertEqual(200, deleted.status_code, deleted.text)
        self.assertFalse((config.SESSIONS / "wcurrent.json").exists())
        self.assertTrue((config.SESSIONS / "wlegacy.json").exists())
        self.assertEqual(0, deleted.json()["removed_files"])
        self.assertTrue((config.REFS / "shared.png").exists())

        # Once the sibling is gone, the legacy world can be deleted and its
        # formerly shared media becomes removable.
        deleted = self.client.delete("/api/worlds/wlegacy")
        self.assertEqual(200, deleted.status_code, deleted.text)
        self.assertFalse((config.SESSIONS / "wlegacy.json").exists())
        self.assertEqual(1, deleted.json()["removed_files"])
        self.assertFalse((config.REFS / "shared.png").exists())


if __name__ == "__main__":
    unittest.main()
