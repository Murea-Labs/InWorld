"""The demo world: imported once for a user without worlds, playable without keys.

No demo archive is bundled at the moment, so these tests build a small one.
"""

import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from murea_agent import config
from murea_agent.memory.world import World
from app import demo


def _node(node_id: str, parent: str | None, clip: str) -> dict:
    return {
        "id": node_id,
        "parent": parent,
        "video_id": node_id,
        "clip": clip,
        "place": "Harbour",
        "beat": f"Beat {node_id}",
        "mode": "continue" if parent else "cut",
    }


def _archive(path: Path) -> None:
    world = {
        "id": "wdemo",
        "title": "Demo",
        "nodes": {"n1": _node("n1", None, "c1.mp4"), "n2": _node("n2", "n1", "c2.mp4")},
        "head": "n2",
        "here": "Harbour",
    }
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("world.json", json.dumps(world))
        z.writestr("clips/c1.mp4", b"clip one")
        z.writestr("clips/c2.mp4", b"clip two")


class DemoWorldTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        archive = self.tmp / "demo.zip"
        _archive(archive)
        self.patches = [
            patch.object(config, "DATA", self.tmp),
            patch.object(config, "SESSIONS", self.tmp / "sessions"),
            patch.object(config, "REFS", self.tmp / "refs"),
            patch.object(config, "CLIPS", self.tmp / "clips"),
            patch.object(config, "FRAMES", self.tmp / "frames"),
            patch.object(demo, "DEMO", archive),
        ]
        for p in self.patches:
            p.start()
        for d in ("sessions", "refs", "clips", "frames"):
            (self.tmp / d).mkdir()

    def tearDown(self):
        for p in self.patches:
            p.stop()

    def test_missing_archive_imports_nothing(self):
        with patch.object(demo, "DEMO", self.tmp / "missing.zip"):
            self.assertFalse(demo.seed_demo("u0", has_worlds=False))
        self.assertEqual([], World.list_all(owner_id="u0"))

    def test_imported_once_with_all_media(self):
        self.assertTrue(demo.seed_demo("u1", has_worlds=False))
        worlds = World.list_all(owner_id="u1")
        self.assertEqual(1, len(worlds))
        w = World.load(worlds[0]["id"])
        self.assertEqual("u1", w.owner_id)
        self.assertGreaterEqual(len(w.lineage()), 2)
        for node in w.lineage():
            self.assertTrue((self.tmp / "clips" / node.clip).exists(), node.clip)
        # Deleting it later must not bring it back.
        self.assertFalse(demo.seed_demo("u1", has_worlds=False))

    def test_not_imported_for_users_who_already_have_worlds(self):
        self.assertFalse(demo.seed_demo("u2", has_worlds=True))
        self.assertEqual([], World.list_all(owner_id="u2"))


if __name__ == "__main__":
    unittest.main()
