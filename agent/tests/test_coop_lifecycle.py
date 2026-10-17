"""Destructive co-op operations, with real isolated files and no model calls."""

import asyncio
import fcntl
import unittest
from unittest.mock import patch

from murea_agent import config
from app.api.sse import mutation_slot
from murea_agent.coop import room as coop, events as coop_events
from murea_agent import reqlog
from murea_agent.memory.world import World
from tests import test_coop as coop_tests


class CoopLifecycleTest(unittest.TestCase):
    create = coop_tests.CoopTest.create
    local_room = coop_tests.CoopTest.local_room
    local_role = coop_tests.CoopTest.local_role
    role = coop_tests.CoopTest.role

    def setUp(self):
        coop_tests.CoopTest.setUp(self)
        frames = self.root / "frames"
        frames.mkdir()
        p = patch.object(config, "FRAMES", frames)
        p.start()
        self.patches.append(p)

    def tearDown(self):
        coop_tests.CoopTest.tearDown(self)

    def test_delete_local_world_cleans_media_previews_history_and_logs(self):
        room = self.local_room()
        for uid, name in [(m["user_id"], f"角色{i}") for i, m in enumerate(room["members"])]:
            room = self.local_role(uid, name)
        world = World.load(room["world_id"])
        world.nodes["n-delete"] = {"refs": ["node.jpg"], "clip": "shot.mp4", "frames": ["frame.jpg"]}
        world.save()
        with coop._edit(self.code) as stored:
            stored["asset_previews"] = {"scene": {"status": "ready", "ref": "preview.jpg"}}
            stored["members"][0]["portrait"] = {"status": "ready", "ref": "portrait.jpg"}
            stored["rounds"] = [{"inputs": [{"ref": "historical.jpg"}]}]
            stored["preview_clip"] = "preview.mp4"
        for name in ("node.jpg", "preview.jpg", "portrait.jpg", "historical.jpg"):
            (config.REFS / name).write_bytes(b"image")
        for name in ("shot.mp4", "preview.mp4"):
            (config.CLIPS / name).write_bytes(b"video")
        (config.FRAMES / "frame.jpg").write_bytes(b"frame")
        token = reqlog.bind(world.id)
        try:
            reqlog.record("test")
        finally:
            reqlog.unbind(token)
        response = self.a.delete(self.url)
        self.assertEqual(200, response.status_code, response.text)
        self.assertEqual(9, response.json()["removed_files"])
        self.assertFalse(world.path.exists())
        self.assertFalse(reqlog.path(world.id).exists())
        self.assertFalse((config.DATA / "coop" / f"{self.code}.json").exists())
        for root in (config.REFS, config.CLIPS, config.FRAMES):
            self.assertEqual([], list(root.iterdir()))
        self.assertEqual([], self.a.get("/api/coop/rooms").json())
        self.assertEqual([], self.a.get("/api/worlds").json())
        self.assertEqual(404, self.a.get(self.url).status_code)
        self.assertEqual(404, self.a.post(self.url + "/join").status_code)
        self.assertEqual(404, self.a.delete(self.url).status_code)

    def test_delete_preserves_media_used_by_other_worlds_and_room_previews(self):
        room = self.local_room()
        url, code = self.url, self.code
        world = World.load(room["world_id"])
        world.add_subject("共享图", "shared.jpg")
        world.save()
        other = World(owner_id=self.ub["id"], title="其他世界")
        other.add_subject("保留图", "shared.jpg")
        other.save()
        with coop._edit(code) as stored:
            stored["asset_previews"] = {"scene": {"status": "ready", "ref": "shared-preview.jpg"}}
        second = self.local_room()
        with coop._edit(second["code"]) as stored:
            stored["asset_previews"] = {"scene": {"status": "ready", "ref": "shared-preview.jpg"}}
        for name in ("shared.jpg", "shared-preview.jpg"):
            (config.REFS / name).write_bytes(b"keep")
        response = self.a.delete(url)
        self.assertEqual(200, response.status_code, response.text)
        self.assertEqual(0, response.json()["removed_files"])
        self.assertTrue(other.path.exists())
        self.assertEqual(2, len(list(config.REFS.iterdir())))
        self.assertEqual(200, self.a.get(self.url).status_code)

    def test_only_current_host_can_delete_online_room(self):
        room = self.create()
        self.b.post(self.url + "/join")
        path = config.DATA / "coop" / f"{self.code}.json"
        original = path.read_bytes()
        for client, status in ((self.anon, 401), (self.b, 403), (self.c, 403)):
            self.assertEqual(status, client.delete(self.url).status_code)
            self.assertEqual(original, path.read_bytes())
            self.assertTrue(World.load(room["world_id"]).path.exists())
        self.assertEqual(self.ua["id"], self.a.get("/api/coop/rooms").json()[0]["host_id"])
        self.a.post(self.url + "/leave")
        self.assertEqual(403, self.a.delete(self.url).status_code)
        self.assertEqual(200, self.b.delete(self.url).status_code)

    def test_local_world_cannot_be_deleted_by_another_user(self):
        self.local_room()
        self.assertEqual(403, self.b.delete(self.url).status_code)
        self.assertEqual(200, self.a.get(self.url).status_code)

    def test_all_generation_locks_block_deletion_without_changing_data(self):
        room = self.local_room()
        path = config.DATA / "coop" / f"{self.code}.json"
        original = path.read_bytes()
        names = [
            f"{self.code}.{kind}.lock" for kind in ("run", "intro", "ai", "recommend", "observation", "story")
        ]
        names.append("portrait-render.lock")
        for name in names:
            with self.subTest(name=name), (coop._directory() / name).open("a") as guard:
                fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self.assertEqual(409, self.a.delete(self.url).status_code)
                self.assertEqual(original, path.read_bytes())
                self.assertTrue(World.load(room["world_id"]).path.exists())
        with mutation_slot(room["world_id"]):
            self.assertEqual(409, self.a.delete(self.url).status_code)
        self.assertEqual(200, self.a.delete(self.url).status_code)

    def test_abandoned_generation_after_restart_does_not_prevent_deletion(self):
        self.local_room()
        with coop._edit(self.code) as stored:
            stored.update(status="generating", ai_status="thinking", intro={"status": "generating"})
            stored["members"][0]["portrait"] = {"status": "generating"}
        self.assertEqual(200, self.a.delete(self.url).status_code)

    def test_missing_world_session_can_still_be_removed_from_lobby(self):
        room = self.local_room()
        World.load(room["world_id"]).path.unlink()
        self.assertEqual(200, self.a.delete(self.url).status_code)
        self.assertEqual([], self.a.get("/api/coop/rooms").json())

    def test_deletion_revokes_connected_teammates(self):
        self.create()
        self.b.post(self.url + "/join")

        async def check():
            stream = coop_events.stream(self.code, self.ub["id"])
            self.assertIn("event: room", await anext(stream))
            self.assertEqual(200, (await asyncio.to_thread(self.a.delete, self.url)).status_code)
            self.assertIn("event: revoked", await asyncio.wait_for(anext(stream), timeout=2))
            await stream.aclose()

        asyncio.run(check())

    def test_concurrent_deletion_has_one_success_and_does_not_recreate_room(self):
        self.local_room()

        async def check():
            responses = await asyncio.gather(*[asyncio.to_thread(self.a.delete, self.url) for _ in range(2)])
            self.assertEqual([200, 404], sorted(r.status_code for r in responses))

        asyncio.run(check())
        self.assertEqual([], self.a.get("/api/coop/rooms").json())


if __name__ == "__main__":
    unittest.main()
