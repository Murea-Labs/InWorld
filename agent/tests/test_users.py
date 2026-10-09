#!/usr/bin/env python3
"""Route-level checks for lightweight users and world ownership."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
import zipfile
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from murea_agent import config
from app.api import create_app
from app.users import DEFAULT_USER_ID
from murea_agent.memory.world import World


class UserIsolationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.original_paths = {
            name: getattr(config, name) for name in ("DATA", "REFS", "CLIPS", "SESSIONS", "FRAMES", "VOICES")
        }
        config.DATA = root / "data"
        config.REFS = config.DATA / "refs"
        config.CLIPS = config.DATA / "clips"
        config.SESSIONS = config.DATA / "sessions"
        config.FRAMES = config.DATA / "frames"
        config.VOICES = config.DATA / "voices"
        for directory in (
            config.DATA,
            config.REFS,
            config.CLIPS,
            config.SESSIONS,
            config.FRAMES,
            config.VOICES,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        self.client = TestClient(create_app(), raise_server_exceptions=False)

    def tearDown(self) -> None:
        self.client.close()
        for name, value in self.original_paths.items():
            setattr(config, name, value)
        self.tmp.cleanup()

    def _create_user(self, name: str) -> dict:
        response = self.client.post("/api/users", json={"name": name})
        self.assertEqual(200, response.status_code, response.text)
        return response.json()

    def _create_world(self, title: str) -> dict:
        response = self.client.post("/api/worlds", json={"title": title})
        self.assertEqual(200, response.status_code, response.text)
        return response.json()

    def test_select_register_and_name_uniqueness(self) -> None:
        self.assertEqual(401, self.client.get("/api/users/me").status_code)
        user = self._create_user("  小 明  ")
        self.assertEqual("小 明", user["name"])
        self.assertEqual(user["id"], self.client.get("/api/users/me").json()["id"])

        duplicate = self.client.post("/api/users", json={"name": "小   明"})
        self.assertEqual(409, duplicate.status_code, duplicate.text)
        selected = self.client.post(f"/api/users/{DEFAULT_USER_ID}/select")
        self.assertEqual(200, selected.status_code, selected.text)
        self.assertEqual(DEFAULT_USER_ID, self.client.get("/api/users/me").json()["id"])

    def test_user_can_be_deleted_and_locked_user_requires_password(self) -> None:
        created = self.client.post(
            "/api/users",
            json={"name": "Locked", "password": "secret1"},
        ).json()
        denied = self.client.request(
            "DELETE",
            f"/api/users/{created['id']}",
            json={"password": "wrong1"},
        )
        self.assertEqual(401, denied.status_code, denied.text)
        deleted = self.client.request(
            "DELETE",
            f"/api/users/{created['id']}",
            json={"password": "secret1"},
        )
        self.assertEqual(200, deleted.status_code, deleted.text)
        self.assertNotIn(created["id"], [user["id"] for user in self.client.get("/api/users").json()])
        self.assertEqual(401, self.client.get("/api/users/me").status_code)
        default = self.client.delete(f"/api/users/{DEFAULT_USER_ID}")
        self.assertEqual(409, default.status_code, default.text)

    def test_deleting_a_user_deletes_their_worlds(self) -> None:
        user = self._create_user("Busy user")
        world = self._create_world("仍在使用的世界")
        deleted = self.client.delete(f"/api/users/{user['id']}")
        self.assertEqual(200, deleted.status_code, deleted.text)
        self.assertFalse((config.SESSIONS / f"{world['id']}.json").exists())

        self.assertNotIn(user["id"], [item["id"] for item in self.client.get("/api/users").json()])

    def test_worlds_are_listed_and_mutated_only_by_owner(self) -> None:
        user_a = self._create_user("Alice")
        world_a = self._create_world("Alice 的世界")
        self.assertEqual(user_a["id"], world_a["owner_id"])

        self._create_user("Bob")
        world_b = self._create_world("Bob 的世界")
        listed = self.client.get("/api/worlds")
        self.assertEqual([world_b["id"]], [world["id"] for world in listed.json()])

        self.assertEqual(404, self.client.get(f"/api/worlds/{world_a['id']}").status_code)
        self.assertEqual(
            404,
            self.client.patch(
                f"/api/worlds/{world_a['id']}",
                json={"title": "偷改"},
            ).status_code,
        )
        self.assertEqual(
            404,
            self.client.post(
                f"/api/worlds/{world_a['id']}/turn",
                json={"text": "继续"},
            ).status_code,
        )
        self.assertEqual(404, self.client.delete(f"/api/worlds/{world_a['id']}").status_code)
        self.assertTrue((config.SESSIONS / f"{world_a['id']}.json").exists())

        self.client.post(f"/api/users/{user_a['id']}/select")
        self.assertEqual([world_a["id"]], [world["id"] for world in self.client.get("/api/worlds").json()])

    def test_import_is_reowned_by_current_user(self) -> None:
        source = World(owner_id="someone-else", title="外部世界")
        payload = io.BytesIO()
        with zipfile.ZipFile(payload, "w") as archive:
            archive.writestr("world.json", json.dumps(asdict(source), ensure_ascii=False))

        user = self._create_user("Importer")
        response = self.client.post(
            "/api/worlds/import",
            files={"file": ("world.zip", payload.getvalue(), "application/zip")},
        )
        self.assertEqual(200, response.status_code, response.text)
        imported = World.load(response.json()["id"])
        self.assertEqual(user["id"], imported.owner_id)

    def _import(self, files: dict[str, bytes], world: dict | None = None):
        payload = io.BytesIO()
        with zipfile.ZipFile(payload, "w") as archive:
            archive.writestr(
                "world.json", json.dumps(world or asdict(World(title="外部世界")), ensure_ascii=False)
            )
            for name, raw in files.items():
                archive.writestr(name, raw)
        return self.client.post(
            "/api/worlds/import", files={"file": ("world.zip", payload.getvalue(), "application/zip")}
        )

    def test_import_writes_only_media_and_refs_are_served_as_images(self) -> None:
        # refs/poc.html from an archive was served as text/html on the app's own origin.
        png = b"\x89PNG\r\n\x1a\n" + b"0" * 32
        response = self._import(
            {
                "refs/poc.html": b"<script>alert(1)</script>",
                "refs/ok.png": png,
                "clips/c.mp4": b"mp4",
                "notes.txt": b"x",
            }
        )
        self.assertEqual(200, response.status_code, response.text)
        self.assertEqual(2, response.json()["files_written"])
        self.assertFalse((config.REFS / "poc.html").exists())
        (config.REFS / "planted.html").write_text("<script>alert(1)</script>")
        self.assertEqual(404, self.client.get("/refs/planted.html").status_code)
        served = self.client.get("/refs/ok.png")
        self.assertEqual(
            ("image/png", "nosniff"),
            (served.headers["content-type"], served.headers["x-content-type-options"]),
        )

    def test_import_accepts_fields_from_a_newer_version(self) -> None:
        world = {**asdict(World(title="新版本")), "field_from_the_future": 1}
        self.assertEqual(200, self._import({}, world).status_code)

    def test_import_refuses_an_archive_that_unpacks_too_large(self) -> None:
        from app.api.routes import transfer

        with patch.object(transfer, "MAX_UNPACKED", 10):
            response = self._import({"refs/big.png": b"0" * 100})
        self.assertEqual(400, response.status_code)
        self.assertFalse((config.REFS / "big.png").exists())
        self.assertEqual([], list(config.SESSIONS.glob("*.json")))

    def test_import_refuses_media_names_that_are_not_plain_file_names(self) -> None:
        bad_names = ["../outside.png", "sub/inner.png", "..\\outside.png", ".hidden.png", "page.html", ".."]

        def world_with(where: str, name: str) -> dict:
            world = asdict(World(title="外部世界"))
            if where == "subject.ref":
                world["subjects"] = {"s1": {"name": "Mira", "ref": name}}
            elif where == "subject.source_ref":
                world["subjects"] = {"s1": {"name": "Mira", "ref": "ok.png", "source_ref": name}}
            elif where == "place.ref":
                world["places"] = {"Harbour": {"ref": name}}
            else:
                key = where.split(".")[1]
                value = name if key == "clip" else [name]
                world["nodes"] = {"n1": {"id": "n1", key: value}}
            return world

        for where in (
            "subject.ref",
            "subject.source_ref",
            "place.ref",
            "node.clip",
            "node.refs",
            "node.frames",
        ):
            for name in bad_names:
                with self.subTest(where=where, name=name):
                    response = self._import({}, world_with(where, name))
                    self.assertEqual(400, response.status_code, response.text)
                    self.assertIn("file name", response.json()["detail"])
        self.assertEqual([], list(config.SESSIONS.glob("*.json")))

        ok = world_with("node.frames", "n1_0.jpg")
        ok["subjects"] = {"s1": {"name": "Mira", "ref": "mira.png", "source_ref": "mira-frame.jpg"}}
        ok["places"] = {"Harbour": {"ref": "harbour.webp"}}
        self.assertEqual(200, self._import({}, ok).status_code)

    def test_export_packs_only_plain_media_names_inside_the_media_directories(self) -> None:
        # Worlds already on disk are not re-checked by import, so export filters names too.
        (config.REFS / "mira.png").write_bytes(b"png")
        (config.DATA / "outside.png").write_bytes(b"not media")
        (config.REFS / "linked.png").symlink_to(config.DATA / "outside.png")
        (config.CLIPS / "c1.mp4").write_bytes(b"mp4")
        world = World(title="A")
        world.add_subject("Mira", "mira.png")
        world.add_subject("Linked", "linked.png")
        world.add_subject("Outside", "../outside.png")
        world.nodes["n1"] = {
            "id": "n1",
            "clip": "c1.mp4",
            "refs": ["../outside.png"],
            "frames": ["../../x.jpg"],
        }
        world.save()

        response = self.client.get(f"/api/worlds/{world.id}/export")
        self.assertEqual(200, response.status_code, response.text)
        names = zipfile.ZipFile(io.BytesIO(response.content)).namelist()
        self.assertEqual(["clips/c1.mp4", "refs/mira.png", "world.json"], sorted(names))

    def test_media_names_are_checked_before_any_path_join(self) -> None:
        from app.api.routes.media import IMAGE_TYPES
        from murea_agent.memory import world_media

        self.assertEqual(set(IMAGE_TYPES), world_media.MEDIA_TYPES["refs"])
        self.assertEqual("a.png", world_media.media_name("refs", "a.png"))
        for name in ("", ".", "..", "../a.png", "a/b.png", "a\\b.png", ".a.png", "a.mp4", None, 3, ["a.png"]):
            with self.subTest(name=name):
                self.assertIsNone(world_media.media_name("refs", name))
                with self.assertRaises(ValueError):
                    world_media.media_path("refs", name)
        self.assertEqual(config.CLIPS / "c.mp4", world_media.media_path("clips", "c.mp4"))
        self.assertFalse(world_media.has_media("refs", "../users.json"))
        room = {
            "members": [{"ref": "../users.json"}, {"ref": "p.jpg"}],
            "clip": "../x.mp4",
            "frames": ["/f.jpg"],
        }
        self.assertEqual({"refs/p.jpg"}, world_media.room_media(room))

    def test_deleting_a_world_keeps_media_while_another_world_is_unreadable(self) -> None:
        from murea_agent.memory.world_media import delete_worlds

        (config.REFS / "shared.png").write_bytes(b"png")
        world = World(title="A")
        world.add_subject("Mira", "shared.png")
        world.save()
        (config.SESSIONS / "wbroken.json").write_text('{"half": ')  # mid-save or corrupt
        self.assertEqual(0, delete_worlds([world.id]))
        self.assertTrue((config.REFS / "shared.png").exists())
        self.assertFalse(world.path.exists())

    def test_deleting_a_world_keeps_media_a_coop_room_still_uses(self) -> None:
        from murea_agent.memory.world_media import delete_worlds

        (config.REFS / "portrait.jpg").write_bytes(b"jpg")
        world = World(title="A")
        world.add_subject("Mira", "portrait.jpg")
        world.save()
        (config.DATA / "coop").mkdir(exist_ok=True)
        (config.DATA / "coop" / "ROOM.json").write_text(json.dumps({"members": [{"ref": "portrait.jpg"}]}))
        delete_worlds([world.id])
        self.assertTrue((config.REFS / "portrait.jpg").exists())

    def test_legacy_world_belongs_to_default_user(self) -> None:
        legacy = asdict(World(id="wlegacy", title="旧世界"))
        legacy.pop("owner_id")
        (config.SESSIONS / "wlegacy.json").write_text(json.dumps(legacy, ensure_ascii=False))

        self._create_user("New user")
        self.assertEqual([], self.client.get("/api/worlds").json())
        self.client.post(f"/api/users/{DEFAULT_USER_ID}/select")
        self.assertEqual(["wlegacy"], [world["id"] for world in self.client.get("/api/worlds").json()])


if __name__ == "__main__":
    unittest.main()
