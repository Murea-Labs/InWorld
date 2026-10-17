#!/usr/bin/env python3
"""Fast local regression checks for reference-asset mutations (no Gemini/video model)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient

from murea_agent import config
from app.api import create_app
from app.api.sse import mutation_slot
from murea_agent.memory.world import World


class AssetRoutesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.original_paths = {
            "DATA": config.DATA,
            "REFS": config.REFS,
            "CLIPS": config.CLIPS,
            "SESSIONS": config.SESSIONS,
        }
        config.DATA = root
        config.REFS = root / "refs"
        config.CLIPS = root / "clips"
        config.SESSIONS = root / "sessions"
        for directory in (config.REFS, config.CLIPS, config.SESSIONS):
            directory.mkdir(parents=True)
        for name in ("old.png", "new.png"):
            (config.REFS / name).write_bytes(name.encode())
        self.client = TestClient(create_app())

    def tearDown(self) -> None:
        self.client.close()
        for name, value in self.original_paths.items():
            setattr(config, name, value)
        self.tmp.cleanup()

    def _world(self) -> World:
        world = World(title="asset route test")
        world.save()
        return world

    def test_replace_subject_keeps_id_and_clears_old_visual_wording(self) -> None:
        world = self._world()
        sid = world.add_subject("角色", "old.png", "旧外观")
        world.subjects[sid]["definition"] = "旧的锁定定义"
        world.save()

        response = self.client.put(
            f"/api/worlds/{world.id}/subjects/{sid}",
            data={"ref": "new.png"},
        )

        self.assertEqual(200, response.status_code, response.text)
        replaced = response.json()
        self.assertEqual(sid, replaced["id"])
        self.assertEqual("new.png", replaced["ref"])
        self.assertEqual("", replaced["desc"])
        self.assertNotIn("definition", replaced)

    def test_uploaded_images_are_capped_and_must_be_images(self) -> None:
        import io

        from PIL import Image

        from app.api.routes import assets

        world = self._world()
        buf = io.BytesIO()
        Image.new("RGB", (8, 8), (200, 10, 10)).save(buf, "PNG")
        png = buf.getvalue()

        def upload(raw: bytes, name: str = "Mira"):
            return self.client.post(
                f"/api/worlds/{world.id}/subjects",
                data={"name": name},
                files={"image": ("photo.png", raw, "image/png")},
            )

        before = set(config.REFS.iterdir())
        with patch.object(assets, "MAX_IMAGE_BYTES", len(png) - 1):
            self.assertEqual(413, upload(png).status_code)
        self.assertEqual(400, upload(b"<html><script></script></html>").status_code)
        self.assertEqual(400, upload(png[:20]).status_code)  # truncated
        self.assertEqual(before, set(config.REFS.iterdir()))  # nothing was stored
        response = upload(png)
        self.assertEqual(200, response.status_code, response.text)
        self.assertTrue(response.json()["ref"].endswith(".png"))

    def test_a_ref_that_is_not_a_plain_image_name_is_not_reused(self) -> None:
        world = self._world()
        for ref in ("..", ".", "../refs", "refs"):
            with self.subTest(ref=ref):
                response = self.client.post(
                    f"/api/worlds/{world.id}/subjects", data={"name": f"Mira {ref}", "ref": ref}
                )
                self.assertEqual(400, response.status_code, response.text)

    def test_voice_uploads_are_read_only_up_to_the_limit(self) -> None:
        from murea_agent.memory import voices

        with patch.object(voices, "MAX_UPLOAD_BYTES", 8):
            response = self.client.post(
                "/api/voices/upload", files={"audio": ("v.wav", b"RIFF0000WAVEfmt " + b"0" * 64, "audio/wav")}
            )
        self.assertEqual(400, response.status_code)
        self.assertIn("too large", response.json()["detail"])

    def test_manual_item_is_stored_as_item_without_character_controls(self) -> None:
        world = self._world()

        response = self.client.post(
            f"/api/worlds/{world.id}/subjects",
            data={
                "name": "黄铜罗盘",
                "ref": "new.png",
                "kind": "物件",
                "control": "user",
                "voice_id": "m_young_high",
            },
        )

        self.assertEqual(200, response.status_code, response.text)
        created = response.json()
        self.assertEqual("物件", created["kind"])
        self.assertEqual("ai", created["control"])
        self.assertEqual("", created["voice"]["ref"])

    def test_manual_item_prompt_uses_object_reference_layout(self) -> None:
        world = self._world()

        with patch(
            "app.api.routes.assets.imagegen.generate",
            return_value=b"drawn-item",
        ) as generate:
            response = self.client.post(
                f"/api/worlds/{world.id}/subjects",
                data={"name": "黄铜罗盘", "image_prompt": "A brass compass", "kind": "物件"},
            )

        self.assertEqual(200, response.status_code, response.text)
        generate.assert_called_once_with(
            "A brass compass",
            kind="object",
            visual_style="",
        )

    def test_replacing_item_image_keeps_the_object_generator(self) -> None:
        world = self._world()
        sid = world.add_subject("黄铜罗盘", "old.png", kind="物件")
        world.subjects[sid]["definition"] = "old appearance"
        world.save()
        with patch("app.api.routes.assets.imagegen.generate", return_value=b"new-item") as generate:
            response = self.client.put(
                f"/api/worlds/{world.id}/subjects/{sid}",
                data={"image_prompt": "A brass compass with a red needle"},
            )
        self.assertEqual(200, response.status_code, response.text)
        generate.assert_called_once_with(
            "A brass compass with a red needle",
            kind="object",
            visual_style="",
        )
        self.assertEqual("物件", response.json()["kind"])
        self.assertNotIn("definition", response.json())

    def test_item_can_be_deleted_without_rewriting_historical_nodes(self) -> None:
        world = self._world()
        person = world.add_subject("角色", "old.png")
        item = world.add_subject("黄铜罗盘", "new.png", kind="物件")
        world.subjects[person]["state"]["relations"] = {
            item: "随身携带",
            "黄铜罗盘": "重要线索",
        }
        world.add_place("会议室", "old.png")
        world.places["会议室"]["state"]["props"] = ["黄铜罗盘", "长桌"]
        world.nodes["n1"] = {
            "id": "n1",
            "subjects": [person, item],
            "pictures": [{"id": item, "ref": "new.png"}],
            "carry": "披着雨衣；手里拿着黄铜罗盘",
        }
        world.head = "n1"
        world.save()

        response = self.client.delete(f"/api/worlds/{world.id}/subjects/{item}")

        self.assertEqual(200, response.status_code, response.text)
        saved = World.load(world.id)
        self.assertNotIn(item, saved.subjects)
        self.assertEqual({}, saved.subjects[person]["state"]["relations"])
        self.assertEqual(["长桌"], saved.places["会议室"]["state"]["props"])
        self.assertEqual("披着雨衣", saved.nodes["n1"]["carry"])
        self.assertEqual([person, item], saved.nodes["n1"]["subjects"])
        self.assertEqual(item, saved.nodes["n1"]["pictures"][0]["id"])

        # A new asset must not reuse S2, which already appears in historical clips.
        self.assertEqual("S3", saved.add_subject("新角色", "old.png"))

    def test_character_cannot_be_deleted_through_the_item_route(self) -> None:
        world = self._world()
        sid = world.add_subject("角色", "old.png")
        world.save()

        response = self.client.delete(f"/api/worlds/{world.id}/subjects/{sid}")

        self.assertEqual(400, response.status_code, response.text)
        self.assertIn(sid, World.load(world.id).subjects)

    def test_missing_item_returns_not_found(self) -> None:
        world = self._world()

        response = self.client.delete(f"/api/worlds/{world.id}/subjects/S404")

        self.assertEqual(404, response.status_code, response.text)

    def test_item_deletion_is_rejected_during_generation(self) -> None:
        world = self._world()
        sid = world.add_subject("黄铜罗盘", "new.png", kind="物件")
        world.save()

        # ⚠️ Locks are per world: the slot taken must be **this world's**; another world's slot must not block here.
        with mutation_slot(world.id):
            response = self.client.delete(f"/api/worlds/{world.id}/subjects/{sid}")
        self.assertEqual(409, response.status_code, response.text)
        self.assertIn(sid, World.load(world.id).subjects)

        # Conversely: another world generating must not block asset changes in this world.
        # This used to be one global lock, so users waited on each other; that was the bug.
        with mutation_slot("w-someone-else"):
            ok = self.client.delete(f"/api/worlds/{world.id}/subjects/{sid}")
        self.assertEqual(200, ok.status_code, ok.text)

    def test_replace_place_preserves_identity_metadata_and_current_place(self) -> None:
        world = self._world()
        world.add_place("会议室", "old.png", "旧外观")
        first_seen = world.places["会议室"]["first_seen"]
        world.here = "会议室"
        world.save()

        response = self.client.post(
            f"/api/worlds/{world.id}/places",
            data={"name": "会议室", "ref": "new.png", "make_current": "false"},
        )

        self.assertEqual(200, response.status_code, response.text)
        saved = World.load(world.id)
        self.assertEqual("会议室", saved.here)
        self.assertEqual(first_seen, saved.places["会议室"]["first_seen"])
        self.assertEqual("new.png", saved.places["会议室"]["ref"])
        self.assertEqual("", saved.places["会议室"]["desc"])

    def test_asset_mutation_is_rejected_during_generation(self) -> None:
        world = self._world()
        sid = world.add_subject("角色", "old.png", "")
        world.save()

        with mutation_slot(world.id):
            response = self.client.put(
                f"/api/worlds/{world.id}/subjects/{sid}",
                data={"ref": "new.png"},
            )

        self.assertEqual(409, response.status_code)
        self.assertEqual("old.png", World.load(world.id).subjects[sid]["ref"])

    def test_place_can_join_a_main_scene_without_changing_the_asset(self) -> None:
        world = self._world()
        world.add_place("潮汕牛肉餐馆", "old.png", "餐厅")
        world.add_place("餐馆包间", "new.png", "包间")
        world.save()

        response = self.client.put(
            f"/api/worlds/{world.id}/places/餐馆包间/main-scene",
            json={"main_scene": "潮汕牛肉餐馆"},
        )

        self.assertEqual(200, response.status_code, response.text)
        saved = World.load(world.id)
        self.assertEqual("new.png", saved.places["餐馆包间"]["ref"])
        self.assertEqual("潮汕牛肉餐馆", saved.places["餐馆包间"]["main_scene"])

    def test_place_can_be_split_back_into_its_own_main_scene(self) -> None:
        world = self._world()
        world.add_place("潮汕牛肉餐馆", "old.png")
        world.add_place("餐馆包间", "new.png", main_scene="潮汕牛肉餐馆")
        world.save()

        response = self.client.put(
            f"/api/worlds/{world.id}/places/餐馆包间/main-scene",
            json={"main_scene": "餐馆包间"},
        )

        self.assertEqual(200, response.status_code, response.text)
        self.assertEqual("餐馆包间", World.load(world.id).places["餐馆包间"]["main_scene"])


if __name__ == "__main__":
    unittest.main()
