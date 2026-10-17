"""Reference generation must leave the HTTP/SSE event loop responsive."""

import asyncio
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from fastapi import FastAPI
from murea_agent import config
from app.api.routes import assets


class ImageConcurrencyTest(unittest.IsolatedAsyncioTestCase):
    async def check_render(self, attached):
        started, release = threading.Event(), threading.Event()

        def draw(*args, **kwargs):
            started.set()
            release.wait(2)
            return b"image-result"

        app = FastAPI()
        app.include_router(assets.router)

        @app.get("/probe")
        async def probe():
            return {"ready": True}

        @app.post("/attached")
        async def attach():
            return {
                "ref": await assets._resolve_ref(
                    ref="", image=None, image_prompt="character", kind="subject", prefix="subj"
                )
            }

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(config, "REFS", Path(directory)),
            patch.object(assets.imagegen, "generate", side_effect=draw),
        ):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                task = asyncio.create_task(
                    client.post(
                        "/attached" if attached else "/api/imagine",
                        json={"prompt": "character", "kind": "subject"},
                    )
                )
                try:
                    self.assertTrue(await asyncio.to_thread(started.wait, 3))
                    self.assertEqual((await client.get("/probe")).json(), {"ready": True})
                    self.assertFalse(task.done(), "image synthesis blocked unrelated HTTP requests")
                finally:
                    release.set()
                    response = await task
                self.assertEqual(response.status_code, 200)
                self.assertEqual((Path(directory) / response.json()["ref"]).read_bytes(), b"image-result")

    async def test_preview_does_not_block(self):
        await self.check_render(False)

    async def test_attached_reference_does_not_block(self):
        await self.check_render(True)


if __name__ == "__main__":
    unittest.main()
