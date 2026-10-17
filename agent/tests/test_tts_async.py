"""Narration synthesis must not block the app's HTTP/SSE event loop."""

import asyncio
import threading
import unittest
from unittest.mock import patch

import httpx
from fastapi import FastAPI
from app.api.routes import tts as route


class NarrationConcurrencyTest(unittest.IsolatedAsyncioTestCase):
    async def test_other_requests_continue_while_audio_is_synthesized(self):
        started, release = threading.Event(), threading.Event()

        def synthesize(*args):
            started.set()
            release.wait(1)
            return b"test-audio"

        app = FastAPI()
        app.include_router(route.router)

        @app.get("/probe")
        async def probe():
            return {"ready": True}

        with (
            patch.object(route.tts, "available", return_value=True),
            patch.object(route.tts, "speak", side_effect=synthesize),
        ):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                task = asyncio.create_task(client.post("/api/tts", json={"text": "你好"}))
                try:
                    self.assertTrue(await asyncio.to_thread(started.wait, 2))
                    response = await client.get("/probe")
                    self.assertEqual(response.json(), {"ready": True})
                    self.assertFalse(task.done(), "synthesis blocked unrelated HTTP requests")
                finally:
                    release.set()
                    audio = await task
                self.assertEqual(audio.status_code, 200)
                self.assertEqual(audio.content, b"test-audio")


if __name__ == "__main__":
    unittest.main()
