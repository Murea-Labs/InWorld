"""TTS "configured" and "alive" must be reported separately.

Real failure: the TTS service refused connections while `/api/health` said `"tts": true`, because it
used `available()` (only checks TTS_URL) while the probing `health()` existed but was not wired in.
The UI lit the voice switch and turning it on did nothing. Same class of silent failure as the voice
library's `in_prompt != pickable`.
"""

from __future__ import annotations

import sys
import json
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from murea_agent import config
from app.api.routes import health as health_route
from murea_agent.providers import tts


class _Resp:
    def __init__(self, code: int, payload: dict | None = None) -> None:
        self.status_code = code
        self._payload = payload

    def json(self) -> dict:
        if self._payload is None:
            raise ValueError("not json")
        return self._payload


class TtsHealthTest(unittest.TestCase):
    def setUp(self) -> None:
        tts._health_cache = (0.0, False)
        self.addCleanup(lambda: setattr(tts, "_health_cache", (0.0, False)))

    def test_configured_but_refused_is_not_healthy(self) -> None:
        with (
            patch.object(config, "TTS_URL", "http://tts.local"),
            patch.object(tts._client, "get", side_effect=ConnectionRefusedError("refused")),
        ):
            self.assertTrue(tts.available(), "配了就是配了")
            self.assertFalse(tts.health(), "但服务连不上就不能说自己是好的")

    def test_health_endpoint_reports_both(self) -> None:
        with (
            patch.object(config, "TTS_URL", "http://tts.local"),
            patch.object(tts._client, "get", side_effect=ConnectionRefusedError("refused")),
        ):
            out = health_route.health()
        self.assertTrue(out["tts"], "开关该不该存在")
        self.assertFalse(out["tts_up"], "服务活没活着")
        # Health needs no login; the video service's (often internal) address is not in it.
        self.assertNotIn("video_url", out)
        self.assertNotIn("http://", json.dumps(out))

    def test_http_process_without_loaded_model_is_unhealthy(self) -> None:
        """Answering /health does not mean the process can really synthesize speech."""
        with (
            patch.object(config, "TTS_URL", "http://tts.local"),
            patch.object(tts._client, "get", return_value=_Resp(200, {"ok": True, "loaded": False})),
        ):
            self.assertFalse(tts.health())

    def test_loaded_model_is_healthy(self) -> None:
        with (
            patch.object(config, "TTS_URL", "http://tts.local"),
            patch.object(tts._client, "get", return_value=_Resp(200, {"ok": True, "loaded": True})),
        ):
            self.assertTrue(tts.health())

    def test_health_shows_when_background_work_shares_the_visual_replica(self) -> None:
        """Background calls hitting the same replica as player requests must be reported, not only documented.

        `config._background_base()` says background work must never compete with the visible path:
        during playback the server makes option prefetches and a critic call, and on the same
        replica a player's request queues behind them. This violation used to be **silent**: after a
        model migration both bases pointed to the same address and the same workload swung between
        3.2 s and 11.0 s (queueing, not compute) with no warning anywhere.
        """
        cases = [
            ("", "", False, "两个都没配 —— 根本没用 openai 后端，不该报警"),
            ("http://a:30000/v1", "", True, "BOOK 空 -> 回落到 BASE，就是共用"),
            ("http://a:30000/v1", "http://a:30000/v1", True, "显式指到同一个"),
            ("http://a:30014/v1", "http://a:30013/v1", False, "分开了"),
        ]
        for base, book, expected, why in cases:
            with self.subTest(base=base, book=book):
                with (
                    patch.object(config, "OPENAI_BASE", base),
                    patch.object(config, "OPENAI_BASE_BOOK", book),
                    patch.object(config, "TTS_URL", ""),
                ):
                    out = health_route.health()
                self.assertEqual(expected, out["background_shares_visual_replica"], why)

    def test_probe_is_cached_so_health_polling_is_not_a_load_test(self) -> None:
        """The frontend polls /api/health; probing every time would load test the TTS service."""
        with (
            patch.object(config, "TTS_URL", "http://tts.local"),
            patch.object(tts._client, "get", return_value=_Resp(200)) as get,
        ):
            self.assertTrue(tts.health())
            self.assertTrue(tts.health())
            self.assertTrue(tts.health())
        self.assertEqual(1, get.call_count)

    def test_not_configured_is_not_an_error(self) -> None:
        with patch.object(config, "TTS_URL", ""):
            self.assertFalse(tts.available())
            self.assertFalse(tts.health())


if __name__ == "__main__":
    unittest.main()
