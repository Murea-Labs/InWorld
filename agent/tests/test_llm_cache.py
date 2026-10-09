"""Regression tests of the explicit system-prompt cache (no external calls).

This is a **cost** optimization, not a latency one: hits 1.01–1.20 s, misses 0.78–0.96 s, prefill is
fast anyway. It saves money: a beat calls the model several times (director + prefetches), each
re-uploading the same large prompt.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from murea_agent.providers import llm

LONG = "指令。" * 5000  # longer than _CACHE_MIN_CHARS
SHORT = "短提示词"


class _Resp:
    def __init__(self, code: int, payload: dict | None = None, text: str = "") -> None:
        self.status_code = code
        self._payload = payload or {}
        self.text = text

    def json(self) -> dict:
        return self._payload


def _clean() -> None:
    llm._caches.clear()
    llm._cache_systems.clear()
    llm._cache_off.clear()


class CacheTest(unittest.TestCase):
    def setUp(self) -> None:
        _clean()
        self.addCleanup(_clean)
        # Caching is skipped without a key; the test never calls the network.
        key = patch.object(llm.config, "GEMINI_API_KEY", "test-key")
        key.start()
        self.addCleanup(key.stop)

    def _body(self, system: str) -> dict:
        return llm._body(system, [], "你好", "m", None, 0.8, None)

    def test_long_system_becomes_a_cache_reference(self) -> None:
        ok = _Resp(200, {"name": "cachedContents/abc"})
        with patch.object(llm._client, "post", return_value=ok) as post:
            body = self._body(LONG)
        self.assertEqual("cachedContents/abc", body["cachedContent"])
        self.assertNotIn("systemInstruction", body, "用了缓存就不能再把 system 原文一起发，否则一分钱没省")
        self.assertEqual(1, post.call_count)

    def test_second_call_reuses_the_same_cache(self) -> None:
        ok = _Resp(200, {"name": "cachedContents/abc"})
        with patch.object(llm._client, "post", return_value=ok) as post:
            self._body(LONG)
            self._body(LONG)
        self.assertEqual(1, post.call_count, "同一份提示词不该建两次缓存")

    def test_short_system_is_never_cached(self) -> None:
        with patch.object(llm._client, "post") as post:
            body = self._body(SHORT)
        post.assert_not_called()
        self.assertIn("systemInstruction", body)

    def test_creation_failure_falls_back_and_stops_retrying(self) -> None:
        """A key without the feature or an unsupported model: retrying only wastes a call every beat."""
        bad = _Resp(403, text="not enabled")
        with patch.object(llm._client, "post", return_value=bad) as post:
            first = self._body(LONG)
            second = self._body(LONG)
        self.assertIn("systemInstruction", first)
        self.assertIn("systemInstruction", second)
        self.assertEqual(1, post.call_count, "失败一次之后不该再试")

    def test_creation_exception_never_escapes(self) -> None:
        with patch.object(llm._client, "post", side_effect=RuntimeError("boom")):
            body = self._body(LONG)
        self.assertIn("systemInstruction", body)

    def test_uncache_restores_the_system_prompt(self) -> None:
        """When the cache expires the original must be restored exactly, or this beat fails outright."""
        ok = _Resp(200, {"name": "cachedContents/abc"})
        with patch.object(llm._client, "post", return_value=ok):
            body = self._body(LONG)
        self.assertTrue(llm._uncache_body(body))
        self.assertNotIn("cachedContent", body)
        self.assertEqual(LONG, body["systemInstruction"]["parts"][0]["text"])
        # once forgotten, it is created again next time
        with patch.object(llm._client, "post", return_value=ok) as post:
            self._body(LONG)
        self.assertEqual(1, post.call_count)


if __name__ == "__main__":
    unittest.main()
