"""Regression tests of the call log (no external calls)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from murea_agent import config
from murea_agent import reqlog


class ReqLogTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        p = patch.object(config, "DATA", Path(self.tmp.name))
        p.start()
        self.addCleanup(p.stop)

    def test_nothing_is_written_without_a_bound_world(self) -> None:
        """Without a bound world it is dropped: entries must belong to a world, or recording them is useless."""
        reqlog.record("gemini", model="x")
        self.assertEqual([], list(Path(self.tmp.name).rglob("*.jsonl")))

    def test_bound_world_gets_its_own_file_newest_first(self) -> None:
        token = reqlog.bind("w1")
        reqlog.record("turn", user_text="第一句")
        reqlog.record("gemini", model="flash", ok=True)
        reqlog.unbind(token)
        rows = reqlog.read("w1")
        self.assertEqual(["gemini", "turn"], [r["kind"] for r in rows])
        self.assertEqual("第一句", rows[1]["user_text"])

    def test_worlds_do_not_leak_into_each_other(self) -> None:
        for wid in ("w1", "w2"):
            t = reqlog.bind(wid)
            reqlog.record("turn", user_text=wid)
            reqlog.unbind(t)
        self.assertEqual(["w1"], [r["user_text"] for r in reqlog.read("w1")])
        self.assertEqual(["w2"], [r["user_text"] for r in reqlog.read("w2")])

    def test_long_text_is_clipped(self) -> None:
        t = reqlog.bind("w1")
        reqlog.record("gemini", response="x" * (reqlog.MAX_TEXT + 500))
        reqlog.unbind(t)
        got = reqlog.read("w1")[0]["response"]
        self.assertLess(len(got), reqlog.MAX_TEXT + 60)
        self.assertIn("500 more characters", got)

    def test_a_traversal_id_cannot_escape_the_log_dir(self) -> None:
        """The world id comes from a path parameter."""
        self.assertEqual(
            (Path(self.tmp.name) / "logs").resolve(), reqlog.path("../../etc/passwd").parent.resolve()
        )

    def test_large_logs_do_not_rescan_for_every_append(self):
        p = reqlog.path("large")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(("x" * 600 + "\n") * 600)
        reqlog._trim_ticks.pop(p, None)
        for _ in range(63):
            reqlog._trim(p)
        self.assertEqual(600, len(p.read_text().splitlines()))
        reqlog._trim(p)
        self.assertEqual(reqlog.MAX_LINES, len(p.read_text().splitlines()))

    def test_record_never_raises(self) -> None:
        t = reqlog.bind("w1")
        with patch.object(reqlog, "path", side_effect=RuntimeError("boom")):
            reqlog.record("gemini", model="x")  # must not raise
        reqlog.unbind(t)

    def test_image_event_keeps_asset_kind_separate(self) -> None:
        """The event kind and the asset kind must not both use record's kind parameter."""
        from murea_agent.providers import imagegen

        response = {
            "candidates": [
                {
                    "content": {"parts": [{"inlineData": {"data": "aGVsbG8="}}]},
                }
            ],
        }
        token = reqlog.bind("w1")
        try:
            with (
                patch.object(config, "IMAGE_BASE", ""),
                patch.object(imagegen, "_post", return_value=response),
            ):
                self.assertEqual(b"hello", imagegen.generate("一间房", kind="place"))
        finally:
            reqlog.unbind(token)

        row = reqlog.read("w1")[0]
        self.assertEqual("image", row["kind"])
        self.assertEqual("place", row["asset_kind"])

    def test_coop_full_scope_preserves_text_and_restores_single_player_defaults(self):
        text = "x" * 12000
        token = reqlog.bind("w1")
        try:
            with reqlog.scope(round=2, request_id="req", stage="agent", full=True):
                reqlog.record("agent_request", prompt=text)
                with reqlog.scope(stage="generation"):
                    reqlog.record("video_request", prompt=text)
                self.assertTrue(reqlog.full())
            self.assertFalse(reqlog.full())
            reqlog.record("turn", prompt=text)
        finally:
            reqlog.unbind(token)
        rows = reqlog.read("w1")
        self.assertNotIn("round", rows[0])
        self.assertLess(len(rows[0]["prompt"]), len(text))
        self.assertEqual(text, rows[1]["prompt"])
        self.assertEqual("generation", rows[1]["stage"])
        self.assertEqual("agent", rows[2]["stage"])
        self.assertEqual(2, rows[2]["round"])

    def test_full_openai_log_contains_text_schema_and_no_headers_or_image_bytes(self):
        from murea_agent.providers import llm

        body = {
            "model": "test",
            "messages": [
                {"role": "system", "content": "system prompt"},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "user prompt"},
                        {"type": "image_url", "image_url": {"url": "data:secret-image"}},
                    ],
                },
            ],
            "response_format": {"type": "json_schema"},
            "Authorization": "secret-key",
        }
        with reqlog.scope(full=True):
            result = llm._log_body(body)
        self.assertEqual("system prompt", result["messages"][0]["content"])
        self.assertEqual(body["response_format"], result["response_format"])
        self.assertNotIn("secret-key", str(result))
        self.assertNotIn("secret-image", str(result))


class BindingScopeTest(unittest.TestCase):
    """The binding must cover the whole turn; the first version failed exactly here."""

    def test_turn_route_binds_around_the_whole_turn(self) -> None:
        """Real failure: unbind was in the finally of the image step, which ran **before** the video call,
        so no video_submit was ever logged. The binding is request-scoped and belongs to the route.
        """
        import inspect
        from app.api.routes import turns

        src = inspect.getsource(turns.turn)
        self.assertIn("reqlog.bind", src)
        self.assertIn("reqlog.unbind", src)
        # loop.turn must not unbind by itself
        from murea_agent import loop

        self.assertNotIn("reqlog.unbind", inspect.getsource(loop.turn))

    def test_streaming_llm_path_is_instrumented(self) -> None:
        """The director and rewriter use _chat_stream, not _post; a hook only on _post would miss them."""
        import inspect
        from murea_agent.providers import llm

        self.assertIn("reqlog.record", inspect.getsource(llm._chat_stream))


if __name__ == "__main__":
    unittest.main()
