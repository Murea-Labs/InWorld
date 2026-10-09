"""The self-hosted (SGLang / vLLM) OpenAI-protocol path keeps the same contract as the Gemini path.

It exists for **the tail**, not the median: a hosted small model had a median first token of 1.3 s /
p90 5.5 s, while a self-hosted server with prefix caching answered in 0.16 s / p90 0.25 s.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from murea_agent import config
from murea_agent.providers import llm

SCHEMA = {"type": "object", "properties": {"reply": {"type": "string"}}, "required": ["reply"]}
HISTORY = [{"role": "user", "text": "推开门"}, {"role": "model", "text": "门开了。"}]


class Dispatch(unittest.TestCase):
    def test_fail_closed_without_a_base(self) -> None:
        """Silently using Gemini because one setting is missing is the worst case: both arms of a comparison would come
        from the same backend, invisibly.
        """
        with mock.patch.object(config, "LLM_BACKEND", "openai"), mock.patch.object(config, "OPENAI_BASE", ""):
            self.assertFalse(llm._use_openai())

    def test_default_stays_on_gemini(self) -> None:
        with (
            mock.patch.object(config, "LLM_BACKEND", "gemini"),
            mock.patch.object(config, "OPENAI_BASE", "http://x/v1"),
        ):
            self.assertFalse(llm._use_openai())

    def test_a_per_call_base_is_enough(self) -> None:
        """When hedging there is one endpoint per replica and maybe no default OPENAI_BASE. Ignoring the per-call base would
        silently fall back to Gemini; on a network without Gemini access the whole call hangs with
        no error.
        """
        with mock.patch.object(config, "LLM_BACKEND", "openai"), mock.patch.object(config, "OPENAI_BASE", ""):
            self.assertTrue(llm._use_openai("http://replica-1/v1"))
            self.assertFalse(llm._use_openai(""))

    def test_both_set_switches_over(self) -> None:
        with (
            mock.patch.object(config, "LLM_BACKEND", "openai"),
            mock.patch.object(config, "OPENAI_BASE", "http://x/v1"),
        ):
            self.assertTrue(llm._use_openai())


class BodyShape(unittest.TestCase):
    def _body(self, **kw):
        return llm._openai_body(
            "系统提示",
            HISTORY,
            "用户说：往前走",
            "m",
            SCHEMA,
            0.9,
            kw.pop("thinking", 0),
            kw.pop("images", ()),
            stream=kw.pop("stream", True),
        )

    def test_history_roles_are_translated(self) -> None:
        msgs = self._body()["messages"]
        self.assertEqual([m["role"] for m in msgs], ["system", "user", "assistant", "user"])
        self.assertEqual(msgs[0]["content"], "系统提示")
        self.assertEqual(msgs[-1]["content"], "用户说：往前走")

    def test_image_rides_on_the_last_user_message(self) -> None:
        """That image is "where the world stands now". Attached to the history the model reads it as an old image."""
        msgs = self._body(images=[("image/jpeg", b"\xff\xd8\xff")])["messages"]
        parts = msgs[-1]["content"]
        self.assertEqual(parts[0]["type"], "text")
        self.assertEqual(parts[1]["type"], "image_url")
        self.assertTrue(parts[1]["image_url"]["url"].startswith("data:image/jpeg;base64,"))

    def test_thinking_zero_turns_it_off_via_chat_template(self) -> None:
        """Qwen3-family models disable thinking through the chat template, not sampling parameters. Otherwise they write
        thousands of thinking tokens first and the first token arrives after 10 s.
        """
        self.assertEqual(self._body(thinking=0)["chat_template_kwargs"], {"enable_thinking": False})

    def test_no_thinking_field_when_budget_is_unset(self) -> None:
        self.assertNotIn("chat_template_kwargs", self._body(thinking=None))

    def test_schema_becomes_strict_json_schema(self) -> None:
        rf = self._body()["response_format"]
        self.assertEqual(rf["type"], "json_schema")
        self.assertTrue(rf["json_schema"]["strict"])
        self.assertIs(rf["json_schema"]["schema"], SCHEMA)

    def test_stream_asks_for_usage(self) -> None:
        """Without usage tok/s cannot be measured, and tok/s is the only criterion for whether this path is worth it."""
        self.assertEqual(self._body(stream=True)["stream_options"], {"include_usage": True})
        self.assertNotIn("stream", self._body(stream=False))


class Signatures(unittest.TestCase):
    """All four functions must pass `base` through."""

    def test_base_is_threaded_all_the_way(self) -> None:
        """⚠️ A missing link is very hard to find: `_use_openai(base)` returns False and the call **silently goes back to
        Gemini**; on a network without Gemini access it just hangs with no error.
        """
        import inspect

        for name in ("chat", "_chat_stream", "_openai_stream", "_openai_chat"):
            with self.subTest(fn=name):
                self.assertIn("base", inspect.signature(getattr(llm, name)).parameters)

    def test_the_streaming_path_actually_forwards_it(self) -> None:
        import inspect

        self.assertIn("base=base", inspect.getsource(llm._chat_stream))


class Salvage(unittest.TestCase):
    """When exceeding max_tokens truncates the JSON, use the fields already complete instead of falling back to non-streaming.

    The most common "streaming collapsed" is not the network but the model writing so much that the
    JSON is cut halfway. Falling back makes the same model write the same overlong output again:
    twice as slow and likely truncated again. Seen in 2 of 14 beats.
    """

    def test_partial_json_never_reports_a_half_written_field(self) -> None:
        """The precondition of the fallback: `partial_json` reports only completed fields. Otherwise it might keep half a
        sentence and send it to the video model.
        """
        buf = '{"a": "完整", "b": "写了一半'
        got = llm.partial_json(buf)
        self.assertEqual(got, {"a": "完整"})
        self.assertNotIn("b", got)

    def test_a_truncated_tail_still_yields_the_finished_head(self) -> None:
        buf = '{"place": "木栈道", "detailed_description": "[Shot 1] fog", "x": "断'
        got = llm.partial_json(buf)
        self.assertEqual(got["place"], "木栈道")
        self.assertEqual(got["detailed_description"], "[Shot 1] fog")

    def test_nothing_finished_yet_gives_nothing(self) -> None:
        self.assertIsNone(llm.partial_json('{"a": "还在写'))


class Usage(unittest.TestCase):
    def test_maps_to_the_same_keys_gemini_uses(self) -> None:
        self.assertEqual(
            llm._openai_usage({"prompt_tokens": 24282, "completion_tokens": 1200}), {"in": 24282, "out": 1200}
        )

    def test_reasoning_tokens_surface_as_think(self) -> None:
        got = llm._openai_usage({"prompt_tokens": 1, "completion_tokens": 2, "reasoning_tokens": 1800})
        self.assertEqual(got["think"], 1800)

    def test_missing_usage_does_not_explode(self) -> None:
        self.assertEqual(llm._openai_usage(None), {"in": 0, "out": 0})


if __name__ == "__main__":
    unittest.main(verbosity=2)
