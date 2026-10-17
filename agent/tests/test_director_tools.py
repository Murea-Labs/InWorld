"""Director memory tools: the history window, recall, and tool-calling rounds."""

from __future__ import annotations

import json
import sys
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from murea_agent import config, tools  # noqa: E402
from murea_agent.director import director  # noqa: E402
from murea_agent.events import Emitter, active  # noqa: E402
from murea_agent.memory.world import Node, World  # noqa: E402
from murea_agent.providers import llm  # noqa: E402


def _world(beats: int) -> World:
    world = World(id="wtools", title="t", setting="harbour town")
    sid = world.add_subject("Mira", "m.png", "a sailor")
    world.add_place("Dock", "d.png", "wooden dock")
    parent = None
    for i in range(beats):
        text = "Mira finds a brass key under the nets" if i == 1 else f"Mira walks on, beat {i}"
        node = Node(
            id=f"n{i}",
            parent=parent,
            video_id="",
            clip=f"{i}.mp4",
            place="Dock",
            beat=text,
            mode="continue",
            user_text=text,
            reply=text,
            subjects=[sid],
        )
        world.add_node(node)
        world.history += [{"role": "user", "text": text}, {"role": "model", "text": text}]
        parent = node.id
    return world


class _Stream:
    """One fake streamed HTTP response made of SSE `data:` lines."""

    def __init__(self, chunks: list[dict] | int) -> None:
        # An int is an HTTP error status with a response_format complaint.
        self.status_code = chunks if isinstance(chunks, int) else 200
        self.text = "This response_format type is unavailable now"
        self._lines = [] if isinstance(chunks, int) else [f"data: {json.dumps(c)}" for c in chunks]

    def iter_lines(self):
        return iter(self._lines)

    def read(self) -> bytes:
        return b""


def _fake_transport(responses: list[list[dict]], bodies: list[dict]):
    @contextmanager
    def stream(_method, _url, json=None, **_kw):
        bodies.append(json)
        yield _Stream(responses[len(bodies) - 1])

    return stream


class MemoryToolTest(unittest.TestCase):
    def test_recall_finds_an_old_beat_and_inspect_reports_an_asset(self) -> None:
        world = _world(30)
        found = tools.get("recall_story").fn(world, "brass key")
        self.assertEqual(2, found[0]["segment"])
        mira = tools.get("inspect_asset").fn(world, "Mira")
        self.assertEqual("a sailor", mira["description"])
        self.assertIn("error", tools.get("inspect_asset").fn(world, "Nobody"))

    def test_window_and_tool_offer(self) -> None:
        with patch.object(config, "HISTORY_WINDOW", 24), patch.object(config, "AGENT_TOOLS", "auto"):
            short, offer = director._context(_world(5))
            self.assertEqual(10, len(short))
            self.assertEqual((), offer["tools"])
            long, offer = director._context(_world(30))
            self.assertEqual(24, len(long))
            self.assertEqual("user", long[0]["role"])
            self.assertEqual({"recall_story", "inspect_asset"}, {t["name"] for t in offer["tools"]})
            self.assertIn("recall_story", offer["note"])
        with patch.object(config, "AGENT_TOOLS", "off"):
            self.assertEqual((), director._context(_world(30))[1]["tools"])

    def test_a_tool_call_is_a_traced_step(self) -> None:
        events = Emitter()
        with active(events):
            out = tools.call("recall_story", _world(30), {"query": "brass key"})
        self.assertEqual(2, out["result"][0]["segment"])
        self.assertEqual(
            ("plan", "recall_story", "done"),
            (events.steps[0].phase, events.steps[0].name, events.steps[0].status),
        )
        self.assertIn("error", tools.call("no_such_tool", None, {}))


class ToolRoundTest(unittest.TestCase):
    DECL = [tools.get("recall_story").declaration()]

    def _calls(self):
        seen: list[tuple[str, dict]] = []

        def call(name: str, args: dict) -> dict:
            seen.append((name, args))
            return {"result": [{"segment": 2, "what_happened": "found a brass key"}]}

        return seen, call

    def test_gemini_runs_the_call_then_streams_the_answer(self) -> None:
        bodies: list[dict] = []
        first = [
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "functionCall": {"name": "recall_story", "args": {"query": "key"}},
                                    "thoughtSignature": "sig",
                                }
                            ]
                        }
                    }
                ]
            }
        ]
        second = [
            {"candidates": [{"content": {"parts": [{"text": '{"reply": "The key", '}]}}]},
            {"candidates": [{"content": {"parts": [{"text": '"beat": "x"}'}]}}]},
        ]
        seen, call = self._calls()
        partial: list[dict] = []
        with (
            patch.object(config, "LLM_BACKEND", "gemini"),
            patch.object(config, "GEMINI_API_KEY", "k"),
            patch.object(llm._client, "stream", _fake_transport([first, second], bodies)),
        ):
            out = llm.chat(
                system="s",
                user="u",
                schema={"type": "object"},
                model="gemini-x",
                on_partial=partial.append,
                tools=self.DECL,
                call_tool=call,
            )
        self.assertEqual({"reply": "The key", "beat": "x"}, out)
        self.assertEqual([("recall_story", {"query": "key"})], seen)
        self.assertNotIn("cachedContent", bodies[0])
        self.assertEqual("recall_story", bodies[0]["tools"][0]["functionDeclarations"][0]["name"])
        turn = bodies[1]["contents"]
        self.assertEqual("sig", turn[-2]["parts"][0]["thoughtSignature"])
        self.assertEqual("recall_story", turn[-1]["parts"][0]["functionResponse"]["name"])
        self.assertTrue(partial)

    def test_gemini_stops_offering_tools_after_the_round_limit(self) -> None:
        bodies: list[dict] = []
        loop_call = [
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [{"functionCall": {"name": "recall_story", "args": {"query": "key"}}}]
                        }
                    }
                ]
            }
        ]
        answer = [{"candidates": [{"content": {"parts": [{"text": '{"reply": "ok"}'}]}}]}]
        responses = [loop_call] * llm.MAX_TOOL_ROUNDS + [answer]
        _seen, call = self._calls()
        with (
            patch.object(config, "LLM_BACKEND", "gemini"),
            patch.object(config, "GEMINI_API_KEY", "k"),
            patch.object(llm._client, "stream", _fake_transport(responses, bodies)),
        ):
            out = llm.chat(
                system="s",
                user="u",
                schema={"type": "object"},
                model="gemini-x",
                on_partial=lambda _p: None,
                tools=self.DECL,
                call_tool=call,
            )
        self.assertEqual({"reply": "ok"}, out)
        self.assertEqual({"functionCallingConfig": {"mode": "NONE"}}, bodies[-1]["toolConfig"])

    def test_openai_compatible_runs_the_call_then_streams_the_answer(self) -> None:
        bodies: list[dict] = []
        first = [
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "c1",
                                    "function": {"name": "recall_story", "arguments": '{"query":'},
                                }
                            ]
                        }
                    }
                ]
            },
            {
                "choices": [
                    {
                        "delta": {"tool_calls": [{"index": 0, "function": {"arguments": ' "key"}'}}]},
                        "finish_reason": "tool_calls",
                    }
                ]
            },
        ]
        second = [{"choices": [{"delta": {"content": '{"reply": "ok"}'}}]}]
        seen, call = self._calls()
        with (
            patch.object(config, "LLM_BACKEND", "openai"),
            patch.object(config, "OPENAI_BASE", "http://llm.test/v1"),
            patch.object(config, "OPENAI_JSON_MAX_WHITESPACE", 0),
            patch.object(llm._client, "stream", _fake_transport([first, second], bodies)),
        ):
            out = llm.chat(
                system="s",
                user="u",
                schema={"type": "object"},
                model="m",
                on_partial=lambda _p: None,
                tools=self.DECL,
                call_tool=call,
            )
        self.assertEqual({"reply": "ok"}, out)
        self.assertEqual([("recall_story", {"query": "key"})], seen)
        msgs = bodies[1]["messages"]
        self.assertEqual("c1", msgs[-2]["tool_calls"][0]["id"])
        self.assertEqual(("tool", "c1"), (msgs[-1]["role"], msgs[-1]["tool_call_id"]))

    def test_openai_compatible_falls_back_to_json_object_without_schema_support(self) -> None:
        bodies: list[dict] = []
        answer = [{"choices": [{"delta": {"content": '{"reply": "ok"}'}}]}]
        llm._no_json_schema.discard("ds-test")
        with (
            patch.object(config, "LLM_BACKEND", "openai"),
            patch.object(config, "OPENAI_BASE", "http://llm.test/v1"),
            patch.object(config, "OPENAI_JSON_MAX_WHITESPACE", 0),
            patch.object(llm._client, "stream", _fake_transport([400, answer, answer], bodies)),
        ):
            schema = {"type": "object", "properties": {"reply": {"type": "string"}}}
            out = llm.chat(system="s", user="u", schema=schema, model="ds-test", on_partial=lambda _p: None)
            self.assertEqual({"reply": "ok"}, out)
            self.assertEqual("json_schema", bodies[0]["response_format"]["type"])
            self.assertEqual({"type": "json_object"}, bodies[1]["response_format"])
            self.assertIn('"reply"', bodies[1]["messages"][0]["content"])
            # Learnt once per model: the next call goes straight to json_object.
            llm.chat(system="s", user="u", schema=schema, model="ds-test", on_partial=lambda _p: None)
            self.assertEqual(3, len(bodies))
            self.assertEqual({"type": "json_object"}, bodies[2]["response_format"])

    def test_without_call_tool_no_tools_are_sent(self) -> None:
        bodies: list[dict] = []
        answer = [{"candidates": [{"content": {"parts": [{"text": '{"reply": "ok"}'}]}}]}]
        with (
            patch.object(config, "LLM_BACKEND", "gemini"),
            patch.object(config, "GEMINI_API_KEY", "k"),
            patch.object(llm, "_cache_for", return_value=None),
            patch.object(llm._client, "stream", _fake_transport([answer], bodies)),
        ):
            llm.chat(
                system="s",
                user="u",
                schema={"type": "object"},
                model="gemini-x",
                on_partial=lambda _p: None,
                tools=self.DECL,
            )
        self.assertNotIn("tools", bodies[0])


if __name__ == "__main__":
    unittest.main()
