"""Submit the video once the six sections are written: field order, readiness check, merging the tail.

The model writes field by field in schema order. Submitting the video only needs the six sections
and the motion plan; options / quest_updates / carry are used after the video, about 16% of the
output. With them last, the server can send the job about 0.4 s earlier.
"""

from __future__ import annotations

import sys
import time
import unittest
from concurrent import futures
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from murea_agent.director import agent_schema
from murea_agent.director import director  # noqa: E402
from murea_agent import loop
from murea_agent.memory.world import World


class SchemaOrderTest(unittest.TestCase):
    def test_required_drives_generation_order_not_properties(self) -> None:
        """⚠️ Generation order is decided by `required`, not `properties`.

        Measured (same schema, only required reversed):

            current            output order vs required +1.000  vs properties +0.641
            required reversed  output order vs required +1.000  vs properties -0.625
                               the first 6 fields flipped entirely

        Before this only properties had been reordered, so two optimizations saved nothing: "submit
        once the six sections are written" (the late fields were at positions 1/4/17/18 of required,
        `mainline` the 2nd written) and "move new_subjects early" (21st in required, so early
        drawing never got its window).

        This test guards that both orders agree.
        """
        schema = agent_schema._agent_schema()
        req = list(schema["required"])
        props = [k for k in schema["properties"] if k in set(req)]
        self.assertEqual(props, req, "required 必须和 properties 同序，否则重排无效")

    def test_image_warming_fields_are_written_early(self) -> None:
        """`new_subjects` / `place_image_prompt` decide which images to draw. The earlier they are written, the more drawing
        overlaps with the rest of the director's output.
        """
        req = list(agent_schema._agent_schema()["required"])
        for key in ("place", "place_image_prompt", "new_subjects"):
            with self.subTest(key=key):
                self.assertLess(req.index(key), len(req) // 2, f"{key} 排在后半段，提前开画抢不到窗口")

    def test_six_section_is_written_before_the_after_video_fields(self) -> None:
        """The order of properties is functional: it is the model's output order."""
        # ⚠️ Look at required, not properties: the former is the generation order
        keys = list(agent_schema._agent_schema()["required"])
        # Only the sections the model really writes: retention_analysis is built by the server now
        last_section = max(keys.index(k) for k in agent_schema._MODEL_SECTIONS)
        for late in agent_schema.AGENT_LATE_FIELDS:
            with self.subTest(field=late):
                self.assertGreater(keys.index(late), last_section, f"{late} 排在六段式前面，视频模型就得等它")

    def test_no_video_critical_field_is_marked_late(self) -> None:
        """Wrongly listing one extra field would build the prompt from half a decision."""
        for late in agent_schema.AGENT_LATE_FIELDS:
            self.assertNotIn(late, agent_schema.SECTION_ORDER)
        self.assertNotIn(
            "retention_analysis", agent_schema._MODEL_SECTIONS, "它是服务端拼的，不该再出现在 schema 里"
        )
        for key in (
            "state_updates",
            "motion_plan",
            "beat",
            "action_sequence",
            "place",
            "subjects",
            "dialogue",
        ):
            self.assertNotIn(key, agent_schema.AGENT_LATE_FIELDS)

    def test_nothing_read_before_video_may_be_deferred(self) -> None:
        """The list can only be guarded by **reading the source**; memory misses things.

        The first version missed three: `implicit_player` / `quest_updates` / `carry`. Their
        literals do not appear in `turn()`; they are read **inside** `_inject_implicit_player` /
        `_sanitize_intermediate_directive` / `_stage_declared_places`, all called before the video.
        The early submission path then proceeded without them, and the implicit player was never
        registered, silently.
        """
        import inspect
        import re

        src = inspect.getsource(loop)
        start = src.index("    def plan(self) -> None:")
        # Anchor on explicit markers in the source, not on code literals that change
        submit = src.index("VIDEO-SUBMIT-BOUNDARY", start)
        before = src[start:submit]

        def body_of(name: str) -> str:
            fn = getattr(loop.TurnLoop, name, None) or getattr(loop, name, None)
            try:
                return inspect.getsource(fn)
            except (TypeError, OSError):
                return ""

        offenders: list[tuple[str, list[str]]] = []
        # Check every function that receives d before the video model (in any module), including its
        # body
        for fn in sorted(set(re.findall(r"\b(_[a-z_]+)\s*\([^)]*\bd\b", before))):
            body = body_of(fn)
            hit = [f for f in agent_schema.AGENT_LATE_FIELDS if re.search(rf"[\"']{f}[\"']", body)]
            if hit:
                offenders.append((fn, hit))
        # Also check what is written directly in turn()
        inline = [f for f in agent_schema.AGENT_LATE_FIELDS if re.search(rf"[\"']{f}[\"']", before)]
        self.assertEqual([], offenders, f"这些函数在视频模型之前读了被延后的字段：{offenders}")
        self.assertEqual([], inline, f"turn() 在视频模型之前直接读了被延后的字段：{inline}")


class PartialReadyTest(unittest.TestCase):
    """By subtraction: a missing entry only delays submission (safe), it never makes it earlier."""

    def setUp(self) -> None:
        self.schema = agent_schema._agent_schema()
        self.required = list(self.schema["required"])

    def test_not_ready_while_a_needed_field_is_missing(self) -> None:
        part = {k: "x" for k in self.required if k != "detailed_description"}
        self.assertFalse(agent_schema._agent_partial_ready(part, self.schema))

    def test_ready_when_only_late_fields_are_missing(self) -> None:
        part = {k: "x" for k in self.required if k not in agent_schema.AGENT_LATE_FIELDS}
        self.assertTrue(agent_schema._agent_partial_ready(part, self.schema))

    def test_an_unknown_new_field_defaults_to_blocking(self) -> None:
        schema = {"required": [*self.required, "brand_new_field"]}
        part = {k: "x" for k in self.required}
        self.assertFalse(
            agent_schema._agent_partial_ready(part, schema), "没列进 late 的字段一律当成视频模型需要的"
        )


class MergeLateTest(unittest.TestCase):
    def test_only_late_fields_are_taken_from_the_tail(self) -> None:
        """Code in between rewrites beat / recap / continue_with in place; those values must not be overwritten.

        The most dangerous part of this change: updating with the whole result would replace the
        server's values with the model's originals, and picture and subtitles would disagree at
        once.
        """
        d = {
            "beat": "服务端改写过的 beat",
            "recap": "服务端写的字幕",
            "continue_with": "服务端算的续跑",
            "options": [],
        }
        full = {
            "beat": "模型的原始 beat",
            "recap": "模型的原话",
            "continue_with": "模型瞎写的",
            "options": ["A", "B", "C"],
            "objective": "走到码头尽头",
            "carry": "手里拿着灯笼",
        }
        fut: futures.Future = futures.Future()
        fut.set_result(full)
        director._merge_late(d, fut)
        self.assertEqual("服务端改写过的 beat", d["beat"])
        self.assertEqual("服务端写的字幕", d["recap"])
        self.assertEqual("服务端算的续跑", d["continue_with"])
        self.assertEqual(["A", "B", "C"], d["options"])
        self.assertEqual("走到码头尽头", d["objective"])
        self.assertNotIn(
            "carry",
            d,
            "carry 不在延后名单里（视频模型之前 _stage_declared_places 就读它了），所以不能从尾巴里合回来",
        )

    def test_a_failed_tail_never_breaks_the_turn(self) -> None:
        """The six sections already went to the video model; this beat must produce its clip as usual."""
        d = {"beat": "b"}
        fut: futures.Future = futures.Future()
        fut.set_exception(RuntimeError("boom"))
        director._merge_late(d, fut)
        self.assertEqual({"beat": "b"}, d)

    def test_none_future_is_a_noop(self) -> None:
        d = {"beat": "b"}
        director._merge_late(d, None)
        self.assertEqual({"beat": "b"}, d)


class DecideEarlyTest(unittest.TestCase):
    def test_returns_before_the_tail_is_written(self) -> None:
        schema = agent_schema._agent_schema()
        early = {k: "x" for k in schema["required"] if k not in agent_schema.AGENT_LATE_FIELDS}
        full = {**early, **{k: "late" for k in agent_schema.AGENT_LATE_FIELDS}}

        def slow(world, text, on_partial=None, **kw):
            if on_partial:
                on_partial(early)
            time.sleep(1.0)  # the tail is still being written
            return full

        with patch.object(director, "_decide", slow):
            t0 = time.time()
            d, fut = director._decide_early(
                World(title="t"), "走出去", lambda part: None, duration=None, asset_mentions=None
            )
            took = time.time() - t0
        self.assertIsNotNone(fut, "尾巴还没写完，应该拿到 future")
        self.assertLess(took, 0.6, "不该等模型把尾巴写完")
        self.assertTrue(agent_schema._agent_partial_ready(d, schema))
        director._merge_late(d, fut)
        self.assertEqual("late", d["options"])


if __name__ == "__main__":
    unittest.main()
