"""Every line fed back into the director's history must be narrative in the story language.

Real failure: after "open the door and walk out" was split into four segments, 6 of 14 history lines
were internal scheduling language, 3 of them as "the user said":

    [07] MODEL: …[clip 4 generated · … · the acting hand first reaches the visible
                 handle or movable part, then moves only that part along its hinge…]
    [08] USER : (an internal continuation instruction about passing the whole body through
                 the open boundary, with continuity anchors…)

The director continued as if the player were issuing CAD commands; replies degraded into imperatives
and the mainline did not advance for seven beats. This file makes sure that never happens again.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from murea_agent import loop as P


class _Node:
    def __init__(self, **kw):
        self.recap = kw.get("recap", "")
        self.beat = kw.get("beat", "")
        self.reply = kw.get("reply", "")


ENGLISH_BEAT = (
    "the acting hand first reaches the visible handle or movable part, then moves "
    "only that part along its hinge, joint, or track until fully open. "
    "End state: the movable boundary is fully open."
)
MACHINE_USER = (
    '继续完成同一个目标"打开门走出去"：沿可见的关节或轨道完成原始目标要求的构型变化'
    "（衔接锚点：同一个人、同一个目标、同一个运动方向，以及上一段最后可见的接触状态）"
)


class HistoryStaysNarrative(unittest.TestCase):
    def test_english_physics_never_reaches_history(self) -> None:
        node = _Node(
            recap="林远山用力拉开沉重的木门。", beat=ENGLISH_BEAT, reply="木门被拉开，湿冷的雾气涌了进来。"
        )
        line = P._history_beat(node)
        self.assertEqual("林远山用力拉开沉重的木门。", line)
        for bad in ("hinge", "End state", "movable part"):
            self.assertNotIn(bad, line)

    def test_falls_back_to_reply_when_only_english_survives(self) -> None:
        """Old clips have no recap and beat is English; fall back to reply, never write English into the history."""
        node = _Node(beat=ENGLISH_BEAT, reply="木门被拉开，湿冷的雾气涌了进来。")
        self.assertEqual("木门被拉开，湿冷的雾气涌了进来。", P._history_beat(node))

    def test_old_node_with_chinese_beat_still_works(self) -> None:
        node = _Node(beat="死死抵住剧烈颤抖的木门", reply="外面的撞击声沉闷而疯狂。")
        self.assertEqual("死死抵住剧烈颤抖的木门", P._history_beat(node))

    def test_planning_language_is_rejected_even_in_chinese(self) -> None:
        """Chinese can be machine text too. Words like "configuration change" or "continuity anchor" must stay out of the history."""
        node = _Node(recap="沿可见的关节或轨道完成原始目标要求的构型变化", beat="", reply="他伸手去够门闩。")
        self.assertEqual("他伸手去够门闩。", P._history_beat(node))

    def test_machine_continuation_is_not_attributed_to_the_player(self) -> None:
        """Auto-continued beat: user_text is built by the server. If machine text slips in, use what was actually played."""
        node = _Node(recap="林远山跨过门槛，走到雾气里的木栈道上。")
        line = P._history_line(MACHINE_USER, node)
        self.assertEqual("林远山跨过门槛，走到雾气里的木栈道上。", line)
        for bad in ("衔接锚点", "构型变化", "原始目标"):
            self.assertNotIn(bad, line)

    def test_real_user_input_is_never_rewritten(self) -> None:
        node = _Node(recap="他朝门外怒吼。")
        for said in ("打开门走出去", "冲门外疯狂撞击的人大声质问", "@林远山 快跑"):
            self.assertEqual(said, P._history_line(said, node))


class ContinuationReadsLikeChinese(unittest.TestCase):
    def test_auto_beat_does_not_split_again(self) -> None:
        """A continuation beat plays one step and produces no new continue_with, or the same plan would be reordered forever."""
        directive = {
            "beat": "走出去",
            "action_sequence": {
                "goal": "打开门走出去",
                "steps": [
                    {
                        "relation_change": "containment",
                        "phase": "transform",
                        "instruction": "走进街道",
                        "end_state": "整个人在街道上",
                        "continuity_anchor": "同一个人",
                    },
                    {
                        "relation_change": "distance",
                        "phase": "approach",
                        "instruction": "沿栈道往前走",
                        "end_state": "停在栈道中段",
                        "continuity_anchor": "同一个人",
                    },
                ],
            },
        }
        motion = {
            "interaction_type": "locomotion",
            "phase": "transform",
            "relation_change": "containment",
            "actors": ["林远山"],
            "target": "水上市集街道",
            "path": "walks through the doorway",
            "end_state": "fully outside",
            "contact_points": ["feet"],
            "framing": "wide",
        }
        seq = P._apply_action_sequence(
            directive, "走出去", motion, request_text="打开门走出去", is_continuation=True
        )
        self.assertEqual(1, len(seq["steps"]))
        self.assertEqual("", directive["continue_with"])


class HistoryLineMatchesWhatWasFilmed(unittest.TestCase):
    """The bracketed summary and the reply above must not tell different stories.

    A real beat: the director's reply was "you grab the figure in front of you", while its own beat
    said "after stopping, chat naturally with the person in front"; two sources in one message
    contradicted each other and the history told the director they had a nice chat. The reply has
    been narrowed to the segment and is the only trustworthy record of the picture, so with a reply
    the summary is not added.
    """

    def test_no_second_summary_when_reply_exists(self) -> None:
        node = _Node(
            recap="停稳后与眼前的人自然交谈。", reply="你顺着木栈道快步追赶，一把抓住了前方那个身影。"
        )
        summary = P._history_beat(node) if not node.reply.strip() else ""
        self.assertEqual("", summary)

    def test_summary_still_used_when_reply_is_empty(self) -> None:
        node = _Node(recap="林远山拉开木门。", reply="")
        summary = P._history_beat(node) if not node.reply.strip() else ""
        self.assertEqual("林远山拉开木门。", summary)


class SoundscapeIsNeverEmpty(unittest.TestCase):
    def test_silent_shot_gets_the_original_short_no_speech_direction(self) -> None:
        """Without dialogue the video model still gets an explicit instruction, in the short original six-section format."""
        fields = {"overall_soundscape": "N/A"}
        P._ensure_soundscape(fields, dialogue="")
        self.assertIn("No speech", fields["overall_soundscape"])
        self.assertIn("nobody talks in this shot", fields["overall_soundscape"])
        self.assertNotIn("ABSOLUTE NO-VOICE RULE:", fields["overall_soundscape"])
        self.assertNotIn("mouth remains closed", fields["overall_soundscape"])

    def test_shot_with_dialogue_keeps_the_speech(self) -> None:
        fields = {"overall_soundscape": ""}
        P._ensure_soundscape(fields, dialogue="门外到底是什么人！")
        self.assertIn("speech", fields["overall_soundscape"])
        self.assertNotIn("No speech", fields["overall_soundscape"])

    def test_a_real_soundscape_is_kept_and_strengthened(self) -> None:
        fields = {"overall_soundscape": "Rain on the wooden roof, water lapping."}
        P._ensure_soundscape(fields, dialogue="")
        self.assertTrue(fields["overall_soundscape"].startswith("Rain on the wooden roof, water lapping."))
        self.assertIn("No speech, no voice-over, no narration", fields["overall_soundscape"])
        self.assertNotIn("ABSOLUTE NO-VOICE RULE:", fields["overall_soundscape"])

    def test_no_voice_lock_is_idempotent(self) -> None:
        fields = {"overall_soundscape": "Wind. "}
        P._ensure_soundscape(fields, dialogue="")
        once = fields["overall_soundscape"]
        P._ensure_soundscape(fields, dialogue="")
        self.assertEqual(once, fields["overall_soundscape"])

    def test_legacy_long_voice_lock_is_removed(self) -> None:
        fields = {
            "overall_soundscape": (
                "Wind through the mine. No speech, no voice-over, no narration -- "
                "nobody talks in this shot. ABSOLUTE NO-VOICE RULE: legacy details"
            )
        }
        P._ensure_soundscape(fields, dialogue="")
        self.assertEqual(
            "Wind through the mine. No speech, no voice-over, no narration -- nobody talks in this shot.",
            fields["overall_soundscape"],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
