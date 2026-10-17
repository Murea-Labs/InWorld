"""Pure-logic regression tests of splitting one input into 1–3 short clips."""

from __future__ import annotations

import sys
import unittest
from unittest.mock import patch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from murea_agent.director import agent_schema, narration
from murea_agent.guards import density_audit
from murea_agent.guards import action_sequence
from murea_agent.director import director  # noqa: E402
from murea_agent.director import prefetch as prefetch_mod  # noqa: E402
from murea_agent.guards import motion
from murea_agent import loop
from murea_agent import config  # noqa: E402
from murea_agent.providers.video import murea  # noqa: E402
from murea_agent.memory.world import World


def _motion(phase: str = "approach", relation: str = "distance") -> dict:
    return {
        "interaction_type": "none" if phase == relation == "none" else "locomotion",
        "phase": phase,
        "relation_change": relation,
        "path": "moves toward the target",
        "end_state": "stands outside the target",
        "actors": ["宇航员"],
        "target": "舱门",
    }


def _step(
    instruction: str,
    phase: str,
    relation: str,
    end_state: str,
    anchor: str,
) -> dict:
    return {
        "instruction": instruction,
        "phase": phase,
        "relation_change": relation,
        "end_state": end_state,
        "continuity_anchor": anchor,
    }


class ActionSequenceTest(unittest.TestCase):
    def test_first_step_takes_the_motion_plan_phase_not_the_relation_default(self) -> None:
        # Real failure: the plan was settle/distance (legal: come to rest a distance away) while the
        # director's first step was settle/none. Alignment rewrote the step to the default phase
        # for distance (approach), so the beat failed "first action step's phase differs".
        directive = {
            "action_sequence": {
                "goal": "Step past Mara and look down the lane",
                "steps": [
                    _step("Step past Mara and turn to face down the lane", "settle", "none", "", ""),
                ],
            },
        }
        sequence = narration._apply_action_sequence(
            directive, "Step past Mara", _motion("settle", "distance")
        )
        self.assertEqual(
            ("settle", "distance"), (sequence["steps"][0]["phase"], sequence["steps"][0]["relation_change"])
        )

    def test_dialogue_is_compiled_as_a_narrative_step(self) -> None:
        directive = {
            "action_sequence": {
                "goal": "进入内殿与神官交谈",
                "steps": [
                    _step(
                        "走近神官并建立接触",
                        "engage",
                        "contact",
                        "双方面对面",
                        "双方保持原有位置",
                    )
                ],
            },
        }
        directive["action_sequence"]["steps"][0]["kind"] = "dialogue"
        plan = action_sequence._reconcile_dialogue_motion(
            World(title="对白"),
            directive,
            motion._normalize_motion_plan(
                {
                    **_motion("engage", "contact"),
                    "actors": ["英雄"],
                    "target": "神官",
                    "start_state": "the actors stand apart",
                    "path": "the actors meet",
                    "contact_points": "",
                    "end_state": "the actors face each other",
                    "camera_side": "same side",
                    "framing": "medium",
                    "visual_evidence": ["contact_point"],
                    "invariants": ["identity"],
                }
            ),
            "与神官交谈",
        )
        sequence = narration._apply_action_sequence(
            directive,
            "交谈",
            plan,
            request_text="与神官交谈",
        )
        self.assertEqual("none", plan["relation_change"])
        self.assertEqual("dialogue", sequence["steps"][0]["kind"])
        self.assertEqual("none", sequence["steps"][0]["relation_change"])

    def test_object_impact_is_never_rewritten_as_dialogue_by_word_lists(self) -> None:
        step = _step(
            "角色将道具掷向目标",
            "engage",
            "contact",
            "道具命中目标",
            "同一道具和目标",
        )
        step.update(
            {
                "kind": "motion",
                "actors": ["角色"],
                "instruments": ["道具"],
                "targets": ["目标"],
                "events": [
                    {
                        "description": "道具沿连续路径飞向目标并命中",
                        "visible_result": "道具与目标在指定位置接触",
                        "estimated_seconds": 3,
                    }
                ],
            }
        )
        directive = {"action_sequence": {"steps": [step]}}
        plan = motion._normalize_motion_plan(
            {
                **_motion("engage", "contact"),
                "interaction_type": "impact",
                "instrument": "道具",
                "actors": ["角色"],
                "target": "目标",
                "start_state": "the actor holds the object away from the target",
                "path": "the object travels continuously from the hand to the target",
                "contact_points": "the object face meeting the target surface",
                "end_state": "the object visibly contacts the target",
                "camera_side": "same side",
                "framing": "wide",
                "visual_evidence": ["contact_point", "target_full_structure"],
                "invariants": ["one object and one target"],
            }
        )

        reconciled = action_sequence._reconcile_dialogue_motion(
            World(title="物体撞击"),
            directive,
            plan,
            "角色让道具飞出去，目标随后发生变化",
        )

        self.assertEqual("impact", reconciled["interaction_type"])
        self.assertEqual("contact", reconciled["relation_change"])
        self.assertEqual("motion", directive["action_sequence"]["steps"][0]["kind"])
        self.assertNotIn("_server_repairs", directive)
        self.assertNotIn("contact", action_sequence._GOAL_RELATION_PATTERNS)
        self.assertNotIn("contact", action_sequence._PRIMARY_ACTION_PATTERNS)

    def test_impact_configuration_is_structurally_compiled_as_contact(self) -> None:
        original_path = "the object travels from the hand and visibly strikes the target"
        directive = {
            "action_sequence": {
                "steps": [
                    {
                        "instruction": "角色掷出道具并使它命中目标",
                        "kind": "motion",
                        "phase": "transform",
                        "relation_change": "configuration",
                        "place": "场景",
                        "visible_subjects": ["角色", "道具", "目标"],
                        "actors": ["角色"],
                        "instruments": ["道具"],
                        "targets": ["目标"],
                        "event_ids": [1],
                    }
                ]
            }
        }
        raw_plan = motion._normalize_motion_plan(
            {
                **_motion("transform", "configuration"),
                "interaction_type": "impact",
                "instrument": "道具",
                "target": "目标",
                "actors": ["角色"],
                "path": original_path,
                "start_state": "the actor holds the object away from the target",
                "contact_points": "the object surface and the target surface",
                "end_state": "the object visibly contacts the target",
                "camera_side": "same side",
                "framing": "wide",
                "visual_evidence": ["contact_point", "target_full_structure"],
                "must_show": ["object and target"],
                "invariants": ["one object"],
            }
        )
        plan = motion._repair_motion_plan(raw_plan, directive)
        self.assertEqual("engage", plan["phase"])
        self.assertEqual("contact", plan["relation_change"])
        self.assertEqual(original_path, plan["path"])

        sequence = narration._apply_action_sequence(
            directive,
            "角色掷出道具",
            plan,
            request_text="让道具飞向目标",
        )
        self.assertEqual("contact", sequence["steps"][0]["relation_change"])
        self.assertIn("掷出道具", sequence["steps"][0]["instruction"])

        fallback = narration._current_segment_reply(
            sequence,
            plan,
            directive_reply="",
            player_actor="角色",
        )
        self.assertIn("道具", fallback)
        self.assertIn("命中目标", fallback)
        self.assertNotIn("与对方碰面", fallback)

    def test_semantic_event_budget_rejects_compressed_causal_chain(self) -> None:
        raw = {
            "steps": [
                {
                    "instruction": "完成整条因果链",
                    "place": "场景",
                    "visible_subjects": ["角色", "道具", "目标"],
                    "actors": ["角色"],
                    "instruments": ["道具"],
                    "targets": ["目标"],
                    "events": [
                        {"description": "释放道具", "visible_result": "道具离手", "estimated_seconds": 1.5},
                        {
                            "description": "道具移动并命中",
                            "visible_result": "道具命中目标",
                            "estimated_seconds": 2.5,
                        },
                        {
                            "description": "目标失稳倒下",
                            "visible_result": "目标倒地",
                            "estimated_seconds": 2.5,
                        },
                    ],
                    "kind": "motion",
                    "phase": "engage",
                    "relation_change": "contact",
                    "end_state": "",
                    "continuity_anchor": "",
                }
            ]
        }
        errors = action_sequence._action_sequence_semantic_errors(raw, 4)
        self.assertTrue(any("over the 4s budget" in error for error in errors))

    def test_semantic_event_budget_accepts_distinct_split_steps(self) -> None:
        def semantic_step(instruction: str, result: str, seconds: float) -> dict:
            step = _step(instruction, "engage", "contact", result, "同一道具和目标")
            step.update(
                {
                    "place": "场景",
                    "visible_subjects": ["角色", "道具", "目标"],
                    "actors": ["角色"],
                    "instruments": ["道具"],
                    "targets": ["目标"],
                    "events": [
                        {
                            "description": instruction,
                            "visible_result": result,
                            "estimated_seconds": seconds,
                        }
                    ],
                    "kind": "motion",
                }
            )
            return step

        raw = {
            "steps": [
                semantic_step("释放道具并沿路径飞行", "道具接近目标", 3.5),
                semantic_step("道具命中并发生形变", "接触位置发生可见变化", 3),
                semantic_step("目标失稳并落到支撑面", "目标稳定倒地", 3.5),
            ]
        }
        self.assertEqual([], action_sequence._action_sequence_semantic_errors(raw, 4))

    def test_master_events_must_be_assigned_once_in_order(self) -> None:
        raw = {
            "events": [
                {"id": 1, "description": "动作一", "visible_result": "结果一", "estimated_seconds": 2},
                {"id": 2, "description": "动作二", "visible_result": "结果二", "estimated_seconds": 3},
            ],
            "steps": [
                {
                    "instruction": "把两个事件塞进一段",
                    "place": "场景",
                    "visible_subjects": [],
                    "actors": [],
                    "instruments": [],
                    "targets": [],
                    "event_ids": [1, 2],
                    "kind": "motion",
                    "phase": "engage",
                    "relation_change": "contact",
                }
            ],
        }
        errors = action_sequence._action_sequence_semantic_errors(raw, 4)
        self.assertTrue(any("exactly one video step" in error for error in errors))

    def test_saturated_single_event_is_reviewed_by_semantic_gm(self) -> None:
        raw = {
            "events": [
                {
                    "id": 1,
                    "description": "一个过长的因果事件描述，把多个不同的可见结果全部压在一起",
                    "visible_result": "最终结果",
                    "estimated_seconds": 4,
                }
            ],
            "steps": [
                {
                    "instruction": "完成全部结果",
                    "place": "场景",
                    "visible_subjects": [],
                    "actors": [],
                    "instruments": [],
                    "targets": [],
                    "event_ids": [1],
                    "kind": "motion",
                    "phase": "engage",
                    "relation_change": "contact",
                }
            ],
        }
        reviewed = {
            "required_segments": 3,
            "reason": "存在三个不同的可见结果",
            "events": [
                {"description": "动作一", "visible_result": "结果一", "estimated_seconds": 3},
                {"description": "动作二", "visible_result": "结果二", "estimated_seconds": 3},
                {"description": "动作三", "visible_result": "结果三", "estimated_seconds": 3},
            ],
        }
        with patch.object(loop.llm, "chat", return_value=reviewed) as chat:
            result = density_audit._audit_single_event_density(raw, "完整用户目标", 4)
        self.assertEqual(3, result["required_segments"])
        chat.assert_called_once()

    def test_completed_approach_advances_to_destination_crossing(self) -> None:
        from murea_agent.memory.world import Node

        world = World(title="阶段状态", here="神庙之巅")
        world.add_place("神庙之巅", "top.png")
        world.add_place("内殿", "hall.png")
        previous = Node(
            id="n1",
            parent=None,
            video_id="v1",
            clip="v1.mp4",
            place="神庙之巅",
            beat="走到内殿入口前并停稳",
            mode="continue",
            directive={
                "motion_plan": {
                    "interaction_type": "locomotion",
                    "phase": "approach",
                    "relation_change": "distance",
                    "target": "内殿入口",
                    "start_state": "standing",
                    "path": "walks to the entrance",
                    "end_state": "stops outside",
                    "camera_side": "terrace",
                    "framing": "wide",
                    "actors": ["英雄"],
                    "visual_evidence": ["actor_full_body", "target_full_structure", "support_surface"],
                    "invariants": ["temple"],
                }
            },
        )
        world.add_node(previous)
        directive = {
            # The model's structured place field is authoritative; the server no longer extracts
            # places from the user's sentence.
            "place": "内殿",
            "action_sequence": {
                "steps": [
                    _step("再次靠近入口", "approach", "distance", "", ""),
                    _step("进入内殿", "transform", "containment", "人在内殿", "敞开的入口"),
                ]
            },
        }
        directive["action_sequence"]["steps"][0]["place"] = "神庙之巅"
        directive["action_sequence"]["steps"][1]["place"] = "内殿"
        plan = motion._normalize_motion_plan(
            {
                **_motion("approach", "distance"),
                "target": "神官",
                "start_state": "at the entrance",
                "path": "moves toward the priest",
                "end_state": "stops outside",
                "camera_side": "terrace",
                "framing": "wide",
                "visual_evidence": ["actor_full_body", "target_full_structure", "support_surface"],
                "invariants": ["identity"],
            }
        )
        advanced = action_sequence._advance_completed_transition(
            world,
            directive,
            plan,
            "进入内殿与神官交谈",
        )
        self.assertEqual("containment", advanced["relation_change"])
        self.assertEqual("内殿", directive["place"])
        self.assertEqual("containment", directive["action_sequence"]["steps"][0]["relation_change"])

    def test_three_segments_only_execute_first_and_queue_the_rest(self) -> None:
        raw = {
            "goal": "里昂进入车内",
            "budget_reason": "4 秒不能同时完成接近、开门和跨越边界",
            "steps": [
                _step("走到车门外并站稳", "approach", "distance", "里昂站在车门外", "里昂面向右侧车门"),
                _step(
                    "握住把手并把车门完全打开",
                    "transform",
                    "configuration",
                    "车门完全打开，里昂仍在车外",
                    "敞开的车门和右手",
                ),
                _step(
                    "从敞开的门框进入车内",
                    "transform",
                    "containment",
                    "里昂身体完全位于车内",
                    "敞开门框和向右运动方向",
                ),
            ],
        }
        directive = {"action_sequence": raw, "beat": "把所有动作挤在一起", "continue_with": "错误的旧值"}
        sequence = narration._apply_action_sequence(directive, directive["beat"], _motion())

        self.assertEqual(
            "moves toward the target. End state: stands outside the target.",
            directive["beat"],
        )
        self.assertEqual(3, len(sequence["steps"]))
        # continue_with holds only **the next step**. It used to join the remaining two into one
        # sentence, and given "A, then B" the director replanned the whole goal and finished steps
        # reappeared in the queue.
        self.assertIn("握住把手并把车门完全打开", directive["continue_with"])
        self.assertNotIn("里昂进入车内", directive["continue_with"])
        self.assertNotIn("从敞开的门框进入车内", directive["continue_with"])
        self.assertNotIn("走到车门外并站稳", directive["continue_with"])
        # Later steps are pinned on plan and travel with the node: that is "done up to step N".
        self.assertEqual(1, len(directive["plan"]))
        self.assertIn("从敞开的门框进入车内", directive["plan"][0])

    def test_continuation_contains_only_its_own_distinct_action(self) -> None:
        """Continuation must not carry the full goal back, or the director redoes the first segment already generated."""
        directive = {
            "action_sequence": {
                "steps": [
                    _step("走到舱门外并站稳", "approach", "distance", "", ""),
                    _step(
                        "双手把舱门完全推开", "transform", "configuration", "舱门完全打开", "人物仍在舱门外"
                    ),
                    _step(
                        "穿过敞开的舱门进入机械舱",
                        "transform",
                        "containment",
                        "人物完全位于机械舱内",
                        "人物与舱门的运动方向不变",
                    ),
                ]
            },
            "beat": "走向舱门",
        }
        narration._apply_action_sequence(
            directive,
            directive["beat"],
            _motion(),
            request_text="走到舱门前，推开舱门并进入机械舱",
        )
        self.assertEqual("接着双手把舱门完全推开", directive["continue_with"])
        self.assertEqual(
            ["接着穿过敞开的舱门进入机械舱"],
            directive["plan"],
        )

    def test_continuation_follows_the_carried_plan_not_a_fresh_replan(self) -> None:
        """Real failure: "open the door and walk out" spent four continuations on the same door.

        On continuation beats the director replanned the whole goal and the finished step
        reappeared. Now the remaining queue pinned on the node by the previous beat is authoritative
        and only shrinks.
        """
        raw = {
            "goal": "里昂进入车内",
            "budget_reason": "4 秒装不下",
            "steps": [
                _step("走到车门外并站稳", "approach", "distance", "里昂站在车门外", "里昂面向右侧车门"),
                # The director put the finished "open the door" back again in this beat
                _step(
                    "握住把手并把车门完全打开",
                    "transform",
                    "configuration",
                    "车门完全打开",
                    "敞开的车门和右手",
                ),
            ],
        }
        directive = {"action_sequence": raw, "beat": "b", "continue_with": ""}
        narration._apply_action_sequence(
            directive,
            directive["beat"],
            _motion(),
            is_continuation=True,
            carried_plan=["接着里昂进入车内：从敞开的门框进入车内"],
        )

        self.assertEqual("接着里昂进入车内：从敞开的门框进入车内", directive["continue_with"])
        self.assertEqual([], directive["plan"], "队列必须变短，不能原地打转")

    def test_the_plan_queue_drains_to_empty(self) -> None:
        """After the last step of a plan, continue_with must be empty, or the frontend keeps continuing."""
        directive = {
            "action_sequence": {
                "goal": "g",
                "steps": [_step("最后一步", "approach", "distance", "到位", "锚点")],
            },
            "beat": "b",
            "continue_with": "旧值",
        }
        narration._apply_action_sequence(directive, "b", _motion(), is_continuation=True, carried_plan=[])
        self.assertEqual("", directive["continue_with"])
        self.assertEqual([], directive["plan"])

    def test_audited_missing_event_keeps_repaired_substeps_before_existing_queue(self) -> None:
        directive = {
            "_repair_current_event": True,
            "action_sequence": {
                "steps": [
                    _step("走到近处", "approach", "distance", "到位", "原地"),
                    _step("打开束缚", "transform", "configuration", "束缚张开", "双手保持位置"),
                ]
            },
        }
        sequence = narration._apply_action_sequence(
            directive, "当前动作", _motion(), is_continuation=True, carried_plan=["接着走向出口"]
        )
        self.assertEqual(2, len(sequence["steps"]))
        self.assertIn("打开束缚", directive["continue_with"])
        self.assertEqual(["接着走向出口"], directive["plan"])
        self.assertNotIn("_repair_current_event", directive)

    def test_single_segment_clears_stale_continuation(self) -> None:
        directive = {
            "beat": "旧 beat",
            "continue_with": "不该继续",
            "action_sequence": {
                "goal": "抬头看向钟表",
                "budget_reason": "单一视线动作可在 4 秒内完成",
                "steps": [
                    _step("抬头看向墙上的钟表", "none", "none", "目光停在钟表上", "同一视线目标"),
                ],
            },
        }
        narration._apply_action_sequence(directive, directive["beat"], _motion("none", "none"))
        self.assertEqual("抬头看向墙上的钟表", directive["beat"])
        self.assertEqual("", directive["continue_with"])

    def test_gm_partial_boundary_step_is_not_rewritten_by_service(self) -> None:
        directive = {
            "beat": "靠近车门",
            "action_sequence": {
                "goal": "进入警车并检查内部",
                "budget_reason": "需要两段",
                "steps": [
                    _step("靠近车门", "approach", "distance", "站在车外", "敞开的车门"),
                    _step(
                        "探身检查车内",
                        "transform",
                        "containment",
                        "upper body leans into the car",
                        "door frame",
                    ),
                ],
            },
        }
        sequence = narration._apply_action_sequence(
            directive, directive["beat"], _motion(), request_text="进入警车"
        )
        self.assertEqual("进入警车", sequence["goal"])
        self.assertEqual("探身检查车内", sequence["steps"][-1]["instruction"])
        self.assertEqual("upper body leans into the car", sequence["steps"][-1]["end_state"])
        self.assertIn("探身检查车内", directive["continue_with"])

    def test_service_does_not_invent_a_relation_missing_from_gm_sequence(self) -> None:
        directive = {
            "beat": "走向警车",
            "action_sequence": {
                "goal": "模型错误地缩短了目标",
                "budget_reason": "4 秒只能接近",
                "steps": [
                    _step("走到车门外", "approach", "distance", "站在车门外", "敞开的车门"),
                ],
            },
        }
        sequence = narration._apply_action_sequence(
            directive, directive["beat"], _motion(), request_text="进入右侧警车"
        )
        self.assertEqual(
            ["distance"],
            [step["relation_change"] for step in sequence["steps"]],
        )
        self.assertEqual("", directive["continue_with"])

    def test_gm_semantic_open_then_enter_sequence_is_kept(self) -> None:
        sequence = {
            "goal": "打开舱门并进入舱内",
            "budget_reason": "需要多个阶段",
            "steps": [
                _step("走到舱门前", "approach", "distance", "站在舱门外", "舱门轮廓"),
                _step("打开舱门", "transform", "configuration", "舱门已经打开", "人物仍在舱门外"),
                _step("进入舱内", "transform", "containment", "人物完全位于舱内", "同一道敞开的舱门"),
            ],
        }
        action_sequence._ensure_goal_relations(sequence)
        self.assertEqual(
            ["distance", "configuration", "containment"],
            [step["relation_change"] for step in sequence["steps"]],
        )

    def test_service_does_not_insert_text_matched_relation_in_goal_order(self) -> None:
        sequence = {
            "goal": "打开舱门并进入舱内",
            "budget_reason": "模型漏掉了第一阶段",
            "steps": [
                _step("进入舱内", "transform", "containment", "身体完全在舱内", "舱门轮廓"),
            ],
        }
        action_sequence._ensure_goal_relations(sequence)
        self.assertEqual(
            ["containment"],
            [step["relation_change"] for step in sequence["steps"]],
        )

    def test_existing_opening_is_not_mistaken_for_opening_action(self) -> None:
        path = "the actor passes through the opening of the hatch into the cabin"
        self.assertIsNone(action_sequence._CONFIGURATION_ACTION.search(path))
        self.assertIsNotNone(action_sequence._BOUNDARY_CROSSING_ACTION.search(path))

    def test_first_step_must_match_current_motion_plan(self) -> None:
        sequence = action_sequence._normalize_action_sequence(
            {
                "goal": "进入房间",
                "budget_reason": "需要先接近再跨越",
                "steps": [
                    _step("直接穿过门框", "transform", "containment", "人在门内", "敞开的门框"),
                ],
            }
        )
        errors = action_sequence._action_sequence_errors(sequence, _motion())
        self.assertTrue(any("phase" in error for error in errors))
        self.assertTrue(any("relation_change" in error for error in errors))

    def test_apply_action_sequence_repairs_first_step_mismatch(self) -> None:
        directive = {
            "beat": "进入房间",
            "action_sequence": {
                "goal": "进入房间",
                "steps": [
                    _step(
                        "穿过门框进入房间",
                        "transform",
                        "containment",
                        "人在门内",
                        "敞开的门框",
                    )
                ],
            },
        }
        # This beat's motion_plan says "approach first", but the head of the queue is still the old
        # crossing relation.
        sequence = narration._apply_action_sequence(
            directive,
            directive["beat"],
            _motion("approach", "distance"),
            request_text="进入房间",
        )
        self.assertEqual("approach", sequence["steps"][0]["phase"])
        self.assertEqual("distance", sequence["steps"][0]["relation_change"])

    def test_disengage_distance_step_is_not_rejected(self) -> None:
        directive = {
            "beat": "与目标拉开距离",
            "action_sequence": {
                "goal": "先靠近目标，再退后拉开距离",
                "steps": [
                    _step("走到目标前停下", "approach", "distance", "停在目标前", "目标保持原位"),
                    _step(
                        "退后与目标拉开距离", "disengage", "distance", "双方之间出现明显间距", "目标保持原位"
                    ),
                ],
            },
        }
        sequence = narration._apply_action_sequence(
            directive,
            directive["beat"],
            _motion("approach", "distance"),
            request_text=directive["action_sequence"]["goal"],
        )
        self.assertEqual("disengage", sequence["steps"][1]["phase"])
        self.assertEqual(
            [],
            action_sequence._action_sequence_errors(
                sequence,
                _motion("approach", "distance"),
            ),
        )

    def test_segment_budget_matches_video_backend_limits(self) -> None:
        world = World(id="wbudget", title="时长", setting="测试")
        with (
            patch.object(murea, "backend", return_value="murea"),
            patch.object(config, "VIDEO_MAX_SECONDS", 15),
        ):
            self.assertEqual(4, director._segment_seconds(world, 2))
            self.assertEqual(8, director._segment_seconds(world, 8))
            self.assertEqual(15, director._segment_seconds(world, 99))

    def test_the_segment_budget_follows_the_config_within_the_model_limit(self) -> None:
        world = World(id="wfast", title="时长", setting="测试")
        with patch.object(murea, "backend", return_value="fast"):
            self.assertEqual(10, director._segment_seconds(world))  # default video.max_seconds
            self.assertEqual(6, director._segment_floor(world))
            self.assertEqual(8, director._segment_seconds(world, 8))
            self.assertEqual(4, director._segment_seconds(world, 2))
            self.assertEqual(20, director._segment_seconds(world, 30))
            with patch.object(config, "VIDEO_MAX_SECONDS", 20):
                self.assertEqual(20, director._segment_seconds(world))
        with (
            patch.object(murea, "backend", return_value="murea"),
            patch.object(config, "VIDEO_MAX_SECONDS", 20),
        ):
            self.assertEqual(15, director._segment_seconds(world))  # the full model takes 4-15 s

    def test_prefetch_cache_key_includes_segment_budget(self) -> None:
        world = World(id="wkey", title="缓存", setting="测试")
        self.assertNotEqual(
            prefetch_mod._prefetch_key(world, "向前走", 4),
            prefetch_mod._prefetch_key(world, "向前走", 8),
        )

    def test_missing_observation_is_not_inserted_by_keyword_budget(self) -> None:
        """One observation event must not be split into duplicate clips because it matches several observation words."""
        goal = "站在黄铜钟楼顶端，看着远处沉睡的齿轮巨兽缓缓睁开赤红的双眼"
        directive = {
            "action_sequence": {
                "goal": goal,
                "steps": [
                    _step(
                        "站在黄铜钟楼顶端，看着远处的齿轮巨兽",
                        "none",
                        "none",
                        "",
                        "",
                    )
                ],
            },
            "beat": "站在钟楼顶端观察远处的巨兽",
        }
        sequence = narration._apply_action_sequence(
            directive,
            directive["beat"],
            _motion("none", "none"),
            request_text=goal,
            segment_seconds=4,
        )
        self.assertEqual(1, len(sequence["steps"]))
        self.assertEqual("", directive["continue_with"])

    def test_missing_physical_relation_is_left_for_gm_not_text_matching(self) -> None:
        goal = "走到警车前，观察车内，进入警车"
        directive = {
            "action_sequence": {
                "goal": goal,
                "steps": [_step("走到警车前", "approach", "distance", "", "")],
            },
            "beat": "走向警车",
        }
        sequence = narration._apply_action_sequence(
            directive,
            directive["beat"],
            _motion(),
            request_text=goal,
            segment_seconds=4,
        )
        self.assertEqual(
            ["distance"],
            [step["relation_change"] for step in sequence["steps"]],
        )
        self.assertEqual("", directive["continue_with"])
        self.assertNotIn("停稳后观察", directive["continue_with"])

    def test_gm_schema_requires_sequence_but_not_freeform_continuation(self) -> None:
        required = agent_schema.GM_SCHEMA["required"]
        self.assertIn("action_sequence", required)
        self.assertNotIn("continue_with", required)
        self.assertNotIn("continue_with", agent_schema.GM_SCHEMA["properties"])

    def test_intermediate_distance_does_not_announce_or_apply_future_state(self) -> None:
        sequence = {
            "goal": "进入右侧警车",
            "budget_reason": "需要先接近再进入",
            "steps": [
                _step("走到车门外", "approach", "distance", "站在车门外", "敞开的车门"),
                _step("进入车内", "transform", "containment", "身体完全在车内", "敞开的车门"),
            ],
        }
        directive = {
            "reply": "里昂探头看到车内有血迹。",
            "state_updates": {
                "subjects": [
                    {
                        "name": "里昂",
                        "mood": "保持警惕",
                        "goal": "检查车内血迹",
                        "knowledge_add": ["知道车内有血迹"],
                        "knowledge_remove": ["不知道车内情况"],
                    }
                ],
                "places": [
                    {
                        "name": "街道",
                        "weather": "小雨",
                        "condition": "警车门已经打开",
                        "props_add": ["车内血迹"],
                        "props_remove": ["关闭的车门"],
                    }
                ],
            },
        }
        narration._sanitize_intermediate_directive(directive, sequence, _motion())

        self.assertEqual("宇航员走到车门外。", directive["reply"])
        self.assertNotIn("稳稳停下", directive["reply"])
        self.assertNotIn("分成", directive["reply"])
        self.assertNotIn("随后", directive["reply"])
        self.assertNotIn("目标", directive["reply"])
        self.assertNotIn("交互", directive["reply"])
        self.assertNotIn("血迹", directive["reply"])
        subject = directive["state_updates"]["subjects"][0]
        self.assertEqual("保持警惕", subject["mood"])
        self.assertNotIn("goal", subject)
        self.assertNotIn("knowledge_add", subject)
        self.assertNotIn("knowledge_remove", subject)
        place = directive["state_updates"]["places"][0]
        self.assertEqual("小雨", place["weather"])
        self.assertNotIn("condition", place)
        self.assertNotIn("props_add", place)
        self.assertNotIn("props_remove", place)

    def test_distance_caption_does_not_force_stop_after_retreat(self) -> None:
        """Subtitles of retreating or gaining distance must not be rewritten by the fallback into stopping."""
        sequence = {
            "goal": "撤离燃烧的巨兽并拉开安全距离",
            "steps": [
                _step(
                    "你迅速向后撤离雪地，与燃烧的巨兽拉开安全距离。",
                    "approach",
                    "distance",
                    "与巨兽保持安全距离",
                    "巨兽仍在原处",
                )
            ],
        }
        motion = {**_motion("approach", "distance"), "actors": ["你"]}

        result = narration._current_segment_reply(
            sequence,
            motion,
            "",
            player_actor="你",
        )

        self.assertEqual(
            "你迅速向后撤离雪地，与燃烧的巨兽拉开安全距离。",
            result,
        )
        self.assertNotIn("稳稳停下", result)

    def test_intermediate_transform_keeps_visible_current_state_only(self) -> None:
        sequence = {
            "goal": "打开舱门并进入舱内",
            "budget_reason": "构型变化和跨越必须分开",
            "steps": [
                _step("打开舱门", "transform", "configuration", "舱门完全打开", "敞开的舱门"),
                _step("进入舱内", "transform", "containment", "身体完全在舱内", "敞开的舱门"),
            ],
        }
        directive = {
            "reply": "舱门打开后，他已经进入舱内。",
            "state_updates": {
                "subjects": [
                    {
                        "name": "宇航员",
                        "knowledge_add": ["知道舱内情况"],
                    }
                ],
                "places": [{"name": "气闸", "condition": "舱门已经打开"}],
            },
        }
        motion = _motion("transform", "configuration")
        narration._sanitize_intermediate_directive(directive, sequence, motion)

        self.assertEqual("宇航员打开舱门。", directive["reply"])
        self.assertNotIn("构型变化", directive["reply"])
        self.assertNotIn("已经进入", directive["reply"])
        self.assertNotIn("knowledge_add", directive["state_updates"]["subjects"][0])
        self.assertEqual(
            "舱门已经打开",
            directive["state_updates"]["places"][0]["condition"],
        )

    def test_single_segment_keeps_original_reply_and_state(self) -> None:
        sequence = {
            "goal": "拿起钥匙",
            "budget_reason": "已经站在钥匙旁，一段足够",
            "steps": [
                _step("拿起钥匙", "transform", "possession", "钥匙在手中", "右手中的钥匙"),
            ],
        }
        directive = {
            "reply": "他拿起了钥匙。",
            "state_updates": {
                "subjects": [{"name": "阿明", "knowledge_add": ["钥匙在手中"]}],
                "places": [],
            },
        }
        original = repr(directive)
        narration._sanitize_intermediate_directive(directive, sequence, _motion("transform", "possession"))
        self.assertEqual(original, repr(directive))

    def test_planning_instruction_falls_back_to_natural_caption(self) -> None:
        sequence = {
            "goal": "进入警车",
            "steps": [
                _step(
                    "让人物整个身体只通过已经敞开的边界，并停在目标空间内",
                    "transform",
                    "containment",
                    "身体完全在车内",
                    "敞开的车门",
                ),
                _step("检查车内", "none", "none", "看到车内", "车内光线"),
            ],
        }
        motion = {
            **_motion("transform", "containment"),
            "actors": ["里昂"],
            "target": "警车",
        }
        directive = {"reply": "这个动作会分成 2 段完成；随后继续完成“进入警车”。"}

        narration._sanitize_intermediate_directive(directive, sequence, motion)

        self.assertEqual("里昂穿过敞开的入口，来到警车。", directive["reply"])
        self.assertNotIn("分成", directive["reply"])
        self.assertNotIn("随后", directive["reply"])
        self.assertNotIn("目标", directive["reply"])

    def test_natural_reply_is_preferred_and_player_uses_second_person(self) -> None:
        """The current segment prefers the full narrative and is no longer overwritten by the short instruction."""
        sequence = {
            "goal": "走到老城区主街",
            "steps": [
                _step("走到跟前，停稳", "approach", "distance", "", ""),
                _step("继续走到主街", "approach", "distance", "站在主街入口", "湿漉漉的巷口"),
            ],
        }
        motion = {**_motion("approach", "distance"), "actors": ["隐秘旅人"]}
        reply = "隐秘旅人沿着湿漉漉的狭窄巷弄继续向前，转过拐角，在通往主街的入口前停稳。"

        result = narration._current_segment_reply(
            sequence,
            motion,
            reply,
            player_actor="隐秘旅人",
        )

        self.assertTrue(result.startswith("你"))
        self.assertIn("湿漉漉", result)
        self.assertGreater(len(result), len(sequence["steps"][0]["instruction"]) * 2)

    def test_brief_player_reply_gets_current_path_context(self) -> None:
        """Short subtitles from old caches also get the current path, without going back to the third person."""
        sequence = {
            "goal": "走到老城区主街",
            "steps": [_step("走到跟前，停稳", "approach", "distance", "", "")],
        }
        motion = {
            **_motion("approach", "distance"),
            "actors": ["隐秘旅人"],
            "target": "老城区主街",
        }

        result = narration._current_segment_reply(
            sequence,
            motion,
            "隐秘旅人走到跟前，停稳。",
            player_actor="隐秘旅人",
        )

        self.assertEqual(
            "你沿着眼前的通路向前，来到老城区主街前，稳稳停下。",
            result,
        )

    def test_reply_with_future_relation_is_rejected(self) -> None:
        """The current segment must not write later relations such as opening or entering into its subtitle."""
        sequence = {
            "goal": "进入警车",
            "steps": [
                _step("走到车门外并停稳", "approach", "distance", "", ""),
                _step("打开车门", "transform", "configuration", "车门全开", "敞开的车门"),
                _step("进入车内", "transform", "containment", "身体在车内", "车内座椅"),
            ],
        }
        motion = {**_motion("approach", "distance"), "actors": ["里昂"]}
        reply = "你走到车门前，打开车门并进入车内。"

        result = narration._current_segment_reply(
            sequence,
            motion,
            reply,
            player_actor="里昂",
        )

        self.assertEqual("你沿着眼前的通路向前，来到舱门前，稳稳停下。", result)
        self.assertNotIn("打开", result)
        self.assertNotIn("进入", result)

    def test_split_notice_is_replaced_for_single_segment_too(self) -> None:
        sequence = {
            "goal": "观察车内",
            "steps": [_step("里昂透过车窗观察车内", "none", "none", "", "")],
        }
        motion = {**_motion("none", "none"), "actors": ["里昂"], "target": "警车"}
        directive = {"reply": "这个动作会分成 2 段完成；随后继续完成“观察车内”。"}

        self.assertTrue(narration._has_split_planning_reply(directive["reply"]))
        directive["reply"] = narration._current_segment_reply(sequence, motion)

        self.assertEqual("里昂透过车窗观察车内。", directive["reply"])
        self.assertNotIn("分成", directive["reply"])
        self.assertNotIn("随后", directive["reply"])


if __name__ == "__main__":
    unittest.main()
