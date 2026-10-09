"""Pure-logic regression tests of the generic physical relation plan.

No model calls; covers movement, boundaries, taking, support, two-person contact, audit retries and
prompt injection.
"""

from __future__ import annotations

import sys
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from murea_agent.director import agent_schema
from murea_agent.guards import density_audit
from murea_agent.shot import video_prompt
from murea_agent.guards import motion
from murea_agent import loop
from murea_agent.memory.world import Node, World


def _plan(
    interaction_type: str,
    phase: str,
    relation_change: str,
    *,
    target: str = "the target",
) -> dict:
    return {
        "interaction_type": interaction_type,
        "phase": phase,
        "relation_change": relation_change,
        "actors": ["Amin"],
        "target": target,
        "start_state": "Amin stands one step away with both feet on the floor",
        "path": "Amin performs only the named movement toward the target",
        "contact_points": (
            "his right hand meets the target while both feet remain supported"
            if relation_change not in {"none", "distance"}
            else ""
        ),
        "end_state": "Amin finishes in the stated end relation to the target",
        "camera_side": "outside the target on the same side as the actor",
        "framing": "medium-wide",
        "visual_evidence": [
            "actor_full_body",
            "actor_hands",
            "actor_feet",
            "target_full_structure",
            "contact_point",
            "support_surface",
            "boundary_opening",
            "both_actors_contact",
        ],
        "must_show": ["Amin's full body", "the contact point", "the target structure"],
        "invariants": [
            "all rigid objects keep one stable shape and scale",
            "the number and identity of every visible object remain constant",
        ],
    }


class MotionPlanTest(unittest.TestCase):
    def test_world_snapshot_carries_previous_distance_anchor(self) -> None:
        world = World(title="t")
        node = Node(
            id="n1",
            parent=None,
            video_id="v",
            clip="c.mp4",
            place="浮桥市集",
            beat="沿浮桥走到泊位",
            mode="continue",
            directive={
                "motion_plan": {
                    "relation_change": "distance",
                    "target": "巨轮靠岸的泊位边缘",
                    "path": "沿右侧木板桥经过蓝色遮棚",
                    "end_state": "双脚停在泊位边缘",
                }
            },
        )
        world.nodes[node.id] = asdict(node)
        world.head = node.id
        anchor = world.snapshot()["previous_motion_anchor"]
        self.assertEqual("巨轮靠岸的泊位边缘", anchor["target"])
        self.assertIn("蓝色遮棚", anchor["path"])
        self.assertEqual("双脚停在泊位边缘", anchor["end_state"])

    def test_every_interaction_family_uses_the_same_relation_rules(self) -> None:
        cases = (
            _plan("locomotion", "approach", "distance"),
            _plan("boundary", "transform", "containment", target="the open doorway"),
            _plan("manipulation", "transform", "possession", target="the cup"),
            _plan("manipulation", "transform", "configuration", target="the drawer"),
            _plan("support", "transform", "support", target="the chair seat"),
            _plan("contact", "engage", "contact", target="Xiaohong's right hand"),
        )
        for raw in cases:
            with self.subTest(
                interaction_type=raw["interaction_type"],
                relation_change=raw["relation_change"],
            ):
                plan = motion._normalize_motion_plan(raw)
                self.assertEqual([], motion._motion_plan_errors(plan))

    def test_phase_cannot_hide_a_second_relation_change(self) -> None:
        plan = motion._normalize_motion_plan(
            _plan("boundary", "approach", "containment", target="the doorway")
        )
        errors = motion._motion_plan_errors(plan)
        self.assertTrue(any("phase=approach" in error for error in errors))

    def test_disengage_distance_is_a_valid_separation(self) -> None:
        plan = motion._normalize_motion_plan(_plan("locomotion", "disengage", "distance"))
        self.assertEqual([], motion._motion_plan_errors(plan))
        fields = video_prompt._lock_motion_plan(
            {"detailed_description": "[Shot 1] Amin steps away."},
            plan,
        )
        self.assertIn("gap widens continuously", fields["detailed_description"])

    def test_invalid_phase_is_repaired_before_generation(self) -> None:
        raw = _plan("manipulation", "engage", "configuration", target="the door")
        directive = {}
        repaired = motion._repair_motion_plan(
            motion._normalize_motion_plan(raw),
            directive,
        )
        self.assertEqual("transform", repaired["phase"])
        self.assertEqual([], motion._motion_plan_errors(repaired))
        self.assertIn("phase and relation disagree", directive["_server_repairs"][0])

    def test_contact_relation_requires_visible_contact_points(self) -> None:
        raw = _plan("contact", "engage", "contact", target="Xiaohong")
        raw["contact_points"] = ""
        errors = motion._motion_plan_errors(motion._normalize_motion_plan(raw))
        self.assertTrue(any("contact_points" in error for error in errors))

    def test_freeform_visibility_details_are_optional_when_evidence_is_structured(self) -> None:
        raw = _plan("support", "transform", "support", target="the stair tread")
        raw["must_show"] = []
        errors = motion._motion_plan_errors(motion._normalize_motion_plan(raw))
        self.assertEqual([], errors)

    def test_full_body_motion_promotes_cropped_framing(self) -> None:
        raw = _plan("locomotion", "approach", "distance")
        raw["framing"] = "medium"
        plan = motion._normalize_motion_plan(raw)
        self.assertEqual("wide", plan["framing"])
        self.assertEqual([], motion._motion_plan_errors(plan))

    def test_camera_side_drops_conflicting_shot_scale(self) -> None:
        raw = _plan("locomotion", "approach", "distance")
        raw["camera_side"] = "medium shot from the street side, facing screen right"
        plan = motion._normalize_motion_plan(raw)
        self.assertEqual("from the street side, facing screen right", plan["camera_side"])

    def test_full_body_motion_promotes_medium_wide_to_wide(self) -> None:
        for relation, interaction, phase in (
            ("distance", "locomotion", "approach"),
            ("containment", "boundary", "transform"),
            ("support", "support", "transform"),
        ):
            with self.subTest(relation=relation):
                plan = motion._normalize_motion_plan(_plan(interaction, phase, relation))
                self.assertEqual("wide", plan["framing"])

    def test_relation_adds_its_required_visual_evidence(self) -> None:
        raw = _plan("boundary", "transform", "containment", target="the doorway")
        raw["visual_evidence"].remove("boundary_opening")
        plan = motion._normalize_motion_plan(raw)
        self.assertIn("boundary_opening", plan["visual_evidence"])
        self.assertEqual([], motion._motion_plan_errors(plan))

    def test_motion_clause_preserves_geometry_and_object_identity(self) -> None:
        raw = _plan("support", "transform", "support", target="the chair seat")
        fields = video_prompt._lock_motion_plan(
            {
                "detailed_description": "[Shot 1] Amin bends his knees.",
            },
            raw,
        )
        detail = fields["detailed_description"]
        self.assertIn("Physical continuity", detail)
        self.assertIn("the chair seat", detail)
        self.assertIn("both feet remain supported", detail)
        self.assertIn("Amin's full body", detail)
        self.assertIn("stable shape and scale", detail)
        self.assertIn("number and identity", detail)
        self.assertIn("center of mass", detail)
        self.assertIn("from the first frame through the final frame", detail)

    def test_distance_clause_requires_visible_progress_to_endpoint(self) -> None:
        raw = _plan(
            "locomotion",
            "approach",
            "distance",
            target="the berth edge beside the ship",
        )
        fields = video_prompt._lock_motion_plan(
            {"detailed_description": "[Shot 1] Amin walks along the pontoon."},
            raw,
        )
        detail = fields["detailed_description"]
        self.assertIn("covering a meaningful stretch", detail)
        self.assertIn("Reach the exact end state at the endpoint", detail)
        self.assertIn("one or two steps", detail)

    def test_rewriter_audit_retries_then_strips_internal_fields(self) -> None:
        plan = _plan("boundary", "transform", "containment", target="the open doorway")
        base = {
            "summary": "[reference_generation] <Subject 1> crosses the doorway.",
            "retention_analysis": ("<Subject 1> (appears in [Shot 1] (full body)): fully_preserved - same"),
            "detailed_description": "[Shot 1] <Subject 1> crosses the open doorway.",
            "overall_soundscape": "N/A",
            "non_diegetic_music": "N/A",
        }
        wrong = dict(
            base,
            motion_phase="approach",
            motion_relation_change="distance",
            motion_target="the open doorway",
            motion_framing="medium",
        )
        right = dict(
            base,
            motion_phase="transform",
            motion_relation_change="containment",
            motion_target="the open doorway",
            motion_framing="wide",
        )
        with patch.object(loop.llm, "chat", side_effect=[wrong, right]) as chat:
            out = video_prompt._rewrite(
                pictures=[],
                place="corridor",
                beat="Amin crosses the doorway",
                dialogue="",
                continues=True,
                previous_beat="Amin reaches the door",
                setting="office",
                visual_style="cinematic",
                previous_shot="at the door",
                motion_plan=plan,
            )
        self.assertEqual(2, chat.call_count)
        self.assertNotIn("motion_phase", out)
        self.assertEqual(base["detailed_description"], out["detailed_description"])
        self.assertIn("motion_correction", chat.call_args_list[1].kwargs["user"])

    def test_previous_ending_drops_dynamic_tail_but_keeps_physical_end(self) -> None:
        node = Node(
            id="n1",
            parent=None,
            video_id="",
            clip="1.mp4",
            place="走廊",
            beat="阿明走到门边",
            mode="continue",
            six_section={
                "detailed_description": (
                    "[Shot 1] Amin stops outside the doorway with both feet on the floor. "
                    "Physical continuity — By the end, his torso remains outside. "
                    "Dynamic world state — preserve these visible facts exactly: 天气=下雨."
                )
            },
        )
        ending = video_prompt._ending_of(node)
        self.assertIn("torso remains outside", ending)
        self.assertNotIn("Dynamic world state", ending)
        self.assertNotIn("天气", ending)


class MergedAuditToleranceTest(unittest.TestCase):
    """The merged agent's physics audit: wording differences pass, real contradictions are still caught.

    All four fields used to be compared exactly. In 258 segments **51%** fell back to the separate
    rewriter, a median of 2.60 s and one more LLM call each, 699 s in total, on what the comments
    called a rare path. motion_target / motion_framing are free text, and verbatim copying is
    unrealistic.
    """

    PLAN = {
        "phase": "transform",
        "relation_change": "containment",
        "target": "the arched airlock door",
        "framing": "full body, wide shot",
    }

    def audit(self, **over) -> str:
        result = {
            "motion_phase": "transform",
            "motion_relation_change": "containment",
            "motion_target": "the arched airlock door",
            "motion_framing": "full body, wide shot",
        }
        result.update(over)
        return motion._motion_audit_error(result, self.PLAN)

    def test_identical_passes(self) -> None:
        self.assertEqual("", self.audit())

    def test_wording_differences_pass(self) -> None:
        """The same door must not be judged a conflict."""
        for target in ("arched airlock door", "the arched airlock door,", "that arched airlock door"):
            with self.subTest(target=target):
                self.assertEqual("", self.audit(motion_target=target))
        self.assertEqual("", self.audit(motion_framing="full body wide shot"))

    def test_a_different_target_is_still_caught(self) -> None:
        """A different noun is a real contradiction; letting it pass would send a wrong action to the video model."""
        self.assertIn("motion_target", self.audit(motion_target="the control panel"))

    def test_no_relation_change_means_nothing_to_audit(self) -> None:
        """Without a relation change no physical constraint appears in the prompt at all; auditing it buys nothing.

        With `relation_change == "none"`, `_motion_active` is False, `_motion_plan_clause` returns
        an empty string and `_lock_motion_plan` is a no-op. The audit still demanded `none / '' /
        none` and fell back on a mismatch: 10 of 25 fallbacks in 102 beats were this path, 2.45–3.18
        s each, buying nothing.
        """
        idle = {
            "phase": "none",
            "relation_change": "none",
            "target": "",
            "framing": "none",
            "interaction_type": "none",
        }
        noisy = {
            "motion_phase": "engage",
            "motion_relation_change": "contact",
            "motion_target": "信纸的防伪钢印",
            "motion_framing": "medium",
        }
        self.assertEqual("", motion._motion_audit_error(noisy, idle))

    def test_relation_change_stays_strict(self) -> None:
        """relation_change is the physical fact to show; a mismatch is a real conflict."""
        self.assertIn("motion_relation_change", self.audit(motion_relation_change="distance"))

    def test_phase_is_not_audited(self) -> None:
        """phase is not audited: it is a function of relation and is removed before the video model sees the prompt.

        Real failure: the merged call emitted motion_plan and the six sections together and the
        model copied its own phase; the server then changed phase in _repair_motion_plan and
        compared the changed value, judging it a conflict. 2 of 7 fallbacks were this, with the same
        repair recorded in that beat's _server_repairs.
        """
        self.assertEqual("", self.audit(motion_phase="approach"))
        self.assertEqual("", self.audit(motion_phase="engage"))
        # But a real change of relation is still caught
        self.assertIn(
            "motion_relation_change", self.audit(motion_phase="approach", motion_relation_change="distance")
        )

    def test_empty_where_the_plan_has_a_target_is_caught(self) -> None:
        self.assertIn("motion_target", self.audit(motion_target=""))

    def test_none_sentinels_on_both_sides_agree(self) -> None:
        """ "No target" may be written as an empty string / none / n/a on either side; that is consistent, not a conflict.
        The first tolerant version missed this and failed cases it should have passed.
        """
        plan = dict(self.PLAN, target="none", framing="none")
        for got in ("", "none", "N/A"):
            with self.subTest(got=got):
                result = {
                    "motion_phase": "transform",
                    "motion_relation_change": "containment",
                    "motion_target": got,
                    "motion_framing": got,
                }
                self.assertEqual("", motion._motion_audit_error(result, plan))


class InactivePlanIsSlimTest(unittest.TestCase):
    """Static shots do not write an empty skeleton: this guards "no empty values", not "fewer fields".

    In 300 recorded beats (166 static / 134 with action), `actors` `instrument` `path`
    `visual_evidence` `must_show` were empty in 166/166 static beats and almost always filled in the
    134 action beats. So the static branch does not need them.

    A/B (same prompt and replica, only the schema changed): static beats 1356 -> 1239 tokens, 9%
    less.
    """

    def _variants(self):
        plan_schema = density_audit._motion_replan_schema(agent_schema._agent_schema(False))["properties"][
            "motion_plan"
        ]
        self.assertIn("anyOf", plan_schema)
        return plan_schema["anyOf"]

    def test_inactive_variant_only_keeps_the_four_enums(self) -> None:
        inactive, _ = self._variants()
        self.assertEqual(
            sorted(inactive["required"]), ["framing", "interaction_type", "phase", "relation_change"]
        )
        # Turning off additionalProperties is what really saves tokens: otherwise the model fills
        # the 11 empty fields out of habit, the schema allows it, and nothing is saved.
        self.assertFalse(inactive.get("additionalProperties", True))
        for key in agent_schema._MOTION_DETAIL_FIELDS:
            self.assertNotIn(key, inactive["properties"])

    def test_active_variant_still_demands_the_full_plan(self) -> None:
        """With action no field may be missing: that would remove the anti-clipping layer."""
        _, active = self._variants()
        for key in agent_schema._MOTION_DETAIL_FIELDS:
            self.assertIn(key, active["required"])
        self.assertEqual(active["properties"]["path"].get("minLength"), 1)

    def test_missing_detail_fields_normalize_to_an_empty_plan(self) -> None:
        """A missing key must equal the empty string received before, or downstream code gets None."""
        plan = motion._normalize_motion_plan(
            {"interaction_type": "none", "phase": "none", "relation_change": "none", "framing": "wide"}
        )
        for key in agent_schema._MOTION_DETAIL_FIELDS:
            self.assertIn(key, plan)
        self.assertEqual(plan["path"], "")
        self.assertEqual(plan["actors"], [])
        self.assertEqual(motion._motion_plan_errors(plan), [])


class PromotedFramingIsNotAuditedTest(unittest.TestCase):
    """Framing the server raised must not be audited: the same trap as `motion_phase`.

    `_repair_motion_plan` raises framing to wide for distance / containment / support. The model
    writes the same value in `motion_plan.framing` and `motion_framing`; the server afterwards
    changes only the former and compares it to the latter, so a consistent model is judged a
    conflict and falls back for nothing.

    Measured (cold start, relation=support):

        merged_fallback: motion_framing expected 'wide', got 'medium'
        → rewrite 1.342 s + physics translation 0.583 s = 1.9 s wasted
    """

    def _plan(self, relation, framing="medium"):
        plan = motion._normalize_motion_plan(
            {
                "interaction_type": "support",
                "phase": "transform",
                "relation_change": relation,
                "framing": framing,
                "actors": ["林恩"],
                "target": "地面",
                "start_state": "站着",
                "path": "蹲下",
                "end_state": "蹲在地上",
                "camera_side": "正面",
                "visual_evidence": ["actor_full_body"],
                "must_show": ["痕迹"],
                "invariants": ["同一盏灯"],
            }
        )
        return motion._repair_motion_plan(plan, {})

    def test_model_keeps_its_own_framing_when_the_service_promoted(self):
        for relation in ("distance", "containment", "support"):
            with self.subTest(relation=relation):
                plan = self._plan(relation)
                self.assertEqual("wide", plan["framing"], "这几种关系必须被提升")
                self.assertEqual(
                    "",
                    motion._motion_audit_error(
                        {
                            "motion_relation_change": relation,
                            "motion_target": "地面",
                            "motion_framing": "medium",
                        },
                        plan,
                    ),
                )

    def test_a_real_framing_disagreement_is_still_caught(self):
        """Relations that were not raised are still audited; this must not be switched off along the way."""
        plan = self._plan("configuration")
        self.assertEqual("medium", plan["framing"], "这种关系不该被提升")
        self.assertIn(
            "motion_framing",
            motion._motion_audit_error(
                {
                    "motion_relation_change": "configuration",
                    "motion_target": "地面",
                    "motion_framing": "wide",
                },
                plan,
            ),
        )


class AuditComparesAgainstTheModelsOwnPlanTest(unittest.TestCase):
    """The audit asks whether the model contradicts **its own** plan, not the plan the server modified.

    Before the audit the server modifies it three times: `_advance_completed_transition` consumes a
    completed approach, `_reconcile_dialogue_motion` turns pure dialogue beats into none,
    `_repair_motion_plan` aligns impact to contact. Comparing the modified plan flags a model that
    wrote the same thing in both places.

    In 305 beats there were 61 fallbacks (20%), 25 of them on `motion_relation_change`, mostly
    "expected 'none', got 'contact'", left behind by the downgrade.
    """

    def _plan(self, relation):
        return motion._normalize_motion_plan(
            {
                "interaction_type": "contact",
                "phase": "engage",
                "relation_change": relation,
                "framing": "medium",
                "actors": ["林恩"],
                "target": "门把手",
                "start_state": "站着",
                "path": "伸手",
                "end_state": "握住门把手",
                "camera_side": "正面",
                "visual_evidence": ["actor_full_body"],
                "must_show": [],
                "invariants": [],
            }
        )

    def test_a_self_consistent_model_passes_even_after_the_service_demotes(self):
        """The model wrote contact in both places and the server downgraded the plan to none: not a conflict."""
        model_plan = self._plan("contact")
        result = {"motion_relation_change": "contact", "motion_target": "门把手", "motion_framing": "medium"}
        self.assertEqual("", motion._motion_audit_error(result, model_plan))
        # Comparing the downgraded plan gives a false alarm: exactly the bug that was fixed
        self.assertIn("motion_relation_change", motion._motion_audit_error(result, self._plan("none")))

    def test_a_real_self_contradiction_is_still_caught(self):
        """The model itself writing two different values is still caught."""
        self.assertIn(
            "motion_relation_change",
            motion._motion_audit_error(
                {
                    "motion_relation_change": "containment",
                    "motion_target": "门把手",
                    "motion_framing": "medium",
                },
                self._plan("contact"),
            ),
        )


if __name__ == "__main__":
    unittest.main()
