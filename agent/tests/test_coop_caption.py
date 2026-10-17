"""Regress a real four-person caption that once lost player 3."""

import copy
from dataclasses import replace
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from murea_agent.director import agent_schema
from murea_agent.shot import prompt_budget
from murea_agent.coop import (
    caption as coop_caption,
    generation as coop_generation,
    prefetch as coop_prefetch,
)
from murea_agent.providers import video
from murea_agent.coop.scene_plan import ScenePlan, ShotCompiler


class CaptionBudgetTest(unittest.TestCase):
    def setUp(self):
        self.fixture = json.loads(
            (Path(__file__).parent / "fixtures/coop_four_actor_caption.json").read_text()
        )
        f = self.fixture
        self.plan = ScenePlan.from_legacy(f["plan"], f["players"], duration=f["duration"])
        self.pictures = f["pictures"]
        self.context = {
            "camera": f["plan"]["camera"],
            "environment": "The action unfolds inside <Subject 5>, with its established layout and lighting.",
            "cast_count": "Exactly 4 people share the frame: <Subject 1>, <Subject 2>, <Subject 3>, <Subject 4>.",
            "style": "Visual style: cinematic lighting.",
        }
        self.source = dict(zip(agent_schema.SECTION_ORDER, f["original_prompt"].split("\n\n")))
        self.source["detailed_description"] = ShotCompiler.compile_timeline(self.plan, **self.context)

    def assert_motion_intact(self, six):
        """Every actor's action clause and the world event reach the video model verbatim.

        Start / end states and routes stay in the internal plan only (see ShotCompiler._timeline),
        so they are not expected in the caption."""
        body = six["detailed_description"]
        for i, action in enumerate(self.plan.actions, 1):
            self.assertIn(ShotCompiler.action_clause(action, f"<Subject {i}>").rstrip(" .;"), body)
        self.assertIn(self.plan.event_action.rstrip(" .;"), body)
        self.assertNotIn("[compressed]", prompt_budget.assemble(six))

    def prepared(self):
        f = self.fixture
        return {
            "plan": copy.deepcopy(f["plan"]),
            "players": f["players"],
            "pictures": self.pictures,
            "scene_plan": self.plan,
            "world_snapshot": None,
            "caption_source": copy.deepcopy(self.source),
            "caption_context": self.context,
            "six": copy.deepcopy(f["six"]),
            "refs": [f"{i}.jpg" for i in range(5)],
            "duration": f["duration"],
            "previous": "parent-latent",
            "base_head": None,
        }

    def test_real_long_caption_keeps_all_four_actions_and_exact_detail(self):
        before = copy.deepcopy(self.source)
        limit = len(prompt_budget.assemble(before)) - 300
        six = coop_caption.fit(self.source, self.plan, self.pictures, limit=limit)
        self.assertLessEqual(len(prompt_budget.assemble(six)), limit)
        self.assertEqual(before, self.source)
        self.assertEqual(before["detailed_description"], six["detailed_description"])
        self.assert_motion_intact(six)
        # The old prompt-too-long retry did corrupt this actor in this same fixture.
        third = self.plan.actions[2].action
        self.assertIn(third, self.fixture["original_prompt"])
        self.assertNotIn(third, self.fixture["six"]["detailed_description"])

    def test_a_legacy_phase_caption_is_rebuilt_not_trimmed(self):
        """Captions from the old phase-by-phase compiler no longer match the plan; fit() refuses
        them so the caller recompiles from the actors' plans instead of trimming stale text."""
        f = json.loads((Path(__file__).parent / "fixtures/coop_four_actor_explicit_starts.json").read_text())
        plan = ScenePlan.from_legacy(f["plan"], f["players"], duration=f["duration"])
        before = copy.deepcopy(f["six"])
        with self.assertRaisesRegex(ValueError, "motion paths are incomplete"):
            coop_caption.fit(before, plan, f["pictures"])
        self.assertEqual(before, f["six"])
        fresh = dict(
            before,
            detailed_description=ShotCompiler.compile_timeline(
                plan,
                camera="A continuous wide shot.",
                environment="Inside <Subject 5>.",
                cast_count="Exactly 4 people share the frame.",
            ),
        )
        after = coop_caption.fit(fresh, plan, f["pictures"])
        self.assertLessEqual(len(prompt_budget.assemble(after)), 6000)
        for index, action in enumerate(plan.actions, 1):
            self.assertIn(
                ShotCompiler.action_clause(action, f"<Subject {index}>").rstrip(" .;"),
                after["detailed_description"],
            )

    def test_short_caption_is_unchanged(self):
        short = coop_caption.fit(self.source, self.plan, self.pictures)
        self.assertEqual(short, coop_caption.fit(short, self.plan, self.pictures))

    def test_all_metadata_tiers_keep_complete_reference_rows(self):
        six = copy.deepcopy(self.source)
        six["subject_definitions"] = "\n".join(
            coop_caption.definition(p) + " " + "Optional appearance detail. " * 100 for p in self.pictures
        )
        six["summary"] += " Optional summary detail." * 100
        six["overall_soundscape"] = "Ambient room sound. " * 200
        out = coop_caption.fit(six, self.plan, self.pictures)
        self.assertLessEqual(len(prompt_budget.assemble(out)), 6000)
        self.assertEqual(5, len(out["subject_definitions"].splitlines()))
        self.assertEqual(5, len(out["retention_analysis"].splitlines()))
        self.assert_motion_intact(out)

    def test_cast_list_cannot_hide_a_deleted_or_reassigned_action(self):
        original = ShotCompiler.action_clause(self.plan.actions[2], "<Subject 3>")
        for replacement in ("", original.replace("<Subject 3>", "<Subject 2>", 1)):
            broken = copy.deepcopy(self.source)
            broken["detailed_description"] = broken["detailed_description"].replace(original, replacement)
            self.assertIn("<Subject 3>", broken["detailed_description"])
            with self.assertRaisesRegex(ValueError, "motion paths are incomplete"):
                coop_caption.fit(broken, self.plan, self.pictures)

    def test_compression_rejects_extra_picture_or_incomplete_retention(self):
        for field, extra in (
            ("subject_definitions", " <Picture 99>"),
            ("retention_analysis", "\n<Subject 3>: fully_preserved."),
        ):
            broken = copy.deepcopy(self.source)
            broken[field] += extra
            with self.assertRaises(ValueError):
                coop_caption.fit(broken, self.plan, self.pictures)

    def test_missing_actor_plan_cannot_pass_using_the_cast_list(self):
        for actions in (
            self.plan.actions[:2] + self.plan.actions[3:],
            self.plan.actions[:2] + (replace(self.plan.actions[2], action=""),) + self.plan.actions[3:],
        ):
            broken = replace(self.plan, actions=actions)
            with self.assertRaisesRegex(ValueError, "for every player"):
                coop_caption.fit(self.source, broken, self.pictures)

    def test_budget_and_cached_rebuild_keep_individual_retention(self):
        details = (
            "Her silver braid and back-mounted quiver remain visible.",
            "His cropped hair and sword belt remain visible.",
            "Her copper curls and shoulder clasp remain visible.",
            "His dark beard and wrist guards remain visible.",
        )
        self.plan = replace(
            self.plan,
            actions=tuple(
                replace(action, retention=detail) for action, detail in zip(self.plan.actions, details)
            ),
        )
        rows = []
        for picture in self.pictures:
            if picture["kind"] == "character":
                rows.append(coop_caption.retention(picture, detail=details[picture["n"] - 1]))
            else:
                rows.append(coop_caption.retention(picture))
        self.source["retention_analysis"] = "\n".join(rows)
        # Force the budget-shortening path; equipment ownership must not be replaced by a generic
        # sentence.
        limit = len(prompt_budget.assemble(self.source)) - 1
        out = coop_caption.fit(self.source, self.plan, self.pictures, limit=limit)
        self.assertEqual(self.source["retention_analysis"], out["retention_analysis"])
        prepared = self.prepared()
        prepared.pop("caption_source", None)
        result = coop_generation._rebuild_prepared_caption(prepared)
        for detail in details:
            self.assertIn(detail, result["retention_analysis"])

    def test_impossible_budget_preserves_source_instead_of_cutting_an_actor(self):
        original = copy.deepcopy(self.source)
        with self.assertRaisesRegex(ValueError, "动作已保留"):
            coop_caption.fit(self.source, self.plan, self.pictures, limit=1000)
        self.assertEqual(original, self.source)

    def test_persisted_cache_restores_full_motion_and_synchronizes_fallback(self):
        prepared = coop_prefetch._serialize_prepared(self.prepared())
        self.assertIsInstance(prepared["scene_plan"], dict)
        result = coop_generation._rebuild_prepared_caption(prepared)
        self.assert_motion_intact(result)
        self.assertEqual(prepared["six"], prepared["fallback_six"])
        self.assertEqual(prepared["prompt"], prepared["fallback_prompt"])
        self.assertLessEqual(len(prepared["prompt"]), 6000)

    def test_legacy_cache_rebuilds_truncated_caption_from_actor_facts(self):
        prepared = self.prepared()
        prepared.pop("caption_source")
        prepared.pop("caption_context")
        prepared["scene_plan"] = self.plan.to_legacy()
        self.assertIn("[compressed]", prompt_budget.assemble(prepared["six"]))
        result = coop_generation._rebuild_prepared_caption(prepared)
        self.assert_motion_intact(result)
        self.assertNotIn("At the start, <Subject 1> takes", result["detailed_description"])

    def test_prompt_limit_retry_uses_same_structural_guard_and_saved_fallback(self):
        # Leave enough optional metadata to exercise a lower prompt limit.
        prepared = self.prepared()
        prepared["plan"]["actions"] = prepared["plan"]["actions"][:2]
        prepared["players"] = prepared["players"][:2]
        prepared["scene_plan"] = ScenePlan.from_legacy(prepared["plan"], prepared["players"])
        prepared["pictures"] = [
            {"n": 1, "kind": "character"},
            {"n": 2, "kind": "character"},
            {"n": 3, "kind": "scene"},
        ]
        prepared["refs"] = ["1.jpg", "2.jpg", "3.jpg"]
        prepared["caption_context"] = {
            "camera": "A continuous wide shot.",
            "environment": "The action unfolds inside <Subject 3>.",
            "cast_count": "Exactly 2 people share the frame: <Subject 1>, <Subject 2>.",
        }
        source = prepared["caption_source"]
        source["subject_definitions"] = "\n".join(coop_caption.definition(p) for p in prepared["pictures"])
        source["retention_analysis"] = "\n".join(
            f"<Subject {p['n']}> (appears in [Shot 1] (full body)): fully_preserved - "
            + "Optional identity prose. " * 22
            for p in prepared["pictures"]
        )
        source["summary"] = "[reference_generation] Both players move."
        observed = []

        def generate(prompt, **kwargs):
            observed.append((prompt, kwargs))
            if len(observed) == 1:
                raise video.PromptTooLongError("prompt limit changed")
            return video.Generation(
                "test-job",
                Path("/tmp/coop-caption-test.mp4"),
                10,
                1,
                {"continuation_fallback": True},
                submitted_prompt=prompt,
            )

        with (
            # The path where a continuation carries the full prompt (and its images).
            patch.object(video.murea.Murea, "references_on_continuation", True),
            patch.object(video, "generate", side_effect=generate),
            patch.object(coop_generation.coop_media, "remove_white_reference_tail", return_value=10),
            patch.object(coop_generation.coop_media, "grab_frames", return_value=[]),
        ):
            rendered = coop_generation.render_prepared(
                prepared, progress=lambda *a: None, cancelled=lambda: False
            )
        self.assertEqual(2, len(observed))
        self.assertLess(len(observed[1][0]), len(observed[0][0]))
        self.assertEqual(observed[1][0], observed[1][1]["continuation_fallback_prompt"])
        self.assertEqual(observed[1][0], rendered["submitted_prompt"])
        self.assertEqual(observed[1][0], prompt_budget.assemble(rendered["submitted_six"]))
        for i, action in enumerate(prepared["scene_plan"].actions, 1):
            self.assertIn(ShotCompiler.action_clause(action, f"<Subject {i}>"), observed[1][0])

    def test_this_four_player_round_fits_murea_fast_without_losing_an_action(self):
        # Under the old 6000-character limit this prompt was 1114 Gemma tokens; Murea-Fast reads
        # 1024 and silently dropped the end. Fitted to its budget it is 956.
        from murea_agent.providers.video import murea

        limit = murea.FAST_MAX_PROMPT_CHARS
        self.assertGreater(len(prompt_budget.assemble(self.source)), limit)
        six = coop_caption.fit(self.source, self.plan, self.pictures, limit=limit)
        self.assertLessEqual(prompt_budget.prompt_size(prompt_budget.assemble(six)), limit)
        self.assert_motion_intact(six)

    def test_a_reference_free_continuation_is_fitted_as_sent(self):
        def sent(six):
            return coop_generation._reference_free_prompt(six, self.pictures)

        plain = prompt_budget.prompt_size(sent(self.source))
        # The prose drops optional sound and framing to fit, never an actor's motion.
        out = coop_caption.fit_continuation(
            self.source, self.plan, self.pictures, limit=plain - 50, sent=sent
        )
        self.assertLessEqual(prompt_budget.prompt_size(sent(out)), plain - 50)
        self.assert_motion_intact(out)
        # Already fitting: unchanged.
        self.assertEqual(
            self.source,
            coop_caption.fit_continuation(self.source, self.plan, self.pictures, limit=plain, sent=sent),
        )
        # Too long even then: the round fails with the request to shorten the actions.
        with self.assertRaisesRegex(ValueError, "请减少本轮动作步骤"):
            coop_caption.fit_continuation(self.source, self.plan, self.pictures, limit=200, sent=sent)

    def test_only_what_is_sent_must_fit(self):
        # 3500: the full prompt cannot fit (3607 at least), the reference-free prose can (3333).
        class Fast:
            references_on_continuation = False
            max_prompt_chars = 3500

        with patch.object(coop_generation.video, "model", return_value=Fast()):
            # A cut or the opening sends the full prompt: the round is refused.
            with self.assertRaisesRegex(ValueError, "请减少本轮动作步骤"):
                coop_generation._fit_caption(
                    self.source, self.plan, self.pictures, video_model="murea", previous=None
                )
            # A continuation sends only the prose; the full prompt is merely its fallback.
            six = coop_generation._fit_caption(
                self.source, self.plan, self.pictures, video_model="murea", previous="parent"
            )
        sent = coop_generation._reference_free_prompt(six, self.pictures)
        self.assertLessEqual(prompt_budget.prompt_size(sent), 3500)
        self.assert_motion_intact(six)


class PhaseOrderingTest(unittest.TestCase):
    def test_known_start_all_actions_routes_and_endpoints_share_one_time_axis(self):
        members = [{"user_id": "a"}, {"user_id": "b"}]
        plan = ScenePlan.from_legacy(
            {
                "actions": [
                    {
                        "user_id": "a",
                        "start_state": "standing in front of the dragon",
                        "action": "runs around the dragon and reaches its rear flank.",
                        "path": "a left arc around the dragon",
                        "target": "the dragon",
                        "end_state": "standing at the rear flank",
                    },
                    {
                        "user_id": "b",
                        "start_state": "beside the pillar",
                        "action": "runs backward to the gateway.",
                        "path": "a straight line away from the dragon",
                        "target": "the dragon",
                        "end_state": "at the gateway",
                    },
                ]
            },
            members,
            duration=10,
        )
        # The internal plan orders every actor's phases on one time axis.
        numeric = ShotCompiler.timeline(plan)
        start, rest = numeric.split("From 0.00s to 10.00s, ")
        self.assertIn("At 0.00s, <Subject 1> is standing in front", start)
        self.assertIn("<Subject 2> is beside the pillar", start)
        self.assertNotIn("reaches its rear flank", start)
        self.assertIn("runs around the dragon and reaches its rear flank", rest)
        body, end = rest.split("By 10.00s, ")
        self.assertIn("<Subject 2> follows a straight line away from the dragon", body)
        self.assertIn("<Subject 1> is standing at the rear flank", end)
        self.assertIn("<Subject 2> is at the gateway", end)
        # The caption sent to the video model carries only the visible performances.
        caption = ShotCompiler.timeline(plan, caption=True)
        self.assertEqual(
            "<Subject 1> runs around the dragon and reaches its rear flank. "
            "<Subject 2> runs backward to the gateway.",
            caption,
        )
        self.assertNotRegex(caption, r"\d+\.\d+s")

    def test_unknown_start_and_unrequested_end_are_not_invented(self):
        plan = ScenePlan.from_legacy(
            {"actions": [{"user_id": "a", "action": "draws the bow."}]}, [{"user_id": "a"}]
        )
        text = ShotCompiler.timeline(plan, caption=True)
        self.assertEqual("<Subject 1> draws the bow.", text)
        self.assertNotIn("releases", text)
        self.assertNotIn("At the start", text)
        self.assertNotIn("By the end", text)


if __name__ == "__main__":
    unittest.main()
