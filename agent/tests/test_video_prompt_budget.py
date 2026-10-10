"""Final caption limits include server-added identity and physics clauses."""

import unittest
from unittest.mock import patch

from murea_agent import config
from murea_agent.shot import prompt_budget
from murea_agent.shot import video_prompt
from murea_agent import loop
from murea_agent.memory.world import World


class PromptBudgetTest(unittest.TestCase):
    def test_repeated_reference_rules_fit_without_replanning_or_cutting_prose(self):
        fields = {
            "subject_definitions": "<Subject 1> is the elf in <Picture 1>.",
            "summary": "[reference_generation] Rescue.",
            "retention_analysis": "\n".join(
                f"<Subject {i}> (appears in [Shot 1] (full body)): fully_preserved - An elf; "
                "only the entity's identity and appearance are retained. The reference "
                "backdrop, floor, shadows, lighting and framing are replaced by the current "
                "story scene (<Subject 4>) throughout the shot. The reference image itself "
                "is never displayed as a frame or transition target."
                for i in range(1, 4)
            ) + "\n<Subject 4> (appears in [Shot 1] (wide scene view)): fully_preserved - A courtyard.",
            "detailed_description": "[Shot 1] The arrow hits. (S1)<d>[Chinese]快动手！</d>",
            "overall_soundscape": "Rain and thunder.",
            "non_diegetic_music": "N/A",
        }
        missing = 6200 - len(prompt_budget.assemble(fields))
        fields["detailed_description"] += "x" * missing
        with patch.object(config, "VIDEO_MAX_PROMPT_CHARS", 6000), patch.object(loop.llm, "chat") as rewrite:
            fitted = prompt_budget._fit_video_prompt(fields)
        rewrite.assert_not_called()
        self.assertLessEqual(len(prompt_budget.assemble(fitted)), 6000)
        for key in fields:
            if key != "retention_analysis":
                self.assertEqual(fields[key], fitted[key])
        self.assertEqual(1, fitted["retention_analysis"].count("never displayed as a frame"))
        for i in range(1, 5):
            self.assertIn(f"<Subject {i}>", fitted["retention_analysis"])
        self.assertEqual(fitted, prompt_budget._fit_video_prompt(fitted))

    def test_compact_retention_fits_without_an_extra_model_call(self):
        world = World(here="court")
        world.subjects = {
            f"S{i}": {"name": f"elf{i}", "definition": "An elf in a green cloak. " * 12}
            for i in range(1, 5)
        }
        world.places = {"court": {"definition": "A wet stone courtyard. " * 12}}
        pictures = [{"n": i, "kind": "character", "id": f"S{i}"} for i in range(1, 5)]
        pictures.append({"n": 5, "kind": "scene", "id": "court"})
        locked, missing = video_prompt._compose_definitions(world, pictures)
        self.assertEqual([], missing)
        fields = {
            "subject_definitions": locked,
            "summary": "[reference_generation] The elves cross the courtyard.",
            "retention_analysis": video_prompt._build_retention(pictures, world, {}),
            "detailed_description": "[Shot 1] The four elves walk across the wet courtyard.",
            "overall_soundscape": "Rain and footsteps.",
            "non_diegetic_music": "N/A",
        }
        self.assertLess(len(fields["retention_analysis"]), 1200)
        with patch.object(loop.llm, "chat") as rewrite:
            fitted = prompt_budget._fit_video_prompt(fields, limit=4200)
        rewrite.assert_not_called()
        self.assertEqual(fields, fitted)

    def test_caption_only_repair_preserves_locked_sections_and_dialogue(self):
        fields = {
            "subject_definitions": "<Subject 1> is the elf in <Picture 1>.",
            "summary": "[reference_generation] An elf moves.",
            "detailed_description": "[Shot 1] <Subject 1> (S1)<d>[Chinese]快走！</d>" + " verbose" * 300,
            "retention_analysis": "Keep the same identity.",
            "overall_soundscape": "<Audio 1> is the voice of <Subject 1>. Rain.",
            "non_diegetic_music": "N/A",
        }
        repaired = {
            "summary": fields["summary"],
            "detailed_description": "[Shot 1] <Subject 1> moves. (S1)<d>[Chinese]快走！</d>",
        }
        with (
            patch.object(config, "VIDEO_MAX_PROMPT_CHARS", 1200),
            patch.object(loop.llm, "chat", return_value=repaired) as model,
        ):
            fitted = prompt_budget._fit_video_prompt(fields)
        model.assert_called_once()
        self.assertLessEqual(len(prompt_budget.assemble(fitted)), 1200)
        for key in ("subject_definitions", "retention_analysis", "overall_soundscape", "non_diegetic_music"):
            self.assertEqual(fields[key], fitted[key])
        # Dropping a marker or dialogue must never pass; raise only after all retries, with the
        # concrete reason.
        for bad, why in (
            ("[Shot 1] <Subject 1> moves.", "dialogue"),
            ("[Shot 1] (S1)<d>[Chinese]快走！</d>", "markers"),
        ):
            with (
                patch.object(config, "VIDEO_MAX_PROMPT_CHARS", 1200),
                patch.object(config, "VIDEO_PROMPT_BUDGET_TRIES", 3),
                patch.object(
                    loop.llm, "chat", return_value={**repaired, "detailed_description": bad}
                ) as retried,
            ):
                with self.assertRaisesRegex(ValueError, "after 3 compressions"):
                    prompt_budget._fit_video_prompt(fields)
            self.assertEqual(3, retried.call_count, why)
            # From the second attempt on, the model must be told what went wrong, or retrying is
            # pointless.
            self.assertIn("previous_attempt_problem", retried.call_args_list[1].kwargs["user"])

    def test_when_compression_keeps_failing_trailing_plain_sentences_are_dropped(self):
        filler = " ".join(f"Rain streaks across the window number {i}." for i in range(60))
        fields = {
            "subject_definitions": "<Subject 1> is the elf in <Picture 1>.",
            "summary": "[reference_generation] An elf moves. It is raining hard outside.",
            "detailed_description": "[Shot 1] <Subject 1> walks to the door. "
            "(S1)<d>[Chinese]快走！</d> " + filler,
            "retention_analysis": "Keep the same identity.",
            "overall_soundscape": "Rain.",
            "non_diegetic_music": "N/A",
        }
        still_long = {"summary": fields["summary"], "detailed_description": fields["detailed_description"]}
        with (
            patch.object(config, "VIDEO_MAX_PROMPT_CHARS", 1200),
            patch.object(loop.llm, "chat", return_value=still_long),
        ):
            fitted = prompt_budget._fit_video_prompt(fields)
        self.assertLessEqual(len(prompt_budget.assemble(fitted)), 1200)
        self.assertTrue(fitted["detailed_description"].startswith("[Shot 1] <Subject 1> walks to the door."))
        self.assertIn("(S1)<d>[Chinese]快走！</d>", fitted["detailed_description"])
        self.assertEqual("[reference_generation] An elf moves.", fitted["summary"])

    def test_a_second_attempt_can_still_succeed(self):
        fields = {
            "subject_definitions": "<Subject 1> is the elf in <Picture 1>.",
            "summary": "[reference_generation] An elf moves.",
            "detailed_description": "[Shot 1] <Subject 1> (S1)<d>[Chinese]快走！</d>" + " verbose" * 300,
            "retention_analysis": "Keep the same identity.",
            "overall_soundscape": "<Audio 1> is the voice of <Subject 1>. Rain.",
            "non_diegetic_music": "N/A",
        }
        good = {
            "summary": fields["summary"],
            "detailed_description": "[Shot 1] <Subject 1> moves. (S1)<d>[Chinese]快走！</d>",
        }
        too_long = {**good, "detailed_description": good["detailed_description"] + " verbose" * 300}
        with (
            patch.object(config, "VIDEO_MAX_PROMPT_CHARS", 1200),
            # The default is one attempt; retrying is still supported when configured.
            patch.object(config, "VIDEO_PROMPT_BUDGET_TRIES", 3),
            patch.object(loop.llm, "chat", side_effect=[too_long, good]) as model,
        ):
            fitted = prompt_budget._fit_video_prompt(fields)
        self.assertEqual(2, model.call_count)
        self.assertLessEqual(len(prompt_budget.assemble(fitted)), 1200)

    def test_trim_drops_marker_sentences_while_another_mention_remains(self):
        fields = {
            "subject_definitions": "<Subject 1> is a traveller.",
            "summary": "<Subject 1> walks in.",
            "detailed_description": "[Shot 1] <Subject 1> steps through the door. "
            + " ".join(f"<Subject 1> keeps a steady pose number {i}." for i in range(40)),
            "retention_analysis": "Keep the same identity.",
            "overall_soundscape": "Rain.",
            "non_diegetic_music": "N/A",
        }
        trimmed = prompt_budget._trim_to_budget(fields, 600)
        self.assertIsNotNone(trimmed)
        self.assertLessEqual(len(prompt_budget.assemble(trimmed)), 600)
        self.assertIn("[Shot 1] <Subject 1> steps through the door.", trimmed["detailed_description"])

    def test_retries_compress_the_best_previous_result(self):
        fields = {
            "subject_definitions": "<Subject 1> is a traveller.",
            "summary": "A walk.",
            # One sentence: trimming never drops the opening one, so only the model can shorten it.
            "detailed_description": "[Shot 1] <Subject 1> walks" + " long" * 400 + ".",
            "retention_analysis": "Keep.",
            "overall_soundscape": "Rain.",
            "non_diegetic_music": "N/A",
        }
        shorter = {
            "summary": "A walk.",
            "detailed_description": "[Shot 1] <Subject 1> walks" + " long" * 200 + ".",
        }
        good = {"summary": "A walk.", "detailed_description": "[Shot 1] <Subject 1> walks."}
        with (
            patch.object(config, "VIDEO_MAX_PROMPT_CHARS", 800),
            patch.object(config, "VIDEO_PROMPT_BUDGET_TRIES", 3),
            patch.object(loop.llm, "chat", side_effect=[shorter, good]) as model,
        ):
            prompt_budget._fit_video_prompt(fields)
        second_input = model.call_args_list[1].kwargs["user"]
        self.assertIn(
            f'"current_characters": {len(shorter["summary"]) + len(shorter["detailed_description"])}',
            second_input,
        )

    def test_trim_handles_punctuation_right_before_a_closing_dialogue_tag(self):
        """Real caption: `...storm in!</d>, a sharp call.` used to break sentence splitting."""
        desc = (
            "[Shot 1] <Subject 1> steps in. <Subject 2> speaks as S1: (S1)<d>[English] Shut "
            "that door, you are letting the storm in!</d>, a sharp call across the room. "
            + " ".join(f"<Subject 1> holds a steady pose number {i}." for i in range(40))
        )
        fields = {
            "subject_definitions": "<Subject 1> is a traveller. <Subject 2> is a keeper.",
            "summary": "<Subject 1> walks in.",
            "detailed_description": desc,
            "retention_analysis": "Keep.",
            "overall_soundscape": "Rain.",
            "non_diegetic_music": "N/A",
        }
        trimmed = prompt_budget._trim_to_budget(fields, 700)
        self.assertIsNotNone(trimmed)
        self.assertLessEqual(len(prompt_budget.assemble(trimmed)), 700)
        self.assertIn("you are letting the storm in!</d>, a sharp call", trimmed["detailed_description"])

    def test_chinese_dialogue_costs_about_a_token_per_character(self):
        self.assertEqual(10, prompt_budget.prompt_size("Run, Mara!"))
        # Two Chinese characters and a full-width "！": four each.
        self.assertEqual(12, prompt_budget.prompt_size("快走！"))

    def test_murea_fast_budgets_by_its_text_encoder(self):
        from murea_agent.providers.video import murea

        with patch.object(murea, "backend", return_value="fast"):
            self.assertEqual(murea.FAST_MAX_PROMPT_CHARS, murea.PROVIDER.max_prompt_chars)
        with patch.object(murea, "backend", return_value="murea"):
            self.assertEqual(config.VIDEO_MAX_PROMPT_CHARS, murea.PROVIDER.max_prompt_chars)


class ContinuationPromptBudgetTest(unittest.TestCase):
    """A reference-free continuation is budgeted as sent, not as the full prompt."""

    class Fast:
        references_on_continuation = False
        max_prompt_chars = 900

    def fields(self) -> dict[str, str]:
        return {
            "subject_definitions": "<Subject 1> is the courier in <Picture 1>. " + "Detail. " * 60,
            "summary": "<Subject 1> walks to the door.",
            "detailed_description": "[Shot 1] <Subject 1> walks to the door. She stops and knocks twice.",
            "retention_analysis": "Keep the identity. " * 30,
            "overall_soundscape": "Rain on cobbles.",
            "non_diegetic_music": "N/A",
        }

    def test_the_sent_prose_keeps_its_last_sentence_and_no_model_is_called(self):
        six = self.fields()
        full = prompt_budget.assemble(six)
        self.assertGreater(len(full), self.Fast.max_prompt_chars)
        with patch.object(loop.llm, "chat") as model:
            plain, extra = loop._continuation_prompt(self.Fast(), six, [], None, full, "mu-parent")
        model.assert_not_called()
        self.assertIn("She stops and knocks twice.", plain)
        self.assertIn("Rain on cobbles.", plain)
        # The references alone fill the budget: the rarely used fallback is kept whole rather than
        # failing a beat that will most likely continue normally.
        self.assertEqual(full, extra["continuation_fallback_prompt"])

    def test_a_long_shot_is_trimmed_to_fit_without_a_model_call(self):
        six = {**self.fields(), "subject_definitions": "<Subject 1> is the courier in <Picture 1>."}
        six["retention_analysis"] = "Keep the identity."
        six["detailed_description"] += " " + " ".join(f"She waits, beat {i}." for i in range(80))
        full = prompt_budget.assemble(six)
        with patch.object(loop.llm, "chat") as model:
            plain, extra = loop._continuation_prompt(self.Fast(), six, [], None, full, "mu-parent")
        model.assert_not_called()
        self.assertLessEqual(prompt_budget.prompt_size(plain), 900)
        self.assertLessEqual(prompt_budget.prompt_size(extra["continuation_fallback_prompt"]), 900)
        # Trimming drops the end, never the opening action (markers are compiled away here).
        self.assertIn("walks to the door. She stops and knocks twice.", plain)
        self.assertNotIn("beat 79.", plain)

    def test_a_full_prompt_is_still_fitted_when_it_is_what_gets_sent(self):
        self.assertTrue(loop._sends_full_prompt(self.Fast(), None))
        self.assertFalse(loop._sends_full_prompt(self.Fast(), "mu-parent"))


if __name__ == "__main__":
    unittest.main()
