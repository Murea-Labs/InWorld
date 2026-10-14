#!/usr/bin/env python3
"""A continuation whose prefix lacks a cast subject is filmed fresh from the references.

Murea continues from the last ~0.7 s of the previous clip and sends no reference images. When
that tail does not show a subject of this clip (the clip ended on a close-up of the dragon), the
subject is redrawn from text alone: a two-handed runic greatsword came back as a short glowing
axe. The director reports such subjects in `missing_from_last_frame`; the turn then cuts.
"""

from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from murea_agent import config  # noqa: E402
from murea_agent import loop  # noqa: E402
from murea_agent.director import agent_schema  # noqa: E402
from murea_agent.director import prefetch as prefetch_mod  # noqa: E402
from murea_agent.guards import replan  # noqa: E402
from murea_agent.memory.world import Node, World  # noqa: E402
from murea_agent.shot import beat_assets  # noqa: E402


class Submitted(BaseException):
    pass


def _world() -> tuple[World, str, str, str]:
    world = World(id="w-prefix-anchor", title="dragon", here="熔岩峭壁")
    world.add_place("熔岩峭壁", "scene.png")
    hunter = world.add_subject("龙裔猎手", "hunter.png", "a silver-haired warrior", kind="人物")
    dragon = world.add_subject("熔岩巨龙", "dragon.png", "a colossal obsidian dragon", kind="生物")
    sword = world.add_subject(
        "铭文大剑", "sword.png", "a two-handed dark steel greatsword with blue runes", kind="物件"
    )
    world.subjects[sword]["scale_anchor"] = "two-handed weapon, about 1.5 meters long"
    # Locked wording, so the merged director output is used as is (no rewriter call).
    world.subjects[hunter]["definition"] = "a silver-haired warrior in dark iron armor"
    world.subjects[dragon]["definition"] = "a colossal dragon with obsidian scales"
    world.subjects[sword]["definition"] = "a two-handed dark steel greatsword with blue runes"
    world.places["熔岩峭壁"]["definition"] = "a volcanic canyon with rivers of lava"
    parent = Node(
        id="parent",
        parent=None,
        video_id="previous-video",
        clip="previous.mp4",
        place=world.here,
        beat="Close-up of the sleeping dragon.",
        mode="continue",
        refs=["hunter.png", "dragon.png", "scene.png"],
        artifact_id="previous-latent",
    )
    world.nodes[parent.id] = asdict(parent)
    world.head = parent.id
    return world, hunter, dragon, sword


def _decision(cast: list[str], missing: list[str] | None) -> replan._PreparedDecision:
    d = {
        "reply": "你握紧大剑，借着岩石掩护向巨龙潜行。",
        "mode": "continue",
        "place": "熔岩峭壁",
        "subjects": cast,
        "beat": "The hunter creeps toward the dragon, holding the greatsword.",
        "carry": "龙裔猎手 holds 铭文大剑 in both hands",
        "dialogue": "",
        "new_subjects": [],
        "new_places": [],
        "options": [],
        "summary": "[reference_generation] <Subject 1> creeps toward <Subject 2> in <Subject 4>.",
        "detailed_description": "[Shot 1] <Subject 1> creeps toward <Subject 2> holding <Subject 3>.",
        "overall_soundscape": "Lava rumble.",
        "non_diegetic_music": "N/A",
    }
    if missing is not None:
        d["missing_from_last_frame"] = missing
    return replan._PreparedDecision(d)


class MissingFromPrefixHelperTest(unittest.TestCase):
    def test_resolves_names_and_ids_inside_the_cast_only(self) -> None:
        world, hunter, dragon, sword = _world()
        d = {"missing_from_last_frame": ["龙裔猎手", sword, "熔岩峭壁", "不存在的人", hunter]}
        # The place, an unknown name and a duplicate are ignored; order is kept.
        self.assertEqual([hunter, sword], loop._missing_from_prefix(world, d, [hunter, dragon, sword]))

    def test_a_subject_not_cast_in_this_clip_never_forces_a_cut(self) -> None:
        world, hunter, dragon, sword = _world()
        d = {"missing_from_last_frame": ["铭文大剑"]}
        self.assertEqual([], loop._missing_from_prefix(world, d, [hunter, dragon]))

    def test_absent_or_malformed_field_is_empty(self) -> None:
        world, hunter, _dragon, _sword = _world()
        for d in ({}, {"missing_from_last_frame": None}, {"missing_from_last_frame": "龙裔猎手"}):
            with self.subTest(d=d):
                self.assertEqual([], loop._missing_from_prefix(world, d, [hunter]))


class ForcedItemReferenceTest(unittest.TestCase):
    def test_a_forced_passive_item_gets_its_own_reference(self) -> None:
        world, hunter, _dragon, sword = _world()
        plain = beat_assets._pictures_and_refs(
            world, "熔岩峭壁", [hunter, sword], focus_text="猎手拿着大剑潜行"
        )
        self.assertNotIn("sword.png", plain[1])
        forced = beat_assets._pictures_and_refs(
            world, "熔岩峭壁", [hunter, sword], focus_text="猎手拿着大剑潜行", force_item_ids={sword}
        )
        self.assertEqual(["hunter.png", "sword.png", "scene.png"], forced[1])
        self.assertEqual("object", forced[0][1]["kind"])


class SchemaTest(unittest.TestCase):
    def test_field_is_required_and_arrives_before_the_video_is_submitted(self) -> None:
        schema = agent_schema._agent_schema()
        self.assertIn("missing_from_last_frame", schema["required"])
        self.assertNotIn("missing_from_last_frame", agent_schema.AGENT_LATE_FIELDS)
        # Early submission waits for every required field except the late ones.
        part = {k: None for k in schema["required"] if k not in agent_schema.AGENT_LATE_FIELDS}
        self.assertTrue(agent_schema._agent_partial_ready(part, schema))
        part.pop("missing_from_last_frame")
        self.assertFalse(agent_schema._agent_partial_ready(part, schema))


class TurnDecisionTest(unittest.TestCase):
    def _run(self, world: World, decision) -> tuple[tuple, dict, list]:
        progress: list[tuple[str, str]] = []
        with tempfile.TemporaryDirectory() as tmp:
            for ref in ("hunter.png", "dragon.png", "sword.png", "scene.png"):
                (Path(tmp) / ref).touch()
            with (
                patch.object(config, "REFS", Path(tmp)),
                patch.object(prefetch_mod, "_take_prefetched", return_value=copy.deepcopy(decision)),
                patch.object(loop.llm, "chat", side_effect=AssertionError("No extra model call")),
                patch.object(loop.video, "supports_continuation", return_value=True),
                patch.object(loop.video, "generate", side_effect=Submitted) as generate,
                patch.object(World, "save", side_effect=AssertionError("No world writes")),
            ):
                with self.assertRaises(Submitted):
                    loop.turn(world, "握紧巨剑，悄悄接近巨龙", progress=lambda s, m: progress.append((s, m)))
            return generate.call_args.args, generate.call_args.kwargs, progress

    def test_missing_subject_turns_the_continuation_into_a_referenced_cut(self) -> None:
        world, hunter, dragon, sword = _world()
        args, kwargs, progress = self._run(
            world, _decision([hunter, dragon, sword], ["龙裔猎手", "铭文大剑"])
        )
        prompt, refs = args[:2]
        self.assertIsNone(kwargs["continue_from"])
        # Every cast subject is filmed from its reference, including the passively carried sword.
        self.assertEqual(["hunter.png", "dragon.png", "sword.png", "scene.png"], [p.name for p in refs])
        self.assertIn("<Picture", prompt)
        self.assertNotIn("continuation_fallback_prompt", kwargs)
        # The user is told who was missing (zh or en, by the configured language).
        self.assertTrue(
            any("龙裔猎手" in m and "铭文大剑" in m for s, m in progress if s == "image"), progress
        )

    def test_everything_visible_keeps_the_continuation(self) -> None:
        world, hunter, dragon, sword = _world()
        _args, kwargs, _progress = self._run(world, _decision([hunter, dragon, sword], []))
        self.assertEqual("previous-latent", kwargs["continue_from"])

    def test_old_decisions_without_the_field_keep_the_continuation(self) -> None:
        # Prefetched or cached decisions written before this field existed.
        world, hunter, dragon, sword = _world()
        _args, kwargs, _progress = self._run(world, _decision([hunter, dragon, sword], None))
        self.assertEqual("previous-latent", kwargs["continue_from"])

    def test_a_provider_that_sends_references_on_continuation_still_continues(self) -> None:
        world, hunter, dragon, sword = _world()
        with patch.object(type(loop.video.model("murea")), "references_on_continuation", True):
            _args, kwargs, _progress = self._run(world, _decision([hunter, dragon, sword], ["龙裔猎手"]))
        self.assertEqual("previous-latent", kwargs["continue_from"])


if __name__ == "__main__":
    unittest.main()
