"""Object reference selection and prop-conditioning regression tests."""

from __future__ import annotations

import sys
import copy
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from murea_agent import config
from murea_agent.shot import definitions
from murea_agent.director import prefetch as prefetch_mod  # noqa: E402
from murea_agent.guards import replan  # noqa: E402
from murea_agent.shot import beat_assets, video_prompt
from murea_agent import loop
from murea_agent.memory.world import Node, World


def _world() -> tuple[World, str, str]:
    world = World(id="w-object-policy", title="objects", here="会议室")
    world.add_place("会议室", "scene.png")
    actor = world.add_subject("小明", "actor.png", kind="人物")
    item = world.add_subject(
        "铜灯笼",
        "lantern.png",
        "a brass lantern with a warm flame",
        kind="物件",
    )
    world.subjects[item]["scale_anchor"] = "handheld, about 25 cm tall"
    return world, actor, item


class ObjectReferencePolicyTest(unittest.TestCase):
    def test_submission_keeps_refs_and_latent_with_clean_object_definition(self) -> None:
        class Submitted(BaseException):
            pass

        for continuation in (False, True):
            with self.subTest(continuation=continuation), tempfile.TemporaryDirectory() as tmp:
                world, actor, item = _world()
                world.subjects[actor]["definition"] = "a traveler wearing a white jacket"
                world.subjects[item]["definition"] = (
                    "the character shown in <Picture 2> as a single full-body view; "
                    "a brass lantern with a warm flame and a curved copper handle"
                )
                world.places[world.here]["definition"] = "a room with white walls and a wooden table"
                refs = ["actor.png", "lantern.png", "scene.png"]
                for ref in refs:
                    (Path(tmp) / ref).touch()
                if continuation:
                    parent = Node(
                        id="parent",
                        parent=None,
                        video_id="previous-video",
                        clip="previous.mp4",
                        place=world.here,
                        beat="The traveler holds the lantern.",
                        mode="continue",
                        refs=refs,
                        artifact_id="previous-latent",
                    )
                    world.nodes[parent.id] = asdict(parent)
                    world.head = parent.id
                decision = replan._PreparedDecision(
                    {
                        "reply": "你仔细端详手中的铜灯笼。",
                        "mode": "continue",
                        "place": world.here,
                        "subjects": [actor, item],
                        "beat": "The traveler examines the brass lantern.",
                        "dialogue": "",
                        "new_subjects": [],
                        "new_places": [],
                        "options": [],
                        "summary": "[reference_generation] <Subject 1> examines <Subject 2> in <Subject 3>.",
                        "detailed_description": "[Shot 1] <Subject 1> examines <Subject 2> at the table in <Subject 3>.",
                        "overall_soundscape": "Quiet room tone.",
                        "non_diegetic_music": "N/A",
                    }
                )
                with (
                    patch.object(config, "REFS", Path(tmp)),
                    patch.object(prefetch_mod, "_take_prefetched", return_value=copy.deepcopy(decision)),
                    patch.object(loop.llm, "chat", side_effect=AssertionError("No extra model call")),
                    patch.object(loop.video, "supports_continuation", return_value=True),
                    patch.object(loop.video, "generate", side_effect=Submitted) as generate,
                    patch.object(World, "save", side_effect=AssertionError("No world writes")),
                ):
                    with self.assertRaises(Submitted):
                        loop.turn(world, "给铜灯笼一个特写")
                prompt, sent_refs = generate.call_args.args[:2]
                self.assertEqual(refs, [p.name for p in sent_refs])
                self.assertIn("curved copper handle", prompt)
                self.assertIn("white jacket", prompt)
                if continuation:
                    # Murea continues from the frames without images: plain prose, and the full
                    # prompt as the fallback for a missing parent clip.
                    self.assertEqual("previous-latent", generate.call_args.kwargs["continue_from"])
                    self.assertNotIn("<Picture", prompt)
                    full = generate.call_args.kwargs["continuation_fallback_prompt"]
                    self.assertIn("<Subject 3> provides the background and lighting", full)
                    self.assertNotIn("the character shown in <Picture 2>", full)
                else:
                    self.assertIsNone(generate.call_args.kwargs["continue_from"])
                    self.assertNotIn("the character shown in <Picture 2>", prompt)
                    self.assertIn("<Subject 3> provides the background and lighting", prompt)

    def test_legacy_object_definition_keeps_appearance_without_character_wrapper(self) -> None:
        world, actor, item = _world()
        appearance = (
            "A rectangular heavy-duty metal case, dark gray with rust and silt, "
            "a sturdy ribbed handle on top, two brass latches on its front and "
            "a small red enamel badge on the right corner"
        )
        world.subjects[item]["definition"] = (
            "the single inanimate object prop shown in <Picture 2> as a single "
            "isolated object reference; the character shown in <Picture 2> as a "
            f"single full-body view; {appearance}; scale anchor: handheld; "
            "use the reference only for identity, shape, material, and color"
        )
        world.subjects[actor]["definition"] = "a traveler wearing a white jacket"
        world.places["会议室"]["definition"] = "a room with white walls and a wooden table"
        # Deliberately change the old numbers to check that continuation keeps asset identity, not
        # historical Picture numbers.
        pictures = [
            {"n": 1, "kind": "object", "id": item, "name": "铜灯笼"},
            {"n": 2, "kind": "character", "id": actor, "name": "小明"},
            {"n": 3, "kind": "scene", "id": "会议室", "name": "会议室"},
        ]
        locked, missing = video_prompt._compose_definitions(world, pictures)
        self.assertEqual([], missing)
        fields = {"subject_definitions": locked}
        out = video_prompt._lock_object_definitions(fields, pictures, world, {}, "")
        item_line = out["subject_definitions"].splitlines()[0]
        self.assertIn(appearance, item_line)
        self.assertEqual(1, item_line.count("<Picture 1>"))
        self.assertNotIn("the character", item_line)
        self.assertNotIn("full-body", item_line)
        self.assertEqual(out, video_prompt._lock_object_definitions(out, pictures, world, {}, ""))
        self.assertEqual(appearance, definitions._locked_definitions(world)["铜灯笼"])

        retention = video_prompt._build_retention(pictures, world, {"framing": "medium"})
        item_line, person_line, scene_line = retention.splitlines()
        self.assertNotIn(appearance, item_line)
        self.assertIn("shape, material and color retained", item_line)
        self.assertNotIn("the character", item_line)
        self.assertNotIn("full-body", item_line)
        for line in (item_line, person_line):
            self.assertIn("<Subject 3> provides the background and lighting", line)
        self.assertNotIn("white jacket", person_line)
        self.assertNotIn("white walls", scene_line)
        self.assertIn("layout, materials and lighting retained", scene_line)

    def test_passive_carried_item_uses_parent_latent(self) -> None:
        world, actor, item = _world()
        pictures, refs, _ = beat_assets._pictures_and_refs(
            world,
            "会议室",
            [actor, item],
            motion_plan={"target": "", "relation_change": "none"},
            focus_text="小明继续往前走，手里拿着铜灯笼",
        )
        self.assertEqual([actor, "会议室"], [p["id"] for p in pictures])
        self.assertEqual(["actor.png", "scene.png"], refs)

    def test_interaction_target_keeps_object_reference(self) -> None:
        world, actor, item = _world()
        pictures, refs, _ = beat_assets._pictures_and_refs(
            world,
            "会议室",
            [actor, item],
            motion_plan={
                "target": "铜灯笼",
                "relation_change": "possession",
            },
        )
        self.assertEqual([actor, item], [p["id"] for p in pictures[:2]])
        self.assertEqual("object", pictures[1]["kind"])
        self.assertEqual(["actor.png", "lantern.png", "scene.png"], refs)

    def test_structured_instrument_keeps_object_reference(self) -> None:
        world, actor, item = _world()
        pictures, refs, _ = beat_assets._pictures_and_refs(
            world,
            "会议室",
            [actor, item],
            motion_plan={
                "interaction_type": "impact",
                "instrument": "铜灯笼",
                "target": "墙上的标记",
                "relation_change": "contact",
            },
            action_step={
                "actors": ["小明"],
                "instruments": ["铜灯笼"],
                "targets": ["墙上的标记"],
            },
        )
        self.assertEqual([actor, item], [p["id"] for p in pictures[:2]])
        self.assertEqual(["actor.png", "lantern.png", "scene.png"], refs)

    def test_step_instrument_keeps_reference_when_plan_uses_another_target(self) -> None:
        world, actor, item = _world()
        pictures, _, _ = beat_assets._pictures_and_refs(
            world,
            "会议室",
            [actor, item],
            motion_plan={"target": "机械装置", "relation_change": "contact"},
            action_step={"instruments": ["铜灯笼"], "targets": ["机械装置"]},
        )
        self.assertIn(item, [p["id"] for p in pictures])

    def test_close_up_keeps_object_reference(self) -> None:
        world, actor, item = _world()
        pictures, _, _ = beat_assets._pictures_and_refs(
            world,
            "会议室",
            [actor, item],
            focus_text="给铜灯笼一个特写",
        )
        self.assertIn(item, [p["id"] for p in pictures])

    def test_character_close_up_does_not_promote_carried_item(self) -> None:
        world, actor, item = _world()
        pictures, _, _ = beat_assets._pictures_and_refs(
            world,
            "会议室",
            [actor, item],
            focus_text="给小明一个特写，他手里拿着铜灯笼继续说话",
        )
        self.assertNotIn(item, [p["id"] for p in pictures])

    def test_hard_cut_also_omits_passive_item(self) -> None:
        world, actor, item = _world()
        pictures, _, _ = beat_assets._pictures_and_refs(
            world,
            "会议室",
            [actor, item],
        )
        self.assertNotIn(item, [p["id"] for p in pictures])

    def test_at_mention_alone_does_not_promote_passive_item(self) -> None:
        world, actor, item = _world()
        pictures, _, _ = beat_assets._pictures_and_refs(
            world,
            "会议室",
            [actor, item],
            required_subject_ids={item},
            motion_plan={"target": "", "relation_change": "none"},
            focus_text="@铜灯笼 仍然被小明普通地拿在手里",
        )
        self.assertNotIn(item, [p["id"] for p in pictures])

    def test_focal_item_without_a_ref_is_queued_as_object_image(self) -> None:
        world, actor, item = _world()
        world.subjects[item]["ref"] = ""
        pending = []
        beat_assets._queue_missing_focal_object_refs(
            world,
            [actor, item],
            pending,
            motion_plan={"target": "铜灯笼", "relation_change": "possession"},
            focus_text="拿起铜灯笼",
        )
        self.assertEqual(1, len(pending))
        self.assertEqual("object", pending[0].asset_kind)
        self.assertEqual("pending", world.subjects[item]["ref_status"])

    def test_video_constraints_make_object_role_and_scale_explicit(self) -> None:
        world, actor, item = _world()
        pictures = [
            {"n": 1, "kind": "character", "id": actor, "name": "小明"},
            {"n": 2, "kind": "object", "id": item, "name": "铜灯笼"},
        ]
        fields = {
            "subject_definitions": "<Subject 2> is a brass lantern",
            "detailed_description": "[Shot 1] 小明拿起铜灯笼。",
        }
        out = video_prompt._lock_object_reference_constraints(
            fields,
            pictures,
            world,
            {"actors": ["小明"], "target": "铜灯笼", "relation_change": "possession"},
            "手里拿着铜灯笼",
            focus_text="拿起铜灯笼",
        )
        self.assertIn("single inanimate object", out["subject_definitions"])
        self.assertIn("<Picture 2>", out["subject_definitions"])
        self.assertIn("handheld, about 25 cm tall", out["detailed_description"])
        self.assertIn("no body, face, limbs", out["detailed_description"])
        self.assertIn(
            "only for its identity, shape, material, and color",
            out["detailed_description"],
        )
        self.assertIn("no duplicate foreground copy", out["detailed_description"])

        again = video_prompt._lock_object_reference_constraints(
            out,
            pictures,
            world,
            {"actors": ["小明"], "target": "铜灯笼", "relation_change": "possession"},
            "手里拿着铜灯笼",
            focus_text="拿起铜灯笼",
        )
        self.assertEqual(out, again)


if __name__ == "__main__":
    unittest.main()
