"""Prompt regression tests per image asset type (no calls to the image model)."""

from __future__ import annotations

import base64
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from murea_agent.providers import imagegen


def _image_response() -> dict:
    encoded = base64.b64encode(b"fake-image").decode()
    return {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"inlineData": {"data": encoded}},
                    ]
                }
            }
        ]
    }


class ImagePromptTest(unittest.TestCase):
    def test_object_is_white_background_on_both_image_backends(self) -> None:
        description = "A white delivery van with blue stripes and a roof rack"
        style = "Cinematic industrial realism, moody warehouse lighting"
        prompts = []
        for base in ("", "http://image.test"):
            with (
                self.subTest(backend=base),
                patch.object(imagegen.config, "IMAGE_BASE", base),
                patch.object(imagegen, "_post", return_value=_image_response()) as post,
                patch.object(imagegen, "_local_draw", return_value=b"image") as local,
                patch.object(imagegen.reqlog, "record"),
            ):
                imagegen.generate(description, kind="object", visual_style=style)
                prompt = (
                    local.call_args.args[0]
                    if base
                    else post.call_args.args[1]["contents"][0]["parts"][0]["text"]
                )
                prompts.append(prompt)
                self.assertIn(description, prompt)
                self.assertIn("pure white (#FFFFFF) background", prompt)
                self.assertIn("materials, colors and functional parts", prompt)
                self.assertNotIn("no vehicle", prompt)
                self.assertNotIn("SUBJECT-ONLY", prompt)
        self.assertTrue(prompts[1].endswith(prompts[0]))

    def test_story_location_cannot_override_entity_reference_instruction(self) -> None:
        for kind in ("subject", "object", "place"):
            with (
                self.subTest(kind=kind),
                patch.object(imagegen.config, "IMAGE_BASE", ""),
                patch.object(imagegen, "_post", return_value=_image_response()) as post,
                patch.object(imagegen.reqlog, "record"),
            ):
                imagegen.generate("A student standing in an abandoned classroom", kind=kind)
                body = post.call_args.args[1]
                if kind == "place":
                    self.assertNotIn("systemInstruction", body)
                else:
                    instruction = body["systemInstruction"]["parts"][0]["text"]
                    self.assertIn("Discard all locations", instruction)
                    self.assertIn("uniform pure white", instruction)
                    self.assertIn(
                        "inanimate object" if kind == "object" else "character or creature", instruction
                    )

    def _prompt(self, kind: str, style: str = "") -> str:
        with (
            patch.object(imagegen.config, "IMAGE_BASE", ""),
            patch.object(imagegen, "_post", return_value=_image_response()) as post,
            patch.object(imagegen.reqlog, "record"),
        ):
            imagegen.generate(
                "A person standing behind a counter in a warehouse",
                kind=kind,
                visual_style=style,
            )
        return post.call_args.args[1]["contents"][0]["parts"][0]["text"]

    def test_entity_reference_overrides_cinematography_but_scene_keeps_it(self) -> None:
        style = "cyberpunk neon cinematography, volumetric fog, heavy bokeh, slight chromatic aberration"
        for kind in ("subject", "object"):
            for art in (style, ""):
                with self.subTest(kind=kind, style=art):
                    prompt = self._prompt(kind, art)
                    self.assertIn("sharp focus across the entire entity", prompt)
                    self.assertIn("intrinsic colors and materials", prompt)
                    self.assertNotIn("Do not switch medium", prompt)
        scene = self._prompt("place", style)
        self.assertIn(style, scene)
        self.assertIn("lighting style, or camera treatment", scene)
        self.assertNotIn("appearance only", scene)
        self.assertNotIn("No depth-of-field blur", scene)

    def test_subject_is_isolated_from_story_environment(self) -> None:
        prompt = self._prompt("subject", "dark cinematic lighting")
        self.assertIn("one complete full-body character or creature", prompt)
        self.assertIn("uniform pure white (#FFFFFF) background", prompt)
        self.assertIn("neutral standing pose", prompt)
        self.assertIn("Use the description for its stable appearance", prompt)
        self.assertIn("dark cinematic lighting", prompt)

    def test_place_and_object_do_not_receive_subject_rule(self) -> None:
        for kind in ("place", "object"):
            with self.subTest(kind=kind):
                prompt = self._prompt(kind)
                self.assertNotIn("SUBJECT-ONLY REFERENCE", prompt)


if __name__ == "__main__":
    unittest.main()
