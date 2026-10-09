"""Co-op default portrait prompts avoid an unnecessary translation request."""

from __future__ import annotations

import base64
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from murea_agent.coop import portrait as coop_portrait
from murea_agent.providers import llm


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


class CoopPortraitPromptTest(unittest.TestCase):
    def test_peer_design_context_is_opt_in(self) -> None:
        description = "Adult ranger wearing brown leather armor and carrying a sword"
        default = coop_portrait._prompt(
            "Ranger", description, "", media_description=description, distinguish_from_peers=True
        )
        manual = coop_portrait._prompt("Ranger", description, "", media_description=description)
        self.assertIn("existing teammates for design comparison", default)
        self.assertIn("distinct facial structure", default)
        self.assertIn(description, default)
        self.assertNotIn("design comparison", manual)

    def test_builtin_role_uses_english_media_description(self) -> None:
        room = {"setting": "共同对抗沉睡的巨龙", "members": [{"character": ""}]}
        coop_portrait.default_members(room)
        member = room["members"][0]
        prompt = coop_portrait._prompt(
            member["character"],
            member["description"],
            "",
            media_description=coop_portrait._default_media_description(member["character"], room["setting"]),
        )
        self.assertIn("Adult forest guardian", prompt)
        self.assertNotRegex(prompt, r"[\u3400-\u9fff]")

    def test_builtin_prompt_skips_translation_but_manual_prompt_keeps_it(self) -> None:
        room = {"setting": "共同对抗沉睡的巨龙", "members": [{"character": ""}]}
        coop_portrait.default_members(room)
        member = room["members"][0]
        builtin = coop_portrait._prompt(
            member["character"],
            member["description"],
            "",
            media_description=coop_portrait._default_media_description(member["character"], room["setting"]),
        )
        manual = coop_portrait._prompt("自定义角色", "银发蓝眼，轻甲长剑", "")

        with (
            patch.object(llm, "chat", return_value={"value": "A custom adult character"}) as chat,
            patch("murea_agent.providers.imagegen._post", return_value=_image_response()),
            patch("murea_agent.providers.imagegen.reqlog.record"),
        ):
            from murea_agent.providers import imagegen

            imagegen.generate(builtin, kind="subject")
            self.assertEqual(0, chat.call_count)
            imagegen.generate(manual, kind="subject")
            self.assertEqual(1, chat.call_count)


if __name__ == "__main__":
    unittest.main()
