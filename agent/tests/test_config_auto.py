"""Providers are chosen from the keys in .env when config.yaml does not name them."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROBE = (
    "import json; from murea_agent import config as c; print(json.dumps([c.LLM_PROVIDER, c.LLM_BACKEND, "
    "c.GM_MODEL, c.IMAGE_PROVIDER, c.IMAGE_MODEL, c.VIDEO_MODEL, c.OPENAI_BASE, c.MUREA_URL, "
    "c.MUREA_API_KEY]))"
)


def resolve(keys: dict[str, str], config: str = "", dotenv: str = "") -> list[str]:
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Path(tmp) / "config.yaml"
        cfg.write_text(config or "mode: single\n")
        if dotenv:
            (Path(tmp) / ".env").write_text(dotenv)
        # The key variables come only from this test, never from the developer's shell or .env.
        env = {k: v for k, v in os.environ.items() if not k.endswith("_API_KEY") and k != "MUREA_URL"}
        env.update(keys, MUREA_AGENT_CONFIG=str(cfg), PYTHONPATH=str(ROOT))
        out = subprocess.run([sys.executable, "-c", PROBE], env=env, cwd=tmp, capture_output=True, text=True)
        if out.returncode:
            raise RuntimeError(out.stderr.strip().splitlines()[-1])
        return json.loads(out.stdout)


class AutoProviderTest(unittest.TestCase):
    def test_two_keys_are_enough(self) -> None:
        llm, backend, model, image, image_model, video, base, *_ = resolve(
            {"DASHSCOPE_API_KEY": "d", "MUREA_URL": "http://gpu:30260"}
        )
        self.assertEqual(("dashscope", "openai", "deepseek-v4.1-flash"), (llm, backend, model))
        self.assertEqual(("dashscope", "qwen-image-2.0", "murea"), (image, image_model, video))
        self.assertIn("dashscope", base)

    def test_gemini_key_alone(self) -> None:
        llm, backend, _, image, *_ = resolve({"GEMINI_API_KEY": "g"})
        self.assertEqual(("gemini", "gemini", "gemini"), (llm, backend, image))

    def test_dashscope_wins_over_gemini(self) -> None:
        self.assertEqual("dashscope", resolve({"GEMINI_API_KEY": "g", "DASHSCOPE_API_KEY": "d"})[0])

    def test_openai_needs_model_names(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "llm.models.agent"):
            resolve({"OPENAI_API_KEY": "o"})
        got = resolve({"OPENAI_API_KEY": "o"}, "llm: {provider: openai, models: {agent: my-model}}\n")
        self.assertEqual(("openai", "my-model"), (got[0], got[2]))

    def test_config_names_win_over_keys(self) -> None:
        got = resolve({"DASHSCOPE_API_KEY": "d"}, "llm: {provider: gemini}\n")
        self.assertEqual("gemini", got[0])


class MureaAddressTest(unittest.TestCase):
    def test_from_env_or_config(self) -> None:
        got = resolve({"MUREA_URL": "http://10.0.0.5:30260/"})
        self.assertEqual(("murea", "http://10.0.0.5:30260", ""), (got[5], got[7], got[8]))
        got = resolve({"MUREA_API_KEY": "k"}, "video: {url: 'http://gpu:30260'}\n")
        self.assertEqual(("http://gpu:30260", "k"), (got[7], got[8]))

    def test_dotenv_inline_comments_are_not_part_of_the_value(self) -> None:
        # The README writes `MUREA_URL=http://...   # your Murea-Fast service`; the comment was
        # read into the address (and into API keys written the same way).
        got = resolve(
            {},
            dotenv="MUREA_URL=http://gpu:30260   # your Murea-Fast service\nMUREA_API_KEY='k#1' # quoted\n",
        )
        self.assertEqual(("http://gpu:30260", "k#1"), (got[7], got[8]))


if __name__ == "__main__":
    unittest.main()
