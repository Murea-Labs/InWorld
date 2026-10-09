"""Every model call that really runs must match the signature of `llm.chat`.

⚠️ Added after a real failure: removing the `stop_when` parameter from `llm.chat` missed the call in `_agent_step`,
so **every beat raised**

    TypeError: chat() got an unexpected keyword argument 'stop_when'

while all offline suites were green, because they all mocked `llm.chat` and none bound arguments
against the real signature. The first beat on the site hit it.
"""

from __future__ import annotations

import contextlib
import inspect
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from murea_agent.providers import llm
from murea_agent.director import director  # noqa: E402
from murea_agent import loop
from murea_agent.memory.world import World


class BindsAgainstTheRealSignature(unittest.TestCase):
    """Bind the arguments against the real signature once; a mock taking `**kwargs` tests nothing."""

    def _spy(self):
        """A stand-in whose signature is **identical** to `llm.chat`. One extra argument raises TypeError."""
        sig = inspect.signature(llm.chat)
        seen = {}

        def spy(**kw):
            sig.bind(**kw)  # ← the real check is this line
            seen.update(kw)
            return {}

        return spy, seen

    def test_agent_step_matches(self) -> None:
        spy, seen = self._spy()
        with mock.patch.object(llm, "chat", spy), contextlib.suppress(ValueError):
            director._agent_step(World(title="t"), "走")
        self.assertIn("system", seen)
        self.assertIn("schema", seen)

    def test_prefetch_path_matches(self) -> None:
        """Prefetch goes through `_decide`, the same path as a normal beat, but passes an extra base."""
        spy, seen = self._spy()
        with (
            mock.patch.object(llm, "chat", spy),
            contextlib.suppress(ValueError),
        ):  # the spy's empty reply is not a valid plan
            director._decide(World(title="t"), "走", base="http://x/v1")
        self.assertEqual(seen.get("base"), "http://x/v1")

    def test_every_llm_call_names_its_model(self) -> None:
        """⚠️ Without `model=` it falls back to `config.TEXT_MODEL`.

        After switching the agent to another model, some calls would silently keep using the old
        one; the critic did, visible only by reading the logs.
        """
        import re

        src = inspect.getsource(loop)
        for m in re.finditer(r"llm\.chat\(", src):
            # the text from this call to its closing parenthesis
            i = m.end()
            depth, j = 1, i
            while j < len(src) and depth:
                depth += (src[j] == "(") - (src[j] == ")")
                j += 1
            call = src[i:j]

            tag = re.search(r'tag="([^"]+)"', call)
            with self.subTest(tag=(tag.group(1) if tag else "?")):
                self.assertIn("model=", call, "每次 llm.chat 都要指名模型，否则落回 TEXT_MODEL")

    def test_no_dead_keyword_survives_in_the_source(self) -> None:
        """Missing a call site when removing a parameter is the general shape of this failure. Removed parameters must not appear in the source again."""
        src = inspect.getsource(loop)
        for gone in ("stop_when", "no_frame", "SPLIT_AGENT", "_decide_split"):
            with self.subTest(gone=gone):
                self.assertNotIn(gone, src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
