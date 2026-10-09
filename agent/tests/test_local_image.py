"""Self-hosted image generation: Qwen-Image + Lightning distillation LoRA.

⚠️ Image drawing is **awaited with `.result()` before the video submission** (`_collect_pending`), not a background
task. When the agent took 4.7 s the drawing hid inside it; once the agent was down to 0.88 s, beats
that draw new images (about 21%) waited on drawing instead.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from murea_agent import config
from murea_agent.providers import imagegen


class Routing(unittest.TestCase):
    def test_empty_base_keeps_gemini(self) -> None:
        with mock.patch.object(config, "IMAGE_BASE", ""), mock.patch.object(imagegen, "_post") as post:
            post.return_value = {"candidates": []}
            with self.assertRaises(imagegen.ImageRefused):
                imagegen.generate("一条木栈道", kind="place")
            post.assert_called_once()

    def test_base_set_goes_local_and_never_touches_gemini(self) -> None:
        with (
            mock.patch.object(config, "IMAGE_BASE", "http://127.0.0.1:8791"),
            mock.patch.object(imagegen, "_post") as post,
            mock.patch.object(imagegen, "_local_draw", return_value=b"\x89PNG") as loc,
        ):
            got = imagegen.generate("一条木栈道", kind="place")
        self.assertEqual(got, b"\x89PNG")
        loc.assert_called_once()
        post.assert_not_called()

    def test_local_failure_is_loud_not_silent_fallback(self) -> None:
        """Silently falling back to the cloud would hide whether the local service is used at all, and latency is the only
        reason this path exists. Failures must be reported.
        """

        class R:
            status_code = 500
            text = "boom"
            content = b""

        with (
            mock.patch.object(config, "IMAGE_BASE", "http://127.0.0.1:8791"),
            mock.patch.object(imagegen.httpx, "post", return_value=R()),
        ):
            with self.assertRaises(imagegen.ImageRefused):
                imagegen._local_draw("x", "place")

    def test_empty_body_is_rejected(self) -> None:
        """200 with zero bytes: written to disk it is a broken image used as a reference in every later beat."""

        class R:
            status_code = 200
            text = ""
            content = b""

        with (
            mock.patch.object(config, "IMAGE_BASE", "http://127.0.0.1:8791"),
            mock.patch.object(imagegen.httpx, "post", return_value=R()),
        ):
            with self.assertRaises(imagegen.ImageRefused):
                imagegen._local_draw("x", "place")


class Replicas(unittest.TestCase):
    """Several replicas round robin: when a beat draws two images they run in parallel."""

    def test_round_robin_spreads_across_replicas(self) -> None:
        seen = []

        class R:
            status_code = 200
            text = ""
            content = b"\x89PNG"

        def fake_post(url, json=None, timeout=None):
            seen.append(url)
            return R()

        bases = "http://a/,http://b/,http://c/"
        with (
            mock.patch.object(config, "IMAGE_BASE", bases),
            mock.patch.object(imagegen.httpx, "post", fake_post),
        ):
            for _ in range(6):
                imagegen._local_draw("x", "place")
        hosts = {u.split("/generate")[0] for u in seen}
        self.assertEqual(len(hosts), 3, "六次调用要摊到三个副本上")

    def test_trailing_slash_and_spaces_are_tolerated(self) -> None:
        with mock.patch.object(config, "IMAGE_BASE", " http://a/ , http://b "):
            self.assertEqual(imagegen._local_bases(), ["http://a", "http://b"])

    def test_a_configured_but_unparsable_base_is_loud(self) -> None:
        """With IMAGE_BASE set but no endpoint parsed, never fall back to the cloud silently."""
        with mock.patch.object(config, "IMAGE_BASE", " , , "):
            with self.assertRaises(imagegen.ImageRefused):
                imagegen._local_draw("x", "place")


class Sizes(unittest.TestCase):
    def test_everything_is_wide_like_the_cloud_does(self) -> None:
        """⚠️ Match the cloud model's aspect: it draws 1376×768 (places) and 1408×768 (characters), **character images are
        wide too**. A square image would be a different kind of reference and change the video
        model's composition habits.
        """
        for kind, (w, h) in imagegen._LOCAL_SIZE.items():
            with self.subTest(kind=kind):
                self.assertGreater(w, h, f"{kind} 应该是宽的")
                self.assertAlmostEqual(w / h, 16 / 9, places=1)

    def test_unknown_kind_falls_back_to_place(self) -> None:
        class R:
            status_code = 200
            text = ""
            content = b"\x89PNG"

        seen = {}

        def fake_post(url, json=None, timeout=None):
            seen.update(json)
            return R()

        with (
            mock.patch.object(config, "IMAGE_BASE", "http://x"),
            mock.patch.object(imagegen.httpx, "post", fake_post),
        ):
            imagegen._local_draw("x", "没见过的类型")
        self.assertEqual((seen["width"], seen["height"]), imagegen._LOCAL_SIZE["place"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
