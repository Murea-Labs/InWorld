"""Regression tests for the two consistency layers most easily broken silently.

Pure logic, no model calls, milliseconds; run before every commit.

Both are problems invisible in the UI while generation quality has already collapsed:
  · locked descriptions (layer 2) corrupted by bad splitting, and never fixable afterwards
  · continuation handles (layer 1) that cannot be sent silently degrade to unconditioned generation, and people change
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from murea_agent import util
from murea_agent.shot import beat_assets, video_prompt
from murea_agent.providers import video
from murea_agent.memory.world import World

MAN = "the man in a blue shirt shown in <Picture 1>, standing with a neutral expression."
ROOM = "the meeting room shown in <Picture 2>, with a long wooden table and a whiteboard."

PICTURES = [
    {"n": 1, "kind": "character", "id": "S1", "name": "阿明", "desc": "", "ref": "s1.png"},
    {"n": 2, "kind": "scene", "id": "会议室", "name": "会议室", "desc": "", "ref": "p1.png"},
]


def _world() -> World:
    w = World(id="wtest", title="t", setting="", visual_style="", here="会议室")
    w.subjects["S1"] = {"name": "阿明", "ref": "s1.png", "desc": ""}
    w.places["会议室"] = {"ref": "p1.png", "desc": "", "first_seen": ""}
    return w


class LockedDefinitionTest(unittest.TestCase):
    """The boundary in `subject_definitions` is the next `<Subject N> is`, not a newline."""

    def test_two_definitions_on_one_line_do_not_merge(self):
        """The rewriter sometimes omits newlines. Splitting by line lets the first definition swallow the second."""
        w = _world()
        video_prompt._learn_definitions(
            w, PICTURES, {"subject_definitions": f"<Subject 1> is {MAN} <Subject 2> is {ROOM}"}
        )
        self.assertEqual(MAN, w.subjects["S1"]["definition"])
        self.assertEqual(ROOM, w.places["会议室"]["definition"])
        self.assertNotIn("<Subject", w.subjects["S1"]["definition"])

    def test_newlines_still_work(self):
        w = _world()
        video_prompt._learn_definitions(
            w, PICTURES, {"subject_definitions": f"<Subject 1> is {MAN}\n<Subject 2> is {ROOM}"}
        )
        self.assertEqual(MAN, w.subjects["S1"]["definition"])
        self.assertEqual(ROOM, w.places["会议室"]["definition"])

    def test_wrapped_definition_keeps_its_tail(self):
        """When a description spans lines, splitting by line drops the second half."""
        w = _world()
        video_prompt._learn_definitions(
            w,
            PICTURES,
            {
                "subject_definitions": "<Subject 1> is the man in a blue shirt shown in <Picture 1>,\n"
                "standing with a neutral expression.\n"
                f"<Subject 2> is {ROOM}",
            },
        )
        self.assertEqual(MAN, w.subjects["S1"]["definition"])

    def test_settled_definition_is_never_overwritten(self):
        """Settled wording must never change: this is the foundation against character drift and must survive the fix above."""
        w = _world()
        w.subjects["S1"]["definition"] = MAN
        video_prompt._learn_definitions(
            w, PICTURES, {"subject_definitions": "<Subject 1> is somebody completely different."}
        )
        self.assertEqual(MAN, w.subjects["S1"]["definition"])

    def test_polluted_definition_is_healed_instead_of_locked_in(self):
        """Worlds already corrupted must recover, or they carry contradictory conditioning forever."""
        w = _world()
        w.subjects["S1"]["definition"] = f"{MAN} <Subject 2> is {ROOM}"
        video_prompt._learn_definitions(
            w, PICTURES, {"subject_definitions": f"<Subject 1> is {MAN}\n<Subject 2> is {ROOM}"}
        )
        self.assertEqual(MAN, w.subjects["S1"]["definition"])

    def test_polluted_definition_is_not_sent_out(self):
        """Even before recovery, contaminated text must not be put into the prompt."""
        w = _world()
        w.subjects["S1"]["definition"] = f"{MAN} <Subject 2> is {ROOM}"
        w.places["会议室"]["definition"] = ROOM
        locked, missing = video_prompt._compose_definitions(w, PICTURES)
        self.assertEqual([], missing)
        self.assertEqual(1, locked.count("<Subject 1>"))
        self.assertEqual(1, locked.count("<Subject 2>"))


class RefCapTest(unittest.TestCase):
    """The provider accepts 1–5 images and the place takes one, so at most 4 people per shot."""

    def _world(self, n: int) -> World:
        w = _world()
        for i in range(n):
            w.subjects[f"S{i + 1}"] = {"name": f"人{i + 1}", "ref": f"s{i + 1}.png", "desc": ""}
        return w

    def test_four_people_all_get_their_reference(self):
        w = self._world(4)
        pics, refs, dropped = beat_assets._pictures_and_refs(w, "会议室", ["S1", "S2", "S3", "S4"])
        self.assertEqual(5, len(refs))  # 4 people + place
        self.assertEqual([], dropped)
        self.assertEqual("scene", pics[-1]["kind"])  # the place is always the last image

    def test_a_fifth_person_is_dropped_not_fatal(self):
        """A VideoError would show a red bar, a wasted wait and an orphan file on disk. Without one reference that person looks
        less alike, but the clip exists; the latter is better.
        """
        w = self._world(6)
        pics, refs, dropped = beat_assets._pictures_and_refs(w, "会议室", [f"S{i + 1}" for i in range(6)])
        self.assertEqual(5, len(refs))
        self.assertEqual(["人5", "人6"], dropped)
        self.assertEqual(["人1", "人2", "人3", "人4"], [p["name"] for p in pics if p["kind"] == "character"])

    def test_picture_numbers_stay_contiguous_after_a_drop(self):
        """<Picture N> numbers follow the images sent; dropping someone must not leave a gap in the middle."""
        w = self._world(6)
        pics, refs, _ = beat_assets._pictures_and_refs(w, "会议室", [f"S{i + 1}" for i in range(6)])
        self.assertEqual(list(range(1, len(refs) + 1)), [p["n"] for p in pics])

    def test_transition_keeps_only_destination_scene_reference(self):
        """A transition still uploads only the destination place, so start and end images do not compete in one shot."""
        w = _world()
        w.places["走廊"] = {"ref": "p0.png", "desc": "origin", "first_seen": ""}
        pics, refs, dropped = beat_assets._pictures_and_refs(w, "会议室", ["S1"])
        self.assertEqual([], dropped)
        self.assertEqual(["阿明", "会议室"], [p["name"] for p in pics])
        self.assertEqual(["s1.png", "p1.png"], refs)
        self.assertEqual([1, 2], [p["n"] for p in pics])

    def test_scene_reference_does_not_reduce_character_capacity(self):
        """With a single place image the provider still allows up to four characters or objects in the shot."""
        w = self._world(4)
        pics, refs, dropped = beat_assets._pictures_and_refs(
            w,
            "会议室",
            [f"S{i + 1}" for i in range(4)],
        )
        self.assertEqual(5, len(refs))
        self.assertEqual([], dropped)
        self.assertEqual(["人1", "人2", "人3", "人4", "会议室"], [p["name"] for p in pics])


class ImageTypeTest(unittest.TestCase):
    """Image models often return JPEG while it was saved as png and uploaded as image/png. The provider did not mind, but
    it once caused a confusing invalid_image error.
    """

    def test_sniffs_real_type_not_the_extension(self):
        from murea_agent.providers import imagegen

        self.assertEqual((".jpg", "image/jpeg"), imagegen.sniff(b"\xff\xd8\xff\xe0..."))
        self.assertEqual((".png", "image/png"), imagegen.sniff(b"\x89PNG\r\n\x1a\n.."))
        self.assertEqual((".webp", "image/webp"), imagegen.sniff(b"RIFF????WEBP"))

    def test_unknown_bytes_fall_back_to_png(self):
        """Unknown bytes keep the old behavior instead of turning an image into an upload failure."""
        from murea_agent.providers import imagegen

        self.assertEqual((".png", "image/png"), imagegen.sniff(b"????"))


class VideoQueueTest(unittest.TestCase):
    def test_queue_is_visible_and_still_serializes_video(self):
        import threading
        from concurrent.futures import ThreadPoolExecutor

        lock = threading.Lock()
        lock.acquire()
        waiting = threading.Event()
        result = video.Generation("test", Path("unused.mp4"), 4, 0, {})
        with (
            patch.object(video, "_gpu_lock", lock),
            patch.object(video.model(), "generate", return_value=result) as generate,
        ):
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(video.generate, "p", on_queue=waiting.set)
                try:
                    self.assertTrue(waiting.wait(1))
                    self.assertFalse(future.done())
                    generate.assert_not_called()
                finally:
                    lock.release()
                got = future.result(timeout=1)
        self.assertIn("local_queue_seconds", got.params)
        generate.assert_called_once_with("p")


class FastStartTest(unittest.TestCase):
    """Provider mp4s must be remuxed to faststart, or the browser downloads the whole file before playing."""

    def test_moov_is_moved_to_the_front_after_download(self):
        """Real problem: 8 of 8 recent clips had moov at the end of the file.

        moov at the end means the player must download the whole file before decoding the first
        frame; no progressive playback. Users see a finished video still spinning, and seeking waits
        again.

        Remuxing is lossless (-c copy only moves the index): a 3 MB clip took 0.12 s and moov moved
        from byte 3147014 to 36, negligible next to the render.
        """
        import subprocess

        exe = util.ffmpeg_exe()
        if not exe:
            self.skipTest("没有 ffmpeg")
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "clip.mp4"
            # Build an mp4 with moov at the end (the default without +faststart)
            subprocess.run(
                [
                    exe,
                    "-nostdin",
                    "-loglevel",
                    "error",
                    "-y",
                    "-f",
                    "lavfi",
                    "-i",
                    "testsrc=size=64x64:rate=8:duration=1",
                    "-pix_fmt",
                    "yuv420p",
                    str(src),
                ],
                check=True,
            )
            raw = src.read_bytes()
            self.assertGreater(raw.find(b"moov"), len(raw) // 2, "样本本身就该是 moov 在尾部")
            video.base.faststart(src)
            out = src.read_bytes()
            self.assertLess(out.find(b"moov"), len(out) // 4, "moov 应该挪到了开头")
            self.assertGreater(len(out), 0)

    def test_a_broken_clip_is_left_alone_rather_than_destroyed(self):
        """If remuxing fails, keep the original: it is only a speed-up and must not break the beat."""
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.mp4"
            bad.write_bytes(b"not a video at all")
            video.base.faststart(bad)
            self.assertEqual(b"not a video at all", bad.read_bytes())
            self.assertFalse((Path(tmp) / "bad.fast.mp4").exists(), "临时文件要清掉")


if __name__ == "__main__":
    unittest.main(verbosity=2)
