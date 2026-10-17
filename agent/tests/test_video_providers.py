"""The video provider registry, and the Murea clients against a fake Murea service."""

from __future__ import annotations

import base64
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx  # noqa: E402

from murea_agent import config  # noqa: E402
from murea_agent.providers import video  # noqa: E402
from murea_agent.providers.video import murea  # noqa: E402
from murea_agent.util import ffmpeg_exe  # noqa: E402


class ProviderRegistryTest(unittest.TestCase):
    def test_every_provider_implements_the_interface(self) -> None:
        for pid, provider in video.PROVIDERS.items():
            with self.subTest(provider=pid):
                self.assertEqual(pid, provider.id)
                self.assertTrue(provider.label and provider.note)
                self.assertGreater(provider.max_references, 0)
                self.assertIsInstance(provider.is_configured(), bool)
                self.assertIsInstance(provider.supports_continuation(), bool)

    def test_worlds_saved_with_a_retired_service_use_murea(self) -> None:
        self.assertIs(murea.PROVIDER, video.model("a-retired-service"))
        self.assertIs(murea.PROVIDER, video.model(""))

    def test_capabilities(self) -> None:
        self.assertTrue(video.model("murea").supports_continuation())
        self.assertFalse(video.model("murea").references_on_continuation)
        self.assertFalse(video.cancel("job-1", "murea"))

    def test_configured_once_the_address_is_set(self) -> None:
        with patch.object(config, "MUREA_URL", "http://gpu:30260"):
            self.assertEqual(["murea"], video.configured())
            self.assertTrue(murea.PROVIDER.is_configured())
        with patch.object(config, "MUREA_URL", ""):
            self.assertFalse(murea.PROVIDER.is_configured())

    def test_generate_dispatches_to_the_chosen_provider(self) -> None:
        result = video.Generation("v", Path("unused.mp4"), 4, 0, {})
        with patch.object(video.PROVIDERS["murea"], "generate", return_value=result) as gen:
            got = video.generate("prompt", [], model_id="murea", duration=6)
        gen.assert_called_once_with("prompt", [], duration=6)
        self.assertIn("local_queue_seconds", got.params)


def _clip(path: Path, frames: int, color: str = "red") -> Path:
    """A real 24 fps clip with audio, so the frame cutting is checked by ffmpeg itself."""
    subprocess.run(
        [ffmpeg_exe(), "-v", "error", "-y", "-f", "lavfi", "-i", f"color=c={color}:s=128x64:r=24",
         "-f", "lavfi", "-i", "sine=f=440:r=48000", "-frames:v", str(frames), "-shortest",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(path)],
        check=True,
    )  # fmt: skip
    return path


def _image(size: tuple[int, int], fmt: str = "PNG", *, portrait: bool = False) -> bytes:
    """A textured picture (a photo or a place), or a figure on a plain light studio backdrop."""
    from PIL import Image, ImageDraw

    if portrait:
        im = Image.new("RGB", size, (247, 247, 245))
        draw = ImageDraw.Draw(im)
        w, h = size
        draw.ellipse((w * 0.3, h * 0.9, w * 0.7, h * 0.98), fill=(225, 225, 222))  # floor shadow
        draw.rectangle((w * 0.42, h * 0.08, w * 0.58, h * 0.95), fill=(70, 60, 50))  # the figure
    else:
        im = Image.effect_noise(size, 60).convert("RGB")
    out = io.BytesIO()
    im.save(out, fmt)
    return out.getvalue()


def _size(data_uri: str) -> tuple[int, int]:
    from PIL import Image

    return Image.open(io.BytesIO(base64.b64decode(data_uri.split(",", 1)[1]))).size


class ReferenceFitTest(unittest.TestCase):
    """What Murea-Fast gets: every reference token costs time in every DiT step."""

    def test_a_large_photo_is_scaled_down_to_the_drawn_size(self) -> None:
        # A 12 MP phone photo made a 5 s clip take 29.9 s instead of 18.5 s on four H20s.
        out = murea._fit_reference(_image((4000, 3000), "JPEG"))
        w, h = _size("x," + base64.b64encode(out).decode())
        self.assertLessEqual(w * h, murea._REF_MAX_PIXELS)
        self.assertGreater(w * h, 0.97 * murea._REF_MAX_PIXELS)
        self.assertAlmostEqual(4 / 3, w / h, places=2)

    def test_a_place_or_photo_within_the_limit_goes_unchanged(self) -> None:
        raw = _image((1376, 768))
        self.assertEqual(raw, murea._fit_reference(raw))

    def test_a_portrait_on_a_plain_backdrop_is_cropped_to_the_figure(self) -> None:
        out = murea._fit_reference(_image((1328, 1328), portrait=True))
        w, h = _size("x," + base64.b64encode(out).decode())
        figure = 0.16 * 1328
        self.assertLess(w, 1.4 * figure)  # the backdrop beside the figure is gone
        self.assertGreater(w, figure)  # the figure and a margin stay, at full resolution
        self.assertGreater(h, 0.9 * 1328)

    def test_a_rotated_phone_photo_keeps_its_orientation(self) -> None:
        # EXIF orientation 6: stored landscape, shown portrait. Re-encoding drops the tag, so
        # the pixels must be turned first or the person arrives lying on their side.
        from PIL import Image

        stored = Image.effect_noise((1200, 800), 60).convert("RGB")
        exif = stored.getexif()
        exif[0x0112] = 6
        buf = io.BytesIO()
        stored.save(buf, "JPEG", quality=95, exif=exif)
        raw = buf.getvalue()
        self.assertGreater(len(raw), 120_000)  # large enough for for_upload to re-encode
        for name, fit in (("for_upload", video.base.for_upload), ("_fit_reference", murea._fit_reference)):
            with self.subTest(name=name):
                out = Image.open(io.BytesIO(fit(raw)))
                self.assertEqual((800, 1200), out.size)
                self.assertIn(out.getexif().get(0x0112, 1), (1,))
        # Both together, as Murea-Fast gets it, and the 4000x3000 case scaled down.
        stored = Image.effect_noise((4000, 3000), 60).convert("RGB")
        buf = io.BytesIO()
        stored.save(buf, "JPEG", quality=90, exif=exif)
        w, h = _size(murea._reference(buf.getvalue(), fit=True))
        self.assertAlmostEqual(3 / 4, w / h, places=2)

    def test_an_unreadable_image_goes_as_it_is(self) -> None:
        raw = b"\x89PNG\r\n\x1a\n" + b"0" * 64
        self.assertEqual(raw, murea._fit_reference(raw))


class FakeMurea:
    """Records submissions and answers like the Murea-Fast API."""

    def __init__(self, content: bytes) -> None:
        self.content = content
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"pipeline_class": "LTX2Pipeline"}]})
        self.requests.append(request)
        if request.method == "POST" and request.url.path == "/v1/videos":
            return httpx.Response(200, json={"id": "job-1", "status": "queued"})
        if request.url.path == "/v1/videos/job-1":
            return httpx.Response(200, json={"id": "job-1", "status": "completed", "inference_time_s": 9.5})
        if request.url.path == "/v1/videos/job-1/content":
            return httpx.Response(200, content=self.content)
        return httpx.Response(404)

    @property
    def submitted(self) -> httpx.Request:
        return next(r for r in self.requests if r.method == "POST")


@unittest.skipUnless(ffmpeg_exe(), "needs ffmpeg")
class MureaClientTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.clips = self.tmp / "clips"
        self.clips.mkdir()
        self.ref = self.tmp / "ref.png"
        self.ref.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 64)
        for p in (
            patch.object(config, "CLIPS", self.clips),
            # A fresh URL per test: the detected backend is cached per URL.
            patch.object(config, "MUREA_URL", f"http://gpu-box-{id(self)}:30260"),
        ):
            p.start()
            self.addCleanup(p.stop)

    def run_murea(self, fake: FakeMurea, **kw) -> video.Generation:
        def client() -> httpx.Client:
            return httpx.Client(base_url=config.MUREA_URL, transport=httpx.MockTransport(fake))

        with patch.object(murea, "_client", client):
            return murea.generate(out_dir=self.clips, poll_interval=0, **kw)

    def test_reference_generation_follows_the_model_contract(self) -> None:
        fake = FakeMurea(_clip(self.tmp / "out.mp4", 121).read_bytes())
        gen = self.run_murea(fake, prompt="<Subject 1> walks.", refs=[self.ref], duration=6)
        body = json.loads(fake.submitted.content)
        self.assertEqual(
            (1280, 704, 24, 8, 1.0),
            tuple(body[k] for k in ("width", "height", "fps", "num_inference_steps", "guidance_scale")),
        )
        self.assertEqual(145, body["num_frames"])  # 6 s, rounded to 8n+1
        self.assertEqual(0, (body["num_frames"] - 1) % 8)  # 8n+1
        self.assertTrue(body["reference_image_paths"][0].startswith("data:image/png;base64,"))
        self.assertNotIn("video_path", body)
        self.assertFalse(gen.continued)
        self.assertEqual(gen.video_id, gen.artifact_id)
        self.assertTrue(gen.video_path.exists())
        # Frame grabbing reads this; without it no clip had frames for the critic or the director.
        self.assertAlmostEqual(145 / 24, gen.params["duration_seconds"])

    def test_references_are_fitted_for_murea_fast(self) -> None:
        big = self.tmp / "photo.jpg"
        original = _image((4000, 3000), "JPEG")
        big.write_bytes(original)
        fake = FakeMurea(_clip(self.tmp / "out.mp4", 121).read_bytes())
        self.run_murea(fake, prompt="<Subject 1> walks.", refs=[big], duration=5)
        w, h = _size(json.loads(fake.submitted.content)["reference_image_paths"][0])
        self.assertLessEqual(w * h, murea._REF_MAX_PIXELS)
        self.assertEqual(original, big.read_bytes())  # only the copy sent is fitted

    def test_a_localhost_address_gets_no_local_paths(self) -> None:
        # 127.0.0.1 may be an SSH tunnel to another machine, which cannot read our files.
        _clip(self.clips / "mu-parent.mp4", 121)
        generation = FakeMurea(_clip(self.tmp / "out.mp4", 121).read_bytes())
        continuation = FakeMurea(_clip(self.tmp / "cont.mp4", 137).read_bytes())
        with patch.object(config, "MUREA_URL", "http://127.0.0.1:30260"):
            self.run_murea(generation, prompt="p", refs=[self.ref], duration=5)
            self.run_murea(continuation, prompt="p", continue_from="mu-parent", duration=5)
        ref = json.loads(generation.submitted.content)["reference_image_paths"][0]
        self.assertTrue(ref.startswith("data:image/png;base64,"))
        self.assertIn(b'name="video_reference"', continuation.submitted.content)
        self.assertNotIn(b'name="video_path"', continuation.submitted.content)

    def test_continuation_uploads_the_last_frames_and_drops_the_repeated_prefix(self) -> None:
        _clip(self.clips / "mu-parent.mp4", 121)
        # Murea returns the 17-frame prefix followed by the new frames.
        fake = FakeMurea(_clip(self.tmp / "out.mp4", 161, "blue").read_bytes())
        gen = self.run_murea(
            fake, prompt="He keeps walking.", refs=[self.ref], continue_from="mu-parent",
            continuation_fallback_prompt="<Subject 1> walks.", duration=6,
        )  # fmt: skip
        request = fake.submitted
        self.assertIn(b'name="video_reference"', request.content)
        self.assertIn(b'name="continuation_prefix_frames"\r\n\r\n17', request.content)
        self.assertIn(b'name="num_frames"\r\n\r\n161', request.content)
        # A multipart request carries the resolution only as `size`.
        self.assertIn(b'name="size"\r\n\r\n1280x704', request.content)
        self.assertNotIn(b'name="width"', request.content)
        self.assertNotIn(b"reference_image_paths", request.content)  # cannot be combined
        self.assertIn(b"He keeps walking.", request.content)
        self.assertTrue(gen.continued)
        self.assertEqual(144, murea._frame_count(gen.video_path))

    def test_clips_go_up_to_20_seconds_with_the_prefix_inside_the_limit(self) -> None:
        # Murea-Fast's positional range is 20 s (481 frames), a continuation's prefix included.
        self.assertEqual(241, murea._new_frames(10))
        self.assertEqual(481, murea._new_frames(20))
        self.assertEqual(481, murea._new_frames(30))  # the hard limit
        self.assertEqual(481, murea.PREFIX_FRAMES + murea._new_frames(20, murea.PREFIX_FRAMES) - 1)
        with patch.object(murea, "backend", return_value="fast"):
            self.assertEqual(20, murea.PROVIDER.max_seconds)
        with patch.object(murea, "backend", return_value="murea"):
            self.assertEqual(15, murea.PROVIDER.max_seconds)

    def test_a_missing_parent_clip_is_filmed_from_the_references(self) -> None:
        fake = FakeMurea(_clip(self.tmp / "out.mp4", 121).read_bytes())
        gen = self.run_murea(
            fake, prompt="He keeps walking.", refs=[self.ref], continue_from="mu-gone",
            continuation_fallback_prompt="<Subject 1> walks.", duration=5,
        )  # fmt: skip
        body = json.loads(fake.submitted.content)
        self.assertEqual("<Subject 1> walks.", body["prompt"])
        self.assertEqual(1, len(body["reference_image_paths"]))
        self.assertFalse(gen.continued)

    def test_a_failed_job_reports_the_server_message(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/v1/models":
                return httpx.Response(200, json={"data": [{"pipeline_class": "LTX2Pipeline"}]})
            if request.method == "POST":
                return httpx.Response(200, json={"id": "job-1", "status": "queued"})
            return httpx.Response(
                200, json={"id": "job-1", "status": "failed", "error": {"message": "CUDA OOM"}}
            )

        with self.assertRaisesRegex(video.VideoError, "CUDA OOM"):
            self.run_murea(handler, prompt="p", refs=[self.ref])


class FakeMureaServer:
    """Answers like a Murea SGLang server; `continuable` is whether its build returns latent ids."""

    def __init__(self, content: bytes, continuable: bool = True, parent_gone: bool = False) -> None:
        self.content, self.continuable, self.parent_gone = content, continuable, parent_gone
        self.bodies: list[dict] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/v1/models":
            return httpx.Response(200, json={"data": [{"pipeline_class": "MureaPipeline"}]})
        if request.method == "POST" and path == "/v1/videos":
            body = json.loads(request.content)
            self.bodies.append(body)
            if self.parent_gone and body.get("continuation_id"):
                return httpx.Response(404, json={"detail": "continuation artifact not found"})
            return httpx.Response(200, json={"id": f"job-{len(self.bodies)}", "status": "queued"})
        if path.startswith("/v1/videos/job-") and not path.endswith("/content"):
            job = {"id": path.rsplit("/", 1)[1], "status": "completed", "seconds": "6.0", "size": "1344x768"}
            if self.continuable:
                job["continuation_id"] = job["id"]
            return httpx.Response(200, json=job)
        if path.endswith("/content"):
            return httpx.Response(200, content=self.content)
        return httpx.Response(404)


class MureaModelTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.ref = self.tmp / "ref.png"
        self.ref.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 64)
        self.voice = self.tmp / "voice.wav"
        self.voice.write_bytes(b"RIFF0000WAVE")
        url = f"http://murea-{id(self)}:30010"  # a fresh URL: backend and continuation caches are per URL
        p = patch.object(config, "MUREA_URL", url)
        p.start()
        self.addCleanup(p.stop)

    def run_model(self, fake: FakeMureaServer, **kw) -> video.Generation:
        def client() -> httpx.Client:
            return httpx.Client(base_url=config.MUREA_URL, transport=httpx.MockTransport(fake))

        with patch.object(murea, "_client", client):
            return murea.generate(out_dir=self.tmp, poll_interval=0, **kw)

    def test_the_server_reports_its_pipeline_and_the_capabilities_follow(self) -> None:
        fake = FakeMureaServer(b"mp4")
        with patch.object(
            murea,
            "_client",
            lambda: httpx.Client(base_url=config.MUREA_URL, transport=httpx.MockTransport(fake)),
        ):
            self.assertEqual("murea", murea.backend())
            self.assertTrue(murea.PROVIDER.references_on_continuation)
            self.assertTrue(murea.PROVIDER.voice_references)
            self.assertTrue(murea.PROVIDER.supports_continuation())

    def test_an_error_from_the_model_list_is_not_cached_as_the_full_model(self) -> None:
        # A 401 (wrong MUREA_API_KEY) used to read as "no LTX2 pipeline", pinning the full-model
        # request format until a restart, even after the key was fixed.
        state = {"status": 401}

        def answer(request: httpx.Request) -> httpx.Response:
            if state["status"] != 200:
                return httpx.Response(state["status"], json={"detail": "unauthorized"})
            return httpx.Response(200, json={"data": [{"pipeline_class": "LTX2Pipeline"}]})

        client = lambda: httpx.Client(base_url=config.MUREA_URL, transport=httpx.MockTransport(answer))  # noqa: E731
        with patch.object(murea, "_client", client):
            self.assertEqual("fast", murea.backend())
            self.assertNotIn(config.MUREA_URL, murea._backends)
            state["status"] = 200
            murea._unreachable.pop(config.MUREA_URL, None)  # skip the 30 s retry wait
            self.assertEqual("fast", murea.backend())
            self.assertEqual("fast", murea._backends[config.MUREA_URL])

    def test_a_single_gpu_service_gets_the_validated_5_second_clips(self) -> None:
        cards = {"one": [{"pipeline_class": "LTX2Pipeline", "num_gpus": 1}],
                 "four": [{"pipeline_class": "LTX2Pipeline", "num_gpus": 4}],
                 "unknown": [{"pipeline_class": "LTX2Pipeline"}],
                 "full": [{"pipeline_class": "MureaPipeline", "num_gpus": 1}]}  # fmt: skip
        expected = {"one": 5, "four": 20, "unknown": 20, "full": 15}
        asked = []
        for name, card in cards.items():

            def answer(request: httpx.Request, card=card) -> httpx.Response:
                asked.append(request.url.path)
                return httpx.Response(200, json={"data": card})

            client = lambda answer=answer: httpx.Client(  # noqa: E731
                base_url=config.MUREA_URL, transport=httpx.MockTransport(answer)
            )
            with self.subTest(name=name), patch.object(config, "MUREA_URL", f"{config.MUREA_URL}/{name}"), \
                    patch.object(murea, "_client", client):  # fmt: skip
                self.assertEqual(expected[name], murea.PROVIDER.max_seconds)
                self.assertEqual(expected[name], murea.PROVIDER.max_seconds)  # cached
        self.assertEqual(4, len(asked))  # one /v1/models call per service

    def test_old_ffmpeg_gets_vsync_instead_of_fps_mode(self) -> None:
        # ffmpeg before 5.1 (Ubuntu 22.04 ships 4.4) has no -fps_mode; every continuation failed.
        calls = []

        def run(cmd, **kw):
            calls.append(cmd)
            if "-fps_mode" in cmd:
                return subprocess.CompletedProcess(cmd, 1, "", "Unrecognized option 'fps_mode'.")
            return subprocess.CompletedProcess(cmd, 0, "", "")

        with (
            patch.object(murea, "ffmpeg_exe", return_value="ffmpeg"),
            patch.object(murea.subprocess, "run", run),
        ):
            done = murea._ffmpeg("-i", "in.mp4", "-fps_mode", "passthrough", "out.mp4")
        self.assertEqual(0, done.returncode)
        self.assertEqual(["-vsync", "passthrough"], calls[-1][calls[-1].index("-vsync") :][:2])

    def test_reference_generation_request(self) -> None:
        fake = FakeMureaServer(b"mp4")
        gen = self.run_model(
            fake, prompt="<Subject 1> waves.", refs=[self.ref], ref_audios=[self.voice], duration=6
        )
        body = fake.bodies[0]
        self.assertEqual("ref2va", body["task"])
        self.assertEqual({"aspect_ratio": "16:9", "duration_seconds": 6}, body["target"])
        self.assertTrue(body["return_continuation"])
        self.assertNotIn("continuation_id", body)
        kinds = [(c["type"], c["role"], c["uri"].split(";")[0]) for c in body["conditions"]]
        self.assertEqual(
            [("image", "reference", "data:image/png"), ("audio", "reference", "data:audio/wav")], kinds
        )
        self.assertEqual("job-1", gen.artifact_id)
        self.assertFalse(gen.continued)

    def test_the_full_model_gets_references_at_their_own_size(self) -> None:
        big = self.tmp / "photo.jpg"
        big.write_bytes(_image((4000, 3000), "JPEG"))
        fake = FakeMureaServer(b"mp4")
        self.run_model(fake, prompt="<Subject 1> waves.", refs=[big], duration=6)
        self.assertEqual((4000, 3000), _size(fake.bodies[0]["conditions"][0]["uri"]))

    def test_continuation_sends_the_parent_latent_and_the_references(self) -> None:
        fake = FakeMureaServer(b"mp4")
        gen = self.run_model(
            fake, prompt="He keeps going.", refs=[self.ref], continue_from="job-0", duration=8
        )
        self.assertEqual("job-0", fake.bodies[0]["continuation_id"])
        self.assertEqual(1, len(fake.bodies[0]["conditions"]))
        self.assertTrue(gen.continued)

    def test_an_expired_parent_latent_is_filmed_from_the_references(self) -> None:
        fake = FakeMureaServer(b"mp4", parent_gone=True)
        gen = self.run_model(fake, prompt="p", refs=[self.ref], continue_from="job-0")
        self.assertEqual(2, len(fake.bodies))
        self.assertNotIn("continuation_id", fake.bodies[1])
        self.assertFalse(gen.continued)

    def test_a_required_continuation_does_not_fall_back_unless_allowed(self) -> None:
        with self.assertRaisesRegex(video.VideoError, "continuation"):
            self.run_model(FakeMureaServer(b"mp4", parent_gone=True), prompt="p", refs=[self.ref], continue_from="job-0",
                        require_continuation=True)  # fmt: skip

    def test_a_build_without_continuation_stops_offering_it(self) -> None:
        fake = FakeMureaServer(b"mp4", continuable=False)
        gen = self.run_model(fake, prompt="p", refs=[self.ref])
        self.assertIsNone(gen.artifact_id)
        with patch.object(
            murea,
            "_client",
            lambda: httpx.Client(base_url=config.MUREA_URL, transport=httpx.MockTransport(fake)),
        ):
            self.assertFalse(murea.PROVIDER.supports_continuation())


if __name__ == "__main__":
    unittest.main()
