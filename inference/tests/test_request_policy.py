"""Request fields the service refuses; standard library only, no GPU, torch or FFmpeg needed.

The policy module lives in the vendored engine but imports only the standard library, so it
is loaded here straight from its file.
"""

import importlib.util
import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from murea_fast import client, service

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "vendor/sglang/python/sglang/multimodal_gen/runtime/entrypoints/openai/request_policy.py"
VIDEO_API = POLICY.with_name("video_api.py")
SAMPLE_CONFIG = ROOT / "vendor/sglang/python/sglang/multimodal_gen/configs/sample/ltx_2.py"

spec = importlib.util.spec_from_file_location("request_policy", POLICY)
policy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(policy)

PNG = "data:image/png;base64,iVBORw0KGgo="


class UnsupportedFieldTests(unittest.TestCase):
    def test_post_processing_fields_are_always_refused(self):
        values = {"enable_frame_interpolation": True, "frame_interpolation_exp": 2,
                  "frame_interpolation_scale": 0.5, "frame_interpolation_model_path": "org/repo",
                  "enable_upscaling": True, "upscaling_model_path": "org/repo",
                  "upscaling_scale": 2, "diffusers_kwargs": {"flow_shift": 3}}
        self.assertEqual(set(values), set(policy.UNSUPPORTED_FIELDS))
        for allowed in (False, True):
            for name, value in values.items():
                with self.subTest(name=name, allow_server_paths=allowed), \
                        self.assertRaisesRegex(policy.RequestRefused, name):
                    policy.check_video_request({name: value}, allow_server_paths=allowed)

    def test_unset_values_pass(self):
        # Multipart requests carry every form field, unset ones as None; false is harmless.
        fields = {"enable_frame_interpolation": False, "enable_upscaling": None,
                  "upscaling_model_path": None, "diffusers_kwargs": None, "output_path": ""}
        policy.check_video_request(fields, allow_server_paths=False)

    def test_refused_fields_never_reach_sampling_params(self):
        # _build_video_sampling_params must not forward them even if a check is bypassed.
        source = VIDEO_API.read_text()
        build = source[source.index("def _build_video_sampling_params"):]
        build = build[:build.index("\ndef ")]
        for name in policy.UNSUPPORTED_FIELDS:
            with self.subTest(name=name):
                self.assertNotIn(f'"{name}":', build)
        self.assertNotIn('batch.extra["diffusers_kwargs"]', source)

    def test_every_create_video_branch_is_checked(self):
        source = VIDEO_API.read_text()
        create = source[source.index("async def create_video"):]
        create = create[:create.index("\n@router.")]
        # One check after both the multipart and the JSON branch, before any output path
        # or sampling parameter is resolved.
        self.assertEqual(create.count("_refuse_request("), 1)
        self.assertLess(create.index("else:\n        try:\n            body = await request.json()"),
                        create.index("_refuse_request("))
        self.assertLess(create.index("_refuse_request("),
                        create.index("_build_video_sampling_params(request_id, req)"))

    def test_sampling_params_refuse_post_processing_on_every_endpoint(self):
        source = SAMPLE_CONFIG.read_text()
        for name in ("enable_frame_interpolation", "frame_interpolation_model_path",
                     "enable_upscaling", "upscaling_model_path", "perf_dump_path"):
            self.assertIn(f"self.{name}", source)


class ServerPathTests(unittest.TestCase):
    def test_server_paths_are_refused_by_default(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(policy.server_paths_allowed())
        cases = {"output_path": "/srv/outputs", "perf_dump_path": "/tmp/perf.json",
                 "video_path": "/srv/clip.mp4", "video_url": "relative/clip.mp4",
                 "input_reference": "~/person.png", "reference_url": "file:///etc/hosts.mp4",
                 "reference_image_paths": [PNG, "/srv/cup.png"]}
        for name, value in cases.items():
            with self.subTest(name=name), patch.dict(os.environ, {}, clear=True), \
                    self.assertRaisesRegex(policy.RequestRefused, f"{name}.*allow_server_paths"):
                policy.check_video_request({name: value})

    def test_urls_and_server_saved_uploads_are_accepted(self):
        saved = "/srv/runtime/inputs/uploads/req_prefix.mp4"
        policy.check_video_request(
            {"reference_image_paths": [PNG, "https://example.com/a.png", "HTTP://example.com/b.jpg"],
             "video_path": saved, "input_reference": "data:image/jpeg;base64,/9j/"},
            server_inputs={saved}, allow_server_paths=False)
        # Only the exact file the server saved; a client path next to it is still refused.
        with self.assertRaises(policy.RequestRefused):
            policy.check_video_request({"video_path": saved + ".other"}, server_inputs={saved},
                                       allow_server_paths=False)

    def test_the_setting_reaches_the_server_as_an_environment_variable(self):
        for value, expected in (("true", True), ("1", True), ("TRUE", True),
                                ("false", False), ("", False), ("no", False)):
            with self.subTest(value=value):
                self.assertIs(policy.server_paths_allowed({policy.SERVER_PATHS_ENV: value}), expected)
        with patch.dict(os.environ, {policy.SERVER_PATHS_ENV: "true"}):
            policy.check_video_request({"output_path": "/srv/out", "video_path": "/srv/clip.mp4",
                                        "reference_image_paths": ["/srv/a.png"]})

    def test_single_server_file_fields(self):
        with self.assertRaisesRegex(policy.RequestRefused, "image"):
            policy.check_server_file("image", "/srv/person.png", allow_server_paths=False)
        policy.check_server_file("image", PNG, allow_server_paths=False)
        policy.check_server_file("image", None, allow_server_paths=False)
        policy.check_server_file("image", "/srv/person.png", allow_server_paths=True)


class UploadNameTests(unittest.TestCase):
    def test_upload_names_stay_inside_the_upload_directory(self):
        cases = {"prefix.mp4": "prefix.mp4", "../../escape.mp4": "escape.mp4",
                 "a/b/c.png": "c.png", "..\\..\\c.png": "c.png", "/abs/path.jpg": "path.jpg",
                 ".hidden.mp4": "hidden.mp4", "name\x00.png": "name.png",
                 "..": "upload", "": "upload", None: "upload", "dir/": "upload"}
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                name = policy.upload_file_name(raw)
                self.assertEqual(expected, name)
                target = os.path.normpath(os.path.join("/srv/uploads", f"req_{name}"))
                self.assertEqual("/srv/uploads", os.path.dirname(target))
        self.assertEqual("url_image", policy.upload_file_name(None, "url_image"))
        self.assertEqual(200, len(policy.upload_file_name("x" * 500)))


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.tmp = Path(self.directory.name)
        self.base = json.loads((ROOT / "configs/single-gpu.json").read_text())
        self.base["runtime_dir"] = str(self.tmp / "runtime")

    def environment(self, **updates):
        path = self.tmp / "config.json"
        path.write_text(json.dumps({**self.base, **updates}))
        with patch.dict(os.environ, {"PATH": "/usr/bin", policy.SERVER_PATHS_ENV: "true"}):
            return service.environment(service.config(path))

    def test_allow_server_paths_defaults_to_off(self):
        for name in ("single-gpu.json", "4gpu-ulysses.json", "4gpu-ulysses-packed-qkv.json"):
            with self.subTest(config=name):
                shipped = json.loads((ROOT / "configs" / name).read_text())
                self.assertIs(shipped["allow_server_paths"], False)
        # Missing means off, and an inherited shell variable does not switch it on.
        del self.base["allow_server_paths"]
        self.assertEqual(self.environment()[policy.SERVER_PATHS_ENV], "false")
        self.assertEqual(self.environment(allow_server_paths=True)[policy.SERVER_PATHS_ENV], "true")
        with self.assertRaisesRegex(ValueError, "allow_server_paths must be true or false"):
            self.environment(allow_server_paths="yes")


class ClientHintTests(unittest.TestCase):
    def test_server_file_fields(self):
        self.assertEqual(client.server_file_fields(
            {"reference_image_paths": [PNG], "video_path": "https://example.com/a.mp4"}), [])
        self.assertEqual(client.server_file_fields(
            {"perf_dump_path": "/tmp/p.json", "video_path": "/srv/tail.mp4",
             "reference_image_paths": [PNG, "/srv/a.png"]}),
            ["perf_dump_path", "video_path", "reference_image_paths"])

    def test_refusal_names_the_setting(self):
        payload = {"prompt": "Go on.", "width": 1280, "height": 704, "num_frames": 137,
                   "video_path": "/srv/tail.mp4", "continuation_prefix_frames": 17}
        with TemporaryDirectory() as directory, \
                patch.object(client, "media_tool", side_effect=lambda name: name), \
                patch.object(client, "request_json",
                             side_effect=RuntimeError('HTTP 400: {"detail": "video_path ..."}')):
            with self.assertRaisesRegex(RuntimeError, 'video_path.*"allow_server_paths": true'):
                client.generate("http://127.0.0.1:1", payload, Path(directory) / "out")


if __name__ == "__main__":
    unittest.main()
