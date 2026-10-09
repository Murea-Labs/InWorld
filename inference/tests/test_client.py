import json
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from murea_fast import client
from murea_fast.client import validate_request
from murea_fast.service import config, command


class MediaToolTests(unittest.TestCase):
    def test_ffmpeg_and_ffprobe_never_read_stdin(self):
        # ffmpeg reads keyboard commands from stdin; in `while read name; do murea-fast generate
        # ...; done` it swallowed the loop's input and every other request was skipped.
        with TemporaryDirectory() as directory, \
                patch.object(client, "media_tool", return_value="ffmpeg"), \
                patch.object(client.subprocess, "check_output", return_value="{}") as check_output, \
                patch.object(client.subprocess, "run") as run:
            client.probe(Path(directory) / "clip.mp4")
            info = {"width": 1280, "height": 704, "source_frames": 121, "prefix_frames": 17}
            client.prepare_tail(Path(directory) / "clip.mp4", Path(directory) / "tail.mp4", 17, info)
        for call in [*check_output.call_args_list, *run.call_args_list]:
            self.assertIs(subprocess.DEVNULL, call.kwargs.get("stdin"))


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.request = {"prompt": "A person walks.", "width": 768, "height": 416,
                        "num_frames": 121, "reference_image_paths": ["a.png", "b.png"]}

    def test_ordered_refs(self):
        validate_request(self.request)
        self.assertEqual(self.request["reference_image_paths"], ["a.png", "b.png"])

    def test_invalid_sampling(self):
        for updates in ({"num_frames": 120}, {"num_frames": 489}, {"num_inference_steps": 4},
                        {"guidance_scale": 3}, {"fps": 30}, {"width": 770}):
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                validate_request({**self.request, **updates})

    def test_clips_up_to_20_seconds(self):
        for frames in (241, 361, 481):
            validate_request({**self.request, "num_frames": frames})

    def test_continuation(self):
        request = {**self.request, "reference_image_paths": None, "video_path": "tail.mp4",
                   "num_frames": 137, "continuation_prefix_frames": 17}
        validate_request(request)
        for updates in ({"continuation_prefix_frames": 16}, {"continuation_prefix_frames": 137},
                        {"continuation_prefix_frames": None}, {"reference_image_paths": ["a.png"]}):
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                validate_request({**request, **updates})

    def test_launcher_rejects_duplicate_gpus_and_ports(self):
        original = json.loads((Path(__file__).resolve().parents[1] / "configs/4gpu-ulysses.json").read_text())
        with TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            for update in ({"gpus": "0,0"}, {"master_port": original["port"]}):
                path.write_text(json.dumps({**original, **update}))
                with self.assertRaises(ValueError):
                    config(path)
            path.write_text(json.dumps(original))
            cmd = command(config(path))
            self.assertIn("--ulysses-degree", cmd)
            self.assertIn("--strict-ports", cmd)
            self.assertEqual(cmd[cmd.index("--attention-backend") + 1], "torch_sdpa")
            self.assertEqual(cmd[cmd.index("--served-model-name") + 1], "Murea-Fast")


if __name__ == "__main__":
    unittest.main()
