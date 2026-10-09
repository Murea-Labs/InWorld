"""Launcher and CLI checks; standard library only, no GPU, torch or FFmpeg needed."""

import contextlib
import io
import json
import os
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from murea_fast import cli, client, service

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def run_cli(*argv):
    """Run the CLI in-process; returns (stdout, stderr)."""
    out, err = io.StringIO(), io.StringIO()
    with patch.object(sys, "argv", ["murea-fast", *argv]), \
            contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        cli.main()
    return out.getvalue(), err.getvalue()


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.tmp = Path(self.directory.name)
        self.base = json.loads((CONFIGS / "single-gpu.json").read_text())
        self.base["runtime_dir"] = str(self.tmp / "runtime")

    def write(self, data):
        path = self.tmp / "config.json"
        path.write_text(json.dumps(data))
        return path

    def test_host_defaults_to_loopback(self):
        data = dict(self.base)
        del data["host"]
        c = service.config(self.write(data))
        self.assertEqual(c["host"], "127.0.0.1")
        cmd = service.command(c)
        self.assertEqual(cmd[cmd.index("--host") + 1], "127.0.0.1")

    def test_gpus_must_be_a_string(self):
        for gpus in (0, [0, 1], None):
            with self.subTest(gpus=gpus), self.assertRaisesRegex(ValueError, "gpus must be a string"):
                service.config(self.write({**self.base, "gpus": gpus}))

    def test_missing_required_key_is_named(self):
        data = dict(self.base)
        del data["parallel"]
        with self.assertRaisesRegex(ValueError, "Missing configuration keys: parallel"):
            service.config(self.write(data))

    def test_flags_must_be_booleans(self):
        with self.assertRaisesRegex(ValueError, "compile must be true or false"):
            service.config(self.write({**self.base, "compile": "yes"}))

    def test_requires_source_checkout(self):
        # A non-editable install puts the package in site-packages, without vendor/ or scripts/.
        with patch.object(service, "ROOT", self.tmp / "site-packages"), \
                self.assertRaisesRegex(RuntimeError, "pip install -e"):
            service.config(self.write(self.base))

    def test_environment_numbers_gpus_like_nvidia_smi(self):
        c = service.config(self.write({**self.base, "gpus": "2,3"}))
        with patch.dict(os.environ, {"PATH": "/usr/bin", "CUDA_DEVICE_ORDER": "FASTEST_FIRST"}):
            env = service.environment(c)
        self.assertEqual(env["CUDA_DEVICE_ORDER"], "PCI_BUS_ID")
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "2,3")

    def test_health_url(self):
        for host, url in (("127.0.0.1", "http://127.0.0.1:30260/health"),
                          ("192.0.2.10", "http://192.0.2.10:30260/health"),
                          ("service.example", "http://service.example:30260/health"),
                          ("0.0.0.0", "http://127.0.0.1:30260/health"),
                          ("", "http://127.0.0.1:30260/health"),
                          ("::", "http://[::1]:30260/health"),
                          ("::1", "http://[::1]:30260/health"),
                          ("2001:db8::10", "http://[2001:db8::10]:30260/health")):
            with self.subTest(host=host):
                self.assertEqual(service.health_url({**self.base, "host": host}), url)

    def test_status_reports_configured_host(self):
        path = self.write({**self.base, "host": "192.0.2.10"})
        with patch.object(service, "state", return_value={"running": False}):
            out, _ = run_cli("status", "--config", str(path))
        self.assertEqual(json.loads(out)["health_url"], "http://192.0.2.10:30260/health")


class StopTests(unittest.TestCase):
    def test_waits_for_every_process_of_the_group(self):
        # The leader exits first; two workers of its group take a few more seconds.
        answers = iter([None, None, None, ProcessLookupError()])

        def killpg(pgid, sig):
            self.assertEqual((pgid, sig), (4242, 0))
            answer = next(answers)
            if answer is not None:
                raise answer

        with patch.object(service.os, "killpg", side_effect=killpg), \
                patch.object(service.time, "sleep") as sleep:
            self.assertTrue(service.wait_for_group(4242))
        self.assertEqual(sleep.call_count, 3)

    def test_reports_a_group_that_does_not_exit(self):
        clock = iter(range(100))
        with patch.object(service.os, "killpg", return_value=None), \
                patch.object(service.time, "monotonic", side_effect=lambda: next(clock)), \
                patch.object(service.time, "sleep"):
            self.assertFalse(service.wait_for_group(4242, timeout=5))

    def test_a_member_owned_by_another_user_still_counts(self):
        with patch.object(service.os, "killpg", side_effect=PermissionError()):
            self.assertTrue(service.group_alive(4242))


def parent_probe(width=1280, height=704, frames=121):
    return {"streams": [{"codec_type": "video", "width": width, "height": height,
                         "avg_frame_rate": "24/1", "nb_read_frames": str(frames)}]}


class ContinueTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.tmp = Path(self.directory.name)
        self.output = self.tmp / "continuation"
        # Nothing may run FFmpeg or reach a service in these tests.
        self.subprocess_run = self.mock("murea_fast.client.subprocess.run")
        self.subprocess_check_output = self.mock("murea_fast.client.subprocess.check_output")
        self.generate = self.mock("murea_fast.client.generate", return_value={"ok": True})
        self.mock("murea_fast.client.media_tool", side_effect=lambda name: name)

    def mock(self, target, **kwargs):
        patcher = patch(target, **kwargs)
        self.addCleanup(patcher.stop)
        return patcher.start()

    def run_continue(self, *extra):
        return run_cli("continue", "--parent", str(self.tmp / "parent.mp4"), "--prompt", "Go on.",
                       "--output", str(self.output), *extra)

    def assert_untouched(self):
        self.assertFalse(self.output.exists())
        self.subprocess_run.assert_not_called()
        self.generate.assert_not_called()

    def test_frame_limits_checked_before_anything_runs(self):
        # 17 + 472 = 489 frames exceeds the 481-frame limit; 16 is not 8n+1.
        for extra in (("--new-frames", "472"), ("--prefix-frames", "16"),
                      ("--new-frames", "12")):
            with self.subTest(extra=extra):
                with self.assertRaises(SystemExit) as raised:
                    self.run_continue(*extra)
                self.assertEqual(raised.exception.code, 2)
                self.subprocess_check_output.assert_not_called()  # parent not even probed
                self.assert_untouched()

    def test_longest_continuation_is_accepted(self):
        with patch.object(client, "probe", return_value=parent_probe()):
            self.run_continue("--new-frames", "464")  # 17 + 464 = 481
        self.assertEqual(self.generate.call_args.args[1]["num_frames"], 481)

    def test_parent_checked_before_output_is_created(self):
        cases = ((parent_probe(height=1080), ValueError, "height must be a positive multiple of 32"),
                 (parent_probe(frames=9), ValueError, "shorter than requested prefix"))
        for probe, error, message in cases:
            with self.subTest(message=message), patch.object(client, "probe", return_value=probe):
                with self.assertRaisesRegex(error, message):
                    self.run_continue()
                self.assert_untouched()

    def test_valid_continuation(self):
        def generate(base_url, payload, output, timeout):
            self.assertTrue((self.output / "parent.json").is_file())
            return {"ok": True}

        self.generate.side_effect = generate
        with patch.object(client, "probe", return_value=parent_probe()):
            out, _ = self.run_continue()
        self.assertEqual(json.loads(out), {"ok": True})
        trim = self.subprocess_run.call_args.args[0]
        self.assertIn("trim=start_frame=104,setpts=PTS-STARTPTS", trim)
        self.assertEqual(trim[-1], str(self.output.resolve() / "prefix.mp4"))
        payload = self.generate.call_args.args[1]
        self.assertEqual((payload["width"], payload["height"], payload["num_frames"],
                          payload["continuation_prefix_frames"]), (1280, 704, 137, 17))


if __name__ == "__main__":
    unittest.main()
