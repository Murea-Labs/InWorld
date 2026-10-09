"""Run every offline test script: `python tests/run_all.py [name ...]`.

Each tests/test_*.py is a standalone script (exit code 0 = pass). They run
against a throw-away config and data directory with no credentials, so nothing
here calls a real model.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent

CONFIG = """\
server: {{data_dir: "{data}", demo_world: false}}
gemini: {{api_key: ""}}
llm: {{provider: gemini}}
agent: {{language: zh}}
video:
  provider: murea
  murea: {{url: "http://127.0.0.1:9"}}
"""


def run(path: Path, env: dict[str, str]) -> tuple[str, int, str]:
    source = path.read_text(encoding="utf-8")
    if "unittest.TestCase" in source and "unittest.main(" not in source:
        # Run as a script, such a file defines its tests and exits 0 without running them.
        return path.stem, -1, "defines unittest tests but has no `unittest.main()` entry point"
    try:
        p = subprocess.run(
            [sys.executable, str(path)], cwd=ROOT, env=env, capture_output=True, text=True, timeout=180
        )
        return path.stem, p.returncode, (p.stdout + p.stderr)[-2000:]
    except subprocess.TimeoutExpired:
        return path.stem, -1, "timed out"


def main() -> int:
    names = sys.argv[1:]
    files = sorted(
        f
        for f in HERE.glob("test_*.py")
        if not names or f.stem in names or f.stem.removeprefix("test_") in names
    )
    with tempfile.TemporaryDirectory() as tmp:
        envs = {}
        for mode in ("single", "coop"):
            cfg = Path(tmp) / f"config-{mode}.yaml"
            cfg.write_text(CONFIG.format(data=Path(tmp) / f"data-{mode}") + f"mode: {mode}\n")
            # No real credentials: keys are read from the environment too.
            clean = {k: v for k, v in os.environ.items() if not k.endswith("_API_KEY") and k != "MUREA_URL"}
            envs[mode] = {**clean, "MUREA_AGENT_CONFIG": str(cfg), "PYTHONPATH": str(ROOT)}

        def run_one(f: Path):
            # Co-op tests run against a server in co-op mode.
            return run(f, envs["coop" if f.stem.startswith("test_coop") else "single"])

        with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 2)) as ex:
            results = list(ex.map(run_one, files))
    failed = [r for r in results if r[1] != 0]
    for name, code, out in failed:
        print(f"--- FAIL {name} (exit {code})\n{out}\n")
    print(f"{len(results) - len(failed)}/{len(results)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
