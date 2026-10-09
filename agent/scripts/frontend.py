"""The prebuilt front end in frontend/dist and the sources it was built from.

The repo ships frontend/dist so running Murea Agent needs no Node.js. A fingerprint of
the front-end sources is stored next to the build:

    python scripts/frontend.py check   # exit 1 when dist is older than the sources
    python scripts/frontend.py build   # npm ci (build deps only), vite build, fingerprint

Rebuild and commit dist together with any front-end change; CI runs `check`.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

FRONTEND = Path(__file__).resolve().parent.parent / "frontend"
SOURCES = ("src", "index.html", "package-lock.json", "vite.config.ts")
STAMP = FRONTEND / "dist" / ".source-hash"


def fingerprint() -> str:
    h = hashlib.sha256()
    for name in SOURCES:
        root = FRONTEND / name
        files = sorted(p for p in root.rglob("*") if p.is_file()) if root.is_dir() else [root]
        for path in files:
            h.update(path.relative_to(FRONTEND).as_posix().encode() + b"\0")
            h.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()


def fresh() -> bool:
    return (
        (FRONTEND / "dist" / "index.html").exists()
        and STAMP.exists()
        and STAMP.read_text().strip() == fingerprint()
    )


def build() -> None:
    npm = shutil.which("npm")
    if not npm:
        sys.exit("npm is needed to rebuild the front end (Node.js 20.19+ or 22.12+)")
    subprocess.run([npm, "ci", "--omit=dev", "--silent", "--no-fund", "--no-audit"], cwd=FRONTEND, check=True)
    subprocess.run([npm, "exec", "--", "vite", "build", "--logLevel", "warn"], cwd=FRONTEND, check=True)
    STAMP.write_text(fingerprint() + "\n")


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "check"
    if command == "build":
        build()
    elif command == "check":
        if not fresh():
            sys.exit("frontend/dist is out of date: run `python scripts/frontend.py build` and commit it")
    else:
        print(fingerprint())
