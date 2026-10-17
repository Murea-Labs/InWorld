"""List (and optionally delete) reference images and clips no world refers to.

Orphans appear when a beat fails halfway: the reference image is already in data/refs but
the world is never saved (world.save is the last step), so nothing refers to it.
`delete_world` cannot remove it either; it only removes files a world refers to.

Worlds and co-op rooms (data/coop/*.json) both count as references. A beat that is still
running has not saved its world yet, so files younger than --min-age (default one hour)
are never treated as orphans.

    .venv/bin/python scripts/gc_refs.py          # list only
    .venv/bin/python scripts/gc_refs.py --delete # delete
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from murea_agent import config
from murea_agent.memory.world_media import _media, room_media


def _unreadable(f: Path) -> SystemExit:
    # An unreadable file counts as "refers to everything" rather than risk deleting; skip this run
    return SystemExit(f"!! {f.name} 读不出来，为安全起见不做任何清理")


def referenced() -> tuple[set[str], set[str]]:
    """File names of all (reference image, clip) files referred to by any world or co-op room."""
    media: set[str] = set()
    for f in config.SESSIONS.glob("*.json"):
        try:
            d = json.loads(f.read_text())
            media |= _media(d.get("subjects") or {}, d.get("places") or {}, d.get("nodes") or {})
        except Exception:
            raise _unreadable(f) from None
    for f in (config.DATA / "coop").glob("*.json"):
        try:
            media |= room_media(json.loads(f.read_text()))
        except Exception:
            raise _unreadable(f) from None
    refs = {Path(m).name for m in media if m.startswith("refs/")}
    clips = {Path(m).name for m in media if m.startswith("clips/")}
    return refs, clips


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--delete", action="store_true", help="真删；不加只列出来")
    ap.add_argument("--min-age", type=float, default=1.0, help="只清理至少这么多小时前的文件（默认 1）")
    a = ap.parse_args()
    cutoff = time.time() - a.min_age * 3600

    used_refs, used_clips = referenced()
    total = 0
    for label, folder, used in (("参考图", config.REFS, used_refs), ("片段", config.CLIPS, used_clips)):
        orphans = [
            p
            for p in sorted(folder.iterdir())
            if p.is_file()
            and not p.name.startswith(".")
            and p.name not in used
            and p.stat().st_mtime < cutoff
        ]
        mb = sum(p.stat().st_size for p in orphans) / 1048576
        total += len(orphans)
        print(f"{label}：{len(list(folder.iterdir()))} 个文件，没人引用的 {len(orphans)} 个（{mb:.1f} MB）")
        for p in orphans[:5]:
            print(f"    {p.name}")
        if len(orphans) > 5:
            print(f"    …还有 {len(orphans) - 5} 个")
        if a.delete:
            for p in orphans:
                p.unlink()
            print(f"    已删除 {len(orphans)} 个")
    if total and not a.delete:
        print("\n加 --delete 才会真删。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
