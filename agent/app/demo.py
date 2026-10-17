"""The demo world: a few finished clips, when the repository ships them in examples/demo/.

A fresh checkout without API keys would otherwise open on an empty stage. The
first time a user has no worlds, the demo archive is imported for them, so the
player, the story drawers and the agent trace can be explored right away. It is
an ordinary world afterwards: it can be deleted, and is never imported again for
that user (a marker file remembers it).
"""

from __future__ import annotations

import logging
from pathlib import Path

from murea_agent import config

DEMO = Path(__file__).resolve().parent.parent / "examples" / "demo" / "rainy-harbour.zip"
log = logging.getLogger(__name__)


def _marker(user_id: str) -> Path:
    return config.DATA / "demo-seeded" / user_id


def seed_demo(user_id: str, has_worlds: bool) -> bool:
    """Import the demo world for a user who has none yet. Returns True if imported."""
    marker = _marker(user_id)
    if has_worlds or not DEMO.exists() or marker.exists():
        return False
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch()
    try:
        from app.api.routes.transfer import import_archive

        import_archive(DEMO.read_bytes(), user_id)
        return True
    except Exception:  # noqa: BLE001 -- the demo is optional
        log.exception("could not import the demo world")
        return False
