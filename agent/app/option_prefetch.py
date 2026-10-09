"""Start single-player option planning before waiting on the browser."""

from __future__ import annotations

import copy
from murea_agent import config
from murea_agent import reqlog
from murea_agent.director import prefetch
from murea_agent.memory.world import World


def warm(world: World, *, source: str, expected_head: str | None = None, allow_busy: bool = False) -> int:
    if not config.SERVER_OPTION_PREFETCH:
        return 0
    node = world.head_node
    if node is None or node.continue_with or (expected_head is not None and node.id != expected_head):
        return 0
    from app.api import sse

    if not allow_busy and sse.busy(world.id):
        return 0
    options = node.options or []
    rec = node.recommend if isinstance(node.recommend, int) and 0 <= node.recommend < len(options) else 0
    ordered = ([options[rec]] + [t for i, t in enumerate(options) if i != rec]) if options else []
    token = reqlog.bind(world.id)
    accepted = 0
    try:
        for text in dict.fromkeys(t.strip() for t in ordered if isinstance(t, str) and t.strip()):
            try:
                # Match independent browser requests: each worker owns its snapshot.
                accepted += bool(prefetch.prefetch(copy.deepcopy(world), text, kind="option"))
            except Exception as exc:
                reqlog.record(
                    "option_prefetch_error",
                    source=source,
                    head=node.id,
                    user_text=text,
                    error=f"{type(exc).__name__}: {exc}",
                )
        reqlog.record("option_prefetch_batch", source=source, head=node.id, accepted=accepted)
    finally:
        reqlog.unbind(token)
    return accepted
