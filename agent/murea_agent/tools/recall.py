"""Memory tools for the director.

The director sees only the recent part of the story (`agent.history_window`).
When older beats exist it may look them up instead of having the whole story
in every prompt.
"""

from __future__ import annotations

import re
from typing import Any

from murea_agent.tools import tool

_WORD = re.compile(r"[a-z0-9]+|[一-鿿]", re.I)


def _terms(text: str) -> set[str]:
    """Words for Latin text, characters and character pairs for Chinese."""
    chars = _WORD.findall(str(text or "").lower())
    out = {c for c in chars if len(c) > 1 or "一" <= c <= "鿿"}
    out |= {a + b for a, b in zip(chars, chars[1:]) if "一" <= a <= "鿿" and "一" <= b <= "鿿"}
    return out


@tool(
    "recall_story",
    "Search earlier beats of this story (older than the recent history you were given) "
    "for events, people, places or items. Returns the best matching beats, oldest first.",
    {
        "query": {"type": "string", "description": "What to look for, e.g. 'the brass key'"},
        "limit": {"type": "integer", "description": "At most this many beats (1-8)"},
    },
    required=("query",),
    for_director=True,
)
def recall_story(world: Any, query: str, limit: int = 5) -> list[dict[str, Any]]:
    wanted = _terms(query)
    if not wanted:
        return []
    beats = world.lineage()
    scored = []
    for index, node in enumerate(beats, 1):
        text = " ".join(str(x or "") for x in (node.user_text, node.reply, node.place, node.carry))
        score = len(wanted & _terms(text))
        if score:
            scored.append((score, index, node))
    best = sorted(scored, key=lambda s: (-s[0], -s[1]))[: max(1, min(8, int(limit or 5)))]
    return [
        {
            "segment": index,
            "place": node.place,
            "player": node.user_text,
            "what_happened": node.reply,
            "carried": node.carry,
        }
        for _score, index, node in sorted(best, key=lambda s: s[1])
    ]


@tool(
    "inspect_asset",
    "Look up one character, creature, object or place of this world by name: its "
    "description, locked look, current state and the segments it appeared in.",
    {"name": {"type": "string", "description": "The asset's name as used in the story"}},
    required=("name",),
    for_director=True,
)
def inspect_asset(world: Any, name: str) -> dict[str, Any]:
    key = " ".join(str(name or "").split())
    for sid, subject in world.subjects.items():
        if key and (key == subject.get("name") or key == sid or key in str(subject.get("name"))):
            seen = [i for i, node in enumerate(world.lineage(), 1) if sid in (node.subjects or [])]
            return {
                "id": sid,
                "name": subject.get("name"),
                "kind": subject.get("kind"),
                "description": subject.get("desc", ""),
                "look": subject.get("definition", ""),
                "state": subject.get("state") or {},
                "segments": seen[-12:],
            }
    for place_name, place in world.places.items():
        if key and (key == place_name or key in place_name):
            seen = [i for i, node in enumerate(world.lineage(), 1) if node.place == place_name]
            return {
                "name": place_name,
                "kind": "place",
                "description": place.get("desc", ""),
                "main_scene": place.get("main_scene", ""),
                "connected_to": world.adjacency().get(place_name, []),
                "state": place.get("state") or {},
                "segments": seen[-12:],
            }
    return {
        "error": f"no asset named {name!r}",
        "known": [s.get("name") for s in world.subjects.values()] + list(world.places),
    }
