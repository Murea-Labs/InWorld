"""Use the agent from Python: one beat, with every agent step printed.

    .venv/bin/python examples/play.py

For an interactive session in the terminal, run `python -m murea_agent` instead.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from murea_agent import MureaAgent  # noqa: E402

agent = MureaAgent()
world = agent.create_world(
    title="Rainy Harbour",
    setting="A small harbour town on a stormy night; lanterns, wet cobblestones, an old inn.",
    visual_style="cinematic live action, rainy night, warm lantern light",
)

for event in agent.stream(world, "I step into the inn, shake off the rain and look around."):
    if event["type"] == "step" and event["status"] != "running":
        print(f"{event['phase']:>8} · {event['name']:<16} {event['seconds']:.1f}s")
    elif event["type"] == "say":
        print(f"\n  {event['text']}\n")
    elif event["type"] == "done":
        node = event["node"]
        print(f"\nclip: {node.clip}\nnext: {node.options}")
    elif event["type"] == "error":
        print(event["message"])

# After playback: let the critic compare the clip with the plan.
print(agent.reflect(world))
