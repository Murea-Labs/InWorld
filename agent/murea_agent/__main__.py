"""Play a world in the terminal and watch the agent work.

    python -m murea_agent "I step into the inn and look around."
    python -m murea_agent --world w1a2b3c4          # continue a saved world
    python -m murea_agent --config my.yaml --setting "A desert outpost at dawn."

Every agent step is printed as it happens; the clip path, the narration and
the three suggested actions follow. Type an action, a number, or press enter
for the recommended one.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

import murea_agent

PHASE_MARK = {"perceive": "◉", "plan": "✎", "check": "✓", "act": "▶", "commit": "■", "reflect": "↺"}


def _show(event: dict[str, Any]) -> None:
    kind = event.get("type")
    if kind == "step" and event.get("status") != "running":
        data = event.get("data") or {}
        brief = ", ".join(
            f"{k}={v}" for k, v in data.items() if v not in ("", None, [], {}) and k != "error"
        )[:110]
        status = "" if event["status"] == "done" else f" [{event['status']}]"
        seconds = event.get("seconds")
        took = f"{seconds:5.1f}s" if isinstance(seconds, (int, float)) else "      "
        print(
            f"  {PHASE_MARK.get(event['phase'], '·')} {event['phase']:<8} "
            f"{event['name']:<16} {took}{status}  {brief}"
        )
        if data.get("error"):
            print(f"      error: {data['error']}")
    elif kind == "say":
        print(f"\n  “{event['text']}”\n")
    elif kind == "sequence":
        print(f"  (split into {event['total']} clips)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m murea_agent", description=__doc__.split("\n")[0])
    parser.add_argument("action", nargs="?", default="I look around.")
    parser.add_argument("--config", help="config file (default: ./config.yaml)")
    parser.add_argument("--world", help="id of a saved world to continue")
    parser.add_argument("--title", default="Rainy Harbour")
    parser.add_argument(
        "--setting", default="A small harbour town on a stormy night: lanterns, wet cobblestones, an old inn."
    )
    parser.add_argument(
        "--style", default="cinematic live action, rainy night, warm lantern light, shallow depth of field"
    )
    args = parser.parse_args(argv)
    if args.config:
        murea_agent.configure(args.config)

    from murea_agent import config

    agent = murea_agent.MureaAgent()
    world = (
        agent.load_world(args.world)
        if args.world
        else agent.create_world(title=args.title, setting=args.setting, visual_style=args.style)
    )
    print(
        f"world {world.id} ({world.title}) · video: {config.VIDEO_MODEL} · "
        f"tools: {', '.join(t.name for t in agent.tools)}"
    )

    action, auto = args.action, False
    while True:
        node = None
        for event in agent.stream(world, action, auto=auto):
            if event["type"] == "done":
                node = event["node"]
            elif event["type"] in {"error", "cancelled"}:
                print(f"  {event['type']}: {event.get('message', '')}", file=sys.stderr)
            else:
                _show(event)
        if node is None:
            return 1
        print(f"\n  clip: {config.CLIPS / node.clip}")
        verdict = agent.reflect(world, node)
        if verdict:
            print(f"  ↺ reflect  critic saw: {verdict.get('observed', '')}")
            node = world.head_node
        if node.continue_with:
            print(f"  … continuing: {node.continue_with}")
            action, auto = node.continue_with, True
            continue
        auto = False
        for i, option in enumerate(node.options, 1):
            print(f"  {'*' if i - 1 == node.recommend else ' '}{i}. {option}")
        try:
            typed = input("\n> ").strip()
        except EOFError:
            return 0
        if typed in {"q", "quit", "exit"}:
            return 0
        if typed.isdigit() and 1 <= int(typed) <= len(node.options):
            typed = node.options[int(typed) - 1]
        action = typed or (node.options[node.recommend] if node.options else "Continue.")


if __name__ == "__main__":
    sys.exit(main())
