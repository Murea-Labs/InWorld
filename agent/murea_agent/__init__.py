"""Murea Agent: an open-world agent that turns each player action into a video beat.

    import murea_agent
    murea_agent.configure("config.yaml")          # optional; defaults to ./config.yaml
    agent = murea_agent.MureaAgent()

Package layout:

    agent.py      MureaAgent, the public API
    loop.py       the turn loop: perceive -> plan -> check -> act -> commit
    events.py     steps, events and the per-clip trace
    director/     the planning model: prompts, output schema, prefetch
    guards/       deterministic checks of the plan, with re-planning
    shot/         reference images, voices and the video prompt of one beat
    critic.py     reflect: compare the rendered clip with the plan
    memory/       the world: assets, story tree, quests, voices
    tools/        tool registry; memory tools for the director
    providers/    LLM, image, video and TTS clients
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

__all__ = ["MureaAgent", "World", "Node", "configure"]


def configure(path: str | os.PathLike[str]) -> None:
    """Use this config file. Must be called before the agent is first used."""
    resolved = str(Path(path).expanduser().resolve())
    if "murea_agent.config" in sys.modules and os.environ.get("MUREA_AGENT_CONFIG") != resolved:
        raise RuntimeError("murea_agent is already configured; call murea_agent.configure() first")
    os.environ["MUREA_AGENT_CONFIG"] = resolved


def __getattr__(name: str) -> Any:
    # Lazy, so `murea_agent.configure()` can run before config.yaml is read.
    if name == "MureaAgent":
        from murea_agent.agent import MureaAgent

        return MureaAgent
    if name in {"World", "Node"}:
        from murea_agent.memory import world

        return getattr(world, name)
    raise AttributeError(name)
