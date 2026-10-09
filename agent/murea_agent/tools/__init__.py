"""Tools the agent works with.

The turn loop calls act tools itself (fixed control flow). Director tools are
offered to the planning model, which may call them before it answers; they are
plain functions of the world plus JSON arguments, declared with a JSON schema:

    @tool("recall_story", "Search earlier beats ...", {"query": {"type": "string"}},
          for_director=True)
    def recall_story(world, query): ...

Every call runs as a traced step of the current turn (see `murea_agent.events`).
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any
from collections.abc import Callable

from murea_agent import events


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    fn: Callable[..., Any]
    parameters: dict[str, Any] = field(default_factory=dict)  # JSON schema properties
    required: tuple[str, ...] = ()
    phase: events.Phase = "act"
    for_director: bool = False

    def declaration(self) -> dict[str, Any]:
        """The function declaration a model sees (OpenAI / Gemini shape)."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {"type": "object", "properties": self.parameters, "required": list(self.required)},
        }


_TOOLS: dict[str, Tool] = {}


def tool(
    name: str,
    description: str,
    parameters: dict[str, Any] | None = None,
    *,
    required: tuple[str, ...] = (),
    phase: events.Phase = "act",
    for_director: bool = False,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def register(fn: Callable[..., Any]) -> Callable[..., Any]:
        _TOOLS[name] = Tool(
            name, description, fn, parameters or {}, required, "plan" if for_director else phase, for_director
        )
        return fn

    return register


def get(name: str) -> Tool:
    _load()
    return _TOOLS[name]


def all_tools() -> list[Tool]:
    _load()
    return list(_TOOLS.values())


def director_tools() -> list[Tool]:
    return [t for t in all_tools() if t.for_director]


def call(name: str, world: Any, args: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run a director tool for the model. Errors become a result, never an exception."""
    args = dict(args or {})
    emitter = events.current()
    try:
        spec = get(name)
    except KeyError:
        return {"error": f"unknown tool {name!r}"}
    summary = f"{name}({', '.join(f'{k}={v!r}' for k, v in args.items())})"[:120]
    try:
        if emitter is None:
            return {"result": spec.fn(world, **args)}
        with emitter.step(spec.phase, name, summary) as step:
            result = spec.fn(world, **args)
            step.data.update(args=args, result=result)
            return {"result": result}
    except Exception as exc:  # noqa: BLE001 -- a bad call is reported back to the model
        return {"error": f"{type(exc).__name__}: {exc}"}


def _load() -> None:
    # Importing registers the built-in tools.
    importlib.import_module("murea_agent.tools.recall")
