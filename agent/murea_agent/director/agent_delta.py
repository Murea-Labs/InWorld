"""Sparse world changes, expanded before any existing Agent consumers run."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from murea_agent.memory.world import World


FIELDS = (
    "mainline_phase",
    "mainline_milestones",
    "quest_updates",
    "mode",
    "transition",
    "place",
    "place_image_prompt",
    "place_brief",
    "main_scene",
    "carry",
    "state_updates",
)

RULES = """

# World changes: output only the differences from the snapshot

The world snapshot is already stored by the service. The fields mainline_phase,
mainline_milestones, quest_updates, mode, transition, place, place_image_prompt,
place_brief, main_scene, carry and state_updates belong inside the required
world_delta object in this response.
All of their semantic requirements above still apply. Review each of them, but
write only values that actually change this turn. world_delta={} explicitly means
that none of these world facts or asset declarations change. Never copy unchanged
milestones, carried objects or empty state-update records into world_delta.

Omitted location/mode/transition mean stay in the snapshot's current place with
mode=continue and transition=false. A camera cut in the same place MUST still set
mode=cut. A place change MUST specify place, mode and transition and provide the
full creation fields for a new place. New subjects, newly visible implicit player
bodies and future places MUST be fully declared with the same identity, voice,
reference image and stage constraints as before. The root fields new_subjects,
implicit_player and new_places remain REQUIRED every turn, outside world_delta.
Do not refer to a new character with an invented S-number while leaving its asset
declaration empty: every visible new body needs its own stable reference image.

Omitted carry means preserve the previous carried-items sentence. To put down or
lose all carried items, explicitly write carry="". Otherwise write the complete
new carried-items sentence. State and quest updates remain incremental changes,
never full copies of old states. New knowledge, injury, changed weather or actual
quest progress MUST be recorded even if the shot stays in the same room.

The service expands completed world_delta into the full original contract before
image preparation, world-state compilation, motion validation and video submission.
Keep action_sequence, motion_plan, beat, dialogue, subjects, all video sections and
motion audit fields complete as before. The video description MUST describe this
new shot even when the world has no persistent changes; it cannot be empty or LOCKED.

"""


def defaults(world: World) -> dict[str, Any]:
    head = world.head_node
    return {
        "mainline_phase": world.mainline_phase,
        "mainline_milestones": deepcopy(world.mainline_milestones),
        "quest_updates": {"create": [], "update": []},
        "mode": "continue",
        "transition": False,
        "place": world.here or "",
        "place_image_prompt": "",
        "place_brief": "",
        "main_scene": "",
        "carry": head.carry if head else "",
        "state_updates": {"subjects": [], "places": []},
    }


def schema(canonical: dict[str, Any], inherited: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(canonical)
    original = out["properties"]
    delta = {key: original[key] for key in FIELDS}
    # Within an update object the two independent lists are also changes, not
    # obligations to retype empty arrays. Expansion restores the existing shape.
    for key in ("quest_updates", "state_updates"):
        delta[key]["required"] = []
    # Mainline fields are required only before the world is set up.
    base_required = [name for name in ("mainline_phase", "mainline_milestones") if not inherited[name]]
    scene_fields = ["place", "mode", "transition", "place_image_prompt", "place_brief", "main_scene"]
    # ⚠️ **The six scene-change fields are all present or all absent.**
    #
    # They used to be required only for a brand-new world. Once the world had a place, the
    # model could write `place` without `mode`/`transition`, and `expand()` then raised
    # "a scene change must decide mode and transition", failing the whole beat. In a scan
    # of 12 beats, 2 failed this way, one of them for "push open the door next to me".
    #
    # The rule is right (a scene change must be explicit, the server must not guess); it
    # just lived in a later check instead of the schema. Now anyOf expresses it: the branch
    # without place has none of the six keys (additionalProperties off), the branch with
    # place requires all six. A brand-new world only has the second branch.
    without_scene = {key: value for key, value in delta.items() if key not in scene_fields}
    variants = [
        {
            "type": "object",
            "properties": deepcopy(delta),
            "required": base_required + scene_fields,
            "additionalProperties": False,
        }
    ]
    if inherited["place"]:
        variants.insert(
            0,
            {
                "type": "object",
                "properties": deepcopy(without_scene),
                "required": list(base_required),
                "additionalProperties": False,
            },
        )
    out["properties"] = {}
    for key, value in original.items():
        if key in FIELDS:
            if "world_delta" not in out["properties"]:
                out["properties"]["world_delta"] = variants[0] if len(variants) == 1 else {"anyOf": variants}
        else:
            out["properties"][key] = value
    required = set(canonical["required"]) - set(FIELDS)
    out["required"] = [key for key in out["properties"] if key in required or key == "world_delta"]
    return out


class UnregisteredAssetsError(ValueError):
    """The director showed names it never declared; the message is a correction for a re-plan."""

    def __init__(self, names: list[str]) -> None:
        self.names = names
        super().__init__(
            "Your last answer put these in shot without registering them: "
            + ", ".join(map(str, names))
            + ". Declare every new character, creature or object in new_subjects (with its look), "
            "and a new location through world_delta.place with place_image_prompt. Only names "
            "already in the world snapshot may appear without a declaration."
        )


def validate_assets(decision: dict[str, Any], world: World) -> None:
    """An omitted declaration must never silently turn a new body into a lost ref.

    New declarations still run through the existing registration machinery, which
    assigns their final IDs. With no declarations, every named visible asset must
    already exist; an invented S-number is not an existing reference.
    """
    if decision.get("new_subjects") or (decision.get("implicit_player") or {}).get("appears"):
        return
    known = set(world.subjects) | set(world.places)
    known.update(rec.get("name") for rec in world.subjects.values())
    known.update(rec.get("name") for rec in decision.get("new_places") or [])
    # The place this beat moves to (new or not) is declared through world_delta.place.
    known.add(decision.get("place"))
    visible = list(decision.get("subjects") or [])
    sequence = decision.get("action_sequence")
    steps = sequence.get("steps") or [] if isinstance(sequence, dict) else []
    if steps and isinstance(steps[0], dict):
        visible.extend(steps[0].get("visible_subjects") or [])
    missing = list(dict.fromkeys(name for name in visible if name not in known))
    if missing:
        raise UnregisteredAssetsError(missing)


def _misplaced_fields(delta: dict[str, Any], out: dict[str, Any]) -> dict[str, Any]:
    """Move top-level fields the model nested in the delta back out; anything else fails closed.

    Models sometimes write e.g. `mainline` next to `mainline_phase` inside world_delta. Its
    meaning is unambiguous, so it is read where it belongs instead of failing the beat.
    """
    extra = set(delta) - set(FIELDS)
    if not extra:
        return delta
    from murea_agent import reqlog
    from murea_agent.director.agent_schema import GM_SCHEMA

    top_level = set(GM_SCHEMA["properties"])
    if not all(k in top_level and k not in out for k in extra):
        raise ValueError("Agent world_delta is malformed")
    for key in extra:
        out[key] = delta[key]
    reqlog.record("world_delta_repaired", moved=sorted(extra))
    return {k: v for k, v in delta.items() if k in FIELDS}


def expand(part: dict[str, Any], inherited: dict[str, Any]) -> dict[str, Any]:
    """Absent streamed delta stays absent; only a completed object means no change."""
    out = deepcopy(part)
    if "world_delta" not in out:
        return out
    delta = out.pop("world_delta")
    if not isinstance(delta, dict):
        raise ValueError("Agent world_delta is malformed")
    delta = _misplaced_fields(delta, out)
    if (
        "place" in delta
        and delta["place"] != inherited["place"]
        and not {"mode", "transition"}.issubset(delta)
    ):
        raise ValueError("Agent must set mode and transition when changing place")
    for key in FIELDS:
        if key in out:
            # The same value written twice is harmless; two different values are ambiguous.
            if key in delta and out[key] == delta[key]:
                del out[key]
            else:
                raise ValueError(f"Agent returned both world_delta and top-level {key}")
        value = deepcopy(delta[key] if key in delta else inherited[key])
        if key in ("quest_updates", "state_updates"):
            if not isinstance(value, dict):
                raise ValueError(f"Agent {key} must be an object")
            value = {**deepcopy(inherited[key]), **value}
        out[key] = value
    return out
