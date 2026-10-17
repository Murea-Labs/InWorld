"""JSON schemas and vocabularies of the agent's structured output."""

from __future__ import annotations

from concurrent import futures
from typing import Any


SECTION_ORDER = (
    "subject_definitions",
    "summary",
    "retention_analysis",
    "detailed_description",
    "overall_soundscape",
    "non_diegetic_music",
)

# The motion plan does not describe a particular prop but "which single spatial relation changes in
# this beat". Vehicles, doors, chairs, stairs, picking up, hugging and riding share one relation
# vocabulary, so fixing one case does not reveal another kind of clipping.
MOTION_TYPES = (
    "none",
    "locomotion",
    "boundary",
    "manipulation",
    "support",
    "contact",
    "impact",
)

MOTION_PHASES = ("none", "approach", "engage", "transform", "disengage", "settle")

MOTION_RELATIONS = (
    "none",
    "distance",
    "contact",
    "containment",
    "support",
    "possession",
    "configuration",
)

MOTION_FRAMINGS = ("none", "wide", "medium-wide", "medium")

MOTION_EVIDENCE = (
    "actor_full_body",
    "actor_hands",
    "actor_feet",
    "target_full_structure",
    "contact_point",
    "support_surface",
    "boundary_opening",
    "both_actors_contact",
    "instrument_full_structure",
    "instrument_path",
)

# Fields that only matter when real body movement happens. They are always empty on static /
# dialogue shots; see the measurements in `_motion_replan_schema`.
_MOTION_DETAIL_FIELDS = (
    "actors",
    "instrument",
    "target",
    "start_state",
    "path",
    "contact_points",
    "end_state",
    "camera_side",
    "visual_evidence",
    "must_show",
    "invariants",
)

MOTION_PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "interaction_type": {"type": "string", "enum": list(MOTION_TYPES)},
        "phase": {"type": "string", "enum": list(MOTION_PHASES)},
        "relation_change": {"type": "string", "enum": list(MOTION_RELATIONS)},
        "actors": {"type": "array", "items": {"type": "string"}},
        # An inanimate intermediary that carries the action to the target: a thrown
        # object, tool, vehicle part, liquid, and so on.  Keeping it separate from
        # `actors` and `target` prevents object-to-object impact from being rewritten
        # as body contact between two people.
        "instrument": {"type": "string"},
        "target": {"type": "string"},
        "start_state": {"type": "string"},
        "path": {"type": "string"},
        "contact_points": {"type": "string"},
        "end_state": {"type": "string"},
        "camera_side": {"type": "string"},
        "framing": {"type": "string", "enum": list(MOTION_FRAMINGS)},
        "visual_evidence": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": list(MOTION_EVIDENCE),
            },
        },
        "must_show": {"type": "array", "items": {"type": "string"}},
        "invariants": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "interaction_type",
        "phase",
        "relation_change",
        "actors",
        "instrument",
        "target",
        "start_state",
        "path",
        "contact_points",
        "end_state",
        "camera_side",
        "framing",
        "visual_evidence",
        "must_show",
        "invariants",
    ],
}

# ⚠️ This schema decides **how much the director writes** each beat, and the director's time is almost all
# output (dropping action_sequence + motion_plan cut output from 2274 to 981 characters and the
# median from 3.49 s to 2.57 s, while shrinking the system prompt from 43.9 KB to 14.7 KB saved
# nothing). So only fields **someone actually reads** remain:
#   · `goal` is gone: `_apply_action_sequence` overwrites it with `request_text` right away.
#   · `budget_reason` is gone: the code only checked it was non-empty; asking the model to explain a
#     segment count it already chose was pure cost.
#   · `end_state` / `continuity_anchor` are only read for steps[1:] (`_sequence_continue_with` builds the
#     next beat, `_repair_partial_final_boundary` fixes the last segment), so the first step no longer
#     requires them, see `_action_sequence_errors`.
ACTION_SEQUENCE_SCHEMA = {
    "type": "object",
    "properties": {
        # The complete user intent is enumerated before segment grouping so a
        # terminal consequence cannot disappear merely to make one MP4 fit.
        "events": {
            "type": "array",
            "minItems": 1,
            "maxItems": 3,
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "description": {"type": "string"},
                    "visible_result": {"type": "string"},
                    "estimated_seconds": {"type": "number"},
                },
                "required": [
                    "id",
                    "description",
                    "visible_result",
                    "estimated_seconds",
                ],
            },
        },
        "steps": {
            "type": "array",
            "minItems": 1,
            "maxItems": 3,
            "items": {
                "type": "object",
                "properties": {
                    "instruction": {"type": "string"},
                    # Each step is a separate clip; place / visible_subjects let the server separate
                    # "appears later" from "filmed in this beat" instead of guessing afterwards.
                    "place": {"type": "string"},
                    "visible_subjects": {"type": "array", "items": {"type": "string"}},
                    # These roles are semantic decisions made by the GM, not values
                    # inferred from action-word lists in the service.
                    "actors": {"type": "array", "items": {"type": "string"}},
                    "instruments": {"type": "array", "items": {"type": "string"}},
                    "targets": {"type": "array", "items": {"type": "string"}},
                    # Every intent event is assigned to exactly one segment by id.
                    "event_ids": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 1,
                        "items": {"type": "integer"},
                    },
                    "kind": {"type": "string", "enum": ["motion", "dialogue", "observation"]},
                    "phase": {"type": "string", "enum": list(MOTION_PHASES)},
                    "relation_change": {
                        "type": "string",
                        "enum": list(MOTION_RELATIONS),
                    },
                    "end_state": {"type": "string"},
                    "continuity_anchor": {"type": "string"},
                },
                # ⚠️ The three role fields are not required: they are empty in most beats.
                #
                # In 300 recorded beats `instruments` was empty 94%, `targets` 92%, `actors` 78%.
                # Required, the model would write the key and an empty pair of brackets every beat.
                # `_normalize_action_sequence` uses `.get()` throughout, so a missing key equals an
                # empty list.
                #
                # `event_ids` stays: it backs the check "each intended event goes to exactly one
                # segment" (`_action_sequence_semantic_errors`).
                "required": [
                    "instruction",
                    "place",
                    "visible_subjects",
                    "kind",
                    "event_ids",
                    "phase",
                    "relation_change",
                ],
            },
        },
    },
    "required": ["events", "steps"],
}

QUEST_UPDATES_SCHEMA = {
    "type": "object",
    "properties": {
        # The mainline quest is created by the server from the locked mainline; the director only
        # creates real optional side quests.
        "create": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": ["side"]},
                    "title": {"type": "string"},
                    "summary": {"type": "string"},
                    "progress": {"type": "string"},
                    "place": {"type": "string"},
                },
                "required": ["type", "title", "summary", "progress", "place"],
            },
        },
        "update": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "status": {
                        "type": "string",
                        "enum": ["active", "completed", "failed"],
                    },
                    "progress": {"type": "string"},
                    "place": {"type": "string"},
                },
                "required": ["id", "status", "progress", "place"],
            },
        },
    },
    "required": ["create", "update"],
}

GM_SCHEMA = {
    "type": "object",
    "properties": {
        "reply": {"type": "string"},
        # The real maximum is set per request (`director._segment_seconds`).
        "duration_seconds": {"type": "integer", "minimum": 4, "maximum": 20},
        # The mainline is filled once, while the world has none; it must be a long-term goal
        # spanning many beats.
        "mainline": {"type": "string"},
        # Mainline stage spanning many beats; it changes as the story moves, without rewriting the
        # mainline title.
        "mainline_phase": {"type": "string"},
        # 2–6 stages when the mainline is first set; afterwards keep the order from the snapshot
        # instead of reordering every beat.
        "mainline_milestones": {
            "type": "array",
            "items": {"type": "string"},
        },
        # The goal after this beat, required every beat. It is where the "advance the mainline"
        # choice leads.
        "objective": {"type": "string"},
        # Side quests are created only when a real optional goal appears; existing quests are
        # updated by their stable ids from the snapshot.
        "quest_updates": QUEST_UPDATES_SCHEMA,
        "mode": {"type": "string", "enum": ["continue", "cut"]},
        # On a place change: did the person **walk** there (the camera can follow) or **jump** there
        "transition": {"type": "boolean"},
        "place": {"type": "string"},
        "place_image_prompt": {"type": "string"},
        "place_brief": {"type": "string"},
        "main_scene": {"type": "string"},
        # ⚠️ The position matters; do not move it later. Schema field order is the order the model streams, and
        # **which reference images to draw is known only when this field is complete**; the server
        # queues drawing at that moment (`_warm_images`) to overlap it with the rest of the
        # director's output. Placed late, little remained to overlap. It only needs reply and place,
        # both of which come before it.
        "new_subjects": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "kind": {"type": "string", "enum": ["人物", "生物", "物件"]},
                    "voice": {"type": "string"},
                    "image_prompt": {"type": "string"},
                    "brief": {"type": "string"},
                },
                "required": ["name", "kind", "voice", "image_prompt", "brief"],
            },
        },
        "beat": {"type": "string"},
        "action_sequence": ACTION_SEQUENCE_SCHEMA,
        "motion_plan": MOTION_PLAN_SCHEMA,
        "dialogue": {"type": "string"},
        "carry": {"type": "string"},
        "options": {"type": "array", "items": {"type": "string"}},
        # Which of the three is recommended (0/1/2). Usually the one advancing the mainline; another
        # if the story needs it.
        "recommend": {"type": "integer"},
        "subjects": {"type": "array", "items": {"type": "string"}},
        # Of this clip's `subjects`, the ones **not visible in the last attached frame**. A
        # continuation only inherits the last ~0.7 s of the previous clip and no reference images,
        # so a subject missing there is redrawn from text alone (a greatsword came back as a
        # short glowing axe). The server answers a non-empty list with a referenced fresh shot
        # instead of a continuation. Read before the video is submitted: never a late field.
        "missing_from_last_frame": {"type": "array", "items": {"type": "string"}},
        # While the world has no user-played character, the director must decide explicitly whether
        # this beat shows the implicit player's body for the first time. If so the server registers
        # it; an off-screen voice does not count.
        "implicit_player": {
            "type": "object",
            "properties": {
                "appears": {"type": "boolean"},
                "name": {"type": "string"},
                "voice": {"type": "string"},
                "image_prompt": {"type": "string"},
                "brief": {"type": "string"},
            },
            "required": ["appears", "name", "voice", "image_prompt", "brief"],
        },
        # The destination space may be registered before the current clip enters it; it is still a
        # place asset and takes no character slot.
        "new_places": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "image_prompt": {"type": "string"},
                    "brief": {"type": "string"},
                    "main_scene": {"type": "string"},
                },
                "required": ["name", "image_prompt", "brief", "main_scene"],
            },
        },
        "state_updates": {
            "type": "object",
            "properties": {
                "subjects": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "location": {"type": "string"},
                            "outfit": {"type": "string"},
                            "condition": {"type": "string"},
                            "mood": {"type": "string"},
                            "goal": {"type": "string"},
                            "knowledge_add": {"type": "array", "items": {"type": "string"}},
                            "knowledge_remove": {"type": "array", "items": {"type": "string"}},
                            "relation_updates": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "target": {"type": "string"},
                                        "relation": {"type": "string"},
                                    },
                                    "required": ["target", "relation"],
                                },
                            },
                        },
                        "required": ["name"],
                    },
                },
                "places": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "time_of_day": {"type": "string"},
                            "weather": {"type": "string"},
                            "lighting": {"type": "string"},
                            "condition": {"type": "string"},
                            "props_add": {"type": "array", "items": {"type": "string"}},
                            "props_remove": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["name"],
                    },
                },
            },
            "required": ["subjects", "places"],
        },
    },
    "required": [
        # mainline is required too: the first time it sets the long-term goal, later it is an empty
        # string; stages and milestones may update as the story moves. If optional, the model mostly
        # never fills it and the world never gets a mainline.
        "reply",
        "duration_seconds",
        "mainline",
        "mainline_phase",
        "mainline_milestones",
        "objective",
        "quest_updates",
        "mode",
        "transition",
        "place",
        "place_image_prompt",
        "place_brief",
        "main_scene",
        "beat",
        "action_sequence",
        "motion_plan",
        "dialogue",
        "carry",
        "options",
        "recommend",
        "subjects",
        "missing_from_last_frame",
        "implicit_player",
        "new_subjects",
        "new_places",
        "state_updates",
    ],
}

# Fields used only after the video. Last in the schema, so the server can submit the video once the
# rest is written while these finish during rendering.
#
# ⚠️ This list is defined **by subtraction**: `_agent_partial_ready` checks "is everything except these
# written". Forgetting a field here only delays submission (safe); wrongly listing a field the video
# model needs is what breaks things.
#
# Together they are about 16% of the output. `state_updates` looks like post-video data but is not:
# dynamic state must be computed first and locked into the six sections.
AGENT_LATE_FIELDS = (
    "options",
    "objective",
    "recommend",
    "mainline",
)

# ⚠️ Do not add fields to the list above without first checking:
#
#     does any helper that receives d before the video model read this key **inside its body**.
#
# The first version failed exactly here: `implicit_player` / `quest_updates` / `carry` were listed
# as late fields because their literals did not appear in `turn()`; they are read inside
# `_inject_implicit_player` / `_sanitize_intermediate_directive` / `_stage_declared_places`, all
# called before the video. The early submission path then proceeded without them: the implicit
# player was never registered, **silently** (`raw` is None and it just returns).
#
# The check is by subtraction, so a missing entry only delays submission (safe); a wrong entry
# breaks things.


def _agent_schema(drop_definitions: bool = False) -> dict[str, Any]:
    """Merged schema = the director's decision fields + the rewriter's six sections + the physics audit.

    A separate schema instead of changing GM_SCHEMA: both paths must coexist (`config.MERGED_AGENT`
    off goes back to two calls), and the prefetch cache may still hold results in the old format.

    ⚠️ **The order of properties is functional**: the model writes fields in this order. The six sections come
    before `AGENT_LATE_FIELDS`, so the server can send the job to the video model as soon as they
    are written instead of waiting for options / quest_updates. Same trick as moving `new_subjects`
    early to overlap drawing with the rest of the director's output.
    """
    import copy as _copy

    out = _copy.deepcopy(GM_SCHEMA)
    props = out["properties"]
    late = {k: props.pop(k) for k in AGENT_LATE_FIELDS if k in props}
    # ⚠️ When every entity has locked wording, do not ask the model for `subject_definitions`.
    #
    # The mechanism existed (`REWRITER_SCHEMA_NO_DEFS`, about 30% fewer output tokens) but only on
    # the separate rewriter path; the merged path still wrote it every beat, and the server
    # overwrites the whole section with the locked assets anyway, so everything the model wrote was
    # **discarded**.
    #
    # Measured on the same world and inputs: one model wrote 2 characters (it learned to skip it), a
    # local model wrote 238 characters ≈ 133 tokens ≈ 0.31 s wasted per beat.
    #
    # The check is `_locked_definitions`: non-empty means the world's entities are all locked.
    sections = _MODEL_SECTIONS
    if drop_definitions:
        sections = tuple(k for k in _MODEL_SECTIONS if k != "subject_definitions")
    for key in (*sections, *_MOTION_AUDIT):
        props[key] = {"type": "string"}
    if drop_definitions:
        props.pop("subject_definitions", None)
    props.update(late)  # late fields go last
    # ⚠️ **`required` must be reordered too**: it, not properties, decides generation order.
    #
    # Measured (same schema, only required reversed):
    #     current            output order vs required +1.000  vs properties +0.641
    #     required reversed  output order vs required +1.000  vs properties -0.625
    #                        the first 6 fields flipped entirely
    #
    # So the order of properties has **no effect** on generation order. Before this only properties
    # had been reordered, and two careful optimizations saved nothing:
    #
    #   · "submit the video once the six sections are written": the late fields were at positions 1/4/17/18 of
    #     required, saving only 0.06 s
    #   · "move new_subjects early to hide drawing": it was 21st in required, so early drawing never got its window
    #
    # After the fix: new_subjects arrives at 1.27 s instead of 2.56 s, the six sections at 3.07 s
    # instead of 3.46 s.
    need = set(out["required"]) | set(_MODEL_SECTIONS) | set(_MOTION_AUDIT)
    out["required"] = [k for k in props if k in need]
    return out


def _agent_partial_ready(part: dict[str, Any], schema: dict[str, Any]) -> bool:
    """Whether the stream so far already has everything the video model needs.

    By subtraction: every required field except `AGENT_LATE_FIELDS` must be present. Better to
    submit a little later than to build a prompt from half a decision.
    """
    need = set(schema.get("required") or ()) - set(AGENT_LATE_FIELDS)
    return need.issubset(part)


_early_pool = futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="agent-early")

_density_pool = futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="agent-density")

# Six-section parts **the server builds itself** instead of asking the model. All three parts of
# `retention_analysis` can be derived: structure from pictures, shot size from motion_plan.framing,
# the retention level is always fully_preserved in this design, and descriptions come from the
# locked English definitions. Rewriting it every beat cost ~126 tokens ≈ 0.4 s for uninformative
# text like "character attributes match reference image".
_SERVER_SECTIONS = ("retention_analysis",)

_MODEL_SECTIONS = tuple(k for k in SECTION_ORDER if k not in _SERVER_SECTIONS)

_MOTION_AUDIT = (
    "motion_phase",
    "motion_relation_change",
    "motion_target",
    "motion_framing",
)

REWRITER_SCHEMA = {
    "type": "object",
    "properties": {k: {"type": "string"} for k in (*_MODEL_SECTIONS, *_MOTION_AUDIT)},
    "required": [*_MODEL_SECTIONS, *_MOTION_AUDIT],
}

# When every entity already has locked wording there is nothing to write for
# subject_definitions, and asking for it anyway costs ~30% of the output tokens.
_NO_DEFS = [k for k in _MODEL_SECTIONS if k != "subject_definitions"]

REWRITER_SCHEMA_NO_DEFS = {
    "type": "object",
    "properties": {k: {"type": "string"} for k in (*_NO_DEFS, *_MOTION_AUDIT)},
    "required": [*_NO_DEFS, *_MOTION_AUDIT],
}
