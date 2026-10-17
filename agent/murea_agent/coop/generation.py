"""One joint shot, with one reference-bound action for every co-op participant.

Single-player planning intentionally decomposes complex actions into a single
physical interaction. Co-op needs concurrent actors instead: validate the full
cast, compose one caption and invoke the existing Ref2VA client exactly once.
"""

from __future__ import annotations
import copy
import json
import re
import time
import uuid
from dataclasses import dataclass

from murea_agent.events import Emitter
from murea_agent.shot import prompt_budget
from murea_agent import config
from murea_agent.shot import video_prompt
from murea_agent.coop import (
    duration as coop_duration,
    trace as coop_trace,
    media as coop_media,
    caption as coop_caption,
    memory as coop_memory,
    boss as coop_boss,
)
from murea_agent.providers import llm, video
from murea_agent import reqlog
from murea_agent.coop.scene_plan import (
    AssetResolver,
    PlannedAction,
    ScenePlan,
    ShotCompiler,
    TurnContext,
    WorldSnapshot,
)
from murea_agent.director.prompt_language import contains_non_latin_text, validate_english_fields
from murea_agent.memory.world import Node
from murea_agent.memory.world_media import has_media, media_path
from murea_agent.coop.messages import msg
from murea_agent.coop.generation_prompts import (
    _ACTION_SCHEMA,
    _ACTION_SYSTEM,
    _CAMERA,
    _CONTINUATION_CAMERA,
    _REFERENCE_LAYOUT,
    _SCHEMA,
    _SYSTEM,
)
from murea_agent.coop.action_text import (  # re-exported for callers of generation
    _ACTION_REPAIR_FIELDS,
    _action_metadata_aliases,
    _english,
    _generation_narration,
    _locked_reference_definitions,
    _normalize_action_ids,
    _normalize_action_metadata,
    _optional_english,
    _plain_beat,
    _prompt_text,
    _recover_missing_action_entries,
    _repair_english_fields_batch,
    _repair_plan_english_fields,
    _repairable_plan_fields,
    _unsheet_definitions,
    _validate,
    _validate_model_timeline,
    _validate_prompt_sections,
)


PROMPT_VERSION = 11


def frame_images(node):
    names = (
        (node.get("frames", []) if isinstance(node, dict) else getattr(node, "frames", [])) if node else []
    )
    return [
        ("image/jpeg", coop_media.read_image(media_path("frames", name)))
        for name in names[-3:]
        if has_media("frames", name)
    ]


def _previous_reference_traits(members, head):
    """Reference traits from the previous shot, when the same players continue.

    They keep the reference wording stable across rounds, and the model is then not
    asked to describe the characters again (nor sent their sheets).
    """
    locked = {}
    if (
        head
        and head.directive.get("prompt_version") in (4, 5, 6, 7, 8, 9, 10, PROMPT_VERSION)
        and head.refs[: len(members)] == [m["ref"] for m in members]
    ):
        candidates = {e["user_id"]: e for e in head.directive.get("shot", {}).get("actions", [])}
        previous_players = {m["user_id"]: m for m in head.directive.get("players", [])}
        if all(
            m["user_id"] in candidates
            and all(
                previous_players.get(m["user_id"], {}).get(k) == m.get(k)
                for k in ("character", "description")
            )
            for m in members
        ):
            locked = {uid: {k: e[k] for k in ("appearance",) if k in e} for uid, e in candidates.items()}
    return locked


def _action_request(members, context, locked):
    """The schema and system prompt for one planning call."""
    schema = json.loads(json.dumps(_ACTION_SCHEMA))
    entry_schema = schema["properties"]["actions"]["items"]
    if locked:
        # reference_layout is no longer asked from the model (fixed by the server); only appearance
        # remains.
        for key in ("appearance",):
            entry_schema["properties"].pop(key, None)
            if key in entry_schema["required"]:
                entry_schema["required"].remove(key)
    if context is not None:
        schema["properties"].update(_SCHEMA["properties"])
        schema["required"].extend(_SCHEMA["required"])
        if context.get("shared_creature"):
            schema["properties"]["monster_appearance"] = {"type": "string"}
            schema["required"].append("monster_appearance")
    system = (
        _ACTION_SYSTEM
        + coop_boss.PLANNING_INSTRUCTION
        + ("\nAlso provide scene, sound and event_action. " + _SYSTEM if context is not None else "")
    )
    if context is not None and not context.get("opening"):
        schema["properties"]["timing"] = coop_duration.timing_schema(
            members, bool(context.get("upcoming_world_event"))
        )
        schema["required"].append("timing")
        system += coop_duration.TIMING_INSTRUCTION
    if context is not None and context.get("upcoming_world_event"):
        system += (
            "\nupcoming_world_event is an agent preview for the next shot, not an observed fact. "
            "Translate this single BOSS/environment behavior into event_action only when it can happen alongside the locked player actions; "
            "never replace, cancel or finish a player action because of it."
        )
        if context.get("upcoming_world_event_type") == "boss_introduction":
            system += (
                "\nThis event deliberately introduces one new unreferenced BOSS. Follow upcoming_boss_candidate "
                "for its identity and stable visible design, and show one clear full silhouette. "
                # It has no reference image yet, so subject_bindings has no number for it.
                # Without saying so the model invents a <Subject N+1>, which `_text()`
                # rejects (only <Subject 1>..<Subject N> are allowed), failing the whole
                # beat.
                "It has NO reference picture and therefore NO Subject token: describe it in plain "
                "English words only, and never invent a <Subject N> or <Picture N> number for it."
            )
    if context is not None and context.get("shared_creature"):
        system += (
            "\nThe shared creature reference is a separate non-player subject. "
            "monster_appearance: 15-40 English words describing its visible anatomy, colors and stable traits. "
            "Keep exactly this creature as the shared target; never replace it with a different species or player. "
            "Use shared_scene and its reference for the environment, never a character or creature reference backdrop. "
            "scene establishes current relative positions; event_action describes their changes. "
            "The creature is NOT a player action entry."
        )
    return schema, system


def _action_payload(members, setting, *, duration, context, shared_refs, boss_life, locked):
    """The planner's input: players, locked traits, image order and subject bindings."""
    payload = {
        "duration_seconds": duration,
        "duration_policy": coop_duration.policy_signature(),
        "setting": setting,
        "boss_life": boss_life or {},
        "locked_reference_traits": locked,
        "image_order": {
            "character_references": [] if locked else [m["user_id"] for m in members],
            "shared_references_next": shared_refs,
            "then": "previous video frames in time order",
        },
        "players": [{k: m[k] for k in ("user_id", "character", "description", "action")} for m in members],
    }
    if context is not None:
        payload.update(context)
        payload["subject_bindings"] = [
            {"subject": f"<Subject {i}>", "user_id": m["user_id"], "character": m["character"]}
            for i, m in enumerate(members, 1)
        ]
        for offset, asset in enumerate(shared_refs, len(members) + 1):
            if asset.get("kind") == "scene":
                payload["subject_bindings"].append(
                    {"subject": f"<Subject {offset}>", "environment": context["shared_scene"]}
                )
            else:
                payload["subject_bindings"].append(
                    {"subject": f"<Subject {offset}>", "creature": context["shared_creature"]}
                )
    return payload


_VISIBILITY = {
    "full body",
    "most of the body",
    "waist-up",
    "chest-up",
    "face-only",
    "rear view",
    "crop",
    "occlusion",
}
_RETENTION_LEVELS = {"fully_preserved", "partially_preserved", "attribute_transfer", "weak_reference"}


def _finish_action(action, entry, m, *, subject_count, aliases):
    """Make one player's validated entry video-ready: English text, normalized metadata, enums."""
    for field in ("appearance", "action", "beat", "retention"):
        value = entry.get(field)
        if field == "retention" and not value:
            continue  # old plans did not keep per-shot descriptions; keep the known appearance
        if field == "beat":
            value = _plain_beat(value)
        # action may reference bound Subjects; the final shot still uses only the
        # timeline.
        action[field] = _english(value, field, subject_count=subject_count if field == "action" else 0)
        if field in ("action", "beat"):
            action[field] = re.sub(r"^(?:he|she|they)\s+", "", action[field], flags=re.I)
        if field == "beat":
            action[field] = action[field].rstrip(" .;")
    for field in ("start_state", "target", "path", "end_state"):
        action[field] = _normalize_action_metadata(
            entry.get(field),
            field,
            aliases,
            actor_user_id=m["user_id"],
        )
    action["action_id"] = PlannedAction.stable_id(m["user_id"], m["action"])
    for field, allowed in {"visibility": _VISIBILITY, "retention_level": _RETENTION_LEVELS}.items():
        value = entry.get(field)
        if value not in allowed:
            raise ValueError("video prompt " + field + " breaks the six-section format")
        action[field] = value


def _english_plan_fields(plan, members, setting, *, context, subject_count):
    """The environment fields (scene, sound, event action, creature) in English."""
    fields = ["scene", "sound", "event_action"] + (
        ["monster_appearance"] if context.get("shared_creature") else []
    )
    contaminated_plan = {
        field: plan.get(field)
        for field in _repairable_plan_fields(fields)
        if isinstance(plan.get(field), str) and contains_non_latin_text(plan.get(field))
    }
    if contaminated_plan:
        repair_context = dict(context)
        repair_context["setting"] = setting
        repair_context["locked_actions"] = [
            {k: m[k] for k in ("user_id", "character", "action")} for m in members
        ]
        plan.update(
            _repair_plan_english_fields(
                plan, contaminated_plan, context=repair_context, subject_count=subject_count
            )
        )
    result = {
        k: (
            _optional_english(plan.get(k, ""), k, subject_count=subject_count)
            if k == "event_action"
            else _english(plan.get(k, ""), k)
        )
        for k in fields
    }
    # A legacy model may still return timeline. It is deliberately
    # ignored; the compiler below owns the only player choreography.
    result["timeline"] = ""
    result["narration"] = ""
    result["timing"] = plan.get("timing")
    return result


def _locked_actions(
    members, setting, *, head=None, duration=6, context=None, shared_refs=None, boss_life=None
):
    """Translate every player's action into the shot plan; with `context`, plan the environment too.

    Returns `actions` (per user id), or `(actions, environment)` when `context` is given. One
    retry carries the validation error back to the model as `correction_required`.
    """
    shared_refs = shared_refs or []
    locked = _previous_reference_traits(members, head)
    images = (
        []
        if locked
        else [("image/jpeg", coop_media.read_image(media_path("refs", m["ref"]))) for m in members]
    )
    images += [
        ("image/jpeg", coop_media.read_image(media_path("refs", asset["ref"]))) for asset in shared_refs
    ]
    images += frame_images(head)
    schema, system = _action_request(members, context, locked)
    reqlog.record(
        "video_observation_context", node=getattr(head, "id", None), frames=getattr(head, "frames", [])
    )
    payload = _action_payload(
        members,
        setting,
        duration=duration,
        context=context,
        shared_refs=shared_refs,
        boss_life=boss_life,
        locked=locked,
    )
    subject_count = len(payload.get("subject_bindings", []))
    metadata_aliases = {
        member["user_id"]: _action_metadata_aliases(members, shared_refs, member["user_id"])
        for member in members
    }
    for attempt in range(2):
        try:
            plan = coop_trace.chat(
                system=system,
                user=json.dumps(payload, ensure_ascii=False),
                schema=schema,
                model=config.GM_MODEL,
                thinking=0,
                temperature=0,
                tag="coop_plan" if context is not None else "coop_action_translation",
                images=images,
            )
            if context is not None:
                _validate_model_timeline(
                    plan.get("timeline") if isinstance(plan, dict) else None, subject_count
                )
            _normalize_action_ids(plan, members)
            present_ids = (
                {entry.get("user_id") for entry in plan.get("actions", []) if isinstance(entry, dict)}
                if isinstance(plan, dict)
                else set()
            )
            missing = [member for member in members if member["user_id"] not in present_ids]
            if missing:
                # Qwen can collapse two identical-looking opening actions into one row.
                # Recover only the absent players, then keep the normal strict validation.
                recovered = _recover_missing_action_entries(missing, images=images, schema=schema)
                plan.setdefault("actions", []).extend(recovered.values())
            if locked and isinstance(plan, dict):
                for entry in plan.get("actions", []):
                    if isinstance(entry, dict) and entry.get("user_id") in locked:
                        entry.update(locked[entry["user_id"]])
            actions = _validate(plan, members, subject_count=subject_count)
            entries = {e["user_id"]: e for e in plan["actions"]}
            originals = {uid: e.get("source_action") for uid, e in entries.items()}
            # ⚠️ Collect everyone's Chinese contamination first, then repair **once**; this used to call the model once
            # per person inside the loop, on the critical path.
            dirty = []
            for m in members:
                if originals[m["user_id"]] != m["action"]:
                    raise ValueError("the action translation dropped the player's original text")
                contaminated = {
                    field: entries[m["user_id"]].get(field)
                    for field in _ACTION_REPAIR_FIELDS
                    if isinstance(entries[m["user_id"]].get(field), str)
                    and contains_non_latin_text(entries[m["user_id"]].get(field))
                }
                if contaminated:
                    dirty.append((m, contaminated))
            for user_id, repaired in _repair_english_fields_batch(
                dirty, subject_count=subject_count, metadata_aliases=metadata_aliases
            ).items():
                entries[user_id].update(repaired)
            for m in members:
                _finish_action(
                    actions[m["user_id"]],
                    entries[m["user_id"]],
                    m,
                    subject_count=subject_count,
                    aliases=metadata_aliases[m["user_id"]],
                )
            if context is not None:
                return actions, _english_plan_fields(
                    plan, members, setting, context=context, subject_count=subject_count
                )
            return actions
        except ValueError as exc:
            reqlog.record("action_translation_rejected", ok=False, error=str(exc))
            if attempt:
                raise
            payload["correction_required"] = str(exc)
        except llm.GeminiError as exc:
            reqlog.record("action_translation_model_retry", ok=False, error=str(exc))
            if attempt:
                raise
            payload["correction_required"] = (
                "The last reply was not valid JSON; output only JSON that matches the schema."
            )


def _order_shared_references(shared, head):
    """Keep shared assets that appeared before in their original position; new ones are appended at the end.

    ⚠️ The order is not cosmetic: it decides the `<Subject N>` numbers, and when a number changes the latent
    continuation must be dropped. `_shared_references` used to fix "monster first, place last", so
    in one game:

        beat 1  [character 1, character 2, place]            place = <Picture 3>
        beat 2  [character 1, character 2, monster, place]   place pushed to <Picture 4>

    The monster's arrival pushed the place's number and broke continuation. In 64 recorded
    submissions after the first round, 17% could not continue, **all in beats 1–2**, exactly when the
    place and the monster are introduced and players form their first impression of continuity.
    """
    if head is None or getattr(head, "origin", "") != "coop":
        return list(shared)
    prior = list(getattr(head, "refs", None) or [])
    index = {ref: i for i, ref in enumerate(prior)}
    # A new frame-derived image of an existing scene/creature keeps its slot.
    # Otherwise replacing the creature could push the unchanged scene forward.
    kind_index = {
        ("monster" if p.get("kind") == "creature" else p.get("kind")): i
        for i, p in enumerate(getattr(head, "pictures", None) or [])
        if p.get("kind") in ("scene", "creature")
    }
    return sorted(
        shared, key=lambda asset: index.get(asset.get("ref"), kind_index.get(asset.get("kind"), len(prior)))
    )


def _continuation_decision(world, room, refs, shared):
    """Keep the video chain when an observation only registers visible assets.

    User edits start a new composition, including edits that reuse an image.
    An automatic reference must come from this head or one of its ancestors;
    a reference extracted on another branch is not evidence for this latent.
    Confirmed death may retire just that Boss's reference. Surviving assets
    are rebound in the same relative order, without introducing a new scene.
    """
    head = world.head_node
    decision = {"continue": False, "reason": "opening", "observed_assets": []}
    if room.get("opening") or head is None:
        return decision
    decision["reason"] = "no_coop_parent"
    if head.origin != "coop":
        return decision
    decision["reason"] = "user_asset_edit"
    if room.get("asset_edit_revision", 0) != head.directive.get("asset_edit_revision", 0):
        return decision
    old_players = head.directive.get("players") or []
    decision["reason"] = "players_changed"
    if old_players and [p["user_id"] for p in old_players] != [p["user_id"] for p in room["members"]]:
        return decision
    prior = list(head.refs or [])
    prior_pictures = list(getattr(head, "pictures", None) or [])
    retired_monster = room.get("monster") or head.directive.get("monster") or {}
    retired = retired_monster.get("ref") if coop_boss.status(room, retired_monster) == "dead" else ""
    # Only a proven dead Boss may leave the reference set without restarting
    # the shot. Preserve all surviving identities and their relative order.
    removed = [
        i
        for i, ref in enumerate(prior)
        if ref == retired
        and i >= len(room["members"])
        and i < len(prior_pictures)
        and prior_pictures[i].get("kind") == "creature"
        and prior_pictures[i].get("id") == retired_monster.get("subject_id")
    ]
    if removed:
        prior_pictures = [p for i, p in enumerate(prior_pictures) if i not in removed]
        prior = [ref for i, ref in enumerate(prior) if i not in removed]
        decision["retired_boss_reference"] = retired
    decision["reason"] = "references_removed_or_reordered"
    if not prior or len(refs) < len(prior):
        return decision
    if refs == prior:
        return {
            **decision,
            "continue": True,
            "reason": "dead_boss_reference_retired" if removed else "unchanged_references",
        }
    ancestors = set()
    node_id = head.id
    while node_id and node_id not in ancestors:
        ancestors.add(node_id)
        node_id = (world.nodes.get(node_id) or {}).get("parent")
    by_ref = {asset["ref"]: asset for asset in shared}
    for index, ref in enumerate(refs):
        if index < len(prior) and ref == prior[index]:
            continue
        asset = by_ref.get(ref, {}) if index >= len(room["members"]) else {}
        source_node = asset.get("from_node")
        if not source_node and asset.get("source") == "opening_video":
            # Opening assets from older saves did not record from_node.
            source_node = (room.get("opening_node") or {}).get("id")
        decision["reason"] = "manual_or_unverified_reference"
        if asset.get("source") not in ("opening_video", "round_video") or source_node not in ancestors:
            return decision
        if index < len(prior):
            prior_picture = prior_pictures[index] if index < len(prior_pictures) else {}
            kind = "creature" if asset["kind"] == "monster" else asset["kind"]
            decision["reason"] = "references_removed_or_reordered"
            if prior_picture.get("kind") != kind or ref in prior:
                return decision
        decision["observed_assets"].append(
            {
                "picture": index + 1,
                "kind": asset["kind"],
                "source": asset["source"],
                "from_node": source_node,
            }
        )
    return {**decision, "continue": True, "reason": "observed_assets_registered"}


def _shared_references(room):
    monster = room.get("monster", {})
    shared = (
        [{"kind": "monster", **monster}] if monster.get("ref") and coop_boss.status(room) != "dead" else []
    )
    if room.get("scene_ref"):
        shared.append(
            {
                "kind": "scene",
                "name": room["scene_name"],
                "ref": room["scene_ref"],
                "description": room.get("scene_description", ""),
                "source": room.get("scene_asset_source", ""),
                "from_node": room.get("scene_asset_from_node", ""),
            }
        )
    return shared


def _environment_plan(members, actions, setting, *, head, duration, context, shared_refs):
    """Finish only the joint environment after every actor has been translated."""
    schema = copy.deepcopy(_SCHEMA)
    if context.get("shared_creature"):
        schema["properties"]["monster_appearance"] = {"type": "string"}
        schema["required"].append("monster_appearance")
    system = _SYSTEM + coop_boss.PLANNING_INSTRUCTION
    if not context.get("opening"):
        schema["properties"]["timing"] = coop_duration.timing_schema(
            members, bool(context.get("upcoming_world_event"))
        )
        schema["required"].append("timing")
        system += coop_duration.TIMING_INSTRUCTION
    if context.get("upcoming_world_event"):
        system += (
            "\nupcoming_world_event is an agent preview for the next shot, not an observed fact. "
            "Include this single BOSS/environment behavior only when it can happen alongside the locked player actions; "
            "never replace, cancel or finish a player action because of it."
        )
        if context.get("upcoming_world_event_type") == "boss_introduction":
            system += (
                "\nThis event deliberately introduces one new unreferenced BOSS. Follow upcoming_boss_candidate "
                "for its identity and stable visible design, and show one clear full silhouette. "
                # It has no reference image yet, so subject_bindings has no number for it.
                # Without saying so the model invents a <Subject N+1>, which `_text()`
                # rejects (only <Subject 1>..<Subject N> are allowed), failing the whole
                # beat.
                "It has NO reference picture and therefore NO Subject token: describe it in plain "
                "English words only, and never invent a <Subject N> or <Picture N> number for it."
            )
    if context.get("shared_creature"):
        system += (
            "\nThe shared creature reference is a separate non-player subject. "
            "monster_appearance: 15-40 English words describing its visible anatomy, colors and stable traits. "
            "Keep exactly this creature as the shared target; never replace it with a different species or player. "
            "Use shared_scene and its reference for the environment. scene must contain ONLY environment and "
            "the creature location, not a second creature or a new attack."
        )
    images = [
        ("image/jpeg", coop_media.read_image(media_path("refs", asset["ref"]))) for asset in shared_refs
    ]
    images += frame_images(head)
    payload = {
        "duration_seconds": duration,
        "duration_policy": coop_duration.policy_signature(),
        "setting": setting,
        "locked_actions": [{"user_id": m["user_id"], **actions[m["user_id"]]} for m in members],
        "image_order": {
            "shared_references_first": shared_refs,
            "then": "previous video frames in time order",
        },
        **context,
    }
    payload["subject_bindings"] = [
        {"subject": f"<Subject {i}>", "user_id": m["user_id"], "character": m["character"]}
        for i, m in enumerate(members, 1)
    ]
    for offset, asset in enumerate(shared_refs, len(members) + 1):
        if asset.get("kind") == "scene":
            payload["subject_bindings"].append(
                {"subject": f"<Subject {offset}>", "environment": context["shared_scene"]}
            )
        else:
            payload["subject_bindings"].append(
                {"subject": f"<Subject {offset}>", "creature": context["shared_creature"]}
            )
    for attempt in range(2):
        try:
            plan = coop_trace.chat(
                system=system,
                user=json.dumps(payload, ensure_ascii=False),
                schema=schema,
                model=config.GM_MODEL,
                thinking=0,
                temperature=0,
                tag="coop_environment_plan",
                images=images,
            )
            if isinstance(plan, dict) and "timeline" in plan:
                _validate_model_timeline(plan.get("timeline"), len(payload["subject_bindings"]))
            fields = ["scene", "sound", "event_action"] + (
                ["monster_appearance"] if context.get("shared_creature") else []
            )
            contaminated_plan = {
                field: plan.get(field)
                for field in _repairable_plan_fields(fields)
                if isinstance(plan.get(field), str) and contains_non_latin_text(plan.get(field))
            }
            if contaminated_plan:
                repair_context = dict(context)
                repair_context["setting"] = setting
                repair_context["locked_actions"] = [
                    {k: m[k] for k in ("user_id", "character", "action")} for m in members
                ]
                plan.update(
                    _repair_plan_english_fields(
                        plan,
                        contaminated_plan,
                        context=repair_context,
                        subject_count=len(payload["subject_bindings"]),
                    )
                )
            result = {
                key: (
                    _optional_english(plan.get(key, ""), key, subject_count=len(payload["subject_bindings"]))
                    if key == "event_action"
                    else _english(plan.get(key, ""), key)
                )
                for key in fields
            }
            result["timeline"] = ""
            result["narration"] = ""
            result["timing"] = plan.get("timing")
            return result
        except (ValueError, llm.GeminiError) as exc:
            reqlog.record("environment_plan_retry", ok=False, error=str(exc))
            if attempt:
                raise
            payload["correction_required"] = str(exc)


def prefetch_member_action(world, room, member):
    """Translate one ready action while the other players are still deciding."""
    head = world.head_node
    shared_refs = _order_shared_references(_shared_references(room), head)
    result = _locked_actions(
        [member],
        room["setting"],
        head=head,
        duration=coop_duration.planning_seconds(room),
        shared_refs=shared_refs,
        boss_life=coop_boss.context(room),
    )
    return result[member["user_id"]]


def _cached_actions(members, cached, *, shared_refs=None):
    if not isinstance(cached, dict) or any(m["user_id"] not in cached for m in members):
        return None
    actions = {}
    allowed = {
        "visibility": {
            "full body",
            "most of the body",
            "waist-up",
            "chest-up",
            "face-only",
            "rear view",
            "crop",
            "occlusion",
        },
        "retention_level": {"fully_preserved", "partially_preserved", "attribute_transfer", "weak_reference"},
    }
    metadata_aliases = {
        member["user_id"]: _action_metadata_aliases(members, shared_refs or [], member["user_id"])
        for member in members
    }
    for member in members:
        raw = cached[member["user_id"]]
        if not isinstance(raw, dict):
            return None
        entry = {}
        if raw.get("retention"):
            entry["retention"] = _english(raw["retention"], "retention")
        for field in ("appearance", "action", "beat"):
            raw_value = _plain_beat(raw.get(field)) if field == "beat" else raw.get(field)
            entry[field] = _english(raw_value, field)
        for field in ("start_state", "target", "path", "end_state"):
            entry[field] = _normalize_action_metadata(
                raw.get(field), field, metadata_aliases[member["user_id"]], actor_user_id=member["user_id"]
            )
        entry["action_id"] = str(
            raw.get("action_id") or PlannedAction.stable_id(member["user_id"], member["action"])
        )
        for field, values in allowed.items():
            if raw.get(field) not in values:
                return None
            entry[field] = raw[field]
        actions[member["user_id"]] = entry
    return actions


def video_model_of(world) -> str:
    """The video service a co-op world films with: the one it was created with, or the
    configured default when that one is not available in this deployment."""
    chosen = str(getattr(world, "video_model", "") or "")
    return chosen if chosen in video.configured() else config.VIDEO_MODEL


@dataclass
class _RoundInputs:
    """What the perceive step reads from the room and the head node."""

    members: list
    head: object
    boss_life: dict
    monster: dict
    shared_refs: list
    refs: list
    continuation_decision: dict
    video_model: str
    same_scene: bool
    previous: str | None
    duration: float
    intro: dict
    action_order: list
    context: dict
    creature_subject_number: int


@dataclass
class _StagedReferences:
    """Reference images in submission order, with the caption lines that bind them."""

    refs: list
    pictures: list
    definitions: list
    retention: list
    scene_subject: int
    creature_line: str


def _observe(world, room, mentions) -> _RoundInputs:
    """Members, shared assets, the continuation decision and the director's context."""
    from murea_agent.coop import chapter as coop_chapter

    if coop_chapter.blocked(room):
        raise ValueError("the next chapter's assets are not ready")
    members = room["members"]
    head = world.head_node
    if mentions != [m["subject_id"] for m in members]:
        raise ValueError("joint shot reference characters do not match")
    boss_life = coop_boss.context(room)
    monster = {} if boss_life["status"] == "dead" else room.get("monster", {})
    shared_refs = _order_shared_references(_shared_references(room), head)
    refs = [m["ref"] for m in members] + [asset["ref"] for asset in shared_refs]
    if not 2 <= len(members) <= 4 or len(refs) > 5 or any(not has_media("refs", ref) for ref in refs):
        raise ValueError("joint shot has no valid references or more than five")
    continuation_decision = _continuation_decision(world, room, refs, shared_refs)
    video_model = video_model_of(world)
    provider = video.model(video_model)
    if continuation_decision["continue"] and not provider.supports_continuation():
        # The video service cannot continue a clip: film the same scene again
        # from the references instead of stopping the round.
        continuation_decision = {
            **continuation_decision,
            "continue": False,
            "reason": "continuation_unavailable",
        }
    elif (
        continuation_decision["continue"]
        and not provider.references_on_continuation
        and continuation_decision["reason"] not in ("unchanged_references", "dead_boss_reference_retired")
    ):
        # A continuation without reference images cannot introduce a newly
        # registered asset; film the round from the references.
        continuation_decision = {
            **continuation_decision,
            "continue": False,
            "reason": "new_reference_needs_images",
        }
    same_scene = continuation_decision["continue"]
    previous = head.artifact_id if same_scene else None
    reqlog.record(
        "coop_continuation_decision",
        ok=True,
        node=getattr(head, "id", None),
        has_parent_artifact=bool(getattr(head, "artifact_id", None)),
        **continuation_decision,
    )
    if same_scene and not previous:
        raise ValueError(
            "the previous clip has no usable latent continuation handle; stopped so the scene is not regenerated"
        )
    duration = coop_duration.planning_seconds(room)
    intro = dict(room.get("intro") or {})
    if boss_life["status"] == "dead" and intro.get("world_event_type") == "boss_action":
        # Accepted forecasts may predate the visual death audit. Keep player
        # intent and omit this obsolete automated Boss action.
        intro.update(world_event="", world_event_type="none", world_event_key="", boss_candidate={})
        reqlog.record("coop_dead_boss_forecast_omitted", boss=boss_life)
    by_id = {member["user_id"]: member for member in members}
    action_order = [uid for uid in room.get("action_order", []) if uid in by_id]
    action_order.extend(member["user_id"] for member in members if member["user_id"] not in action_order)
    context = {
        "visual_style": room["visual_style"],
        "continuation": bool(previous),
        "boss_life": boss_life,
        "opening": bool(room.get("opening")),
        "previous_scene": head.directive.get("shot", {}).get("scene", "") if previous else "",
        "opening_situation": (
            intro.get("narration", "") if room.get("opening") else coop_chapter.entry(room)
        ),
        "upcoming_world_event": (intro.get("world_event", "") if not room.get("opening") else ""),
        "upcoming_world_event_type": (
            intro.get("world_event_type", "none") if not room.get("opening") else "none"
        ),
        "upcoming_boss_candidate": (intro.get("boss_candidate") or {} if not room.get("opening") else {}),
        "observed_story_state": coop_memory.current(room.get("story_state") or {}),
        "recent_story_history": [
            {
                "round": item.get("round"),
                "forecast": item.get("guidance", {}),
                "observation": item.get("observation", {}),
            }
            for item in (room.get("rounds") or [])[-4:]
        ],
        "narration_inputs": [
            {k: by_id[uid].get(k, "") for k in ("character", "action")} for uid in action_order
        ],
        "shared_scene": {k: room.get(k, "") for k in ("scene_name", "scene_description")},
        "shared_creature": {k: monster[k] for k in ("name", "description")} if monster.get("ref") else {},
    }
    creature_subject_number = next(
        (len(members) + index for index, asset in enumerate(shared_refs, 1) if asset.get("kind") != "scene"),
        0,
    )
    return _RoundInputs(
        members=members,
        head=head,
        boss_life=boss_life,
        monster=monster,
        shared_refs=shared_refs,
        refs=refs,
        continuation_decision=continuation_decision,
        video_model=video_model,
        same_scene=same_scene,
        previous=previous,
        duration=duration,
        intro=intro,
        action_order=action_order,
        context=context,
        creature_subject_number=creature_subject_number,
    )


def _plan_actions(obs: _RoundInputs, room, pretranslated):
    """Lock every player's action and plan the environment; returns (actions, plan, duration decision, cached)."""
    members, head, duration, context, shared_refs = (
        obs.members,
        obs.head,
        obs.duration,
        obs.context,
        obs.shared_refs,
    )
    actions = _cached_actions(members, pretranslated, shared_refs=shared_refs)
    cached = actions is not None
    if actions is None:
        actions, plan = _locked_actions(
            members,
            room["setting"],
            head=head,
            duration=duration,
            context=context,
            shared_refs=shared_refs,
        )
    else:
        plan = _environment_plan(
            members,
            actions,
            room["setting"],
            head=head,
            duration=duration,
            context=context,
            shared_refs=shared_refs,
        )
    duration_decision = coop_duration.decide(room, plan.pop("timing", None))
    duration = duration_decision.seconds
    reqlog.record("coop_duration_selected", duration=duration, duration_policy=duration_decision.as_dict())
    return actions, plan, duration_decision, cached


def _check_world_event(obs: _RoundInputs, room, plan):
    """Validate the accepted forecast and write it into the plan; returns (players, text, type, key)."""
    members, monster, shared_refs, context, intro = (
        obs.members,
        obs.monster,
        obs.shared_refs,
        obs.context,
        obs.intro,
    )
    creature_subject_number = obs.creature_subject_number
    players = [{k: m[k] for k in ("user_id", "character", "description", "action")} for m in members]
    event_text = _prompt_text(context.get("upcoming_world_event", ""), "动态预告", 300)
    event_type = context.get("upcoming_world_event_type", "none")
    event_key = intro.get("world_event_key", "") if not room.get("opening") else ""
    if intro.get("status") == "ready":
        if event_text and (
            event_type == "none"
            or not isinstance(event_key, str)
            or not re.fullmatch(r"[a-z0-9_]{2,80}", event_key.strip().casefold())
        ):
            raise ValueError(
                "joint shot forecast has no valid world_event_key; stopped using the old forecast"
            )
        if not event_text and event_type != "none":
            raise ValueError("joint shot forecast type and content disagree")
    raw_candidate = context.get("upcoming_boss_candidate") or {}
    if not isinstance(raw_candidate, dict):
        raise ValueError("joint shot boss forecast is malformed")
    candidate = {
        "name": _prompt_text(raw_candidate.get("name", ""), "BOSS 名称", 120),
        "description": _prompt_text(raw_candidate.get("description", ""), "BOSS 外观", 700),
    }
    plan["world_event"] = event_text
    plan["world_event_type"] = event_type
    plan["world_event_key"] = event_key
    plan["boss_candidate"] = candidate
    # The model's old timeline is intentionally ignored.  ScenePlan is the
    # immutable boundary object and ShotCompiler owns the sole choreography.
    event_action = plan.get("event_action", "") if isinstance(plan, dict) else ""
    if event_text:
        if not str(event_action).strip():
            if event_type == "boss_introduction" and not monster.get("ref"):
                # Candidate display text may be Chinese. The normal planner
                # translates it; this deterministic recovery must stay English
                # so one omitted field never contaminates the video model boundary.
                event_action = (
                    "A single nonhuman creature enters the scene "
                    "and presents a clear, stable full-body silhouette."
                )
            elif event_type == "boss_introduction":
                event_action = (
                    f"<Subject {creature_subject_number}> enters the scene and becomes clearly visible."
                )
            elif monster.get("ref"):
                event_action = f"<Subject {creature_subject_number}> performs the upcoming creature behavior in the scene."
            else:
                event_action = "The environment changes as previewed."
        plan["event_action"] = _english(
            event_action, "event_action", 900, subject_count=len(members) + len(shared_refs)
        )
    else:
        plan["event_action"] = ""
    return players, event_text, event_type, event_key


def _stage_references(world, obs: _RoundInputs, room, plan, actions, event_text) -> _StagedReferences:
    """Number the references (players first, then shared assets) and write their definitions."""
    members, head, previous, shared_refs, monster = (
        obs.members,
        obs.head,
        obs.previous,
        obs.shared_refs,
        obs.monster,
    )
    # Waiting narration is presentation only. Compile it from the locked
    # player inputs and permitted forecast instead of asking the planner to
    # create another account of what happens in the video.
    plan["narration"] = _generation_narration(members, event_text)
    plan["reply"] = plan["narration"]
    plan["camera"] = _CONTINUATION_CAMERA if previous else _CAMERA
    if not previous:
        for action in actions.values():
            action["visibility"] = "full body"
    plan["actions"] = [
        {
            "user_id": m["user_id"],
            "source_action": m["action"],
            **actions[m["user_id"]],
            "action_id": actions[m["user_id"]].get("action_id")
            or PlannedAction.stable_id(m["user_id"], m["action"]),
        }
        for m in members
    ]
    refs = [m["ref"] for m in members]
    pictures = [
        {"n": i, "id": m["subject_id"], "kind": "character", "name": m["character"], "ref": m["ref"]}
        for i, m in enumerate(members, 1)
    ]
    definitions = [
        f"<Subject {i}> is the character shown in <Picture {i}> as {_REFERENCE_LAYOUT}; {actions[m['user_id']]['appearance']}"
        for i, m in enumerate(members, 1)
    ]
    if previous and head.directive.get("prompt_version") in (4, 5, 6, 7, 8, 9, 10, PROMPT_VERSION):
        # Stable reference wording survives subsequent rounds verbatim.
        locked = _locked_reference_definitions(world, head, len(members))
        if (
            locked
            and not any("[compressed]" in line for line in locked)
            and all(f"<Subject {i}>" in line and f"<Picture {i}>" in line for i, line in enumerate(locked, 1))
        ):
            definitions = _unsheet_definitions(locked)
    retention = [
        f"<Subject {i}> (appears in [Shot 1] ({actions[m['user_id']]['visibility']})): {actions[m['user_id']]['retention_level']} - {actions[m['user_id']].get('retention') or actions[m['user_id']]['appearance'].rstrip(' .') + '; these reference traits are retained.'}"
        for i, m in enumerate(members, 1)
    ]
    scene_subject = 0
    creature_line = ""
    # ⚠️ The order follows `shared_refs`; never assume "monster right after characters, place always last";
    # that assumption is what pushed numbers and broke continuation (see
    # `_order_shared_references`).
    for asset in shared_refs:
        refs.append(asset["ref"])
        i = len(refs)
        if asset.get("kind") == "scene":
            scene_subject = i
            definitions.append(
                f"<Subject {i}> is the environment shown in <Picture {i}> as a single wide scene view."
            )
            retention.append(
                f"<Subject {i}> (appears in [Shot 1] (wide scene view)): partially_preserved - The referenced environment retains its structure, materials and lighting with framing adapted to the characters."
            )
            pictures.append(
                {
                    "n": i,
                    "id": room["scene_name"],
                    "kind": "scene",
                    "name": room["scene_name"],
                    "ref": room["scene_ref"],
                }
            )
        else:
            definitions.append(
                f"<Subject {i}> is the single nonhuman creature shown in <Picture {i}> as a single full-body view; {plan['monster_appearance']}"
            )
            retention.append(
                f"<Subject {i}> (appears in [Shot 1] (full body)): partially_preserved - The creature retains its anatomy, silhouette, colors and surface details as its stance changes."
            )
            pictures.append(
                {
                    "n": i,
                    "id": monster["subject_id"],
                    "kind": "creature",
                    "name": monster["name"],
                    "ref": monster["ref"],
                }
            )
            creature_line = f"<Subject {i}> is visible in the scene."
    if not 2 <= len(members) <= 4 or len(refs) > 5 or any(not has_media("refs", ref) for ref in refs):
        raise ValueError("joint shot has no valid character references")
    return _StagedReferences(refs, pictures, definitions, retention, scene_subject, creature_line)


def _write_prompt(
    obs: _RoundInputs, room, plan, actions, staged: _StagedReferences, event, duration_decision
):
    """Compile the single timeline and assemble the six-section video prompt."""
    members, head, shared_refs, previous, action_order, video_model = (
        obs.members,
        obs.head,
        obs.shared_refs,
        obs.previous,
        obs.action_order,
        obs.video_model,
    )
    event_text, event_type, event_key = event
    duration = duration_decision.seconds
    refs, pictures, definitions, retention = (
        staged.refs,
        staged.pictures,
        staged.definitions,
        staged.retention,
    )
    scene_subject, creature_line = staged.scene_subject, staged.creature_line
    # The shared style bible is user-facing text and may be Chinese.  The video model receives
    # an English caption, so translate it once at the media boundary while keeping
    # Chinese player actions/source narration intact elsewhere.
    style_text = video_prompt._english_visual_style(room.get("visual_style", ""))
    style = (" The scene's visual treatment is " + style_text.rstrip(" .") + ".") if style_text else ""
    assets = AssetResolver.resolve(members, room, shared_refs)
    scene_snapshot = WorldSnapshot.from_room(
        room,
        node_id=getattr(head, "id", "") if head else "",
        revision=room.get("state_revision", room.get("revision", 0)),
    )
    scene_plan = ScenePlan.from_legacy(
        {
            **plan,
            "actions": plan["actions"],
            "world_event": event_text,
            "world_event_type": event_type,
            "world_event_key": event_key,
            "event_action": plan.get("event_action", ""),
        },
        members,
        base_revision=scene_snapshot.revision,
        duration=duration,
        continuation=bool(previous),
        assets=assets,
        submission_order=action_order,
    )
    plan["timeline"] = ShotCompiler.timeline(scene_plan, caption=True)
    plan["timeline_source"] = "server:ShotCompiler"
    plan["planned_event"] = scene_plan.event.as_dict() if scene_plan.event else None
    environment_anchor = (
        f"The shot is inside <Subject {scene_subject}>. {plan['scene'].rstrip(' .')}."
        if scene_subject
        else plan["scene"].rstrip(" .") + "."
    )
    six = {
        "subject_definitions": "\n".join(definitions),
        "summary": "[reference_generation] "
        + "; ".join(f"<Subject {i}> {actions[m['user_id']]['beat']}" for i, m in enumerate(members, 1)),
        "retention_analysis": "\n".join(retention),
        # Submit one timeline only, so per-person actions and the world preview are not joined
        # into repeated performances.
        "detailed_description": ShotCompiler.compile_timeline(
            scene_plan,
            camera=plan["camera"],
            creature_clause=creature_line,
            environment=environment_anchor,
            style=style,
        ),
        "overall_soundscape": plan["sound"],
        "non_diegetic_music": "N/A",
    }
    # Defensive final pass: plan repair covers scene/sound/event_action; the timeline is
    # compiled in code, but legacy locked definitions and style clauses can still contain
    # Chinese.
    six.update(video_prompt._ensure_prompt_english(six))
    caption_source = copy.deepcopy(six)
    six = _fit_caption(six, scene_plan, pictures, video_model=video_model, previous=previous)
    validate_english_fields(six, allow_dialogue=True)
    _validate_prompt_sections(six, len(refs))
    prompt = prompt_budget.assemble(six)
    # Reference submission remains governed by the model's existing policy.
    send_refs_on_continuation = True
    if prompt_budget.prompt_size(prompt) > prompt_budget.PROMPT_MAX_CHARS:
        raise ValueError("joint shot description is too long; shorten the actions and retry")
    turn_context = TurnContext(
        request_id=str(room.get("request_id", "")),
        snapshot=scene_snapshot,
        assets=assets,
        log_context={"world_id": scene_snapshot.world_id, "round": scene_snapshot.round_number},
    )
    turn_context.final_video_request = {
        "model": video_model,
        "duration": duration,
        "refs": [asset.ref for asset in assets],
        "duration_policy": duration_decision.as_dict(),
        "send_refs_on_continuation": send_refs_on_continuation,
        "continue_from": previous,
        "prompt": prompt,
        "continuation_fallback_prompt": prompt if previous else None,
    }
    reqlog.record(
        "coop_scene_plan_compiled",
        ok=True,
        base_revision=scene_snapshot.revision,
        node_id=scene_snapshot.node_id,
        action_count=len(scene_plan.actions),
        asset_order=[asset.as_dict() for asset in assets],
        subject_order=list(scene_plan.subject_order),
        submission_order=list(scene_plan.submission_order),
        timeline_source=plan["timeline_source"],
        event_source=(scene_plan.event.source if scene_plan.event else "none"),
        duration=duration,
        duration_policy=duration_decision.as_dict(),
        prompt_version=PROMPT_VERSION,
        prompt_chars=len(prompt),
    )
    return {
        "style": style,
        "scene_snapshot": scene_snapshot,
        "scene_plan": scene_plan,
        "environment_anchor": environment_anchor,
        "six": six,
        "caption_source": caption_source,
        "prompt": prompt,
        "send_refs_on_continuation": send_refs_on_continuation,
        "turn_context": turn_context,
    }


def prepare(world, user_text, *, room, mentions, pretranslated=None, events=None):
    """Plan and compile one round's shot without rendering or committing it.

    Runs the first stages of the round loop as traced steps (see
    `murea_agent.coop.loop`): perceive -> plan -> check -> act (stage, write prompt).
    The steps are returned in `prepared['trace']`.
    """
    events = events or Emitter()
    started = time.monotonic()
    with events.step("perceive", "observe", "读房间、上一段画面和共享资产") as step:
        obs = _observe(world, room, mentions)
        members = obs.members
        step.data.update(
            players=[m["character"] for m in members],
            continuation=obs.same_scene,
            references=len(obs.refs),
            boss=obs.boss_life.get("status", ""),
        )
    with events.step("plan", "director", "导演为每位玩家锁定动作，规划环境") as step:
        actions, plan, duration_decision, cached = _plan_actions(obs, room, pretranslated)
        duration = duration_decision.seconds
        step.data.update(
            cached=cached,
            duration=duration,
            actions={m["character"]: actions[m["user_id"]].get("beat", "") for m in members},
        )
    with events.step("check", "world_event", "检查世界事件与 BOSS 预告") as step:
        players, event_text, event_type, event_key = _check_world_event(obs, room, plan)
        step.data.update(world_event=event_text, event_type=event_type)
    with events.step("act", "stage_assets", "排定参考图顺序和角色定义") as step:
        staged = _stage_references(world, obs, room, plan, actions, event_text)
        step.data.update(cast=[p["name"] for p in staged.pictures], references=len(staged.refs))
    with events.step("act", "write_prompt", "编排同一个镜头，写视频 prompt") as step:
        out = _write_prompt(
            obs, room, plan, actions, staged, (event_text, event_type, event_key), duration_decision
        )
        prompt, six, previous = out["prompt"], out["six"], obs.previous
        step.data.update(chars=len(prompt), duration=duration, continuation=bool(previous))
    environment_anchor, scene_plan, scene_snapshot = (
        out["environment_anchor"],
        out["scene_plan"],
        out["scene_snapshot"],
    )
    boss_life, continuation_decision, video_model = obs.boss_life, obs.continuation_decision, obs.video_model
    caption_source, send_refs_on_continuation, turn_context = (
        out["caption_source"],
        out["send_refs_on_continuation"],
        out["turn_context"],
    )
    style, refs, pictures, creature_line = out["style"], staged.refs, staged.pictures, staged.creature_line
    return {
        "base_head": world.head,
        "boss_life": boss_life,
        "asset_edit_revision": room.get("asset_edit_revision", 0),
        "continuation_decision": continuation_decision,
        "caption_source": caption_source,
        "caption_context": {
            "environment": environment_anchor,
            "camera": plan["camera"],
            "creature_clause": creature_line,
            "style": style,
        },
        "duration": duration,
        "duration_policy": duration_decision.as_dict(),
        "previous": previous,
        "video_model": video_model,
        # ``turn_context`` is deliberately omitted when a recommendation prompt
        # is persisted.  Keep the model policy beside the serializable request so
        # consuming a completed prefetch cannot turn a video-model continuation into a
        # zero-reference request.
        "send_refs_on_continuation": send_refs_on_continuation,
        "refs": refs,
        "pictures": pictures,
        "players": players,
        "plan": plan,
        "scene_plan": scene_plan,
        "world_snapshot": scene_snapshot,
        "turn_context": turn_context,
        "six": six,
        # Compatibility aliases: continuation fallback submits the same caption.
        "fallback_six": six,
        "fallback_prompt": prompt,
        "prompt": prompt,
        "merge_seconds": time.monotonic() - started,
        "trace": events.trace(),
    }


def _reference_free_prompt(six, pictures):
    """What a continuation sends when the video model takes no references with it."""
    return prompt_budget.assemble(video_prompt._plain_continuation_from_merged(six, pictures))


def _fit_caption(source, scene_plan, pictures, *, video_model, previous, limit=None):
    """Fit the caption to what the video model reads, as it will actually be sent.

    `limit` defaults to the model's budget (`max_prompt_chars`): Murea-Fast reads 1024 text
    tokens and silently drops the rest, the end of the prompt. A continuation on a model that
    takes no references sends reference-free prose, which is fitted itself; the full prompt is
    then only the fallback and may keep the general ceiling.
    """
    model = video.model(video_model)
    limit = model.max_prompt_chars if limit is None else min(limit, model.max_prompt_chars)
    if not previous or model.references_on_continuation:
        return coop_caption.fit(source, scene_plan, pictures, limit=limit)
    try:
        six = coop_caption.fit(source, scene_plan, pictures, limit=limit)
    except ValueError:
        six = coop_caption.fit(source, scene_plan, pictures)
    return coop_caption.fit_continuation(
        six, scene_plan, pictures, limit=limit, sent=lambda out: _reference_free_prompt(out, pictures)
    )


def _rebuild_prepared_caption(prepared, *, limit=None):
    """Rebuild persisted and retried captions from the complete actor plans."""
    plan = prepared["plan"]
    pictures = prepared["pictures"]
    scene_plan = prepared.get("scene_plan")
    if not isinstance(scene_plan, ScenePlan):
        saved = scene_plan or {}
        scene_plan = ScenePlan.from_legacy(
            plan,
            prepared["players"],
            duration=prepared["duration"],
            base_revision=saved.get("base_revision", 0),
            continuation=bool(prepared.get("previous")),
        )
        prepared["scene_plan"] = scene_plan
    context = prepared.get("caption_context")
    if context is None:
        # Old prepared bundles have all motion facts but no caption context.
        # Do not reuse their potentially truncated or obsolete choreography.
        scene = next((p["n"] for p in pictures if p["kind"] == "scene"), 0)
        old_detail = (prepared.get("six") or {}).get("detailed_description", "")
        style = old_detail.partition("Visual style: ")[2]
        context = {
            "environment": (
                f"The action unfolds inside <Subject {scene}>, with its established layout and lighting."
                if scene
                else plan["scene"].rstrip(" .") + "."
            ),
            "camera": plan.get("camera") or _CAMERA,
            "style": style if style and "[compressed]" not in style else "",
        }
        prepared["caption_context"] = context
    # Rebuild old cached template clauses from their existing structured data.
    context.pop("cast_count", None)
    creature = next((p["n"] for p in pictures if p["kind"] == "creature"), 0)
    context["creature_clause"] = f"<Subject {creature}> is visible in the scene." if creature else ""
    context["camera"] = _CONTINUATION_CAMERA if prepared.get("previous") else _CAMERA
    style = context.get("style", "").strip()
    if style.startswith("Visual style: "):
        style = style.removeprefix("Visual style: ")
    if style and not style.startswith("The scene's visual treatment is "):
        style = "The scene's visual treatment is " + style.rstrip(" .") + "."
    context["style"] = style
    source = copy.deepcopy(prepared.get("caption_source") or prepared.get("six") or {})
    if not prepared.get("caption_source"):
        # Restore whole metadata rows too: a v7 cached body may already have
        # lost the middle of a definition or its retention row.
        source["subject_definitions"] = "\n".join(coop_caption.definition(p) for p in pictures)
        actions = {action.user_id: action for action in scene_plan.actions}
        ordered_actions = [actions[uid] for uid in scene_plan.subject_order]
        source["retention_analysis"] = "\n".join(
            coop_caption.retention(
                p,
                visibility=ordered_actions[p["n"] - 1].visibility,
                level=ordered_actions[p["n"] - 1].retention_level,
                detail=ordered_actions[p["n"] - 1].retention,
            )
            if p["kind"] == "character"
            else coop_caption.retention(p)
            for p in pictures
        )
        source["summary"] = "[reference_generation] " + "; ".join(
            f"<Subject {i}> {action.beat or action.action}" for i, action in enumerate(ordered_actions, 1)
        )
        source["non_diegetic_music"] = "N/A"
    source["overall_soundscape"] = plan.get("sound", "Natural ambient sound.")
    # Replace only the former server-authored environment definition in cached captions.
    source["subject_definitions"] = source.get("subject_definitions", "").replace(
        "is the shared environment shown in", "is the environment shown in"
    )
    source["detailed_description"] = ShotCompiler.compile_timeline(scene_plan, **context)
    prepared["caption_source"] = copy.deepcopy(source)
    six = _fit_caption(
        source,
        scene_plan,
        pictures,
        video_model=prepared.get("video_model") or config.VIDEO_MODEL,
        previous=prepared.get("previous"),
        limit=limit,
    )
    validate_english_fields(six, allow_dialogue=True)
    prompt = prompt_budget.assemble(six)
    prepared.update(six=six, fallback_six=six, prompt=prompt, fallback_prompt=prompt)
    plan["timeline"] = ShotCompiler.timeline(scene_plan, caption=True)
    plan["timeline_source"] = "server:ShotCompiler"
    turn_context = prepared.get("turn_context")
    if turn_context and turn_context.final_video_request is not None:
        turn_context.final_video_request.update(
            prompt=prompt, continuation_fallback_prompt=prompt if prepared.get("previous") else None
        )
    return six


def render_prepared(prepared, *, progress, cancelled, on_video_ready=None, on_status=None, events=None):
    """Render and extract frames, leaving world persistence to the caller.

    One traced `act / render_clip` step; `rendered['trace']` carries the
    planning steps of `prepared` plus this one.
    """
    events = events or Emitter(earlier=prepared.get("trace"))
    # A recommendation may have been prepared before the room was submitted (or
    # persisted by an older process). Rebuild the video caption at this final boundary
    # so stale oversized text can never bypass the current length policy.
    _rebuild_prepared_caption(prepared)
    prompt = prepared["prompt"]
    refs = prepared["refs"]
    duration = prepared["duration"]
    previous = prepared["previous"]
    video_model = prepared.get("video_model") or config.VIDEO_MODEL
    turn_context = prepared.get("turn_context")
    request_policy = getattr(turn_context, "final_video_request", None) or {}
    send_refs_on_continuation = prepared.get("send_refs_on_continuation")
    if send_refs_on_continuation is None:
        send_refs_on_continuation = request_policy.get("send_refs_on_continuation")
    if send_refs_on_continuation is None:
        # Backward compatibility for recommendation prompts persisted before the
        # policy became a top-level field: they sent the images with a continuation.
        send_refs_on_continuation = True
    send_refs_on_continuation = bool(send_refs_on_continuation)
    # Murea continues from the previous clip's frames alone: it gets the shot as
    # reference-free prose, and the full prompt stays the fallback.
    reference_free = bool(previous) and not video.model(video_model).references_on_continuation

    def sent_prompt(full):
        if not reference_free:
            return full
        return _reference_free_prompt(prepared["six"], prepared["pictures"])

    full_prompt = prompt
    prompt = sent_prompt(full_prompt)
    token = getattr(prepared.get("turn_context"), "cancellation", None)

    def is_cancelled():
        return bool(cancelled() or (token and token.cancelled()))

    if is_cancelled():
        raise RuntimeError("joint generation was cancelled")
    with events.step("act", "render_clip", "提交一次联合视频生成并抽帧") as step:
        step.data.update(duration=duration, references=len(refs), continues=bool(previous))
        progress("video", msg("联合镜头已就绪，提交一次 Ref2VA 生成"))
        last_status = None

        def status(job):
            nonlocal last_status
            state = (job.get("status"), bool(job.get("started_at")))
            if state != last_status:
                reqlog.record(
                    "video_status",
                    response={
                        k: job[k]
                        for k in (
                            "id",
                            "status",
                            "started_at",
                            "completed_at",
                            "seconds",
                            "error",
                            "inference_time_s",
                        )
                        if k in job
                    },
                )
                last_status = state
            if on_status:
                on_status(job)
            progress(
                "video",
                msg("正在渲染共同画面")
                if job.get("status") == "running" or job.get("started_at")
                else "视频任务已提交，等待模型处理",
            )

        with reqlog.scope(stage="generation"):
            reqlog.record(
                "video_request",
                prompt=prompt,
                refs=prepared["pictures"],
                model=video_model,
                duration=duration,
                continue_from=previous,
                send_refs_on_continuation=send_refs_on_continuation,
                require_continuation=bool(previous),
            )
            try:
                gen = video.generate(
                    prompt,
                    refs=[media_path("refs", ref) for ref in refs],
                    duration=duration,
                    model_id=video_model,
                    continue_from=previous,
                    send_refs_on_continuation=send_refs_on_continuation,
                    require_continuation=bool(previous),
                    allow_continuation_fallback=bool(previous),
                    continuation_fallback_prompt=(prepared.get("fallback_prompt") if previous else None),
                    cancelled=is_cancelled,
                    on_status=status,
                    on_queue=lambda: progress("video", msg("双方动作已就绪，正在等待视频生成槽")),
                )
            except video.PromptTooLongError:
                # The video service can accept less than its stated limit. Retry once by removing
                # whole optional metadata units; never cut an actor's motion.
                compact_limit = min(5000, len(full_prompt) - 1)
                _rebuild_prepared_caption(prepared, limit=compact_limit)
                retry_prompt = prepared["prompt"]
                if retry_prompt == full_prompt or len(retry_prompt) > compact_limit:
                    raise
                prompt = sent_prompt(retry_prompt)
                reqlog.record(
                    "video_prompt_compacted_retry", ok=True, prompt_chars=len(prompt), limit=compact_limit
                )
                gen = video.generate(
                    prompt,
                    refs=[media_path("refs", ref) for ref in refs],
                    duration=duration,
                    model_id=video_model,
                    continue_from=previous,
                    send_refs_on_continuation=send_refs_on_continuation,
                    require_continuation=bool(previous),
                    allow_continuation_fallback=bool(previous),
                    continuation_fallback_prompt=(retry_prompt if previous else None),
                    cancelled=is_cancelled,
                    on_status=status,
                    on_queue=lambda: progress("video", msg("双方动作已就绪，正在等待视频生成槽")),
                )
            reqlog.record(
                "video_result",
                ok=True,
                video_id=gen.video_id,
                clip=gen.video_path.name,
                seconds=gen.elapsed,
                params=gen.params,
                artifact_id=gen.artifact_id,
            )
        # The video model may drift toward an uploaded white-background reference near the tail.
        # Remove only that invalid plate before exposing the clip to the client.
        clean_duration = coop_media.remove_white_reference_tail(gen.video_path, gen.duration)
        if clean_duration < gen.duration:
            gen.duration = clean_duration
            gen.params["white_reference_tail_trimmed"] = True
        if on_video_ready:
            on_video_ready(gen.video_path.name)
        node_id = "n" + uuid.uuid4().hex[:8]
        frame_started = time.monotonic()
        frames = coop_media.grab_frames(gen.video_path, node_id, gen.duration)
        frame_seconds = time.monotonic() - frame_started
        step.data.update(
            clip=gen.video_path.name,
            continued=bool(gen.continued),
            fallback=bool(gen.params.get("continuation_fallback")),
        )
    used_fallback = bool(gen.params.get("continuation_fallback"))
    submitted_prompt = gen.submitted_prompt or (prepared.get("fallback_prompt") if used_fallback else prompt)
    submitted_six = prepared.get("fallback_six") if used_fallback else prepared["six"]
    return {
        "node_id": node_id,
        "generation": gen,
        "frames": frames,
        "submitted_prompt": submitted_prompt,
        "submitted_six": submitted_six,
        "used_prompt_fallback": used_fallback,
        "frame_seconds": frame_seconds,
        "trace": events.trace(),
    }


def commit_prepared(world, user_text, *, room, mentions, prepared, rendered, origin, events=None):
    """Atomically turn an accepted rendered draft into the next world node.

    The node keeps the round's whole trace: planning, rendering and this
    `commit / update_world` step.
    """
    events = events or Emitter(earlier=rendered.get("trace") or prepared.get("trace"))
    with events.step("commit", "update_world", "把这一轮写进世界") as step:
        node = _commit(
            world,
            user_text,
            room=room,
            mentions=mentions,
            prepared=prepared,
            rendered=rendered,
            origin=origin,
        )
        step.data.update(node=node.id, players=[m["character"] for m in room["members"]])
    node.trace = events.trace()
    world.nodes[node.id]["trace"] = node.trace
    world.save()
    return node


def _commit(world, user_text, *, room, mentions, prepared, rendered, origin):
    if world.head != prepared.get("base_head"):
        raise ValueError("the speculative clip's parent node changed")
    snapshot = prepared.get("world_snapshot")
    snapshot_revision = (
        snapshot.revision
        if hasattr(snapshot, "revision")
        else snapshot.get("revision")
        if isinstance(snapshot, dict)
        else None
    )
    if snapshot_revision is not None and int(room.get("state_revision", room.get("revision", 0))) != int(
        snapshot_revision
    ):
        raise ValueError("the room changed during the joint shot; discarded the stale result")
    gen = rendered["generation"]
    plan = copy.deepcopy(prepared["plan"])
    if rendered.get("used_prompt_fallback") or bool(
        (getattr(rendered.get("generation"), "params", None) or {}).get("continuation_fallback")
    ):
        plan["timeline"] = (rendered.get("submitted_six") or {}).get(
            "detailed_description", plan.get("timeline", "")
        )
        plan["timeline_source"] = "server:ShotCompiler:continuation_fallback"
        plan.pop("director_plan", None)
    refs = prepared["refs"]
    pictures = prepared["pictures"]
    players = prepared["players"]
    six = rendered.get("submitted_six") or prepared["six"]
    submitted_prompt = rendered.get("submitted_prompt") or prepared["prompt"]
    previous = prepared["previous"]
    monster = room.get("monster", {})
    node_id = rendered["node_id"]
    worker_queue = max(0, time.time() - room.get("queued_at", time.time()))
    node = Node(
        id=node_id,
        parent=world.head,
        video_id=gen.video_id,
        clip=gen.video_path.name,
        place=room.get("scene_name") or world.here or room["title"],
        beat=plan["reply"],
        mode="continue" if gen.continued else "cut",
        subjects=mentions + ([monster["subject_id"]] if monster.get("ref") in refs else []),
        refs=refs,
        six_section=six,
        prompt=submitted_prompt,
        reply=plan["reply"],
        user_text=user_text,
        directive={
            "coop": True,
            "prompt_version": PROMPT_VERSION,
            "mode": room.get("mode", "online"),
            "asset_edit_revision": prepared.get("asset_edit_revision", 0),
            "continuation_decision": copy.deepcopy(prepared.get("continuation_decision") or {}),
            "boss_life": copy.deepcopy(prepared.get("boss_life") or {}),
            "players": players,
            "shot": plan,
            "monster": monster,
            "duration_policy": copy.deepcopy(prepared.get("duration_policy") or {}),
            "reference_subject_definitions": prepared.get("fallback_six", {}).get("subject_definitions", ""),
            "scene_plan": prepared.get("scene_plan").as_dict()
            if hasattr(prepared.get("scene_plan"), "as_dict")
            else prepared.get("scene_plan"),
            "world_snapshot": prepared.get("world_snapshot").as_dict()
            if hasattr(prepared.get("world_snapshot"), "as_dict")
            else dict(prepared.get("world_snapshot") or {}),
        },
        pictures=pictures,
        model_params=gen.params,
        artifact_id=gen.artifact_id,
        continued_from=previous if gen.continued else None,
        origin=origin,
        elapsed=prepared["merge_seconds"] + gen.elapsed + rendered["frame_seconds"],
        timings={
            "merge": prepared["merge_seconds"],
            "video": gen.elapsed,
            "frames": rendered["frame_seconds"],
            "worker_queue": worker_queue,
        },
        frames=rendered["frames"],
    )
    world.add_node(node)
    world.history.extend([{"role": "user", "text": user_text}, {"role": "model", "text": node.reply}])
    world.save()
    reqlog.record(
        "coop_done",
        stage="generation",
        node=node_id,
        video=gen.video_id,
        refs=len(refs),
        players=[m["user_id"] for m in room["members"]],
        prompt=submitted_prompt,
        seconds=node.elapsed,
    )
    return node
