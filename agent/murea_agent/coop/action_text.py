"""Validating and repairing the English texts of the joint shot: actions, beats, metadata and narration."""

from __future__ import annotations
import json
import re

from murea_agent import config
from murea_agent.director import agent_schema
from murea_agent.coop import (
    trace as coop_trace,
)
from murea_agent import reqlog
from murea_agent.director.prompt_language import contains_non_latin_text
from murea_agent.util import cap
from murea_agent.coop.generation_prompts import (
    _ACTION_RECOVERY_SYSTEM,
    _ACTION_REPAIR_SYSTEM,
    _ACTION_SCHEMA,
    _PLAN_REPAIR_SYSTEM,
    _REFERENCE_LAYOUT,
    _SERVER_TIMELINE,
    _SHEET_LAYOUT_PHRASES,
)


def _unsheet_definitions(lines):
    """Replace wording that describes a "reference sheet" with the reference image's real form."""
    out = []
    for line in lines:
        for phrase in _SHEET_LAYOUT_PHRASES:
            if phrase in line:
                line = line.replace(phrase, _REFERENCE_LAYOUT)
        out.append(line)
    return out


_REFERENCE_TOKEN = re.compile(r"<\s*(?:Subject|Picture)\s*\d+\s*>", re.I)


_RAW_BOUND_REFERENCE = re.compile(r"(?:<\s*)?\b(?:Subject|Picture)\s*\d+\b(?:\s*>)?", re.I)


def _locked_reference_definitions(world, head, count):
    """Find the last reference-bound cast definitions across latent children."""
    current = head
    while current:
        directive = (
            current.directive
            if hasattr(current, "directive")
            else current.get("directive", {})
            if isinstance(current, dict)
            else {}
        )
        stored = directive.get("reference_subject_definitions", "")
        if not stored:
            six = (
                current.six_section
                if hasattr(current, "six_section")
                else current.get("six_section", {})
                if isinstance(current, dict)
                else {}
            )
            stored = six.get("subject_definitions", "")
        lines = str(stored or "").splitlines()
        if len(lines) >= count:
            return lines[:count]
        parent = (
            current.parent
            if hasattr(current, "parent")
            else current.get("parent")
            if isinstance(current, dict)
            else None
        )
        current = world.nodes.get(parent) if parent else None
    return []


def _validate_prompt_sections(six, subject_count):
    """Check the shared video caption contract without injecting visual rules."""
    if any(not str(six.get(field, "")).strip() for field in agent_schema.SECTION_ORDER):
        raise ValueError("joint shot is missing the six video prompt fields")
    if not six["summary"].startswith("[reference_generation]"):
        raise ValueError("joint shot summary is missing [reference_generation]")
    if not six["detailed_description"].startswith("[Shot 1]"):
        raise ValueError("joint shot detailed description is missing [Shot 1]")
    expected = set(range(1, subject_count + 1))
    for field in ("subject_definitions", "retention_analysis", "detailed_description"):
        found = {int(n) for n in re.findall(r"<Subject (\d+)>", six[field])}
        if found != expected:
            raise ValueError(f"joint shot {field} reference subjects do not match")
    pictures = {int(n) for n in re.findall(r"<Picture (\d+)>", six["subject_definitions"])}
    if pictures != expected:
        raise ValueError("joint shot reference image numbers do not match")


def _text(value, field, limit, *, subject_count=0):
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError("joint shot " + field + " is malformed")
    unbound = value
    for i in range(1, subject_count + 1):
        unbound = unbound.replace(f"<Subject {i}>", "")
    if re.search(r"<(?:Subject|Picture)\s", unbound, re.I) or _REFERENCE_TOKEN.search(unbound):
        allowed = ", ".join(f"<Subject {i}>" for i in range(1, subject_count + 1))
        raise ValueError(
            f"联合镜头 {field} 引用无效；"
            + (
                f"only {allowed} allowed, no Picture or other numbers"
                if allowed
                else "this field is plain English without Subject/Picture tags"
            )
        )
    return value.strip()


def _validate(plan, members, *, subject_count=0):
    if not isinstance(plan, dict) or not isinstance(plan.get("actions"), list):
        raise ValueError("joint shot is missing the characters' actions")
    expected = {m["user_id"] for m in members}
    actions = {}
    for entry in plan["actions"]:
        if not isinstance(entry, dict) or entry.get("user_id") not in expected or entry["user_id"] in actions:
            raise ValueError("joint shot characters do not match the room")
        actions[entry["user_id"]] = {
            k: _text(entry.get(k), k, 900, subject_count=subject_count if k == "action" else 0)
            for k in ("appearance", "action")
        }
    if set(actions) != expected:
        raise ValueError("joint shot left out a teammate's action; generate again")
    return actions


def _normalize_action_ids(plan, members):
    """Recover a model-truncated ID only when its verbatim source action is unique."""
    if not isinstance(plan, dict) or not isinstance(plan.get("actions"), list):
        return
    expected = {m["user_id"] for m in members}
    reserved = {
        entry.get("user_id")
        for entry in plan["actions"]
        if isinstance(entry, dict) and entry.get("user_id") in expected
    }
    by_source = {}
    for member in members:
        by_source.setdefault(member["action"], []).append(member["user_id"])
    for entry in plan["actions"]:
        if not isinstance(entry, dict) or entry.get("user_id") in expected:
            continue
        candidates = [uid for uid in by_source.get(entry.get("source_action"), []) if uid not in reserved]
        if len(candidates) == 1:
            entry["user_id"] = candidates[0]
            reserved.add(candidates[0])
            reqlog.record("action_user_id_recovered", ok=True, user_id=candidates[0])


def _english(value, field, limit=1600, *, subject_count=0):
    value = _text(value, field, limit, subject_count=subject_count)
    if contains_non_latin_text(value):
        raise ValueError(
            "joint shot " + field + " must be translated to English, not replaced by older Chinese narration"
        )
    return value


def _plain_beat(value):
    """Replace Subject/Picture numbers in `beat` with natural language instead of failing the whole beat.

    The server prefixes `beat` with `<Subject i>` before putting it into summary, so `beat` itself
    should carry no numbers and `_text()` rejects them. But the model still sometimes writes:

        follows closely behind <Subject 1>
        stands beside <Subject 1> and holds a defensive posture
        crouches low, scans the rock wall, and signals <Subject 1>

    Failing the beat is too costly (one retry took 100 seconds) and the meaning is clear. Deleting
    the number would leave a broken sentence ("follows closely behind"), so it becomes a
    natural-language reference, worded like `video_prompt._plain_continuation_from_merged`: the same
    problem should look the same on both paths.

    Only `beat` changes. `action` may carry bound numbers (the timeline uses them to find targets),
    and `appearance` still rejects them.
    """
    if not isinstance(value, str) or not value:
        return value
    out = _REFERENCE_TOKEN.sub("the other character", value)
    if out == value:
        return value
    out = re.sub(r"\s+", " ", out).strip()
    reqlog.record("coop_beat_token_replaced", ok=True, before=value[:120], after=out[:120])
    return out


def _optional_english(value, field, limit=1600, *, subject_count=0):
    """Validate an optional compiled field without making legacy plans fail."""
    if value is None or not str(value).strip():
        return ""
    return _english(value, field, limit, subject_count=subject_count)


def _validate_model_timeline(value, subject_count):
    """Validate legacy timeline structure, but never use it as choreography."""
    if value == _SERVER_TIMELINE:
        return False
    if isinstance(value, str) and contains_non_latin_text(value):
        # The compatibility field can be repaired below. Its repaired value is
        # still discarded because ShotCompiler owns the final timeline.
        return True
    timeline = _english(value, "timeline", _plan_field_limit("timeline"), subject_count=subject_count)
    # ⚠️ **A malformed structure is only logged, never rejects the round.**
    #
    # This value is **discarded anyway** (docstring: never use it as choreography; the caller then
    # overwrites it with `plan['timeline'] = compiled_timeline`). Raising for text about to be
    # overwritten makes the user wait a round for an error about a field that does not affect the
    # clip.
    #
    # What matters is guarded elsewhere: each action passes `_english`, missing people are filled by
    # `_recover_missing_action_entries`, ids are normalized by `_normalize_action_ids`. Another
    # check here adds no coverage and has nothing to repair.
    problems = []
    if len(timeline.split()) < max(8, subject_count * 4):
        problems.append("too_short")
    if any(f"<Subject {i}>" not in timeline for i in range(1, subject_count + 1)):
        problems.append("missing_subject")
    reqlog.record(
        "legacy_model_timeline_ignored",
        ok=not problems,
        subject_count=subject_count,
        chars=len(timeline),
        problems=problems,
    )
    return False


_ACTION_REPAIR_FIELDS = {
    "appearance": "Concise English preserving individual facial, hair, garment and equipment details visible in the reference.",
    "retention": "A concise English declarative sentence describing retained visible traits, equipment ownership and differences in this shot.",
    "action": (
        "30-70 English words describing faithful visible physical staging, as a "
        "present-tense predicate starting with a verb and no subject of its own."
    ),
    "beat": "6-16 English words as a present-tense predicate without a subject pronoun.",
    "start_state": "An explicitly stated initial pose/location, phrased after is; empty when absent.",
    "target": "A short English target or spatial anchor, or an empty string when absent.",
    "path": (
        "A concise English route phrase with ordered waypoints, phrased after follows "
        "(a left arc around the pillar); preserve direction, without repeating the whole action."
    ),
    "end_state": (
        "A short English visible completion boundary or final spatial state, "
        "phrased after is, without inventing an outcome; empty when absent."
    ),
}


def _repair_english_fields_batch(requests, *, subject_count=0, metadata_aliases=None):
    """Repair the Chinese-contaminated fields of everyone in one call instead of one call per person.

    ⚠️ This repair used to run in a `for m in members:` loop, **one call per player**, on the critical path
    (the round waits for it before submitting the video). Measured, the repair ran 74 times
    against 28 translations, about 2.6 per round; a four-player room meant 4 sequential ~1 s calls.

    Models writing Chinese here is normal, so this path runs almost every round. Merged, it is one
    call regardless of party size.

    Fields are the union over everyone: a few more fields are much cheaper than another call; when
    filling back, each person only takes the fields that were actually contaminated, the rest stay
    as they were.
    """
    metadata_aliases = metadata_aliases or {}
    pending = [
        (member, {f: fields[f] for f in fields if f in _ACTION_REPAIR_FIELDS}) for member, fields in requests
    ]
    pending = [(member, fields) for member, fields in pending if fields]
    if not pending:
        return {}
    union = [field for field in _ACTION_REPAIR_FIELDS if any(field in fields for _, fields in pending)]
    schema = {
        "type": "object",
        "properties": {
            "players": {
                "type": "array",
                "minItems": len(pending),
                "maxItems": len(pending),
                "items": {
                    "type": "object",
                    "properties": {
                        "user_id": {"type": "string"},
                        **{
                            field: {"type": "string", "description": _ACTION_REPAIR_FIELDS[field]}
                            for field in union
                        },
                    },
                    "required": ["user_id"] + union,
                },
            }
        },
        "required": ["players"],
    }
    base_payload = {
        "players": [
            {
                "user_id": member["user_id"],
                "character_role": "the character",
                "source_action": member["action"],
                "rejected_fields": fields,
            }
            for member, fields in pending
        ],
        "required_fields": union,
        "allowed_subject_tokens": (
            [f"<Subject {i}>" for i in range(1, subject_count + 1)] if subject_count else []
        ),
    }
    last_exc = None
    for attempt in range(3):
        payload = dict(base_payload)
        if attempt:
            payload["correction_required"] = (
                "Previous repair contained non-English text. Rewrite every returned field "
                "with ASCII English only; do not include names from the source."
            )
        try:
            repaired = coop_trace.chat(
                system=_ACTION_REPAIR_SYSTEM,
                user=json.dumps(payload, ensure_ascii=False),
                schema=schema,
                model=config.GM_MODEL,
                thinking=0,
                temperature=0,
                tag="coop_action_translation_repair",
            )
            rows = (repaired or {}).get("players") if isinstance(repaired, dict) else None
            if not isinstance(rows, list):
                raise ValueError("joint shot repair result is malformed")
            by_id = {row.get("user_id"): row for row in rows if isinstance(row, dict)}
            result = {}
            for member, fields in pending:
                row = by_id.get(member["user_id"])
                if row is None:
                    raise ValueError("joint shot repair result left out a teammate")
                # Fill back only the fields this person actually had contaminated; leave the others
                # untouched.
                result[member["user_id"]] = {
                    field: (
                        _normalize_action_metadata(
                            row.get(field),
                            field,
                            metadata_aliases.get(member["user_id"], {}),
                            actor_user_id=member["user_id"],
                        )
                        if field in ("start_state", "target", "path", "end_state")
                        else _english(
                            row.get(field), field, subject_count=subject_count if field == "action" else 0
                        )
                    )
                    for field in fields
                }
            reqlog.record(
                "action_translation_repaired",
                ok=True,
                user_ids=[m["user_id"] for m, _ in pending],
                fields=union,
                repaired=result,
            )
            return result
        except ValueError as exc:
            last_exc = exc
            reqlog.record(
                "action_translation_repair_rejected",
                ok=False,
                user_ids=[m["user_id"] for m, _ in pending],
                fields=union,
                error=str(exc),
            )
    raise last_exc


# A field must use the same length limit in "validate" and "repair". Overlong `_text` **raises**
# instead of truncating; if they differ, a valid repaired result is rejected by the repair itself,
# retries burn out and the whole round fails. Only values different from the default are listed; the
# rest use `_english`'s default of 1600.
_PLAN_FIELD_LIMITS = {"timeline": 4000}


def _plan_field_limit(field):
    return _PLAN_FIELD_LIMITS.get(field, 1600)


def _repairable_plan_fields(fields):
    """Include the legacy timeline so Chinese output can be repaired once."""
    return list(fields) + ["timeline"]


def _repair_plan_english_fields(plan, rejected_fields, *, context=None, subject_count=0):
    """Repair contaminated shared environment fields before strict video prompt validation."""
    descriptions = {
        "scene": "Concise English describing the current visible layout, boundaries and relative positions.",
        "sound": "10-30 English words: The audible soundscape consists entirely of ...; complete the sentence with this shot's environmental and physical action sounds.",
        "timeline": "90-170 English words describing one continuous shot with every bound subject.",
        "event_action": "8-35 English words describing only the single permitted forecast behavior.",
        "monster_appearance": "15-40 English words describing the shared creature stable visible traits.",
    }
    fields = [field for field in rejected_fields if field in descriptions]
    if not fields:
        return {}
    schema = {
        "type": "object",
        "properties": {field: {"type": "string", "description": descriptions[field]} for field in fields},
        "required": fields,
    }
    payload = {
        "setting": (context or {}).get("setting", ""),
        "shared_scene": (context or {}).get("shared_scene", {}),
        "shared_creature": (context or {}).get("shared_creature", {}),
        "locked_player_actions": (context or {}).get("locked_actions", []),
        "rejected_fields": {field: rejected_fields[field] for field in fields},
        "required_fields": fields,
        "allowed_subject_tokens": [f"<Subject {i}>" for i in range(1, subject_count + 1)],
    }
    last_exc = None
    for attempt in range(3):
        request = dict(payload)
        if attempt:
            request["correction_required"] = (
                "Previous repair contained non-English text. Rewrite every returned field "
                "with ASCII English only; preserve all allowed Subject tokens in event_action."
            )
        try:
            repaired = coop_trace.chat(
                system=_PLAN_REPAIR_SYSTEM,
                user=json.dumps(request, ensure_ascii=False),
                schema=schema,
                model=config.GM_MODEL,
                thinking=0,
                temperature=0,
                tag="coop_plan_translation_repair",
            )
            result = {}
            for field in fields:
                # ⚠️ The limit must match the caller: the timeline uses 4000, this used the default 1600, so a valid long
                # timeline was rejected by its own repair.
                result[field] = _english(
                    repaired.get(field) if isinstance(repaired, dict) else None,
                    field,
                    _plan_field_limit(field),
                    subject_count=subject_count if field in ("event_action", "timeline") else 0,
                )
            reqlog.record("plan_translation_repaired", ok=True, fields=fields, repaired=result)
            return result
        except ValueError as exc:
            last_exc = exc
            reqlog.record("plan_translation_repair_rejected", ok=False, fields=fields, error=str(exc))
    raise last_exc


def _recover_missing_action_entries(members, *, images=(), schema=None):
    """Ask for omitted players only; identical source actions must still stay separate."""
    recovery_schema = json.loads(json.dumps(schema or _ACTION_SCHEMA))
    # The normal planning schema also contains scene/sound/timeline. Recovery only
    # fills action entries; asking for those unrelated fields encourages truncation.
    recovery_schema["properties"] = {"actions": recovery_schema["properties"]["actions"]}
    recovery_schema["required"] = ["actions"]
    payload = {
        "players": [
            {k: member[k] for k in ("user_id", "character", "description", "action")} for member in members
        ],
        "required_user_ids": [member["user_id"] for member in members],
        "instruction": "Return one complete actions entry per required_user_id, including duplicate-looking actions.",
    }
    recovered = coop_trace.chat(
        system=_ACTION_RECOVERY_SYSTEM,
        user=json.dumps(payload, ensure_ascii=False),
        schema=recovery_schema,
        model=config.GM_MODEL,
        thinking=0,
        temperature=0,
        tag="coop_action_entry_recovery",
        images=images,
    )
    if not isinstance(recovered, dict) or not isinstance(recovered.get("actions"), list):
        raise ValueError("joint shot left out a teammate's action and the repair returned none")
    expected = {member["user_id"] for member in members}
    entries = {}
    for entry in recovered["actions"]:
        if not isinstance(entry, dict) or entry.get("user_id") not in expected:
            raise ValueError("joint shot characters do not match the room")
        uid = entry["user_id"]
        if uid in entries:
            raise ValueError("joint shot repair duplicated a teammate's action")
        member = next(member for member in members if member["user_id"] == uid)
        if entry.get("source_action") != member["action"]:
            raise ValueError("the action translation dropped the player's original text")
        entries[uid] = entry
    if set(entries) != expected:
        raise ValueError("joint shot repair still leaves out a teammate's action")
    return entries


_CLAUSE_END = re.compile(r"[，,；;、：:]")


def _clause_cap(text, zh_limit):
    """`cap`, but a cut ends at the last clause boundary instead of mid-phrase.

    The narration is read aloud and kept as the round's text: "…pulls open three drawers one
    after." read as a mistake. Without a usable boundary the cut is marked with "…".
    """
    full = " ".join(str(text or "").split())
    short = cap(full, zh_limit)
    if short == full:
        return full
    ends = [m.start() for m in _CLAUSE_END.finditer(short)]
    if ends and ends[-1] >= len(short) // 3:
        return short[: ends[-1]].rstrip()
    return short + "…"


def _generation_narration(members, event_text):
    """Build waiting-room TTS from locked inputs without creating new facts."""
    event = str(event_text or "").strip().rstrip("。！？!?.")
    if config.LANGUAGE != "zh":

        def sentence(text, limit):
            part = _clause_cap(text, limit).rstrip(".")
            return part if part.endswith("…") else part + "."

        # English actions are already full sentences that start with the character's name.
        sentences = [
            sentence(str(m.get("action") or ""), 70) for m in members if str(m.get("action") or "").strip()
        ]
        narration = " ".join(sentences) or "This round is being filmed."
        if event:
            narration += " " + sentence(event, 60)
        return narration if len(narration) <= 600 else sentence(narration, 240)
    parts = []
    for member in members:
        # "沈舟 · 雾港侦探" is spoken as "沈舟"; actions usually start with it already.
        name = str(member.get("character") or "角色").split("·")[0].strip()[:30]
        action = _clause_cap(str(member.get("action") or "").strip().rstrip("。！？!?"), 100)
        if action:
            parts.append(action if action.startswith(name) else f"{name}{action}")
    narration = ("本轮行动：" + "；".join(parts) + "。") if parts else "本轮行动正在生成。"
    if event:
        # Forecasts open with their own connective; the narration adds 与此同时 itself.
        event = re.sub(r"^(?:接下来|与此同时|同时|随后|紧接着)[，,、\s]*", "", event)
        narration += f"与此同时，{_clause_cap(event, 120)}。"
    return narration if len(narration) <= 300 else _clause_cap(narration, 300).rstrip("。") + "。"


def _prompt_text(value, field, limit):
    """Validate optional user/model text before inserting it into the video model."""
    if not isinstance(value, str) or len(value) > limit:
        raise ValueError("joint shot " + field + " is malformed")
    value = value.strip()
    if _REFERENCE_TOKEN.search(value):
        raise ValueError("joint shot must not renumber characters")
    return value


def _action_metadata_aliases(members, shared_refs, actor_user_id):
    """Map reference-bound subjects to plain-English action metadata labels."""
    aliases = {
        index: ("the actor" if member["user_id"] == actor_user_id else "the teammate")
        for index, member in enumerate(members, 1)
    }
    for index, asset in enumerate(shared_refs or [], len(members) + 1):
        aliases[index] = "the shared environment" if asset.get("kind") == "scene" else "the shared creature"
    return aliases


def _normalize_action_metadata(value, field, aliases, limit=500, *, actor_user_id=""):
    """Turn bound subjects in optional metadata into plain-English descriptions.

    target/path/end_state guide composition but do not own the player's action. A model
    may repeat a valid binding there even though the schema requests plain English. Keep
    that useful meaning without allowing an auxiliary formatting mistake to veto a shot.
    Unknown subjects and all Picture references are discarded with the auxiliary field;
    the authoritative action field continues to use strict reference validation.
    """
    if value is None or not str(value).strip():
        return ""
    if not isinstance(value, str):
        return _english(value, field, limit)

    invalid = None
    for match in _RAW_BOUND_REFERENCE.finditer(value):
        parsed = re.search(r"\b(Subject|Picture)\s*(\d+)\b", match.group(0), re.I)
        if not parsed:
            continue
        kind, raw_number = parsed.groups()
        number = int(raw_number)
        if kind.casefold() == "picture" or number not in aliases:
            invalid = match.group(0)
            break
    if invalid is not None:
        reqlog.record(
            "action_metadata_reference_dropped",
            ok=True,
            user_id=actor_user_id,
            field=field,
            reference=invalid,
            reason=("picture" if re.search(r"Picture", invalid, re.I) else "unknown_subject"),
        )
        return ""

    def replace(match):
        parsed = re.search(r"\bSubject\s*(\d+)\b", match.group(0), re.I)
        return aliases[int(parsed.group(1))] if parsed else match.group(0)

    return _english(_RAW_BOUND_REFERENCE.sub(replace, value), field, limit)
