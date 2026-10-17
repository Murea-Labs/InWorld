"""Checking forecasts, suggestions and picture observations returned by the co-op planner, and recovering ids the model miswrote."""

import re


from murea_agent.coop import director as coop_director
from murea_agent import reqlog
from murea_agent.coop.messages import msg


_EVENT_TYPES = ("none", "environment", "boss_introduction", "boss_action")


_EVENT_OUTCOMES = ("observed", "partially_observed", "not_observed", "unknown")


_STORY_PHASES = (
    "opening",
    "exploration",
    "threat",
    "boss_reveal",
    "confrontation",
    "escalation",
    "resolution",
    "aftermath",
)


def _text(*args, **kwargs):
    # room imports this module indirectly; import it when first needed.
    from murea_agent.coop import room

    return room._text(*args, **kwargs)


class _GuidanceValidationError(ValueError):
    """Safe UI category; the chained exception retains the diagnostic detail."""

    def __init__(self, label, cause):
        self.label = "规划重复" if isinstance(cause, coop_director.CompletedPlanError) else label
        super().__init__(f"{self.label}: {getattr(cause, 'detail', cause)}")


def _guidance_error(exc):
    if isinstance(exc, _GuidanceValidationError):
        return msg("{label}，可以重试或先写下动作。", label=exc.label)
    from murea_agent.providers.llm import GeminiError
    from httpx import HTTPError

    if isinstance(exc, (GeminiError, HTTPError)):
        return msg("规划接口不可用或请求失败，可以重试或先写下动作。")
    return msg("行动建议生成异常，可以重试或先写下动作。")


def _clean_optional(value, limit=500):
    if not isinstance(value, str):
        return ""
    return value.strip()[:limit]


_BOSS_STATE_NEGATIVE = re.compile(
    r"(?:未|没|没有|不见|看不到|看不见|无法确认|无法判断|不明确).{0,12}"
    r"(?:BOSS|boss|怪兽|巨龙|恶龙|魔王|首领|怪物|敌人|生物)|"
    r"(?:BOSS|boss|怪兽|巨龙|恶龙|魔王|首领|怪物|敌人|生物).{0,12}"
    r"(?:未|没|没有|不在|不见|看不到|看不见|无法确认|无法判断|不明确)|"
    r"\b(?:no|not visible|not seen|cannot confirm|unable to confirm|absent|off[- ]screen)\b",
    re.IGNORECASE,
)


# Models often fill boss_state with enum-like phrases instead of prose: not_present / no_change /
# none / unknown. The regex above was written for Chinese prose and missed them all, so
# `boss_visible=False, boss_state="not_present"` was read as "stated present" and the whole
# observation was rejected. `not_present` literally means absent.
# "Negative" = explicitly says the boss is not in the picture, contradicting boss_visible=True.
_BOSS_STATE_DENIAL = {
    "not present",
    "none",
    "no boss",
    "absent",
    "not visible",
    "unseen",
    "off screen",
    "未出现",
    "不在场",
    "无",
}


# "Neutral" = no information, neither present nor absent. Valid when the boss is visible
# ("no change since the last beat"), and not a presence claim when it is not.
_BOSS_STATE_NEUTRAL = {"no change", "unknown", "n/a", "na", "未知", "没有变化"}


def _normalized_boss_state(value):
    text = _clean_optional(value)
    if not text:
        return ""
    return " ".join(text.replace("_", " ").replace("-", " ").split()).casefold()


def _denies_boss_presence(value):
    """Explicitly denies the boss is in the picture; only this contradicts boss_visible=True."""
    normalized = _normalized_boss_state(value)
    if not normalized:
        return False
    # Neutral first: `\bno\b` in the prose regex would read "no change" as a denial.
    if normalized in _BOSS_STATE_NEUTRAL:
        return False
    return normalized in _BOSS_STATE_DENIAL or bool(_BOSS_STATE_NEGATIVE.search(normalized))


def _positive_boss_state(value):
    text = _clean_optional(value)
    if not text:
        return False
    normalized = _normalized_boss_state(text)
    if normalized in _BOSS_STATE_NEUTRAL or normalized in _BOSS_STATE_DENIAL:
        return False
    return not _BOSS_STATE_NEGATIVE.search(text)


def _has_known_boss(state):
    state = state or {}
    if state.get("boss_known") is True:
        return True
    return _positive_boss_state(state.get("boss"))


def _has_shared_creature(context):
    creature = (context or {}).get("shared_creature") or {}
    return isinstance(creature, dict) and bool(
        str(creature.get("name") or "").strip()
        or str(creature.get("description") or "").strip()
        or str(creature.get("ref") or "").strip()
    )


def _raw_boss_visible(raw, visual_evidence_available):
    if not visual_evidence_available or not isinstance(raw, dict):
        return False
    value = raw.get("boss_visible")
    if isinstance(value, bool):
        return value
    return _positive_boss_state(raw.get("boss_state"))


def _recover_suggestion_ids(plan, members):
    """Recover a user_id the model truncated instead of failing the whole opening; only when unambiguous.

    `coop_generation` already has this defense (`_normalize_action_ids`: match by an identical
    source action, accept only a unique candidate); `coop_intro` did not. Suggestions only have
    `user_id` and `action`, no second anchor, so a wrong id raised "suggestions do not match the
    party" and failed the whole opening.

    Two signals, each effective only when **unique**:

    1. The written id is a prefix of exactly one real id: that is what truncation looks like.
    2. Exactly one unknown id and exactly one missing: a one-to-one match without ambiguity.

    With more than one unknown, do not guess: attaching a suggestion to the wrong player is worse
    than failing.
    """
    if not isinstance(plan, dict) or not isinstance(plan.get("suggestions"), list):
        return
    _recover_entry_ids(
        [s for s in plan["suggestions"] if isinstance(s, dict)], members, event="suggestion_user_id_recovered"
    )


def _recover_entry_ids(entries, members, *, event):
    """Recover wrongly written user_ids. The two rules of `_recover_suggestion_ids`, plus character names."""
    expected = [m["user_id"] for m in members]
    taken = {s.get("user_id") for s in entries if s.get("user_id") in expected}
    # The model often only has character names (the observation path used to send only names, see
    # player_characters in `coop_assets`), so it writes a name into user_id. When names are unique
    # in the room this mapping is unambiguous and more reliable than prefix matching, so try it
    # first.
    by_character = {}
    for m in members:
        name = " ".join(str(m.get("character") or "").split())
        if name:
            by_character.setdefault(name.casefold(), []).append(m["user_id"])
    for entry in [s for s in entries if s.get("user_id") not in expected]:
        name = " ".join(str(entry.get("user_id") or "").split()).casefold()
        hits = [uid for uid in by_character.get(name, []) if uid not in taken]
        if len(by_character.get(name, [])) == 1 and len(hits) == 1:
            entry["user_id"] = hits[0]
            taken.add(hits[0])
            reqlog.record(event, ok=True, reason="character_name", user_id=hits[0])
    unknown = [s for s in entries if s.get("user_id") not in expected]
    for entry in unknown:
        raw = str(entry.get("user_id") or "")
        hits = [uid for uid in expected if uid not in taken and raw and uid.startswith(raw)]
        if len(hits) == 1:
            entry["user_id"] = hits[0]
            taken.add(hits[0])
            reqlog.record(event, ok=True, reason="truncated_prefix", user_id=hits[0])
    unknown = [s for s in entries if s.get("user_id") not in expected]
    missing = [uid for uid in expected if uid not in taken]
    if len(unknown) == 1 and len(missing) == 1:
        unknown[0]["user_id"] = missing[0]
        reqlog.record(event, ok=True, reason="single_unambiguous_slot", user_id=missing[0])


def _validate_suggestions(plan, members):
    if not isinstance(plan, dict) or not isinstance(plan.get("suggestions"), list):
        raise ValueError("the opening has no action suggestions")
    _recover_suggestion_ids(plan, members)
    expected = {m["user_id"] for m in members}
    suggestions = {}
    for s in plan["suggestions"]:
        if not isinstance(s, dict) or s.get("user_id") not in expected or s["user_id"] in suggestions:
            raise ValueError("suggestions do not match the party")
        suggestions[s["user_id"]] = _text(s.get("action"), "开场行动建议", 500)
    if set(suggestions) != expected:
        raise ValueError("the suggestions leave out a teammate")
    return [{"user_id": m["user_id"], "action": suggestions[m["user_id"]]} for m in members]


def _validate_forecast(plan, context=None):
    if not isinstance(plan, dict):
        raise ValueError("no boss/environment forecast")
    world_event = _text(plan.get("world_event", ""), "局势变化", 300, False)
    event_type = plan.get("world_event_type")
    explicit_event_type = event_type in _EVENT_TYPES
    if event_type not in _EVENT_TYPES:
        if not world_event:
            event_type = "none"
        elif isinstance(plan.get("boss_candidate"), dict) and plan["boss_candidate"].get("name"):
            event_type = "boss_introduction"
        elif _has_shared_creature(context):
            event_type = "boss_action"
        else:
            event_type = "environment"
    if bool(world_event) != (event_type != "none"):
        raise ValueError("forecast type and content disagree")
    raw_event_key = plan.get("world_event_key")
    explicit_event_key = isinstance(raw_event_key, str) and bool(raw_event_key.strip())
    event_key = _clean_optional(raw_event_key, 80).casefold()
    if explicit_event_key and not re.fullmatch(r"[a-z0-9_]{2,80}", event_key):
        raise ValueError("the forecast event key must be English snake_case")
    if not world_event:
        if explicit_event_key:
            raise ValueError("the event key must be empty without a forecast")
        event_key = ""
    elif not event_key:
        raise ValueError("a forecast needs an English snake_case event key")
    story_context = (context or {}).get("story_context") or {}
    if (
        event_type == "boss_action"
        and (
            (context or {}).get("boss_life") or story_context.get("current_state", {}).get("boss_life") or {}
        ).get("status")
        == "dead"
    ):
        raise ValueError(
            "the boss is confirmed dead; forecast the aftermath or remaining goals, not the boss acting"
        )
    if (
        explicit_event_type
        and story_context.get("boss_introduction_due")
        and not (context or {}).get("observed_new_boss")
        and event_type != "boss_introduction"
    ):
        raise ValueError(
            "the story has reached the boss entrance; this round must forecast the boss appearing"
        )
    if (
        event_type == "boss_action"
        and not _has_shared_creature(context)
        and not (context or {}).get("observed_new_boss")
        and not _has_known_boss(story_context.get("current_state"))
    ):
        raise ValueError("no boss has appeared yet; use boss_introduction first")
    candidate = plan.get("boss_candidate") if isinstance(plan.get("boss_candidate"), dict) else {}
    if event_type == "boss_introduction":
        if not story_context.get("boss_introduction_allowed"):
            raise ValueError("the reference budget does not allow a new boss")
        candidate = {
            "name": _text(candidate.get("name"), "待登场 BOSS 名称", 60),
            "description": _text(candidate.get("description"), "待登场 BOSS 外观", 500),
        }
    else:
        candidate = {"name": "", "description": ""}
    return {
        "narration": _text(plan.get("narration"), "当前处境", 900),
        "world_event": world_event,
        "world_event_type": event_type,
        "world_event_key": event_key,
        "boss_candidate": candidate,
        "objective": _text(plan.get("objective"), "共同目标", 200),
    }


def _validate(plan, members, context=None):
    suggestions = _validate_suggestions(plan, members)
    return {**_validate_forecast(plan, context), "suggestions": suggestions}


def _validate_reactions(plan, members, forecast):
    if not isinstance(plan, dict) or plan.get("world_event_key") != forecast["world_event_key"]:
        raise ValueError("suggestions must match target_forecast's event key, not the previous forecast")
    suggestions = _validate_suggestions(plan, members)
    for item in plan["suggestions"]:
        _text(item.get("forecast_cue"), "行动依据", 300)
        _text(item.get("response_reason"), "应对说明", 300)
    return suggestions


def _validate_observation(plan, members, context=None, *, allow_empty_players=False):
    raw = plan.get("observation") if isinstance(plan, dict) else None
    if not isinstance(raw, dict):
        return None
    outcome = raw.get("event_outcome")
    if outcome not in _EVENT_OUTCOMES:
        raise ValueError("the observation does not say whether the last forecast happened")
    # user_ids in observations get miswritten like in suggestions: one wrong character in a 26-char
    # random id (`local-4120da325e7931ab00fa3e22`) voided the whole observation, roughly every few
    # beats, and the world lost its picture-based correction. Recover with the same two conservative
    # rules, accepting only a unique match.
    _recover_entry_ids(
        [i for i in (raw.get("player_states") or []) if isinstance(i, dict)],
        members,
        event="observation_user_id_recovered",
    )
    expected = {member["user_id"] for member in members}
    states = {}
    for item in raw.get("player_states") or []:
        if not isinstance(item, dict) or item.get("user_id") not in expected or item["user_id"] in states:
            raise ValueError("the observation's characters do not match the room")
        states[item["user_id"]] = _text(item.get("state"), "角色画面状态", 300)
    # ⚠️ **One missing person must not void the whole observation.**
    #
    # The observation also holds the environment, the boss and the state of the reported characters,
    # all real facts from the picture. Voiding it for one unreported person loses all picture
    # feedback for the beat; in two-player rooms it was usually the long `local-…` id that went
    # missing.
    #
    # The old version built suggestions from character positions and required at least one person.
    # The independent fact observation allows everyone off screen and still records environment /
    # boss; unreported characters are kept by publish_observation.
    if not states and not allow_empty_players:
        raise ValueError("the observation gives no character states")
    if set(states) != expected:
        reqlog.record(
            "observation_partial_player_states", ok=True, reported=len(states), expected=len(expected)
        )
    # ⚠️ **A missing story_phase must not void the whole observation.**
    #
    # This call's schema does not require story_phase and the system prompt says it is optional. The
    # model complied and sent only observation, but this check demanded it and the whole picture
    # confirmation was dropped. Same class of bug as requiring a user_id the payload never sent:
    # **validation enforcing what the schema does not and the prompt calls optional**.
    #
    # If missing, leave it: the phase stays (`publish_observation` already uses `.get`), and the
    # character state, environment and boss facts are written as usual. When a phase is given, the
    # "never go back to an earlier phase" rule below still applies.
    phase = plan.get("story_phase")
    if phase not in _STORY_PHASES:
        phase = ""
    story_context = (context or {}).get("story_context") or {}
    visual_evidence_available = bool(story_context.get("visual_evidence_available"))
    boss_state = _clean_optional(raw.get("boss_state"), 500)
    if not visual_evidence_available and raw.get("boss_visible") is True:
        raise ValueError("cannot confirm a boss without video keyframes")
    boss_visible = _raw_boss_visible(raw, visual_evidence_available)
    if not visual_evidence_available:
        if boss_visible:
            raise ValueError("cannot confirm a boss without video keyframes")
        # A textual plan is not evidence. Discard a model's negative or
        # speculative description so it cannot become a persistent boss fact.
        boss_state = ""
    elif boss_visible and (not boss_state or _denies_boss_presence(boss_state)):
        # ⚠️ What this blocks is the contradiction "saw the boss, but its state says it is absent".
        # Neutral values (no_change / unknown) are not contradictions: the boss is in the picture
        # and nothing changed since the last beat, a normal observation. Treating neutral as invalid
        # broke right after it shipped.
        raise ValueError("a visible boss needs a valid boss state")
    elif not boss_visible and boss_state and _positive_boss_state(boss_state):
        raise ValueError("an invisible boss cannot have an on-screen state")
    else:
        # Phrases such as “画面中未发现 BOSS” describe absence of evidence,
        # not a durable world state.
        boss_state = "" if not boss_visible else boss_state
    return {
        "node_id": "",
        "event_outcome": outcome,
        "summary": _text(raw.get("summary"), "画面变化摘要", 500),
        "environment_state": _text(raw.get("environment_state"), "环境状态", 500),
        "boss_visible": boss_visible,
        "boss_state": boss_state,
        "player_states": [
            {"user_id": member["user_id"], "state": states[member["user_id"]]}
            for member in members
            if member["user_id"] in states
        ],
        "story_phase": phase,
        "_visual_evidence_available": visual_evidence_available,
    }


def validate_response(plan, members, context=None, *, latest=False, forecast_only=False):
    context = dict(context or {})
    if latest and isinstance(plan, dict) and "observed_new_boss" not in context:
        context["observed_new_boss"] = _raw_boss_visible(
            plan.get("observation"),
            bool((context.get("story_context") or {}).get("visual_evidence_available")),
        )
    result = _validate_forecast(plan, context) if forecast_only else _validate(plan, members, context)
    observation = _validate_observation(plan, members, context) if latest else None
    if observation:
        story_context = (context or {}).get("story_context") or {}
        if not story_context.get("visual_evidence_available") and observation["event_outcome"] != "unknown":
            raise ValueError("cannot claim the forecast happened without video keyframes")
        current_phase = (story_context.get("current_state") or {}).get("phase")
        if current_phase in _STORY_PHASES and observation.get("story_phase") in _STORY_PHASES:
            current_index = _STORY_PHASES.index(current_phase)
            next_index = _STORY_PHASES.index(observation["story_phase"])
            if current_phase != "aftermath" and next_index < current_index:
                raise ValueError("the story cannot go back to a completed stage")
    return result, observation
