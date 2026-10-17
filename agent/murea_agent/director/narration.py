"""Story history, narration captions and suggested options."""

from __future__ import annotations

import re
from typing import Any
from murea_agent import config
from murea_agent.util import _in_story_language, _say
from murea_agent.guards.motion import (
    _PHASE_RELATIONS,
    _clean_motion_text,
    _motion_active,
)
from murea_agent.guards.action_sequence import (
    _DIALOGUE_ACTION,
    _GOAL_RELATIONS,
    _goal_relation_instruction,
    _GOAL_RELATION_PATTERNS,
    _OBSERVATION_ACTION,
    _action_sequence_errors,
    _complete_sequence_step_fields,
    _dialogue_step_from,
    _ensure_action_sequence_budget,
    _ensure_goal_relations,
    _normalize_action_sequence,
    _repair_action_sequence_phases,
    _safe_relation_step,
    _sequence_goal,
)


# "[beat 3 · place · summary]"; worlds saved before it was language-neutral say "已生成第 3 段".
_BEAT_TAIL = re.compile(r"\n\[(?:beat |已生成第 )(\d+)(?: 段)? · ([^·\]]+) · (.*)\]$", re.S)


def _clean_history(history: list[dict[str, str]]) -> list[dict[str, str]]:
    """Filter machine text out of the history before sending it to the model.

    The write side already blocks it (`_history_line` / `_history_beat`), but **worlds already saved
    do not clean themselves**: one world's 14 history lines held 6 English physics descriptions and
    continuation instructions, and the director kept reading and imitating them every beat.

    So filter again on the sending side. Nothing is written back: user data must not be changed
    silently by a display-level cleanup, and recomputing takes microseconds.
    """
    out: list[dict[str, str]] = []
    for item in history:
        text = str(item.get("text") or "")
        if item.get("role") == "user":
            if _CAPTION_PLANNING_LANGUAGE.search(text) or not _in_story_language(text):
                # This must be a continuation instruction built by the server; users do not talk
                # like this. Keep a neutral placeholder so the fact that a beat happened remains,
                # without teaching the model machine language.
                text = _say("（接着上一步演下去）", "(continue from the last step)")
        else:
            hit = _BEAT_TAIL.search(text)
            if hit:
                beat = hit.group(3).strip()
                if _CAPTION_PLANNING_LANGUAGE.search(beat) or not _in_story_language(beat):
                    # The bracketed part is a one-line summary for the director. When it is
                    # mechanical English, use the reply of the same message instead, which is this
                    # beat's narrative anyway.
                    reply = text[: hit.start()].strip()
                    text = f"{reply}\n[beat {hit.group(1)} · {hit.group(2).strip()} · {reply[:60]}]"
        out.append({"role": str(item.get("role") or "user"), "text": text})
    return out


def _history_beat(node: Any) -> str:
    """The line that goes into the director's history. Only narrative in the story language; machine text is replaced.

    Candidates in order: recap (written for the history) → beat (old clips only have this) → reply.
    """
    for value in (getattr(node, "recap", ""), getattr(node, "beat", "")):
        text = _clean_motion_text(value)
        if text and _in_story_language(text) and not _CAPTION_PLANNING_LANGUAGE.search(text):
            return text[:120]
    return _clean_motion_text(getattr(node, "reply", ""))[:120]


def _history_line(user_text: str, node: Any) -> str:
    """The "user said" line in the director's history.

    The user_text of an auto-continued beat is the server's continuation instruction. It is plain
    language now (see `_goal_relation_instruction`), but if planning terms still slip in, use what
    the clip actually showed: the history must never contain machine text the player never said.
    """
    text = " ".join(str(user_text or "").split())
    if text and not _CAPTION_PLANNING_LANGUAGE.search(text) and _in_story_language(text):
        return text
    return _history_beat(node) or text


def _sequence_steps_text(sequence: dict[str, Any]) -> list[str]:
    """The steps not played yet, one sentence each, in order.

    ⚠️ This used to join steps[1:] into one "next X: A, then B, then C" for the next beat. Given that, the
    director **replanned the whole goal from the start**, the finished step reappeared in the queue,
    and the server could only drop the head without knowing what was done: four auto-continuations
    were spent on the same door. One step at a time leaves nothing to split and so nothing to
    reorder.
    """
    steps = sequence.get("steps") or []
    if len(steps) <= 1:
        return []
    safe_steps: list[str] = []
    for step in steps[1:]:
        relation = step.get("relation_change", "")
        # Keep the natural action the director wrote for the next stage; only when missing or still
        # in planning terms fall back to the relation template, so auto-continuation carries real
        # narrative instead of collapsing every action into the same internal instruction.
        original = _clean_motion_text(step.get("instruction"))
        instruction = (
            original
            if original and _in_story_language(original) and not _CAPTION_PLANNING_LANGUAGE.search(original)
            else _goal_relation_instruction(relation, original)
        )
        place = _clean_motion_text(step.get("place"))
        if place and place not in instruction and relation == "containment":
            instruction = _say(f"进入{place}", f"Go into {place}")
        # The continuity anchor is no longer sent out. It is an audit field for the planner; putting
        # it in this sentence reads internal reasoning to the user and pollutes the director's
        # history as "the user said". Continuity has a better carrier: the previous clip's last
        # frame is sent with every beat.
        safe_steps.append(instruction)
    # The next beat only gets its own task. The full original goal used to be prefixed: "continuing
    # to crawl into the room: open the valve". The director saw "crawl into the room" again and the
    # second segment repeated the first one's crossing. A standalone instruction keeps continuity
    # without rescheduling finished actions.
    if config.LANGUAGE != "zh":
        return [x[:400] for x in safe_steps if x]
    return [(("" if re.match(r"^(?:接着|继续)", x) else "接着") + x)[:400] for x in safe_steps if x]


def _align_first_action_step(
    directive: dict[str, Any],
    sequence: dict[str, Any],
    motion_plan: dict[str, Any],
) -> None:
    """Align the first step of a misordered action sequence with the actual motion plan."""
    steps = sequence.get("steps") or []
    if not steps:
        return
    first = steps[0]
    expected_phase = motion_plan.get("phase", "none")
    expected_relation = motion_plan.get("relation_change", "none")
    if first.get("phase") == expected_phase and first.get("relation_change") == expected_relation:
        return

    if motion_plan.get("interaction_type") == "impact" and expected_relation == "contact":
        # Preserve the GM's natural instruction and structured event roles. The
        # body-contact fallback is intentionally inapplicable to an instrument.
        first["phase"] = "engage"
        first["relation_change"] = "contact"
        directive.setdefault("_server_repairs", []).append(
            "projectile step aligned to the impact motion plan (engage/contact)"
        )
        return

    # motion_plan has passed structured physics checks this round and is what enters the video
    # prompt; action_sequence is only the auto-continue queue. On conflict repair from the former,
    # so the user is not stuck at the input.
    original_relation = first.get("relation_change")
    if expected_relation in _GOAL_RELATIONS:
        anchor = _clean_motion_text(first.get("continuity_anchor"))
        repaired = _safe_relation_step(expected_relation, anchor)
        # The fallback step carries the relation's default phase (distance -> approach), but a plan
        # may legally use another one (settle/distance: come to rest a distance away). Keep the
        # plan's phase, or the check below rejects the step this function just aligned.
        if expected_relation in _PHASE_RELATIONS.get(expected_phase, set()):
            repaired["phase"] = expected_phase
        steps[0] = {**first, **repaired}
        # If the original first step is another relation from the user's goal not done yet (e.g. the
        # step was "enter" but motion_plan says "approach" first), do not swallow it while aligning;
        # keep it as the second step so auto-continue still completes the whole goal.
        if (
            original_relation in _GOAL_RELATIONS
            and original_relation != expected_relation
            and not any(step.get("relation_change") == original_relation for step in steps[1:])
            and len(steps) < 3
        ):
            steps.insert(1, dict(first))
            steps[1].update(_safe_relation_step(original_relation, anchor))
    else:
        first["phase"] = expected_phase
        first["relation_change"] = expected_relation
    directive.setdefault("_server_repairs", []).append("first action step aligned to motion_plan")


def _apply_action_sequence(
    directive: dict[str, Any],
    fallback_current: str,
    motion_plan: dict[str, Any],
    request_text: str = "",
    segment_seconds: int = 4,
    is_continuation: bool = False,
    carried_plan: list[str] | None = None,
    player_actor: str = "",
) -> dict[str, Any]:
    """The current beat runs only the head of the queue; continue_with is derived deterministically from the rest.

    `is_continuation` = this beat was continued by the system from the previous continue_with. No
    re-splitting by default; only when the independent goal audit confirms the current event was
    missed may its local steps be repaired.
    """
    raw = directive.get("action_sequence")
    if not isinstance(raw, dict):
        # Compatibility with old caches and test stubs; the real schema always returns a
        # structured sequence.
        return {"goal": "", "budget_reason": "", "steps": []}
    sequence = _normalize_action_sequence(raw)
    # The director no longer returns goal: on the normal path it would be overwritten by
    # `request_text` anyway. The fallback still checks `sequence["goal"]` first: old worlds store
    # it, and replacing it with `fallback_current` would change their goal to the current beat (goal
    # drift, exactly what this code prevents). Only when both are empty use the caller's current
    # instruction.
    sequence["goal"] = _sequence_goal(request_text, sequence.get("goal") or fallback_current)
    _ensure_goal_relations(sequence)
    # Dialogue is a structured GM decision. Never infer it from the presence or
    # absence of words in the user's sentence: object impact and body contact share
    # relation=contact, while spoken dialogue changes no physical relation.
    sequence["steps"] = [
        _dialogue_step_from(step)
        if step.get("kind") == "dialogue" and step.get("relation_change") == "contact"
        else step
        for step in sequence.get("steps") or []
        if isinstance(step, dict)
    ]
    _complete_sequence_step_fields(sequence)
    _ensure_action_sequence_budget(sequence, motion_plan, segment_seconds)
    _repair_action_sequence_phases(sequence, directive)
    _align_first_action_step(directive, sequence, motion_plan)
    errors = _action_sequence_errors(sequence, motion_plan)
    if errors:
        raise ValueError("Invalid action sequence: " + "; ".join(errors))
    # Auto-continue plays one step by default, so finished events are never queued again.
    #
    # Real failure: "open the door and walk out" was split into "open → walk out" and the door was
    # opened. The next beat came in with the continuation and the director **replanned the whole
    # goal**, so "open" reappeared in the queue. The server only dropped the head without knowing
    # what was done, every beat reordered the same plan, four continuations were spent on one door
    # and it hit AUTO_MAX.
    #
    # The server's carried_plan is kept by default and the queue shrinks step by step. When an
    # independent audit confirms the current event was missed, its local steps may be repaired and
    # the existing queue appended.
    repair_current_event = bool(directive.pop("_repair_current_event", False))
    if is_continuation and not repair_current_event and len(sequence.get("steps") or []) > 1:
        sequence["steps"] = sequence["steps"][:1]
    directive["action_sequence"] = sequence
    first = sequence["steps"][0]
    # A physical action's instruction sometimes smuggles in the next stage (e.g. "grab" inside
    # approach). The beat sent to the rewriter only uses the checked path and end_state; static /
    # dialogue beats use the original instruction.
    # ⚠️ `beat` has two consumers with opposite needs: the rewriter wants English physical detail, the history a
    # narrative sentence. Sharing one field filled the history with "the acting hand first reaches
    # the visible handle…". Now `beat` goes to the rewriter and `recap` into the history (written in
    # `_history_beat`).
    narrative = _clean_motion_text(directive.get("beat"))
    if not (narrative and _in_story_language(narrative) and not _CAPTION_PLANNING_LANGUAGE.search(narrative)):
        narrative = ""
    if _motion_active(motion_plan):
        path = _clean_motion_text(motion_plan.get("path")).rstrip(" .;:")
        end_state = _clean_motion_text(motion_plan.get("end_state")).rstrip(" .;:")
        directive["beat"] = f"{path}. End state: {end_state}."
    else:
        directive["beat"] = first["instruction"] or fallback_current
    directive["recap"] = narrative or _current_segment_reply(
        sequence, motion_plan, directive.get("reply"), player_actor
    )
    # ⚠️ When continuing, the remaining queue is **the one carried from the previous beat**, not the director's
    # new reordering.
    #
    # The director replans the whole goal every beat and has no notion of which steps were played;
    # that is server state. The remaining steps were fixed when the previous beat computed
    # continue_with; walking them, the queue only shrinks and always converges.
    if is_continuation and repair_current_event:
        # Only an independent goal audit can expand this unfinished event. Keep
        # the pre-existing later queue after its repaired local substeps.
        rest = _sequence_steps_text(sequence) + list(carried_plan or [])
        directive["continue_with"] = rest[0] if rest else ""
        directive["plan"] = rest[1:]
    elif is_continuation and carried_plan:
        directive["continue_with"] = carried_plan[0]
        directive["plan"] = list(carried_plan[1:])
    else:
        rest = _sequence_steps_text(sequence)
        directive["continue_with"] = rest[0] if rest else ""
        directive["plan"] = rest[1:]
    return sequence


_CAPTION_PLANNING_LANGUAGE = re.compile(
    r"这个动作|(?:分成|拆成)\s*\d*\s*段|随后|继续完成|本段|这一段|"
    r"当前(?:步骤|这一步)|交互|构型变化|关系变化|原始目标|目标空间|"
    r"交互目标|衔接锚点|服务端|完成(?:当前|这)?(?:步骤|关系)",
    re.I,
)

_NATURAL_RELATION_CAPTION = {
    "distance": "来到眼前的位置，稳稳停下",
    "contact": "在指定位置与对方碰面",
    "containment": "穿过敞开的入口，来到另一侧",
    "support": "把身体重量稳稳交给支撑面",
    "possession": "握住物件，将它稳稳拿在手中",
    "configuration": "把眼前那件东西彻底打开，人还停在外面",
    "none": "完成眼前的动作",
}


def _caption_action_kinds(text: str) -> set[str]:
    """Read the action relations stated in the narration, to detect leaks of later actions.

    Reuses the generic relation vocabulary, with no lists of places, objects or vehicles. The
    narration may add environment detail but must not claim another irreversible action happened
    beyond the current relation.
    """
    value = _clean_motion_text(text)
    if not value:
        return set()
    kinds: set[str] = set()
    for relation, pattern in _GOAL_RELATION_PATTERNS.items():
        if pattern.search(value):
            kinds.add(relation)
    if _OBSERVATION_ACTION.search(value):
        kinds.add("observation")
    if _DIALOGUE_ACTION.search(value):
        kinds.add("dialogue")
    return kinds


def _compact_caption_actor(value: Any) -> str:
    """Normalize a character name into a comparable label so spaces and punctuation do not break the perspective check."""
    return re.sub(r"[\s，。！？!?、：:；;（）()\"'“”‘’]+", "", str(value or "")).casefold()


def _is_player_caption_actor(actor: str, player_actor: str) -> bool:
    """Rewrite only the player character into the second person; NPCs stay in the third person."""
    actor_key = _compact_caption_actor(actor)
    player_key = _compact_caption_actor(player_actor)
    return bool(actor_key and player_key and actor_key == player_key)


def _as_player_narration(text: str, actor: str, player_actor: str) -> str:
    """Replace the player character's name / third person at the start with "you", leaving NPC narration alone."""
    if config.LANGUAGE != "zh":
        return text  # English narration is written by the agent as-is
    if not text or not _is_player_caption_actor(actor, player_actor):
        return text
    escaped = re.escape(actor)
    # Names usually directly precede the verb without punctuation; replace only at the start of the
    # sentence so mentions in dialogue or the environment are untouched. If the model uses he / she
    # for the player, only that leading occurrence is handled.
    text = re.sub(rf"^{escaped}(?=[^A-Za-z0-9]|$)", "你", text, count=1)
    text = re.sub(r"^(?:他|她|该角色)(?=[^A-Za-z0-9]|$)", "你", text, count=1)
    return text


def _caption_char_count(text: str) -> int:
    """Count meaningful narration characters, ignoring punctuation and internal tokens.

    English stories count two per word, so the same thresholds mean roughly
    the same amount of narration.
    """
    if config.LANGUAGE != "zh":
        return 2 * len(re.findall(r"[A-Za-z]+", text))
    return sum(1 for char in text if "\u4e00" <= char <= "\u9fff")


def _expand_brief_player_caption(
    text: str,
    relation: str,
    actor: str,
    player_actor: str,
    target: str,
    interaction_type: str = "",
) -> str:
    """Give a too-short player action an audible path and result, without inventing extra events.

    Normally the director prompt produces full narration; this is only a deterministic fallback for
    old caches or a model that still returns "walks up, stops". Content comes only from the current
    motion_plan's relation and target.
    """
    if interaction_type == "impact" or config.LANGUAGE != "zh":
        return text
    if not _is_player_caption_actor(actor, player_actor) or _caption_char_count(text) >= 16:
        return text
    destination = (
        target
        if target and _in_story_language(target) and not _CAPTION_PLANNING_LANGUAGE.search(target)
        else "眼前的位置"
    )
    if relation == "distance":
        return f"你沿着眼前的通路向前，来到{destination}前，稳稳停下。"
    if relation == "containment":
        return f"你沿着敞开的入口向前，穿过边界来到{destination}，在另一侧停稳。"
    if relation == "possession":
        return f"你伸手握住{destination}，把它稳稳留在手中。"
    if relation == "support":
        # Helping another actor stand is not transferring the player's own weight.
        # Preserve a specific target-bearing caption instead of changing its subject.
        if target and _in_story_language(target) and target in text:
            return text
        return "你把身体重量交给支撑面，稳稳站住。"
    if relation == "contact":
        return f"你伸手触到{destination}，在接触处停稳。"
    return text


def _reply_matches_current_segment(
    value: Any,
    sequence: dict[str, Any],
    motion_plan: dict[str, Any],
) -> bool:
    """Whether the director's reply only tells what already happened in the current clip.

    A multi-segment reply may still read like natural narration but must contain the current
    segment's relation result and no other relation change. So "walks to the door and opens it /
    gets into the car" falls back to a safe description of the current segment, while "you walk down
    the alley and stop at the corner" is kept.
    """
    text = _clean_motion_text(value)
    if not text or not _in_story_language(text) or _CAPTION_PLANNING_LANGUAGE.search(text):
        return False
    steps = sequence.get("steps") or []
    first = steps[0] if steps and isinstance(steps[0], dict) else {}
    relation = (
        _clean_motion_text(motion_plan.get("relation_change") or first.get("relation_change")).lower()
        or "none"
    )
    kind = _clean_motion_text(first.get("kind")).lower()
    detected = _caption_action_kinds(text)

    # A clip with a physical relation must state the current relation's result in the narration;
    # otherwise the next stage's "look / search" or "enter" is easily mistaken for this segment's
    # subtitle.
    if relation != "none" and relation not in detected:
        return False
    if relation == "none" and kind in {"observation", "dialogue"} and kind not in detected:
        return False

    # Any other high-risk relation means the model wrote a future action into the current subtitle.
    # Looking and dialogue are allowed as natural results of the same relation, but blocked when
    # they are the next step's own event.
    physical = {"distance", "configuration", "possession", "containment", "support", "contact"}
    if (detected & physical) - ({relation} if relation in physical else set()):
        return False
    later_kinds = {
        _clean_motion_text(step.get("kind")).lower() for step in steps[1:] if isinstance(step, dict)
    }
    return not detected & {"observation", "dialogue"} & later_kinds


# Verbs that commonly follow "The player ... and" in step instructions; only these are
# turned back to the base form after "and", so nouns ending in -s are left alone.
_ENGLISH_ACTION_VERBS = frozenset(
    [
        "approach",
        "ask",
        "carry",
        "check",
        "climb",
        "close",
        "cross",
        "crouch",
        "draw",
        "drop",
        "enter",
        "face",
        "follow",
        "glance",
        "grab",
        "greet",
        "head",
        "hold",
        "jump",
        "keep",
        "kneel",
        "knock",
        "lean",
        "leave",
        "let",
        "lift",
        "light",
        "listen",
        "look",
        "move",
        "nod",
        "open",
        "pause",
        "pick",
        "place",
        "point",
        "press",
        "pull",
        "push",
        "put",
        "raise",
        "reach",
        "read",
        "return",
        "run",
        "say",
        "scan",
        "set",
        "settle",
        "shake",
        "shout",
        "sit",
        "slip",
        "smile",
        "speak",
        "stand",
        "step",
        "stop",
        "study",
        "swing",
        "take",
        "tap",
        "touch",
        "turn",
        "wait",
        "walk",
        "watch",
        "wave",
        "whisper",
        "wipe",
        "write",
    ]
)


def _english_base_verb(word: str) -> str:
    """Third-person singular -> base form ("pushes" -> "push", "is" -> "are")."""
    low = word.lower()
    irregular = {"is": "are", "has": "have", "does": "do", "goes": "go", "was": "were"}
    if low in irregular:
        return irregular[low]
    if low.endswith("ies") and len(low) > 4:
        return word[:-3] + "y"
    if re.search(r"(?:ch|sh|ss|x|z|o)es$", low):
        return word[:-2]
    if low.endswith("s") and not low.endswith("ss"):
        return word[:-1]
    return word


def _english_second_person(text: str, player_actor: str = "") -> str:
    """Address the player character as "you" in an English caption.

    Step instructions often say "The player pushes the door" or use the character's name. Only
    the player is rewritten, never other characters or pronouns that may mean someone else.
    """
    if not text:
        return text
    names = ["the player character", "the player"]
    if player_actor and not re.fullmatch(r"S\d+", player_actor):
        names.insert(0, player_actor)
    subject = "|".join(re.escape(n) for n in names)
    text = re.sub(
        rf"(^|[.!?]\s+)(?:{subject})\s+(\w+)",
        lambda m: f"{m.group(1)}You {_english_base_verb(m.group(2))}",
        text,
        flags=re.I,
    )
    if re.match(r"You\b", text):
        # "You push the door open and steps in" -> "... and step in"
        text = re.sub(
            r"\band (\w+)",
            lambda m: (
                "and "
                + (
                    _english_base_verb(m.group(1))
                    if _english_base_verb(m.group(1)).lower() in _ENGLISH_ACTION_VERBS
                    else m.group(1)
                )
            ),
            text,
        )
    text = re.sub(rf"\b(?:{subject})'s\b", "your", text, flags=re.I)
    return re.sub(
        rf"\b(at|to|toward|towards|with|for|behind|beside) (?:{subject})\b", r"\1 you", text, flags=re.I
    )


def _caption_sentence(value: Any) -> str:
    """Turn an action description into one narration line ready to show / read aloud."""
    text = _clean_motion_text(value).rstrip("。！？!?；;，,")
    return f"{text}。" if text else ""


def _current_segment_reply(
    sequence: dict[str, Any],
    motion_plan: dict[str, Any],
    directive_reply: Any = "",
    player_actor: str = "",
) -> str:
    """Generate natural narration only for the current short clip, never exposing the split or the continuation plan."""
    steps = sequence.get("steps") or []
    first = steps[0] if steps and isinstance(steps[0], dict) else {}
    instruction = _clean_motion_text(first.get("instruction"))
    if config.LANGUAGE != "zh":
        # The Chinese caption templates below do not apply. The director's reply is the real
        # narration and already addresses the player as "you"; the step instruction is a machine
        # instruction (often imperative or "The player ...") and only a fallback.
        spoken = _clean_motion_text(directive_reply)
        text = spoken if spoken and not _CAPTION_PLANNING_LANGUAGE.search(spoken) else instruction
        text = _english_second_person(text, player_actor)
        return text if not text or text[-1] in ".!?\"'" else text + "."
    step_place = first.get("place") or ""
    actor_values = motion_plan.get("actors") or []
    # Static / dialogue actions may have no explicit actors; never take actors[0] just to name the
    # subtitle, or an empty list turns the whole round into an IndexError. Without a character the
    # relation template uses a neutral "the person" and still gives audible narration.
    actor = _clean_motion_text(actor_values[0]) if isinstance(actor_values, list) and actor_values else ""
    if re.fullmatch(r"S\d+", actor):
        actor = ""  # Stable asset IDs are not spoken character names.
    target = _clean_motion_text(motion_plan.get("target"))
    relation = _clean_motion_text(motion_plan.get("relation_change")).lower() or "none"

    # The director's reply is the only real narrative of this beat. Use it first rather than the
    # short instruction from the action sequence, which is a machine instruction for auto-continue,
    # often just "walks up, stops" and often third person. If the reply already describes later
    # relations, fall back to the current step so subtitles and picture stay in sync.
    spoken = _clean_motion_text(directive_reply)
    if _reply_matches_current_segment(spoken, sequence, motion_plan):
        spoken = _as_player_narration(spoken, actor, player_actor)
        spoken = _expand_brief_player_caption(
            spoken,
            relation,
            actor,
            player_actor,
            target,
            _clean_motion_text(motion_plan.get("interaction_type")),
        )
        return _caption_sentence(spoken)

    # The instruction is the second choice: it must be in the story language and free of planning
    # terms. A character name is added only for NPCs; the player character always becomes second
    # person, keeping every clip's narration in the same perspective.
    if instruction and _in_story_language(instruction) and not _CAPTION_PLANNING_LANGUAGE.search(instruction):
        instruction = _as_player_narration(instruction, actor, player_actor)
        if (
            actor
            and not _is_player_caption_actor(actor, player_actor)
            and actor not in instruction
            and not re.match(r"^(人物|他|她|它|双方|两人)", instruction)
        ):
            instruction = f"{actor}{instruction}"
        # No longer force "comes to a steady stop" on every distance action. Retreating, chasing and
        # gaining distance may end in motion; whether to stop is decided explicitly by the
        # director's narration and motion_plan.end_state, so subtitles never change the action's
        # meaning.
        if _is_player_caption_actor(actor, player_actor) and not re.match(r"^你", instruction):
            instruction = f"你{instruction}"
        instruction = _expand_brief_player_caption(
            instruction,
            relation,
            actor,
            player_actor,
            target,
            _clean_motion_text(motion_plan.get("interaction_type")),
        )
        return _caption_sentence(instruction)

    # Third choice: the director's own narration. It was rejected above usually for saying a bit
    # more about what follows, but a little extra is far better than the relation template below,
    # which was written for opening doors and lids and produces nonsense for actions like shooting
    # an arrow or teleporting ("you fully open the movable part, your body still outside").
    if spoken and _in_story_language(spoken) and not _CAPTION_PLANNING_LANGUAGE.search(spoken):
        fallback = _as_player_narration(spoken, actor, player_actor)
        return _caption_sentence(
            _expand_brief_player_caption(
                fallback,
                relation,
                actor,
                player_actor,
                target,
                _clean_motion_text(motion_plan.get("interaction_type")),
            )
        )

    template = _NATURAL_RELATION_CAPTION.get(relation, _NATURAL_RELATION_CAPTION["none"])
    if motion_plan.get("interaction_type") == "impact":
        instrument = _clean_motion_text(motion_plan.get("instrument")) or "物件"
        target_name = target or "接触位置"
        template = f"{actor or '人物'}使{instrument}沿连续路径运动并命中{target_name}"
    # Include the name only if it is in the story language and not an internal placeholder, so the
    # fallback never shows terms like "target".
    if (
        actor
        and actor not in template
        and _in_story_language(actor)
        and not _CAPTION_PLANNING_LANGUAGE.search(actor)
    ):
        template = f"{actor}{template}"
    if (
        target
        and _in_story_language(target)
        and "目标" not in target
        and not _CAPTION_PLANNING_LANGUAGE.search(target)
    ):
        if relation == "distance":
            template = f"{actor or '人物'}来到{target}前，稳稳停下"
        elif relation == "containment":
            # ⚠️ Use the destination, not motion_plan.target. Real failure: someone walked from a cabin to the street,
            # and target was the cabin whose door they **left** through, so the subtitle named the
            # starting point. step.place is the destination.
            destination = _clean_motion_text(step_place) or target
            template = f"{actor or '人物'}穿过敞开的入口，来到{destination}"
    template = _as_player_narration(template, actor, player_actor)
    if _is_player_caption_actor(actor, player_actor) and not re.match(r"^你", template):
        template = f"你{template}"
    template = _expand_brief_player_caption(
        template,
        relation,
        actor,
        player_actor,
        target,
        _clean_motion_text(motion_plan.get("interaction_type")),
    )
    return _caption_sentence(template)


def _has_split_planning_reply(value: Any) -> bool:
    """Detect segment-planning notes the model or an old cache wrote into the user's subtitle."""
    text = _clean_motion_text(value)
    return bool(_CAPTION_PLANNING_LANGUAGE.search(text))


def _sanitize_intermediate_directive(
    directive: dict[str, Any],
    sequence: dict[str, Any],
    motion_plan: dict[str, Any],
    player_actor: str = "",
) -> None:
    """A multi-segment goal settles only facts already visible in the current segment; it must not rewrite world history early."""
    steps = sequence.get("steps") or []
    if len(steps) <= 1:
        return

    relation = motion_plan.get("relation_change", "none")
    # Show only the current step's natural description; later steps stay in continue_with for
    # internal continuation.
    directive["reply"] = _current_segment_reply(sequence, motion_plan, directive.get("reply"), player_actor)

    # Intermediate beats of a multi-segment goal must not mark quests completed / failed early. Keep
    # creation and progress text; the final state is confirmed by the director after the last beat
    # really happens.
    quest_updates = directive.get("quest_updates")
    if isinstance(quest_updates, dict) and isinstance(quest_updates.get("update"), list):
        for update in quest_updates["update"]:
            if isinstance(update, dict) and update.get("status") in {"completed", "failed"}:
                update["status"] = "active"

    updates = directive.get("state_updates")
    if not isinstance(updates, dict):
        return
    subjects = updates.get("subjects")
    if isinstance(subjects, list):
        for update in subjects:
            if not isinstance(update, dict):
                continue
            # Discoveries, forgetting and new character goals usually belong to story not filmed
            # yet. An intermediate segment has no evidence they happened; let the director settle
            # them in the final segment.
            update.pop("knowledge_add", None)
            update.pop("knowledge_remove", None)
            if relation == "distance":
                update.pop("goal", None)
    if relation != "distance":
        return
    places = updates.get("places")
    if isinstance(places, list):
        for update in places:
            if not isinstance(update, dict):
                continue
            # A pure approach only changes the distance between the person and the target; it cannot
            # change a door, prop or space before the person actually interacts with it.
            update.pop("condition", None)
            update.pop("props_add", None)
            update.pop("props_remove", None)


# The frontend lays out three fixed options (plus "other" for typing). A missing one leaves a hole,
# so pad here. A padded option is weaker than the director's own, but a hole is worse: it reads as
# "the world has nothing to say".
_FALLBACK_OPTIONS_ZH = (
    "四下看看这儿还有什么",
    "朝旁边的人开口问一句",
    "往前再走一步",
)
_FALLBACK_OPTIONS_EN = (
    "Look around for anything else here",
    "Ask the person nearby a question",
    "Take another step forward",
)

# Normal form for deciding whether two options are "the same thing". Punctuation and spacing do not
# change the meaning; keeping them would make identical options look different.
_OPT_PUNCT = re.compile(r"[\s，。！？、；：,.!?;:~…—－·\-]+")


def _norm_opt(text: str) -> str:
    return _OPT_PUNCT.sub("", text)


def _three_options(raw: Any, just_did: str = "") -> list[str]:
    """Exactly three: drop empties and duplicates, **remove what was just done**, truncate extras, pad missing ones.

    Removing what was just done came from observation: the user picked "charge forward to draw the
    beast's attention" and the director's first option next beat was the same. Playing it again
    makes the world look stuck, and moving forward is the whole point of options. Code can decide
    this, so it is not left to the model.
    """
    did = _norm_opt(just_did.strip())
    seen: set[str] = set()
    out: list[str] = []
    for item in raw or []:
        text = str(item).strip()
        key = _norm_opt(text)
        if not text or key in seen:
            continue
        # Identical, or one fully contains the other (only when both are long enough: containment
        # between short ones is noise, "walk" would also match "walk into the control room")
        if did and (key == did or (len(key) >= 6 and len(did) >= 6 and (key in did or did in key))):
            continue
        seen.add(key)
        out.append(text)
        if len(out) == 3:
            return out
    for text in _FALLBACK_OPTIONS_ZH if config.LANGUAGE == "zh" else _FALLBACK_OPTIONS_EN:
        if len(out) == 3:
            break
        if _norm_opt(text) not in seen:
            seen.add(_norm_opt(text))
            out.append(text)
    return out


def _recommend_index(raw: Any, count: int) -> int:
    """Which option is recommended. Not a number, out of range, or missing all fall back to the first.

    The fallback is correct by itself: the prompt says the first option always advances the
    mainline.
    """
    try:
        index = int(raw)
    except (TypeError, ValueError):
        return 0
    return index if 0 <= index < count else 0
