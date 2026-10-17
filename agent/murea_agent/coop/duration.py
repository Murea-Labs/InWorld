"""Choose shot length from causal action timing, without another model call."""

from __future__ import annotations

from dataclasses import dataclass
import copy
import math
import re
from typing import Any
from collections.abc import Mapping

from murea_agent import config
from murea_agent.providers import video

POLICY_VERSION = 2
END_HOLD_SECONDS = 0.75
EVENT_ACTOR = "world_event"

TIMING_SCHEMA = {
    "type": "object",
    "properties": {
        "steps": {
            "type": "array",
            "minItems": 1,
            "maxItems": 16,
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "actors": {"type": "array", "items": {"type": "string"}},
                    "action": {
                        "type": "string",
                        "description": "Short 2-6 word name of the visible step, not another action paragraph.",
                    },
                    "seconds": {"type": "number"},
                    "after": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["id", "actors", "action", "seconds", "after"],
            },
        },
    },
    "required": ["steps"],
}

TIMING_INSTRUCTION = """\nAlso return timing.steps for the visible performance you just staged, including every supplied player and any upcoming_world_event.
duration_seconds is a planning CEILING, not the selected clip length. Estimate the natural time needed; do not pad actions to fill it or quote a fixed shot duration in action/event_action.
Use at most 16 short steps in causal order, normally 1-3 per player. Each step has a unique id, actors (exact user_id values, or world_event for the permitted world behavior), a 2-6 word action label, seconds (0.5-30), and after (ids of earlier steps that must FINISH before this step can begin).
Independent players and concurrent world behavior start together: after=[] unless a real prerequisite requires waiting. seconds excludes time already spent waiting for predecessors. Never serialize players just because of their input order. Within a player, list successive actions in order; combine things done simultaneously by that player into one step. A coordinated simultaneous action may list multiple actors once.
Estimate from the path, distance, movement speed, preparation, execution and visible ending. Distinguish a few steps from crossing a hall; do not use description length or cast size. Do not assume an unknown distance is exactly measured. Short speech uses its actual spoken length; mentioning magic/chanting or a BOSS does not automatically need a long clip.
Example: player A opens a gate (3s), then player B crosses it (4s, after=[gate]); another player can cover both concurrently. That chain needs 7s, not 4s, and covering fire is not added afterward.
Only time the CURRENT requested actions and permitted event, never later short-term-plan goals. Waiting for an uncertain outcome is not confirmation that it happened. Keep any estimated overrun honest; never delete requested actions to make timing fit. The server selects the final length and adds a brief ending hold.
"""


def timing_schema(members, has_event):
    schema = copy.deepcopy(TIMING_SCHEMA)
    actors = [m["user_id"] for m in members]
    if has_event:
        actors.append(EVENT_ACTOR)
    schema["properties"]["steps"]["items"]["properties"]["actors"]["items"]["enum"] = actors
    return schema


@dataclass(frozen=True)
class DurationDecision:
    seconds: int
    tier: str
    reasons: tuple[str, ...]
    source: str = "fallback"
    estimated_seconds: float = 0
    schedule: tuple[dict, ...] = ()
    limited: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": POLICY_VERSION,
            "seconds": self.seconds,
            "tier": self.tier,
            "reasons": list(self.reasons),
            "source": self.source,
            "estimated_seconds": round(self.estimated_seconds, 2),
            "end_hold_seconds": END_HOLD_SECONDS if self.source != "opening" else 0,
            "limited": self.limited,
            "schedule": [dict(step) for step in self.schedule],
        }


# The configured lengths, capped at the longest clip the video model renders (Murea-Fast 20 s,
# the full model 15 s), so a round is never planned longer than it can be filmed.
def opening_seconds() -> int:
    return min(config.COOP_OPENING_SECONDS, video.model().max_seconds)


def action_max_seconds() -> int:
    return min(config.COOP_ACTION_SECONDS, video.model().max_seconds)


def action_min_seconds() -> int:
    return min(config.COOP_ACTION_MIN_SECONDS, action_max_seconds())


def candidate_seconds() -> list[int]:
    # Every 2 s from the minimum, then the maximum: with a 15 s ceiling the old {min, 8, max} tiers
    # filmed a round that needs 9 s as 15 s.
    minimum = action_min_seconds()
    maximum = action_max_seconds()
    return sorted({*range(minimum, maximum, 2), maximum})


def planning_seconds(room: Mapping[str, Any]) -> int:
    # Individual action prefetch must not be staged to a shorter fixed length
    # before a teammate submits a longer action. Final timing uses the full cast.
    return opening_seconds() if room.get("opening") else action_max_seconds()


def policy_signature() -> dict[str, Any]:
    return {
        "version": POLICY_VERSION,
        "opening_seconds": opening_seconds(),
        "action_min_seconds": action_min_seconds(),
        "action_max_seconds": action_max_seconds(),
        "candidate_seconds": candidate_seconds(),
        "end_hold_seconds": END_HOLD_SECONDS,
    }


def _schedule(timing, actors):
    if not isinstance(timing, dict) or not isinstance(timing.get("steps"), list):
        raise ValueError("missing_steps")
    steps = timing["steps"]
    if not 1 <= len(steps) <= 16:
        raise ValueError("invalid_step_count")
    ends, previous, covered, result = {}, {}, set(), []
    for step in steps:
        if not isinstance(step, dict):
            raise ValueError("invalid_step")
        sid, cast, seconds, after = (step.get(k) for k in ("id", "actors", "seconds", "after"))
        if not isinstance(sid, str) or not re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_]{0,59}", sid) or sid in ends:
            raise ValueError("invalid_step_id")
        if (
            not isinstance(cast, list)
            or not cast
            or any(not isinstance(a, str) or a not in actors for a in cast)
            or len(cast) != len(set(cast))
        ):
            raise ValueError("invalid_actors")
        if (
            isinstance(seconds, bool)
            or not isinstance(seconds, (int, float))
            or not math.isfinite(seconds)
            or not 0.5 <= seconds <= 30
        ):
            raise ValueError("invalid_seconds")
        if not isinstance(after, list) or any(not isinstance(dep, str) or dep not in ends for dep in after):
            # Reject dangling edges, forward references and cycles without retrying.
            raise ValueError("invalid_dependencies")
        label = step.get("action")
        if not isinstance(label, str) or not label.strip() or len(label) > 120:
            raise ValueError("invalid_action_label")
        # One actor cannot execute two successive steps in parallel merely
        # because the model forgot their dependency. Simultaneous work is one step.
        deps = set(after) | {previous[a] for a in cast if a in previous}
        start = max((ends[d] for d in deps), default=0)
        ends[sid] = start + seconds
        previous.update({a: sid for a in cast})
        covered.update(cast)
        result.append(
            {
                "id": sid,
                "actors": list(cast),
                "action": label.strip(),
                "seconds": seconds,
                "after": sorted(deps),
                "start": round(start, 2),
                "end": round(ends[sid], 2),
            }
        )
    if covered != actors:
        raise ValueError("incomplete_actors")
    return max(ends.values()), tuple(result)


# Best-effort fallback for older/offline/malformed model responses. Deliberately
# no cast count, overall text length, generic event floor or magic-word penalty.
_VERBS = re.compile(
    r"拔剑|举剑|挥剑|斩击|劈砍|刺击|冲锋|翻滚|绕后|射击|射箭|释放|拉开|推开|攀爬|跳跃|落地|转身|"
    r"\b(?:draws?|raises?|slashes?|strikes?|charges?|rolls?|flanks?|shoots?|releases?|opens?|climbs?|jumps?|lands?|turns?)\b",
    re.I,
)
_LONG_ROUTE = re.compile(
    r"(?:横穿|穿过|跑过|跨越).{0,14}(?:大厅|大殿|庭院|广场)|(?:跑|走|移动).{0,25}(?:尽头|另一端|远端)|"
    r"\bacross\b.{0,25}\b(?:hall|courtyard|room|square)\b|\b(?:far|opposite) end\b",
    re.I,
)
_MOVE = re.compile(
    r"跑|走|移动|靠近|后退|绕|冲锋|\b(?:runs?|walks?|moves?|approaches?|retreats?|flanks?|charges?)\b", re.I
)
_SPEECH = re.compile(
    r'(?:喊|说|问|答|低语|\b(?:say|says|shout|shouts|whisper|whispers)\b).{0,8}[“"「]([^”"」]+)[”"」]', re.I
)
_SERIAL = re.compile(r"然后|随后|接着|继而|再|\bthen\b|\bafter\b", re.I)
_PARALLEL = re.compile(r"同时|一边|\bwhile\b|\bsimultaneously\b", re.I)


def _fallback_seconds(text):
    # Sequential verbs are counted within one actor, never across players.
    count = len(_VERBS.findall(text))
    if _PARALLEL.search(text) and not _SERIAL.search(text):
        count = min(count, 1)
    moving = bool(_MOVE.search(text))
    motion = 7.0 if _LONG_ROUTE.search(text) else 3.0 if moving else 0
    distance = re.search(r"(\d+(?:\.\d+)?)\s*(?:米|meters?\b|metres?\b)", text, re.I)
    if moving and distance:
        speed = 4 if re.search(r"跑|冲锋|\brun|\bcharg", text, re.I) else 1.5
        motion = max(motion, float(distance[1]) / speed)
    seconds = max(2.5, count * 1.5, motion)
    if moving and _SERIAL.search(text):
        seconds = max(seconds, motion + max(0, count - 1) * 1.5)
    spoken = _SPEECH.search(text)
    if spoken:
        line = spoken[1]
        speech = len(re.findall(r"[\u3400-\u9fff]", line)) / 4.5 + len(re.findall(r"[A-Za-z]+", line)) / 2.5
        seconds = seconds + speech if _SERIAL.search(text) else max(seconds, speech)
    return seconds


def decide(room: Mapping[str, Any], timing=None) -> DurationDecision:
    """Freeze duration after the existing shared-shot planning call completes."""
    if room.get("opening"):
        return DurationDecision(
            opening_seconds(),
            "opening",
            ("configured_opening",),
            "opening",
            estimated_seconds=opening_seconds(),
        )
    members = [m for m in room.get("members", []) if str(m.get("action") or "").strip()]
    intro = room.get("intro") or {}
    event = str(intro.get("world_event") or "").strip() if intro.get("status") == "ready" else ""
    actors = {m.get("user_id") for m in members}
    if event:
        actors.add(EVENT_ACTOR)
    schedule, source, reasons = (), "structured", []
    try:
        estimated, schedule = _schedule(timing, actors)
        reasons.append("causal_action_timing")
    except ValueError as exc:
        source = "fallback"
        reasons.append("missing_timing" if timing is None else str(exc))
        estimates = [_fallback_seconds(str(m["action"])) for m in members]
        if event:
            estimates.append(_fallback_seconds(event))
        estimated = max(estimates, default=2.5)
        # A visible cross-player dependency in an unstructured response should
        # not be treated as two independent actions finishing together.
        if any(
            re.search(
                r"等.{1,25}(?:后|再)|待.{1,25}(?:后|再)|\bwait\b.{0,30}\b(?:until|for)\b",
                str(m["action"]),
                re.I,
            )
            for m in members
        ):
            estimated = sum(estimates)
            reasons.append("unresolved_causal_wait")
    required = estimated + END_HOLD_SECONDS
    candidates = candidate_seconds()
    seconds = next((s for s in candidates if s >= required), candidates[-1])
    limited = required > candidates[-1]
    if limited:
        reasons.append("configured_limit")
    tier = "compact" if seconds <= 6 else "standard" if seconds <= 8 else "extended"
    return DurationDecision(seconds, tier, tuple(reasons), source, required, schedule, limited)
