"""Pack video captions without cutting a player's choreography into fragments."""

from __future__ import annotations

import re

from murea_agent.shot import prompt_budget
from murea_agent.director import agent_schema
from murea_agent import reqlog
from murea_agent.coop.scene_plan import ScenePlan, ShotCompiler


def definition(picture: dict) -> str:
    number = picture["n"]
    kind = picture["kind"]
    subject = {"character": "character", "scene": "environment", "creature": "single nonhuman creature"}[kind]
    layout = "single wide scene view" if kind == "scene" else "single full-body view"
    return f"<Subject {number}> is the {subject} shown in <Picture {number}> as a {layout}."


def retention(picture: dict, *, visibility: str = "", level: str = "", detail: str = "") -> str:
    """A complete, concise retention row, including its visible differences."""
    kind = picture["kind"]
    visibility = visibility or ("wide scene view" if kind == "scene" else "full body")
    level = level or ("fully_preserved" if kind == "character" else "partially_preserved")
    if not detail:
        if level == "attribute_transfer":
            detail = "Reference attributes guide the appearance."
        elif level == "weak_reference":
            detail = "Reference loosely guides the appearance."
        else:
            detail = {
                "character": "Visible traits retained; pose follows the action.",
                "scene": "Layout and lighting retained; framing changes.",
                "creature": "Anatomy and colors retained; pose changes.",
            }[kind]
    return f"<Subject {picture['n']}> (appears in [Shot 1] ({visibility})): {level} - {detail}"


_RETENTION_ROW = re.compile(
    r"^<Subject (\d+)> \(appears in \[Shot 1\] \(([^\n()]+)\)\): "
    r"(fully_preserved|partially_preserved|attribute_transfer|weak_reference) - .+$"
)


def validate(six: dict, plan: ScenePlan, pictures: list[dict]) -> None:
    """Validate the choreography itself, not merely the set of Subject tokens."""
    actor_ids = [action.user_id for action in plan.actions]
    subject_order = plan.subject_order or tuple(actor_ids)
    characters = [p["n"] for p in pictures if p["kind"] == "character"]
    if (
        len(set(actor_ids)) != len(actor_ids)
        or set(actor_ids) != set(subject_order)
        or len(subject_order) != len(actor_ids)
        or characters != list(range(1, len(actor_ids) + 1))
        or any(not action.action.strip() for action in plan.actions)
    ):
        raise ValueError("joint shot needs a complete action plan for every player")
    if any(not str(six.get(key, "")).strip() for key in agent_schema.SECTION_ORDER):
        raise ValueError("joint shot is missing the six video prompt fields")
    if any("[compressed]" in str(value) for value in six.values()):
        raise ValueError("joint shot contains truncated text; recompile it from the action plans")
    if not six["summary"].startswith("[reference_generation]") or not six["detailed_description"].startswith(
        "[Shot 1]"
    ):
        raise ValueError("joint shot is missing the video model shot markers")
    expected = {picture["n"] for picture in pictures}
    for field in ("subject_definitions", "retention_analysis", "detailed_description"):
        found = {int(n) for n in re.findall(r"<Subject (\d+)>", six[field])}
        if found != expected:
            raise ValueError(f"joint shot {field} reference subjects do not match")
    bindings = re.findall(r"^<Subject (\d+)>[^\n]*?<Picture (\d+)>", six["subject_definitions"], re.M)
    if bindings != [(str(p["n"]), str(p["n"])) for p in pictures]:
        raise ValueError("joint shot subject-to-image bindings are incomplete or out of order")
    if re.findall(r"<Picture (\d+)>", six["subject_definitions"]) != [str(p["n"]) for p in pictures]:
        raise ValueError("joint shot reference image numbers do not match")
    rows = [_RETENTION_ROW.fullmatch(line) for line in six["retention_analysis"].splitlines()]
    if any(row is None for row in rows) or [row[1] for row in rows] != [str(p["n"]) for p in pictures]:
        raise ValueError("joint shot subject preservation notes are incomplete or out of order")
    # The entire server-compiled, phase-ordered trajectory is mandatory. This
    # catches a deleted middle actor even when the cast list contains its token.
    if ShotCompiler.timeline(plan, caption=True) not in six["detailed_description"]:
        raise ValueError("joint shot motion paths are incomplete; recompile them from the players' plans")


def fit(six: dict, plan: ScenePlan, pictures: list[dict], *, limit: int | None = None) -> dict[str, str]:
    """Remove optional, whole metadata units before touching any motion facts.

    The action timeline (including every actor, route, start/end and world event)
    is immutable. Even the smallest representation retains one complete image
    binding and retention row per subject. Never trade an actor for a budget.
    """
    limit = prompt_budget.PROMPT_MAX_CHARS if limit is None else limit
    out = {key: str(six.get(key) or "").strip() for key in agent_schema.SECTION_ORDER}
    validate(out, plan, pictures)
    before = prompt_budget.prompt_size(prompt_budget.assemble(out))
    changes = []

    def fits():
        return prompt_budget.prompt_size(prompt_budget.assemble(out)) <= limit

    if fits():
        return out
    # Replace repetitive explanations with complete, shorter retention rows.
    # Keep each subject's visibility and retention level unchanged.
    compact_rows = []
    actions = {action.user_id: action for action in plan.actions}
    ordered = [actions[uid] for uid in (plan.subject_order or tuple(actions))]
    for picture, line in zip(pictures, out["retention_analysis"].splitlines()):
        row = _RETENTION_ROW.fullmatch(line)
        detail = ordered[picture["n"] - 1].retention if picture["kind"] == "character" else ""
        short = retention(picture, visibility=row[2], level=row[3], detail=detail)
        compact_rows.append(short if len(short) < len(line) else line)
    out["retention_analysis"] = "\n".join(compact_rows)
    changes.append("retention_explanations")
    if not fits():
        cast = ", ".join(f"<Subject {p['n']}>" for p in pictures if p["kind"] == "character")
        out["summary"] = f"[reference_generation] {cast} act together."
        changes.append("duplicate_action_summary")
    if not fits():
        # Drop optional appearance descriptions one whole Subject block at a
        # time, longest saving first; never splice a binding or layout label.
        lines = out["subject_definitions"].splitlines()
        if len(lines) != len(pictures):
            raise ValueError("joint shot subject definitions cannot be grouped by reference image")
        replacements = [definition(picture) for picture in pictures]
        order = sorted(range(len(lines)), key=lambda i: len(lines[i]) - len(replacements[i]), reverse=True)
        for i in order:
            if len(lines[i]) <= len(replacements[i]):
                continue
            lines[i] = replacements[i]
            out["subject_definitions"] = "\n".join(lines)
            changes.append(f"subject_{pictures[i]['n']}_appearance")
            if fits():
                break
    if not fits():
        changes.extend(_drop_optional_prose(out, plan, pictures, fits))
    validate(out, plan, pictures)
    after = prompt_budget.prompt_size(prompt_budget.assemble(out))
    reqlog.record(
        "coop_caption_budget",
        ok=after <= limit,
        before_chars=before,
        prompt_chars=after,
        limit=limit,
        removed_metadata=changes,
        protected_actors=len(plan.actions),
        protected_timeline_chars=len(ShotCompiler.timeline(plan, caption=True)),
    )
    if after > limit:
        # Arbitrarily long indispensable instructions cannot fit a finite
        # prompt budget. Preserve them for the caller instead of silently
        # producing a successful video with a missing player's action.
        raise ValueError(
            f"binding the full actions to the references needs {after} characters, over the video model limit of {limit}; "
            "动作已保留，请减少本轮动作步骤"
        )
    return out


def _drop_optional_prose(out: dict[str, str], plan: ScenePlan, pictures: list[dict], fits) -> list[str]:
    """Shorten the shot prose by whole optional units until `fits()`; returns what was removed.

    Sound goes first, as a complete section value rather than a broken dialogue fragment; then
    the framing and style around the timeline. The exact complete timeline, environment binding
    and cast remain present and validated.
    """
    changes = []
    out["overall_soundscape"] = "Natural ambient sound. No speech or narration."
    out["non_diegetic_music"] = "N/A"
    changes.append("optional_sound")
    if fits():
        return changes
    scene = next((p["n"] for p in pictures if p["kind"] == "scene"), 0)
    creature = next((p["n"] for p in pictures if p["kind"] == "creature"), 0)
    brief = ShotCompiler.compile_timeline(
        plan,
        camera="A continuous wide shot.",
        environment=f"Inside <Subject {scene}>." if scene else plan.scene.rstrip(" .") + ".",
        cast_count=f"Exactly {len(plan.actions)} people share the frame.",
        creature_clause=f"<Subject {creature}> is also present." if creature else "",
    )
    if len(brief) < len(out["detailed_description"]):
        out["detailed_description"] = brief
        changes.append("optional_shot_framing_and_style")
    return changes


def fit_continuation(six: dict, plan: ScenePlan, pictures: list[dict], *, limit: int, sent) -> dict[str, str]:
    """Fit the reference-free prose a continuation sends (`sent(six)`) into `limit`.

    A video model that cannot take references with a continuation (Murea-Fast) gets only the
    shot prose; the full prompt is the fallback. Only the same optional prose units as `fit`
    are removed, never an actor's motion; when even that is too long the round fails with the
    same request to shorten the actions.
    """
    out = dict(six)

    def fits():
        return prompt_budget.prompt_size(sent(out)) <= limit

    if fits():
        return out
    before = prompt_budget.prompt_size(sent(out))
    changes = _drop_optional_prose(out, plan, pictures, fits)
    validate(out, plan, pictures)
    after = prompt_budget.prompt_size(sent(out))
    reqlog.record(
        "coop_caption_budget",
        ok=after <= limit,
        sent="reference_free_continuation",
        before_chars=before,
        prompt_chars=after,
        limit=limit,
        removed_metadata=changes,
        protected_actors=len(plan.actions),
    )
    if after > limit:
        raise ValueError(
            f"the continuation needs {after} characters for the full actions, over the video model limit of {limit}; "
            "动作已保留，请减少本轮动作步骤"
        )
    return out
