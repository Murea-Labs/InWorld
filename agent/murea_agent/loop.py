"""The agent's turn loop: perceive -> plan -> check -> act -> commit.

`turn()` plays one beat of the story; `regenerate()` re-renders the latest clip.
The model makes the decisions (`murea_agent.director`); this module owns the control
flow, so every beat runs the same, traceable sequence of steps (`murea_agent.events`).
"""

from __future__ import annotations

import re
import time
from contextlib import ExitStack
from concurrent import futures
import uuid
from pathlib import Path
from typing import Any
from collections.abc import Callable

from murea_agent import config
from murea_agent.director import agent_delta
from murea_agent.providers import imagegen, llm, video
from murea_agent import reqlog
from murea_agent.memory import voices
from murea_agent.memory.world import Node, World
from murea_agent.memory.world_media import media_path
from murea_agent.util import _Clock, _no_event, _noop, _say, cap
from murea_agent.director.agent_schema import (
    _MODEL_SECTIONS,
    _MOTION_AUDIT,
)
from murea_agent.guards.motion import (
    _clean_motion_text,
    _motion_audit_error,
    _motion_plan_errors,
    _normalize_motion_plan,
    _repair_motion_plan,
)
from murea_agent.guards.action_sequence import (
    _goal_relation_instruction,
    _advance_completed_transition,
    _annotate_sequence_context,
    _reconcile_dialogue_motion,
    _stage_declared_places,
)
from murea_agent.shot.definitions import (
    _PendingImage,
    _english_asset_prompt,
)
from murea_agent.director.narration import (
    _CAPTION_PLANNING_LANGUAGE,
    _apply_action_sequence,
    _current_segment_reply,
    _has_split_planning_reply,
    _history_beat,
    _history_line,
    _recommend_index,
    _sanitize_intermediate_directive,
    _three_options,
)
from murea_agent.shot.video_prompt import (
    _build_retention,
    _compose_definitions,
    _ending_of,
    _ensure_prompt_english,
    _learn_definitions,
    _lock_bound_subject_tokens,
    _lock_motion_plan,
    _lock_object_reference_constraints,
    _lock_required_ref_assets,
    _lock_visual_style,
    _plain_continuation_from_merged,
    _reference_token_error,
    _repair_reference_tokens,
    _rewrite,
    _six_from_merged,
    _substitute_refs,
)
from murea_agent.shot.dialogue import (
    _has_dialogue,
    _lock_dialogue_language,
    _lock_dialogue_performance,
    _lock_dialogue_speakers,
)
from murea_agent.shot.prompt_budget import (
    _fit_video_prompt,
    assemble,
)
from murea_agent.shot.soundscape import (
    SOUNDSCAPE_LOCKED,
    _ensure_soundscape,
    _learn_soundscape,
    _lock_soundscape,
)
from murea_agent.shot.beat_assets import (
    MAX_CHARACTER_REFS,
    _apply_asset_mentions,
    _asset_mentions,
    _audio_refs,
    _collect_pending,
    _draw_pending,
    _drop_warm,
    _inject_implicit_player,
    _lock_audio_refs,
    _pictures_and_refs,
    _player_actor_name,
    _prepare_subjects,
    _prune_passive_object_pending,
    _queue_missing_focal_object_refs,
    _six_section_speakers,
    _warm_images,
)
from murea_agent.tools.frames import _grab_frames
from murea_agent.critic import _as_dict
from murea_agent import events
from murea_agent.events import Cancelled, Emitter, bail
from murea_agent.director import director, prefetch
from murea_agent.guards import replan


def _drop_drawn(world: World, pending: list[_PendingImage]) -> None:
    """Delete the reference images drawn in this beat.

    The turn is not saved (`world.save` comes last), but the images are already in data/refs; left
    there they become orphans no world refers to and `delete_world` never removes. Both errors and
    user cancels go through here.
    """
    for pend in pending:
        store = world.subjects if pend.kind == "subject" else world.places
        fn = (store.get(pend.key) or {}).get("ref")
        if fn:
            (config.REFS / fn).unlink(missing_ok=True)


def _sketches(world: World, pending: list[_PendingImage]) -> list[dict[str, Any]]:
    """Reference images drawn in this beat: the minimal fields the UI needs to render them."""
    out: list[dict[str, Any]] = []
    for p in pending:
        store = world.subjects if p.kind == "subject" else world.places
        rec = store.get(p.key) or {}
        if not rec.get("ref"):
            continue
        out.append(
            {
                "kind": p.kind,
                "id": p.key,  # opening a card looks up the asset by id
                "entity": rec.get("kind", "人物") if p.kind == "subject" else "场景",
                "name": rec.get("name", p.key) if p.kind == "subject" else p.key,
                "brief": p.brief or rec.get("brief", ""),
                "ref": rec["ref"],
            }
        )
    return out


def _drawing_message(world: World, pending: list[_PendingImage]) -> str:
    def label(p: _PendingImage) -> str:
        if p.kind != "subject":
            return _say(f"新场景「{p.key}」", f'new place "{p.key}"')
        name = world.subjects[p.key]["name"]
        if p.asset_kind == "object":
            return _say(f"新物品「{name}」", f'new item "{name}"')
        return _say(f"新角色「{name}」", f'new character "{name}"')

    items = [label(p) for p in pending]
    return _say(f"画{'、'.join(items)}（和写 prompt 并行）", "Drawing " + ", ".join(items))


def _video_watch(
    progress: Callable[[str, str], None],
    label: str,
    cancelled: Callable[[], bool] | None = None,
) -> Callable[[dict[str, Any]], None]:
    """Turn the provider's job status into a readable message while waiting.

    Why: on a shared video service much of a beat can be queueing (`created_at` → `started_at`)
    rather than rendering. With a single "rendering" message users blame the model for the whole
    wait; reporting them separately makes the wait explainable.

    When the user presses stop, the video client calls this on its next poll. This callback used to
    only update the text, so the video job kept the global lock and later requests waited forever.
    """
    t0 = time.time()

    def watch(job: dict[str, Any]) -> None:
        if cancelled is not None and cancelled():
            raise Cancelled("cancelled by the user")
        secs = int(time.time() - t0)
        if job.get("started_at"):
            progress("generate", _say(f"{label} 正在渲染（已 {secs}s）", f"{label} is rendering ({secs}s)"))
        else:
            progress(
                "generate",
                _say(f"排队中，前面还有任务（已等 {secs}s）", f"Queued behind other jobs ({secs}s)"),
            )

    return watch


# ---------------------------------------------------------------- turn


def _continuous_scene_boundary(
    world: World,
    source: str,
    destination: str,
    *,
    new_destination: bool = False,
) -> bool:
    """Reachable main scenes need a cut; only shared subspaces share geometry."""
    return world.main_scene_for(source) == world.main_scene_for(destination) and (
        new_destination or destination in world.adjacency().get(source, [])
    )


def _walked_over(
    world: World,
    source: str,
    destination: str,
    *,
    mode: str,
    transition: bool,
    new_destination: bool,
) -> bool:
    """Whether this beat is "walking over": walking keeps the previous latent instead of a new shot.

    If the director fills transition it knows this is movement, but it often leaves it out, and the
    beat degrades to a hard cut where the person teleports. When the destination is an adjacent
    sub-space that already has an image, walking is the only way between them, so do not wait for
    it. A new place without a reference image still needs an explicit statement, so nobody is moved
    into an imagined space.
    """
    if mode != "cut" or not source or source == destination:
        return False
    if not (transition or not new_destination):
        return False
    # Adjacency means reachable, not visually continuous. Distinct main scenes
    # need a fresh referenced shot; reusing courtyard latents made the bridge
    # materialize inside that courtyard. Subspaces of one main scene may still
    # share a continuous boundary crossing.
    return _continuous_scene_boundary(world, source, destination, new_destination=new_destination)


def turn(
    world: World,
    user_text: str,
    *,
    progress: Callable[[str, str], None] = _noop,
    notify: Callable[[dict[str, Any]], None] = _no_event,
    cancelled: Callable[[], bool] = lambda: False,
    duration: float | None = None,
    origin: str = "turn",
    mentions: list[str] | None = None,
    request_id: str = "",
) -> Node:
    """Play one beat: perceive -> plan -> check -> act -> commit.

    `notify` receives every event (agent steps, the early reply, drawn
    references, the action sequence); `progress(stage, detail)` is the one-line
    status. Raises `Cancelled` when `cancelled()` turns true before commit.
    """
    return TurnLoop(
        world,
        user_text,
        events=Emitter(notify, progress),
        cancelled=cancelled,
        duration=duration,
        origin=origin,
        mentions=mentions,
        request_id=request_id,
    ).run()


# Words that narrate what happens after this clip ("then", "finally").
_LATER_STEPS = re.compile(r"然后|接着|随后|最后|紧接着|\bthen\b|\bfinally\b|\bafterwards\b", re.I)


def _sends_full_prompt(vmodel, continue_from) -> bool:
    """This clip goes out with the full six-section prompt (references), not reference-free prose."""
    return not continue_from or vmodel.references_on_continuation


def _continuation_prompt(vmodel, six, pictures, world, prompt, continue_from):
    """The prompt for this clip, and extra arguments for `video.generate`.

    A provider that cannot send reference images with a continuation (Murea) gets the shot as
    reference-free prose; the full prompt goes along for when the parent clip is gone.

    Here `six` has not been fitted to the budget (see `_finish_prompt`): fitting the full prompt
    first would trim the end of the shot out of the prose that is actually sent. Both prompts are
    fitted here by trimming only; the fallback is rarely used, so it gets no model edit.
    """
    if _sends_full_prompt(vmodel, continue_from):
        return prompt, {}
    limit = vmodel.max_prompt_chars
    plain = assemble(
        _fit_video_prompt(_plain_continuation_from_merged(six, pictures, world), limit, compress=False)
    )
    try:
        fallback = assemble(_fit_video_prompt(six, limit, compress=False))
    except ValueError:
        # Too many references to fit: still better than failing a beat that will most likely
        # continue normally.
        fallback = prompt
    return plain, {"continuation_fallback_prompt": fallback}


class TurnLoop:
    """The agent's loop for one beat.

    The control flow is fixed; the model only makes the decisions:

        perceive  snapshot + last frames of the previous clip
        plan      the director's one structured call (or a prefetched plan)
        check     guards validate the plan; errors go back to the director
        act       stage assets, draw references, write the prompt, render
        commit    apply the world changes, save the clip with its trace

    Every stage is a traced step (`murea_agent.events`). State that later stages need
    is kept on the instance, so each stage reads like a short function.
    """

    def __init__(
        self,
        world: World,
        user_text: str,
        *,
        events: Emitter,
        cancelled: Callable[[], bool] = lambda: False,
        duration: float | None = None,
        origin: str = "turn",
        mentions: list[str] | None = None,
        request_id: str = "",
    ) -> None:
        self.world = world
        self.user_text = user_text
        self.events = events
        self.progress = events.progress
        self.cancelled = cancelled
        self.duration = duration
        self.origin = origin
        self.mentions = mentions
        self.request_id = request_id
        self.clock = _Clock()
        self.pending: list[_PendingImage] = []

    def run(self) -> Node:
        with events.active(self.events):  # tools called by the director report here
            self.perceive()
            self.plan()
            self.check()
            bail(self.cancelled)  # nothing drawn or rendered yet
            self.stage_assets()
            self.write_prompt()
            self.render()
            return self.commit()

    # ---- perceive ------------------------------------------------------------

    def perceive(self) -> None:
        world = self.world
        with self.events.step("perceive", "observe", _say("观察世界", "Looking at the world")) as step:
            self.state_before = world.dynamic_state_snapshot()
            self.quest_before = world.quest_state_snapshot()
            self.asset_mentions = _asset_mentions(world, self.user_text, self.mentions)
            self.head = world.head_node
            reqlog.record(
                "turn", origin=self.origin, user_text=self.user_text, head=world.head, duration=self.duration
            )
            step.data.update(
                here=world.here,
                objective=world.objective,
                beats=len(world.lineage()),
                frames=len(self.head.frames or []) if self.head else 0,
                mentions=[m["name"] for m in self.asset_mentions],
            )

    # ---- plan ----------------------------------------------------------------

    def plan(self) -> None:
        world, text = self.world, self.user_text
        self.progress("gm", _say("推演这一拍发生什么", "Planning what happens next"))
        # `reply` comes first in the schema, but on multi-step goals the model may
        # announce later results. Partial output is only cached; the reply is sent
        # after the action plan has been checked.
        self.early_say = ""

        def say_early(part: dict[str, Any]) -> None:
            reply = str(part.get("reply") or "").strip()
            if reply and not self.early_say:
                self.early_say = reply
            # New subjects are known as soon as `new_subjects` is written: start
            # drawing them before the rest of the JSON arrives (pure speed-up).
            try:
                _warm_images(world, part)
            except Exception:
                pass

        self.say_early = say_early
        with self.events.step("plan", "director", _say("导演规划这一拍", "Director plans the beat")) as step:
            started = time.monotonic()
            d = prefetch._take_prefetched(
                world,
                text,
                self.duration,
                cancelled=self.cancelled,
                on_wait=lambda: self.progress(
                    "gm", _say("该选项正在预推演，接着等待已有结果", "Already being planned; waiting for it")
                ),
            )
            waited = time.monotonic() - started
            if waited >= 0.01:
                self.clock.marks["gm_prefetch_wait"] = round(waited, 2)
            reqlog.record("prefetch_result", hit=d is not None, wait_seconds=round(waited, 3))
            if d is not None:
                self.progress("gm", _say("这一拍已预推演，直接开演", "Already planned; starting"))
                self.clock.marks["gm"] = 0.0
                self.late_fut = None
                say_early(d)
            else:
                with self.clock.stage("gm"):
                    d, self.late_fut = director._decide_early(
                        world,
                        text,
                        say_early,
                        duration=self.duration,
                        asset_mentions=self.asset_mentions,
                    )
            self.d = d
            step.data.update(
                prefetched=d is not None and self.clock.marks.get("gm") == 0.0,
                beat=str(d.get("beat") or ""),
                place=str(d.get("place") or ""),
                mode=str(d.get("mode") or ""),
                subjects=list(d.get("subjects") or []),
                new_subjects=[
                    str(s.get("name") or "") for s in d.get("new_subjects") or [] if isinstance(s, dict)
                ],
                dialogue=str(d.get("dialogue") or ""),
            )

    # ---- check ---------------------------------------------------------------

    def check(self) -> None:
        world, text, d = self.world, self.user_text, self.d
        with self.events.step(
            "check", "action_plan", _say("检查动作计划", "Checking the action plan")
        ) as step:
            d, replanned = replan._prepare_action_plan(
                world,
                text,
                d,
                self.say_early,
                duration=self.duration,
                asset_mentions=self.asset_mentions,
                clock=self.clock,
                progress=self.progress,
                is_continuation=(self.origin == "auto"),
            )
            if replanned:
                # A discarded first call's late fields must not overwrite the new plan.
                self.late_fut = None
            step.data.update(replanned=replanned, repairs=list(d.get("_server_repairs") or []))

        with self.events.step(
            "check", "motion_plan", _say("检查物理关系", "Checking the physical plan")
        ) as step:
            # Keep the model's own plan: the prompt audit compares the video prompt
            # with what the model itself planned, not with the server's repairs.
            self.model_motion_plan = _normalize_motion_plan(d.get("motion_plan"))
            d["motion_plan"] = self._repaired_motion_plan(d)
            self.required_subject_ids = _apply_asset_mentions(world, d, self.asset_mentions)
            self.required_asset_ids = {item["id"] for item in self.asset_mentions}
            # The director decided whether the player's body is visible; registering
            # it is deterministic and happens before references are assembled.
            _inject_implicit_player(world, d)
            motion_plan = self._repaired_motion_plan(d)
            errors = _motion_plan_errors(motion_plan)
            if errors:
                # A clip costs tens of seconds and real money: stop on a plan that
                # mixes, say, approaching and crossing, instead of filming it.
                raise ValueError("Invalid motion plan: " + "; ".join(errors))
            d["motion_plan"] = self.motion_plan = motion_plan
            step.data.update(
                {k: motion_plan.get(k, "") for k in ("phase", "relation_change", "target", "framing")}
            )

        with self.events.step("check", "action_sequence", _say("拆分动作", "Splitting the action")) as step:
            self._sequence(d)
            step.data.update(
                steps=len(self.sequence_steps),
                seconds=self.generation_seconds,
                goal=self.sequence.get("goal") or text,
            )
        self.d = d
        if self.say:
            self.events.notify({"type": "say", "text": self.say})

    def _repaired_motion_plan(self, d: dict[str, Any]) -> dict[str, Any]:
        plan = _normalize_motion_plan(d.get("motion_plan"))
        plan = _advance_completed_transition(self.world, d, plan, self.user_text)
        plan = _reconcile_dialogue_motion(self.world, d, plan, self.user_text)
        return _repair_motion_plan(plan, d)

    def _sequence(self, d: dict[str, Any]) -> None:
        """Run step 1 of the director's action sequence; queue the rest."""
        world, text, head = self.world, self.user_text, self.head
        player = _player_actor_name(world)
        sequence = _apply_action_sequence(
            d,
            d.get("beat") or text,
            self.motion_plan,
            request_text=text,
            segment_seconds=director._segment_seconds(world, self.duration),
            is_continuation=(self.origin == "auto"),
            # The queue pinned on the previous clip decides during auto-continue.
            carried_plan=list(getattr(head, "plan", None) or []),
            player_actor=player,
        )
        deferred = _annotate_sequence_context(world, d, sequence, self.motion_plan, text)
        _sanitize_intermediate_directive(d, sequence, self.motion_plan, player)
        self.generation_seconds = director._generation_seconds(world, d, self.duration)
        d["duration_seconds"] = self.generation_seconds
        # The caption describes only the current step: never future characters
        # or the internal split notes.
        if deferred and any(name in str(d.get("reply") or "") for name in deferred):
            d["reply"] = _current_segment_reply(sequence, self.motion_plan, d.get("reply"), player)
        if _has_split_planning_reply(d.get("reply")):
            d["reply"] = _current_segment_reply(sequence, self.motion_plan, d.get("reply"), player)
        if len(sequence.get("steps") or []) > 1 and _LATER_STEPS.search(str(d.get("reply") or "")):
            # A split action whose caption already tells the later clips.
            d["reply"] = _current_segment_reply(sequence, self.motion_plan, "", player)
        self.say = str(d.get("reply") or self.early_say).strip()
        self.sequence, self.sequence_steps = sequence, sequence.get("steps") or []
        if len(self.sequence_steps) > 1:
            contents: list[str] = []
            for index, step in enumerate(self.sequence_steps):
                content = self.say if index == 0 and self.say else _clean_motion_text(step.get("instruction"))
                if not content or _CAPTION_PLANNING_LANGUAGE.search(content):
                    relation = _clean_motion_text(step.get("relation_change")).lower()
                    content = _goal_relation_instruction(
                        relation, _say("完成这一段动作", "Finish this action")
                    )
                contents.append(content)
            self.events.notify(
                {
                    "type": "sequence",
                    "goal": sequence.get("goal") or text,
                    "current": 1,
                    "total": len(self.sequence_steps),
                    "segments": [{"index": i + 1, "content": c} for i, c in enumerate(contents)],
                }
            )
            self.progress(
                "gm",
                _say(
                    f"按动作时长拆成 {len(self.sequence_steps)} 段，先完成第 1 段",
                    f"Split into {len(self.sequence_steps)} clips; playing the first",
                ),
            )

    # ---- act: assets -----------------------------------------------------------

    def stage_assets(self) -> None:
        """Decide the place, register new assets and pick the references."""
        world, d, head = self.world, self.d, self.head
        with self.events.step(
            "act", "stage_assets", _say("准备场景和角色", "Staging place and cast")
        ) as step:
            mode = d.get("mode", "continue")
            place = (d.get("place") or world.here or _say("未命名场景", "Unnamed place")).strip()
            # `place` decides, not `mode`: the two can disagree, and walking to the
            # door and back is one room, so it must stay a continuation.
            if head is None or head.place != place:
                mode = "cut"
            else:
                mode = "continue"
            pending = self.pending
            place_is_new = place not in world.places
            if place_is_new:
                img_prompt = (d.get("place_image_prompt") or "").strip() or (
                    f"A wide establishing shot of {place}, showing its spatial layout, "
                    "architecture, surfaces, and primary light sources."
                )
                img_prompt = _english_asset_prompt(
                    img_prompt,
                    fallback="A wide establishing shot of the location, showing spatial "
                    "layout, surfaces, and primary light sources.",
                )
                brief = cap(d.get("place_brief"), 60)
                world.add_place(place, "", img_prompt, str(d.get("main_scene") or place), brief=brief)
                pending.append(_PendingImage("place", place, img_prompt, world.visual_style, brief))
                mode = "cut"  # a new place cannot be a continuation
            # Future places the director declared are drawn ahead of time.
            _stage_declared_places(world, d, pending)
            subject_ids = _prepare_subjects(world, d, pending, set(d.get("subjects") or []))
            d["subjects"] = subject_ids
            # Apply the end-of-beat state before writing the prompt, so the clip
            # shows the same outfit, condition and lighting the next plan reads.
            world.apply_dynamic_updates(
                d.get("state_updates") or {}, current_place=place, visible_subject_ids=subject_ids
            )
            self.dynamic_state = world.dynamic_state_for_prompt(place, subject_ids)
            self.walked_over = head is not None and _walked_over(
                world,
                head.place,
                place,
                mode=mode,
                transition=bool(d.get("transition")),
                new_destination=place_is_new,
            )
            self.continues = (mode == "continue" and head is not None) or self.walked_over
            motion_plan = self.motion_plan
            self.focus_text = " ".join(
                str(value).strip()
                for value in (
                    self.user_text,
                    d.get("beat"),
                    d.get("reply"),
                    d.get("dialogue"),
                    d.get("carry"),
                    d.get("prompt"),
                    motion_plan.get("instrument"),
                    motion_plan.get("target"),
                    motion_plan.get("start_state"),
                    motion_plan.get("path"),
                    motion_plan.get("contact_points"),
                    motion_plan.get("end_state"),
                )
                if str(value or "").strip()
            )
            self.action_step = self.sequence_steps[0] if self.sequence_steps else None
            _prune_passive_object_pending(
                world,
                pending,
                motion_plan=motion_plan,
                focus_text=self.focus_text,
                action_step=self.action_step,
            )
            _queue_missing_focal_object_refs(
                world,
                subject_ids,
                pending,
                motion_plan=motion_plan,
                focus_text=self.focus_text,
                action_step=self.action_step,
            )
            self.mode, self.place, self.subject_ids = mode, place, subject_ids
            self._pick_references()
            step.data.update(
                place=place,
                mode=mode,
                new_place=place_is_new,
                continues=self.continues,
                cast=[p["name"] for p in self.pictures if p["kind"] != "scene"],
                to_draw=[p.key if p.kind == "place" else world.subjects[p.key]["name"] for p in pending],
                dropped=list(self.dropped),
            )

    def _pick_references(self) -> None:
        self.pictures, self.ref_files, self.dropped = _pictures_and_refs(
            self.world,
            self.place,
            self.subject_ids,
            required_subject_ids=self.required_subject_ids,
            motion_plan=self.motion_plan,
            focus_text=self.focus_text,
            action_step=self.action_step,
        )

    # ---- act: references + prompt ----------------------------------------------

    def write_prompt(self) -> None:
        """Draw new references in parallel while the video prompt is written."""
        world, pending = self.world, self.pending
        vmodel = video.model(world.video_model)
        self.vmodel = vmodel
        pool = futures.ThreadPoolExecutor(max_workers=max(1, len(pending))) if pending else None
        jobs = _draw_pending(pending, pool) if pool else {}
        self.drew_for = 0.0
        self.sketches: list[dict[str, Any]] = []
        with ExitStack() as stack:
            if pool:
                stack.callback(pool.shutdown, wait=False)
            # Unclaimed warm drawings must not be picked up by a later beat.
            stack.callback(_drop_warm)
            if pending:
                names = [(world.subjects[p.key]["name"] if p.kind == "subject" else p.key) for p in pending]
                # Drawing is the one silent multi-second wait: say what is drawn.
                self.progress("image", _drawing_message(world, pending))
                drawing = stack.enter_context(
                    self.events.step("act", "draw_references", _say("画参考图", "Drawing reference images"))
                )
                drawing.data.update(items=names)
            with self.events.step(
                "act",
                "write_prompt",
                _say(f"给 {vmodel.label} 写 prompt", f"Writing the prompt for {vmodel.label}"),
            ) as step:
                self.progress(
                    "rewrite", _say(f"给 {vmodel.label} 写 prompt", f"Writing the prompt for {vmodel.label}")
                )
                six, source = self._six_sections()
                step.data.update(source=source)
            if pending:
                self.drew_for = _collect_pending(world, pending, jobs)
                # Assets now point at real files; rebuild the list from the store
                # so picture order and filenames cannot drift apart.
                self._pick_references()
                # The drawn references are facts of this world from now on; show them.
                self.sketches = _sketches(world, pending)
                self.events.notify({"type": "sketch", "items": self.sketches})
        with self.events.step("act", "lock_prompt", _say("锁定一致性", "Locking consistency")) as step:
            self.prompt = self._finish_prompt(six)
            step.data.update(
                chars=len(self.prompt),
                audios=len(self.audio_bind),
                repairs=list(self.d.get("_server_repairs") or []),
            )

    def _six_sections(self) -> tuple[dict[str, str], str]:
        """The director's video prompt, or a separate rewrite when it is unusable."""
        world, d, head, pictures = self.world, self.d, self.head, self.pictures
        motion_plan = self.motion_plan
        locked_defs, needs_def = _compose_definitions(world, pictures)
        source = "director"
        merged = _six_from_merged(d, definitions_locked=bool(locked_defs) and not needs_def)
        if merged is None:
            reqlog.record(
                "merged_fallback",
                mismatch="merged output has no usable video sections",
                missing_sections=[
                    k
                    for k in _MODEL_SECTIONS
                    if k not in d and (k != "subject_definitions" or needs_def or not locked_defs)
                ],
                missing_asset_definitions=[pic["id"] for pic in needs_def],
                missing_shot=str(d.get("detailed_description") or "").strip() in ("", SOUNDSCAPE_LOCKED),
            )
        else:
            six, _unknown = _substitute_refs(merged, pictures, world)
            # Compare with the model's own plan, not the server-repaired one.
            audit = _motion_audit_error(
                {**merged, **{k: d.get(k, "") for k in _MOTION_AUDIT}}, self.model_motion_plan
            )
            if not audit:
                # Out-of-range reference numbers are deterministic to repair.
                six, fixes = _repair_reference_tokens(six, pictures)
                for note in fixes:
                    d.setdefault("_server_repairs", []).append(note)
                audit = _reference_token_error(six, pictures)
            if audit:
                # The prompt contradicts the plan: rewrite only the prompt, which is
                # cheaper than re-planning the whole beat.
                self.progress(
                    "rewrite",
                    _say("物理审计不一致，单独重写这一段", "Rewriting the prompt to match the motion plan"),
                )
                reqlog.record(
                    "merged_fallback",
                    mismatch=audit,
                    phase=motion_plan.get("phase"),
                    relation=motion_plan.get("relation_change"),
                )
                merged = None
            else:
                self.clock.marks["rewrite"] = 0.0
        if merged is None:
            source = "rewriter"
            with self.clock.stage("rewrite"):
                six = _rewrite(
                    pictures=pictures,
                    place=self.place,
                    beat=d.get("beat", self.user_text),
                    dialogue=d.get("dialogue", ""),
                    continues=self.continues,
                    previous_beat=head.beat if head else "",
                    setting=world.setting,
                    visual_style=world.visual_style,
                    previous_shot=_ending_of(head) if self.continues else "",
                    carry=(head.carry if head else ""),
                    needs_definition=needs_def,
                    dynamic_state=self.dynamic_state,
                    walked_from=(head.place if self.walked_over and head else ""),
                    motion_plan=motion_plan,
                )
        # Old caches and hand-written prompts may still carry `<Ref:Name>`.
        six, _unknown = _substitute_refs(six, pictures, world)
        six["subject_definitions"] = "\n".join(
            x for x in (locked_defs, six.get("subject_definitions", "")) if x.strip()
        )
        six = _lock_visual_style(six, world.visual_style)
        six = _lock_bound_subject_tokens(six, pictures)
        six = _lock_motion_plan(six, motion_plan, pictures=pictures)
        six = _lock_required_ref_assets(six, pictures, self.required_asset_ids)
        six = _lock_object_reference_constraints(
            six, pictures, world, motion_plan, carry=(head.carry if head else ""), focus_text=self.focus_text
        )
        return _lock_dialogue_language(six), source

    def _finish_prompt(self, six: dict[str, str]) -> str:
        """Deterministic locks that keep the world consistent, then assemble."""
        world, d, head, pictures = self.world, self.d, self.head, self.pictures
        dialogue = d.get("dialogue", "")
        _learn_definitions(world, pictures, six)
        six = _lock_dialogue_speakers(six, dialogue, world, pictures)
        six = _lock_dialogue_performance(six, dialogue)
        # A line the director wrote but the prompt dropped makes the model invent speech.
        self.dropped_line = ""
        if dialogue.strip() and not _has_dialogue(six):
            self.dropped_line = _say(
                "这一段的台词没写进 prompt，画面里的说话声可能对不上",
                "The line was left out of the prompt; speech in the clip may not match",
            )
            self.progress("rewrite", self.dropped_line)
        # Voice samples are bound deterministically, every beat that has speech:
        # continuation keeps the picture, but audio is generated anew each time.
        self.audio_paths, self.audio_bind = _audio_refs(
            world, self.subject_ids, dialogue, also_speaking=_six_section_speakers(six)
        )
        if self.audio_bind:
            six = _lock_audio_refs(six, self.audio_bind, pictures)
        six = {
            **six,
            "retention_analysis": _build_retention(
                pictures, world, self.motion_plan, carry=(head.carry if head else "")
            ),
        }
        _ensure_soundscape(six, dialogue)
        # Same place, same ambient bed; the first visit locks it.
        six = _lock_soundscape(six, world, self.place, dialogue)
        _learn_soundscape(world, self.place, six)
        six, fixes = _repair_reference_tokens(six, pictures)
        for note in fixes:
            d.setdefault("_server_repairs", []).append(note)
        token_error = _reference_token_error(six, pictures)
        if token_error:
            _drop_drawn(world, self.pending)
            raise RuntimeError(f"Invalid reference numbering in the video prompt: {token_error}")
        six.update(_ensure_prompt_english(six))
        # Fit only a prompt that is sent as is. A reference-free continuation is fitted when it is
        # compiled (`_continuation_prompt`); compressing the full prompt here used to cost up to
        # three model calls and cut the end of the shot out of the prose that was actually sent.
        if _sends_full_prompt(self.vmodel, self.continue_from):
            try:
                six = _fit_video_prompt(six, self.vmodel.max_prompt_chars)
            except Exception:
                _drop_drawn(world, self.pending)
                raise
        self.six = six
        return assemble(six)

    @property
    def continue_from(self) -> str | None:
        """The previous clip this beat continues from (its artifact id), if it continues."""
        return self.head.artifact_id if self.continues and self.head else None

    # ---- act: render -----------------------------------------------------------

    def render(self) -> None:
        world, head, vmodel = self.world, self.head, self.vmodel
        if self.dropped:
            self.progress(
                "image",
                _say(
                    f"同框超过 {MAX_CHARACTER_REFS} 个主体，这一段没带上 {'、'.join(self.dropped)} 的参考图",
                    f"More than {MAX_CHARACTER_REFS} characters in shot; left out the references "
                    f"of {', '.join(self.dropped)}",
                ),
            )
        # The last cheap exit: the video request has not been sent.
        if self.cancelled():
            _drop_drawn(world, self.pending)
            bail(self.cancelled)
        reqlog.record(
            "agent_ready",
            seconds=round(self.clock.total(), 3),
            timings=dict(self.clock.marks),
            prompt_chars=len(self.prompt),
        )

        # ─── VIDEO-SUBMIT-BOUNDARY ───────────────────────────────────────────
        # Every `d` field read above must be written before the video is
        # submitted, i.e. never be in AGENT_LATE_FIELDS (test_agent_early.py
        # slices the source at this marker).
        self.progress("generate", _say(f"{vmodel.label} 生成画面", f"{vmodel.label} is generating the clip"))
        continue_from = self.continue_from
        prompt, fallback = _continuation_prompt(
            vmodel, self.six, self.pictures, world, self.prompt, continue_from
        )
        with self.events.step(
            "act", "render_clip", _say(f"{vmodel.label} 拍摄", f"Rendering with {vmodel.label}")
        ) as step:
            step.data.update(
                model=vmodel.id,
                seconds=self.generation_seconds,
                references=len(self.ref_files),
                voices=len(self.audio_paths),
                continues=bool(self.continues and head),
            )
            try:
                with self.clock.stage("video"):
                    self.gen = video.generate(
                        prompt,
                        [media_path("refs", f) for f in self.ref_files],
                        ref_audios=self.audio_paths,
                        continue_from=continue_from,
                        **fallback,
                        duration=self.generation_seconds,
                        model_id=world.video_model,
                        on_status=_video_watch(self.progress, vmodel.label, self.cancelled),
                    )
            except Exception:
                _drop_drawn(world, self.pending)
                raise
            step.data.update(clip=self.gen.video_path.name, continued=bool(self.gen.continued))
        # The clip exists but the user withdrew: keep nothing of this beat.
        if self.cancelled():
            self.gen.video_path.unlink(missing_ok=True)
            _drop_drawn(world, self.pending)
            bail(self.cancelled)

    # ---- commit ----------------------------------------------------------------

    def commit(self) -> Node:
        world, d, head, text = self.world, self.d, self.head, self.user_text
        with self.events.step("commit", "update_world", _say("更新世界", "Updating the world")) as step:
            # The director's late fields (options, quests...) finished while the
            # clip rendered. Only those fields are merged back.
            director._merge_late(d, self.late_fut)
            objective = str(d.get("objective") or "").strip()[:120] or world.objective
            world.objective = objective
            world.set_mainline(
                d.get("mainline") or "",
                phase=d.get("mainline_phase") or "",
                milestones=d.get("mainline_milestones"),
                opening_situation=d.get("reply") or text,
            )
            options = _three_options(d.get("options"), just_did=text)
            node_id = f"n{uuid.uuid4().hex[:8]}"
            world.apply_quest_updates(d.get("quest_updates") or {}, current_place=self.place, node_id=node_id)
            gen, pictures = self.gen, self.pictures
            step.data.update(
                node=node_id, objective=objective, options=options, next_steps=len(d.get("plan") or [])
            )
            node = Node(
                id=node_id,
                request_id=self.request_id,
                parent=world.head,
                video_id=gen.video_id,
                frames=_grab_frames(gen.video_path, node_id, float(gen.params.get("duration_seconds") or 0)),
                clip=gen.video_path.name,
                place=self.place,
                beat=d.get("beat", text),
                recap=str(d.get("recap") or "").strip(),
                mode=self.mode,
                carry=(d.get("carry") or "").strip(),
                options=options,
                recommend=_recommend_index(d.get("recommend"), len(options)),
                objective=objective,
                continue_with=(d.get("continue_with") or "").strip(),
                plan=[str(x).strip() for x in (d.get("plan") or []) if str(x).strip()],
                # Every non-scene reference used, including a focal object.
                subjects=[p["id"] for p in pictures if p["kind"] != "scene"],
                six_section=self.six,
                refs=self.ref_files,
                elapsed=self.clock.total(),
                user_text=text,
                reply=d.get("reply", ""),
                directive=d,
                pictures=pictures,
                audios=self.audio_bind,
                model_params=gen.params,
                continued_from=head.artifact_id if gen.continued else None,
                artifact_id=gen.artifact_id,
                timings=self.clock.marks,
                images_drawn=len(self.pending),
                image_seconds=self.drew_for,
                sketches=self.sketches,
                prompt=self.prompt,
                origin=self.origin,
                note="; ".join(
                    x
                    for x in (
                        (
                            ""
                            if not self.dropped
                            else _say(
                                f"同框超过 {MAX_CHARACTER_REFS} 人，这一段没带上 {'、'.join(self.dropped)} 的参考图",
                                f"More than {MAX_CHARACTER_REFS} characters in shot; left out the references "
                                f"of {', '.join(self.dropped)}",
                            )
                        ),
                        self.dropped_line,
                    )
                    if x
                ),
                state_before=self.state_before,
                state_after=world.dynamic_state_snapshot(),
                quest_before=self.quest_before,
                quest_after=world.quest_state_snapshot(),
            )
        node.trace = self.events.trace()
        world.add_node(node)
        # Only one line per beat goes back into the director's history, never the prompt.
        world.history.append({"role": "user", "text": _history_line(text, node)})
        summary = _history_beat(node) if not node.reply.strip() else ""
        world.history.append(
            {
                "role": "model",
                "text": f"{node.reply}\n[beat {len(world.lineage())} · {self.place}"
                + (f" · {summary}]" if summary else "]"),
            }
        )
        world.save()
        reqlog.record(
            "turn_done",
            node=node.id,
            place=node.place,
            beat=node.beat,
            recap=node.recap,
            reply=node.reply,
            continue_with=node.continue_with,
            plan=list(node.plan),
            options=list(node.options),
            objective=node.objective,
            refs=len(node.refs),
            audios=len(node.audios),
            images_drawn=int(node.images_drawn or 0),
            timings=dict(node.timings or {}),
            seconds=round(float(node.elapsed or 0), 3),
            repairs=list((node.directive or {}).get("_server_repairs") or []),
            six_section=dict(node.six_section or {}),
        )
        return node


def regenerate(
    world: World,
    *,
    six_section: dict[str, str] | None = None,
    seed: int | None = None,
    duration: float | None = None,
    progress: Callable[[str, str], None] = _noop,
    notify: Callable[[dict[str, Any]], None] = _no_event,
) -> Node:
    """Re-roll the head segment with an edited prompt and/or a new seed.

    The new segment is a sibling: same parent, references and continuation source.
    Only the head can be regenerated.
    """
    old = world.head_node
    if old is None:
        raise ValueError("There is no clip to regenerate")

    clock = _Clock()
    emitter = Emitter(notify, progress)
    vmodel = video.model(world.video_model)
    # ⚠️ A prompt edited by hand in the inspector must **not be changed by a single character**.
    #
    # The `_lock_*` steps exist because **models drift**: the visual style gets rewritten, state
    # gets dropped, speakers get mixed up. But the inspector is a debugging tool for people; locking
    # a hand-written prompt back would make the feature pointless: edit, rerun, and the same clip
    # comes out without any notice.
    #
    # Real bug: editing `overall_soundscape` and rerunning, `_lock_soundscape` replaced it with the
    # place's locked version. Style edits in `detailed_description` were likewise overwritten by
    # `_lock_visual_style`.
    #
    # The only one kept is `_ensure_soundscape`, and only when the field is **empty or N/A**: with
    # no audio direction the video model invents a soundtrack, including people speaking who have no
    # lines. It is not a lock but a guard against sending an input known to fail.
    edited = six_section is not None
    fields = six_section or old.six_section
    six = _lock_visual_style(dict(fields), world.visual_style)
    six = _lock_bound_subject_tokens(six, old.pictures)
    six = _lock_dialogue_language(six)
    six = _lock_dialogue_speakers(six, str(old.directive.get("dialogue") or ""), world, old.pictures)
    _ensure_soundscape(six, str(old.directive.get("dialogue") or ""))
    # ⚠️ No ambience lock after a manual edit.
    #
    # The locks above **add** things: inject the visual style, append dynamic state, add language
    # tags and speakers. They never destroy a written sentence and keep a hand-edited prompt within
    # the world's style and state (the tests check the style prefix survives a rerun). Only
    # `_lock_soundscape` **replaces the whole section**, silently throwing away the ambience the
    # user just edited. That was the reported bug.
    if not edited:
        six = _lock_soundscape(six, world, old.place, str(old.directive.get("dialogue") or ""))
    token_error = _reference_token_error(six, old.pictures)
    if token_error:
        raise ValueError(f"Invalid reference numbering in the video prompt: {token_error}")
    # Preserve an explicitly hand-edited prompt verbatim (the inspector is a
    # deliberate escape hatch), while sanitizing legacy/model-generated prompts
    # before they are sent again.
    if not edited:
        six.update(_ensure_prompt_english(six))
        if _sends_full_prompt(vmodel, old.continued_from):
            six = _fit_video_prompt(six, vmodel.max_prompt_chars)
    prompt = assemble(six)

    progress(
        "generate",
        f"{vmodel.label} "
        + (
            _say("按你改的 prompt 原样重跑", "regenerating with your edited prompt")
            if edited
            else _say("重跑", "regenerating")
        ),
    )
    # Send the reference voices again as they were. This used to be missing, and the rerun clip had
    # a different voice while the prompt still said "<Audio 1> is the voice of <Subject 1>". But "as
    # they were" excludes sending voices without dialogue: old nodes were generated before
    # _audio_refs got this gate, and resending them would repeat the same bug on a rerun, exactly
    # when the user reruns because nobody should be talking.
    reused: list[Path] = []
    if str(old.directive.get("dialogue") or "").strip():
        reused = [p for p in (voices.path(a.get("file") or "") for a in old.audios) if p]
    with emitter.step(
        "act", "render_clip", _say(f"{vmodel.label} 重拍", f"Re-rendering with {vmodel.label}")
    ) as step:
        step.data.update(model=vmodel.id, edited=edited, seed=seed)
        with clock.stage("video"):
            prompt, fallback = _continuation_prompt(
                vmodel, six, old.pictures, world, prompt, old.continued_from
            )
            gen = video.generate(
                prompt,
                [media_path("refs", f) for f in old.refs],
                ref_audios=reused,
                continue_from=old.continued_from,
                **fallback,
                duration=duration or float(old.model_params.get("duration_seconds", config.VIDEO_SECONDS)),
                seed=seed,
                model_id=world.video_model,
            )

    node_id = f"n{uuid.uuid4().hex[:8]}"
    node = Node(
        id=node_id,
        parent=old.parent,  # sibling, not child
        video_id=gen.video_id,
        frames=_grab_frames(gen.video_path, node_id, float(gen.params.get("duration_seconds") or 0)),
        clip=gen.video_path.name,
        place=old.place,
        beat=old.beat,
        recap=old.recap,
        mode=old.mode,
        carry=old.carry,
        # A rerun is the same beat, so it gets the same guidance; otherwise the recommendation falls
        # back to the first choice, which is usually not the one recommended originally.
        options=list(old.options),
        recommend=old.recommend,
        objective=old.objective,
        subjects=list(old.subjects),
        six_section=six,
        refs=list(old.refs),
        elapsed=clock.total(),
        user_text=old.user_text,
        reply=old.reply,
        directive=dict(old.directive),
        pictures=list(old.pictures),
        audios=list(old.audios),
        model_params=gen.params,
        continued_from=old.continued_from if gen.continued else None,
        artifact_id=gen.artifact_id,
        timings=clock.marks,
        images_drawn=0,  # a rerun reuses old.refs and draws no image
        image_seconds=0.0,
        prompt=prompt,
        origin="regenerate",
        trace=emitter.trace(),
        state_before=old.state_before,
        state_after=old.state_after,
        quest_before=old.quest_before,
        quest_after=old.quest_after,
    )
    world.nodes[node.id] = _as_dict(node)
    world.head = node.id
    world.here = node.place
    world.save()
    return node


__all__ = ["agent_delta", "imagegen", "llm", "reqlog", "video", "voices"]
