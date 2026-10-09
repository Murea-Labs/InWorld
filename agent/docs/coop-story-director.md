# Co-op story planning and observation

[中文](coop-story-director_zh.md)

This note describes how a co-op room plans its story and checks the plan against
what was actually filmed. For the rules of play and the settings, see
[Co-op mode](coop.md).

## Planning levels

The story is planned at three levels:

- **Chapter outline.** While the opening clip is filmed, a planner writes the
  chapter goal and its stages in the background. It revises the outline every
  five rounds from what actually happened, and plans the next chapter once the
  current one is complete.
- **Forecast.** Each round's forecast proposes 2–3 short-term goals for the
  current stage, each with its prerequisites, who does what, a completion
  condition that can be seen on screen, and a fallback if it is blocked.
- **Suggested actions.** The actions suggested to the players carry out only the
  first step of the plan. Later steps are candidates: they never become part of
  this round's actions and are never treated as facts.

## Observation

After a clip is rendered, an independent observer checks the adopted forecast,
the players' actions and the first short-term goal against the clip. Stage
completion and goal completion are recorded separately, and both need keyframe
evidence from this round. A missing observation is not a failure. A goal
already confirmed cannot become the first step of the same stage's plan again.

## Current state

`story_state.facts` stores state per entity and attribute, with evidence, node
and round, for example `object:stone_gate.passage`, `boss.posture`,
`<user_id>.location`. A new observation replaces the old value of the same
field; fields not on screen keep theirs. Past changes live in the per-round
observation records. The planner reads the current state and the last eight
rounds of observations; the full history is not repeated in the current state.
After an asset is changed by hand, the structured state of the old composition
is void; late observations of the old one are kept as history only.

## Parallel generation and correction

While the current clip renders, the next forecast is drafted in the
background. The latest observation, outline revisions and the entry point of
recommended prefetching check what each suggestion is based on, including
changes of passage, position, action results and remaining goals within the
same stage. A background correction first decides whether to keep the plan,
change only the actions, or rewrite the forecast and short-term goals, then
publishes atomically.

Player actions already edited or adopted, and a next clip already rendering,
keep the plan they adopted. A recommended clip not yet submitted may finish
its choreography first and reserves at most six seconds in total for
observation and correction before submission. A finished observation resumes
the work actively, and a one-shot wake-up covers timeouts, without extra
polling. Corrections run in their own thread pool; on timeout the planning
lock is released and late results cannot overwrite later work. After leaving
immersive mode, the existing pause-prefetch flag stays in effect.

## Repetition and stalls

Suggestions record the real goal, route, tactic and role in the cooperation;
only suggestions players actually adopt count toward tactic history. Repeating
the same kind of action is fine while it makes progress. When clearly blocked,
or after two observations without progress, the short-term plan changes at
once; the full outline keeps its five-round cycle. Stalls get one extra
semantic check and at most one correction of the affected part; if that
service is unavailable the already validated plan is kept. When an
observation is unclear, prefer a single action whose result is easy to
confirm, and never invent success or failure.

The outline tab shows "action plan" and "global outline" separately. The
action plan separates "this round" from "later, to be confirmed", with
conditions and alternatives expandable; stage numbers are used only in the
global outline. The action log keeps the scope of corrections, tactic checks
and degraded fallbacks.

Display fields of short-term plans are generated directly in the story
language (`agent.language`); only ids and enums are English. In a Chinese room,
when the outline is opened, fully English text from older saves is translated
on demand in one batch via
`/story/display-zh` and cached; ordinary Chinese content calls no model. This
only produces display text: it changes neither the original outline, the
adopted plan, the observation conditions, the room revision nor the
recommended video cache, and it never blocks forecasts, suggestions or video
generation. If translation fails the outline can still be read, with a retry
button.
