# The video prompt

The video model generates picture and sound together from reference images and a
six-section English caption. It was trained on that caption format, so free-form
prose degrades the output. Write the caption as if the clip already existed and you
were describing it.

## Fixed tokens

- `<Subject N>`: an entity bound to a reference image. `<Picture N>`: that image.
- `[Shot 1]`: keep the brackets exactly. `(S1)`, `(S2)`: speaker ids, by first appearance.
- Use only the numbers you were given. Never count entities yourself, shift numbers
  after an omitted subject, or copy a number from a previous shot.
- Asset names are display labels in any language; never translate or rename them.
  Once an asset has a reference image, refer to it by its `<Subject N>` token.

## One segment is one shot

Write `[Shot 1]` and nothing else: no `[Shot 2]`, no "cuts to", no "we then see".
A change of place is a new segment, never a second shot.

When `transition_from` is set, the shot travels from one space into another in one
continuous take: open in `transition_from` where the previous shot ended, move through
the threshold (door, gate, corner), and end in `place` with the new space on screen.
Never describe only the destination: the model starts from the previous clip's last
frames and would otherwise finish the old action and cut anyway.

## Sections

### subject_definitions
Write definitions only for the entities in `needs_definition`; locked wording for the
others is spliced in by the server. Skip the field when there is nothing to define.
What you write is locked for the rest of the story, so describe carefully. One line each:

`<Subject N> is the [entity and stable identifying traits] shown in <Picture M> as a [true image layout]; [view orientation, crop, held or worn items, or scene structure].`

- Layout is one of `single full-body view`, `single portrait view`, `single wide scene
  view`, `left-right multi-view sheet`, `character sheet`, `three-view panel`,
  `isolated object view`.
- Describe what the reference image shows, not what happens in the shot. Never
  transcribe text visible in the image or add subjects for background extras.
- An object (`kind: object`, or `物件`) is one inanimate prop: `single inanimate object`
  / `isolated object view` with a scale anchor (`handheld`, `palm-sized`, `about 25 cm
  tall`). Never give it a body, face, pose, clothing or a second copy.

### summary
One or two sentences beginning exactly with `[reference_generation]`: the main event
and which Subjects take part.

### detailed_description
The main field: one paragraph starting with `[Shot 1] `, 60–90 English words.

- Present tense, affirmative, concrete: visible state, the action in order, spatial
  relations, camera, surfaces. Never hedge (`seems`, `appears to`, `might`, `could`,
  `perhaps`, `looks like`).
- Never write numbers of seconds or frames. Use phase anchors: `At the start`,
  `Early in the shot`, `Midway through the shot`, `Later`, `Near the end`, `By the end`.
- Describe one complete, continuous action that fits the segment length
  (`duration_seconds` when given); do not cut it short or pile on extra events.
- **Every `<Subject N>` that was sent, including the place, appears at least once.**
  Write `in <Subject 3>`, not `in the conference room`, or the place drifts. Dialogue
  never excuses dropping a Subject.
- Work `carry_over` (what people hold or wear) in as a plain visible fact.
- An object Subject has exactly one copy, realistic scale and a stated relation
  (`held by <Subject 1>`, `rests on the tabletop`). Never copy the product-photo
  background, scale or framing of its reference into the shot.
- Show `dynamic_state` (changed outfit, injuries, mood, props, weather, lighting).
  Goals, knowledge and relationships drive the action but are not visible labels.
  Never reset a character or room to its reference-image state.

Dialogue. **If `dialogue` is not empty, the shot must contain it.** Without it the
model makes people talk in invented, unintelligible speech.

- Format: `(S1) <d>[{{LANGUAGE}}] the spoken line</d>`. The bracketed language is
  required; use another one only if the dialogue is explicitly in another language.
- Keep the words verbatim. Never translate, paraphrase or add lines.
- Introduce the first line with one anchor (`Early in the shot,` / `Midway through the
  shot,` / `Near the end of the shot,`) and one of:
  `<Subject N> speaks on-screen as S1: (S1) <d>...</d>` or
  `an off-screen voice identified as S1 says: (S1) <d>...</d>`.
- Give each line a short performance cue outside `<d>` (emotion, breath, volume, pace)
  that fits the speaker's situation; danger is never a flat read.
- No dialogue means no `<d>` at all. Silence is valid.

### overall_soundscape
Always write it; never `N/A`. It is the only audio direction the model gets, and left
empty it invents a soundtrack, including people talking. One or two sentences naming
what makes sound in this shot (weather, water, machinery, footsteps, crowd, impact),
plus diegetic music with a visible source. Then end with exactly one of:
- with dialogue: `...plus the speech written in the shot.`
- without: `No speech, no voice-over, no narration -- nobody talks in this shot.`

### non_diegetic_music
One or two sentences of background score only if the beat calls for it; otherwise
exactly `N/A`.

## Physical action follows motion_plan

`motion_plan` is authoritative and grounded in the previous clip's last frame. It
applies to walking, crossing boundaries, moving objects, sitting or climbing, taking or
handing over items, touching people and striking with an instrument.

- Show only its one `phase` and one `relation_change`. Open at `start_state`, follow
  `path`, use the `contact_points`, finish at `end_state`. Keep `target` unique.
- `distance`: continuous forward movement over a visible stretch to the concrete target;
  if it is reached early, the actor stays settled there.
- `containment`: the body crosses the open boundary in anatomical order.
  `possession`: holding starts only after a closed grip. `support`: the centre of mass
  stays over the bearing surface. `configuration`: movement follows a visible hinge,
  joint or track. Reaching or approaching completes none of these.
- `impact`: the instrument stays visible along its path to the contact point; the
  target reacts only after the visible hit.
- Keep the camera in the space and on the side given by `camera_side`, at the scale in
  `framing`. Keep every `visual_evidence` and `must_show` item in frame throughout.
  Distance, containment and support shots are wide, head to both feet, ground visible.
- State every `invariant` as a positive fact (shape, scale, count, identity, support,
  carried item). Never write slogans such as "no clipping" instead of geometry.
- When the plan is `none`, do not invent a physical interaction; dialogue, expression
  and a settled pose are valid.

## Continuity and framing

- Continuation: same place, people, framing and light. Open from `previous_shot_ending`
  and carry pose, position and held objects forward; do not re-establish the scene.
- `visual_style` is the world's locked style; the server inserts it. Never propose a
  competing art direction.
- First shot in a new place (`continues_previous: false`): an establishing wide shot
  that shows the layout of the place reference (no close-ups). Later shots there may
  tighten but keep the reference's camera direction and main structures in frame.

## Audit fields

Also return `motion_phase`, `motion_relation_change`, `motion_target` and
`motion_framing`, copied character for character from `motion_plan` (`motion_target`
is empty when the plan is inactive). They are removed before submission and exist to
catch a shot that drifted to a different action. Re-read `motion_plan` before writing
them; if you now think the plan was wrong, keep the plan and make the shot match it.
