# Co-op rooms

[中文](coop_zh.md)

> Co-op follows `agent.language` like single-player: English or Chinese prompts,
> messages and UI. Rooms film with the same Murea service; within one place each
> round continues from the previous clip.

Set `mode: coop` in `config.yaml` and the web UI opens the co-op lobby
instead of the single-player stage. Two to four players share one world:
everyone writes what their own character does, and each round all actions
are filmed together as **one** clip.

## Rooms

| Kind | Players | Notes |
|---|---|---|
| `local` | 2 seats, one person | You write the actions of both A and B on one page. For trying things out alone. No invite link, no AI teammates. |
| `online` | 2–4 people | Others join with the invite link or the 12-character code. The host can add AI teammates. |

Each player binds a character: upload a reference image, or let the room
draw one (random or from a description). A portrait a player asks for is a
preview until they adopt it, so a failed or slow drawing never replaces a
character. The room also draws default portraits for empty seats and for AI
teammates; those are bound only while the seat still has no image of its own.

The host may also set two shared assets, both optional:

- **Place**: an empty environment image, no people or creatures.
- **Boss**: a non-player creature with its own reference image. It takes no
  seat and submits no actions.

The video model accepts at most 5 reference images, so with a boss the room
holds at most 3 players (one image is kept for the place).

## A round, end to end

1. **Opening.** When every character is bound the host starts the adventure.
   The server films a short opening clip (`coop.opening_seconds`) from the
   world setting and the adopted assets. Players submit nothing for it.
2. **Asset extraction.** After the opening, a critic reads its keyframes. For
   a place or boss that was not set by hand it extracts what is **really
   visible** and draws a fixed reference image. A boss that is only mentioned,
   or not clearly visible, is never registered.
3. **Forecast and suggestions.** A planner writes what the boss / environment
   will do next and a shared goal, then suggests an action for each player.
   This is a plan, not a fact: it is never written into the world until a clip
   confirms it.
4. **Actions.** Each player submits an action (or adopts the suggestion).
   Actions are translated to English in the background while teammates are
   still typing.
5. **Filming.** When the last player submits, the server freezes an
   immutable snapshot of the room and compiles one timeline: every player's
   action, the world event, stable `<Subject N>` numbers for every reference.
   One video request is sent for the whole party.
6. **Observation.** After the clip is saved, an observer reads its keyframes
   and records what actually happened: the environment, the boss, each
   character's last visible state, the story phase, and whether the forecast
   came true. Only observations become facts.

Clips continue from the previous clip's latent while the set of reference
images is unchanged. Changing an asset by hand (upload, adopt a preview, edit
a description, remove it, change a character) starts a new chain once.

Logs: `coop_forecast_started`, `coop_scene_plan_compiled`, `video_request`,
`video_result`, `coop_story_observation`, `coop_done` and
`coop_continuation_decision` (why a chain was kept or broken) mark the stages.

## Story outline

When the opening is filmed, a planner writes a chapter goal and 3–5 stages,
each with a completion condition that a picture can confirm. Progress is
judged in the same observation call, from keyframe evidence only: an unclear
or missing observation counts as unknown, never as failure. After two clearly
stalled rounds the forecast must try a new route. Every five rounds the
outline is revised from what really happened; when all stages are done the
next chapter starts. See [coop-story-director.md](coop-story-director.md).

Set `COOP_STORY_ENABLED=0` to turn the outline off and keep only the
forecast / suggestion loop.

## AI teammates

In online rooms the host can add AI teammates (axe warrior, guardian, mage).
Each has its own portrait and takes a seat. Once every human has submitted,
the AI teammates write matching actions in one model call and the round
starts. If every human adopted their suggestion unchanged, the AI teammates
take theirs too, without a model call, so the clip filmed in advance
(`coop.speculative_video`) can be used. With no new human action, AI teammates
never start a round by themselves.

`coop.ai_mode: offline` replaces the model with simple rules, for testing
without a model. Turn off generation as well if the whole instance should run
offline.

## Immersive mode

Immersive mode adopts each player's suggestion automatically and keeps the
story moving. The next clip may be generated while the current one plays,
but the player only switches after the current clip ends, so the story never
runs ahead of playback. Exiting (button, Esc, leaving the room) cancels the
pending round; a clip already committed is kept.

## Configuration

`config.yaml`:

| Key | Default | Meaning |
|---|---|---|
| `coop.public_url` | `""` | Public address used in invite links. Empty uses the browser's address. |
| `coop.opening_seconds` | `6` | Length of the opening clip. |
| `coop.min_seconds` / `coop.max_seconds` | `6` / `15` | Range for rounds, in 2-second steps up to the maximum. The length is estimated from the actions: independent actions run in parallel, dependent ones add up. A round is one clip; one that needs more than the maximum is compressed into it. At most 20 s on Murea-Fast (5 s on a single-GPU service), 15 s on the full model. |
| `coop.speculative_video` | `true` | Film the recommended actions while players are deciding. Adopting all suggestions unchanged plays it at once; any change cancels it. Costs one extra video per round. |
| `coop.ai_mode` | `llm` | `llm` or `offline` AI teammates. |
| `coop.ai_model` | `""` | Model for the AI teammates; empty uses `llm.models.agent`. |

Environment switches, mostly for testing:

| Variable | Default | Meaning |
|---|---|---|
| `COOP_GENERATION_ENABLED` | `1` | `0` allows rooms, uploads and previews but never generates video. |
| `COOP_AUTO_GENERATE` | `1` | Start the round as soon as everyone is ready, and draw default portraits. `0` needs the host to start it and draws none. |
| `COOP_INTRO_ENABLED` | `1` | Forecasts and suggestions. `0` also turns off the story outline, and since the opening waits for the forecast, no room can start. |
| `COOP_ACTION_PREFETCH` | `1` | Translate each action while teammates are still typing. |
| `COOP_RECOMMENDATION_PREFETCH` | `1` | Prepare the full prompt for the suggested actions in advance. `0` also turns off `coop.speculative_video`. |
| `COOP_STORY_ENABLED` | `1` | The chapter outline and progress tracking. |
| `COOP_PARALLEL_ROUNDS` | `4` | How many rooms may run a round at the same time (1–4). |

Portraits, places and bosses are drawn with the configured `image.provider`.

## Code

| Path | What |
|---|---|
| `murea_agent/coop/room.py` | rooms, seats, locks, recovery after a restart |
| `murea_agent/coop/loop.py` | one round through the agent's traced turn loop |
| `murea_agent/coop/generation.py` | joint shot planning, the video request |
| `murea_agent/coop/action_text.py` | translation repair of actions and narration |
| `murea_agent/coop/scene_plan.py` | `ShotCompiler`: the timeline and subject numbering |
| `murea_agent/coop/duration.py` | round length |
| `murea_agent/coop/intro.py` | forecasts and suggestions |
| `murea_agent/coop/prefetch.py` | prompts and speculative clips for the suggested actions |
| `murea_agent/coop/observation.py` | what each clip really showed |
| `murea_agent/coop/story.py`, `chapter.py` | chapter outline, progress and the next chapter |
| `murea_agent/coop/immersive.py` | immersive mode |
| `murea_agent/coop/assets.py`, `portrait.py` | place / boss assets and character portraits |
| `murea_agent/coop/ai.py` | AI teammates |
| `app/api/routes/coop.py` | HTTP API |
| `frontend/src/coop/` | lobby, room and immersive UI |

Tests: `python tests/run_all.py` runs the co-op tests (`tests/test_coop*.py`)
against a co-op server config, offline, with fake models and temporary data
directories.
