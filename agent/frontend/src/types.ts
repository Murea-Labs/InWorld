/** A world rolled by the dice. Not created yet; whoever clicks decides. */
export interface RandomWorld {
  title: string
  setting: string
  mainline: string
  /** An id from STYLE_PRESETS. styleById() turns it into the real style bible text. */
  style_id: string
  /** What the first beat plays. One sentence, second person. */
  opening: string
  /** References created before the opening video for a player-supplied world idea. */
  assets?: { kind: 'player' | 'character' | 'creature' | 'object' | 'place'; name: string; prompt: string }[]
  visual_style?: string
  refs?: { kind: 'subject' | 'place'; name: string; mine?: boolean; ref: string; prompt?: string }[]
}

export interface Picture {
  n: number
  kind: 'character' | 'scene'
  id: string
  name: string
  desc: string
  ref: string
}

/** A reference image drawn in this beat: what the world shows the user it drew. */
export interface Sketch {
  kind: 'subject' | 'place'
  /** Asset id: S1 and so on for characters, the place name for places. Old clips lack it; fall back to name. */
  id?: string
  /** 人物 / 生物 / 物件 / 场景 (character / creature / object / place): anything that may reappear has a reference image. */
  entity?: string
  name: string
  brief: string
  ref: string
}

/** One reference voice actually sent to the video model for this clip. Old clips lack it (node.audios defaults to empty). */
export interface AudioRef {
  /** The <Audio N> in the prompt. */
  n: number
  /** Bound to <Subject N> in the prompt. 0 = this person's reference image was dropped, so it is not bound. */
  subject_n?: number
  sid: string
  name: string
  /** File name under data/voices. Library entries start with library/. */
  file?: string
  voice_id?: string
  voice_name?: string
}

export type TracePhase = 'perceive' | 'plan' | 'check' | 'act' | 'commit' | 'reflect'

/** One step of the agent loop (murea_agent/events.py). */
export interface TraceStep {
  id: string
  phase: TracePhase
  name: string
  status: 'running' | 'done' | 'error' | 'cancelled'
  summary: string
  data: Record<string, unknown>
  /** Seconds since the turn started; null for steps recorded afterwards. */
  started: number | null
  seconds: number | null
}

export interface Node {
  id: string
  parent: string | null
  video_id: string
  clip: string
  place: string
  beat: string
  mode: 'continue' | 'cut'
  carry: string
  options: string[]
  /** Which option the world recommends (the one advancing the mainline). Old clips lack it. */
  recommend?: number
  /** The goal of this step, so the user knows where they are heading. */
  objective?: string
  /** What the next beat plays while the user's sentence is not finished yet. Empty = done. */
  continue_with: string
  /**
   * Steps after `continue_with` not played yet, one sentence each. With the server's
   * "done up to step N" on the node the frontend knows how many beats remain instead of
   * guessing every beat.
   */
  plan?: string[]
  subjects: string[]
  six_section: Record<string, string>
  refs: string[]
  elapsed: number
  created: string
  user_text: string
  reply: string
  directive: Record<string, unknown>
  pictures: Picture[]
  /** Who <Audio N> is bound to. Old clips lack it. */
  audios?: AudioRef[]
  model_params: Record<string, unknown>
  continued_from: string | null
  timings: Record<string, number>
  images_drawn: number
  image_seconds: number
  sketches?: Sketch[]
  prompt: string
  origin: 'turn' | 'regenerate' | 'auto'
  note: string
  state_before?: DynamicStateSnapshot | null
  state_after?: DynamicStateSnapshot | null
  quest_before?: QuestStateSnapshot | null
  quest_after?: QuestStateSnapshot | null
  /** The agent steps that produced this clip (empty for older clips). */
  trace?: TraceStep[]
}

export interface SubjectState {
  location: string
  outfit: string
  condition: string
  mood: string
  goal: string
  knowledge: string[]
  relations: Record<string, string>
}

export interface PlaceState {
  time_of_day: string
  weather: string
  lighting: string
  condition: string
  props: string[]
}

export interface DynamicStateSnapshot {
  subjects: Record<string, SubjectState>
  places: Record<string, PlaceState>
}

export interface Quest {
  id: string
  type: 'main' | 'side'
  title: string
  summary: string
  status: 'active' | 'completed' | 'failed'
  progress: string
  place: string
  created_node: string
  updated_node: string
  /** Current mainline stage; usually empty for side quests. */
  phase?: string
  /** Mainline stage milestones in causal order. */
  milestones?: string[]
}

/** Quests are saved along the clip tree branch; reviewing an old clip shows the quest state of its time. */
export interface QuestStateSnapshot {
  quests: Quest[]
  objective: string
  mainline?: string
  mainline_phase?: string
  mainline_milestones?: string[]
  opening_situation?: string
}

/** A character's locked voice: what they sound like in the video. */
export interface VoiceRef {
  /** Id in the voice library. Empty for uploads. */
  id: string
  name: string
  /** File name under data/voices. Empty = this character has no voice. */
  ref: string
  /** `auto` = picked by the server from the name and description; the user never chose it. */
  source?: 'library' | 'upload' | 'auto' | ''
  /** The automatic pick could not even determine the gender; roughly a coin flip, so the UI asks to confirm. */
  guessed?: boolean
  /** Old worlds stored the director's attribute words, no file. */
  instruct?: string
}

/** One voice library entry. Fixed, previewable and replaceable; not made up per character. */
export interface LibraryVoice {
  id: string
  name: string
  desc: string
  /** 男 / 女 (male / female) */
  gender: string
  /** 儿童 / 少年-少女 / 青年 / 中年 / 老年 (age band) */
  age: string
  /** Dialect tag; empty for Mandarin. */
  accent: string
  /** Narrative-role group; only the curated tier has it. */
  group: string
  /** The curated tier the director picks from. The user can pick any library entry. */
  gm: boolean
  seconds: number
  ref: string
}

export interface Subject {
  name: string
  ref: string
  /** The English prompt used to draw this image, for the rewriter. */
  desc: string
  /** A short human-readable description. */
  brief?: string
  definition?: string
  /** Who drives this character. 'user' is the one you play; the world plays the rest. */
  control?: 'user' | 'ai'
  /** 人物 / 生物 / 物件 (character / creature / object). Monsters and key props need their own reference image or they change next time. */
  kind?: string
  /** Locked voice, sent to the video model as reference_audios; unrelated to the narrator. */
  voice?: VoiceRef
  state?: SubjectState
  /** Items found by the critic use the last frame first, replaced once the background image is ready. */
  ref_status?: 'pending' | 'ready' | 'failed'
  /** Where the current ref came from: video last frame, background drawing or the user. */
  ref_source?: 'frame' | 'generated' | 'user' | string
  /** The item's original last frame, kept as evidence of its appearance. */
  source_ref?: string
  ref_error?: string
}

export interface Place {
  ref: string
  desc: string
  brief?: string
  first_seen: string
  definition?: string
  /** Background places prepared for the agent; shown on the map and in the assets only once a clip reaches them. */
  hidden_until_visited?: boolean
  /** Root place of the map group; a main place points to itself. */
  main_scene?: string
  state?: PlaceState
}

export interface PlaceLink {
  from: string
  to: string
  n?: number
}

export interface VideoModel {
  id: string
  label: string
  note: string
}

export interface World {
  /** Background assets are used by the opening, but not revealed to the player before the first clip finishes. */
  id: string
  owner_id: string
  video_model: string
  title: string
  setting: string
  /** Mainline: the big thing happening in this world. If empty the director sets one in the first beat. */
  mainline?: string
  /** Current mainline stage, updated as the story moves. */
  mainline_phase?: string
  /** Mainline stage milestones from start to end. */
  mainline_milestones?: string[]
  /** Opening facts kept from the old instant mainline after migration; not used as a quest title. */
  opening_situation?: string
  /** The goal of the current step, updated by the director every beat. */
  objective?: string
  visual_style: string
  here: string
  head: string | null
  subjects: Record<string, Subject>
  places: Record<string, Place>
  place_links?: PlaceLink[]
  quests?: Quest[]
  nodes: Record<string, Node>
  history: { role: 'user' | 'model'; text: string }[]
  created: string
}

export interface WorldSummary {
  id: string
  owner_id: string
  title: string
  here: string
  segments: number
  created: string
}

/**
 * Local identity. Passwords are optional: users created before passwords existed have none, so the
 * rule is "whoever set one must type it".
 */
export interface User {
  id: string
  name: string
  created: string
  /** This user has a password and must type it to enter. Others enter with one click. */
  locked?: boolean
}

export interface Health {
  version: string
  /** The commit at process start differs from the working tree = old code running, not restarted after a change. */
  stale?: boolean
  head?: string
  video: { id: string; label: string; note: string; choices: VideoModel[] }
  video_online: boolean
  /** Voice exists only when the backend has TTS_URL. Otherwise the switch is disabled. */
  tts?: boolean
  /**
   * Whether the TTS service answers **right now**. `tts` only says whether it is configured;
   * they are separate because "configured but unreachable" used to fail silently.
   */
  tts_up?: boolean
  continuation: boolean
  text_model: string
  image_model: string
  gemini_key: boolean
  language?: 'en' | 'zh'
  /** `coop` servers host two-player rooms instead of single-player worlds. */
  mode?: 'single' | 'coop'
  /** Which providers have credentials. Missing ones mean only the demo world can be played. */
  ready?: { llm: boolean; image: boolean; video: boolean }
}

export type Stage = 'queue' | 'gm' | 'image' | 'rewrite' | 'generate'

export type TurnEvent =
  | { type: 'progress'; stage: Stage; detail?: string }
  // When one sentence is split into several clips, announce the whole plan before the first clip
  // starts.
  | {
      type: 'sequence'
      goal: string
      current: number
      total: number
      segments: { index: number; content: string }[]
    }
  // Sent as soon as the director finishes, without waiting for the video; the user reads the
  // world's answer within seconds.
  | { type: 'say'; text: string }
  // An agent step started or finished; see AgentTrace.
  | ({ type: 'step' } & TraceStep)
  // Reference images drawn in this beat, sent as soon as each is drawn
  | { type: 'sketch'; items: Sketch[] }
  // Cancelled by the user, not a failure: no red bar, the sentence goes back into the input
  | { type: 'cancelled' }
  | { type: 'error'; message: string; trace?: string }
  | { type: 'done'; reply: string; node: Node; world: World }
