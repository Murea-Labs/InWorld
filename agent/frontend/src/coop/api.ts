import { t } from '../i18n'
import { BASE } from "../api";
import type { Node, TraceStep } from "../types";

export interface Player {
  user_id: string;
  name: string;
  subject_id: string;
  character: string;
  description: string;
  ref: string;
  action: string;
  ready: boolean;
  kind?: "human" | "ai" | "local";
  controller_id?: string;
  ai_profile?: string;
  ai_source?: "offline" | "llm";
  portrait?: {
    request_id: string;
    status: "generating" | "ready" | "failed" | "applied";
    name: string;
    description: string;
    ref: string;
    error: string;
    elapsed?: number;
  };
}
export interface Round {
  round: number;
  merged_prompt: string;
  inputs: Player[];
  guidance?: {
    world_event: string;
    world_event_type: "none" | "environment" | "boss_introduction" | "boss_action";
    world_event_key?: string;
    boss_candidate?: { name: string; description: string };
    objective: string;
  };
  observation?: {
    node_id: string;
    event_outcome: "observed" | "partially_observed" | "not_observed" | "unknown";
    summary: string;
    environment_state: string;
    boss_state: string;
    player_states: { user_id: string; state: string }[];
    story_phase: string;
  };
  node: Node;
  story_progress?: StoryProgress;
}
export interface StoryProgress {
  version: number;
  last_round?: number;
  node_id?: string;
  chapter?: number;
  stage_title?: string;
  stage_goal?: string;
  summary?: string;
  next_gap?: string;
  assessment?: "advanced" | "stalled" | "unknown";
  completed_stages?: number;
  total_stages?: number;
  active_stage_id?: string;
  stages?: Record<string, {
    status: "pending" | "active" | "completed" | "blocked" | "abandoned";
    evidence?: string;
    remaining?: string;
    reason?: string;
  }>;
}
export interface ShortTermPlan {
  stage_id: string;
  purpose: "approach" | "discover" | "opportunity" | "breakthrough" | "transition" | "closure" | "observe";
  change_reason: string;
  steps: { id: string; goal: string; prerequisite: string; cooperation: string;
    completion_evidence: string; if_blocked: string }[];
}
export interface Room {
  chapter_transition?: { pending: boolean; status?: string; error?: string; scene?: string; prepared?: string };
  story_progress?: StoryProgress;
  story_outline?: {
    overall_goal: string;
    version: number;
    chapter: number;
    active_stage_id?: string;
    reviewed_round?: number;
    stages: { id: string; title: string; goal: string; completion_evidence: string; alternative: string }[];
  };
  story_outline_job?: { status: string; round: number };
  shared_image_backend?: string;
  scene_description?: string;
  scene_asset_source?: "manual" | "opening_video" | "round_video" | "";
  monster?: SharedAsset & { subject_id: string };
  asset_previews?: Partial<Record<AssetKind, AssetPreview>>;
  asset_extraction?: {
    status: "generating" | "ready" | "failed";
    request_id?: string;
    error: string;
    scene?: { status: string; name?: string; description?: string; ref?: string; frame_index?: number; reason?: string };
    monster?: { status: string; name?: string; description?: string; ref?: string; frame_index?: number; reason?: string };
  };
  /** Latest post-video visual check for an automatically discovered BOSS. */
  boss_detection?: {
    status: "generating" | "ready" | "failed" | "superseded" | "skipped";
    request_id?: string;
    node_id?: string;
    result?: "applied" | "none" | "";
    name?: string;
    description?: string;
    ref?: string;
    error?: string;
  };
  /** Background preparation for the exact Agent-recommended joint action. */
  recommendation_prefetch?: {
    status: "planning" | "prompt_ready" | "rendering" | "ready" | "consuming" | "failed" | "superseded";
    round?: number;
    actions?: Record<string, string>;
    error?: string;
  };
  asset_recommendations?: { id: string; scene: AssetSpec; monster: AssetSpec };
  mode: "local" | "online";
  code: string;
  world_id: string;
  title: string;
  setting: string;
  visual_style: string;
  capacity: number;
  host_id: string;
  members: Player[];
  /** Server-accepted order of the current round's ready actions. */
  action_order?: string[];
  round: number;
  revision: number;
  /** Reviewed turn inputs; background progress updates do not change this token. */
  generation_signature?: string;
  story_state?: {
    phase: string;
    environment: string;
    boss: string;
    boss_known?: boolean;
    boss_visible?: boolean;
    players: Record<string, string>;
    last_observation: string;
    last_node_id: string;
    recent_events: { round: number; type: string; key?: string; event: string; outcome: string }[];
    persistent_history?: {
      node_id: string;
      round: number;
      requested_actions: { user_id: string; character?: string; action?: string }[];
      forecast: Record<string, unknown>;
      observation: Record<string, unknown>;
    }[];
  };
  status: "waiting" | "generating" | "failed" | "closed";
  error: string;
  progress: string;
  generation_prompt?: string;
  /** Short Chinese narration played only while the current video is pending. */
  generation_narration?: string;
  scene_ref: string;
  scene_name: string;
  rounds: Round[];
  opening_node?: Node;
  preview_clip?: string;
  opening_generation?: boolean;
  generation_round?: number;
  generation_cancelling?: boolean;
  merged_prompt: string;
  generation_enabled: boolean;
  auto_generate: boolean;
  progress_stage?: "agent" | "generation";
  /** Agent steps of the round being generated, live for every player. */
  agent_trace?: TraceStep[];
  intro?: {
    round?: number;
    status: "generating" | "ready" | "failed";
    narration: string;
    /** The agent's preview of what the boss / environment will do in the next clip; not something that has happened. */
    world_event?: string;
    world_event_type?: "none" | "environment" | "boss_introduction" | "boss_action";
    world_event_key?: string;
    boss_candidate?: { name: string; description: string };
    objective: string;
    short_term_plan?: ShortTermPlan;
    suggestions: { user_id: string; action: string }[];
    error: string;
  };
  image_generation_enabled: boolean;
  ai_mode: "offline" | "llm";
  ai_status?: "waiting" | "thinking" | "ready" | "failed";
  ai_error?: string;
  invite_url: string;
  created: number;
}

export type AssetKind = 'scene' | 'monster';
export interface AssetSpec { name: string; description: string }
export interface SharedAsset extends AssetSpec { ref: string; source?: "manual" | "opening_video" | "round_video" | "" }
export interface AssetPreview extends SharedAsset {
  request_id: string;
  status: 'generating' | 'ready' | 'failed' | 'applied';
  error: string;
}
export const recommendAssets = (code: string, refresh = false) =>
  request<Room>(`/${code}/assets/recommendations`, 'POST', { refresh });
export const generateAsset = (code: string, kind: AssetKind, spec: AssetSpec) =>
  request<Room>(`/${code}/assets/${kind}/generate`, 'POST', spec);
export const useAsset = (code: string, kind: AssetKind, request_id: string) =>
  request<Room>(`/${code}/assets/${kind}/use`, 'POST', { request_id });
export const uploadAsset = (code: string, kind: AssetKind, body: FormData) =>
  request<Room>(`/${code}/assets/${kind}/upload`, 'POST', body);
export const removeAsset = (code: string, kind: AssetKind) =>
  request<Room>(`/${code}/assets/${kind}`, 'DELETE');
export type RoomSummary = Pick<
  Room,
  "code" | "title" | "status" | "round" | "members" | "created" | "mode" | "host_id"
>;

export class RoomRequestError extends Error {
  constructor(
    message: string,
    public status: number,
    public code?: string,
  ) {
    super(message);
  }
}

async function request<T>(
  path = "",
  method = "GET",
  body?: object | FormData,
): Promise<T> {
  const form = body instanceof FormData;
  const r = await fetch(`${BASE.replace(/\/$/, "")}/api/coop/rooms${path}`, {
    method,
    cache: "no-store",
    headers: body && !form ? { "Content-Type": "application/json" } : undefined,
    body: body ? (form ? (body as FormData) : JSON.stringify(body)) : undefined,
  });
  if (!r.ok) {
    const error = await r.json().catch(() => null);
    throw new RoomRequestError(
      typeof error?.detail === "string"
        ? error.detail
        : typeof error?.detail?.message === "string" ? error.detail.message : t("请求失败（{status}）", { status: r.status }),
      r.status,
      typeof error?.detail?.code === "string" ? error.detail.code : undefined,
    );
  }
  return r.json() as Promise<T>;
}
export const listRooms = () => request<RoomSummary[]>();
export const getRoom = (code: string) =>
  request<Room>(`/${encodeURIComponent(code)}`);
export const getStoryDisplay = (code: string) =>
  request<{ translations: Record<string, string> }>(`/${encodeURIComponent(code)}/story/display-zh`, "POST");
export const createRoom = (body: {
  title: string;
  setting: string;
  visual_style: string;
  capacity: number;
  mode: "local" | "online";
}) => request<Room>("", "POST", body);
export const joinRoom = (code: string) =>
  request<Room>(`/${encodeURIComponent(code)}/join`, "POST");
export const leaveRoom = (code: string) =>
  request<{ ok: boolean }>(`/${code}/leave`, "POST");
export const deleteRoom = (code: string) =>
  request<{ ok: boolean; removed_files: number }>(`/${encodeURIComponent(code)}`, "DELETE");
export const rollback = (code: string, node_id: string) =>
  request<Room>(`/${encodeURIComponent(code)}/rollback`, "POST", { node_id });
export const setReference = (
  code: string,
  kind: "character" | "scene",
  body: FormData,
) => request<Room>(`/${code}/${kind}`, "POST", body);
export const submitAction = (
  room: Room,
  action: string,
  ready = true,
  target_uid?: string,
) =>
  request<Room>(`/${room.code}/action`, "PUT", {
    action,
    ready,
    target_uid,
    round: room.round,
  });
export const generate = (room: Room) =>
  request<Room>(`/${room.code}/generate`, "POST", {
    round: room.round,
    revision: room.revision,
    generation_signature: room.generation_signature,
  });

export const startImmersive = (code: string, session_id: string) =>
  request<Room>(`/${code}/immersive/start`, "POST", { session_id });
export const advanceImmersive = (code: string, session_id: string, round: number) =>
  request<Room>(`/${code}/immersive/advance`, "POST", { session_id, round });
export const stopImmersive = (code: string, session_id: string) =>
  request<Room>(`/${code}/immersive/stop`, "POST", { session_id });

export const addAI = (code: string, profile: string) =>
  request<Room>(`/${code}/ai`, "POST", { profile });
export const removeAI = (code: string, id: string) =>
  request<Room>(`/${code}/ai/${encodeURIComponent(id)}`, "DELETE");
export const respondAI = (code: string) =>
  request<Room>(`/${code}/ai/respond`, "POST");

export const generatePortrait = (
  code: string,
  body: {
    name?: string;
    description?: string;
    randomize?: boolean;
    target_uid?: string;
  },
) => request<Room>(`/${code}/portrait`, "POST", body);
export const usePortrait = (
  code: string,
  request_id: string,
  target_uid?: string,
) => request<Room>(`/${code}/portrait/use`, "POST", { request_id, target_uid });

export interface LogRow {
  t: number;
  kind: string;
  round?: number;
  request_id?: string;
  stage?: "agent" | "generation";
  tag?: string;
  ok?: boolean;
  [key: string]: unknown;
}
export const getLog = (code: string, limit = 300) =>
  request<LogRow[]>(`/${encodeURIComponent(code)}/log?limit=${limit}`);
export const retryIntro = (code: string) => request<Room>(`/${code}/intro/retry`, "POST");
export const startOpening = (code: string) => request<Room>(`/${code}/opening/start`, "POST");

export function watchRoom(code: string, onRoom: (room: Room) => void, onRevoked: () => void, onStatus: (connected: boolean) => void) {
  const source = new EventSource(`${BASE.replace(/\/$/, "")}/api/coop/rooms/${encodeURIComponent(code)}/events`, { withCredentials: true });
  source.onopen = () => onStatus(true);
  source.onerror = () => onStatus(false);
  source.addEventListener("room", (event) => onRoom(JSON.parse((event as MessageEvent).data) as Room));
  source.addEventListener("revoked", () => { source.close(); onRevoked(); });
  return () => source.close();
}
