import type {
  Health, LibraryVoice, Node, RandomWorld, TurnEvent, User, VoiceRef, World,
  WorldSummary,
} from './types'
import { t } from './i18n'

// Relative to the page, so the app works under any path prefix.
export const BASE = import.meta.env.BASE_URL

async function json<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(BASE.replace(/\/$/, '') + '/' + path, init)
  if (!r.ok) {
    const raw = (await r.text()).slice(0, 300)
    try {
      const detail = (JSON.parse(raw) as { detail?: unknown }).detail
      throw new Error(typeof detail === 'string' ? detail : raw)
    } catch (error) {
      if (error instanceof SyntaxError) throw new Error(raw || `HTTP ${r.status}`, { cause: error })
      throw error
    }
  }
  return r.json() as Promise<T>
}

export const mediaUrl = (kind: 'clips' | 'refs', name: string) =>
  `${BASE.replace(/\/$/, '')}/${kind}/${name}`

export const getHealth = () => json<Health>('api/health')
export const listUsers = () => json<User[]>('api/users')
export const getCurrentUser = () => json<User>('api/users/me')
export const selectUser = (id: string, password = '') =>
  json<User>(`api/users/${encodeURIComponent(id)}/select`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ password }),
  })
export const createUser = (name: string, password = '') => json<User>('api/users', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ name, password }),
})
export const logoutUser = () =>
  json<{ ok: boolean }>('api/users/logout', { method: 'POST' })
export const deleteUser = (id: string, password = '') =>
  json<{ ok: boolean; user: User }>(`api/users/${encodeURIComponent(id)}`, {
    method: 'DELETE',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ password }),
  })
/** Roll a world setting. Roll only, never create; creating goes through createWorld. */
export const randomWorld = (styles: { id: string; label: string }[]) =>
  json<RandomWorld>('api/worlds/random', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ styles }),
  })


export const listWorlds = () => json<WorldSummary[]>('api/worlds')
export const getWorld = (id: string) => json<World>(`api/worlds/${id}`)

export const renameWorld = (id: string, title: string) =>
  json<World>(`api/worlds/${id}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ title }),
  })

export const createWorld = (
  title: string, setting: string, visualStyle: string, videoModel: string,
  mainline = '',
) =>
  json<World>('api/worlds', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      title, setting, visual_style: visualStyle, video_model: videoModel, mainline,
    }),
  })

/** Change who drives this character. */
export const setSubjectControl = (wid: string, sid: string, control: 'user' | 'ai') =>
  json(`api/worlds/${wid}/subjects/${sid}/control`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ control }),
  })

export const exportUrl = (id: string) =>
  `${BASE.replace(/\/$/, '')}/api/worlds/${id}/export`

export const importWorld = (file: File) => {
  const fd = new FormData()
  fd.append('file', file)
  return json<{ id: string; title: string; segments: number; files_written: number }>(
    'api/worlds/import', { method: 'POST', body: fd },
  )
}

export const deleteWorld = (id: string) =>
  json<{ ok: boolean; removed_files: number }>(`api/worlds/${id}`, { method: 'DELETE' })

export const rollback = (id: string) =>
  json<{ ok: boolean }>(`api/worlds/${id}/rollback`, { method: 'POST' })

export const imagine = (
  prompt: string, kind: 'place' | 'subject' | 'object', visualStyle = '',
) =>
  json<{ ref: string; url: string; elapsed: number; model: string }>('api/imagine', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ prompt, kind, visual_style: visualStyle }),
  })

export const addSubject = (id: string, form: FormData) =>
  json<{ id: string; ref: string }>(`api/worlds/${id}/subjects`, { method: 'POST', body: form })

export const replaceSubject = (id: string, sid: string, form: FormData) =>
  json<{ id: string; ref: string }>(
    `api/worlds/${id}/subjects/${encodeURIComponent(sid)}`,
    { method: 'PUT', body: form },
  )

export const dropItem = (id: string, sid: string) =>
  json<{ ok: boolean }>(`api/worlds/${id}/subjects/${encodeURIComponent(sid)}`, {
    method: 'DELETE',
  })

export const addPlace = (id: string, form: FormData) =>
  json<{ name: string; ref: string }>(`api/worlds/${id}/places`, { method: 'POST', body: form })

export const updatePlaceMainScene = (
  id: string,
  name: string,
  mainScene: string,
) => json<{ name: string; main_scene: string }>(
  `api/worlds/${id}/places/${encodeURIComponent(name)}/main-scene`,
  {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ main_scene: mainScene }),
  },
)

export const dropPlace = (id: string, name: string) =>
  json<{ ok: boolean }>(`api/worlds/${id}/places/${encodeURIComponent(name)}`, {
    method: 'DELETE',
  })

export const setNote = (id: string, nid: string, note: string) =>
  json<{ ok: boolean }>(`api/worlds/${id}/nodes/${nid}/note`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ note }),
  })

/** Transport failures are distinct from model, HTTP, and rendering errors. */
export class StreamTransportError extends Error {
  constructor() { super(t('连接中断，正在确认生成结果')); this.name = 'StreamTransportError' }
}

/** POST once. A terminal event is success even if the connection closes after it. */
export async function streamTurn(
  path: string,
  body: unknown,
  onEvent: (e: TurnEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  let r: Response
  try {
    r = await fetch(BASE.replace(/\/$/, '') + '/' + path, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body), signal,
    })
  } catch (e) {
    if (signal?.aborted || (e as Error).name === 'AbortError') throw e
    throw new StreamTransportError()
  }
  if (!r.ok) throw new Error((await r.text()).slice(0, 300) || `HTTP ${r.status}`)
  if (!r.body) throw new StreamTransportError()
  const reader = r.body.getReader()
  const dec = new TextDecoder()
  let buf = ''
  try {
    for (;;) {
      let chunk: ReadableStreamReadResult<Uint8Array>
      try { chunk = await reader.read() } catch (e) {
        if (signal?.aborted || (e as Error).name === 'AbortError') throw e
        throw new StreamTransportError()
      }
      if (chunk.done) throw new StreamTransportError() // EOF before done is not success
      buf = (buf + dec.decode(chunk.value, { stream: true })).replace(/\r\n/g, '\n')
      let i: number
      while ((i = buf.indexOf('\n\n')) >= 0) {
        const block = buf.slice(0, i)
        buf = buf.slice(i + 2)
        const data = block.split('\n').filter(line => line.startsWith('data:'))
          .map(line => line.slice(5).trimStart()).join('\n')
        if (!data) continue // SSE heartbeat comments
        const event = JSON.parse(data) as TurnEvent
        onEvent(event)
        if (['done', 'error', 'cancelled'].includes(event.type)) return
      }
    }
  } finally {
    void reader.cancel().catch(() => {})
    reader.releaseLock()
  }
}

export async function turn(
  id: string, text: string, onEvent: (e: TurnEvent) => void, auto = false,
  signal?: AbortSignal, mentions: string[] = [],
): Promise<void> {
  const requestId = crypto.randomUUID()
  try {
    await streamTurn(`api/worlds/${id}/turn`,
      { text, auto, mentions, request_id: requestId }, onEvent, signal)
    return
  } catch (e) {
    if (!(e instanceof StreamTransportError)) throw e
  }
  onEvent({ type: 'progress', stage: 'queue', detail: t('连接中断，正在查询后台结果，请勿重复发送') })
  // Never repeat the POST: the server may still be working after the stream drops.
  let failures = 0
  for (let attempt = 0; attempt < 90; attempt++) {
    signal?.throwIfAborted()
    if (attempt) await new Promise(resolve => setTimeout(resolve, 2000))
    signal?.throwIfAborted()
    let result: { status: string; event?: Extract<TurnEvent, { type: 'done' }> }
    try {
      result = await json(`api/worlds/${id}/turn-status/${requestId}`, { signal })
      failures = 0
    } catch (e) {
      if (signal?.aborted) throw e
      if (++failures < 3) continue
      throw new Error(t('连接暂时不可用，输入已保留。请恢复连接后刷新确认生成结果，避免重复发送。'), { cause: e })
    }
    if (result.status === 'done' && result.event) { onEvent(result.event); return }
    if (result.status !== 'running') {
      throw new Error(t('连接中断，未找到本次已完成的片段。输入已保留，请确认网络后重试。'))
    }
  }
  throw new Error(t('后台仍未返回结果。请稍后刷新查看，避免重复发送。'))
}

/**
 * Let the director plan the recommended choice ahead, so the world answers at once when the
 * user picks it and generation starts right away.
 * 
 * Only the director (LLM only) runs ahead, never the video: the video model renders one job
 * at a time, and a clip nobody picked would delay the one the user actually picks.
 * Failures do not matter here, it is a free head start, so errors are swallowed.
 */
export const prefetch = (id: string, text: string, kind: 'option' | 'draft' = 'option') =>
  json<{ ok: boolean; cached: boolean }>(`api/worlds/${id}/prefetch`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, kind }),
  }).catch(() => ({ ok: false, cached: false }))

/** Changed my mind: stop the running beat. The backend runs one beat at a time, so no world id is needed. */
export const cancelTurn = () =>
  json<{ ok: boolean; was_running: boolean }>('api/worlds/cancel', { method: 'POST' })

export const regenerate = (
  id: string,
  body: { six_section?: Record<string, string> | null; seed?: number | null },
  onEvent: (e: TurnEvent) => void,
) => streamTurn(`api/worlds/${id}/regenerate`, body, onEvent)

/** URL of a voice file. Library refs start with library/; the backend route accepts the slash. */
export const voiceUrl = (ref: string) =>
  `${BASE.replace(/\/$/, '')}/voices/${ref}`

/** The voice library. Its content is fixed, so it is fetched once per session. */
let voiceLibrary: Promise<LibraryVoice[]> | null = null
export const listVoices = () => (voiceLibrary ??= json<{ voices: LibraryVoice[] }>(
  'api/voices').then((r) => r.voices).catch((e) => {
    voiceLibrary = null      // do not cache a failure; the next open can retry
    throw e
  }))

/** Upload your own voice. Saved first to get a file name, bound to the character once the preview sounds right. */
export const uploadVoice = (file: File) => {
  const fd = new FormData()
  fd.append('audio', file)
  return json<{ ref: string; name: string; url: string }>(
    'api/voices/upload', { method: 'POST', body: fd },
  )
}

/** Change or clear a character's voice. */
export const setSubjectVoice = (
  wid: string, sid: string, v: { voice_id?: string; ref?: string; name?: string } | null,
) => json<{ id: string; voice: VoiceRef }>(
  `api/worlds/${wid}/subjects/${encodeURIComponent(sid)}/voice`,
  {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(v ?? { clear: true }),
  },
)

export type { Node, World }

/**
 * This world's outbound call log: what each director / rewriter / image / video call sent.
 * For debugging only; it survives restarts (data/logs/<world>.jsonl) and keeps the latest 500.
 */
export interface LogEntry {
  t: number
  kind: string
  /** Which step: director, rewriter… The most useful field on the two streaming entries. */
  tag?: string
  stream?: boolean
  first_token_seconds?: number
  chars?: number
  /** The output on turn_done */
  place?: string
  beat?: string
  recap?: string
  reply?: string
  objective?: string
  asset_kind?: string
  /** merged_fallback: which field disagreed with the motion plan */
  mismatch?: string
  phase?: string
  relation?: string
  six_section?: Record<string, string>
  options?: string[]
  continue_with?: string
  plan?: string[]
  timings?: Record<string, number>
  images_drawn?: number
  repairs?: string[]
  model?: string
  ok?: boolean
  seconds?: number
  upload_seconds?: number
  retries?: number
  status?: number
  usage?: Record<string, number>
  request?: { system_chars?: number; turns?: { role?: string; text?: string }[]
              schema?: boolean; thinking?: number }
  response?: string
  error?: string
  prompt?: string
  payload?: Record<string, string>
  refs?: number
  audios?: number
  continuation?: boolean
  attempt?: number
  origin?: string
  user_text?: string
  head?: string
  duration?: number | null
}

export const worldLog = (id: string, limit = 200) =>
  json<LogEntry[]>(`api/worlds/${id}/log?limit=${limit}`)

export const clearWorldLog = (id: string) =>
  json<{ ok: boolean }>(`api/worlds/${id}/log`, { method: 'DELETE' })

/**
 * Critic: looks at the frames just rendered and replaces the three choices and the objective with
 * ones that match the picture.
 * 
 * Called when the clip starts playing: the user is watching and thinking, so this call is free
 * for them. On failure it returns ok:false and the director's version stays; play goes on.
 */
export interface Verdict {
  ok: boolean
  observed?: string
  options?: string[]
  recommend?: number
  objective?: string
  /** Things really picked up in the picture. When non-empty, `world` comes back too and the asset panel refreshes. */
  picked_up?: string
  /** Items visibly dropped, destroyed or used up in the picture; already removed from the assets. */
  dropped?: string[]
  /** Characters who appeared in the narration but were not registered in that beat; the critic adds them during playback so they are in the next beat. */
  appeared?: string
  world?: World
}

export const adjudicate = (id: string, node: string): Promise<Verdict> =>
  json<Verdict>(
    `api/worlds/${id}/adjudicate`,
    { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ node }) },
  ).catch((): Verdict => ({ ok: false }))

