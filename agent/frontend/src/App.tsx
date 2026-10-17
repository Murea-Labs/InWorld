import { useCallback, useEffect, useRef, useState, type CSSProperties } from 'react'
import * as api from './api'
import Assets from './components/Assets'
import Header from './components/Header'
import Hud from './components/Hud'
import Inspector from './components/Inspector'
import SideRail, { type RailTab } from './components/SideRail'
import Stage from './components/Stage'
import UserGate from './components/UserGate'
import CoopRoom from './coop/CoopRoom'
import WorldMap from './components/WorldMap'
import AddAsset from './dialogs/AddAsset'
import NewWorld from './dialogs/NewWorld'
import WorldLog from './dialogs/WorldLog'
import { OPENING, STAGE_CN, lineage } from './lib'
import { styleMenu } from './presets'
import { prepareWorld, type WorldStart } from './worldStart'
import { loadTheme, saveTheme, type MureaTheme } from './theme'
import { loadTts, saveTts, speak, stopSpeaking, finishSpeaking } from './tts'
// `Stage` is taken by the component above, so the type is imported under another name
import type {
  Health, Node, Stage as PipeStage, Sketch, TraceStep, TurnEvent, User, World, WorldSummary,
} from './types'
import { upsertStep } from './components/AgentTrace'
import { t } from './i18n'

/**
 * How many beats the system may continue on its own via continue_with.
 * 
 * It used to be a fixed 4: `continue_with` held "A, then B, then C", the director replanned every
 * beat and the queue never converged, so a hard cap was needed, and "go downstairs, cross the
 * street to the restaurant" was cut off at beat 4.
 * 
 * Now the server pins the remaining steps on `node.plan`, one step per beat; the queue only shrinks
 * and always converges. So the limit follows the sentence's own plan, and `AUTO_HARD_MAX` is only a
 * defensive ceiling (a dozen steps usually means the director went wrong, and every beat is a paid
 * video generation).
 */
const AUTO_MAX = 4
const AUTO_HARD_MAX = 8

export interface TurnState {
  stage: PipeStage
  detail: string
  started: number
  /** What the director already said. The video is still rendering, but this arrives within seconds. */
  say: string
  /** Reference images drawn in this beat. */
  sketches: Sketch[]
  /** Agent steps of this turn so far (live trace). */
  steps?: TraceStep[]
}

export interface SequenceSegmentState {
  index: number
  content: string
  subtitle: string
  status: 'pending' | 'generating' | 'done'
}

export interface SequenceState {
  goal: string
  current: number
  total: number
  segments: SequenceSegmentState[]
}

/* Immersive sizing needs both the ratio and its inverse. Computing the inverse here leaves only
   multiplication in CSS; dividing by a var in calc is newer syntax. */
type AppStyle = CSSProperties & {
  '--video-ratio': number
  '--video-ratio-inv': number
}

export default function App() {
  const [health, setHealth] = useState<Health | null>(null)
  // Co-op servers show the two-player room; `#coop/CODE` selects a room.
  const isCoop = health?.mode === 'coop'
  const missingKeys = health?.ready
    ? ([['llm', t('语言模型 key')], ['image', t('生图 key')], ['video', t('视频 key')]] as const)
        .filter(([k]) => !health.ready![k]).map(([, label]) => label)
    : []
  const [coopRoute, setCoopRoute] = useState(() => window.location.hash)
  useEffect(() => {
    const changed = () => setCoopRoute(window.location.hash)
    window.addEventListener('hashchange', changed)
    return () => window.removeEventListener('hashchange', changed)
  }, [])
  const [users, setUsers] = useState<User[]>([])
  const [user, setUser] = useState<User | null>(null)
  const [userReady, setUserReady] = useState(false)
  const [worlds, setWorlds] = useState<WorldSummary[]>([])
  const [world, setWorld] = useState<World | null>(null)
  const [currentNode, setCurrent] = useState<Node | null>(null)
  // The selected clip always belongs to the world on screen: a late callback from the previous
  // world's player (e.g. auto-advance while switching) must not show its clip in a new world.
  const current = currentNode && world?.nodes[currentNode.id] ? currentNode : null
  const shownWorld = world
  const [busy, setBusy] = useState(false)
  // The line already sent but not in world.history yet. Without showing it right away nothing on
  // screen moves for a while after Enter, which looks like it was never sent.
  const [pending, setPending] = useState('')
  const [draft, setDraft] = useState('')
  // Start early: do not wait for Enter. There is usually 0.5–2 s between the user stopping typing
  // and pressing send, free time.
  // 
  // Why: clicking a choice already costs 0 s (all three are prefetched), but typing your own line
  // paid the full agent time with nothing on screen. The prefetch key is world + text, so if the
  // early text equals what is finally sent, `_take_prefetched` hits and the agent part of the beat
  // drops to zero.
  // 
  // A miss costs nothing for the user: just one wasted call.
  const spec = useRef('')
  useEffect(() => {
    const text = draft.trim()
    // Too short: skip; a half-typed sentence will differ from the final one. Also skip while a beat
    // is playing: the backend is busy and an early call would queue ahead of the real one.
    if (busy || text.length < 6 || spec.current === text) return
    const id = world?.id
    if (!id) return
    const t = setTimeout(() => {
      spec.current = text
      // 'draft': a half-typed sentence. The server keeps only one and drafts only evict drafts; it
      // used to evict and cancel the first queued choice (the recommended one).
      void api.prefetch(id, text, 'draft').catch(() => {})
    }, 400)
    return () => clearTimeout(t)
  }, [draft, busy, world?.id])
  const [turn, setTurn] = useState<TurnState | null>(null)
  // Which world this beat was sent to and its text. SSE wraps backend errors in a 200 error event
  // (sse.py), so fetch does not reject and the most common failure never reaches takeTurn's catch.
  // Recovery hangs on the error branch of onEvent; both share this ref.
  const flight = useRef<
    { wid: string; text: string; system: boolean; auto: boolean } | null
  >(null)
  const lastNode = useRef<Node | null>(null)
  // This beat's fetch. Undo must close the local stream and tell the backend not to save the
  // result.
  const abort = useRef<AbortController | null>(null)
  const autoStop = useRef(false)
  const [autoRunning, setAutoRunning] = useState(false)
  // When one input is split into several clips, this state lives across several oneTurn calls. turn
  // is only the current request and is rebuilt on each continuation; putting the whole plan in turn
  // would lose the first segment as soon as the second started.
  const [sequence, setSequence] = useState<SequenceState | null>(null)
  const sequenceStep = useRef(0)
  const [error, setError] = useState('')
  const [inspecting, setInspecting] = useState(false)
  const [newWorld, setNewWorld] = useState(false)
  const [rolling, setRolling] = useState(false)
  const worldStart = useRef<WorldStart | null>(null)
  const [mapOpen, setMapOpen] = useState(false)
  // The stage comes first: both drawers start closed and open from the header.
  const [assetsOpen, setAssetsOpen] = useState(false)
  // The left drawer defaults to the map: dialogue and choices are on the picture; the chat is for
  // looking back.
  const [tab, setTab] = useState<RailTab>('map')
  // The left drawer (map / chat) can close too, symmetric with the asset drawer on the right.
  const [railOpen, setRailOpen] = useState(false)
  const [logOpen, setLogOpen] = useState(false)
  const [immersive, setImmersive] = useState(false)
  // The picture is really playing. While it plays, dim the surrounding panels; hovering restores
  // them.
  const [playing, setPlaying] = useState(false)
  // In immersive mode the header and transport bar are hidden until the mouse reaches the top edge.
  const [chrome, setChrome] = useState(false)
  const [videoRatio, setVideoRatio] = useState(16 / 9)
  const [theme, setTheme] = useState<MureaTheme>(loadTheme)
  // null = the backend has no TTS_URL and the switch is disabled. Known only after health returns.
  const [tts, setTts] = useState<boolean | null>(null)
  const [speaking, setSpeaking] = useState(false)
  /** TTS is configured but unreachable: disable the switch and say so, never pretend it works */
  const [ttsUp, setTtsUp] = useState(true)
  // onEvent is a new closure on every render but must read the switch state **now**: use a ref.
  const ttsOn = useRef(false)
  ttsOn.current = tts === true
  const [asset, setAsset] = useState<{
    kind: 'subject' | 'place'
    /** subjects also includes objects; tells the editor not to show character-only controls for a new one. */
    subjectKind?: '人物' | '物件'
    key?: string
    file?: File          // an image pasted / uploaded from the chat, filled in as the candidate
  } | null>(null)
  // Items registered by the critic show the last frame first; the background image replaces ref in
  // the world file later. Poll only while pending items exist, so normal turns make no extra
  // requests.
  const hasPendingItemRefs = !!world && Object.values(world.subjects).some((subject) =>
    (subject.kind === '物件' || subject.kind === '物品' || subject.kind === '道具')
    && subject.ref_status === 'pending',
  )

  const load = useCallback(async (id: string) => {
    const w = await api.getWorld(id)
    // Switching world clears the previous world's in-flight state, or a stuck bubble and stopwatch
    // would follow
    stopSpeaking()
    setSpeaking(false)
    setPending('')
    setTurn(null)
    setSequence(null)
    setWorld(w)
    const chain = lineage(w)
    setCurrent(chain.length ? chain[chain.length - 1] : null)
  }, [])

  const refresh = useCallback(
    async (select?: string) => {
      const list = await api.listWorlds()
      setWorlds(list)
      const id = select ?? list[0]?.id
      if (id) await load(id)
      else setWorld(null)
    },
    [load],
  )

  useEffect(() => {
    let active = true
    let pending = false
    let loaded = false
    const refreshHealth = async () => {
      // A page opened in a background tab still needs its first health check:
      // it decides the server mode (single player or co-op rooms).
      if (pending || (document.hidden && loaded)) return
      pending = true
      try {
        const h = await api.getHealth()
        if (!active) return
        loaded = true
        setHealth(h)
        setTts(h.tts ? loadTts() : null)
        setTtsUp(h.tts_up !== false)
      } catch {
        if (active) setHealth(null)
      } finally {
        pending = false
      }
    }
    void refreshHealth()
    const timer = window.setInterval(refreshHealth, 15000)
    window.addEventListener('focus', refreshHealth)
    document.addEventListener('visibilitychange', refreshHealth)
    return () => {
      active = false
      window.clearInterval(timer)
      window.removeEventListener('focus', refreshHealth)
      document.removeEventListener('visibilitychange', refreshHealth)
    }
  }, [])

  useEffect(() => {
    let active = true
    Promise.all([
      api.listUsers(),
      api.getCurrentUser().catch(() => null),
    ]).then(async ([available, selected]) => {
      if (!active) return
      // A single profile without a password (the usual local setup) needs no picking screen.
      const only = available.length === 1 && !available[0].locked ? available[0] : null
      if (!selected && only) selected = await api.selectUser(only.id, '')
      if (!active) return
      setUsers(available)
      setUser(selected)
      setUserReady(true)
    }).catch((e) => {
      if (!active) return
      setError(String(e))
      setUserReady(true)
    })
    return () => { active = false }
  }, [])

  useEffect(() => {
    if (!userReady || !user) return
    refresh().catch((e) => setError(String(e)))
  }, [userReady, user, refresh])

  useEffect(() => {
    if (!world?.id || !hasPendingItemRefs) return
    let active = true
    const refreshPendingAssets = async () => {
      try {
        const latest = await api.getWorld(world.id)
        if (active) setWorld((currentWorld) => (
          currentWorld?.id === latest.id ? latest : currentWorld
        ))
      } catch {
        // Asset enhancement is best-effort; a failure must not interrupt the clip being played.
      }
    }
    const timer = window.setInterval(refreshPendingAssets, 2500)
    return () => {
      active = false
      window.clearInterval(timer)
    }
  }, [world?.id, hasPendingItemRefs])

  useEffect(() => {
    saveTheme(theme)
  }, [theme])

  const onEvent = (e: TurnEvent) => {
    const f = flight.current
    if (e.type === 'progress')
      setTurn((t) => ({
        ...(t ?? { say: '', sketches: [] }),
        stage: e.stage,
        detail: e.detail || STAGE_CN[e.stage] || e.stage,
        started: t?.started ?? Date.now(),
      }))
    else if (e.type === 'sequence') {
      const completed = sequenceStep.current
      setSequence((value) => {
        // The server plan starts at this clip, not at the user's first clip.
        // Preserve the completed prefix and replace only the remaining plan.
        const segments: SequenceSegmentState[] = [
          ...Array.from({ length: completed }, (_, i) => ({
            index: i + 1,
            content: value?.segments[i]?.content ?? '',
            subtitle: value?.segments[i]?.subtitle ?? '',
            status: 'done' as const,
          })),
          ...e.segments.map((segment, i) => ({
            index: completed + i + 1,
            content: segment.content,
            subtitle: '',
            status: i === 0 ? 'generating' as const : 'pending' as const,
          })),
        ]
        return {
          goal: value?.goal || e.goal,
          current: completed + 1,
          total: segments.length,
          segments,
        }
      })
    }
    // The world speaks first, the picture comes later. These are the earliest things to show in a
    // beat; they hang on turn instead of waiting for done so they appear well before the video.
    else if (e.type === 'say') {
      setTurn((t) => (t ? { ...t, say: e.text } : t))
      const active = sequenceStep.current + 1
      setSequence((value) => value ? {
        ...value,
        current: active,
        segments: value.segments.map((segment) => (
          segment.index === active
            ? { ...segment, subtitle: e.text, status: 'generating' }
            : segment
        )),
      } : value)
      // The video still takes a while; read this line now so something happens during the wait
      if (ttsOn.current) {
        void speak(e.text, setSpeaking).catch((err: Error) => {
          // Say so when it cannot be read aloud. This used to be silent: the switch was on, the
          // world said nothing, and users concluded the feature did not work.
          // 
          // The two failures are reported separately because the fixes differ: an unreachable
          // service needs the backend checked, a browser block only needs one click on the page.
          if (err?.message === 'autoplay-blocked') {
            setError(t('浏览器拦下了自动播放 —— 在页面上点一下，下一句就有声音了'))
            return
          }
          setTtsUp(false)
          setError(t('语音服务连不上，这一段没有旁白（画面不受影响）'))
        })
      }
    }
    else if (e.type === 'step') {
      const { type: _type, ...step } = e
      setTurn((t) => ({
        ...(t ?? { stage: 'gm', detail: '', say: '', sketches: [] }),
        started: t?.started ?? Date.now(),
        steps: upsertStep(t?.steps, step),
      }))
    }
    else if (e.type === 'sketch')
      setTurn((t) => (t ? { ...t, sketches: e.items } : t))
    else if (e.type === 'cancelled') {
      setTurn(null)
      setPending('')
      // Undo usually means "I want to change that line": put it back in the input instead of
      // retyping. The opening and auto-continued lines were not written by the user and are not
      // returned.
      if (f && !f.system && !f.auto) setDraft((d) => d || f.text)
      flight.current = null
      autoStop.current = true
    }
    else if (e.type === 'error') {
      stopSpeaking()
      setSpeaking(false)
      setTurn(null)
      setPending('')
      // Put the unfinished line back into the input so the user can edit and resend. Not the
      // opening; the user did not write it.
      if (f && !f.system) setDraft((d) => d || f.text)
      flight.current = null
      // The instructions of auto-continued beats are written by the system. Silently filling the
      // input would show a sentence the user never typed with no explanation; say so.
      setError(f?.auto
        ? t('自动续跑断了：{msg}　没走完的那句已经放回输入框，改一下或直接发。', { msg: e.message })
        : e.message)
    } else {
      const completed = sequenceStep.current + 1
      setSequence((value) => value ? {
        ...value,
        current: completed,
        segments: value.segments.map((segment) => (
          segment.index === completed
            ? { ...segment, subtitle: e.node.reply || segment.subtitle, status: 'done' }
            : segment
        )),
      } : value)
      setTurn(null)
      setPending('')
      flight.current = null
      // This beat belongs to another world (the user switched away midway): drop it, do not pull
      // them back
      if (f && f.wid !== e.world.id) return
      lastNode.current = e.node
      setWorld(e.world)
      setWorlds((items) => items.map((item) => item.id === e.world.id
        ? { ...item, title: e.world.title, here: e.world.here,
            segments: Object.keys(e.world.nodes).length }
        : item))
      setCurrent(e.node)
      // Run the director step of all three choices ahead. On click the world answers at once and
      // generation starts, skipping the whole agent part. Not while continue_with is unfinished:
      // the system is about to send the next beat itself, and a prefetch would be for a path nobody
      // clicks.
      // 
      // With only the recommended one prefetched, a sample of 109 beats hit 8 times (7%).
      // 
      // ⚠️ **All three at once.** This used to be sequential to avoid rate limits, assuming the user watches a
      // long clip. Measured over 278 beats the hit rate was only **22%**: three sequential
      // prefetches take ~12 s while users click a few seconds after a short clip ends, usually
      // before their choice's turn. The wait is for the **next** clip; the clip they watch is
      // short.
      // 
      // Rate limits matter much less now: the backoff is 0.5/1/2 s **with jitter**, added for
      // concurrent prefetch. A failed prefetch is harmless anyway; it just falls back to normal
      // speed.
      if (!e.node.continue_with) {
        const wid = e.world.id
        const nid = e.node.id

        // ⚠️ Send the prefetch **immediately**, never behind anything.
        // 
        // The previous version put it in `adjudicate().then()`, a sequential prerequisite: the
        // prefetches started 8 s after the output. Many users click within 10 s, and those clicks
        // always missed and waited for the director.
        const fire = (opts: string[], rec: number) => {
          // Send the recommended one a moment earlier so it is first in any rate-limit backoff
          const order = [opts[rec], ...opts.filter((_, i) => i !== rec)]
          void Promise.all(order.filter(Boolean).map((text, i) =>
            new Promise((r) => setTimeout(r, i * 120))
              .then(() => api.prefetch(wid, text, 'option'))
              .catch(() => {})))
        }
        fire(e.node.options ?? [], e.node.recommend ?? 0)

        // The critic runs **in parallel**: it looks at the frames really rendered and replaces the
        // choices with ones matching the picture. Only changed choices need a new prefetch;
        // unchanged ones were sent above.
        void api.adjudicate(wid, nid).then((v) => {
          if (!v.ok) return
          const opts = v.options?.length === 3 ? v.options : null
          const rec = v.recommend ?? 0
          if (opts) {
            // Hud reads the node from world.nodes before falling back to current.
            // Keep both copies aligned or it displays the old, uncorrected options.
            setWorld((w) => {
              if (!w || w.id !== wid || w.head !== nid || !w.nodes[nid]) return w
              return { ...w, nodes: { ...w.nodes, [nid]: {
                ...w.nodes[nid], options: opts, recommend: rec,
                objective: v.objective ?? w.nodes[nid].objective,
              } } }
            })
            setCurrent((n) => (n && n.id === nid
              ? { ...n, options: opts, recommend: rec,
                  objective: v.objective ?? n.objective }
              : n))
            const before = new Set(e.node.options ?? [])
            const fresh = opts.filter((t) => !before.has(t))
            if (fresh.length) fire(opts, rec)
          }
          // Something was really picked up / dropped in the picture: the assets must reflect it
          // now, not next beat
          if ((v.picked_up || v.dropped?.length || v.appeared) && v.world) {
            setWorld(v.world)
            const said = [
              v.picked_up && t('捡到了「{name}」', { name: v.picked_up }),
              v.dropped?.length && t('失去了「{name}」', { name: v.dropped.join(t('、')) }),
              // The narration said something arrived but the picture did not show it; the critic
              // registered it, and next beat it really appears with its own reference image.
              v.appeared && t('「{name}」已登记，下一拍开始出现在画面里', { name: v.appeared }),
            ].filter(Boolean).join('，')
            if (said) setError(said)
          }
        })
      }
    }
  }

  const run = async (fn: () => Promise<void>) => {
    setBusy(true)
    setError('')
    try {
      await fn()
    } catch (e) {
      setTurn(null)
      setError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  /**
   * One beat. The text shows first and returns to the input on failure: never swallow what the user
   * typed. `system` marks lines the system sends itself (the opening), which are not returned to
   * the input. Returns this beat's node, or null on failure.
   */
  const oneTurn = async (
    wid: string, text: string, system: boolean, auto: boolean,
    mentions: string[] = [], sequenceIndex = 0,
  ): Promise<Node | null> => {
    sequenceStep.current = sequenceIndex
    if (sequenceIndex > 0) {
      const active = sequenceIndex + 1
      setSequence((value) => {
        if (!value) return value
        const segments: SequenceSegmentState[] = value.segments.map((segment) => ({
          ...segment,
          status: segment.index < active
            ? 'done'
            : segment.index === active ? 'generating' : 'pending',
        }))
        // An unplanned continuation may start without a new split event.
        if (!segments.some((segment) => segment.index === active)) {
          segments.push({ index: active, content: text, subtitle: '', status: 'generating' })
        }
        return { ...value, current: active, total: segments.length, segments }
      })
    }
    flight.current = { wid, text, system, auto }
    lastNode.current = null
    setPending(system ? '' : text)
    setTurn({ stage: 'gm', detail: STAGE_CN.gm, started: Date.now(),
      say: '', sketches: [] })
    abort.current = new AbortController()
    try {
      await api.turn(wid, text, onEvent, auto, abort.current.signal, mentions)
    } catch (e) {
      // Aborted by ourselves, not an error: the cancelled event (or stopTurn below) already cleaned
      // up
      if ((e as Error).name === 'AbortError') return null

      // Only a network failure ends up here (backend errors arrive as SSE error events)
      setPending('')
      if (!system) setDraft((d) => d || text)
      flight.current = null
      throw e
    }
    return lastNode.current
  }

  /**
   * One sentence may span several spaces. The director puts the unfinished part in `continue_with`
   * and this walks it for the user; otherwise "go downstairs, cross the street to the restaurant"
   * would take five inputs, and the user would be operating a text-to-video page, not an agent.
   */
  const takeTurn = (
    wid: string, text: string, system = false, mentions: string[] = [],
  ) =>
    run(async () => {
      autoStop.current = false
      sequenceStep.current = 0
      setSequence(null)
      let next = text
      let sys = system
      let auto = false
      for (let step = 0; ; step++) {
        const node = await oneTurn(
          wid, next, sys, auto, step === 0 ? mentions : [], step,
        )
        setAutoRunning(false)
        const cw = (node?.continue_with || '').trim()
        if (!node || !cw || autoStop.current) return
        // Beats this sentence needs = already played + the step at hand + the rest of the plan.
        // Nodes without plan are old (generated before the field existed); fall back to the fixed
        // limit.
        const planned = node.plan?.length ?? 0
        const cap = planned > 0
          ? Math.min(step + 2 + planned, AUTO_HARD_MAX)
          : AUTO_MAX
        if (step >= cap - 1) {
          // Cannot finish: hand the wheel back instead of running on forever
          setError(t('这一句还没走完，接着说就行：{rest}', { rest: cw }))
          return
        }
        next = cw
        sys = false
        auto = true
        setAutoRunning(true)
      }
    })

  /** Sent and then changed my mind. A video request already sent cannot be recalled, but this clip is not written into the world. */
  const stopTurn = useCallback(() => {
    const f = flight.current
    autoStop.current = true
    setAutoRunning(false)
    setSequence(null)
    api.cancelTurn().catch(() => {})
    abort.current?.abort()
    abort.current = null
    setTurn(null)
    setPending('')
    if (f && !f.system && !f.auto) setDraft((d) => d || f.text)
    flight.current = null
  }, [])

  const switchUser = async () => {
    if (busy) stopTurn()
    stopSpeaking()
    setSpeaking(false)
    try {
      await api.logoutUser()
    } catch {
      // Clearing the local UI is still the right fallback; the next selection
      // replaces the cookie explicitly.
    }
    setImmersive(false)
    setWorlds([])
    setWorld(null)
    setCurrent(null)
    setPending('')
    setTurn(null)
    setSequence(null)
    setUser(null)
  }

  const send = (text: string, mentions: string[] = []) => {
    if (rolling || busy) return
    if (!world) return startWorld(text)
    stopSpeaking()               // the user speaks, so the world stops reading
    setSpeaking(false)
    autoStop.current = true      // the user interrupting = stop the auto-continue chain
    setDraft('')
    return takeTurn(world.id, text, false, mentions)
  }

  const regenerate = (six: Record<string, string> | null, seed: number | null) =>
    run(async () => {
      if (!world) return
      setTurn({ stage: 'generate', detail: STAGE_CN.generate, started: Date.now(),
        say: '', sketches: [] })
      await api.regenerate(world.id, { six_section: six, seed }, onEvent)
    })

  const removeWorld = () =>
    run(async () => {
      if (!world) return
      const n = Object.keys(world.nodes).length
      if (!confirm(t('删掉「{title}」？{n} 段片段和它独占的参考图会一起清掉，不可恢复。', { title: world.title, n })))
        return
      const r = await api.deleteWorld(world.id)
      setInspecting(false)
      setCurrent(null)
      await refresh()
      setError(t('已删除，清掉 {n} 个文件', { n: r.removed_files }))
    })

  const renameWorld = async (id: string, title: string) => {
    setError('')
    try {
      const updated = await api.renameWorld(id, title)
      setWorlds((items) => items.map((item) =>
        item.id === id ? { ...item, title: updated.title } : item))
      setWorld((active) => active?.id === id ? updated : active)
    } catch (e) {
      setError((e as Error).message)
      throw e
    }
  }

  const importWorld = (f: File) =>
    run(async () => {
      setTurn({ stage: 'gm', started: Date.now(), say: '', sketches: [],
        detail: t('导入 {name}（{mb} MB）…', { name: f.name, mb: (f.size / 1048576).toFixed(1) }) })
      const r = await api.importWorld(f)
      setTurn(null)
      await refresh(r.id)          // switch to it right after importing
      setError(t('已导入「{title}」：{n} 段，新增 {files} 个文件', { title: r.title, n: r.segments, files: r.files_written }))
    })

  const undo = () =>
    run(async () => {
      if (!world) return
      await api.rollback(world.id)
      setInspecting(false)
      await load(world.id)
    })

  /**
   * Both the home input and dice develop a concept, create it, then film the opening.
   */
  const startWorld = async (idea = '') => {
    if (rolling || busy) return
    setRolling(true)
    setError('')
    setTurn({ stage: 'gm', started: Date.now(), say: '', sketches: [],
      detail: idea ? t('正在根据你的想法完善世界、主线与开场…') : t('正在摇一个世界…') })
    try {
      if (!worldStart.current || worldStart.current.idea !== idea || worldStart.current.owner !== user?.id) {
        const concept = await api.randomWorld(styleMenu(), idea)
        worldStart.current = { idea, owner: user?.id ?? '', concept,
          videoModel: health?.video.id ?? '', completed: 0 }
      }
      const start = worldStart.current
      const id = await prepareWorld(start, (name, i, n) => {
        setTurn(previous => previous && ({ ...previous, stage: 'image',
          detail: t('准备世界资产 {i}/{n}：{name}', { i, n, name }) }))
      })
      await refresh(id)
      worldStart.current = null
      if (idea) setDraft('')
      // Rolling is done. What follows is video generation with its own progress and stop button;
      // the dice should not show "rolling…" during all of that.
      setRolling(false)
      // The opening line is followed by the rolled scene, so the first beat lands on this world's
      // own picture instead of a generic establishing shot. Chat recognizes it by prefix and still
      // shows "world ready".
      await takeTurn(id, `${OPENING} ${start.concept.opening}`.trim(), true)
    } catch (e) {
      setTurn(null)
      if (idea) setDraft((d) => d || idea)
      setError(t(idea ? '创建世界失败：{msg}' : '摇世界失败：{msg}', { msg: (e as Error).message }))
    } finally {
      setRolling(false)
    }
  }

  const closeMap = useCallback(() => setMapOpen(false), [])

  // In immersive mode the header and transport bar hide entirely: a permanent toolbar breaks
  // immersion most. They return when the mouse reaches the top edge, so exiting, the map and the
  // inspector are still reachable.
  useEffect(() => {
    if (!immersive) {
      setChrome(false)
      return
    }
    const move = (e: MouseEvent) => setChrome(e.clientY < 72)
    window.addEventListener('mousemove', move)
    return () => window.removeEventListener('mousemove', move)
  }, [immersive])

  // Esc: while running it cancels this beat; otherwise it exits immersive mode.
  // Cancel wins: when people change their mind, they want stop, not a layout switch.
  useEffect(() => {
    if (!immersive && !busy) return
    const esc = (e: KeyboardEvent) => {
      if (e.key !== 'Escape') return
      if (busy) stopTurn()
      else setImmersive(false)
    }
    document.addEventListener('keydown', esc)
    return () => document.removeEventListener('keydown', esc)
  }, [immersive, busy, stopTurn])

  const enterUser = async (candidate: User, password = '') => {
    const selected = await api.selectUser(candidate.id, password)
    setUser(selected)
  }

  const registerUser = async (name: string, password = '') => {
    const created = await api.createUser(name, password)
    setUsers((items) => [...items, created])
    setUser(created)
  }

  const unregisterUser = async (candidate: User, password = '') => {
    await api.deleteUser(candidate.id, password)
    setUsers((items) => items.filter((item) => item.id !== candidate.id))
  }

  if (!userReady || !user) {
    return (
      <div id="app" className="user-entry" data-theme={theme}>
        <UserGate
          users={users}
          loading={!userReady}
          onSelect={enterUser}
          onCreate={registerUser}
          onDelete={unregisterUser}
        />
      </div>
    )
  }

  if (isCoop) {
    return <CoopRoom user={user} code={coopRoute.split('/')[1] || ''} theme={theme} onThemeChange={setTheme}
      onExit={() => { window.location.hash = ''; void refresh().catch(e => setError(String(e))) }}
      onSwitchUser={() => { void switchUser() }}/>
  }

  return (
    // The stage is one column (video, transport bar, controls); drawers float over it, and
    // immersive mode keeps only the large video and the input.
    <div
      id="app"
      data-theme={theme}
      className={[immersive && 'immersive', chrome && 'chrome',
        !assetsOpen && 'no-assets', !railOpen && 'no-rail',
        playing && 'playing'].filter(Boolean).join(' ') || undefined}
      style={{
        '--video-ratio': videoRatio,
        '--video-ratio-inv': 1 / videoRatio,
      } as AppStyle}
    >
      <Stage
        // Keys are per sibling list: the world map below is keyed by the world too, so prefix both.
        key={`stage:${world?.id ?? ''}`}
        world={world}
        current={current}
        turn={turn}
        duck={speaking}
        beforeAutoPlay={finishSpeaking}
        onPlay={setCurrent}
        onInspect={() => setInspecting(true)}
        onVideoRatio={setVideoRatio}
        onPlaying={setPlaying}
        // Autoplay waits for narration already speaking to finish; a manual replay switches back to
        // the video sound at once.
        onAudible={() => {
          stopSpeaking()
          setSpeaking(false)
        }}
      />

      {/* Drawers float over the picture and close on an outside click; the stage stays centered and never narrows. */}
      {!immersive && (railOpen || assetsOpen) && (
        <div className="drawer-scrim" onClick={() => { setRailOpen(false); setAssetsOpen(false) }} />
      )}

      {/* A handle on each side of the picture opens / closes that side's drawer. In immersive mode both are
         closed and the handles step aside. */}
      {!immersive && (
        <>
          <button
            className="rail-handle left"
            onClick={() => setRailOpen((v) => !v)}
            title={railOpen ? t('收起左栏') : t('展开左栏')}
            aria-label={railOpen ? t('收起左栏') : t('展开左栏')}
          >
            <svg viewBox="0 0 8 14" aria-hidden="true">
              <path d={railOpen ? 'M6 1 1 7l5 6' : 'M2 1l5 6-5 6'} />
            </svg>
          </button>
          <button
            className="rail-handle right"
            onClick={() => setAssetsOpen((v) => !v)}
            title={assetsOpen ? t('收起资产表') : t('展开资产表')}
            aria-label={assetsOpen ? t('收起资产表') : t('展开资产表')}
          >
            <svg viewBox="0 0 8 14" aria-hidden="true">
              <path d={assetsOpen ? 'M2 1l5 6-5 6' : 'M6 1 1 7l5 6'} />
            </svg>
          </button>
        </>
      )}

      <Header
        userName={user.name}
        onSwitchUser={() => { void switchUser() }}
        health={health}
        worlds={worlds}
        current={world?.id ?? ''}
        setting={world?.setting ?? ''}
        visualStyle={world?.visual_style ?? ''}
        onSelect={(id) => load(id).catch((e) => setError(String(e)))}
        onRename={renameWorld}
        onOpenLog={() => setLogOpen(true)}
        onOpenAgent={() => setInspecting(true)}
        onNew={() => setNewWorld(true)}
        onRandom={() => { void startWorld() }}
        randomBusy={rolling}
        busy={busy || rolling}
        onDelete={removeWorld}
        onImport={importWorld}
        model={
          health?.video.choices.find((m) => m.id === world?.video_model)?.label ??
          world?.video_model
        }
        assetsOpen={assetsOpen}
        onToggleAssets={() => setAssetsOpen((v) => !v)}
        railOpen={railOpen}
        onToggleRail={() => setRailOpen((v) => !v)}
        immersive={immersive}
        onToggleImmersive={() => setImmersive((v) => !v)}
        theme={theme}
        onThemeChange={setTheme}
        tts={tts}
        ttsUp={ttsUp}
        speaking={speaking}
        onToggleTts={() => {
          setTts((v) => {
            if (v === null) return v
            const next = !v
            saveTts(next)
            if (!next) {
              stopSpeaking()
              setSpeaking(false)
            }
            return next
          })
        }}
      />

      {railOpen && (
      <SideRail
        world={shownWorld}
        current={current}
        theme={theme}
        tab={tab}
        onTab={setTab}
        busy={busy}
        pending={pending}
        turn={turn}
        onSelect={setCurrent}
        onOpenMap={() => setMapOpen(true)}
        onInspect={(n) => {
          setCurrent(n)
          setInspecting(true)
        }}
        onOpenAsset={(kind, key) => setAsset({ kind, key })}
        onUndo={undo}
      />
      )}

      <Hud
        preparing={rolling}
        setupHint={missingKeys.length ? t('演示模式：要拍新片段，在 .env 里填好 {what} 后重启（见 README）', { what: missingKeys.join(' / ') }) : undefined}
        world={shownWorld}
        current={current ? shownWorld?.nodes[current.id] ?? current : null}
        busy={busy}
        turn={turn}
        sequence={sequence}
        draft={draft}
        onDraft={setDraft}
        onSend={send}
        onStop={stopTurn}
        autoRunning={autoRunning}
        onStopAuto={() => {
          autoStop.current = true
          setAutoRunning(false)
        }}
        onAddRef={(kind, file) => setAsset({ kind, file })}
      />

      {assetsOpen && (
        <Assets
          world={shownWorld}
          current={current}
          busy={busy}
          onPlay={setCurrent}
          onAddSubject={() => setAsset({ kind: 'subject', subjectKind: '人物' })}
          onAddItem={() => setAsset({ kind: 'subject', subjectKind: '物件' })}
          onAddPlace={() => setAsset({ kind: 'place' })}
          onEditSubject={(sid) => setAsset({ kind: 'subject', key: sid })}
          onEditPlace={(name) => setAsset({ kind: 'place', key: name })}
          onDeleteItem={async (sid) => {
            if (!world) return
            await api.dropItem(world.id, sid)
            // Refresh only the asset data, without interrupting the clip being played or jumping
            // back from a reviewed branch.
            setWorld(await api.getWorld(world.id))
          }}
        />
      )}

      {error && (
        <div id="toast" role="status">
          <span>{error}</span>
          <div className="spacer" />
          <button className="sm ghost" onClick={() => setError('')}>
            {t('知道了')}
          </button>
        </div>
      )}

      <WorldMap
        key={`map:${world?.id ?? ''}`}
        open={mapOpen}
        world={shownWorld}
        current={current}
        onClose={closeMap}
        onWorldUpdated={() => (world ? load(world.id) : undefined)}
        onPlay={(node) => {
          setCurrent(node)
          setMapOpen(false)
        }}
      />

      <Inspector
        open={inspecting}
        world={world}
        node={current}
        busy={busy}
        onClose={() => setInspecting(false)}
        onSelectTake={setCurrent}
        onRegenerate={regenerate}
        liveSteps={busy ? turn?.steps : undefined}
      />

      <WorldLog
        open={logOpen}
        world={world?.id ?? ''}
        onClose={() => setLogOpen(false)}
      />

      <NewWorld
        open={newWorld}
        onClose={() => setNewWorld(false)}
        onCreated={async (id, opening) => {
          setNewWorld(false)
          try {
            await refresh(id)
            await takeTurn(id, opening || OPENING, true)  // play the first beat right after creating, never leave the user an empty world
          } catch (e) {
            setError(String(e))
          }
        }}
      />

      <AddAsset
        open={!!asset}
        kind={asset?.kind ?? 'subject'}
        subjectKind={asset?.subjectKind}
        worldId={world?.id ?? ''}
        visualStyle={world?.visual_style ?? ''}
        assetKey={asset?.key}
        initialFile={asset?.file}
        current={
          asset?.key
            ? asset.kind === 'subject'
              ? world?.subjects[asset.key]
              : world?.places[asset.key]
            : undefined
        }
        onClose={() => setAsset(null)}
        onDone={async () => {
          if (world) await load(world.id)
        }}
      />
    </div>
  )
}
