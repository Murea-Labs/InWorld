import { useCallback, useEffect, useRef, useState, type CSSProperties } from 'react'
import type { TurnState } from '../App'
import { mediaUrl } from '../api'
import { badge, lineage } from '../lib'
import type { Node, Sketch, World } from '../types'
import { t } from '../i18n'

interface Props {
  world: World | null
  current: Node | null
  turn: TurnState | null
  onPlay: (n: Node) => void
  onInspect: () => void
  onVideoRatio: (ratio: number) => void
  /** The world is speaking: duck the previous clip's sound so two voices do not talk at once */
  duck?: boolean
  /**
   * The picture started making sound. The narrator must stop at once; ducking is not enough: with
   * someone speaking in the video and the narrator reading the same line, two voices say the same
   * words.
   */
  onAudible?: () => void
  beforeAutoPlay?: () => Promise<void> | undefined
  /** The front video is playing / stopped. While playing the UI dims its surroundings. */
  onPlaying?: (playing: boolean) => void
}

type Slot = 0 | 1
const REEL_BLEND_MS = 120
// A 24fps video calls back every ~42ms; 80ms is too close to the end and a busy browser may jump
// straight to ended. 160ms reliably hits one of the last 3–4 frames; the cross-fade itself is still
// 120ms.
const REEL_PREROLL_S = 0.16

/**
 * entity → the card's presence tier. A monster and a prop should not look alike: creatures are
 * bigger, slower and have a glow, objects are small and quick. Sizes are in the CSS .is-* rules.
 */
const POP_TIER: Record<string, string> = {
  生物: 'life', 场景: 'place', 人物: 'cast', 物件: 'prop',
}
/**
 * How long each tier stays on the picture (seconds). CSS uses --pop-life as animation-duration and
 * this value removes the card; the duration has a single source.
 */
const POP_LIFE: Record<string, number> = {
  prop: 3.6, cast: 4.4, place: 4.8, life: 6.4,
}
/** The small line on the card: what happened, not what type it is. */
const POP_LABEL: Record<string, string> = {
  生物: t('遭遇'), 场景: t('抵达'), 人物: t('登场'), 物件: t('出现'),
}
/** At most this many at once. A beat may draw a place and four people; showing all would cover the picture. */
const POP_MAX = 4

/** Old clips stored sketches without entity; fall back to kind. */
const entityOf = (s: Sketch) => s.entity || (s.kind === 'place' ? '场景' : '人物')
/** Each image pops up once. ref is a file name and unique enough; old data without ref falls back to the name. */
const popKey = (s: Sketch) => s.ref || `${s.kind}:${s.name}`

interface Pop {
  key: number
  s: Sketch
  tier: string
  /** Index within a batch: stagger the entrances instead of five cards at once */
  delay: number
}

type PopStyle = CSSProperties & { '--pop-life': string }

/**
 * A card's life on the picture: CSS does enter / stay / fade, this only removes it on time. Both
 * paths stay: animationend is precise, but background tabs get no animation events; the timer is
 * slower but always fires. Without either, cards could pile up and never leave.
 */
function PopCard({ pop, onDone }: { pop: Pop; onDone: (key: number) => void }) {
  const { key, s, tier, delay } = pop
  const life = POP_LIFE[tier] ?? POP_LIFE.cast

  useEffect(() => {
    const t = setTimeout(() => onDone(key), (delay + life) * 1000 + 200)
    return () => clearTimeout(t)
  }, [key, delay, life, onDone])

  const entity = entityOf(s)
  return (
    <figure
      className={'stage-pop is-' + tier}
      style={{ '--pop-life': `${life}s`, animationDelay: `${delay}s` } as PopStyle}
      // Remove when the animation ends, more precise than the timer; the glow's animationend
      // bubbles up, only accept our own
      onAnimationEnd={(e) => {
        if (e.target === e.currentTarget) onDone(key)
      }}
    >
      <span className="stage-pop-glow" aria-hidden="true" />
      <img src={mediaUrl('refs', s.ref)} alt="" />
      <figcaption>
        <em>{POP_LABEL[entity] ?? t('出现')}</em>
        <b>{s.name}</b>
        {s.brief && <span>{s.brief}</span>}
      </figcaption>
    </figure>
  )
}

/**
 * "Something new" popping up on the picture. These images are not intermediate output: they decide
 * what this person, monster or place looks like from now on. In normal mode they are listed in the
 * chat; in immersive mode they pop up on the picture and fade.
 */
function StagePops(
  { pops, onDone }: { pops: Pop[]; onDone: (key: number) => void },
) {
  if (!pops.length) return null
  return (
    <div className="stage-pops" aria-live="polite">
      {pops.map((p) => <PopCard key={p.key} pop={p} onDone={onDone} />)}
    </div>
  )
}

/** The atmosphere layer filling the viewport, with the main video above it. */
export default function Stage({
  world, current, turn, onPlay, onInspect, onVideoRatio, duck = false,
  onAudible, onPlaying, beforeAutoPlay,
}: Props) {
  // Two <video> elements in the same grid cell take turns. With one, changing the source empties
  // the element before the new frame arrives, which caused a black flash between clips. Now the old
  // frame stays on top until the new first frame has really been painted underneath, then the
  // layers swap.
  const vid = [useRef<HTMLVideoElement>(null), useRef<HTMLVideoElement>(null)]
  const [src, setSrc] = useState<[string, string]>(['', ''])
  const [front, setFront] = useState<Slot>(0)
  const want = useRef('')
  const chaining = useRef(false)          // playing the whole chain
  const fromEnded = useRef(false)         // whether this switch is "finished, go to the next" or a user click
  const primed = useRef<[string, string]>(['', ''])
  const primeSeq = useRef<[number, number]>([0, 0])

  const chain = lineage(world)
  const idx = current ? chain.findIndex((n) => n.id === current.id) : -1
  const nextClip = idx >= 0 ? chain[idx + 1]?.clip : undefined

  // While the world reads this beat's line, the picture plays the **previous** clip's sound; two
  // people talking at once are both unintelligible. Duck to 20%, it comes back after the line.
  useEffect(() => {
    for (const r of vid) if (r.current) r.current.volume = duck ? 0.2 : 1
  }, [duck])                                   // eslint-disable-line react-hooks/exhaustive-deps

  // The atmosphere uses the current place's reference image, already on disk.
  const atmos = current && world?.places[current.place]?.ref
  const poster = atmos ? mediaUrl('refs', atmos) : undefined

  useEffect(() => {
    // The user picked another clip: the chain stops here, or playback would keep jumping onward
    if (!fromEnded.current) chaining.current = false
    fromEnded.current = false

    const url = current ? mediaUrl('clips', current.clip) : ''
    want.current = url
    if (!url) {
      setSrc(['', ''])
      chaining.current = false
      return
    }
    if (src[front] === url) {
      // The same clip was selected again (a new world opening with the same clip, or back to a
      // finished clip): the element is still on its last frame, and nothing moves unless it
      // restarts.
      const v = vid[front].current
      if (v && v.ended) {
        v.currentTime = 0
        v.play().catch(() => {})
      }
      return
    }

    const back = (1 - front) as Slot
    // The two slots strictly alternate, so going back one clip always lands in the background slot,
    // which holds exactly that url. setSrc produces the same array, React sees no prop change, the
    // DOM does not change and canplay never fires again. Flip it here.
    if (src[back] === url) {
      // While a clip plays, the chain preloads the next one into the background slot. Usually it
      // can play right away here; if the network is behind, keep the current last frame and flip on
      // onCanPlay, rather than switching to a black picture without a decoded first frame.
      const v = vid[back].current
      if (v && v.readyState >= HTMLMediaElement.HAVE_FUTURE_DATA) show(back)
      return
    }
    setSrc((s) => {
      const n = [...s] as [string, string]
      n[back] = url
      return n
    })
  }, [current?.clip])                          // eslint-disable-line react-hooks/exhaustive-deps

  const switchSeq = useRef(0)
  const beforeAutoPlayRef = useRef(beforeAutoPlay)
  beforeAutoPlayRef.current = beforeAutoPlay
  useEffect(() => () => { ++switchSeq.current }, [])

  /**
   * Decode the preloaded next clip up to its first frame, then hold it there. readyState=4 only
   * means enough bytes, not that the first frame reached the compositor; a first play after the
   * previous clip ended would still add ~80–110ms. The background video is fully covered and muted,
   * so it can safely advance one frame.
   */
  const prime = useCallback((i: Slot) => {
    const v = vid[i].current
    if (!v || !v.currentSrc || primed.current[i] === v.currentSrc) return

    const url = v.currentSrc
    const seq = ++primeSeq.current[i]
    v.currentTime = 0
    const done = () => {
      if (seq !== primeSeq.current[i] || v.currentSrc !== url) return
      v.pause()
      primed.current[i] = url
    }
    const requestFrame = v.requestVideoFrameCallback?.bind(v)
    if (requestFrame) {
      requestFrame(done)
      v.play().catch(() => {})
    } else {
      v.play().then(() => {
        requestAnimationFrame(() => requestAnimationFrame(done))
      }).catch(() => {})
    }
  }, [])                                       // eslint-disable-line react-hooks/exhaustive-deps

  /**
   * Flip to slot i: let it really paint its first frame in the hidden, muted background, then hand
   * over picture and sound. canplay only promises playable data, not that the GPU has shown the
   * first frame; hiding the old video at canplay would expose one black frame.
   * requestVideoFrameCallback marks a committed frame. The other slot must be paused explicitly or
   * the two soundtracks overlap.
   */
  const show = useCallback((i: Slot) => {
    const seq = ++switchSeq.current
    const expected = want.current
    const start = () => {
      // A world switch, a newer clip, or another show() invalidates this wait.
      if (seq !== switchSeq.current || want.current !== expected) return
      const v = vid[i].current
      if (!v) return

      const handoff = () => {
        const old = vid[1 - i].current
        setFront(i)
        // This slot just took over picture and sound. The play event fired while front still
        // pointed at the old slot, and onPlay below ignored it as background priming, so the state
        // is set here.
        onAudible?.()
        onPlaying?.(true)
        // The old clip keeps playing its last frames under the cross-fade; do not pause before the
        // class switch. Stop after the animation; if it quickly becomes the front again, this old
        // timer will not stop it by mistake.
        if (old) setTimeout(() => {
          if (!old.classList.contains('on')) old.pause()
        }, REEL_BLEND_MS + 20)
      }

      ++primeSeq.current[i]                       // an old prime callback must not pause it any more
      if (primed.current[i] === v.currentSrc) {
        primed.current[i] = ''
        v.play().catch(() => {})
        handoff()
        return
      }

      v.currentTime = 0
      const commit = () => {
        if (seq !== switchSeq.current) return
        handoff()
      }
      const requestFrame = v.requestVideoFrameCallback?.bind(v)
      if (requestFrame) {
        // Subscribe before play. Chrome resolves the play promise only once playback really starts;
        // registering the frame callback in then can hang at HAVE_METADATA for some preload / seek
        // combinations.
        requestFrame(commit)
        v.play().catch(() => {})
      } else {
        v.play().then(() => {
          // Old browsers without frame callbacks: wait two more paints, at least not removing the
          // old frame right when the play promise resolves and the compositor has nothing yet.
          requestAnimationFrame(() => requestAnimationFrame(commit))
        }).catch(() => {})
      }
    }
    const narration = beforeAutoPlayRef.current?.()
    if (narration) void narration.then(start)
    else start()
  }, [])                                       // eslint-disable-line react-hooks/exhaustive-deps

  /** Flip to the front once the background slot is ready. */
  const ready = useCallback((i: Slot) => {
    if (front === i) return
    if (src[i] === want.current) show(i)
    else prime(i)
  }, [src, front, show, prime])

  // As soon as the current clip starts, let the other video fully buffer the next one. The old
  // version had two videos but the background slot kept the previous clip, so the next request
  // started only after ended and every switch paused for about a second. Only the next clip is
  // prefetched: downloads hide inside the current clip without loading dozens of videos in a long
  // world.
  useEffect(() => {
    if (!current || !nextClip) return
    const currentUrl = mediaUrl('clips', current.clip)
    if (src[front] !== currentUrl) return

    const back = (1 - front) as Slot
    const nextUrl = mediaUrl('clips', nextClip)
    if (src[back] === nextUrl) return
    // The new clip is still covering the previous last frame; switch the background slot to the
    // clip after next only when the CSS cross-fade is over, or the old picture underneath would be
    // replaced by an empty player halfway through the 120ms animation.
    const timer = setTimeout(() => {
      setSrc((s) => {
        if (s[back] === nextUrl) return s
        const n = [...s] as [string, string]
        n[back] = nextUrl
        return n
      })
    }, REEL_BLEND_MS + 20)
    return () => clearTimeout(timer)
  }, [current?.clip, nextClip, front, src]) // eslint-disable-line react-hooks/exhaustive-deps

  /**
   * Hand over to the next clip. An early hand-over happens only when the next clip is primed; on a
   * slow network the safe ended path is used rather than cutting to a picture that is not ready.
   */
  const advance = useCallback((i: Slot, preparedOnly: boolean) => {
    if (i !== front || !chaining.current) return
    const next = idx >= 0 ? chain[idx + 1] : undefined
    if (next) {
      const back = (1 - front) as Slot
      const nextUrl = mediaUrl('clips', next.clip)
      const v = vid[back].current
      const prepared = src[back] === nextUrl && !!v
        && primed.current[back] === v.currentSrc
      if (preparedOnly && !prepared) return false
      if (prepared) show(back)
      fromEnded.current = true
      onPlay(next)
    } else chaining.current = false
    return true
  // eslint-disable-next-line react-hooks/exhaustive-deps -- vid holds refs, which are stable
  }, [front, idx, chain, onPlay, src, show])

  // By the time ended fires the previous clip has frozen on its last frame. Watch the remaining
  // time frame by frame and start the primed next clip about 3–4 frames before the end, so the
  // 120ms cross-fade finishes before the clip ends instead of starting from a still frame.
  const seamSeq = useRef(0)
  const watchSeam = useCallback((i: Slot) => {
    const v = vid[i].current
    if (i !== front || !v || v.paused || !chaining.current) return
    const requestFrame = v.requestVideoFrameCallback?.bind(v)
    if (!requestFrame) return

    const seq = ++seamSeq.current
    const tick = () => {
      if (seq !== seamSeq.current || v.paused || !v.classList.contains('on')
          || !chaining.current) return
      const remaining = v.duration - v.currentTime
      if (Number.isFinite(remaining) && remaining <= REEL_PREROLL_S) {
        if (advance(i, true)) return
      }
      requestFrame(tick)
    }
    requestFrame(tick)
  // eslint-disable-next-line react-hooks/exhaustive-deps -- vid holds refs, which are stable
  }, [front, advance])

  useEffect(() => {
    if (chaining.current) watchSeam(front)
    // src switches to "the next clip" about 140ms after the current one starts. Re-attach the
    // watcher after that update, or the frame callback closure still sees the old background url
    // and thinks the next clip is not ready.
  }, [front, current?.id, src])               // eslint-disable-line react-hooks/exhaustive-deps

  /** Fallback when the network missed the preroll or the browser has no frame callbacks. */
  const ended = useCallback((i: Slot) => {
    advance(i, false)
  }, [advance])

  /* ── Something new appears ───────────────────────────────────────
     Two sources: images drawn in this beat (turn.sketches, arriving well before the video) and those a clip
     brings when it plays (node.sketches: during a chain the monster pops up on the beat it appears).
     Each image pops up once per world; scrolling back does not repeat it. */
  const [pops, setPops] = useState<Pop[]>([])
  const popSeq = useRef(0)
  const popSeen = useRef<Set<string>>(new Set())

  const reveal = useCallback((items: Sketch[] | undefined) => {
    if (!items?.length) return
    const fresh = items.filter((s) => s.ref && !popSeen.current.has(popKey(s)))
    if (!fresh.length) return
    for (const s of fresh) popSeen.current.add(popKey(s))
    setPops((cur) => [
      ...cur,
      ...fresh.map((s, i) => ({
        key: ++popSeq.current,
        s,
        tier: POP_TIER[entityOf(s)] ?? 'cast',
        delay: i * 0.17,
      })),
    ].slice(-POP_MAX))
  }, [])

  const dropPop = useCallback(
    (key: number) => setPops((cur) => cur.filter((p) => p.key !== key)),
    [],
  )

  // Switching world: mark the images of the current clip as seen, or a pile pops up on entry
  useEffect(() => {
    popSeen.current = new Set(
      (current?.sketches ?? []).filter((s) => s.ref).map(popKey),
    )
    setPops([])
  }, [world?.id])                              // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    reveal(turn?.sketches)
  }, [turn?.sketches, reveal])

  useEffect(() => {
    reveal(current?.sketches)
  }, [current?.id, reveal])                    // eslint-disable-line react-hooks/exhaustive-deps

  /**
   * No native controls: the HUD sits on the bottom edge of the picture, exactly over them, so they
   * could not be clicked; and a player control bar says "I am watching a web video" more than
   * anything. Clicking the picture pauses / resumes, the only playback control needed here.
   */
  const togglePlay = () => {
    const v = vid[front].current
    if (!v) return
    if (v.paused) {
      onAudible?.()
      v.play().catch(() => {})
    } else v.pause()
  }

  const playAll = () => {
    if (!chain.length) return
    if (chain[0].id === current?.id) {
      chaining.current = true
      vid[front].current?.play().catch(() => {})
      // If the first clip is already playing the browser will not fire playing again; attach the
      // boundary watcher here, or this chain would still wait for ended to switch.
      watchSeam(front)
    } else {
      fromEnded.current = true      // this switch comes from the chain, not a user click
      chaining.current = true
      onPlay(chain[0])
    }
  }

  const b = current && badge(current)
  // No clip yet but running: the state right after a new world is created
  const booting = !current && !!turn

  return (
    <div id="stage">
      {atmos && (
        <div
          className="stage-atmos"
          style={{ backgroundImage: `url(${poster})` }}
          aria-hidden="true"
        />
      )}

      {/* Clicking the picture pauses / resumes. No title: a native tooltip over the whole screen breaks the
         mood more than a missing hint, and a wrong click is undone by clicking again. */}
      <div id="stage-body" onClick={togglePlay}>
        {current ? (
          ([0, 1] as Slot[]).map((i) => (
            <video
              key={i}
              ref={vid[i]}
              className={'reel' + (front === i ? ' on' : '')}
              src={src[i] || undefined}
              poster={poster}
              preload="auto"
              muted={front !== i}
              playsInline
              onLoadStart={() => {
                ++primeSeq.current[i]
                primed.current[i] = ''
              }}
              onLoadedMetadata={(event) => {
                if (src[i] !== want.current) return
                const { videoWidth, videoHeight } = event.currentTarget
                if (videoWidth > 0 && videoHeight > 0) onVideoRatio(videoWidth / videoHeight)
              }}
              onCanPlay={() => ready(i)}
              // Only the front slot counts: the background slot also plays briefly to prime its
              // first frame; that is not "the picture is playing" and must not stop the narrator.
              onPlay={() => { if (front === i) { onAudible?.(); onPlaying?.(true) } }}
              onPause={() => { if (front === i) onPlaying?.(false) }}
              // After a pause or a rebuffer the old frame watcher has exited; re-attach on every
              // real resume so later boundaries still hand over early.
              onPlaying={() => watchSeam(i)}
              onEnded={() => {
                if (front === i) onPlaying?.(false)
                ended(i)
              }}
            />
          ))
        ) : booting ? (
          <div id="empty">
            <div className="ring spin" />
            <b>{t('正在把这个世界立起来')}</b>
            {/* Progress only, no dialogue: the HUD shows the dialogue, and writing it here too would show the same
               line twice on screen (always on a new world's first beat). */}
            <span>{turn.detail}</span>
          </div>
        ) : (
          <div id="empty">
            <div className="ring" />
            <b>{t('还没有画面')}</b>
            <span>{t('先说一句话，世界就开始了')}</span>
          </div>
        )}
      </div>

      <StagePops pops={pops} onDone={dropPop} />

      <div id="transport">
        <span id="seg-label">
          {current ? (
            <>
              <span className="idx">
                {idx >= 0 ? `${idx + 1} / ${chain.length}` : t('旁支')}
              </span>
              <span className="sep">·</span>
              {current.place}
              {b && <span className={'tag ' + b.cls}>{b.text}</span>}
              <span className="sep">·</span>
              <span className="idx">{`${Number(current.elapsed || 0).toFixed(1)}s`}</span>
            </>
          ) : (
            <span className="sep">—</span>
          )}
        </span>
        <button className="sm" onClick={playAll} disabled={!chain.length}>
          {t('连播全片')}
        </button>
        <button className="sm" onClick={onInspect} disabled={!current}>
          {t('检查器')}
        </button>
      </div>
    </div>
  )
}
