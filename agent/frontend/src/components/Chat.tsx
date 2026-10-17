import { useEffect, useRef } from 'react'
import { mediaUrl } from '../api'
import type { TurnState } from '../App'
import { OPENING, badge, lineage, takesOf } from '../lib'
import type { Node, Sketch, World } from '../types'
import { kindLabel, t } from '../i18n'

interface Props {
  world: World | null
  /** The line that was sent but is not in history yet */
  pending: string
  turn: TurnState | null
  /** Whether this drawer currently shows the chat tab; scrollTop cannot be set while hidden */
  active?: boolean
  onInspect: (n: Node) => void
  /** Open a reference image: large view, replace image, edit description */
  onOpenAsset: (kind: 'subject' | 'place', key: string) => void
}

/** Segment metadata is parsed back out of the one line the pipeline appends. */
// "[beat 3 · place · summary]"; older worlds stored "[已生成第 3 段 · …]".
const BEAT_RE = /^([\s\S]*?)\n\[(?:beat |已生成第 )(\d+)(?: 段)? · (.+?) · ([\s\S]*)\]$/

/**
 * Reference images drawn in this beat, with a short note.
 * 
 * They are not intermediate output: they decide what this person and this place look like,
 * and every later clip is generated from them. When the world creates something it is shown
 * right away instead of quietly appearing in the asset panel.
 */
function Sketches(
  { items, world, onOpen }:
  { items: Sketch[]; world: World | null; onOpen: Props['onOpenAsset'] },
) {
  if (!items.length) return null
  /** Old clips stored sketches without ids; find them in the assets by name. */
  const keyOf = (s: Sketch) => {
    if (s.id) return s.id
    if (s.kind === 'place') return s.name
    const hit = Object.entries(world?.subjects ?? {}).find(([, v]) => v.name === s.name)
    return hit?.[0]
  }
  return (
    <div className="sketches">
      {items.map((s, i) => {
        const key = keyOf(s)
        return (
          <figure
            key={s.ref || i}
            className={'sketch' + (key ? ' open' : '')}
            title={key ? t('打开「{name}」：看大图、换图', { name: s.name }) : undefined}
            onClick={(e) => {
              if (!key) return
              e.stopPropagation()      // do not also open the inspector of the outer reply
              onOpen(s.kind, key)
            }}
          >
            <img src={mediaUrl('refs', s.ref)} alt={s.name} loading="lazy" />
            <figcaption>
              <b>
                <span className={'dot ' + s.kind} />
                {s.name}
              </b>
              {s.entity && s.entity !== '人物' && (
                <em className="sketch-kind">{kindLabel(s.entity)}</em>
              )}
              {s.brief && <span>{s.brief}</span>}
            </figcaption>
          </figure>
        )
      })}
    </div>
  )
}

/**
 * The full chat log, the content of the left drawer's chat tab. Dialogue and choices are
 * already on the picture (Hud); this is for looking back: which line made this clip.
 */
export default function Chat({
  world, pending, turn, active = true, onInspect, onOpenAsset,
}: Props) {
  const logRef = useRef<HTMLDivElement>(null)

  // active is a dependency too: while hidden (display:none) scrollHeight is 0 and setting
  // scrollTop does nothing, so switching back would stay at the top.
  useEffect(() => {
    if (active && logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight
  }, [world?.history.length, pending, turn?.stage, turn?.say,
      turn?.sketches.length, active])

  const chain = lineage(world)
  let seg = 0

  return (
    <div id="log" ref={logRef}>
      {!world?.history.length && !pending && (
        <div className="msg sys">
          {t('这个世界还是空的。')}
          <br />
          {t('说一句话，比如「三个同事坐在会议室里开会」。')}
        </div>
      )}
      {world?.history.map((h, i) => {
        // The director node right after a user bubble is the beat that line produced.
        // origin==='auto' means the system continued on its own via continue_with.
        const auto = h.role === 'user' && chain[seg]?.origin === 'auto'
        if (h.role === 'user')
          // The opening is sent by the system. It must be in history (the director needs the
          // context) but must not look like something the user typed. A rolled world appends a
          // concrete scene after the opening line, so match by prefix.
          return h.text.startsWith(OPENING) ? (
            <div key={i} className="msg sys opening">
              {t('世界已就位')}
            </div>
          ) : (
            <div key={i} className={'msg user' + (auto ? ' auto' : '')}>
              {auto && <span className="auto-tag">{t('自动')}</span>}
              {h.text}
            </div>
          )
        const m = h.text.match(BEAT_RE)
        const node = m ? chain[seg++] : undefined
        const b = node && badge(node)
        const takes = node && world ? takesOf(world, node) : []
        return (
          <div
            key={i}
            className="msg gm"
            onClick={() => node && onInspect(node)}
            style={{ cursor: node ? 'pointer' : 'default' }}
          >
            {m ? m[1] : h.text}
            {node?.sketches?.length
              ? <Sketches items={node.sketches} world={world} onOpen={onOpenAsset} />
              : null}
            {node && b && (
              <div className="beat">
                <span className={'tag ' + b.cls}>{b.text}</span>
                <span className="tag">{t('第 {n} 段', { n: m![2] })}</span>
                <span className="tag">{m![3]}</span>
                {takes.length > 1 && <span className="tag take">{t('{n} 次生成', { n: takes.length })}</span>}
                <span className="go">{t('检查')} ▸</span>
              </div>
            )}
          </div>
        )
      })}
      {pending && <div className="msg user pending">{pending}</div>}
      {turn && (turn.say || turn.sketches.length > 0) && (
        <div className="msg gm live">
          {turn.say}
          <Sketches items={turn.sketches} world={world} onOpen={onOpenAsset} />
        </div>
      )}
    </div>
  )
}
