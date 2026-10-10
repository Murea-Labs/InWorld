import { useEffect, useRef, useState } from 'react'
import type { SequenceState, TurnState } from '../App'
import { STEPS, STEP_CN, STEP_OF } from '../lib'
import type { Node, TracePhase, World } from '../types'
import { kindLabel, t } from '../i18n'
import { PHASES, PHASE_LABEL } from './AgentTrace'

interface MentionOption {
  id: string
  name: string
  kind: 'subject' | 'place'
  entity: string
}

interface MentionContext {
  start: number
  end: number
  query: string
}

interface MentionRange {
  start: number
  end: number
  name: string
}

function mentionOptions(world: World | null): MentionOption[] {
  if (!world) return []
  const subjects = Object.entries(world.subjects).map(([id, subject]) => ({
    id,
    name: subject.name || id,
    kind: 'subject' as const,
    entity: subject.kind || '人物',
  }))
  const places = Object.keys(world.places).map((name) => ({
    id: name,
    name,
    kind: 'place' as const,
    entity: '场景',
  }))
  return [...subjects, ...places]
}

/** Find an @ token immediately before the caret. */
function mentionContext(value: string, caret: number): MentionContext | null {
  const before = value.slice(0, caret)
  const at = before.lastIndexOf('@')
  if (at < 0) return null
  const prefix = at > 0 ? before[at - 1] : ''
  if (prefix && /[A-Za-z0-9_]/.test(prefix)) return null
  const query = before.slice(at + 1)
  if (!query || (!query.includes(' ') && !query.includes('\n'))) {
    return { start: at, end: caret, query }
  }
  return null
}

/** Match a selected or manually typed asset mention without matching foo@bar. */
function hasMention(text: string, name: string): boolean {
  const needle = `@${name}`
  if (!name) return false
  let from = 0
  while (true) {
    const at = text.indexOf(needle, from)
    if (at < 0) return false
    const before = at ? text[at - 1] : ''
    const after = text[at + needle.length] || ''
    const leftOk = !before || !/[A-Za-z0-9_]/.test(before)
    const rightOk = !after || !/[A-Za-z0-9_]/.test(after)
    if (leftOk && rightOk) return true
    from = at + 1
  }
}

function mentionedIds(text: string, options: MentionOption[]): string[] {
  const seen = new Set<string>()
  // Prefer the longest name if one asset name is a prefix of another.
  for (const option of [...options].sort((a, b) => b.name.length - a.name.length)) {
    if (hasMention(text, option.name) && !seen.has(option.id)) seen.add(option.id)
  }
  return [...seen]
}

/**
 * Return the asset mentions that can be rendered as tokens in the editor.
 *
 * The backend intentionally also accepts manually typed `@asset` names, so the
 * highlighter uses the same boundary rules as `hasMention` instead of keeping a
 * second, UI-only list of mentions. Longest names win when one name is a prefix
 * of another, and overlapping matches are ignored.
 */
function mentionRanges(text: string, options: MentionOption[]): MentionRange[] {
  const candidates: MentionRange[] = []
  for (const option of [...options].sort((a, b) => b.name.length - a.name.length)) {
    if (!option.name) continue
    const needle = `@${option.name}`
    let from = 0
    while (true) {
      const at = text.indexOf(needle, from)
      if (at < 0) break
      const end = at + needle.length
      const before = at ? text[at - 1] : ''
      const after = text[end] || ''
      const leftOk = !before || !/[A-Za-z0-9_]/.test(before)
      const rightOk = !after || !/[A-Za-z0-9_]/.test(after)
      if (leftOk && rightOk) candidates.push({ start: at, end, name: option.name })
      from = at + 1
    }
  }

  // Candidates are already longest-first for equal starts. Sorting by start
  // makes the final pass deterministic while preserving that tie-breaker.
  candidates.sort((a, b) => a.start - b.start || b.end - a.end)
  const ranges: MentionRange[] = []
  let occupiedUntil = -1
  for (const candidate of candidates) {
    if (candidate.start < occupiedUntil) continue
    ranges.push(candidate)
    occupiedUntil = candidate.end
  }
  return ranges
}

interface Props {
  preparing?: boolean
  playbackLocked?: boolean
  world: World | null
  /** The clip being played. Dialogue comes from its reply, choices from its options. */
  current: Node | null
  busy: boolean
  turn: TurnState | null
  sequence: SequenceState | null
  draft: string
  onDraft: (v: string) => void
  onSend: (text: string, mentions?: string[]) => void
  /** Shown in the input when the server has no API keys: only the demo world can be played. */
  setupHint?: string
  onStop: () => void
  /** The system is continuing on its own from the last beat's continue_with */
  autoRunning: boolean
  onStopAuto: () => void
  onAddRef: (kind: 'subject' | 'place', file?: File) => void
}

/** A stopwatch. A turn takes tens of seconds; without a moving number running and stuck look the same. */
function useElapsed(from?: number): string {
  const [now, setNow] = useState(() => Date.now())
  useEffect(() => {
    if (!from) return
    setNow(Date.now())
    const t = setInterval(() => setNow(Date.now()), 200)
    return () => clearInterval(t)
  }, [from])
  return from ? ((now - from) / 1000).toFixed(1) : '0.0'
}

function Progress(
  { turn, sequence, auto, onStop }: {
    turn: TurnState
    sequence: SequenceState | null
    auto: boolean
    onStop: () => void
  },
) {
  const elapsed = useElapsed(turn.started)
  const at = STEPS.indexOf(STEP_OF[turn.stage] ?? 'gm')
  // With agent step events the bar follows the loop itself.
  const steps = turn.steps ?? []
  const loopPhases: TracePhase[] | null = steps.length ? PHASES.filter((phase) => phase !== 'reflect') : null
  const current = [...steps].reverse().find((step) => step.status === 'running') ?? steps[steps.length - 1]
  const phaseAt = loopPhases && current ? loopPhases.indexOf(current.phase) : -1
  const currentSegment = sequence?.segments.find((segment) => segment.index === sequence.current)
  return (
    <div className={'turn-card' + (auto ? ' auto' : '')}>
      {sequence ? (
        // One line: the card replaces the choices and must fit the same HUD height, or it pushes
        // the input out of the window. The goal and each segment's text are in tooltips.
        <div className="turn-sequence" title={t('总内容：{goal}', { goal: sequence.goal })}>
          <b>{t('连续生成 {n} 段', { n: sequence.total })}</b>
          <ol>
            {sequence.segments.map((segment) => {
              const status = segment.status === 'done'
                ? t('已生成')
                : segment.status === 'generating' ? t('生成中') : t('等待')
              return (
                <li
                  key={segment.index}
                  className={'turn-segment-index ' + segment.status}
                  title={`${segment.index} · ${status} · ${segment.subtitle || segment.content}`}
                >
                  {segment.index}
                </li>
              )
            })}
          </ol>
          <span className="turn-sequence-now">
            {t('当前第 {n} 段', { n: sequence.current })}
            {currentSegment && <> · {currentSegment.subtitle || currentSegment.content}</>}
          </span>
          {auto && <button className="sm ghost" onClick={onStop}>{t('停止续跑')}</button>}
        </div>
      ) : auto && (
        <div className="turn-auto">
          {t('自动往下走 —— 这一句还没走完')}
          <button className="sm ghost" onClick={onStop}>{t('停')}</button>
        </div>
      )}
      <div className="turn-steps">
        {loopPhases ? loopPhases.map((phase, i) => (
          <span
            key={phase}
            className={'turn-step' + (i < phaseAt ? ' done' : i === phaseAt ? ' now' : '')}
            title={PHASE_LABEL[phase]}
          >
            {PHASE_LABEL[phase]}
          </span>
        )) : STEPS.map((s, i) => (
          <span
            key={s}
            className={'turn-step' + (i < at ? ' done' : i === at ? ' now' : '')}
          >
            {STEP_CN[s]}
          </span>
        ))}
        <span className="turn-time">{elapsed}s</span>
      </div>
      <div className="turn-detail">{turn.detail}</div>
    </div>
  )
}

/**
 * Subtitles on the picture + the controls under it.
 * 
 * Dialogue sits on the bottom edge of the picture like subtitles; progress, choices and the
 * input belong to the controls and sit outside the video in normal mode, so they never cover
 * people or subtitles. Immersive mode moves only the input onto the picture.
 */
export default function Hud({
  world, current, busy, turn, sequence, draft, onDraft, onSend, onStop, playbackLocked = false,
  autoRunning, onStopAuto, onAddRef, setupHint, preparing = false,
}: Props) {
  const inputRef = useRef<HTMLTextAreaElement>(null)
  const highlightRef = useRef<HTMLDivElement>(null)
  const fileRef = useRef<HTMLInputElement>(null)
  const [addOpen, setAddOpen] = useState(false)
  const [mentionStart, setMentionStart] = useState<number | null>(null)
  const [mentionEnd, setMentionEnd] = useState(0)
  const [mentionQuery, setMentionQuery] = useState('')
  const [mentionIndex, setMentionIndex] = useState(0)
  // After pressing "other" the input flashes; otherwise the cursor lands in a box that looks
  // exactly the same and the user cannot tell what the click did.
  const [own, setOwn] = useState(false)

  const assets = mentionOptions(world)
  const filteredMentions = assets.filter((option) => {
    const q = mentionQuery.trim().toLocaleLowerCase()
    return !q || option.name.toLocaleLowerCase().includes(q)
      || option.id.toLocaleLowerCase().includes(q)
  })
  const visibleMentions = filteredMentions.slice(0, 8)
  const mentionHighlights = mentionRanges(draft, assets)

  const highlightParts = () => {
    if (!draft) return null
    const parts: JSX.Element[] = []
    let cursor = 0
    mentionHighlights.forEach((range, index) => {
      if (range.start > cursor) {
        parts.push(
          <span className="input-text" key={`text-${index}`}>
            {draft.slice(cursor, range.start)}
          </span>,
        )
      }
      parts.push(
        <span
          key={`mention-${range.start}-${range.end}`}
          className="input-mention"
          title={t('已绑定资产：{name}', { name: range.name })}
        >
          {draft.slice(range.start, range.end)}
        </span>,
      )
      cursor = range.end
    })
    if (cursor < draft.length) {
      parts.push(<span className="input-text" key="text-tail">{draft.slice(cursor)}</span>)
    }
    return parts
  }

  const syncMention = (value: string, caret: number) => {
    const context = mentionContext(value, caret)
    setMentionStart(context?.start ?? null)
    setMentionEnd(context?.end ?? caret)
    setMentionQuery(context?.query ?? '')
    setMentionIndex(0)
  }

  const chooseMention = (option: MentionOption) => {
    const start = mentionStart
    if (start === null) return
    const end = mentionEnd
    const replacement = `@${option.name} `
    const next = `${draft.slice(0, start)}${replacement}${draft.slice(end)}`
    const caret = start + replacement.length
    onDraft(next)
    setMentionStart(null)
    setMentionQuery('')
    setMentionIndex(0)
    requestAnimationFrame(() => {
      const input = inputRef.current
      if (!input) return
      input.focus()
      input.setSelectionRange(caret, caret)
    })
  }

  useEffect(() => {
    if (!addOpen) return
    const away = () => setAddOpen(false)
    document.addEventListener('mousedown', away)
    return () => document.removeEventListener('mousedown', away)
  }, [addOpen])

  useEffect(() => {
    if (!own) return
    const t = setTimeout(() => setOwn(false), 1400)
    return () => clearTimeout(t)
  }, [own])

  useEffect(() => {
    setMentionStart(null)
    setMentionQuery('')
    setMentionIndex(0)
  }, [world?.id, busy])

  const send = () => {
    const t = draft.trim()
    if (!t || busy || preparing || playbackLocked) return
    setMentionStart(null)
    onSend(t, mentionedIds(t, assets))
  }

  // Only the latest clip's choices are still valid; no choices when reviewing older clips
  const atHead = !!world?.head && current?.id === world.head
  const options = !busy && atHead ? (current?.options ?? []) : []
  const recommend = current?.recommend ?? 0
  // Prefer the clip's own objective: an older clip shows the objective of its time, not the
  // world's current one. Old clips without one fall back to the world's.
  const objective = (current?.objective || world?.objective || '').trim()

  // Dialogue: while running, the line the world just said (it arrives well before the video);
  // otherwise the line of the clip being played.
  // While the next clip is generated automatically the previous clip is still playing; the
  // subtitle follows the picture and must not be replaced early by the next say.
  const say = ((
    autoRunning
      ? (current?.reply || turn?.say)
      : (turn?.say || current?.reply)
  ) || '').trim()

  return (
    <div id="hud">
      {say && (
        <div className="hud-say" key={turn ? `t${turn.started}` : current?.id}>
          <p>{say}</p>
        </div>
      )}

      <div className="hud-controls">
        {turn ? (
          <Progress
            turn={turn}
            sequence={sequence}
            auto={autoRunning}
            onStop={onStopAuto}
          />
        ) : options.length > 0 ? (
          <div className="hud-deck">
            {objective && (
              <div className="hud-objective">
                <i aria-hidden="true" />
                {objective}
              </div>
            )}
            <div className="hud-options">
              {options.map((o, i) => (
                <button
                  key={i}
                  className={'option' + (i === recommend ? ' pick' : '')}
                  onClick={() => onSend(o)}
                  disabled={busy || playbackLocked}
                  title={i === recommend ? t('世界建议走这条 —— 已经先替你想好了，点下去直接开拍') : undefined}
                >
                  {i === recommend && <em>{t('推荐')}</em>}
                  {o}
                </button>
              ))}
              <button
                className="option other"
                onClick={() => {
                  setOwn(true)
                  inputRef.current?.focus()
                }}
              >
                {t('其他…')}
              </button>
            </div>
          </div>
        ) : null}

        <div className={'hud-input' + (own ? ' own' : '')}>
        <div className="addwrap" onMouseDown={(e) => e.stopPropagation()}>
          <button
            className="icon addbtn"
            title={t('加一张参考图：生成或上传')}
            disabled={busy || !world}
            onClick={() => setAddOpen((v) => !v)}
          >
            <svg viewBox="0 0 24 24" aria-hidden="true">
              <path d="M12 5v14M5 12h14" />
            </svg>
          </button>
          {addOpen && (
            <div className="addmenu">
              <button onClick={() => { setAddOpen(false); onAddRef('subject') }}>
                {t('加角色')}
              </button>
              <button onClick={() => { setAddOpen(false); onAddRef('place') }}>
                {t('加场景')}
              </button>
              <button onClick={() => { setAddOpen(false); fileRef.current?.click() }}>
                {t('从文件上传…')}
              </button>
            </div>
          )}
        </div>

        <div className="input-edit">
          <div ref={highlightRef} className="input-highlight" aria-hidden="true">
            {highlightParts()}
          </div>
          <textarea
            id="input"
            disabled={playbackLocked}
            ref={inputRef}
            rows={1}
            value={draft}
            placeholder={preparing ? t('正在根据你的想法完善世界、主线与开场…') : !world ? t('想去怎样的世界？在这里简单描述，或点击右上角「新世界」完善设定。') : setupHint || (playbackLocked ? t('本段播放结束后，任务和物品状态会更新') : busy ? t('世界正在演这一拍…') : t('你要做什么（输入 @ 绑定资产）'))}
            onChange={(e) => {
              onDraft(e.target.value)
              syncMention(e.target.value, e.target.selectionStart)
            }}
            onClick={(e) => syncMention(e.currentTarget.value, e.currentTarget.selectionStart)}
            onBlur={() => setTimeout(() => setMentionStart(null), 0)}
            onKeyUp={(e) => {
              if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(e.key)) return
              syncMention(e.currentTarget.value, e.currentTarget.selectionStart)
            }}
            onScroll={(e) => {
              if (!highlightRef.current) return
              highlightRef.current.scrollTop = e.currentTarget.scrollTop
              highlightRef.current.scrollLeft = e.currentTarget.scrollLeft
            }}
            onPaste={(e) => {
              // Pasting an image opens the add-asset dialog with the image filled in, only a name
              // missing. A region copied from a web page or a spreadsheet often has both image and
              // text; then let the text paste as usual, or the input looks like it swallows text.
              if (busy || !world) return
              if (e.clipboardData.getData('text/plain').trim()) return
              const f = [...e.clipboardData.files].find((x) => x.type.startsWith('image/'))
              if (!f) return
              e.preventDefault()
              onAddRef('subject', f)
            }}
            onKeyDown={(e) => {
              if (mentionStart !== null && visibleMentions.length > 0) {
                if (e.key === 'ArrowDown') {
                  e.preventDefault()
                  setMentionIndex((i) => (i + 1) % visibleMentions.length)
                  return
                }
                if (e.key === 'ArrowUp') {
                  e.preventDefault()
                  setMentionIndex((i) => (i - 1 + visibleMentions.length) % visibleMentions.length)
                  return
                }
                if (e.key === 'Tab' || (e.key === 'Enter' && !e.shiftKey)) {
                  e.preventDefault()
                  chooseMention(visibleMentions[mentionIndex])
                  return
                }
                if (e.key === 'Escape') {
                  e.preventDefault()
                  setMentionStart(null)
                  return
                }
              }
              if (e.key !== 'Enter' || e.shiftKey) return
              // Chinese / Japanese IMEs use Enter to **confirm a candidate**. Without this guard
              // that Enter counts as "send" and sends half a sentence. keyCode 229 covers browsers
              // that do not report isComposing.
              if (e.nativeEvent.isComposing || e.keyCode === 229) return
              e.preventDefault()
              send()
            }}
          />
        </div>

        {mentionStart !== null && assets.length > 0 && !busy && (
          <div
            className="mention-menu"
            role="listbox"
            aria-label={t('选择要绑定的资产')}
            onMouseDown={(e) => e.preventDefault()}
          >
            {visibleMentions.length > 0 ? visibleMentions.map((option, index) => (
              <button
                type="button"
                key={`${option.kind}:${option.id}`}
                className={index === mentionIndex ? 'is-active' : ''}
                role="option"
                aria-selected={index === mentionIndex}
                onMouseDown={(e) => {
                  e.preventDefault()
                  chooseMention(option)
                }}
              >
                <span className="mention-kind">{kindLabel(option.entity)}</span>
                <b>{option.name}</b>
                <small>{option.kind === 'place' ? t('场景') : option.id}</small>
              </button>
            )) : (
              <div className="mention-empty">{t('没有匹配的资产')}</div>
            )}
          </div>
        )}

        {/* A beat takes a while and changing your mind after sending is common. The button itself
           becomes "stop" so there is no need to look for an undo elsewhere. */}
        {busy ? (
          <button id="btn-send" className="stop" onClick={onStop} title={t('停下这一拍（Esc）')}>
            <svg viewBox="0 0 24 24" aria-hidden="true">
              <rect x="7" y="7" width="10" height="10" rx="2" />
            </svg>
            {t('停止')}
          </button>
        ) : (
          <button id="btn-send" className="primary" disabled={playbackLocked || preparing || !draft.trim()} onClick={send}
            title={t('发送')} aria-label={t('发送')}>
            <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 19V5M5.5 11.5 12 5l6.5 6.5" /></svg>
          </button>
        )}

          <input
            ref={fileRef} type="file" accept="image/*" style={{ display: 'none' }}
            onChange={(e) => {
              const f = e.target.files?.[0]
              e.target.value = ''
              if (f) onAddRef('subject', f)
            }}
          />
        </div>
      </div>
    </div>
  )
}
