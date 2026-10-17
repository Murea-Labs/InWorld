import { useState } from 'react'
import { language, t } from '../i18n'
import type { TracePhase, TraceStep } from '../types'

/** The agent's loop, in order. Each beat runs these phases; `reflect` follows playback. */
export const PHASES: TracePhase[] = ['perceive', 'plan', 'check', 'act', 'commit', 'reflect']

// Phase names match the README and the CLI; 检查 alone is "Inspect" elsewhere in the UI.
export const PHASE_LABEL: Record<TracePhase, string> = language() === 'zh'
  ? { perceive: '感知', plan: '规划', check: '检查', act: '行动', commit: '提交', reflect: '反思' }
  : { perceive: 'Perceive', plan: 'Plan', check: 'Check', act: 'Act', commit: 'Commit', reflect: 'Reflect' }

const PHASE_HINT: Record<TracePhase, string> = {
  perceive: t('读世界状态和上一段的最后几帧'),
  plan: t('导演一次调用决定这一拍发生什么'),
  check: t('确定性规则检查计划，出错就让导演重写'),
  act: t('画参考图、写视频 prompt、拍视频'),
  commit: t('把变化写进世界'),
  reflect: t('看真实画面，修正世界和选项'),
}

const STEP_LABEL: Record<string, string> = {
  observe: t('观察'),
  director: t('导演'),
  recall_story: t('回忆剧情'),
  inspect_asset: t('查资产'),
  action_plan: t('动作计划'),
  motion_plan: t('物理关系'),
  action_sequence: t('拆分动作'),
  world_event: t('世界事件'),
  stage_assets: t('布景选角'),
  draw_references: t('画参考图'),
  write_prompt: t('写 prompt'),
  lock_prompt: t('锁定一致性'),
  render_clip: t('拍视频'),
  update_world: t('更新世界'),
  critic: t('评审画面'),
}

function short(value: unknown): string {
  if (value === null || value === undefined || value === '') return '—'
  if (typeof value === 'string') return value
  if (typeof value === 'number' || typeof value === 'boolean') return String(value)
  if (Array.isArray(value) && value.every((v) => typeof v !== 'object')) return value.join(', ') || '—'
  return JSON.stringify(value, null, 2)
}

function StepRow({ step, total }: { step: TraceStep; total: number }) {
  const [open, setOpen] = useState(false)
  const entries = Object.entries(step.data || {}).filter(([, v]) => v !== '' && v !== null
    && !(Array.isArray(v) && v.length === 0))
  const start = step.started ?? 0
  // Only steps that took noticeable time get a bar; instant checks stay clean.
  const width = step.seconds != null && step.seconds >= 0.05 && total > 0
    ? Math.max(1, (step.seconds / total) * 100) : 0
  return (
    <li className={'trace-step is-' + step.status}>
      <button type="button" className="trace-step-head" onClick={() => setOpen((v) => !v)}
        aria-expanded={open} disabled={!entries.length}>
        <i className="trace-dot" aria-hidden="true" />
        <b>{STEP_LABEL[step.name] ?? step.name}</b>
        <span className="trace-summary">{step.summary}</span>
        <em>{step.status === 'running' ? t('进行中') : step.seconds != null ? `${step.seconds.toFixed(1)}s` : ''}</em>
      </button>
      {width > 0 && (
        <div className="trace-bar" aria-hidden="true">
          <span style={{ left: `${(start / total) * 100}%`, width: `${width}%` }} />
        </div>
      )}
      {open && (
        <dl className="trace-data">
          {entries.map(([key, value]) => (
            <div key={key}><dt>{key}</dt><dd>{short(value)}</dd></div>
          ))}
        </dl>
      )}
    </li>
  )
}

/** What the agent saw, planned, checked and did for one clip, step by step. */
export default function AgentTrace({ steps, live = false }: { steps: TraceStep[]; live?: boolean }) {
  if (!steps.length) {
    return <p className="trace-empty">{t('这一段没有 agent 轨迹（较早生成的片段不记录）。')}</p>
  }
  const total = Math.max(...steps.map((s) => (s.started ?? 0) + (s.seconds ?? 0)), 0.001)
  return (
    <div className={'agent-trace' + (live ? ' is-live' : '')}>
      <p className="trace-intro">
        {t('每一拍都走同一个循环：感知 → 规划 → 检查 → 行动 → 提交，播放后再反思。点开一步看它的输入输出。')}
      </p>
      {PHASES.map((phase) => {
        const mine = steps.filter((s) => s.phase === phase)
        if (!mine.length) return null
        // Wall-clock span of the phase: a tool call runs inside the director's step.
        const timed = mine.filter((s) => s.started != null && s.seconds != null)
        const seconds = timed.length
          ? Math.max(...timed.map((s) => (s.started ?? 0) + (s.seconds ?? 0)))
            - Math.min(...timed.map((s) => s.started ?? 0))
          : 0
        return (
          <section key={phase} className={'trace-phase phase-' + phase}>
            <div className="trace-phase-head">
              <b>{PHASE_LABEL[phase]}</b>
              <span>{PHASE_HINT[phase]}</span>
              {seconds > 0 && <em>{seconds.toFixed(1)}s</em>}
            </div>
            <ol>{mine.map((s) => <StepRow key={s.id} step={s} total={total} />)}</ol>
          </section>
        )
      })}
    </div>
  )
}

/** Live update of the step list from a `step` event (start and end share an id). */
export function upsertStep(steps: TraceStep[] | undefined, step: TraceStep): TraceStep[] {
  const list = steps ?? []
  const at = list.findIndex((s) => s.id === step.id)
  if (at < 0) return [...list, step]
  const next = list.slice()
  next[at] = step
  return next
}
