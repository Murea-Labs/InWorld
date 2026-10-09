import { useEffect, useState } from 'react'
import { mediaUrl, setNote, voiceUrl } from '../api'
import {
  SECTION_CN, STAGE_COLOR, STAGE_SHORT, STAGE_SHORT_MERGED,
  badge, lineage, sectionsOf, takesOf,
} from '../lib'
import type { Node, PlaceState, SubjectState, TraceStep, World } from '../types'
import AgentTrace from './AgentTrace'
import { t } from '../i18n'

type Tab = 'agent' | 'overview' | 'prompt' | 'refs' | 'raw'

interface Props {
  open: boolean
  world: World | null
  node: Node | null
  busy: boolean
  onClose: () => void
  onSelectTake: (n: Node) => void
  onRegenerate: (six: Record<string, string> | null, seed: number | null) => void
  /** Steps of the turn that is running now, shown on the Agent tab. */
  liveSteps?: TraceStep[]
}

export default function Inspector({
  open, world, node, busy, onClose, onSelectTake, onRegenerate, liveSteps,
}: Props) {
  const [tab, setTab] = useState<Tab>('agent')
  const [edited, setEdited] = useState<Record<string, string> | null>(null)
  const [seed, setSeed] = useState('')

  useEffect(() => {
    setEdited(null)
  }, [node?.id])

  useEffect(() => {
    const esc = (e: KeyboardEvent) => e.key === 'Escape' && open && onClose()
    document.addEventListener('keydown', esc)
    return () => document.removeEventListener('keydown', esc)
  }, [open, onClose])

  const live = !!liveSteps?.length
  if (!node || !world) {
    if (!live) return <aside id="inspector" />
    // The very first beat: nothing to inspect yet but the agent at work.
    return (
      <>
        <div id="scrim" className={open ? 'on' : ''} onClick={onClose} />
        <aside id="inspector" className={open ? 'on' : ''}>
          <div id="ins-head">
            <b>{t('Agent 正在工作')}</b>
            <div className="spacer" style={{ flex: 1 }} />
            <button className="sm" onClick={onClose}>{t('关闭')}</button>
          </div>
          <div id="ins-body"><AgentTrace steps={liveSteps ?? []} live /></div>
        </aside>
      </>
    )
  }

  const chain = lineage(world)
  const idx = chain.findIndex((n) => n.id === node.id)
  const takes = takesOf(world, node)
  const b = badge(node)
  const isHead = node.id === world.head

  return (
    <>
      <div id="scrim" className={open ? 'on' : ''} onClick={onClose} />
      <aside id="inspector" className={open ? 'on' : ''}>
        <div id="ins-head">
          <b>{idx >= 0 ? t('第 {n} 段', { n: idx + 1 }) : t('旁支片段')}</b>
          <span className={'tag ' + b.cls}>{b.text}</span>
          <span className="tag">{node.place}</span>
          <div className="spacer" style={{ flex: 1 }} />
          {takes.length > 1 && (
            <select
              value={node.id}
              style={{ width: 'auto', minWidth: 110 }}
              onChange={(e) => onSelectTake(world.nodes[e.target.value])}
            >
              {takes.map((take, i) => (
                <option key={take.id} value={take.id}>
                  {t('第 {n} 次', { n: i + 1 })}{take.origin === 'regenerate' ? t('（重跑）') : ''}
                </option>
              ))}
            </select>
          )}
          <button className="sm" onClick={onClose}>
            {t('关闭')}
          </button>
        </div>

        <div id="ins-tabs">
          {(['agent', 'overview', 'prompt', 'refs', 'raw'] as Tab[]).map((name) => (
            <button key={name} className={tab === name ? 'on' : ''} onClick={() => setTab(name)}>
              {{
                agent: live ? t('Agent · 进行中') : 'Agent', overview: t('概览'), prompt: 'Prompt',
                refs: t('参考图 / 音色'), raw: t('原始 JSON'),
              }[name]}
            </button>
          ))}
        </div>

        <div id="ins-body">
          {tab === 'agent' && (
            live
              ? <AgentTrace steps={liveSteps ?? []} live />
              : <AgentTrace steps={node.trace ?? []} />
          )}
          {tab === 'overview' && <Overview node={node} world={world} />}
          {tab === 'prompt' && (
            <Prompt
              six={edited ?? node.six_section}
              onChange={(k, v) => setEdited({ ...(edited ?? node.six_section), [k]: v })}
            />
          )}
          {tab === 'refs' && <Refs node={node} />}
          {tab === 'raw' && <Raw node={node} />}
        </div>

        <div id="ins-foot">
          <label style={{ color: 'var(--dimmer)', fontSize: 11.5 }}>seed</label>
          <input
            id="seed"
            type="number"
            placeholder={t('默认')}
            value={seed}
            onChange={(e) => setSeed(e.target.value)}
          />
          <span id="regen-note">
            {edited ? t('prompt 已改，未重跑') : isHead ? '' : t('只能重跑最新一段')}
          </span>
          <div className="spacer" />
          <button
            className="primary"
            disabled={busy || !isHead}
            onClick={() => onRegenerate(edited, seed ? Number(seed) : null)}
          >
            {t('用当前 prompt 重跑')}
          </button>
        </div>
      </aside>
    </>
  )
}

function Overview({ node, world }: { node: Node; world: World }) {
  // Phases that took 0 s are neither drawn nor listed. Since the merge into one call
  // `rewrite` is always 0; listing it would suggest a separate rewrite step still exists.
  const timings = Object.fromEntries(
    Object.entries(node.timings ?? {}).filter(([, v]) => v > 0),
  )
  const total = Object.values(timings).reduce((a, b) => a + b, 0) || 1
  const merged = (node.timings?.rewrite ?? 0) === 0 && (node.timings?.gm ?? 0) > 0
  const label = merged ? STAGE_SHORT_MERGED : STAGE_SHORT
  const p = node.model_params ?? {}
  return (
    <>
      <Field label={node.origin === 'auto' ? t('收到（系统自动续跑）') : t('用户说')}>
        {node.user_text || '—'}
      </Field>
      <Field label={t('GM 回复')}>{node.reply || '—'}</Field>
      <Field label={t('这一拍（beat）')}>{node.beat}</Field>

      {/* World state: the three things the director returns each beat; the first thing to check when editing the prompt */}
      <Field label={t('随身 carry（跨场景唯一活下来的状态）')}>
        {node.carry || <span style={{ color: 'var(--dimmer)' }}>{t('无')}</span>}
      </Field>
      <div className="field">
        <label>{t('还剩什么 continue_with')}</label>
        <div className="box">
          {node.continue_with || (
            <span style={{ color: 'var(--dimmer)' }}>{t('空 —— 这一句演完了')}</span>
          )}
        </div>
      </div>
      <div className="field">
        <label>{t('抛出的岔路 options')}</label>
        {node.options?.length ? (
          <ol className="opt-list">
            {node.options.map((o, i) => <li key={i}>{o}</li>)}
          </ol>
        ) : (
          <div className="box"><span style={{ color: 'var(--dimmer)' }}>{t('这一拍没给')}</span></div>
        )}
      </div>

      <DynamicState node={node} world={world} />

      <div className="field">
        <label>{t('耗时')} <b>{node.elapsed}s</b></label>
        <div className="timing">
          {Object.entries(timings).map(([k, v]) => (
            <i key={k} style={{ flex: v, background: STAGE_COLOR[k] ?? 'var(--w-22)' }}>
              {v / total > 0.13 ? `${label[k] ?? k} ${v.toFixed(1)}s` : ''}
            </i>
          ))}
        </div>
        <div className="legend">
          {Object.entries(timings).map(([k, v]) => (
            <span key={k}>
              <s style={{ background: STAGE_COLOR[k] ?? 'var(--w-22)' }} />
              {label[k] ?? k} {v}s
            </span>
          ))}
          {node.images_drawn > 0 && !timings.image && (
            <span title={t('参考图和改写并行跑，时间已含在「改写」里')}>
              <s style={{ background: STAGE_COLOR.image }} />
              {t('生图 ×{n} {s}s（并行）', { n: node.images_drawn, s: node.image_seconds })}
            </span>
          )}
        </div>
      </div>

      <div className="field">
        <label>{t('生成参数')}</label>
        <dl className="kv">
          <dt>{t('时长 / 画幅')}</dt>
          <dd>
            {String(p.duration_seconds ?? '?')}s · {String(p.aspect_ratio ?? '?')} ·{' '}
            {String(p.size ?? '')}
          </dd>
          <dt>{t('推理耗时')}</dt>
          <dd>{p.inference_time_s ? `${Number(p.inference_time_s).toFixed(1)}s` : '—'}</dd>
          <dt>{t('参考图数')}</dt>
          <dd>{String(p.n_conditions ?? node.refs.length)}</dd>
          <dt>video_id</dt>
          <dd>{node.video_id}</dd>
          <dt>{t('续写自')}</dt>
          <dd>{node.continued_from || <span style={{ color: 'var(--dimmer)' }}>{t('无')}</span>}</dd>
          <dt>{t('来源')}</dt>
          <dd>
            {{ regenerate: t('重跑'), auto: t('系统续跑'), turn: t('对话') }[node.origin] ?? node.origin}
          </dd>
        </dl>
      </div>

      <NoteBox world={world} node={node} />
    </>
  )
}

function DynamicState({ node, world }: { node: Node; world: World }) {
  const snapshot = node.state_after ?? {
    subjects: Object.fromEntries(
      Object.entries(world.subjects).map(([id, subject]) => [id, subject.state ?? emptySubjectState()]),
    ),
    places: Object.fromEntries(
      Object.entries(world.places).map(([name, place]) => [name, place.state ?? emptyPlaceState()]),
    ),
  }
  const subjects = Object.entries(snapshot.subjects ?? {})
  const places = Object.entries(snapshot.places ?? {})
  return (
    <>
      <div className="field">
        <label>{t('这一拍后的角色状态')}</label>
        <div className="box">
          {subjects.length ? subjects.map(([id, state]) => (
            <div key={id} style={{ marginBottom: 8 }}>
              <b>{world.subjects[id]?.name ?? id}</b>
              <div style={{ color: 'var(--dim)', marginTop: 3 }}>
                {subjectStateText(state)}
              </div>
            </div>
          )) : <span style={{ color: 'var(--dimmer)' }}>{t('暂无角色状态')}</span>}
        </div>
      </div>
      <div className="field">
        <label>{t('这一拍后的场景状态')}</label>
        <div className="box">
          {places.length ? places.map(([name, state]) => (
            <div key={name} style={{ marginBottom: 8 }}>
              <b>{name}</b>
              <div style={{ color: 'var(--dim)', marginTop: 3 }}>
                {placeStateText(state)}
              </div>
            </div>
          )) : <span style={{ color: 'var(--dimmer)' }}>{t('暂无场景状态')}</span>}
        </div>
      </div>
    </>
  )
}

function emptySubjectState(): SubjectState {
  return { location: '', outfit: '', condition: '', mood: '', goal: '', knowledge: [], relations: {} }
}

function emptyPlaceState(): PlaceState {
  return { time_of_day: '', weather: '', lighting: '', condition: '', props: [] }
}

function subjectStateText(state: SubjectState): string {
  const pieces = [
    state.location && t('位置：{v}', { v: state.location }),
    state.outfit && t('服装：{v}', { v: state.outfit }),
    state.condition && t('状态：{v}', { v: state.condition }),
    state.mood && t('表情/姿态：{v}', { v: state.mood }),
    state.goal && t('目标：{v}', { v: state.goal }),
    state.knowledge?.length ? t('知道：{v}', { v: state.knowledge.join(t('、')) }) : '',
    state.relations && Object.keys(state.relations).length
      ? t('关系：{v}', { v: Object.entries(state.relations).map(([k, v]) => `${k} (${v})`).join(t('、')) })
      : '',
  ].filter(Boolean)
  return pieces.join(' · ') || t('未记录动态变化')
}

function placeStateText(state: PlaceState): string {
  const pieces = [
    state.time_of_day && t('时段：{v}', { v: state.time_of_day }),
    state.weather && t('天气：{v}', { v: state.weather }),
    state.lighting && t('光照：{v}', { v: state.lighting }),
    state.condition && t('环境：{v}', { v: state.condition }),
    state.props?.length ? t('道具：{v}', { v: state.props.join(t('、')) }) : '',
  ].filter(Boolean)
  return pieces.join(' · ') || t('未记录动态变化')
}

function NoteBox({ world, node }: { world: World; node: Node }) {
  const [v, setV] = useState(node.note ?? '')
  // eslint-disable-next-line react-hooks/exhaustive-deps -- reset the draft only when switching nodes, not after our own save
  useEffect(() => setV(node.note ?? ''), [node.id])
  return (
    <div className="field">
      <label>{t('笔记（记下你试了什么）')}</label>
      <textarea
        rows={3}
        value={v}
        placeholder={t('例如：这段人物脸崩了，把 detailed_description 里的特写去掉试试')}
        onChange={(e) => setV(e.target.value)}
        onBlur={() => setNote(world.id, node.id, v).catch(() => {})}
      />
    </div>
  )
}

function Prompt({
  six, onChange,
}: { six: Record<string, string>; onChange: (k: string, v: string) => void }) {
  return (
    <>
      <p style={{ color: 'var(--dimmer)', fontSize: 12, margin: '0 0 14px' }}>
        {t('这六段拼起来就是发过去的原文。改完点右下角重跑，会用同样的参考图再生成一次，作为这一拍的另一个 take。')}
      </p>
      {sectionsOf(six).map((k) => (
        <div className="field" key={k}>
          <label>
            {k} · {SECTION_CN[k] ?? k}
          </label>
          <textarea
            className="sec"
            rows={k === 'detailed_description' || k === 'shot' ? 9 : 3}
            value={six[k] ?? ''}
            onChange={(e) => onChange(k, e.target.value)}
          />
        </div>
      ))}
    </>
  )
}

function Refs({ node }: { node: Node }) {
  return (
    <>
      <p style={{ color: 'var(--dimmer)', fontSize: 12, margin: '0 0 14px' }}>
        {t('这些图按这个顺序发过去，序号就是 prompt 里的 <Picture N>。角色在前、场景在后。')}
      </p>
      <div className="pics">
        {(node.pictures ?? []).map((p) => (
          <div className="pic" key={p.n}>
            <img src={mediaUrl('refs', p.ref)} alt="" />
            <div>
              <b>&lt;Picture {p.n}&gt;</b> {p.name}
              <small>
                {p.kind === 'character' ? t('角色') : t('场景')} · {p.ref}
              </small>
              <small>{(p.desc || '').slice(0, 120)}</small>
            </div>
          </div>
        ))}
      </div>
      <Audios node={node} />
    </>
  )
}

/**
 * The reference voices actually sent with this clip.
 * 
 * Not read from world.subjects: if a character's voice changed later, reviewing this clip
 * would show the new one while the video has the old one. The inspector shows what was sent.
 */
function Audios({ node }: { node: Node }) {
  const list = node.audios ?? []
  return (
    <div style={{ marginTop: 20 }}>
      <p style={{ color: 'var(--dimmer)', fontSize: 12, margin: '0 0 10px' }}>
        {t('音频跟图片分开编号，序号就是 prompt 里的 <Audio N>，绑在对应的 <Subject N> 上（最多 3 条，这一拍台词里说话的人优先）。')}
      </p>
      {list.length === 0 ? (
        <div className="box" style={{ color: 'var(--dimmer)', fontSize: 12 }}>
          {t('这一段没带参考音色 —— 在场的角色都没绑音色，或者这是加上这项记录之前的老片段。')}
        </div>
      ) : (
        <div className="auds">
          {list.map((a) => (
            <div className="aud" key={a.n}>
              <div className="aud-head">
                <b>&lt;Audio {a.n}&gt;</b>
                <span>
                  {a.subject_n
                    ? `→ <Subject ${a.subject_n}>`
                    : t('未绑定（这个人的参考图被挤掉了）')}
                </span>
                <em>{a.name}</em>
              </div>
              <small>{a.voice_name || t('未命名音色')} · {a.file || '?'}</small>
              {/* Metadata is enough for the duration; do not preload the whole clip (at most three per clip) */}
              {!!a.file && (
                <audio controls preload="metadata" src={voiceUrl(a.file)} />
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

function Raw({ node }: { node: Node }) {
  return (
    <>
      <Field label={t('GM 返回的原始 JSON')} pre>
        {JSON.stringify(node.directive ?? {}, null, 2)}
      </Field>
      <Field label={t('发出去的完整 prompt')} pre>
        {node.prompt}
      </Field>
      <Field label={t('生成请求参数')} pre>
        {JSON.stringify(node.model_params ?? {}, null, 2)}
      </Field>
    </>
  )
}

function Field({
  label, children, pre,
}: { label: string; children: React.ReactNode; pre?: boolean }) {
  return (
    <div className="field">
      <label>{label}</label>
      {pre ? <pre className="box">{children}</pre> : <div className="box">{children}</div>}
    </div>
  )
}
