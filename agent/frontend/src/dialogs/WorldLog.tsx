import { useEffect, useMemo, useState } from 'react'
import * as api from '../api'
import type { LogEntry } from '../api'
import { t } from '../i18n'

/**
 * A world's outbound call log: navigation on the left, details on the right.
 * 
 * Why this panel exists: a beat makes four kinds of external calls (director / rewriter /
 * image / video). When something goes wrong, the final clip alone cannot tell whether the
 * model wrote it wrong or we sent it wrong.
 * 
 * Why it groups by beat instead of one flat stream: the unit of debugging is "what happened
 * in that beat": what the user said, what the director replied, what the rewriter wrote,
 * which string went to the video model and what came out. A flat list scatters these.
 * 
 * Deliberately read-only plus one clear button: it is a debugging tool, not a feature.
 */

const KIND_CN: Record<string, string> = {
  turn: t('开始'),
  turn_done: t('产出'),
  llm: t('模型'),
  gemini: t('模型·非流式'),
  image: t('生图'),
  video_submit: t('视频提交'),
  merged_fallback: t('回退改写'),
}

function when(t: number): string {
  const d = new Date(t * 1000)
  const p = (n: number) => String(n).padStart(2, '0')
  return `${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`
}

const secondsOf = (e: LogEntry) => e.seconds ?? e.upload_seconds

const isBad = (e: LogEntry) =>
  e.ok === false || e.kind === 'merged_fallback' || (e.status != null && e.status >= 400)

/** The one-line summary of a row in the list. */
function headline(e: LogEntry): string {
  if (e.kind === 'turn') return e.user_text || t('(空)')
  if (e.kind === 'turn_done') return e.recap || e.beat || e.place || ''
  if (e.kind === 'merged_fallback') return e.mismatch || ''
  if (e.kind === 'image' || e.kind === 'video_submit') return (e.prompt || '').slice(0, 90)
  const turns = e.request?.turns || []
  const last = [...turns].reverse().find((t) => (t.text || '').trim())
  return (last?.text || '').slice(0, 90)
}

/**
 * A beat = a `turn` entry and every call until the next `turn`.
 * The API returns newest first; convert to time order to group, then reverse the groups.
 */
interface Group { key: string; head: LogEntry | null; rows: LogEntry[] }

function groupByTurn(rows: LogEntry[]): Group[] {
  const groups: Group[] = []
  let cur: Group | null = null
  for (const e of [...rows].reverse()) {
    if (e.kind === 'turn' || !cur) {
      cur = { key: `${e.t}-${groups.length}`, head: e.kind === 'turn' ? e : null, rows: [] }
      groups.push(cur)
    }
    cur.rows.push(e)
  }
  return groups.reverse()
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <>
      <div className="log-label">{label}</div>
      <pre className="log-pre">{children}</pre>
    </>
  )
}

function Detail({ e }: { e: LogEntry }) {
  const u = e.usage
  return (
    <div className="log-detail">
      <div className="log-detail-head">
        <b>{KIND_CN[e.kind] || e.kind}</b>
        <span className="log-model">{e.tag || e.model || e.origin || ''}</span>
        <div className="spacer" />
        <span className="log-time">{when(e.t)}</span>
      </div>

      <p className="log-meta">
        {e.first_token_seconds != null && t('首字 {s}s · ', { s: e.first_token_seconds.toFixed(2) })}
        {/* The video entry records the **upload** time, not the render time; calling it "duration"
           would suggest the video model ran two seconds. The full render time is in the beat's
           output (timings.video). */}
        {e.upload_seconds != null
          ? t('上行 {s}s', { s: e.upload_seconds.toFixed(2) })
          : e.seconds != null && t('耗时 {s}s', { s: e.seconds.toFixed(2) })}
        {u && t(' · token 入 {i} 出 {o}', { i: u.in ?? '—', o: u.out ?? '—' })}
        {u?.think ? t(' · 思考 {n}', { n: u.think }) : ''}
        {e.retries ? t(' · 重试 {n} 次', { n: e.retries }) : ''}
        {e.chars != null && t(' · {n} 字', { n: e.chars })}
        {e.request?.system_chars ? t(' · system {n} 字', { n: e.request.system_chars }) : ''}
      </p>

      {e.error && <p className="log-err">{e.error}</p>}

      {e.kind === 'merged_fallback' && (
        <Field label={t('哪个字段对不上（合并版这次白跑了，退回单独改写器）')}>
          {`${e.mismatch || ''}\nphase=${e.phase || ''}  relation=${e.relation || ''}`}
        </Field>
      )}

      {e.kind === 'turn' && (
        <Field label={t('用户这一句')}>
          {`${e.user_text || ''}\n\n` + t('来源 {origin} · 接在 {head} 后面', { origin: e.origin || '', head: e.head || t('(开场)') })}
        </Field>
      )}

      {e.kind === 'image' && (
        <Field label={t('画图 prompt（{kind}）', { kind: e.asset_kind || '' })}>{e.prompt}</Field>
      )}

      {e.kind === 'video_submit' && (
        <>
          <p className="log-meta">
            {t('参考图 {r} 张 · 音色 {a} 条', { r: e.refs ?? 0, a: e.audios ?? 0 })} ·{' '}
            {e.continuation ? t('带续写句柄') : t('不续写')} · HTTP {e.status}
            {e.attempt ? t(' · 第 {n} 次尝试', { n: e.attempt + 1 }) : ''}
            {t(' · 渲染时间看这一拍的「产出」')}
          </p>
          {e.payload && Object.keys(e.payload).length > 0 && (
            <Field label={t('提交参数')}>{JSON.stringify(e.payload, null, 2)}</Field>
          )}
          <Field label={t('发给视频模型的 prompt（原文）')}>{e.prompt}</Field>
        </>
      )}

      {e.kind === 'turn_done' && (
        <>
          <p className="log-meta">
            {t('参考图 {r} 张 · 音色 {a} 条 · 现画 {d} 张', { r: e.refs ?? 0, a: e.audios ?? 0, d: e.images_drawn ?? 0 })}
            {e.timings && ' · ' + Object.entries(e.timings)
              .map(([k, v]) => `${k} ${v.toFixed(1)}s`).join(' / ')}
          </p>
          {!!e.repairs?.length && <Field label={t('服务端修补')}>{e.repairs.join('\n')}</Field>}
          <Field label={t('这一拍演了什么')}>
            {[
              `${t('场景')}  ${e.place || ''}`,
              `${t('字幕')}  ${e.recap || ''}`,
              `${t('回复')}  ${e.reply || ''}`,
              `beat  ${e.beat || ''}`,
              `${t('目标')}  ${e.objective || ''}`,
              `${t('续跑')}  ${e.continue_with || t('(收敛，无)')}`,
              ...(e.plan || []).map((x, i) => `${t('计划')}${i + 1} ${x}`),
              ...(e.options || []).map((x, i) => `${t('选项')}${i + 1} ${x}`),
            ].join('\n')}
          </Field>
          {e.six_section && Object.keys(e.six_section).length > 0 && (
            <Field label={t('六段式（发给视频模型之前拼好的那份）')}>
              {Object.entries(e.six_section).map(([k, v]) => `── ${k}\n${v}`).join('\n\n')}
            </Field>
          )}
        </>
      )}

      {(e.request?.turns || []).map((turn, i) => (
        <Field key={i} label={turn.role === 'model' ? t('上下文 · 模型') : t('发出')}>{turn.text}</Field>
      ))}
      {e.response && <Field label={t('回来的')}>{e.response}</Field>}
    </div>
  )
}

export default function WorldLog(
  { open, world, onClose }: { open: boolean; world: string; onClose: () => void },
) {
  const [rows, setRows] = useState<LogEntry[] | null>(null)
  const [err, setErr] = useState('')
  const [sel, setSel] = useState('')

  const load = () => {
    if (!world) return
    setErr('')
    api.worldLog(world).then((r) => {
      setRows(r)
      // Default to the latest beat's output: that is what people open the panel for
      const first = r.find((e) => e.kind === 'turn_done') || r[0]
      setSel(first ? `${first.t}-${first.kind}` : '')
    }).catch((e) => setErr(String(e)))
  }

  useEffect(() => {
    if (open) load()
    else { setRows(null); setSel('') }
  }, [open, world]) // eslint-disable-line react-hooks/exhaustive-deps

  const groups = useMemo(() => groupByTurn(rows || []), [rows])
  const selected = useMemo(
    () => (rows || []).find((e) => `${e.t}-${e.kind}` === sel),
    [rows, sel],
  )

  if (!open) return null
  return (
    <div className="log-scrim" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <div className="log-panel" role="dialog" aria-label={t('后台日志')}>
        <div className="log-bar">
          <b>{t('后台日志')}</b>
          <span className="log-sub">
            {rows ? t('{g} 拍 · {r} 条', { g: groups.length, r: rows.length }) : t('读取中…')}
          </span>
          <div className="spacer" />
          <button className="sm ghost" onClick={load}>{t('刷新')}</button>
          <button
            className="sm ghost"
            onClick={() => { void api.clearWorldLog(world).then(load) }}
          >{t('清空')}</button>
          <button className="sm ghost" onClick={onClose}>{t('关闭')}</button>
        </div>

        <div className="log-split">
          <nav className="log-nav">
            {err && <p className="log-err">{err}</p>}
            {rows && rows.length === 0 && (
              <p className="log-empty">
                {t('这个世界还没有记录 —— 演一拍就有了。')}
                {t('更早的调用没有记录，补不回来。')}
              </p>
            )}
            {groups.map((g) => (
              <div className="log-group" key={g.key}>
                <div className="log-group-head">
                  <span className="log-time">{when(g.rows[0].t)}</span>
                  <span className="log-group-text">
                    {g.head ? headline(g.head) : t('(这一拍的开头已滚出保留范围)')}
                  </span>
                </div>
                {g.rows.map((e) => {
                  const id = `${e.t}-${e.kind}`
                  return (
                    <button
                      key={id}
                      type="button"
                      className={'log-item' + (id === sel ? ' on' : '')
                        + (isBad(e) ? ' is-bad' : '')}
                      onClick={() => setSel(id)}
                    >
                      <span className="log-kind">{KIND_CN[e.kind] || e.kind}</span>
                      <span className="log-line">{e.tag || headline(e)}</span>
                      {secondsOf(e) != null && (
                        <span className="log-secs">{secondsOf(e)!.toFixed(1)}s</span>
                      )}
                    </button>
                  )
                })}
              </div>
            ))}
          </nav>

          <section className="log-body">
            {selected ? <Detail e={selected} />
              : <p className="log-empty">{t('左边选一条看详情。')}</p>}
          </section>
        </div>
      </div>
    </div>
  )
}
