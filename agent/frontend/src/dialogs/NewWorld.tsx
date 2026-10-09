import { useEffect, useState } from 'react'
import { addPlace, addSubject, createWorld, getHealth, mediaUrl, randomWorld } from '../api'
import type { VideoModel } from '../types'
import RefPicker, { appendRef, emptyRef, type RefValue } from '../components/RefPicker'
import { STYLE_PRESETS, styleById, styleMenu } from '../presets'
import Dialog from './Dialog'
import { t } from '../i18n'

interface Row extends RefValue {
  kind: 'subject' | 'place'
  name: string
  /** The character you play. The rest are driven by the world and act on their own every beat. */
  mine?: boolean
}

const newRow = (kind: Row['kind'], mine = false): Row =>
  ({ ...emptyRef(), kind, name: '', mine })

export default function NewWorld({
  open, onClose, onCreated,
}: { open: boolean; onClose: () => void; onCreated: (id: string, opening?: string) => void }) {
  const [title, setTitle] = useState('')
  const [setting, setSetting] = useState('')
  const [mainline, setMainline] = useState('')
  const [visualStyle, setVisualStyle] = useState('')
  const [models, setModels] = useState<VideoModel[]>([])
  const [videoModel, setVideoModel] = useState('')

  useEffect(() => {
    if (!open || models.length) return
    getHealth()
      .then((h) => {
        setModels(h.video.choices)
        setVideoModel((v) => v || h.video.id)
      })
      .catch(() => {})
  }, [open, models.length])
  const [rows, setRows] = useState<Row[]>([newRow('subject', true), newRow('place')])
  const [busy, setBusy] = useState(false)
  const [rolling, setRolling] = useState(false)
  const [note, setNote] = useState('')
  // A rolled cast first shows as a row of cards; expand one to edit it.
  const [editRefs, setEditRefs] = useState(false)

  /**
   * Fill the form with a rolled world. **Only the four setting fields**, never the
   * reference-image rows: the user may already have uploaded their own character image.
   */
  const roll = async () => {
    setRolling(true)
    setNote('')
    try {
      const w = await randomWorld(styleMenu())
      setTitle(w.title)
      setSetting(w.setting)
      setMainline(w.mainline)
      setVisualStyle(w.visual_style || styleById(w.style_id) || '')
      if (w.refs?.length) {
        setRows(w.refs.map((r) => ({
          ...emptyRef(), kind: r.kind, name: r.name, mine: !!r.mine,
          ref: r.ref, preview: mediaUrl('refs', r.ref), prompt: r.prompt || '',
        })))
        setEditRefs(false)
      }
    } catch (e) {
      setNote(String((e as Error).message).slice(0, 160))
    } finally {
      setRolling(false)
    }
  }

  const patch = (i: number, v: Partial<Row>) =>
    setRows((rs) => rs.map((r, j) => (j === i ? { ...r, ...v } : r)))

  const hasContent = (r: Row) => !!(r.file || r.ref || r.prompt.trim())
  /** Rows the user clearly meant to add but never named -- do not silently drop them. */
  const unnamed = rows.map((r, i) => [r, i] as const).filter(([r]) => hasContent(r) && !r.name.trim())

  const submit = async () => {
    if (unnamed.length) {
      setNote(t('还有 {n} 项没起名字 —— 名字是它在世界里的标识，不能留空', { n: unnamed.length }))
      return
    }
    setBusy(true)
    try {
      const w = await createWorld(
        title || t('未命名世界'), setting, visualStyle, videoModel, mainline)
      const usable = rows.filter((r) => r.name.trim() && hasContent(r))
      for (const [i, r] of usable.entries()) {
        setNote(t('上传 {i}/{n}：{name}', { i: i + 1, n: usable.length, name: r.name }))
        const fd = new FormData()
        fd.append('name', r.name.trim())
        appendRef(fd, r)
        if (r.kind === 'subject') {
          fd.append('control', r.mine ? 'user' : 'ai')
          await addSubject(w.id, fd)
        }
        else {
          fd.append('make_current', String(i === usable.findIndex((x) => x.kind === 'place')))
          await addPlace(w.id, fd)
        }
      }
      setTitle(''); setSetting(''); setMainline(''); setVisualStyle('')
      setRows([newRow('subject', true), newRow('place')]); setNote(''); setEditRefs(false)
      onCreated(w.id)
    } catch (e) {
      setNote(String((e as Error).message).replace(/^\{"detail":"|"\}$/g, '').slice(0, 200))
    } finally {
      setBusy(false)
    }
  }

  return (
    <Dialog
      open={open} title={t('新世界')} className="nw"
      sub={t('描述一个世界，或者摇一个。人物和场景定下来，故事就从这里开始。')}
      busy={busy} okLabel={t('创建世界')} note={note} onClose={onClose} onOk={submit}
    >
      <button type="button" className={'nw-roll' + (rolling ? ' rolling' : '')}
              disabled={rolling || busy} onClick={() => void roll()}>
        <svg className="dice-face" viewBox="0 0 20 20" aria-hidden="true">
          <rect x="2.6" y="2.6" width="14.8" height="14.8" rx="4" />
          <circle cx="6.9" cy="6.9" r="1.35" />
          <circle cx="13.1" cy="6.9" r="1.35" />
          <circle cx="10" cy="10" r="1.35" />
          <circle cx="6.9" cy="13.1" r="1.35" />
          <circle cx="13.1" cy="13.1" r="1.35" />
        </svg>
        <span>
          <b>{rolling ? t('正在摇一个世界…') : t('摇一个')}</b>
          <small>{t('名字、世界观和主线一次编排好；人物和场景会在故事里出现')}</small>
        </span>
      </button>

      <div className="nw-fields">
        <label>{t('名字')}</label>
        <input className="nw-title" value={title} placeholder={t('给这个世界起个名字')}
               onChange={(e) => setTitle(e.target.value)} />
        <label title={t('这是个什么地方。可留空，让 AI 自己定。')}>{t('世界观')}</label>
        <input value={setting} placeholder={t('一句话：这是个什么地方')}
               onChange={(e) => setSetting(e.target.value)} />
        <label title={t('这里正在发生什么。有主线，GM 每一拍推荐的岔路都会奔着它去；留空由 GM 第一拍自己定。')}>{t('主线')}</label>
        <input value={mainline} placeholder={t('一句话：这里正在发生什么')}
               onChange={(e) => setMainline(e.target.value)} />
      </div>

      <div className="nw-section">
        <label>{t('人物与场景')}</label>
        {!editRefs && rows.some(hasContent) && (
          <button type="button" className="sm ghost" onClick={() => setEditRefs(true)}>{t('编辑')}</button>
        )}
      </div>
      {!editRefs && rows.some(hasContent) ? (
        <div className="nw-cast">
          {rows.filter(hasContent).map((r, i) => (
            <button type="button" key={i} className={'nw-cast-card ' + r.kind} onClick={() => setEditRefs(true)}>
              <span className="nw-cast-img">
                {r.preview ? <img src={r.preview} alt="" /> : null}
                {r.kind === 'subject' && r.mine && <em>{t('你')}</em>}
              </span>
              <b>{r.name || t('未命名')}</b>
              <small>{r.kind === 'place' ? t('场景') : r.mine ? t('你扮演') : t('世界驱动')}</small>
            </button>
          ))}
        </div>
      ) : (
      <>
      <div className="nw-grid">
      {rows.map((r, i) => (
        <RefPicker
          key={i}
          kind={r.kind === 'subject' ? 'subject' : 'place'}
          value={r}
          visualStyle={visualStyle}
          onChange={(v) => patch(i, v)}
          placeholder={
            r.kind === 'subject'
              ? t('三十出头的男人，浅蓝色衬衫')
              : t('现代办公室里的会议室，长木桌，城市日光')
          }
          header={
            <>
              <div className="seg">
                {(['subject', 'place'] as const).map((k) => (
                  <button
                    key={k}
                    className={r.kind === k ? 'on' : ''}
                    onClick={() => patch(i, { kind: k, mine: false })}
                  >
                    {k === 'subject' ? t('角色') : t('场景')}
                  </button>
                ))}
              </div>
              <input
                className={hasContent(r) && !r.name.trim() ? 'need' : ''}
                value={r.name}
                placeholder={r.kind === 'subject' ? t('起个名字，比如 小明') : t('起个名字，比如 会议室')}
                onChange={(e) => patch(i, { name: e.target.value })}
              />
              {r.kind === 'subject' && (
                <button
                  className={'mine' + (r.mine ? ' on' : '')}
                  title={r.mine ? t('你扮演这个角色') : t('点一下改成你扮演；其余角色由世界自己驱动，每一拍都会自己做点什么')}
                  // There can be only one "you"; picking someone else clears the others.
                  onClick={() => setRows((rs) => rs.map((x, j) =>
                    x.kind === 'subject' ? { ...x, mine: j === i && !r.mine } : x))}
                >
                  {r.mine ? t('你扮演') : t('世界驱动')}
                </button>
              )}
              <button
                className="refdel"
                title={t('去掉这一项')}
                onClick={() => setRows((rs) => rs.filter((_, j) => j !== i))}
              >
                ×
              </button>
            </>
          }
        />
      ))}
      </div>
      <div className="row" style={{ marginTop: 8 }}>
        <button className="sm" onClick={() => setRows((rs) => [...rs, newRow('subject')])}>
          {t('+ 角色')}
        </button>
        <button className="sm" onClick={() => setRows((rs) => [...rs, newRow('place')])}>
          {t('+ 场景')}
        </button>
      </div>
      </>
      )}

      <details className="nw-adv">
        <summary>{t('高级设置')}<span>{t('画风 · 视频模型')}</span></summary>
        <label title={t('这段文字会原样约束所有角色图、场景图和视频。只写画风、色彩、材质、光照和镜头质感。')}>{t('视觉风格')}</label>
        <div className="preset-row">
          {STYLE_PRESETS.map((p) => (
            <button
              key={p.id}
              className={'preset' + (visualStyle.trim() === p.text ? ' on' : '')}
              title={p.text}
              onClick={() => setVisualStyle(visualStyle.trim() === p.text ? '' : p.text)}
            >
              {p.label}
            </button>
          ))}
        </div>
        <textarea
          rows={2} value={visualStyle}
          placeholder="cinematic live action, natural daylight, shallow depth of field, muted color grade"
          onChange={(e) => setVisualStyle(e.target.value)}
        />
        <label title={t('建好之后不能改，两家的续写句柄不通用')}>{t('视频模型')}</label>
        <div className="seg">
          {models.map((m) => (
            <button
              key={m.id}
              className={videoModel === m.id ? 'on' : ''}
              title={m.note}
              onClick={() => setVideoModel(m.id)}
            >
              {m.label}
            </button>
          ))}
        </div>
      </details>
    </Dialog>
  )
}
