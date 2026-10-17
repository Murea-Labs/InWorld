import { useEffect, useMemo, useRef, useState } from 'react'
import { listVoices, uploadVoice, voiceUrl } from '../api'
import { markFilePickerOpen } from '../filepicker'
import type { LibraryVoice, VoiceRef } from '../types'
import { t } from '../i18n'

interface Props {
  /** The bound voice. null / empty ref = this character has no voice yet. */
  value: VoiceRef | null
  onChange: (v: { voice_id?: string; ref?: string; name?: string } | null) => void
  disabled?: boolean
}

const AGES = ['儿童', '少年/少女', '青年', '中年', '老年']

/**
 * Pick a character's voice: search the library, preview, or upload your own clip.
 * 
 * This is the **character's** voice (sent to the video model as reference_audios; it is what
 * the character sounds like in the video). The header's voice switch is the narrator, always
 * the same voice and unrelated to this.
 * 
 * The library has hundreds of entries, so it cannot be a `<select>`. Collapsed it shows the
 * current one; "change" opens the search panel. The panel lists the curated tier the director
 * picks from by default; "all" shows the whole library.
 */
export default function VoicePicker({ value, onChange, disabled }: Props) {
  const [lib, setLib] = useState<LibraryVoice[]>([])
  const [note, setNote] = useState('')
  const [busy, setBusy] = useState(false)
  const [open, setOpen] = useState(false)
  const [q, setQ] = useState('')
  const [all, setAll] = useState(false)
  const [gender, setGender] = useState('')
  const [age, setAge] = useState('')
  /** ref of the voice being previewed. Empty = nothing playing. */
  const [playing, setPlaying] = useState('')
  const fileRef = useRef<HTMLInputElement>(null)
  const audioRef = useRef<HTMLAudioElement | null>(null)

  const stop = () => {
    audioRef.current?.pause()
    audioRef.current = null
    setPlaying('')
  }

  useEffect(() => {
    let alive = true
    listVoices()
      .then((v) => alive && setLib(v))
      .catch(() => alive && setNote(t('音色库没读出来')))
    return () => {
      alive = false
    }
  }, [])

  // Stop on closing the dialog or switching character, or the preview keeps playing.
  useEffect(() => () => stop(), [])

  const ref = value?.ref ?? ''
  const id = value?.id ?? ''
  const picked = lib.find((v) => v.id === id)
  // Not in the library but has a file: the user's own upload.
  const own = !!ref && !id
  const has = !!(id || ref)
  // Bound to a library id, the library is loaded, but the entry is missing: an old world after
  // a library change, the file is gone. **Report it**: otherwise the UI shows the old name and a
  // live preview button that plays nothing, and generation silently skips it, so the character
  // is mute. The `lib.length` check avoids flashing "invalid" before the library has loaded.
  const gone = !!id && !!lib.length && !picked
  // When creating a character the caller only has the id (the backend decides the file name), so
  // prefer the library entry's ref.
  const playRef = gone ? '' : (picked?.ref || ref)

  const shown = useMemo(() => {
    const kw = q.trim().toLowerCase()
    return lib.filter((v) => {
      if (!all && !v.gm) return false
      if (gender && v.gender !== gender) return false
      if (age && v.age !== age) return false
      if (!kw) return true
      return `${v.name} ${v.desc} ${v.accent} ${v.group} ${v.gender} ${v.age}`
        .toLowerCase().includes(kw)
    })
  }, [lib, q, all, gender, age])

  const play = (r: string) => {
    if (playing === r) return stop()
    stop()
    if (!r) return
    const a = new Audio(voiceUrl(r))
    audioRef.current = a
    setPlaying(r)
    a.onended = a.onerror = () => {
      // After switching, the old onended still fires; do not stop the new one.
      if (audioRef.current === a) stop()
    }
    a.play().catch(() => stop())
  }

  const upload = async (f: File | null) => {
    if (!f) return
    stop()
    setBusy(true)
    setNote(t('上传中…'))
    try {
      const r = await uploadVoice(f)
      onChange({ ref: r.ref, name: r.name })
      setOpen(false)
      setNote(t('{name}，可以试听了', { name: r.name }))
    } catch (e) {
      setNote(String((e as Error).message).replace(/^\{"detail":"|"\}$/g, '').slice(0, 140))
    } finally {
      setBusy(false)
    }
  }

  const label = picked
    ? `${picked.name} — ${picked.desc}`
    : gone ? t('{name} — 这条音色已不在库里，请重新选一条', { name: value?.name || id })
      : own ? (value?.name || t('这个角色现有的音色'))
        : id ? (value?.name || id)
          : t('不设音色（这个角色不出声）')

  const chip = (on: boolean, text: string, click: () => void) => (
    <button key={text} type="button" className={`vchip${on ? ' on' : ''}`}
      disabled={disabled} onClick={click}>{text}</button>
  )

  return (
    <div className="voice-pick">
      <div className={`voice-current${gone ? ' gone' : ''}`} title={label}>{label}</div>

      <button className="sm" disabled={!playRef || disabled} onClick={() => play(playRef)}>
        {playing === playRef && !!playRef ? t('停') : t('试听')}
      </button>
      <button className="sm" disabled={disabled || busy}
        onClick={() => { stop(); setNote(''); setOpen((v) => !v) }}>
        {open ? t('收起') : has ? t('更换') : t('选音色')}
      </button>
      {has && (
        <button className="sm" disabled={disabled || busy}
          onClick={() => { stop(); setNote(''); onChange(null) }}>{t('清除')}</button>
      )}

      {open && (
        <div className="voice-browse">
          <div className="voice-filters">
            <input className="voice-q" value={q} placeholder={t('搜名字、人设、方言…')}
              disabled={disabled} onChange={(e) => setQ(e.target.value)} />
            {chip(!all, t('精选'), () => setAll(false))}
            {chip(all, t('全部'), () => setAll(true))}
            {chip(gender === '男', t('男'), () => setGender((g) => (g === '男' ? '' : '男')))}
            {chip(gender === '女', t('女'), () => setGender((g) => (g === '女' ? '' : '女')))}
            {AGES.map((a) => chip(age === a, t(a),
              () => setAge((x) => (x === a ? '' : a))))}
          </div>

          <div className="voice-list">
            {shown.map((v) => (
              <div key={v.id} className={`voice-row-item${v.id === id ? ' on' : ''}`}
                role="button" tabIndex={0}
                onClick={() => { stop(); onChange({ voice_id: v.id }); setOpen(false) }}
                onKeyDown={(e) => {
                  if (e.key !== 'Enter' && e.key !== ' ') return
                  e.preventDefault()
                  stop(); onChange({ voice_id: v.id }); setOpen(false)
                }}
              >
                <button className="voice-play" type="button" title={t('试听')}
                  // Previewing must not select the entry; listening to a few before deciding is
                  // normal.
                  onClick={(e) => { e.stopPropagation(); play(v.ref) }}
                >{playing === v.ref ? '■' : '▶'}</button>
                <div className="voice-meta">
                  <b>{v.name}</b>
                  <span className="voice-tag">{v.gender}·{v.age}{v.accent ? `·${v.accent}` : ''}</span>
                  <i>{v.desc}</i>
                </div>
              </div>
            ))}
            {!shown.length && (
              <div className="voice-empty">
                {lib.length ? t('没有符合的，把筛选放宽一点') : t('音色库还没读出来')}
              </div>
            )}
          </div>

          <div className="voice-foot">
            <span>{shown.length} / {all ? lib.length : lib.filter((v) => v.gm).length}</span>
            <button className="sm" disabled={disabled || busy}
              onClick={() => { markFilePickerOpen(); fileRef.current?.click() }}>
              {t('传自己的')}
            </button>
          </div>
        </div>
      )}

      <span className="voice-note">
        {note || (gone
          ? t('换音色库之前绑的那条，文件已经不在了 —— 不重新选的话这个人在视频里不出声。')
          : value?.guessed
            ? t('⚠️ 自动挑的，连性别都没从名字和简介里认出来 —— 听一下，不对就换。')
            : value?.source === 'auto'
              ? t('自动挑的，可以换成别的')
              : own ? t('用的是这个角色自带的那段') : '')}
      </span>

      <input
        ref={fileRef} type="file" accept="audio/*" style={{ display: 'none' }}
        onChange={(e) => {
          upload(e.target.files?.[0] ?? null)
          e.target.value = ''            // uploading the same file twice must still trigger change
        }}
      />
    </div>
  )
}
