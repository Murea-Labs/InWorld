import { useEffect, useState } from 'react'
import {
  addPlace, addSubject, mediaUrl, replaceSubject, setSubjectControl, setSubjectVoice,
} from '../api'
import RefPicker, { appendRef, emptyRef, type RefValue } from '../components/RefPicker'
import VoicePicker from '../components/VoicePicker'
import type { Place, Subject, VoiceRef } from '../types'
import Dialog from './Dialog'
import { kindLabel, t } from '../i18n'

/** What the voice picker returns. Binding waits until the character exists, so keep it in the form for now. */
type VoicePick = { voice_id?: string; ref?: string; name?: string } | null

/**
 * Picker result -> the record used for display. The backend decides the library file name,
 * so leave it empty; VoicePicker looks it up for previews.
 */
function shownVoice(pick: VoicePick): VoiceRef | null {
  if (!pick) return null
  if (pick.voice_id) return { id: pick.voice_id, name: '', ref: '', source: 'library' }
  return { id: '', name: pick.name || t('上传的音色'), ref: pick.ref || '', source: 'upload' }
}

interface Props {
  open: boolean
  kind: 'subject' | 'place'
  /** The kind for a new asset; existing assets read current.kind. */
  subjectKind?: '人物' | '物件'
  worldId: string
  visualStyle: string
  /** Stable subject id or place name when inspecting an existing asset. */
  assetKey?: string
  /** An image pasted or uploaded from the chat; prefilled as the candidate when the dialog opens. */
  initialFile?: File
  current?: Subject | Place
  onClose: () => void
  onDone: () => Promise<void> | void
}

export default function AddAsset({
  open, kind, subjectKind, worldId, visualStyle, assetKey, initialFile, current, onClose, onDone,
}: Props) {
  const initialName = kind === 'subject'
    ? (current as Subject | undefined)?.name ?? ''
    : assetKey ?? ''
  const [name, setName] = useState(initialName)
  const [val, setVal] = useState<RefValue>(emptyRef())
  // For a new asset it is kept and submitted with the character; for an existing one it is saved
  // immediately (see applyVoice).
  const [voice, setVoice] = useState<VoicePick>(null)
  const [busy, setBusy] = useState(false)
  const [candidateBusy, setCandidateBusy] = useState(false)
  const [note, setNote] = useState('')
  const editing = !!assetKey && !!current
  const hasCandidate = !!(val.file || val.ref)
  const isSubject = kind === 'subject'
  const currentSubject = isSubject ? current as Subject | undefined : undefined
  const entityKind = currentSubject?.kind || subjectKind || '人物'
  const isItem = ['物件', '物品', '道具'].includes(entityKind)

  useEffect(() => {
    if (!open) return
    setName(initialName)
    const url = initialFile ? URL.createObjectURL(initialFile) : ''
    setVal(url ? { ...emptyRef(), file: initialFile!, preview: url } : emptyRef())
    setVoice(null)
    setCandidateBusy(false)
    setNote(initialFile ? t('图已就位，起个名字就能加进来') : '')
    // Without revoke, blob URLs from the normal flow (paste → name → add) are never freed and pin
    // the image for the tab's lifetime; StrictMode runs the effect twice and orphans the first.
    return () => {
      if (url) URL.revokeObjectURL(url)
    }
  }, [open, initialName, assetKey, initialFile])

  const submit = async () => {
    if (candidateBusy) return
    if (!editing && !name.trim()) return setNote(t('先起个名字'))
    if (!hasCandidate) {
      return setNote(val.prompt.trim() ? t('先点“生成”查看候选图，再决定是否采用') : t('先上传一张图，或生成一张候选图'))
    }
    setBusy(true)
    try {
      const fd = new FormData()
      appendRef(fd, val)
      if (editing && kind === 'subject') {
        await replaceSubject(worldId, assetKey, fd)
      } else {
        fd.append('name', name.trim())
        if (kind === 'place') fd.append('make_current', 'false')
        if (kind === 'subject') fd.append('kind', isItem ? '物件' : '人物')
        if (kind === 'subject' && !isItem && voice?.voice_id) fd.append('voice_id', voice.voice_id)
        if (kind === 'subject' && !isItem && voice?.ref) fd.append('voice_ref', voice.ref)
        await (kind === 'subject' ? addSubject : addPlace)(worldId, fd)
      }
      await onDone()
      onClose()
    } catch (e) {
      setNote((e as Error).message.slice(0, 160))
    } finally {
      setBusy(false)
    }
  }

  /**
   * Changing an existing character's voice saves right away, like "who drives it": it does not
   * wait for "use candidate image", because the user changed the voice, not the image.
   */
  const applyVoice = async (pick: VoicePick) => {
    setVoice(pick)
    if (!editing || !assetKey) return
    try {
      await setSubjectVoice(worldId, assetKey, pick)
      await onDone()
    } catch (e) {
      setNote((e as Error).message.slice(0, 160))
    }
  }

  const voiceRow = (
    <div className="control-row voice-row">
      <span>{t('音色')}</span>
      <VoicePicker
        value={editing ? ((current as Subject | undefined)?.voice ?? null) : shownVoice(voice)}
        onChange={applyVoice}
        disabled={busy}
      />
      <small>
        {t('视频里这个角色开口就是这个声音，绑定之后每一段都一样。')}
        {t('头上那个「语音」开关念的是旁白，跟这儿无关。')}
      </small>
    </div>
  )
  // Monsters, mechs and key props take a reference image like people do; the title must say
  // what this is, otherwise "Character: mech monster" looks like a bug.
  const entity = isItem ? t('物品') : (isSubject && currentSubject?.kind) ? kindLabel(currentSubject.kind) : t('角色')
  return (
    <Dialog
      open={open}
      title={editing
        ? t('{kind}：{name}', { kind: isSubject ? entity : t('地点'), name: initialName ?? '' })
        : isItem ? t('添加物品') : isSubject ? t('添加角色') : t('添加场景')}
      sub={
        editing
          ? t('先上传或生成一张候选图，对比满意后再采用。替换只影响之后生成的片段，已有片段不会改变。')
          : t('上传一张参考图，或先用提示词生成并预览一张')
      }
      busy={busy || candidateBusy}
      okLabel={editing ? t('采用候选图') : t('添加')}
      okDisabled={!hasCandidate}
      note={note}
      onClose={onClose} onOk={submit}
    >
      {!editing && (
        <>
          <label>{t('名字')}</label>
          <input
            value={name}
            placeholder={isItem ? t('黄铜罗盘') : isSubject ? t('小明') : t('会议室')}
            onChange={(e) => setName(e.target.value)}
          />
          {isSubject && !isItem && voiceRow}
        </>
      )}
      <div className={'asset-editor' + (editing ? ' is-comparing' : '')}>
        {editing && isSubject && !isItem && (
          <div className="control-row">
            <span>{t('谁在驱动')}</span>
            <div className="seg">
              {([['user', t('你扮演')], ['ai', t('世界驱动')]] as const).map(([v, label]) => (
                <button
                  key={v}
                  className={((current as Subject)?.control ?? 'ai') === v ? 'on' : ''}
                  disabled={busy}
                  onClick={async () => {
                    try {
                      await setSubjectControl(worldId, assetKey!, v)
                      await onDone()
                    } catch (e) {
                      setNote((e as Error).message.slice(0, 160))
                    }
                  }}
                >
                  {label}
                </button>
              ))}
            </div>
            <small>
              {t('世界驱动的角色每一拍都会自己做点什么，不会站着等指令')}
            </small>
          </div>
        )}
        {editing && isSubject && !isItem && voiceRow}
        {editing && current && (
          <div className="asset-current">
            <div className="asset-editor-label">
              <span>{t('当前参考图')}</span>
              <small>{t('正在使用')}</small>
            </div>
            <div className="asset-current-card">
              <div className="asset-current-image">
                <img src={mediaUrl('refs', current.ref)} alt={t('{name}的当前参考图', { name: initialName ?? '' })} />
              </div>
              {(current as Subject | undefined)?.brief && (
                <div className="asset-current-brief">
                  {(current as Subject).brief}
                </div>
              )}
              <div className="asset-current-description">
                {current.definition || current.desc || t('暂无图像描述')}
              </div>
              <div className="asset-current-foot">
                <span>{t('当前资产已应用')}</span>
              </div>
            </div>
          </div>
        )}
        <div className="asset-candidate">
          {editing && (
            <div className="asset-editor-label">
              <span>{t('候选参考图')}</span>
              <small>{hasCandidate ? t('可采用') : t('尚未选择')}</small>
            </div>
          )}
          <RefPicker
            key={`${worldId}:${kind}:${assetKey ?? 'new'}:${open ? 'open' : 'closed'}:${initialFile?.name ?? ''}`}
            kind={isItem ? 'object' : kind} value={val} onChange={setVal}
            visualStyle={visualStyle}
            onBusyChange={setCandidateBusy}
            placeholder={
              isItem
                ? 'A single vintage brass compass, engraved lid, worn leather strap, clean neutral background'
                : isSubject
                ? 'A Chinese man in his late twenties, light blue shirt, dark trousers'
                : 'A cozy coffee shop interior, warm wood counter, hanging lamps'
            }
          />
        </div>
      </div>
    </Dialog>
  )
}
