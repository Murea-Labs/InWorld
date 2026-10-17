import { useEffect, useRef, useState } from 'react'
import { imagine, mediaUrl } from '../api'
import { markFilePickerOpen } from '../filepicker'
import { t } from '../i18n'

export interface RefValue {
  ref: string          // filename already on the server
  file: File | null    // or a local upload
  preview: string      // object URL or server URL
  prompt: string
}

export const emptyRef = (): RefValue => ({ ref: '', file: null, preview: '', prompt: '' })

interface Props {
  kind: 'place' | 'subject' | 'object'
  value: RefValue
  onChange: (v: RefValue) => void
  placeholder?: string
  visualStyle?: string
  onBusyChange?: (busy: boolean) => void
  /** Type switch, name field and remove button, rendered in the card header. */
  header?: React.ReactNode
}

/** One reference asset: upload an image, or write a prompt and draw one. */
export default function RefPicker({
  kind, value, onChange, placeholder, visualStyle = '', onBusyChange, header,
}: Props) {
  const fileRef = useRef<HTMLInputElement>(null)
  const requestRef = useRef(0)
  const mountedRef = useRef(false)
  const [busy, setBusy] = useState(false)
  const [note, setNote] = useState('')
  const [failed, setFailed] = useState(false)

  useEffect(() => {
    mountedRef.current = true
    return () => {
      mountedRef.current = false
      requestRef.current += 1
    }
  }, [])

  const setDrawing = (next: boolean) => {
    if (!mountedRef.current) return
    setBusy(next)
    onBusyChange?.(next)
  }

  const cancelPending = () => {
    requestRef.current += 1
    setDrawing(false)
  }

  const pickFile = (f: File | null) => {
    if (!f) return
    cancelPending()
    if (value.preview && value.file) URL.revokeObjectURL(value.preview)
    onChange({ ...value, file: f, ref: '', preview: URL.createObjectURL(f) })
    setNote(f.name)
    setFailed(false)
  }

  const draw = async () => {
    const prompt = value.prompt.trim()
    if (!prompt) {
      setFailed(true)
      return setNote(t('先写一句提示词，或者点上面的框上传图片'))
    }
    const request = ++requestRef.current
    setDrawing(true)
    setFailed(false)
    setNote(t('生成中…'))
    try {
      const g = await imagine(prompt, kind, visualStyle)
      if (!mountedRef.current || request !== requestRef.current) return
      if (value.preview && value.file) URL.revokeObjectURL(value.preview)
      onChange({
        ...value, ref: g.ref, file: null,
        preview: `${mediaUrl('refs', g.ref)}?t=${Date.now()}`,
      })
      setNote(`${g.elapsed}s`)
    } catch (e) {
      if (!mountedRef.current || request !== requestRef.current) return
      setFailed(true)
      setNote(String((e as Error).message).replace(/^\{"detail":"|"\}$/g, '').slice(0, 150))
    } finally {
      if (request === requestRef.current) setDrawing(false)
    }
  }

  const open = () => {
    markFilePickerOpen()
    fileRef.current?.click()
  }

  return (
    <div className="refcard">
      {header && <div className="refhead">{header}</div>}

      <div className={'refthumb' + (busy ? ' busy' : '')} onClick={open}>
        {value.preview ? (
          <>
            <img src={value.preview} alt="" />
            <span className="refswap">{t('换一张')}</span>
          </>
        ) : (
          <div className="refempty">
            <b>{kind === 'object' ? t('点这里上传物品图') : kind === 'subject' ? t('点这里上传人物图') : t('点这里上传场景图')}</b>
            <span>{t('或在下面写一句提示词生成')}</span>
          </div>
        )}
      </div>

      <textarea
        className="refprompt"
        rows={2}
        value={value.prompt}
        placeholder={placeholder}
        onChange={(e) => {
          const prompt = e.target.value
          const invalidatesGeneratedRef = !!value.ref
          cancelPending()
          onChange({
            ...value,
            prompt,
            ref: invalidatesGeneratedRef ? '' : value.ref,
            preview: invalidatesGeneratedRef ? '' : value.preview,
          })
          setFailed(false)
          setNote(invalidatesGeneratedRef ? t('提示词已修改，请重新生成') : '')
        }}
      />

      <div className="reffoot">
        <span className={'refnote' + (failed ? ' err' : '')}>{note}</span>
        <button className="sm" onClick={draw} disabled={busy}>
          {busy ? t('生成中…') : value.ref ? t('重新生成') : t('生成图片')}
        </button>
      </div>

      <input
        ref={fileRef} type="file" accept="image/*" style={{ display: 'none' }}
        onChange={(e) => pickFile(e.target.files?.[0] ?? null)}
      />
    </div>
  )
}

/** Turn a picker value into the multipart fields the API expects. */
export function appendRef(form: FormData, v: RefValue) {
  form.append('image_prompt', v.prompt.trim())
  if (v.ref) form.append('ref', v.ref)
  if (v.file) form.append('image', v.file)
}
