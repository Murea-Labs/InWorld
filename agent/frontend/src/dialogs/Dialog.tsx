import { useEffect, useRef } from 'react'
import { isPickingFile } from '../filepicker'
import { t } from '../i18n'

interface Props {
  open: boolean
  title: string
  sub?: string
  busy?: boolean
  okLabel?: string
  /** Optional visual variant for destructive confirmations. */
  okClassName?: string
  okDisabled?: boolean
  note?: string
  className?: string
  onClose: () => void
  onOk: () => void
  children: React.ReactNode
}

export default function Dialog({
  open, title, sub, busy, okLabel = t('确定'), okClassName, okDisabled, note, className, onClose, onOk, children,
}: Props) {
  const ref = useRef<HTMLDialogElement>(null)
  useEffect(() => {
    const d = ref.current
    if (!d) return
    if (open && !d.open) d.showModal()
    if (!open && d.open) d.close()
  }, [open])

  return (
    <dialog
      ref={ref}
      className={className}
      onCancel={(e) => {
        e.preventDefault()
        if (busy) return
        // Esc that merely dismissed a file panel must not close the dialog too.
        if (isPickingFile()) return
        onClose()
      }}
    >
      <div className="body">
        <h2>{title}</h2>
        {sub && <p className="sub">{sub}</p>}
        {children}
      </div>
      <div className="actions">
        {note && <span className="note">{note}</span>}
        <div className="spacer" />
        <button onClick={onClose} disabled={busy}>{t('取消')}</button>
        <button
          className={['primary', okClassName].filter(Boolean).join(' ')}
          onClick={onOk}
          disabled={busy || okDisabled}
        >
          {busy ? t('处理中…') : okLabel}
        </button>
      </div>
    </dialog>
  )
}
