import { useEffect, useRef, useState } from 'react'
import type { WorldSummary } from '../types'
import { t } from '../i18n'

interface Props {
  worlds: WorldSummary[]
  current: string
  onSelect: (id: string) => void
  onRename: (id: string, title: string) => Promise<void>
}

export default function WorldPicker({ worlds, current, onSelect, onRename }: Props) {
  const [open, setOpen] = useState(false)
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState('')
  const [saving, setSaving] = useState(false)
  const box = useRef<HTMLDivElement>(null)
  const active = worlds.find((w) => w.id === current)

  useEffect(() => {
    if (!open && !editing) return
    const away = (e: MouseEvent) => {
      if (!box.current?.contains(e.target as Node)) {
        setOpen(false)
        setEditing(false)
      }
    }
    const esc = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        setOpen(false)
        setEditing(false)
      }
    }
    document.addEventListener('mousedown', away)
    document.addEventListener('keydown', esc)
    return () => {
      document.removeEventListener('mousedown', away)
      document.removeEventListener('keydown', esc)
    }
  }, [open, editing])

  useEffect(() => {
    setEditing(false)
    setOpen(false)
  }, [current])

  const beginRename = () => {
    if (!active) return
    setOpen(false)
    setDraft(active.title || t('未命名世界'))
    setEditing(true)
  }

  return (
    <div className={'picker' + (editing ? ' is-editing' : '')} ref={box}>
      {editing && active ? (
        <form
          className="picker-rename-form"
          onSubmit={async (e) => {
            e.preventDefault()
            const title = draft.trim()
            if (!title || saving) return
            if (title === active.title) {
              setEditing(false)
              return
            }
            setSaving(true)
            try {
              await onRename(active.id, title)
              setEditing(false)
            } catch {
              // App already shows the server error as a toast; stay in edit mode so it can be fixed
              // and retried.
            } finally {
              setSaving(false)
            }
          }}
        >
          <input
            className="picker-rename-input"
            value={draft}
            maxLength={80}
            autoFocus
            aria-label={t('世界名称')}
            onChange={(e) => setDraft(e.target.value)}
            onFocus={(e) => e.currentTarget.select()}
          />
          <button type="submit" className="picker-rename-save" disabled={!draft.trim() || saving}>
            {saving ? t('保存中') : t('保存')}
          </button>
          <button
            type="button"
            className="picker-rename-cancel"
            aria-label={t('取消修改世界名称')}
            title={t('取消')}
            disabled={saving}
            onClick={() => setEditing(false)}
          >
            ×
          </button>
        </form>
      ) : (
        <>
          <button className="picker-btn" onClick={() => setOpen((v) => !v)}>
            <span className="picker-name">{active?.title || t('（还没有世界）')}</span>
            {active && <span className="picker-seg">{active.segments}</span>}
            <svg width="9" height="6" viewBox="0 0 9 6" className={open ? 'up' : ''}>
              <path d="M1 1l3.5 3.5L8 1" stroke="currentColor" strokeWidth="1.6" fill="none" />
            </svg>
          </button>
          {active && (
            <button
              type="button"
              className="picker-rename-trigger"
              onClick={beginRename}
              aria-label={t('修改世界名称：{title}', { title: active.title })}
              title={t('修改当前世界名称')}
            >
              <svg viewBox="0 0 24 24" aria-hidden="true">
                <path d="M4 20h4l11-11-4-4L4 16v4Zm9.5-13.5 4 4" />
              </svg>
            </button>
          )}
        </>
      )}

      {open && !editing && (
        <div className="picker-menu">
          {worlds.length === 0 && <div className="picker-empty">{t('还没有世界')}</div>}
          {worlds.map((w) => (
            <button
              key={w.id}
              className={'picker-item' + (w.id === current ? ' on' : '')}
              onClick={() => {
                setOpen(false)
                if (w.id !== current) onSelect(w.id)
              }}
            >
              <span className="picker-name">{w.title || t('未命名')}</span>
              <span className="picker-seg">{w.segments}</span>
            </button>
          ))}
        </div>
      )}
    </div>
  )
}
