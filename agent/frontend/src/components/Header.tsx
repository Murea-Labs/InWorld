import { useEffect, useRef, useState } from 'react'
import { exportUrl } from '../api'
import { MUREA_THEMES, type MureaTheme } from '../theme'
import WorldPicker from './WorldPicker'
import type { Health, WorldSummary } from '../types'
import { t } from '../i18n'

interface Props {
  /** Prebuilt demos are not marked in the UI: in a screen recording they should look like live generation */
  userName: string
  onSwitchUser: () => void
  health: Health | null
  worlds: WorldSummary[]
  current: string
  setting: string
  visualStyle: string
  onSelect: (id: string) => void
  onRename: (id: string, title: string) => Promise<void>
  onNew: () => void
  /** Roll a world and start playing right away */
  onRandom: () => void
  /** Rolling (about a second); only then show "rolling…" */
  randomBusy: boolean
  /** A beat is running. No new roll while the rolled world is still rendering. */
  busy: boolean
  onDelete: () => void
  onImport: (f: File) => void
  model?: string
  assetsOpen: boolean
  onToggleAssets: () => void
  /** The left drawer: map, quests, story log. */
  railOpen: boolean
  onToggleRail: () => void
  immersive: boolean
  onToggleImmersive: () => void
  theme: MureaTheme
  onThemeChange: (theme: MureaTheme) => void
  /** null = the backend has no TTS configured; the switch is disabled */
  tts: boolean | null
  /** false = configured but unreachable; also disabled, with a different message */
  ttsUp: boolean
  onToggleTts: () => void
  /** The world is speaking; show a moving hint on the button */
  speaking: boolean
  /** Open this world's outbound call log */
  onOpenLog: () => void
  /** Open the agent trace of the current clip (or the beat being made). */
  onOpenAgent: () => void
}

export default function Header({
  userName,
  onSwitchUser,
  health,
  worlds,
  current,
  setting,
  visualStyle,
  onSelect,
  onRename,
  onNew,
  onRandom,
  randomBusy,
  busy,
  onDelete,
  onImport,
  model,
  assetsOpen,
  onToggleAssets,
  immersive,
  onToggleImmersive,
  theme,
  onThemeChange,
  tts,
  ttsUp,
  onToggleTts,
  speaking,
  onOpenLog,
  onOpenAgent,
  railOpen,
  onToggleRail,
}: Props) {
  const fileRef = useRef<HTMLInputElement>(null)
  const actionsRef = useRef<HTMLDivElement>(null)
  const [moreOpen, setMoreOpen] = useState(false)
  const ok = health?.video_online

  useEffect(() => {
    if (!moreOpen) return
    const away = (event: MouseEvent) => {
      if (!actionsRef.current?.contains(event.target as Node)) setMoreOpen(false)
    }
    const escape = (event: KeyboardEvent) => event.key === 'Escape' && setMoreOpen(false)
    document.addEventListener('mousedown', away)
    document.addEventListener('keydown', escape)
    return () => {
      document.removeEventListener('mousedown', away)
      document.removeEventListener('keydown', escape)
    }
  }, [moreOpen])

  return (
    <header>
      <div className="brand">
        <h1>Murea-Mind</h1>
        {setting && <span className="setting">{setting}</span>}
      </div>

      {/* The world picker is the main navigation here, so it sits next to the title. */}
      <WorldPicker
        worlds={worlds}
        current={current}
        onSelect={onSelect}
        onRename={onRename}
      />

      <div className="spacer" />

      <div className="head-status">
        <span className="pill pill-video">
          <span className={'dot ' + (ok ? 'ok' : 'err')} />
          {ok ? t('视频服务在线') : t('视频服务离线')}
        </span>
        {health && <span className="pill pill-model">{health.text_model}</span>}
        {model && (
          <span className="pill pill-video" title={t('这个世界用的视频模型，建好之后不改')}>
            {model}
          </span>
        )}
        {visualStyle && (
          <span className="pill pill-style" title={t('锁定的视觉风格：{style}', { style: visualStyle })}>
            {/* This span is for text-overflow: the pill is inline-flex and bare text in flex
               does not get an ellipsis, it would be cut flat. */}
            <span>Style · {visualStyle.slice(0, 22)}{visualStyle.length > 22 ? '…' : ''}</span>
          </span>
        )}
        {health?.stale ? (
          // Python does not hot-reload. A backend change without a restart is invisible in the UI,
          // and once made agent latency look like a slow model. This one has no pill-version class
          // and never collapses on narrow screens.
          <span
            className="pill stale"
            title={t('进程跑的是 {version}，工作区已经是 {head} 了。改完后端必须重启（Ctrl+C 再 ./run.sh），Python 不会自己热重载。', { version: health.version ?? '', head: health.head ?? '' })}
          >
            <span className="dot err" />
            {t('服务是旧代码 · 重启')}
          </span>
        ) : health?.version ? (
          <span className="pill pill-version" title={t('当前运行的 commit')}>
            {health.version}
          </span>
        ) : null}
      </div>

      <div className="head-sep" />

      <div className="head-actions" ref={actionsRef}>
        <button
          className={'sm head-toggle' + (railOpen ? ' on' : '')}
          onClick={onToggleRail}
          disabled={immersive}
          title={t('地图、主线与任务、剧情记录')}
        >
          {t('世界')}
        </button>
        <button
          className={'sm head-toggle' + (assetsOpen ? ' on' : '')}
          onClick={onToggleAssets}
          disabled={immersive}
          title={assetsOpen ? t('收起资产表，把画面让出来') : t('角色、物品、地点与片段')}
        >
          {t('资产')}
        </button>
        <button
          className="sm head-toggle head-agent"
          onClick={onOpenAgent}
          title={t('看 agent 这一拍是怎么想、怎么做的：感知、规划、检查、行动、反思')}
        >
          Agent
        </button>
        <button className="sm head-new" onClick={onNew}>{t('新世界')}</button>
        <button
          type="button"
          className={'sm head-more-trigger' + (moreOpen ? ' on' : '')}
          onClick={() => setMoreOpen((value) => !value)}
          aria-expanded={moreOpen}
          aria-haspopup="menu"
          aria-label={t('更多')}
          title={t('更多')}
        >
          <svg viewBox="0 0 20 20" aria-hidden="true"><circle cx="4.5" cy="10" r="1.6" /><circle cx="10" cy="10" r="1.6" /><circle cx="15.5" cy="10" r="1.6" /></svg>
        </button>
        <button
          type="button"
          className="sm head-user"
          onClick={onSwitchUser}
          title={t('当前用户：{name}。点击切换用户', { name: userName })}
          aria-label={userName}
        >
          <span aria-hidden="true">{userName.slice(0, 1).toUpperCase()}</span>
        </button>
        {moreOpen && (
          <div className="head-more-menu" role="menu">
            <button type="button" className="head-more-item" role="menuitem"
              onClick={() => { setMoreOpen(false); onRandom() }} disabled={randomBusy || busy}>
              <span>{randomBusy ? t('摇…') : t('随机一个世界')}</span><em>{t('摇完直接开演')}</em>
            </button>
            <button type="button" className="head-more-item" role="menuitem"
              onClick={() => { setMoreOpen(false); onToggleImmersive() }}>
              <span>{immersive ? t('退出沉浸') : t('沉浸模式')}</span><em>Esc</em>
            </button>
            <button type="button" className="head-more-item" role="menuitem"
              onClick={() => { onToggleTts() }} disabled={tts === null || !ttsUp}>
              <span>{t('旁白配音')}</span><em>{tts === null || !ttsUp ? t('不可用') : speaking ? t('朗读中') : tts ? t('开') : t('关')}</em>
            </button>
            <div className="head-more-divider" />
            {/* The header pills collapse one by one on narrow screens (clickable buttons win the space),
               so the same items are listed here in full: what collapses is the display, not the information. */}
            <div className="head-more-label">
              <span>{t('运行状态')}</span>
              <em>{ok ? t('视频服务在线') : t('视频服务离线')}</em>
            </div>
            <div className="head-facts">
              <div className="head-fact"><span>{t('文本模型')}</span><em>{health?.text_model || '—'}</em></div>
              <div className="head-fact"><span>{t('视频模型')}</span><em>{model || '—'}</em></div>
              <div className="head-fact"><span>{t('视觉风格')}</span><em title={visualStyle}>{visualStyle || '—'}</em></div>
              <div className="head-fact">
                <span>{health?.stale ? t('进程 commit') : 'commit'}</span>
                <em className={health?.stale ? 'is-stale' : undefined}>
                  {health?.version || '—'}{health?.stale ? t(' · 旧代码') : ''}
                </em>
              </div>
            </div>
            <div className="head-more-divider" />
            <div className="head-theme-section" role="group" aria-label={t('界面主题')}>
              <div className="head-more-label">
                <span>{t('界面主题')}</span><em>{t('本机偏好')}</em>
              </div>
              <div className="head-theme-grid">
                {MUREA_THEMES.map((option) => (
                  <button
                    key={option.id}
                    type="button"
                    className={'head-theme-option' + (theme === option.id ? ' on' : '')}
                    role="menuitemradio"
                    aria-checked={theme === option.id}
                    title={option.description}
                    onClick={() => onThemeChange(option.id)}
                  >
                    <i style={{ background: option.swatch }} aria-hidden="true" />
                    <span>
                      <b>{option.label}</b>
                      <small>{option.description}</small>
                    </span>
                    <svg viewBox="0 0 16 16" aria-hidden="true">
                      <path d="m3.5 8.2 2.7 2.7 6.3-6.2" />
                    </svg>
                  </button>
                ))}
              </div>
            </div>
            <div className="head-more-divider" />
            <a
              className="head-more-item"
              href={current ? exportUrl(current) : undefined}
              title={t('导出成 zip：世界 + 片段 + 参考图，可以拿到别的环境导入')}
              aria-disabled={!current}
              role="menuitem"
              onClick={() => setMoreOpen(false)}
            >
              <span>{t('导出世界')}</span><em>ZIP</em>
            </a>
            <button
              type="button"
              className="head-more-item"
              role="menuitem"
              onClick={() => {
                setMoreOpen(false)
                fileRef.current?.click()
              }}
            >
              <span>{t('导入世界')}</span><em>ZIP</em>
            </button>
            <button
              type="button"
              className="head-more-item"
              role="menuitem"
              disabled={!current}
              onClick={() => {
                setMoreOpen(false)
                onOpenLog()
              }}
            >
              <span>{t('后台日志')}</span><em>DEBUG</em>
            </button>
            <button
              type="button"
              className="head-more-item is-danger"
              role="menuitem"
              disabled={!current}
              onClick={() => {
                setMoreOpen(false)
                onDelete()
              }}
            >
              <span>{t('删除当前世界')}</span>
            </button>
          </div>
        )}
        <input
          ref={fileRef} type="file" accept=".zip" style={{ display: 'none' }}
          onChange={(e) => {
            const f = e.target.files?.[0]
            e.target.value = ''
            if (f) onImport(f)
          }}
        />
      </div>
    </header>
  )
}
