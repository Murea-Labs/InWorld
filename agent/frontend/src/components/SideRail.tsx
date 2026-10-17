import { useState } from 'react'
import type { TurnState } from '../App'
import type { MureaTheme } from '../theme'
import type { Node, World } from '../types'
import Chat from './Chat'
import WorldMapMini from './WorldMapMini'
import Dialog from '../dialogs/Dialog'
import { t } from '../i18n'

export type RailTab = 'map' | 'chat'

interface Props {
  world: World | null
  current: Node | null
  theme: MureaTheme
  tab: RailTab
  onTab: (t: RailTab) => void
  busy: boolean
  pending: string
  turn: TurnState | null
  onSelect: (n: Node) => void
  onOpenMap: () => void
  onInspect: (n: Node) => void
  onOpenAsset: (kind: 'subject' | 'place', key: string) => void
  onUndo: () => void | Promise<void>
}

/**
 * The left drawer. The map and the full chat log are two tabs of the same panel; the map is the
 * default.
 * 
 * The chat is not the default because the dialogue is already shown over the picture (Hud);
 * here it is for looking back: how did the last clip come about, open a clip in the inspector.
 * The map is the default because "where am I in this world" is what people want at any time.
 */
export default function SideRail({
  world, current, theme, tab, onTab, busy, pending, turn,
  onSelect, onOpenMap, onInspect, onOpenAsset, onUndo,
}: Props) {
  const [confirmingUndo, setConfirmingUndo] = useState(false)
  const [undoing, setUndoing] = useState(false)
  const canUndo = !!(world?.head && world.nodes[world.head]?.parent)

  const confirmUndo = async () => {
    if (!canUndo || busy || undoing) return
    setUndoing(true)
    try {
      await onUndo()
    } finally {
      setUndoing(false)
      setConfirmingUndo(false)
    }
  }

  return (
    <>
      <section id="side-rail" className="rail panel" aria-label={t('世界地图与对话')}>
      <div className="rail-tabs" role="tablist">
        <button
          role="tab"
          aria-selected={tab === 'map'}
          className={'rail-tab' + (tab === 'map' ? ' on' : '')}
          onClick={() => onTab('map')}
        >
          {t('地图')}
        </button>
        <button
          role="tab"
          aria-selected={tab === 'chat'}
          className={'rail-tab' + (tab === 'chat' ? ' on' : '')}
          onClick={() => onTab('chat')}
        >
          {t('对话')}
          {world?.history.length ? <em>{world.history.length}</em> : null}
        </button>
        <div className="spacer" />
        {/* Only on the chat tab. Undo removes the last line of the chat log and the clip it made,
           which is only visible on the chat tab; on the map tab nothing would visibly change. */}
        {tab === 'chat' && (
          <button
            className="sm ghost" onClick={() => setConfirmingUndo(true)}
            disabled={busy || !canUndo}
            title={t(canUndo ? '撤回最新一段' : '世界第一段不能撤回')}
          >
            {t('撤回')}
          </button>
        )}
      </div>

      {/* Both tabs stay mounted and are only hidden. The map would re-measure and redraw on every
         mount and flicker when switching tabs; the two together are few nodes. */}
      <div className="rail-body" hidden={tab !== 'map'}>
        <WorldMapMini
          world={world}
          current={current}
          theme={theme}
          onSelect={onSelect}
          onOpen={onOpenMap}
        />
      </div>
      <div className="rail-body" hidden={tab !== 'chat'}>
        <Chat
          world={world}
          pending={pending}
          turn={turn}
          active={tab === 'chat'}
          onInspect={onInspect}
          onOpenAsset={onOpenAsset}
        />
      </div>
      </section>

      <Dialog
        open={confirmingUndo}
        title={t('撤回最新一段？')}
        sub={t('这个操作不可逆。确认后，当前剧情线会回到上一段生成结果。')}
        busy={busy || undoing}
        okLabel={t('确认撤回')}
        okClassName="danger"
        onClose={() => {
          if (!undoing && !busy) setConfirmingUndo(false)
        }}
        onOk={confirmUndo}
      >
        <div className="rollback-confirm" role="alert">
          <span className="rollback-confirm-icon" aria-hidden="true">!</span>
          <div>
            <strong>{t('请确认这是你想要的操作')}</strong>
            <p>{t('最新片段会从当前剧情线移除，相关对话记录和世界状态也会回退。之后无法通过界面恢复，请在确认前确保不需要保留这一段。')}</p>
          </div>
        </div>
      </Dialog>
    </>
  )
}
