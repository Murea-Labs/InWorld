import { useEffect, useRef, useState } from 'react'
import { mediaUrl } from '../api'
import Dialog from '../dialogs/Dialog'
import { lineage, takesOf, visiblePlaces } from '../lib'
import type { Node, PlaceState, SubjectState, World } from '../types'
import { kindLabel, t } from '../i18n'

interface Props {
  world: World | null
  /** The clip shown in the player. The map and the clip tree share this selection. */
  current?: Node | null
  busy: boolean
  onPlay: (n: Node) => void
  onAddSubject: () => void
  onAddItem: () => void
  onAddPlace: () => void
  onEditSubject: (sid: string) => void
  onEditPlace: (name: string) => void
  onDeleteItem: (sid: string) => Promise<void>
}

export default function Assets({
  world, current, busy, onPlay, onAddSubject, onAddItem, onAddPlace, onEditSubject, onEditPlace,
  onDeleteItem,
}: Props) {
  const [deletingItem, setDeletingItem] = useState<{ sid: string; name: string } | null>(null)
  const [deleteBusy, setDeleteBusy] = useState(false)
  const [deleteNote, setDeleteNote] = useState('')
  const headChain = lineage(world)
  // The inspector can select an alternate take that is not on the current head lineage,
  // and the map can review branch nodes left after a rollback. Attach the selected branch
  // after the current chain from its fork point, for display only; world.head and the next
  // beat are unchanged.
  const currentInWorld = current && world?.nodes[current.id]
    ? world.nodes[current.id]
    : null
  const currentIsOnHead = !!currentInWorld
    && headChain.some((node) => node.id === currentInWorld.id)
  const replayPath = currentInWorld && !currentIsOnHead && world
    ? lineageToNode(world, currentInWorld.id)
    : []
  let shared = 0
  while (
    shared < headChain.length
    && shared < replayPath.length
    && headChain[shared].id === replayPath[shared].id
  ) shared += 1
  const replayNodes = replayPath.slice(shared)
  const chain = replayNodes.length ? [...headChain, ...replayNodes] : headChain
  const splitParts = splitPartLabels(chain)
  const selectedRowId = currentInWorld && chain.some((node) => node.id === currentInWorld.id)
    ? currentInWorld.id
    : ''
  const replayStart = replayNodes.length ? headChain.length : -1
  const selectedRowRef = useRef<HTMLDivElement | null>(null)

  // When an older clip is selected, the map jumps to another place, or the panel reopens,
  // scroll the matching row into view. scrollIntoView is missing in some test DOMs and old
  // browsers; skip silently.
  useEffect(() => {
    const row = selectedRowRef.current
    if (!row || typeof row.scrollIntoView !== 'function') return
    row.scrollIntoView({ block: 'nearest', behavior: 'smooth' })
  }, [selectedRowId, replayStart, world?.id])

  useEffect(() => {
    setDeletingItem(null)
    setDeleteBusy(false)
    setDeleteNote('')
  }, [world?.id])

  const confirmDeleteItem = async () => {
    if (!deletingItem) return
    setDeleteBusy(true)
    setDeleteNote('')
    try {
      await onDeleteItem(deletingItem.sid)
      setDeletingItem(null)
    } catch (error) {
      setDeleteNote((error as Error).message.slice(0, 160))
    } finally {
      setDeleteBusy(false)
    }
  }

  const allSubjects = Object.entries(world?.subjects ?? {})
  const subjects = allSubjects.filter(([, subject]) => !isItem(subject.kind))
  const items = allSubjects.filter(([, subject]) => isItem(subject.kind))
  const places = visiblePlaces(world)

  return (
    <section id="assets" className="rail panel">
      <div className="panel-head">{t('资产表')}</div>

      <div className="scroll">
        <div className="group">
          <h3>{t('当前位置')}</h3>
          <span className="chip here">{world?.here || '—'}</span>
        </div>

        <div className="group">
          <h3>
            {t('所有角色')}
            <button className="sm ghost" onClick={onAddSubject} disabled={busy} title={t('添加角色')}>
              +
            </button>
          </h3>
          <div className="asset-grid">
            {subjects.map(([sid, s]) => (
              <button
                type="button"
                className="asset-card is-subject"
                key={sid}
                title={t('查看并更换「{name}」的参考图', { name: s.name })}
                onClick={() => onEditSubject(sid)}
                disabled={busy}
              >
                <span className="asset-card-image">
                  <img src={mediaUrl('refs', s.ref)} alt={t('{name}的当前参考图', { name: s.name })} />
                  {s.control === 'user' && <em>{t('你')}</em>}
                </span>
                <span className="asset-card-meta">
                  <b>
                    {s.name}
                    {s.kind && s.kind !== '人物' && (
                      <em className="asset-kind">{kindLabel(s.kind)}</em>
                    )}
                  </b>
                  <small>{s.control === 'user' ? t('你扮演') : t('世界驱动')} · {t('点击查看')}</small>
                  {subjectStateLine(s.state) && (
                    <small title={t('这一拍结束时的动态状态')}>{subjectStateLine(s.state)}</small>
                  )}
                </span>
              </button>
            ))}
          </div>
          {!subjects.length && <div className="none">{t('还没有角色')}</div>}
        </div>

        <div className="group">
          <h3>
            {t('所有物品')}
            <button className="sm ghost" onClick={onAddItem} disabled={busy} title={t('添加物品')}>
              +
            </button>
          </h3>
          <div className="asset-grid">
            {items.map(([sid, item]) => (
              <div className="asset-card-wrap" key={sid}>
                <button
                  type="button"
                  className="asset-card is-item"
                  title={t('查看并更换「{name}」的参考图', { name: item.name })}
                  onClick={() => onEditSubject(sid)}
                  disabled={busy}
                >
                  <span className="asset-card-image">
                    <img src={mediaUrl('refs', item.ref)} alt={t('{name}的当前参考图', { name: item.name })} />
                  </span>
                  <span className="asset-card-meta">
                    <b>{item.name}</b>
                    <small>{item.brief || t('点击查看')}</small>
                    {item.state?.condition && <small title={t('物品当前状态')}>{item.state.condition}</small>}
                    {itemReferenceLine(item) && (
                      <small className={`asset-ref-status ${item.ref_status ?? ''}`}>
                        {itemReferenceLine(item)}
                      </small>
                    )}
                  </span>
                </button>
                <button
                  type="button"
                  className="asset-card-delete"
                  aria-label={t('删除物品「{name}」', { name: item.name })}
                  title={t('删除「{name}」', { name: item.name })}
                  disabled={busy}
                  onClick={() => {
                    setDeleteNote('')
                    setDeletingItem({ sid, name: item.name })
                  }}
                >
                  <svg viewBox="0 0 16 16" aria-hidden="true">
                    <path d="M3.5 4.5h9M6 4.5V3h4v1.5m1.4 0-.5 8.5h-5.8l-.5-8.5M6.8 7v3.5M9.2 7v3.5" />
                  </svg>
                </button>
              </div>
            ))}
          </div>
          {!items.length && <div className="none">{t('还没有物品')}</div>}
        </div>

        <div className="group">
          <h3>
            {t('去过的地方')}
            <button className="sm ghost" onClick={onAddPlace} disabled={busy} title={t('添加场景')}>
              +
            </button>
          </h3>
          <div className="asset-grid">
            {places.map(([name, p]) => (
              <button
                type="button"
                className="asset-card is-place"
                key={name}
                title={t('查看并更换「{name}」的参考图', { name })}
                onClick={() => onEditPlace(name)}
                disabled={busy}
              >
                <span className="asset-card-image">
                  <img src={mediaUrl('refs', p.ref)} alt={t('{name}的当前参考图', { name })} />
                  {world?.here === name && <em>{t('当前')}</em>}
                </span>
                <span className="asset-card-meta">
                  <b>{name}</b>
                  {/* If the director wrote a description, show what this place is instead of a "click to view" line */}
                  <small>
                    {world?.here === name ? t('当前位置') : p.brief || t('点击查看')}
                  </small>
                  {placeStateLine(p.state) && (
                    <small title={t('这一拍结束时的场景状态')}>{placeStateLine(p.state)}</small>
                  )}
                </span>
              </button>
            ))}
          </div>
          {!places.length && <div className="none">{t('还没去过任何地方')}</div>}
        </div>

        <div className="group">
          <h3>
            {t('片段树')}
            {replayNodes.length > 0 && (
              <span className="tag take" title={t('当前正在回看主线之外的旁支，点击主线行可返回主线')}>
                {t('旁支回看')}
              </span>
            )}
          </h3>
          <div id="tree">
            {!chain.length && <div className="none">—</div>}
            {chain.map((n, i) => {
              const takes = world ? takesOf(world, n) : []
              const splitPart = splitParts.get(n.id)
              const isReplay = i >= replayStart && replayStart >= 0
              return (
                <div
                  className={[
                    'row',
                    n.id === selectedRowId ? 'is-current' : '',
                    isReplay ? 'is-replay' : '',
                  ].filter(Boolean).join(' ')}
                  ref={n.id === selectedRowId ? selectedRowRef : undefined}
                  key={n.id}
                >
                  <span className="rail-mark">{isReplay ? '└' : n.mode === 'continue' ? '│' : '┌'}</span>
                  <a
                    href="#"
                    className={[
                      n.id === world?.head ? 'head' : '',
                      n.id === selectedRowId ? 'is-current' : '',
                    ].filter(Boolean).join(' ')}
                    aria-current={n.id === selectedRowId ? 'step' : undefined}
                    title={isReplay ? t('旁支回看：{beat}', { beat: n.beat }) : n.beat}
                    onClick={(e) => {
                      e.preventDefault()
                      onPlay(n)
                    }}
                  >
                    <span className="n">{String(i + 1).padStart(2, '0')}</span>
                    {n.place}
                  </a>
                  {splitPart && (
                    <span
                      className="tag cont"
                      title={t('同一次输入自动拆成 {total} 个独立视频，这是第 {part} 段', { total: splitPart.total, part: splitPart.part })}
                    >
                      {t('拆 {part}/{total}', { part: splitPart.part, total: splitPart.total })}
                    </span>
                  )}
                  {takes.length > 1 && (
                    <span
                      className="tag take"
                      title={t('同一拍有 {n} 个生成版本，可在片段检查器中切换', { n: takes.length })}
                    >
                      {t('{n} 版', { n: takes.length })}
                    </span>
                  )}
                </div>
              )
            })}
          </div>
        </div>
      </div>

      <Dialog
        open={!!deletingItem}
        title={t('删除物品：{name}', { name: deletingItem?.name ?? '' })}
        sub={t('删除后，这件物品不会再作为参考资产参与后续生成。')}
        busy={deleteBusy}
        okLabel={t('确认删除')}
        okClassName="danger"
        note={deleteNote}
        onClose={() => {
          if (!deleteBusy) setDeletingItem(null)
        }}
        onOk={confirmDeleteItem}
      >
        <div className="rollback-confirm" role="alert">
          <span className="rollback-confirm-icon" aria-hidden="true">!</span>
          <div>
            <strong>{t('请确认删除这项物品资产')}</strong>
            <p>{t('资产会从物品栏目和当前世界状态中移除，已有视频与历史片段不会改变。删除后无法通过界面恢复，但可以重新添加。')}</p>
          </div>
        </div>
      </Dialog>
    </section>
  )
}

/** Consecutive auto nodes belong to the user/system turn immediately before them. */
function splitPartLabels(chain: Node[]): Map<string, { part: number; total: number }> {
  const out = new Map<string, { part: number; total: number }>()
  for (let start = 0; start < chain.length;) {
    let end = start + 1
    while (end < chain.length && chain[end].origin === 'auto') end += 1
    const total = end - start
    if (total > 1) {
      for (let index = start; index < end; index += 1) {
        out.set(chain[index].id, { part: index - start + 1, total })
      }
    }
    start = end
  }
  return out
}

/** Root -> an arbitrary node, including a detached rollback branch. */
function lineageToNode(world: World, id: string): Node[] {
  const out: Node[] = []
  const seen = new Set<string>()
  let cursor: string | null = id
  while (cursor && world.nodes[cursor] && !seen.has(cursor)) {
    seen.add(cursor)
    const node: Node = world.nodes[cursor]
    out.push(node)
    cursor = node.parent
  }
  return out.reverse()
}

/** The director's canonical value is "物件"; older or hand-imported worlds often use "物品 / 道具". */
function isItem(kind?: string): boolean {
  return ['物件', '物品', '道具'].includes((kind || '').trim())
}

function subjectStateLine(state?: SubjectState): string {
  if (!state) return ''
  return [state.outfit, state.condition, state.mood]
    .map((x) => (x || '').trim()).filter(Boolean).join(' · ').slice(0, 90)
}

function itemReferenceLine(item: { ref_status?: string; ref_source?: string; ref_error?: string }): string {
  if (item.ref_status === 'pending') return t('正在生成独立参考图…')
  if (item.ref_status === 'failed') return t('独立参考图生成失败，当前使用尾帧')
  if (item.ref_status === 'ready' && item.ref_source === 'generated') return t('独立参考图已更新')
  return ''
}

function placeStateLine(state?: PlaceState): string {
  if (!state) return ''
  return [state.time_of_day, state.weather, state.lighting, state.condition]
    .map((x) => (x || '').trim()).filter(Boolean).join(' · ').slice(0, 90)
}
