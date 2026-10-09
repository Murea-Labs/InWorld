import { useEffect, useMemo, useRef, useState } from 'react'
import { mediaUrl } from '../api'
import { lineage, visiblePlaces } from '../lib'
import type { MureaTheme } from '../theme'
import type { Node, Quest, World } from '../types'
import { sceneGlyphFor } from './SceneGlyph'
import { t } from '../i18n'

interface Props {
  world: World | null
  current: Node | null
  theme: MureaTheme
  onOpen: () => void
  onSelect: (node: Node) => void
}

interface MiniNode {
  name: string
  x: number
  y: number
  children: number
  latest: Node | null
  quests: Quest[]
}

interface MiniEdge {
  key: string
  from: MiniNode
  to: MiniNode
}

function mainSceneFor(world: World, name: string): string {
  let current = name
  const seen = new Set<string>()
  while (world.places[current] && !seen.has(current)) {
    seen.add(current)
    const parent = (world.places[current].main_scene || current).trim()
    if (!world.places[parent] || parent === current) return current
    current = parent
  }
  return name
}

function shortName(name: string): string {
  return name.length > 7 ? `${name.slice(0, 6)}…` : name
}

const CANVAS_W = 300
const CANVAS_H = 220

function roundedRect(
  ctx: CanvasRenderingContext2D, x: number, y: number, width: number, height: number, radius: number,
) {
  const r = Math.min(radius, width / 2, height / 2)
  ctx.beginPath()
  ctx.moveTo(x + r, y)
  ctx.lineTo(x + width - r, y)
  ctx.quadraticCurveTo(x + width, y, x + width, y + r)
  ctx.lineTo(x + width, y + height - r)
  ctx.quadraticCurveTo(x + width, y + height, x + width - r, y + height)
  ctx.lineTo(x + r, y + height)
  ctx.quadraticCurveTo(x, y + height, x, y + height - r)
  ctx.lineTo(x, y + r)
  ctx.quadraticCurveTo(x, y, x + r, y)
  ctx.closePath()
}

/** Small map marker: icon on top, a pointer at the bottom. */
function markerPath(ctx: CanvasRenderingContext2D, x: number, y: number, radius = 10) {
  const centerY = y - 3
  ctx.beginPath()
  ctx.moveTo(x, y + radius + 2)
  ctx.bezierCurveTo(x - 2, y + radius - 1, x - radius, y + 4, x - radius, centerY)
  ctx.arc(x, centerY, radius, Math.PI, 0)
  ctx.bezierCurveTo(x + radius, y + 4, x + 2, y + radius - 1, x, y + radius + 2)
  ctx.closePath()
}

function drawSceneGlyph(
  ctx: CanvasRenderingContext2D, name: string, x: number, y: number, color: string,
) {
  const glyph = sceneGlyphFor(name)
  const size = 13
  ctx.save()
  ctx.translate(x - size / 2, y - 3 - size / 2)
  ctx.scale(size / 24, size / 24)
  ctx.strokeStyle = color
  ctx.lineWidth = 1.7 * (24 / size)
  ctx.lineCap = 'round'
  ctx.lineJoin = 'round'
  for (const path of glyph.paths) ctx.stroke(new Path2D(path))
  ctx.restore()
}

export default function WorldMapMini({ world, current, theme, onOpen, onSelect }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const transformRef = useRef({ scale: 1, offsetX: 0, offsetY: 0 })
  const [hovered, setHovered] = useState('')

  // When switching worlds, world and current update separately. Only accept nodes of the
  // current world, and let quest state follow the clip being reviewed instead of leaking
  // future results from the current branch.
  const activeCurrent = world && current ? world.nodes[current.id] ?? null : null
  const questState = useMemo(() => {
    if (!world) return {
      quests: [] as Quest[], objective: '', mainline: '', mainline_phase: '',
      mainline_milestones: [] as string[],
    }
    const node = activeCurrent ?? (world.head ? world.nodes[world.head] : undefined)
    return node?.quest_after ?? {
      quests: world.quests ?? [],
      objective: node?.objective || world.objective || '',
      mainline: world.mainline || '',
      mainline_phase: world.mainline_phase || '',
      mainline_milestones: world.mainline_milestones ?? [],
    }
  }, [world, activeCurrent])

  const graph = useMemo(() => {
    if (!world) return { nodes: [] as MiniNode[], edges: [] as MiniEdge[], current: '' }

    const grouped = new Map<string, string[]>()
    for (const [name] of visiblePlaces(world)) {
      const root = mainSceneFor(world, name)
      const scenes = grouped.get(root) ?? []
      scenes.push(name)
      grouped.set(root, scenes)
    }

    const roots = [...grouped.keys()]
    const chain = lineage(world)
    const latestByRoot = new Map<string, Node>()
    for (const node of chain) latestByRoot.set(mainSceneFor(world, node.place), node)
    // Manually added map nodes may not be on the current story chain; still allow replaying the
    // last clip made there.
    const activeRoots = new Set(latestByRoot.keys())
    for (const node of Object.values(world.nodes).sort((a, b) => a.created.localeCompare(b.created))) {
      const root = mainSceneFor(world, node.place)
      if (!activeRoots.has(root)) latestByRoot.set(root, node)
    }
    const nodes = roots.map((name, index): MiniNode => {
      const angle = roots.length === 1 ? 0 : -Math.PI / 2 + (index * Math.PI * 2) / roots.length
      return {
        name,
        x: roots.length === 1 ? 150 : 150 + Math.cos(angle) * 105,
        y: roots.length === 1 ? 108 : 108 + Math.sin(angle) * 70,
        children: Math.max(0, (grouped.get(name)?.length ?? 1) - 1),
        latest: latestByRoot.get(name) ?? null,
        quests: [],
      }
    })
    const byName = new Map(nodes.map((node) => [node.name, node]))
    for (const quest of questState.quests) {
      if (!quest.place || !world.places[quest.place]) continue
      byName.get(mainSceneFor(world, quest.place))?.quests.push(quest)
    }
    const pairs = new Set<string>()
    const add = (from: string, to: string) => {
      const rootFrom = mainSceneFor(world, from)
      const rootTo = mainSceneFor(world, to)
      if (rootFrom === rootTo || !byName.has(rootFrom) || !byName.has(rootTo)) return
      const a = roots.indexOf(rootFrom) <= roots.indexOf(rootTo) ? rootFrom : rootTo
      const b = a === rootFrom ? rootTo : rootFrom
      pairs.add(`${a}\u0000${b}`)
    }

    for (let index = 1; index < chain.length; index += 1) {
      add(chain[index - 1].place, chain[index].place)
    }
    for (const link of world.place_links ?? []) add(link.from, link.to)

    const edges = [...pairs].flatMap((key): MiniEdge[] => {
      const [a, b] = key.split('\u0000')
      const from = byName.get(a)
      const to = byName.get(b)
      return from && to ? [{ key, from, to }] : []
    })

    return {
      nodes,
      edges,
      current: world.here ? mainSceneFor(world, world.here) : '',
    }
  }, [world, questState.quests])

  const selectedRoot = world && activeCurrent?.place && world.places[activeCurrent.place]
    ? mainSceneFor(world, activeCurrent.place)
    : graph.current

  const context = useMemo(() => {
    if (!world) return {
      root: '', currentScene: '', sceneRef: '', characters: [] as { id: string; name: string; ref: string }[],
      event: '', setting: '', mainline: '', mainlinePhase: '', objective: '',
      mainlineMilestones: [] as string[], mainQuest: null as Quest | null,
      sideQuests: [] as Quest[],
    }

    const currentNode = activeCurrent ?? (world.head ? world.nodes[world.head] : undefined)
    const currentScene = currentNode?.place && world.places[currentNode.place]
      ? currentNode.place
      : world.here
    const root = currentScene ? mainSceneFor(world, currentScene) : graph.current
    const scene = world.places[root] ?? world.places[world.here]
    const activeSubjects = new Set(currentNode?.subjects ?? [])
    const allCharacters = Object.entries(world.subjects).filter(([, subject]) => (
      !['物件', '物品', '道具'].includes((subject.kind || '').trim())
    ))
    const activeCharacters = allCharacters.filter(([id, subject]) =>
      activeSubjects.has(id) || activeSubjects.has(subject.name))
    const visibleCharacters = (activeCharacters.length ? activeCharacters : allCharacters)
      .slice(0, 3)
      .map(([id, subject]) => ({ id, name: subject.name, ref: subject.ref }))
    const mainQuest = questState.quests.find((quest) => quest.type === 'main') ?? null
    const sideQuests = questState.quests
      .filter((quest) => quest.type === 'side')
      .sort((a, b) => Number(a.status !== 'active') - Number(b.status !== 'active'))

    return {
      root,
      currentScene,
      sceneRef: scene?.ref ?? '',
      characters: visibleCharacters,
      event: currentNode?.reply?.trim() || currentNode?.user_text?.trim() || t('这个场景还没有发生事件'),
      setting: world.setting?.trim() || t('这个世界还没有背景说明。'),
      mainline: questState.mainline?.trim() || world.mainline?.trim()
        || mainQuest?.title || t('主线尚未建立'),
      mainlinePhase: questState.mainline_phase?.trim() || mainQuest?.phase
        || world.mainline_phase?.trim() || t('尚未开始'),
      mainlineMilestones: mainQuest?.milestones?.length
        ? mainQuest.milestones
        : (questState.mainline_milestones?.length
          ? questState.mainline_milestones
          : (world.mainline_milestones ?? [])),
      objective: mainQuest?.progress || questState.objective || t('等待故事继续推进'),
      mainQuest,
      sideQuests,
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps -- graph.current is refreshed together with activeCurrent
  }, [world, activeCurrent, graph.current, questState])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const draw = () => {
      const width = canvas.clientWidth
      const height = canvas.clientHeight
      if (!width || !height) return
      const ratio = window.devicePixelRatio || 1
      const pixelWidth = Math.max(1, Math.round(width * ratio))
      const pixelHeight = Math.max(1, Math.round(height * ratio))
      if (canvas.width !== pixelWidth) canvas.width = pixelWidth
      if (canvas.height !== pixelHeight) canvas.height = pixelHeight
      const ctx = canvas.getContext('2d')
      if (!ctx) return

      const scale = Math.min(width / CANVAS_W, height / CANVAS_H)
      const offsetX = (width - CANVAS_W * scale) / 2
      const offsetY = (height - CANVAS_H * scale) / 2
      transformRef.current = { scale, offsetX, offsetY }
      ctx.setTransform(1, 0, 0, 1, 0, 0)
      ctx.clearRect(0, 0, canvas.width, canvas.height)
      ctx.setTransform(ratio * scale, 0, 0, ratio * scale, ratio * offsetX, ratio * offsetY)

      const css = getComputedStyle(canvas)
      const color = (name: string, fallback: string) => css.getPropertyValue(name).trim() || fallback
      const faint = color('--faint', '#6f7580')
      const dim = color('--dim', '#9aa0aa')
      const fg = color('--fg', '#f4f5f7')
      const accent = color('--c-map', '#79c4bc')
      const questColor = color('--c-quest', '#e2bd68')
      const raised = color('--bg-2', '#2d3946')
      const base = color('--bg-0', '#111820')
      const line = color('--w-09', 'rgba(255,255,255,.09)')
      const route = color('--w-22', 'rgba(255,255,255,.22)')
      const font = getComputedStyle(document.body).fontFamily || 'sans-serif'

      ctx.lineWidth = 1
      ctx.strokeStyle = line
      ctx.beginPath()
      ctx.moveTo(0, 44); ctx.lineTo(300, 44)
      ctx.moveTo(0, 110); ctx.lineTo(300, 110)
      ctx.moveTo(0, 176); ctx.lineTo(300, 176)
      ctx.moveTo(50, 0); ctx.lineTo(50, 220)
      ctx.moveTo(150, 0); ctx.lineTo(150, 220)
      ctx.moveTo(250, 0); ctx.lineTo(250, 220)
      ctx.stroke()

      ctx.strokeStyle = route
      ctx.lineWidth = 1.35
      ctx.setLineDash([2, 5])
      for (const edge of graph.edges) {
        const dx = edge.to.x - edge.from.x
        const dy = edge.to.y - edge.from.y
        const length = Math.hypot(dx, dy) || 1
        const sx = edge.from.x + (dx / length) * 17
        const sy = edge.from.y + (dy / length) * 17
        const ex = edge.to.x - (dx / length) * 17
        const ey = edge.to.y - (dy / length) * 17
        const bend = Math.min(12, length * .08)
        ctx.beginPath()
        ctx.moveTo(sx, sy)
        ctx.quadraticCurveTo((sx + ex) / 2 + bend, (sy + ey) / 2 - bend, ex, ey)
        ctx.stroke()
      }
      ctx.setLineDash([])

      for (const node of graph.nodes) {
        const live = node.name === graph.current
        const selected = node.name === selectedRoot
        const hover = node.name === hovered
        if (selected || hover) {
          ctx.beginPath()
          ctx.arc(node.x, node.y - 2, hover ? 18 : 17, 0, Math.PI * 2)
          ctx.strokeStyle = hover ? fg : accent
          ctx.lineWidth = hover ? 1.5 : 1
          ctx.stroke()
        }
        markerPath(ctx, node.x, node.y)
        ctx.save()
        ctx.shadowColor = hover ? 'rgba(0,0,0,.42)' : 'rgba(0,0,0,.28)'
        ctx.shadowBlur = hover ? 8 : 5
        ctx.shadowOffsetY = 2
        ctx.fillStyle = hover ? fg : live ? accent : raised
        ctx.fill()
        ctx.restore()
        markerPath(ctx, node.x, node.y)
        ctx.strokeStyle = hover ? fg : live || selected ? accent : route
        ctx.lineWidth = hover ? 1.4 : 1.15
        ctx.stroke()
        drawSceneGlyph(ctx, node.name, node.x, node.y, hover || live ? base : selected ? accent : dim)

        ctx.textAlign = 'center'
        ctx.textBaseline = 'alphabetic'
        ctx.font = `500 9px ${font}`
        ctx.fillStyle = live || selected ? fg : dim
        ctx.fillText(shortName(node.name), node.x, node.y + 29)
        if (node.children > 0) {
          const badge = `+${node.children}`
          ctx.font = `600 7px ${font}`
          const badgeWidth = Math.max(14, ctx.measureText(badge).width + 7)
          roundedRect(ctx, node.x + 6, node.y - 18, badgeWidth, 11, 5.5)
          ctx.fillStyle = raised
          ctx.fill()
          ctx.strokeStyle = route
          ctx.lineWidth = .8
          ctx.stroke()
          ctx.fillStyle = live ? accent : faint
          ctx.textBaseline = 'middle'
          ctx.fillText(badge, node.x + 6 + badgeWidth / 2, node.y - 12.5)
        }
        const activeQuests = node.quests.filter((quest) => quest.status === 'active')
        if (activeQuests.length > 0) {
          const hasMain = activeQuests.some((quest) => quest.type === 'main')
          const badge = hasMain
            ? (activeQuests.length > 1 ? `◆${activeQuests.length}` : '◆')
            : `${activeQuests.length}`
          ctx.font = `700 7px ${font}`
          const badgeWidth = Math.max(14, ctx.measureText(badge).width + 7)
          roundedRect(ctx, node.x - 6 - badgeWidth, node.y - 18, badgeWidth, 11, 5.5)
          ctx.fillStyle = 'rgba(40,32,20,.94)'
          ctx.fill()
          ctx.strokeStyle = questColor
          ctx.lineWidth = .9
          ctx.stroke()
          ctx.fillStyle = questColor
          ctx.textBaseline = 'middle'
          ctx.fillText(badge, node.x - 6 - badgeWidth / 2, node.y - 12.5)
        }
      }
    }

    const observer = new ResizeObserver(draw)
    observer.observe(canvas)
    draw()
    return () => observer.disconnect()
  }, [graph, selectedRoot, hovered, theme])

  const nodeAt = (clientX: number, clientY: number): MiniNode | undefined => {
    const canvas = canvasRef.current
    if (!canvas) return undefined
    const rect = canvas.getBoundingClientRect()
    const { scale, offsetX, offsetY } = transformRef.current
    const x = (clientX - rect.left - offsetX) / scale
    const y = (clientY - rect.top - offsetY) / scale
    return graph.nodes.find((node) => Math.hypot(node.x - x, node.y - y) <= 19)
  }

  const openQuest = (quest: Quest) => {
    if (!world || !quest.place || !world.places[quest.place]) {
      onOpen()
      return
    }
    const root = mainSceneFor(world, quest.place)
    const target = graph.nodes.find((node) => node.name === root)
    if (target?.latest) onSelect(target.latest)
    else onOpen()
  }

  const statusLabel = (quest: Quest) => {
    if (quest.status === 'completed') return t('已完成')
    if (quest.status === 'failed') return t('已失败')
    return t('进行中')
  }

  return (
    <section className="world-map-mini panel" aria-label={t('迷你世界地图')}>
      <span className="world-map-mini-head">
        <span>{t('世界地图')}</span>
        <em>{t('{n} 个主场景', { n: graph.nodes.length })}</em>
      </span>

      <span className="world-map-mini-canvas">
        <canvas
          ref={canvasRef}
          className="world-map-mini-canvas-element"
          role="button"
          tabIndex={0}
          aria-label={selectedRoot
            ? t('当前主场景：{name}；点击场景节点回看最新事件；点击空白区域打开完整地图', { name: selectedRoot })
            : t('点击场景节点回看最新事件；点击空白区域打开完整地图')}
          title={hovered ? t('回看 {name} 的最近事件', { name: hovered }) : t('点击空白区域打开完整地图')}
          style={{ cursor: hovered ? 'pointer' : 'zoom-in' }}
          onPointerMove={(e) => setHovered(nodeAt(e.clientX, e.clientY)?.name ?? '')}
          onPointerLeave={() => setHovered('')}
          onClick={(e) => {
            const node = nodeAt(e.clientX, e.clientY)
            if (node?.latest) onSelect(node.latest)
            else onOpen()
          }}
          onKeyDown={(e) => {
            if (e.key === 'Enter' || e.key === ' ') {
              e.preventDefault()
              onOpen()
            }
          }}
        />
        {!graph.nodes.length && <span className="world-map-mini-empty">{t('解锁场景后生成地图')}</span>}
      </span>

      {world && (
        <div className="world-map-mini-context">
          <section className="world-story-overview" aria-label={t('世界故事概览')}>
            <div className="world-story-block">
              <small>{t('世界背景')}</small>
              <p>{context.setting}</p>
            </div>
            <div className="world-story-block">
              <small>{t('主线任务')}</small>
              {context.mainQuest ? (
                <button
                  type="button"
                  className={`world-quest-card main ${context.mainQuest.status}`}
                  onClick={() => openQuest(context.mainQuest!)}
                  title={context.mainline}
                >
                  <span className="world-quest-card-head">
                    <strong>{context.mainline}</strong>
                    <em>{statusLabel(context.mainQuest)}</em>
                  </span>
                  {context.mainlinePhase && <span className="world-quest-phase">{t('阶段 · {phase}', { phase: context.mainlinePhase })}</span>}
                  {context.mainQuest.place && <i>◆ {context.mainQuest.place}</i>}
                </button>
              ) : <p>{context.mainline}</p>}
            </div>
            <div className="world-story-block current-progress">
              <small>{t('当前进展')}</small>
              <p>{context.objective}</p>
            </div>
          </section>

          <section className="world-quest-section" aria-label={t('支线任务')}>
            <small>{t('支线任务')}</small>
            <div className="world-quest-list">
              {context.sideQuests.length ? context.sideQuests.map((quest) => (
                <button
                  type="button"
                  className={`world-quest-card side ${quest.status}`}
                  onClick={() => openQuest(quest)}
                  key={quest.id}
                >
                  <span className="world-quest-card-head">
                    <strong>{quest.title}</strong>
                    <em>{statusLabel(quest)}</em>
                  </span>
                  <span>{quest.progress || quest.summary || t('等待推进')}</span>
                  {quest.place && <i>◇ {quest.place}</i>}
                </button>
              )) : <span className="world-map-mini-context-empty">{t('暂无支线任务')}</span>}
            </div>
          </section>

          {graph.nodes.length > 0 && <>
          <div className="world-map-mini-scene">
            <span className="world-map-mini-scene-thumb">
              {context.sceneRef ? (
                <img src={mediaUrl('refs', context.sceneRef)} alt={t('{name}参考图', { name: context.root })} />
              ) : (
                <span className="world-map-mini-scene-empty">{t('暂无参考图')}</span>
              )}
            </span>
            <span className="world-map-mini-scene-copy">
              <small>{t('当前主场景')}</small>
              <strong>{context.root || t('未命名场景')}</strong>
              {context.currentScene && context.currentScene !== context.root && (
                <span>{t('镜头位于 · {place}', { place: context.currentScene })}</span>
              )}
            </span>
          </div>

          <div className="world-map-mini-people">
            <small>{t('主要人物')}</small>
            <span className="world-map-mini-people-list">
              {context.characters.length ? context.characters.map((character) => (
                <span className="world-map-mini-person" key={character.id} title={character.name}>
                  <img src={mediaUrl('refs', character.ref)} alt="" />
                  <span>{character.name}</span>
                </span>
              )) : <span className="world-map-mini-context-empty">{t('暂无登场人物')}</span>}
            </span>
          </div>

          <div className="world-map-mini-event">
            <small>{t('最近事件')}</small>
            <span title={context.event}>{context.event}</span>
          </div>
          </>}
        </div>
      )}

      <button type="button" className="world-map-mini-foot" onClick={onOpen} disabled={!world}>
        <span>{context.currentScene
          ? `${(activeCurrent?.id ?? world?.head) === world?.head ? t('镜头') : t('回看')} · ${context.currentScene}`
          : t('地图尚未生成')}</span>
        <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 5l7 7-7 7" /></svg>
      </button>
    </section>
  )
}
