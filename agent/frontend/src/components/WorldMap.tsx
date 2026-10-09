import {
  useEffect, useMemo, useRef, useState,
  type CSSProperties, type PointerEvent as ReactPointerEvent,
} from 'react'
import { mediaUrl, updatePlaceMainScene } from '../api'
import { lineage, visiblePlaces } from '../lib'
import type { Node, Place, World } from '../types'
import SceneGlyph from './SceneGlyph'
import { t } from '../i18n'

interface Props {
  open: boolean
  world: World | null
  /** The clip shown in the player; the map dialog follows it when opened or when reviewing a clip. */
  current?: Node | null
  onClose: () => void
  onPlay: (node: Node) => void
  onWorldUpdated: () => void | Promise<void>
}

interface Point {
  x: number
  y: number
}

interface SceneGroup {
  name: string
  sceneNames: string[]
}

interface BranchLine {
  key: string
  root: string
  child: string
  from: Point
  to: Point
}

interface GroupLayout {
  points: Map<string, Point>
  groups: SceneGroup[]
  branches: BranchLine[]
  width: number
  height: number
}

interface Connection {
  key: string
  a: string
  b: string
  aToB: number
  bToA: number
}

interface ConnectionCurve {
  d: string
  midpoint: Point
}

interface StoredLayout {
  worldId: string
  points: Record<string, Point>
}

interface NodeDrag {
  name: string
  pointerId: number
  startX: number
  startY: number
  origin: Point
  followers: Record<string, Point>
  moved: boolean
}

interface RenderedConnection extends Connection {
  kind: 'main' | 'internal'
  visible: boolean
}

type PositionedStyle = CSSProperties & { left: number; top: number }
type CanvasStyle = CSSProperties & {
  '--map-canvas-width': string
  '--map-canvas-height': string
}

const SIDE_PADDING = 34
const ITEM_GAP = 40
const ROW_GAP = 42
const MAIN_NODE_WIDTH = 300
const MAIN_NODE_HEIGHT = 190
const EXPANDED_WIDTH = 640
const BRANCH_ROW_HEIGHT = 116
const DRAG_THRESHOLD = 4
const EMPTY_POSITIONS: Record<string, Point> = {}

const layoutStorageKey = (worldId: string) => `murea-agent:map-layout:v1:${worldId}`

function constrainPoint(point: Point, width: number, height: number): Point {
  const margin = 58
  return {
    x: Math.min(Math.max(margin, point.x), Math.max(margin, width - margin)),
    y: Math.min(Math.max(margin, point.y), Math.max(margin, height - margin)),
  }
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

function buildGroupLayout(
  world: World,
  viewportWidth: number,
  viewportHeight: number,
  expandedMainScene: string,
): GroupLayout {
  const grouped = new Map<string, string[]>()
  for (const name of Object.keys(world.places)) {
    const root = mainSceneFor(world, name)
    const scenes = grouped.get(root) ?? []
    scenes.push(name)
    grouped.set(root, scenes)
  }

  const drafts = [...grouped.entries()].map(([name, sceneNames]) => {
    const ordered = [name, ...sceneNames.filter((scene) => scene !== name)]
    const children = ordered.slice(1)
    const expanded = expandedMainScene === name && children.length > 0
    return {
      name,
      sceneNames: ordered,
      children,
      expanded,
      width: expanded ? EXPANDED_WIDTH : MAIN_NODE_WIDTH,
      height: expanded
        ? Math.max(280, children.length * BRANCH_ROW_HEIGHT + 72)
        : MAIN_NODE_HEIGHT,
    }
  })

  const baseWidth = viewportWidth || 920
  const availableWidth = Math.max(720, baseWidth - SIDE_PADDING * 2)
  const rows: { items: typeof drafts; width: number; height: number }[] = []
  let pendingRow: typeof drafts = []
  let pendingWidth = 0
  let pendingHeight = 0
  for (const draft of drafts) {
    const nextWidth = pendingWidth + (pendingRow.length ? ITEM_GAP : 0) + draft.width
    if (pendingRow.length && nextWidth > availableWidth) {
      rows.push({ items: pendingRow, width: pendingWidth, height: pendingHeight })
      pendingRow = []
      pendingWidth = 0
      pendingHeight = 0
    }
    pendingWidth += (pendingRow.length ? ITEM_GAP : 0) + draft.width
    pendingHeight = Math.max(pendingHeight, draft.height)
    pendingRow.push(draft)
  }
  if (pendingRow.length) rows.push({ items: pendingRow, width: pendingWidth, height: pendingHeight })

  const contentWidth = Math.max(availableWidth, ...rows.map((row) => row.width), 0)
  const width = Math.max(baseWidth, contentWidth + SIDE_PADDING * 2)
  const points = new Map<string, Point>()
  const groups: SceneGroup[] = drafts.map(({ name, sceneNames }) => ({ name, sceneNames }))
  const branches: BranchLine[] = []
  let cursorY = 64
  for (const row of rows) {
    let cursorX = (width - row.width) / 2
    for (const item of row.items) {
      const itemY = cursorY + (row.height - item.height) / 2
      const rootPoint = item.expanded
        ? { x: cursorX + 86, y: itemY + item.height / 2 }
        : { x: cursorX + 62, y: itemY + item.height / 2 }
      points.set(item.name, rootPoint)
      item.children.forEach((name, index) => {
        const childPoint = item.expanded
          ? {
            x: cursorX + 370,
            y: itemY + ((index + 1) * item.height) / (item.children.length + 1),
          }
          : rootPoint
        points.set(name, childPoint)
        branches.push({
          key: `${item.name}\u0000${name}`,
          root: item.name,
          child: name,
          from: rootPoint,
          to: childPoint,
        })
      })
      cursorX += item.width + ITEM_GAP
    }
    cursorY += row.height + ROW_GAP
  }
  const contentHeight = rows.length ? cursorY - ROW_GAP + 64 : 320
  const height = Math.max(viewportHeight || 620, contentHeight)

  return { points, groups, branches, width, height }
}

function insetLineEnd(from: Point, to: Point, distance = 31): Point {
  const dx = to.x - from.x
  const dy = to.y - from.y
  const length = Math.hypot(dx, dy) || 1
  return { x: from.x + (dx / length) * distance, y: from.y + (dy / length) * distance }
}

function connectionCurve(from: Point, to: Point, index: number): ConnectionCurve {
  const start = insetLineEnd(from, to)
  const end = insetLineEnd(to, from)
  const dx = end.x - start.x
  const dy = end.y - start.y
  const bend = (index % 2 === 0 ? -1 : 1) * Math.min(58, Math.max(22, Math.abs(dx + dy) * .08))
  const control1 = Math.abs(dx) >= Math.abs(dy)
    ? { x: start.x + dx * .36, y: start.y + bend }
    : { x: start.x + bend, y: start.y + dy * .36 }
  const control2 = Math.abs(dx) >= Math.abs(dy)
    ? { x: end.x - dx * .36, y: end.y + bend }
    : { x: end.x + bend, y: end.y - dy * .36 }
  return {
    d: `M${start.x} ${start.y} C${control1.x} ${control1.y} ${control2.x} ${control2.y} ${end.x} ${end.y}`,
    midpoint: {
      x: (start.x + 3 * control1.x + 3 * control2.x + end.x) / 8,
      y: (start.y + 3 * control1.y + 3 * control2.y + end.y) / 8,
    },
  }
}

function curveIndex(key: string): number {
  return [...key].reduce((sum, char) => sum + char.charCodeAt(0), 0)
}

function sceneConnections(world: World, chain: Node[]): Connection[] {
  const order = new Map(Object.keys(world.places).map((name, index) => [name, index]))
  const directed = new Map<string, { from: string; to: string; count: number }>()
  const add = (from: string, to: string, count = 1) => {
    if (from === to || !order.has(from) || !order.has(to)) return
    const key = `${from}\u0000${to}`
    const existing = directed.get(key)
    if (existing) existing.count += count
    else directed.set(key, { from, to, count })
  }

  for (let index = 1; index < chain.length; index += 1) {
    add(chain[index - 1].place, chain[index].place)
  }
  for (const link of world.place_links ?? []) {
    const forward = directed.has(`${link.from}\u0000${link.to}`)
    const backward = directed.has(`${link.to}\u0000${link.from}`)
    if (!forward && !backward) add(link.from, link.to, Math.max(1, Number(link.n) || 1))
  }

  const grouped = new Map<string, Connection>()
  for (const edge of directed.values()) {
    const fromIndex = order.get(edge.from) ?? 0
    const toIndex = order.get(edge.to) ?? 0
    const a = fromIndex <= toIndex ? edge.from : edge.to
    const b = fromIndex <= toIndex ? edge.to : edge.from
    const key = `${a}\u0000${b}`
    const connection = grouped.get(key) ?? { key, a, b, aToB: 0, bToA: 0 }
    if (edge.from === a) connection.aToB += edge.count
    else connection.bToA += edge.count
    grouped.set(key, connection)
  }
  return [...grouped.values()]
}

function collapseToMainScenes(
  world: World,
  connections: Connection[],
): { main: Connection[]; internal: Map<string, Connection[]> } {
  const rootOrder = new Map<string, number>()
  for (const name of Object.keys(world.places)) {
    const root = mainSceneFor(world, name)
    if (!rootOrder.has(root)) rootOrder.set(root, rootOrder.size)
  }
  const main = new Map<string, Connection>()
  const internal = new Map<string, Connection[]>()

  const addMain = (from: string, to: string, count: number) => {
    if (!count || from === to) return
    const fromIndex = rootOrder.get(from) ?? 0
    const toIndex = rootOrder.get(to) ?? 0
    const a = fromIndex <= toIndex ? from : to
    const b = fromIndex <= toIndex ? to : from
    const key = `${a}\u0000${b}`
    const edge = main.get(key) ?? { key, a, b, aToB: 0, bToA: 0 }
    if (from === a) edge.aToB += count
    else edge.bToA += count
    main.set(key, edge)
  }

  for (const connection of connections) {
    const rootA = mainSceneFor(world, connection.a)
    const rootB = mainSceneFor(world, connection.b)
    if (rootA === rootB) {
      const edges = internal.get(rootA) ?? []
      edges.push(connection)
      internal.set(rootA, edges)
      continue
    }
    addMain(rootA, rootB, connection.aToB)
    addMain(rootB, rootA, connection.bToA)
  }
  return { main: [...main.values()], internal }
}

function branchCurve(from: Point, to: Point): string {
  const start = insetLineEnd(from, to, 34)
  const end = insetLineEnd(to, from, 27)
  const middle = start.x + (end.x - start.x) * .46
  return `M${start.x} ${start.y} C${middle} ${start.y} ${middle} ${end.y} ${end.x} ${end.y}`
}

function formatMoment(value: string): string {
  return value ? value.replace('T', ' ').slice(0, 16) : ''
}

export default function WorldMap({
  open, world, current, onClose, onPlay, onWorldUpdated,
}: Props) {
  const [selectedName, setSelectedName] = useState('')
  const [expandedMainScene, setExpandedMainScene] = useState('')
  const [viewportSize, setViewportSize] = useState({ width: 0, height: 0 })
  const [groupBusy, setGroupBusy] = useState(false)
  const [groupNote, setGroupNote] = useState('')
  const [storedLayout, setStoredLayout] = useState<StoredLayout>({ worldId: '', points: {} })
  const [draggingName, setDraggingName] = useState('')
  const boardWrapRef = useRef<HTMLDivElement>(null)
  const dialogRef = useRef<HTMLDivElement>(null)
  const closeButtonRef = useRef<HTMLButtonElement>(null)
  const returnFocusRef = useRef<HTMLElement | null>(null)
  const dragRef = useRef<NodeDrag | null>(null)
  const suppressClickRef = useRef('')
  const positionsRef = useRef<Record<string, Point>>({})
  const displayedPointsRef = useRef<Map<string, Point>>(new Map())
  const places = useMemo(() => visiblePlaces(world), [world])
  const mapWorld = useMemo(() => {
    if (!world) return null
    const names = new Set(places.map(([name]) => name))
    return {
      ...world,
      places: Object.fromEntries(places),
      place_links: (world.place_links ?? []).filter(
        (link) => names.has(link.from) && names.has(link.to),
      ),
    }
  }, [world, places])
  const chain = useMemo(() => lineage(world), [world])
  const placeKey = places.map(([name]) => name).join('\u0000')

  const eventsByPlace = useMemo(() => {
    const grouped = new Map<string, Node[]>()
    for (const node of chain) {
      const events = grouped.get(node.place) ?? []
      events.push(node)
      grouped.set(node.place, events)
    }
    return grouped
  }, [chain])

  useEffect(() => {
    if (!open || !world) return
    const points: Record<string, Point> = {}
    try {
      const raw = window.localStorage.getItem(layoutStorageKey(world.id))
      const parsed: unknown = raw ? JSON.parse(raw) : null
      if (parsed && typeof parsed === 'object') {
        for (const [name, value] of Object.entries(parsed)) {
          if (!mapWorld?.places[name] || !value || typeof value !== 'object') continue
          const candidate = value as Partial<Point>
          if (Number.isFinite(candidate.x) && Number.isFinite(candidate.y)) {
            points[name] = { x: Number(candidate.x), y: Number(candidate.y) }
          }
        }
      }
    } catch {
      // A broken local layout must not block the map; fall back to automatic layout.
    }
    positionsRef.current = points
    setStoredLayout({ worldId: world.id, points })
    dragRef.current = null
    setDraggingName('')
  // eslint-disable-next-line react-hooks/exhaustive-deps -- rebuild the layout per world and place set, not on every world update
  }, [open, world?.id, mapWorld, placeKey])

  useEffect(() => {
    if (!open) return
    const element = boardWrapRef.current
    if (!element) return
    const update = () => setViewportSize((current) => {
      const next = { width: element.clientWidth, height: element.clientHeight }
      return current.width === next.width && current.height === next.height ? current : next
    })
    update()
    const frame = window.requestAnimationFrame(update)
    const observer = typeof ResizeObserver === 'undefined' ? null : new ResizeObserver(update)
    observer?.observe(element)
    window.addEventListener('resize', update)
    return () => {
      window.cancelAnimationFrame(frame)
      observer?.disconnect()
      window.removeEventListener('resize', update)
    }
  }, [open])

  const layout = useMemo(
    () => mapWorld ? buildGroupLayout(
      mapWorld, viewportSize.width, viewportSize.height, expandedMainScene,
    ) : {
      points: new Map<string, Point>(), groups: [], branches: [], width: 920, height: 620,
    },
    [mapWorld, viewportSize.width, viewportSize.height, expandedMainScene],
  )
  const customPositions = world && storedLayout.worldId === world.id
    ? storedLayout.points
    : EMPTY_POSITIONS
  const displayedPoints = useMemo(() => {
    const points = new Map(layout.points)
    for (const group of layout.groups) {
      const automaticRoot = layout.points.get(group.name)
      if (!automaticRoot) continue
      const storedRoot = customPositions[group.name]
      const root = storedRoot
        ? constrainPoint(storedRoot, layout.width, layout.height)
        : automaticRoot
      points.set(group.name, root)
      const dx = root.x - automaticRoot.x
      const dy = root.y - automaticRoot.y
      for (const name of group.sceneNames) {
        if (name === group.name) continue
        const automatic = layout.points.get(name)
        if (!automatic) continue
        const stored = customPositions[name]
        points.set(name, stored
          ? constrainPoint(stored, layout.width, layout.height)
          : constrainPoint({ x: automatic.x + dx, y: automatic.y + dy }, layout.width, layout.height))
      }
    }
    return points
  }, [layout, customPositions])
  const displayedBranches = useMemo(() => layout.branches.map((branch) => ({
    ...branch,
    from: displayedPoints.get(branch.root) ?? branch.from,
    to: displayedPoints.get(branch.child) ?? branch.to,
  })), [layout.branches, displayedPoints])
  displayedPointsRef.current = displayedPoints
  const sceneLevelConnections = useMemo(
    () => mapWorld ? sceneConnections(mapWorld, chain) : [],
    [mapWorld, chain],
  )
  const connectionGraph = useMemo(
    () => mapWorld ? collapseToMainScenes(mapWorld, sceneLevelConnections) : {
      main: [], internal: new Map<string, Connection[]>(),
    },
    [mapWorld, sceneLevelConnections],
  )
  const compactLabels = viewportSize.width > 0 && viewportSize.width <= 430

  useEffect(() => {
    if (!open || !world) return
    // When reviewing a clip, the map details should point at that clip's place, not the world's
    // current one. A sub-place expands its parent place, otherwise the selected parent's branch
    // node stays hidden by `is-branch-hidden`; the parent itself keeps its collapsed state.
    const currentNode = current?.id ? world.nodes[current.id] : undefined
    const currentPlace = currentNode?.place && world.places[currentNode.place]
      ? currentNode.place
      : ''
    const place = currentPlace || (
      world.here && world.places[world.here]
        ? world.here
        : places[0]?.[0] ?? ''
    )
    const root = place && mapWorld ? mainSceneFor(mapWorld, place) : ''
    // The map selects at the level of main places: while a sub-place is being reviewed, keep
    // its root highlighted and reveal the sub-place by expanding it.
    setSelectedName(root)
    setExpandedMainScene(currentPlace && root !== place ? root : '')
  // eslint-disable-next-line react-hooks/exhaustive-deps -- reselect only when the world, place or current node changes
  }, [open, world?.id, world?.here, placeKey, current?.id, current?.place])

  useEffect(() => {
    setGroupNote('')
  }, [selectedName])

  useEffect(() => {
    if (!open || !selectedName) return
    const point = displayedPointsRef.current.get(selectedName)
    const element = boardWrapRef.current
    if (!point || !element) return
    const frame = window.requestAnimationFrame(() => {
      element.scrollTo({
        left: Math.max(0, point.x - element.clientWidth / 2),
        top: Math.max(0, point.y - element.clientHeight / 2),
        behavior: 'smooth',
      })
    })
    return () => window.cancelAnimationFrame(frame)
  }, [
    open, selectedName, expandedMainScene, layout.width, layout.height,
    storedLayout.worldId, placeKey,
  ])

  useEffect(() => {
    if (!open) return
    returnFocusRef.current = document.activeElement instanceof HTMLElement ? document.activeElement : null
    const frame = window.requestAnimationFrame(() => closeButtonRef.current?.focus())
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        event.preventDefault()
        onClose()
        return
      }
      if (event.key !== 'Tab' || !dialogRef.current) return
      const focusable = Array.from(dialogRef.current.querySelectorAll<HTMLElement>(
        'button:not([disabled]), a[href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])',
      )).filter((element) => element.offsetParent !== null)
      if (!focusable.length) return
      const first = focusable[0]
      const last = focusable[focusable.length - 1]
      const active = document.activeElement
      if (!dialogRef.current.contains(active)) {
        event.preventDefault()
        first.focus()
      } else if (event.shiftKey && active === first) {
        event.preventDefault()
        last.focus()
      } else if (!event.shiftKey && active === last) {
        event.preventDefault()
        first.focus()
      }
    }
    window.addEventListener('keydown', onKeyDown)
    return () => {
      window.cancelAnimationFrame(frame)
      window.removeEventListener('keydown', onKeyDown)
      returnFocusRef.current?.focus()
    }
  }, [open, onClose])

  if (!open || !world) return null

  const selected: Place | undefined = selectedName ? world.places[selectedName] : undefined
  const selectedEvents = eventsByPlace.get(selectedName) ?? []
  const selectedMainScene = selectedName && mapWorld ? mainSceneFor(mapWorld, selectedName) : ''
  const visited = new Set([...eventsByPlace.keys(), ...(world.here ? [world.here] : [])])
  const allMainScenes = layout.groups.map((group) => group.name)
  const groupedRoots = new Set(
    layout.groups.filter((group) => group.sceneNames.length > 1).map((group) => group.name),
  )
  const currentMainScene = world.here && mapWorld ? mainSceneFor(mapWorld, world.here) : ''
  const renderedConnections: RenderedConnection[] = [
    ...connectionGraph.main.map((connection) => ({
      ...connection, kind: 'main' as const, visible: true,
    })),
    ...[...connectionGraph.internal.entries()].flatMap(([mainScene, connections]) => (
      connections.map((connection) => ({
        ...connection,
        kind: 'internal' as const,
        visible: expandedMainScene === mainScene,
      }))
    )),
  ]
  const groupChoices = [selectedName, ...allMainScenes.filter((name) => name !== selectedName)]
  const hasCustomLayout = Object.keys(customPositions).length > 0
  const canvasStyle: CanvasStyle = {
    '--map-canvas-width': `${layout.width}px`,
    '--map-canvas-height': `${layout.height}px`,
  }

  const changeMainScene = async (mainScene: string) => {
    if (!selectedName || mainScene === selectedMainScene) return
    setGroupBusy(true)
    setGroupNote('')
    try {
      await updatePlaceMainScene(world.id, selectedName, mainScene)
      await onWorldUpdated()
    } catch (error) {
      setGroupNote((error as Error).message.slice(0, 140))
    } finally {
      setGroupBusy(false)
    }
  }

  const beginNodeDrag = (
    event: ReactPointerEvent<HTMLButtonElement>,
    name: string,
    isMain: boolean,
  ) => {
    if (event.button !== 0) return
    const origin = displayedPoints.get(name)
    if (!origin) return
    positionsRef.current = { ...customPositions }
    const followers: Record<string, Point> = {}
    if (isMain) {
      const group = layout.groups.find((item) => item.name === name)
      for (const child of group?.sceneNames ?? []) {
        if (child === name || !customPositions[child]) continue
        const point = displayedPoints.get(child)
        if (point) followers[child] = point
      }
    }
    dragRef.current = {
      name,
      pointerId: event.pointerId,
      startX: event.clientX,
      startY: event.clientY,
      origin,
      followers,
      moved: false,
    }
    event.currentTarget.setPointerCapture(event.pointerId)
  }

  const moveNode = (event: ReactPointerEvent<HTMLButtonElement>) => {
    const drag = dragRef.current
    if (!drag || drag.pointerId !== event.pointerId) return
    const dx = event.clientX - drag.startX
    const dy = event.clientY - drag.startY
    if (!drag.moved && Math.hypot(dx, dy) < DRAG_THRESHOLD) return
    event.preventDefault()
    if (!drag.moved) {
      drag.moved = true
      setDraggingName(drag.name)
    }
    const nextPoint = constrainPoint(
      { x: drag.origin.x + dx, y: drag.origin.y + dy },
      layout.width,
      layout.height,
    )
    const appliedX = nextPoint.x - drag.origin.x
    const appliedY = nextPoint.y - drag.origin.y
    const next = { ...positionsRef.current, [drag.name]: nextPoint }
    for (const [name, origin] of Object.entries(drag.followers)) {
      next[name] = constrainPoint(
        { x: origin.x + appliedX, y: origin.y + appliedY },
        layout.width,
        layout.height,
      )
    }
    positionsRef.current = next
    setStoredLayout({ worldId: world.id, points: next })
  }

  const finishNodeDrag = (
    event: ReactPointerEvent<HTMLButtonElement>,
    cancelled = false,
  ) => {
    const drag = dragRef.current
    if (!drag || drag.pointerId !== event.pointerId) return
    dragRef.current = null
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId)
    }
    if (drag.moved) {
      try {
        window.localStorage.setItem(
          layoutStorageKey(world.id), JSON.stringify(positionsRef.current),
        )
      } catch {
        // In private mode or when storage is full, keep the layout for this session.
      }
      if (!cancelled) {
        suppressClickRef.current = drag.name
        window.setTimeout(() => {
          if (suppressClickRef.current === drag.name) suppressClickRef.current = ''
        }, 0)
      }
    }
    setDraggingName('')
  }

  const resetNodeLayout = () => {
    positionsRef.current = {}
    setStoredLayout({ worldId: world.id, points: {} })
    try {
      window.localStorage.removeItem(layoutStorageKey(world.id))
    } catch {
      // Automatic layout is back; unavailable local storage does not affect the UI.
    }
  }

  return (
    <div
      className="world-map-overlay"
      ref={dialogRef}
      role="dialog"
      aria-modal="true"
      aria-labelledby="world-map-title"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onClose()
      }}
    >
      <div className="world-map-shell">
        <div className="world-map-topbar">
          <div className="world-map-heading">
            <span className="world-map-eyebrow">MUREA AGENT · SCENE MAP</span>
            <div>
              <h2 id="world-map-title">{world.title || t('未命名世界')}</h2>
              <span className="world-map-summary">
                {t('{a} 个主场景 · {b} 个场景 · {c} 条主路径', { a: allMainScenes.length, b: places.length, c: connectionGraph.main.length })}
              </span>
            </div>
          </div>
          <div className="world-map-legend" aria-label={t('地图图例')}>
            <span><i className="legend-current" />{t('镜头位置')}</span>
            <span><i className="legend-group" />{t('展开分支')}</span>
            <span><i className="legend-route" />{t('切换路径')}</span>
            <span><i className="legend-drag" />{t('拖动节点')}</span>
          </div>
          {hasCustomLayout && (
            <button
              type="button"
              className="world-map-reset-layout"
              onClick={resetNodeLayout}
              title={t('恢复地图的自动排版')}
            >
              <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 12a8 8 0 1 0 2.3-5.7L4 8.6M4 4v4.6h4.6" /></svg>
              <span>{t('重置布局')}</span>
            </button>
          )}
          <button ref={closeButtonRef} className="world-map-close" onClick={onClose} aria-label={t('关闭地图')}>
            <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 6l12 12M18 6L6 18" /></svg>
          </button>
        </div>

        <div className="world-map-content">
          <div className="world-map-board-wrap" ref={boardWrapRef}>
            <div className={'world-map-board' + (draggingName ? ' is-node-dragging' : '')} style={canvasStyle}>
              <div className="world-map-grid" />
              <div className="world-map-glow" />

              {(layout.branches.length > 0 || renderedConnections.length > 0) && (
                <svg className="world-map-routes" viewBox={`0 0 ${layout.width} ${layout.height}`} width={layout.width} height={layout.height} aria-hidden="true">
                  <defs>
                    <marker id="world-map-route-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
                      <path d="M0 0L8 4L0 8Z" />
                    </marker>
                  </defs>
                  {displayedBranches.map((branch) => (
                    <path
                      className={[
                        'world-map-branch-line',
                        expandedMainScene === branch.root ? '' : 'is-hidden',
                      ].filter(Boolean).join(' ')}
                      d={branchCurve(branch.from, branch.to)}
                      key={branch.key}
                      pathLength={1}
                    />
                  ))}
                  {renderedConnections.map((connection) => {
                    const from = displayedPoints.get(connection.a)
                    const to = displayedPoints.get(connection.b)
                    if (!from || !to) return null
                    const curve = connectionCurve(from, to, curveIndex(connection.key))
                    const total = connection.aToB + connection.bToA
                    const directionClass = connection.aToB && connection.bToA ? 'is-bidirectional' : connection.bToA ? 'is-reverse' : 'is-forward'
                    const direction = [
                      connection.aToB ? `${connection.a} → ${connection.b}${connection.aToB > 1 ? ` ×${connection.aToB}` : ''}` : '',
                      connection.bToA ? `${connection.b} → ${connection.a}${connection.bToA > 1 ? ` ×${connection.bToA}` : ''}` : '',
                    ].filter(Boolean).join('；')
                    const isCurrentRoute = connection.kind === 'main'
                      ? connection.a === currentMainScene || connection.b === currentMainScene
                      : connection.a === world.here || connection.b === world.here
                    return (
                      <g
                        key={`${connection.kind}:${connection.key}`}
                        className={['world-map-route', connection.visible ? '' : 'is-hidden'].filter(Boolean).join(' ')}
                      >
                        <path className={['world-map-route-shadow', connection.kind === 'internal' ? 'is-internal' : ''].filter(Boolean).join(' ')} d={curve.d} />
                        <path
                          className={['world-map-route-line', directionClass, connection.kind === 'internal' ? 'is-internal' : '', isCurrentRoute ? 'is-current' : ''].filter(Boolean).join(' ')}
                          d={curve.d}
                          markerStart={connection.bToA ? 'url(#world-map-route-arrow)' : undefined}
                          markerEnd={connection.aToB ? 'url(#world-map-route-arrow)' : undefined}
                        ><title>{direction}</title></path>
                        {total > 1 && (
                          <g className="world-map-route-count" transform={`translate(${curve.midpoint.x} ${curve.midpoint.y})`}>
                            <circle r="11" /><text textAnchor="middle" dominantBaseline="central">×{total}</text>
                          </g>
                        )}
                      </g>
                    )
                  })}
                </svg>
              )}

              {places.map(([name], index) => {
                const point = displayedPoints.get(name)
                if (!point) return null
                const events = eventsByPlace.get(name) ?? []
                const isCurrent = name === world.here
                const isVisited = visited.has(name)
                const mainScene = mapWorld ? mainSceneFor(mapWorld, name) : name
                const isMain = mainScene === name
                const hasChildren = isMain && groupedRoots.has(name)
                const isExpanded = expandedMainScene === name
                const isBranchNode = !isMain
                const isBranchHidden = isBranchNode && expandedMainScene !== mainScene
                const isCurrentGroup = isMain && currentMainScene === name
                const group = isMain ? layout.groups.find((item) => item.name === name) : undefined
                const mainVisited = isMain
                  ? group?.sceneNames.some((sceneName) => visited.has(sceneName)) ?? isVisited
                  : isVisited
                const style: PositionedStyle = { left: point.x, top: point.y }
                return (
                  <button
                    key={name}
                    className={['world-map-node', isCurrent ? 'is-current' : '', isCurrentGroup ? 'is-current-group' : '', mainVisited ? 'is-visited' : 'is-unlocked', isMain ? 'is-main-node' : '', isBranchNode ? 'is-branch-node' : '', isBranchHidden ? 'is-branch-hidden' : '', isExpanded ? 'is-expanded' : '', selectedName === name ? 'is-selected' : '', draggingName === name ? 'is-dragging' : ''].filter(Boolean).join(' ')}
                    style={style}
                    onClick={() => {
                      if (suppressClickRef.current === name) {
                        suppressClickRef.current = ''
                        return
                      }
                      setSelectedName(name)
                      if (isMain) {
                        setExpandedMainScene((current) => hasChildren && current !== name ? name : '')
                      }
                    }}
                    onPointerDown={(event) => beginNodeDrag(event, name, isMain)}
                    onPointerMove={moveNode}
                    onPointerUp={(event) => finishNodeDrag(event)}
                    onPointerCancel={(event) => finishNodeDrag(event, true)}
                    onLostPointerCapture={(event) => finishNodeDrag(event, true)}
                    aria-pressed={selectedName === name}
                    aria-expanded={hasChildren ? isExpanded : undefined}
                    aria-hidden={isBranchHidden || undefined}
                    aria-label={hasChildren ? t('查看主场景：{name}，{act}子场景；可拖动调整位置', { name, act: isExpanded ? t('收起') : t('展开') }) : t('查看场景：{name}；可拖动调整位置', { name })}
                    title={t('拖动调整节点位置，点击查看场景')}
                    tabIndex={isBranchHidden ? -1 : undefined}
                  >
                    <span className="world-map-node-pulse" />
                    <span className="world-map-node-pin"><SceneGlyph name={name} /></span>
                    <span className={['world-map-node-label', point.x > layout.width - 230 ? 'is-left' : '', compactLabels ? 'is-compact' : ''].filter(Boolean).join(' ')}>
                      <small>{isMain ? 'MAIN SCENE' : 'BRANCH'} · {String(index + 1).padStart(2, '0')}</small>
                      <strong>{name}</strong>
                      <em>{isCurrent
                        ? t('镜头位置')
                        : isCurrentGroup && world.here
                          ? t('镜头在 {place}', { place: world.here })
                          : hasChildren
                            ? t('{n} 个子场景 · {act}', { n: (group?.sceneNames.length ?? 1) - 1, act: isExpanded ? t('点击收起') : t('点击展开') })
                            : events.length ? t('{n} 个剧情事件', { n: events.length }) : t('场景已解锁')}</em>
                    </span>
                  </button>
                )
              })}

              {!places.length && (
                <div className="world-map-empty"><span>◎</span><b>{t('地图还是一片空白')}</b><p>{t('完成第一个场景后，它会自动成为这里的第一个节点。')}</p></div>
              )}
            </div>
            <div className="world-map-route-caption">
              <span className="is-story" />{t('跨场景路径连接主节点')} <em>{t('· 展开后显示组内路径 · 拖动节点自由布局')}</em>
            </div>
          </div>

          <div className="world-map-detail">
            {selected && (
              <>
                <div className="world-map-detail-image">
                  <img src={mediaUrl('refs', selected.ref)} alt={t('{name}场景参考图', { name: selectedName })} />
                  <div className="world-map-detail-shade" />
                  <span className="world-map-detail-status">
                    {selectedName === world.here
                      ? t('● 镜头位置')
                      : selectedName === selectedMainScene && currentMainScene === selectedName && world.here
                        ? t('● 镜头所在主场景 · {place}', { place: world.here })
                        : visited.has(selectedName) ? t('已到访') : t('已解锁')}
                  </span>
                  <div className="world-map-detail-title"><small>{t('主场景 · {name}', { name: selectedMainScene })}</small><h3>{selectedName}</h3></div>
                </div>

                <div className="world-map-detail-body">
                  <div className="world-map-detail-block world-map-group-editor">
                    <label>{t('所属主场景')}</label>
                    <select
                      value={selectedMainScene}
                      disabled={groupBusy}
                      onChange={(event) => void changeMainScene(event.target.value)}
                    >
                      {groupChoices.map((name) => (
                        <option key={name} value={name}>{name === selectedName ? t('{name}（独立主场景）', { name }) : name}</option>
                      ))}
                    </select>
                    <p>{t('同一主场景的子场景，会在点击主节点后以分支展开。')}</p>
                    {groupNote && <p className="world-map-group-note">{groupNote}</p>}
                  </div>

                  <div className="world-map-detail-block"><label>{t('场景档案')}</label><p>{selected.definition || selected.desc || t('这个场景还没有补充描述。')}</p></div>

                  <div className="world-map-detail-block world-map-event-block">
                    <div className="world-map-event-heading"><label>{t('当前剧情线发生过')}</label><span>{selectedEvents.length} EVENT{selectedEvents.length === 1 ? '' : 'S'}</span></div>
                    <div className="world-map-events">
                      {selectedEvents.map((node) => {
                        const storyIndex = chain.findIndex((item) => item.id === node.id) + 1
                        return (
                          <button className="world-map-event" key={node.id} onClick={() => onPlay(node)} title={t('在播放器中查看这一段')}>
                            <span className="world-map-event-dot" />
                            <span className="world-map-event-copy">
                              <small>{t('第 {n} 段', { n: storyIndex })} · {formatMoment(node.created)}</small><b>{node.reply || node.beat}</b>
                              {node.reply && node.beat && <em>{node.beat}</em>}
                            </span>
                            <span className="world-map-event-arrow">↗</span>
                          </button>
                        )
                      })}
                      {!selectedEvents.length && <div className="world-map-no-events">{t('这个场景已经解锁，但当前剧情线还没有在这里发生事件。')}</div>}
                    </div>
                  </div>
                </div>
              </>
            )}

            {!selected && places.length === 0 && (
              <div className="world-map-detail-placeholder"><span>MAP / 00</span><p>{t('场景详情会在解锁地点后出现在这里。')}</p></div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
