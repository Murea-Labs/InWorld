import type { Node, Place, World } from './types'
import { t } from './i18n'

export const SECTIONS = [
  'subject_definitions',
  'summary',
  'retention_analysis',
  'detailed_description',
  'overall_soundscape',
  'non_diegetic_music',
] as const

export const SECTION_CN: Record<string, string> = {
  subject_definitions: t('主体定义'),
  summary: t('摘要'),
  retention_analysis: t('保留分析'),
  detailed_description: t('详细描述（主字段）'),
  overall_soundscape: t('画内音'),
  non_diegetic_music: t('画外音乐'),
}

/**
 * Which fields this clip actually used. The two video routes use different prompt
 * structures, so the inspector cannot assume the six-section format.
 */
export function sectionsOf(six: Record<string, string>): string[] {
  const known = SECTIONS.filter((k) => k in six)
  return known.length ? known : Object.keys(six)
}

export const STAGE_CN: Record<string, string> = {
  queue: t('排队中，一次只能跑一个请求'),
  gm: t('GM 推演中…'),
  image: t('生成场景参考图…'),
  rewrite: t('写六段式 prompt…'),
  generate: t('生成中…'),
}

/* The three steps of a turn the user can see. Drawing images and writing the prompt run
   in parallel, so they share one step; otherwise the progress would jump back and forth. */
/**
 * The first beat the system runs after a world is created. It is in history (the director
 * needs the context) but must not look like something the user typed; render it separately.
 */
export const OPENING = t('开场：用一个定场镜头把这个世界立起来。')

export const STEPS = ['gm', 'rewrite', 'generate'] as const
export const STEP_CN: Record<string, string> = {
  gm: t('推演'), rewrite: t('写 prompt'), generate: t('生成画面'),
}
/** Which step a progress event's stage belongs to. */
export const STEP_OF: Record<string, (typeof STEPS)[number]> = {
  queue: 'gm', gm: 'gm', image: 'rewrite', rewrite: 'rewrite', generate: 'generate',
}

export const STAGE_SHORT: Record<string, string> = {
  gm: 'GM', image: t('生图'), rewrite: t('改写'), video: t('生成'),
  gm_density_audit: t('动作审计'), gm_plan_repair: t('动作重规划'),
  gm_physics_repair: t('物理字段校正'), gm_prefetch_wait: t('预推演等待'),
}

/** In merged mode the director's single call also writes the six-section prompt; say so in the label. */
export const STAGE_SHORT_MERGED: Record<string, string> = {
  ...STAGE_SHORT, gm: t('主模型（含视频描述）'),
}

/* Values are defined in styles/tokens.css; only reference them here, colors have one source. */
export const STAGE_COLOR: Record<string, string> = {
  gm: 'var(--c-gm)', image: 'var(--c-image)',
  rewrite: 'var(--c-rewrite)', video: 'var(--c-video)',
}

/** Root -> head. */
export function lineage(w: World | null): Node[] {
  if (!w?.head) return []
  const out: Node[] = []
  let cur: string | null = w.head
  while (cur && w.nodes[cur]) {
    out.push(w.nodes[cur])
    cur = w.nodes[cur].parent
  }
  return out.reverse()
}

/** Places authored for a later beat stay available to the Agent without spoiling
    the map. A generated node is the durable evidence that the player reached it. */
export function visiblePlaces(w: World | null): [string, Place][] {
  if (!w) return []
  const visited = new Set(Object.values(w.nodes).map((node) => node.place))
  if (w.here) visited.add(w.here)
  return Object.entries(w.places).filter(([name, place]) => (
    !place.hidden_until_visited || visited.has(name)
  ))
}

/** Every take of the same beat. */
export function takesOf(w: World, n: Node): Node[] {
  return Object.values(w.nodes)
    .filter((x) => x.parent === n.parent)
    .sort((a, b) => a.created.localeCompare(b.created))
}

/** What actually happened, not what was intended. */
export function badge(n: Node): { text: string; cls: string } {
  if (n.continued_from) return { text: t('续写'), cls: 'cont' }
  return n.mode === 'continue' ? { text: t('同场景'), cls: 'same' } : { text: t('切场景'), cls: 'cut' }
}

export const assemble = (six: Record<string, string>) => {
  return SECTIONS.map((k) => (six[k] || 'N/A').trim()).join('\n\n')
}
