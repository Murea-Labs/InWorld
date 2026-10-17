/**
 * UI language. The server's `agent.language` decides it (see /api/health), so
 * the interface and the story always speak the same language.
 *
 * Strings are written in Chinese in the source and looked up in `en.ts`; a
 * missing entry falls back to the Chinese text. `{name}` placeholders are
 * filled from `vars`.
 */
import { EN } from './en'

export type Language = 'en' | 'zh'

let current: Language = 'en'

export function setLanguage(lang: string | undefined): void {
  current = lang === 'zh' ? 'zh' : 'en'
  document.documentElement.lang = current === 'zh' ? 'zh-CN' : 'en'
}

export function language(): Language {
  return current
}

export function t(zh: string, vars?: Record<string, string | number>): string {
  const text = current === 'zh' ? zh : (EN[zh] ?? zh)
  return vars ? text.replace(/\{(\w+)\}/g, (m, k) => (k in vars ? String(vars[k]) : m)) : text
}

/** Display label for an asset kind stored in world data (stored values stay Chinese). */
export function kindLabel(kind: string | undefined): string {
  return t((kind || '人物').trim())
}
