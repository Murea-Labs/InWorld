import { t } from './i18n'
export type MureaTheme = 'studio' | 'daylight'

export interface ThemeOption {
  id: MureaTheme
  label: string
  description: string
  swatch: string
}

export const MUREA_THEMES: ThemeOption[] = [
  { id: 'daylight', label: t('日光'), description: t('纯白'), swatch: '#ffffff' },
  { id: 'studio', label: t('暗场'), description: t('纯黑，画面优先'), swatch: '#000000' },
]

const THEME_STORAGE_KEY = 'murea-mind:theme'

export function loadTheme(): MureaTheme {
  let saved: string | null = null
  try {
    saved = window.localStorage.getItem(THEME_STORAGE_KEY)
    if (MUREA_THEMES.some((theme) => theme.id === saved)) return saved as MureaTheme
  } catch {
    // Private mode may block localStorage; the theme still works for this session.
  }
  // White is the default; only the old pure-black theme maps to studio.
  return saved === 'noir' ? 'studio' : 'daylight'
}

export function saveTheme(theme: MureaTheme) {
  try {
    window.localStorage.setItem(THEME_STORAGE_KEY, theme)
  } catch {
    // Failing to save a preference must not break the world itself.
  }
}
