import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { setLanguage } from './i18n'
import './styles/index.css'

// The UI language comes from the server config. Read it before loading the app,
// so module-level labels are created in the right language.
async function start() {
  try {
    const r = await fetch('api/health', { cache: 'no-store' })
    setLanguage((await r.json()).language)
  } catch {
    setLanguage('en')
  }
  const { default: App } = await import('./App')
  createRoot(document.getElementById('root')!).render(
    <StrictMode>
      <App />
    </StrictMode>,
  )
}

void start()
