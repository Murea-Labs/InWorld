/**
 * Narration voice.
 * 
 * The director's line arrives in about a second while the video takes much longer; reading it
 * aloud turns the wait from watching a progress bar into someone telling you a story.
 * 
 * The voice comes from a self-hosted OmniVoice server (proxied by the backend /api/tts), not
 * the browser's speechSynthesis, which sounds too robotic.
 * 
 * ⚠️ This is the **narrator**: one fixed voice from the TTS server's voice.wav, unrelated to
 * who is in the world. Characters speaking in the video is different: each character has its
 * own wav sent to the video model (see murea_agent/memory/voices.py). So no speaker is passed here.
 */
import { BASE } from './api'

const KEY = 'murea-mind:tts'

export function loadTts(): boolean {
  try {
    return window.localStorage.getItem(KEY) === '1'
  } catch {
    return false          // unreadable in private mode: off by default
  }
}

export function saveTts(on: boolean) {
  try {
    window.localStorage.setItem(KEY, on ? '1' : '0')
  } catch {
    // A failed save does not affect this session
  }
}

interface Narration {
  controller: AbortController
  audio: HTMLAudioElement | null
  started: boolean
  done: Promise<void>
  finish: () => void
}
let current: Narration | null = null

/** Explicit user actions cancel both synthesis and playback. */
export function stopSpeaking() {
  const active = current
  if (!active) return
  current = null
  active.controller.abort()
  active.audio?.pause()
  active.finish()
}

/** Automatic video playback lets an audible sentence finish. If synthesis has
 * not started playback yet, skip it so a fast video is not delayed by TTS. */
export function finishSpeaking(): Promise<void> | undefined {
  if (current?.started) return current.done
  stopSpeaking()
  return undefined
}

export async function speak(text: string, onState?: (speaking: boolean) => void) {
  stopSpeaking()
  const t = text.trim()
  if (!t) return
  let resolveDone!: () => void
  const done = new Promise<void>((resolve) => { resolveDone = resolve })
  let finished = false
  const active: Narration = {
    controller: new AbortController(), audio: null, started: false, done,
    finish: () => {
      if (finished) return
      finished = true
      onState?.(false)
      resolveDone()
    },
  }
  current = active
  let url: string | undefined
  try {
    // OmniVoice synthesizes the whole utterance with its punctuation and
    // prosody intact. Twelve-character serial requests cut words in half and
    // inserted a synthesis/network wait after every few spoken words.
    const r = await fetch(`${BASE.replace(/\/$/, '')}/api/tts`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: t }),
      signal: active.controller.signal,
    })
    if (current !== active) return
    if (!r.ok) throw new Error(`TTS ${r.status}`)
    const blob = await r.blob()
    if (current !== active) return
    url = URL.createObjectURL(blob)
    const a = new Audio(url)
    active.audio = a
    // Expose the actual media lifecycle for browser diagnostics, without
    // showing extra playback controls in the world UI.
    a.hidden = true
    a.dataset.narration = 'true'
    a.preload = 'auto'
    document.body.appendChild(a)
    const playback = new Promise<void>((resolve, reject) => {
      a.onplaying = () => {
        if (current !== active) return
        active.started = true
        onState?.(true)
      }
      a.onended = () => resolve()
      a.onerror = () => reject(new Error('audio-playback-failed'))
      a.onpause = () => {
        if (active.controller.signal.aborted) resolve()
      }
      a.play().catch((error: Error) => reject(new Error(
        error.name === 'NotAllowedError' ? 'autoplay-blocked' : 'audio-playback-failed',
      )))
    })
    // Cancellation also resolves when a pending play() has not emitted events.
    await Promise.race([playback, done])
  } catch (error) {
    if (current === active && !active.controller.signal.aborted) throw error
  } finally {
    if (current === active) current = null
    active.audio?.pause()
    active.audio?.remove()
    if (url) URL.revokeObjectURL(url)
    active.finish()
  }
}
