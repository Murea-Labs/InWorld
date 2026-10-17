// Run: cd frontend && npm test
import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import vm from 'node:vm'
import ts from '../frontend/node_modules/typescript/lib/typescript.js'

const source = readFileSync(new URL('../frontend/src/tts.ts', import.meta.url), 'utf8')
  .replace("import { BASE } from './api'", "const BASE = '/murea-mind/'")
const code = ts.transpileModule(source, { compilerOptions: {
  module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022,
} }).outputText
function setup({ delayed = false, blocked = false } = {}) {
  const requests = [], audios = [], revoked = [], states = []
  let respond
  class Audio {
    constructor(src) { this.src = src; this.dataset = {}; audios.push(this) }
    play() {
      if (blocked) return Promise.reject(Object.assign(new Error('blocked'), { name: 'NotAllowedError' }))
      this.onplaying?.()
      return Promise.resolve()
    }
    pause() { this.onpause?.() }
    remove() { this.removed = true }
  }
  const context = { exports: {}, AbortController, Audio,
    document: { body: { appendChild() {} } },
    URL: { createObjectURL: () => 'blob:test', revokeObjectURL: u => revoked.push(u) },
    fetch: (url, init) => {
      requests.push({ url, ...init })
      const response = { ok: true, blob: async () => new Blob(['wav']) }
      if (delayed) return new Promise(resolve => { respond = () => resolve(response) })
      return Promise.resolve(response)
    },
  }
  vm.runInNewContext(code, context)
  return { ...context.exports, requests, audios, revoked, states,
    onState: state => states.push(state), respond: () => respond() }
}
const flush = () => new Promise(resolve => setImmediate(resolve))

test('a complete Chinese utterance is one request and video waits until ended', async () => {
  const h = setup()
  const text = '莱拉弯腰将虚弱的艾拉尔扶了起来，艾拉尔指向断桥，大声喊道：快撤！'
  const speech = h.speak(text, h.onState)
  await flush()
  assert.equal(h.requests.length, 1)
  assert.equal(JSON.parse(h.requests[0].body).text, text)
  assert.deepEqual(h.states, [true])
  let videoReady = false
  const handoff = h.finishSpeaking().then(() => { videoReady = true })
  await flush()
  assert.equal(videoReady, false, 'automatic video must not truncate audible narration')
  h.audios[0].onended()
  await Promise.all([speech, handoff])
  assert.equal(videoReady, true)
  assert.deepEqual(h.states, [true, false])
  assert.equal(h.audios[0].removed, true)
  assert.equal(h.revoked.length, 1)
})

test('fast video cancels pending synthesis without waiting or late audio', async () => {
  const h = setup({ delayed: true })
  const speech = h.speak('还没有开始的旁白', h.onState)
  assert.equal(h.finishSpeaking(), undefined)
  assert.equal(h.requests[0].signal.aborted, true)
  h.respond()
  await speech
  assert.equal(h.audios.length, 0)
  assert.deepEqual(h.states, [false])
})

test('user cancellation releases a waiting video and cleans up playback', async () => {
  const h = setup()
  const speech = h.speak('正在播放的旁白', h.onState)
  await flush()
  const handoff = h.finishSpeaking()
  h.stopSpeaking()
  await Promise.all([speech, handoff])
  assert.deepEqual(h.states, [true, false])
  assert.equal(h.audios[0].removed, true)
})

test('autoplay rejection never reports speaking and is surfaced', async () => {
  const h = setup({ blocked: true })
  await assert.rejects(h.speak('旁白', h.onState), /autoplay-blocked/)
  assert.deepEqual(h.states, [false])
  assert.equal(h.audios[0].removed, true)
})

test('decode failure is surfaced instead of silently pretending the sentence ended', async () => {
  const h = setup()
  const speech = h.speak('旁白', h.onState)
  await flush()
  h.audios[0].onerror()
  await assert.rejects(speech, /audio-playback-failed/)
  assert.deepEqual(h.states, [true, false])
})
