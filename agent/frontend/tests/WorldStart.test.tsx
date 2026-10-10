import { act, cleanup, fireEvent, render, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import App from '../src/App'
import * as api from '../src/api'
import type { Health, RandomWorld, World } from '../src/types'
import { setLanguage } from '../src/i18n'
import { OPENING } from '../src/lib'

vi.mock('../src/api', async (original) => ({ ...await original<typeof api>(),
  getHealth: vi.fn(), listUsers: vi.fn(), getCurrentUser: vi.fn(), listWorlds: vi.fn(),
  getWorld: vi.fn(), randomWorld: vi.fn(), createWorld: vi.fn(), turn: vi.fn(), prefetch: vi.fn(),
  addSubject: vi.fn(), addPlace: vi.fn(),
}))
vi.mock('../src/components/Stage', () => ({ default: () => null }))
vi.mock('../src/components/Header', () => ({ default: () => null }))
vi.mock('../src/dialogs/NewWorld', () => ({ default: () => null }))
vi.mock('../src/dialogs/AddAsset', () => ({ default: () => null }))
vi.mock('../src/dialogs/WorldLog', () => ({ default: () => null }))
vi.mock('../src/tts', () => ({ loadTts: () => false, saveTts: vi.fn(), speak: vi.fn(),
  stopSpeaking: vi.fn(), finishSpeaking: vi.fn() }))

const idea = '去冰塔峰之龙的世界探险'
const concept: RandomWorld = { title: '冰塔峰', setting: '雪山驿站与冰龙的领地', mainline: '寻找冰龙',
  opening: '你站在驿站外，望向雪山。', style_id: '', assets: [
    { kind: 'player', name: '旅人', prompt: 'A traveller in a blue coat.' },
    { kind: 'place', name: '驿站', prompt: 'A stone lodge in snowy mountains.' },
  ] }
const world = { id: 'ice-world', title: concept.title, subjects: {}, places: {}, nodes: {},
  history: [], head: '', quests: [], place_links: [] } as unknown as World

beforeEach(() => {
  setLanguage('zh')
  vi.mocked(api.getHealth).mockResolvedValue({ mode: 'single', tts: false,
    video: { id: 'murea', choices: [] }, ready: { llm: true, image: true, video: true } } as unknown as Health)
  const user = { id: 'bc', name: 'bc', created: '' }
  vi.mocked(api.listUsers).mockResolvedValue([user])
  vi.mocked(api.getCurrentUser).mockResolvedValue(user)
  vi.mocked(api.listWorlds).mockResolvedValue([])
  vi.mocked(api.getWorld).mockResolvedValue(world)
  vi.mocked(api.randomWorld).mockResolvedValue(concept)
  vi.mocked(api.createWorld).mockResolvedValue(world)
  vi.mocked(api.turn).mockResolvedValue(undefined)
  vi.mocked(api.prefetch).mockResolvedValue({ ok: true } as never)
  vi.mocked(api.addSubject).mockResolvedValue({ id: 'S1', ref: 'subject.png' })
  vi.mocked(api.addPlace).mockResolvedValue({ name: '驿站', ref: 'place.png' })
})
afterEach(() => { cleanup(); vi.clearAllMocks() })

it('develops a sentence, saves its concept and starts the opening video automatically', async () => {
  const ui = render(<App />)
  const input = await ui.findByPlaceholderText(/想去怎样的世界/)
  fireEvent.change(input, { target: { value: idea } })
  fireEvent.click(ui.getByRole('button', { name: '发送' }))
  await waitFor(() => expect(api.turn).toHaveBeenCalled())
  expect(api.randomWorld).toHaveBeenCalledWith(expect.any(Array), idea)
  expect(api.createWorld).toHaveBeenCalledWith(concept.title, concept.setting, '', 'murea', concept.mainline)
  expect(api.turn).toHaveBeenCalledWith(world.id, `${OPENING} ${concept.opening}`,
    expect.any(Function), false, expect.any(AbortSignal), [])
  expect(api.addSubject).toHaveBeenCalledTimes(1)
  expect(vi.mocked(api.addSubject).mock.calls[0][1].get('control')).toBe('user')
  expect(vi.mocked(api.addSubject).mock.calls[0][1].get('kind')).toBe('人物')
  expect(api.addPlace).toHaveBeenCalledTimes(1)
  expect(vi.mocked(api.addPlace).mock.calls[0][1].get('make_current')).toBe('true')
  expect(vi.mocked(api.turn).mock.invocationCallOrder[0]).toBeGreaterThan(vi.mocked(api.addSubject).mock.invocationCallOrder[0])
  expect(vi.mocked(api.turn).mock.invocationCallOrder[0]).toBeGreaterThan(vi.mocked(api.addPlace).mock.invocationCallOrder[0])
})

it('waits for assets and resumes failed preparation without recreating the world or player', async () => {
  let reject!: (error: Error) => void
  vi.mocked(api.addPlace).mockImplementationOnce(() => new Promise((_, fail) => { reject = fail }))
  const ui = render(<App />)
  const input = await ui.findByPlaceholderText(/想去怎样的世界/)
  fireEvent.change(input, { target: { value: idea } })
  fireEvent.click(ui.getByRole('button', { name: '发送' }))
  await ui.findByText('准备世界资产 2/2：驿站')
  expect(api.turn).not.toHaveBeenCalled()
  await act(async () => reject(new Error('image unavailable')))
  await ui.findByText('创建世界失败：image unavailable')
  expect((input as HTMLTextAreaElement).value).toBe(idea)
  fireEvent.click(ui.getByRole('button', { name: '发送' }))
  await waitFor(() => expect(api.turn).toHaveBeenCalledTimes(1))
  expect(api.randomWorld).toHaveBeenCalledTimes(1)
  expect(api.createWorld).toHaveBeenCalledTimes(1)
  expect(api.addSubject).toHaveBeenCalledTimes(1)
  expect(api.addPlace).toHaveBeenCalledTimes(2)
})

it('shows preparation progress, prevents double submits and preserves a failed idea', async () => {
  let reject!: (error: Error) => void
  vi.mocked(api.randomWorld).mockReturnValue(new Promise((_, fail) => { reject = fail }))
  const ui = render(<App />)
  const input = await ui.findByPlaceholderText(/想去怎样的世界/)
  fireEvent.change(input, { target: { value: idea } })
  fireEvent.click(ui.getByRole('button', { name: '发送' }))
  await ui.findByText('正在根据你的想法完善世界、主线与开场…')
  expect((ui.getByRole('button', { name: '发送' }) as HTMLButtonElement).disabled).toBe(true)
  fireEvent.keyDown(input, { key: 'Enter' })
  expect(api.randomWorld).toHaveBeenCalledTimes(1)
  await act(async () => reject(new Error('planning unavailable')))
  await ui.findByText('创建世界失败：planning unavailable')
  expect((input as HTMLTextAreaElement).value).toBe(idea)
  expect(api.createWorld).not.toHaveBeenCalled()
  expect(api.turn).not.toHaveBeenCalled()
})

it('keeps input in an existing world on the normal action path', async () => {
  vi.mocked(api.listWorlds).mockResolvedValue([world] as never)
  const ui = render(<App />)
  const input = await ui.findByPlaceholderText('你要做什么（输入 @ 绑定资产）')
  fireEvent.change(input, { target: { value: '沿山路前进' } })
  fireEvent.click(ui.getByRole('button', { name: '发送' }))
  await waitFor(() => expect(api.turn).toHaveBeenCalled())
  expect(api.randomWorld).not.toHaveBeenCalled()
  expect(api.createWorld).not.toHaveBeenCalled()
})
