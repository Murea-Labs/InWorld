import { act, cleanup, fireEvent, render, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import App from '../src/App'
import * as api from '../src/api'
import type { Health, Node, RandomWorld, TurnEvent, World } from '../src/types'
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

it.each([true, false])('keeps cumulative progress when a later clip continues (replanned: %s)', async (replanned) => {
  vi.mocked(api.listWorlds).mockResolvedValue([world] as never)
  const requests: { emit: (event: TurnEvent) => void; finish: () => void }[] = []
  vi.mocked(api.turn).mockImplementation((_id, _text, emit) => new Promise<void>((finish) => {
    requests.push({ emit, finish })
  }))
  const ui = render(<App />)
  const input = await ui.findByPlaceholderText('你要做什么（输入 @ 绑定资产）')
  fireEvent.change(input, { target: { value: '沿山路前进，再穿过石桥' } })
  fireEvent.click(ui.getByRole('button', { name: '发送' }))
  await waitFor(() => expect(requests).toHaveLength(1))
  act(() => requests[0].emit({ type: 'sequence', goal: '沿山路前进，再穿过石桥', current: 1,
    total: 2, segments: [{ index: 1, content: '走向桥头' }, { index: 2, content: '穿过石桥' }] }))
  expect(ui.getByText('连续生成 2 段')).toBeTruthy()

  const complete = async (i: number, reply: string) => {
    const node = { id: `n${i}`, reply, continue_with: '继续过桥', plan: ['过桥'],
      options: [], clip: '', frames: [] } as unknown as Node
    await act(async () => {
      requests[i].emit({ type: 'done', reply, node,
        world: { ...world, head: node.id, nodes: { [node.id]: node } } as unknown as World })
      requests[i].finish()
    })
    await waitFor(() => expect(requests).toHaveLength(i + 2))
  }
  await complete(0, '已经抵达桥头')
  if (replanned) {
    act(() => requests[1].emit({ type: 'sequence', goal: '穿过石桥', current: 1, total: 2,
      segments: [{ index: 1, content: '踏上石桥' }, { index: 2, content: '抵达对岸' }] }))
    expect(ui.getByText('连续生成 3 段')).toBeTruthy()
    expect(ui.container.querySelector('.turn-sequence')?.getAttribute('title'))
      .toBe('总内容：沿山路前进，再穿过石桥')
    expect(ui.container.querySelectorAll('.turn-segment-index')).toHaveLength(3)
    expect(ui.container.querySelector('.turn-segment-index.done')?.getAttribute('title'))
      .toContain('已经抵达桥头')
    expect(ui.container.querySelector('.turn-segment-index.generating')?.textContent).toBe('2')
  }
  await complete(1, '走到了桥中央')
  expect(ui.getByText('连续生成 3 段')).toBeTruthy()
  expect(ui.getByText(/当前第 3 段/)).toBeTruthy()
  expect([...ui.container.querySelectorAll('.turn-segment-index')].map((el) => el.textContent))
    .toEqual(['1', '2', '3'])
  expect(ui.container.querySelectorAll('.turn-segment-index.done')).toHaveLength(2)
  expect(ui.container.querySelector('.turn-segment-index.generating')?.textContent).toBe('3')
  await act(async () => {
    requests[2].emit({ type: 'error', message: '测试结束' })
    requests[2].finish()
  })
})
