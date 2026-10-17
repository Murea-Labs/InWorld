import { cleanup, fireEvent, render } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'
import SideRail from '../src/components/SideRail'
import type { World } from '../src/types'
import { setLanguage } from '../src/i18n'

vi.mock('../src/components/Chat', () => ({ default: () => null }))
vi.mock('../src/components/WorldMapMini', () => ({ default: () => null }))
setLanguage('zh')
afterEach(cleanup)

it('protects the first clip even when detached later clips remain', () => {
  const world = { head: 'one', history: [], nodes: {
    one: { id: 'one', parent: null }, two: { id: 'two', parent: 'one' },
  } } as unknown as World
  const onUndo = vi.fn()
  const props = { world, current: null, theme: 'mist' as const, tab: 'chat' as const,
    onTab: vi.fn(), busy: false, pending: '', turn: null, onSelect: vi.fn(),
    onOpenMap: vi.fn(), onInspect: vi.fn(), onOpenAsset: vi.fn(), onUndo }
  const ui = render(<SideRail {...props} />)
  const button = ui.getByRole('button', { name: '撤回' }) as HTMLButtonElement
  expect(button.disabled).toBe(true)
  fireEvent.click(button)
  expect(onUndo).not.toHaveBeenCalled()
  ui.rerender(<SideRail {...props} world={{ ...world, head: 'two' }} />)
  expect(button.disabled).toBe(false)
})
