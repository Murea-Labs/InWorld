import { cleanup, fireEvent, render } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'
import Hud from '../src/components/Hud'
import type { Node, World } from '../src/types'

afterEach(() => { cleanup(); vi.restoreAllMocks(); vi.unstubAllGlobals() })

const options = [
  '紧盯着快速飙升的解密进度条，准备在解密完成后立即关闭反应堆。',
  '闪身退到一旁，敏捷地避开从上方管道突然喷涌的蒸汽。',
  '对着手环焦急地大声呼喊，询问同伴是否已经抵达安全出口。',
]
const node = { id: 'n1', options, recommend: 0 } as Node
const world = { id: 'w1', head: 'n1', subjects: {}, places: {} } as World

function setup() {
  const onSend = vi.fn()
  const props = { world, current: node, busy: false, turn: null, sequence: null,
    draft: '', onDraft: vi.fn(), onSend, onStop: vi.fn(), autoRunning: false,
    onStopAuto: vi.fn(), onAddRef: vi.fn() }
  return { ...render(<Hud {...props} />), props, onSend }
}

it('shows full choice text above the choices on hover and dismisses when leaving', () => {
  const ui = setup()
  const choices = ui.container.querySelector('.hud-options')!
  expect(ui.queryByRole('tooltip')).toBeNull()
  options.forEach((text) => {
    const button = ui.getByRole('button', { name: new RegExp(text) })
    fireEvent.mouseEnter(button)
    const tooltip = ui.getByRole('tooltip')
    expect(tooltip.textContent).toBe(text)
    expect(button.getAttribute('aria-describedby')).toBe(tooltip.id)
    fireEvent.mouseLeave(choices)
    expect(ui.queryByRole('tooltip')).toBeNull()
  })
})

it('centers every preview on its own button, including wrapped rows', () => {
  vi.stubGlobal('innerWidth', 1600)
  const ui = setup()
  const row = ui.container.querySelector('.hud-options')!
  vi.spyOn(row, 'getBoundingClientRect').mockReturnValue({ left: 100, bottom: 700 } as DOMRect)
  ;[300, 700, 1100].forEach((left, index) => {
    const button = ui.getByRole('button', { name: new RegExp(options[index]) })
    const top = 500 + index * 50
    vi.spyOn(button, 'getBoundingClientRect').mockReturnValue({ left, width: 200, top } as DOMRect)
    fireEvent.mouseEnter(button)
    const tooltip = ui.getByRole('tooltip')
    expect(tooltip.style.left).toBe(`${left + 100 - 100}px`)
    expect(tooltip.style.bottom).toBe(`${700 - top + 12}px`)
    expect(tooltip.style.width).toBe('280px')
    expect((tooltip.querySelector('.option-tooltip-tip') as HTMLElement).style.left).toBe('140px')
  })
  fireEvent(window, new Event('resize'))
  expect(ui.queryByRole('tooltip')).toBeNull()
})

it('keeps the preview inside both viewport edges', () => {
  vi.stubGlobal('innerWidth', 800)
  const ui = setup()
  const row = ui.container.querySelector('.hud-options')!
  vi.spyOn(row, 'getBoundingClientRect').mockReturnValue({ left: 0, bottom: 600 } as DOMRect)
  ;[0, 700].forEach((left, index) => {
    const button = ui.getByRole('button', { name: new RegExp(options[index]) })
    vi.spyOn(button, 'getBoundingClientRect').mockReturnValue({ left, width: 100, top: 550 } as DOMRect)
    fireEvent.focus(button)
    const tooltip = ui.getByRole('tooltip')
    expect(tooltip.style.left).toBe(index === 0 ? '152px' : '648px')
    expect((tooltip.querySelector('.option-tooltip-tip') as HTMLElement).style.left)
      .toBe(index === 0 ? '38px' : '242px')
  })
})

it('supports focus and Escape, and sends the original choice on click', () => {
  const ui = setup()
  const button = ui.getByRole('button', { name: new RegExp(options[1]) })
  fireEvent.focus(button)
  expect(ui.getByRole('tooltip').textContent).toBe(options[1])
  fireEvent.keyDown(button, { key: 'Escape' })
  expect(ui.queryByRole('tooltip')).toBeNull()
  fireEvent.mouseEnter(button)
  fireEvent.click(button)
  expect(ui.onSend).toHaveBeenCalledExactlyOnceWith(options[1])
  expect(ui.queryByRole('tooltip')).toBeNull()
})

it('does not keep an old preview after switching clips or updating choices', () => {
  const ui = setup()
  fireEvent.mouseEnter(ui.getByRole('button', { name: new RegExp(options[0]) }))
  ui.rerender(<Hud {...ui.props} current={{ ...node, options: ['新版建议', ...options.slice(1)] }} />)
  expect(ui.queryByRole('tooltip')).toBeNull()
  ui.rerender(<Hud {...ui.props} world={{ ...world, head: 'n2' }} current={{ ...node, id: 'n2' }} />)
  expect(ui.queryByRole('tooltip')).toBeNull()
})
