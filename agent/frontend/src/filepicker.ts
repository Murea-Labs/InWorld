/** Whether a native file panel is (probably) open.
 *
 * Dismissing the OS file panel with Esc lets that same Esc reach the page, and a
 * <dialog> turns it into `cancel` — so picking a file and backing out would close
 * the whole dialog. Track the panel via window focus: opening it takes focus away,
 * closing it (either way) gives focus back.
 */
let picking = false

export function markFilePickerOpen() {
  picking = true
  const done = () => {
    window.removeEventListener('focus', done)
    // Let the Esc that dismissed the panel land first, then stop guarding.
    setTimeout(() => {
      picking = false
    }, 350)
  }
  window.addEventListener('focus', done)
  // Belt and braces: never guard forever if focus never comes back.
  setTimeout(() => {
    picking = false
  }, 60_000)
}

export const isPickingFile = () => picking
