/** Start narration before a whole paragraph has been synthesized. */
export function speechChunks(text: string, limit = 12): string[] {
  const chunks: string[] = []
  const clauses = text.trim().match(/[^，。！？；、,.!?;\n]+[，。！？；、,.!?;\n]*/gu) ?? []
  for (const clause of clauses) {
    let chars = Array.from(clause.trim())
    while (chars.length > limit) {
      let end = chars.slice(0, limit + 1).lastIndexOf(' ')
      if (end < Math.floor(limit / 2)) {
        end = limit
        // Keep an unspaced Latin word together.
        while (end < chars.length && /[A-Za-z0-9]/.test(chars[end - 1])
               && /[A-Za-z0-9]/.test(chars[end])) end += 1
      }
      const value = chars.slice(0, end).join('').trim()
      if (value) chunks.push(value)
      chars = Array.from(chars.slice(end).join('').trimStart())
    }
    const rest = chars.join('').trim()
    if (rest) chunks.push(rest)
  }
  return chunks
}
