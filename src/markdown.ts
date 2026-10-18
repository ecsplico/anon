// Markdown-aware protection: detection must never rewrite structural Markdown (fenced/inline code,
// link targets, autolinks). Frontmatter and link text are NOT protected: they can hold PII.

/** Fenced code blocks: ``` or ~~~ … matching closing fence (line anchored). */
const FENCE = /^[ \t]*(`{3,}|~{3,})[\s\S]*?^[ \t]*\1[ \t]*$/gm
/** Inline code: `code` or ``co`de`` (matched backtick run length). */
const INLINE_CODE = /(`+).+?\1/g
/** Link / image URL target: `](<url>)`; only the parenthesised target (capture group 1, at the match end). */
const LINK_TARGET = /\]\(\s*(<?[^)\s]+>?)/g
/** Autolinks `<https://…>` / `<mailto:…>`; only the inner URL (group 1) is protected. */
const AUTOLINK = /<((?:https?|mailto|ftp):[^>\s]+)>/g

export type Range = [start: number, end: number]

/** Sorted, merged `[start, end)` ranges that must not be touched. */
export function protectedSpans(text: string): Range[] {
  const spans: Range[] = []
  for (const m of text.matchAll(FENCE)) spans.push([m.index, m.index + m[0].length])
  for (const m of text.matchAll(INLINE_CODE)) spans.push([m.index, m.index + m[0].length])
  for (const m of text.matchAll(LINK_TARGET)) {
    const g = m[1]!
    spans.push([m.index + m[0].length - g.length, m.index + m[0].length])
  }
  for (const m of text.matchAll(AUTOLINK)) {
    const g = m[1]!
    spans.push([m.index + 1, m.index + 1 + g.length])
  }
  return merge(spans)
}

function merge(spans: Range[]): Range[] {
  if (spans.length === 0) return []
  spans.sort((a, b) => a[0] - b[0] || a[1] - b[1])
  const merged: Range[] = [spans[0]!]
  for (const [start, end] of spans.slice(1)) {
    const last = merged[merged.length - 1]!
    if (start <= last[1]) last[1] = Math.max(last[1], end)
    else merged.push([start, end])
  }
  return merged
}

/** True if `[start, end)` intersects any protected span. */
export function overlaps(start: number, end: number, spans: readonly Range[]): boolean {
  return spans.some(([s, e]) => start < e && s < end)
}
