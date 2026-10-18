// Transformers.js token-classification output has no character offsets, only the (normalised)
// `word` of each token. openmed's own aligner searches that word in a lower-cased, accent-stripped
// copy of the text and throws when it cannot: a single character the tokenizer deletes (soft hyphen,
// zero-width space, control characters) makes the whole document fail. This aligner mirrors the
// BERT normaliser (clean_text, whitespace folding, lower-casing, NFD accent stripping), never
// throws, and drops tokens it cannot place instead of letting offsets of later tokens drift.

export type RawToken = {
  entity?: string
  entity_group?: string
  score?: number
  word?: string
  index?: number
  start?: number
  end?: number
}

export type AlignResult = { tokens: RawToken[]; unaligned: RawToken[] }

const SPECIAL_WORDS: Record<string, true> = {
  '[CLS]': true, '[SEP]': true, '[PAD]': true, '[MASK]': true, '<s>': true, '</s>': true,
  '<pad>': true, '<mask>': true, '<cls>': true, '<sep>': true, '<bos>': true, '<eos>': true,
}
const UNK_WORDS: Record<string, true> = { '[UNK]': true, '<unk>': true }
const CONTINUATION = '##'
const WORD_START_MARKERS = /^[▁Ġ]+/u

/** Window (in normalised chars) in which a token is searched when it does not sit exactly at the cursor. */
const CONTINUATION_WINDOW = 16
const WORD_WINDOW = 24

const CJK = /[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u{20000}-\u{2fa1f}]/u
const WHITESPACE = /^[ \t\n\r\p{Zs}\p{Zl}\p{Zp}]$/u
const CONTROL = /^[\p{Cc}\p{Cf}\p{Co}\p{Cn}\p{Cs}]$/u
const MARK = /\p{M}/gu
const PUNCT = /^[\p{P}!-/:-@[-`{-~]$/u

type Normalized = { chars: string; start: number[]; end: number[] }

/** The text as the BERT tokenizer sees it, with every normalised char mapped back to its source range. */
function normalize(text: string): Normalized {
  const out: string[] = []
  const start: number[] = []
  const end: number[] = []
  const push = (c: string, from: number, to: number): void => {
    out.push(c)
    start.push(from)
    end.push(to)
  }
  let offset = 0
  for (const ch of text) {
    const to = offset + ch.length
    if (WHITESPACE.test(ch)) {
      push(' ', offset, to)
    } else if (CONTROL.test(ch) || ch === '\ufffd') {
      // Removed by clean_text. Extend the previous char so a token spanning it covers it.
      if (end.length > 0) end[end.length - 1] = to
    } else if (CJK.test(ch)) {
      push(' ', offset, offset)
      push(ch.toLowerCase(), offset, to)
      push(' ', to, to)
    } else {
      const n = ch.toLowerCase().normalize('NFD').replace(MARK, '')
      if (n.length === 0) {
        if (end.length > 0) end[end.length - 1] = to
      } else {
        for (let i = 0; i < n.length; i++) push(n[i] as string, offset, to)
      }
    }
    offset = to
  }
  return { chars: out.join(''), start, end }
}

function atWordStart(chars: string, pos: number): boolean {
  if (pos === 0) return true
  const prev = chars[pos - 1] as string
  return prev === ' ' || PUNCT.test(prev)
}

/**
 * Gives every token `start`/`end` (UTF-16 offsets into `text`). Tokens that already carry offsets
 * pass through; special tokens are dropped; `[UNK]` receives the offsets of the word it stands for;
 * tokens that cannot be placed are dropped (equivalent to label `O`) and returned in `unaligned`.
 * A failed token never moves the cursor, so the offsets of later tokens cannot drift.
 */
export function alignTokens(text: string, tokens: RawToken[]): AlignResult {
  const norm = normalize(text)
  const { chars } = norm
  const aligned: RawToken[] = []
  const unaligned: RawToken[] = []
  let cursor = 0

  // First normalised char at or after `from` that is not whitespace.
  const skipSpace = (from: number): number => {
    let p = from
    while (p < chars.length && chars[p] === ' ') p++
    return p
  }
  const originalToNormalized = (offset: number): number => {
    // First normalised char whose source range ends after `offset`.
    let lo = 0
    let hi = norm.end.length
    while (lo < hi) {
      const mid = (lo + hi) >> 1
      if ((norm.end[mid] as number) <= offset) lo = mid + 1
      else hi = mid
    }
    return lo
  }

  const locate = (piece: string, continuation: boolean): number | null => {
    if (continuation) {
      if (chars.startsWith(piece, cursor)) return cursor
      const limit = Math.min(chars.length, cursor + CONTINUATION_WINDOW)
      const found = chars.indexOf(piece, cursor)
      return found !== -1 && found <= limit ? found : null
    }
    const pos = skipSpace(cursor)
    if (chars.startsWith(piece, pos)) return pos
    const limit = Math.min(chars.length, pos + WORD_WINDOW)
    let from = pos
    while (from <= limit) {
      const found = chars.indexOf(piece, from)
      if (found === -1 || found > limit) return null
      if (atWordStart(chars, found)) return found
      from = found + 1
    }
    return null
  }

  for (const token of tokens) {
    if (Number.isFinite(token.start) && Number.isFinite(token.end) && (token.end as number) > (token.start as number)) {
      aligned.push(token)
      cursor = Math.max(cursor, originalToNormalized(token.end as number))
      continue
    }
    const word = (token.word ?? '').trim()
    if (Object.hasOwn(SPECIAL_WORDS, word)) continue

    if (Object.hasOwn(UNK_WORDS, word)) {
      // The tokenizer replaced a whole pre-token (word or punctuation char) by [UNK].
      const from = skipSpace(cursor)
      let to = from
      if (to < chars.length && PUNCT.test(chars[to] as string)) {
        to++
      } else {
        while (to < chars.length && chars[to] !== ' ' && !PUNCT.test(chars[to] as string)) to++
      }
      if (to === from) {
        unaligned.push(token)
        continue
      }
      aligned.push({ ...token, start: norm.start[from] as number, end: norm.end[to - 1] as number })
      cursor = to
      continue
    }

    const continuation = word.startsWith(CONTINUATION)
    const piece = (continuation ? word.slice(CONTINUATION.length) : word)
      .replace(WORD_START_MARKERS, '')
      .toLowerCase()
      .normalize('NFD')
      .replace(MARK, '')
      .trim()
    if (!piece) {
      unaligned.push(token)
      continue
    }
    const match = locate(piece, continuation)
    if (match === null) {
      unaligned.push(token)
      continue
    }
    aligned.push({ ...token, start: norm.start[match] as number, end: norm.end[match + piece.length - 1] as number })
    cursor = match + piece.length
  }
  return { tokens: aligned, unaligned }
}

/**
 * Wraps a raw Transformers.js token-classification pipeline so its output carries offsets.
 * `onUnaligned` receives the tokens that were dropped (once per call, only when there are any).
 */
export function alignedPipeline(
  base: unknown,
  onUnaligned?: (info: { text: string; unaligned: RawToken[] }) => void,
): (text: string, options?: unknown) => Promise<RawToken[]> {
  const call = base as (text: string, options?: unknown) => RawToken[] | RawToken[][] | Promise<RawToken[] | RawToken[][]>
  const wrapped = async (text: string, options?: unknown): Promise<RawToken[]> => {
    const raw = await call(text, options)
    const flat = (Array.isArray(raw[0]) ? (raw as RawToken[][]).flat() : raw) as RawToken[]
    const { tokens, unaligned } = alignTokens(text, flat)
    if (unaligned.length > 0) onUnaligned?.({ text, unaligned })
    return tokens
  }
  return wrapped
}
