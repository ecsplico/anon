// Shared text helpers (single copy): word boundaries, literal escaping, date patterns,
// name variants, placeholder masking, chunking.

export const WORD = '\\p{L}\\p{N}_'
/** Wortanfang: davor weder Buchstabe, Ziffer noch Unterstrich. */
export const BEFORE = `(?<![${WORD}])`
/** Wortende: danach kein Wortzeichen, oder genau ein Genitiv-s gefolgt von Nicht-Wortzeichen. */
export const AFTER = `(?:(?![${WORD}])|(?=s(?![${WORD}])))`

/** Invertierte Schreibweise 'Nachname, Vorname' nicht direkt nach einem großgeschriebenen Wort ('Lena Sommerfeld, Lenas Mutter'). */
export const NOT_AFTER_CAPITALIZED = '(?<!\\p{Lu}\\p{L}*\\s+)'

export function escapeRe(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

/** Literal escapen, Whitespace flexibel. */
export function literal(s: string): string {
  return s.trim().split(/\s+/).map(escapeRe).join('\\s+')
}

export function wordPattern(body: string, flags = 'gu'): RegExp {
  return new RegExp(`${BEFORE}(?:${body})${AFTER}`, flags)
}

// ---------------------------------------------------------------------------
// Datum

export const MONTHS: string[][] = [
  ['Januar', 'Jänner', 'Jan'],
  ['Februar', 'Feb'],
  ['März', 'Maerz', 'Mär', 'Mrz'],
  ['April', 'Apr'],
  ['Mai'],
  ['Juni', 'Jun'],
  ['Juli', 'Jul'],
  ['August', 'Aug'],
  ['September', 'Sept', 'Sep'],
  ['Oktober', 'Okt'],
  ['November', 'Nov'],
  ['Dezember', 'Dez']
]

/** Regex-Quelltext (ohne Grenzen) für alle Schreibweisen eines dd.mm.yyyy-Datums, oder null. */
export function dateBody(geb: string): string | null {
  const m = geb.trim().match(/^(\d{1,2})\.(\d{1,2})\.(\d{4})$/)
  if (!m) return null
  const d = String(Number(m[1]))
  const mo = Number(m[2])
  const moStr = String(mo)
  const yyyy = m[3]!
  const yy = yyyy.slice(2)
  const year = `(?:${yyyy}|${yy})`
  const names = (MONTHS[mo - 1] ?? []).map(escapeRe).join('|')
  const alts = [
    // 12.03.2010, 12. 03. 2010, 1.3.10
    `0?${d}\\s?\\.\\s?0?${moStr}\\s?\\.\\s?${year}`,
    // 12/03/2010
    `0?${d}/0?${moStr}/${year}`,
    // 2010-03-12
    `${yyyy}-0?${moStr}-0?${d}`
  ]
  // 12. März 2010, 12 Mär. 2010
  if (names) alts.push(`0?${d}\\.?\\s+(?:${names})\\.?\\s+${yyyy}`)
  return alts.join('|')
}

export function datePattern(geb: string): RegExp {
  const body = dateBody(geb)
  if (!body) return new RegExp(`${BEFORE}${literal(geb)}(?![${WORD}])`, 'giu')
  return new RegExp(`(?<![\\p{N}])(?:${body})(?![\\p{N}])`, 'giu')
}

// ---------------------------------------------------------------------------
// Namen

export function clean(s: string | undefined): string {
  return (s ?? '').trim()
}

/** Ganzes Wort + Teile an Leerzeichen/Bindestrich (nur großgeschriebene Teile ≥ 3 Zeichen, also keine 'von'/'de'). */
export function nameVariants(name: string): string[] {
  const whole = clean(name)
  if (!whole) return []
  const out = [whole]
  const parts = whole.split(/[\s-]+/)
  if (parts.length > 1) {
    for (const p of parts) {
      if (p.length >= 3 && /^\p{Lu}/u.test(p) && !out.includes(p)) out.push(p)
    }
  }
  return out
}

export function slug(s: string): string {
  const t = s
    .toLowerCase()
    .replace(/ä/g, 'ae')
    .replace(/ö/g, 'oe')
    .replace(/ü/g, 'ue')
    .replace(/ß/g, 'ss')
    .replace(/[^a-z0-9]+/g, '')
  return t || 'person'
}

// ---------------------------------------------------------------------------
// Platzhalter maskieren, Chunking

/**
 * Every match of `pattern` (the placeholder pattern of a style) is replaced by spaces of equal
 * length, so indices into the masked text stay valid for the original.
 */
export function maskPlaceholders(text: string, pattern: RegExp): string {
  const re = pattern.global ? pattern : new RegExp(pattern.source, pattern.flags + 'g')
  re.lastIndex = 0
  return text.replace(re, m => ' '.repeat(m.length))
}

/** Overlapping chunks, cut at line breaks (else spaces in the last 40 %), starting at word starts. */
export function chunkText(text: string, size: number, overlap: number): string[] {
  if (text.length <= size) return [text]
  const out: string[] = []
  let start = 0
  while (start < text.length) {
    let end = Math.min(start + size, text.length)
    if (end < text.length) {
      // Am letzten Zeilenumbruch, sonst Leerzeichen im hinteren Drittel trennen.
      const floor = start + Math.floor(size * 0.6)
      const nl = text.lastIndexOf('\n', end)
      const sp = text.lastIndexOf(' ', end)
      const cut = nl >= floor ? nl : sp >= floor ? sp : -1
      if (cut > 0) end = cut
    }
    out.push(text.slice(start, end))
    if (end >= text.length) break
    let next = Math.max(end - overlap, start + 1)
    // Auf Wortanfang vorrücken, damit kein Name mittendrin getrennt beginnt.
    while (next < end && !/\s/.test(text[next - 1] ?? ' ')) next++
    start = next
  }
  return out
}
