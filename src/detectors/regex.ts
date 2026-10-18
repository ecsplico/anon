// Regex detector for German texts: contact data, IBAN/KVNR, dates, salutation surnames.
// All recognizers scan `ctx.masked` (placeholders blanked out); originals are sliced from `text`.

import { findPhoneNumbersInText } from 'libphonenumber-js'
import { isValidIBAN } from 'ibantools'
import { MONTHS } from '../text.ts'
import type { EntityKind, Span } from '../types.ts'
import type { Detector } from './types.ts'

export const REGEX_ENTITIES: EntityKind[] = [
  'CONTACT_EMAIL',
  'CONTACT_PHONE',
  'CONTACT_FAX',
  'CONTACT_URL',
  'CONTACT_IP',
  'ID',
  'DATE',
  'NAME_EXT'
]

const EMAIL = /[\p{L}\p{N}._%+-]+@[\p{L}\p{N}.-]+\.\p{L}{2,}/gu
const IBAN = /\b[A-Z]{2}\d{2}(?:\s?[A-Z0-9]{4}){2,7}\s?[A-Z0-9]{1,4}\b/g
const KVNR = /\b[A-Z]\d{9}\b/g
const URL_RE = /\bhttps?:\/\/[^\s<>()]+|\bwww\.[^\s<>()]+/g
const IP = /(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?!\d|\.\d)/g

const MONTH_ALTERNATION = MONTHS.flat()
  .sort((a, b) => b.length - a.length)
  .join('|')
const DATES = [
  /(?<![\p{N}])\d{1,2}\s?\.\s?\d{1,2}\s?\.\s?(?:\d{4}|\d{2})(?![\p{N}])/gu,
  /\b\d{4}-\d{2}-\d{2}\b/g,
  new RegExp(`(?<![\\p{N}])\\d{1,2}\\.?\\s+(?:${MONTH_ALTERNATION})\\.?\\s+\\d{4}(?![\\p{N}])`, 'gu')
]

/** Surname after Herr/Frau/Hr./Fr. (optionally with Dr./Prof./med.); the title tokens themselves are never matched. */
const SALUTATION =
  /(?<=(?<![\p{L}\p{N}_])(?:Herrn?|Frau|Hr\.|Fr\.)(?:\s+(?:Dr|Prof)\.(?:\s*med\.)?)*\s+)(?!(?:Dr|Prof|Med|Dent|Dipl)\.)\p{Lu}\p{Ll}+(?:-\p{Lu}\p{Ll}+)?(?![\p{L}\p{N}_])/gu

type Range = [number, number]

function scan(masked: string, re: RegExp): Range[] {
  return [...masked.matchAll(re)].map(m => [m.index, m.index + m[0].length] as Range)
}

function digitCount(s: string): number {
  return s.match(/\d/g)?.length ?? 0
}

/** IBAN candidates validated by checksum; an over-greedy match is trimmed group by group. */
function ibanRanges(masked: string): Range[] {
  const out: Range[] = []
  for (const m of masked.matchAll(IBAN)) {
    let text = m[0]
    while (text.length >= 15) {
      if (isValidIBAN(text.replace(/\s/g, ''))) {
        out.push([m.index, m.index + text.length])
        break
      }
      const cut = text.search(/\s\S*$/)
      if (cut < 0) break
      text = text.slice(0, cut)
    }
  }
  return out
}

function ipRanges(masked: string): Range[] {
  return scan(masked, IP).filter(([s, e]) => masked.slice(s, e).split('.').every(o => Number(o) <= 255))
}

/** Strip sentence punctuation that cannot end a URL. */
function urlRanges(masked: string): Range[] {
  return [...masked.matchAll(URL_RE)].map(m => {
    const trimmed = m[0].replace(/[.,;:!?'"]+$/, '')
    return [m.index, m.index + trimmed.length] as Range
  })
}

function phoneSpans(masked: string, suppress: Range[]): { start: number; end: number; fax: boolean }[] {
  const out: { start: number; end: number; fax: boolean }[] = []
  for (const f of findPhoneNumbersInText(masked, 'DE')) {
    const text = masked.slice(f.startsAt, f.endsAt)
    if (!text.includes('+') && digitCount(text) < 6) continue
    if (/^\d+(?:\.\d+){2,}$/.test(text)) continue // dotted numbers: invalid IPs, versions, dates
    if (suppress.some(([s, e]) => f.startsAt < e && s < f.endsAt)) continue
    const context = masked.slice(Math.max(0, f.startsAt - 12), f.startsAt)
    out.push({ start: f.startsAt, end: f.endsAt, fax: /fax/i.test(context) })
  }
  return out
}

export function regexDetector(opts: { entities?: EntityKind[] } = {}): Detector {
  const enabled = new Set<EntityKind>(opts.entities ?? REGEX_ENTITIES)
  return {
    name: 'regex',
    async detect(text, ctx) {
      const masked = ctx.masked
      const spans: Span[] = []
      const add = (start: number, end: number, kind: EntityKind, score: number) => {
        if (enabled.has(kind)) spans.push({ start, end, text: text.slice(start, end), kind, score, detector: 'regex' })
      }

      // Always computed: also used to keep their digits from being reported as phone numbers.
      const emails = scan(masked, EMAIL)
      const urls = urlRanges(masked)
      const ips = ipRanges(masked)
      const ibans = ibanRanges(masked)
      const dates = DATES.flatMap(re => scan(masked, re))

      for (const [s, e] of emails) add(s, e, 'CONTACT_EMAIL', 1)
      for (const [s, e] of urls) add(s, e, 'CONTACT_URL', 0.9)
      for (const [s, e] of ips) add(s, e, 'CONTACT_IP', 0.9)
      for (const [s, e] of ibans) add(s, e, 'ID', 1)
      for (const [s, e] of scan(masked, KVNR)) add(s, e, 'ID', 0.7)
      for (const [s, e] of dates) add(s, e, 'DATE', 0.6)
      for (const [s, e] of scan(masked, SALUTATION)) add(s, e, 'NAME_EXT', 0.6)

      if (enabled.has('CONTACT_PHONE') || enabled.has('CONTACT_FAX')) {
        const suppress = [...emails, ...urls, ...ips, ...ibans, ...dates]
        for (const p of phoneSpans(masked, suppress)) add(p.start, p.end, p.fax ? 'CONTACT_FAX' : 'CONTACT_PHONE', 0.8)
      }

      return spans.sort((a, b) => a.start - b.start || b.end - a.end)
    }
  }
}
