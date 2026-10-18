import { describe, expect, test } from 'bun:test'
import { overlaps, protectedSpans } from '../src/markdown.ts'

const slices = (text: string) => protectedSpans(text).map(([s, e]) => text.slice(s, e))

describe('protectedSpans', () => {
  test('fenced code block is protected as a whole', () => {
    const text = 'Vorher anna@example.de\n```sh\nmail anna@example.de\n```\nNachher'
    expect(slices(text)).toEqual(['```sh\nmail anna@example.de\n```'])
  })

  test('tilde fences and longer fences need a matching closer', () => {
    const text = '~~~\ncode\n~~~\ntext\n````\nx\n```\ny\n````\n'
    expect(slices(text)).toEqual(['~~~\ncode\n~~~', '````\nx\n```\ny\n````'])
  })

  test('inline code, including double backtick runs', () => {
    expect(slices('Nutze `a@example.de` oder ``co`de``.')).toEqual(['`a@example.de`', '``co`de``'])
  })

  test('link URL target is protected, link text is not', () => {
    const text = 'Siehe [Anna Müller](https://example.de/anna) hier'
    expect(slices(text)).toEqual(['https://example.de/anna'])
    const start = text.indexOf('Anna')
    expect(overlaps(start, start + 'Anna Müller'.length, protectedSpans(text))).toBe(false)
  })

  test('image targets and angle-bracket targets', () => {
    expect(slices('![Bild](bild.png) Ende')).toEqual(['bild.png'])
    expect(slices('[x](<https://example.de/y>)')).toEqual(['<https://example.de/y>'])
  })

  test('autolinks protect the inner URL only', () => {
    const text = 'Mail <mailto:anna@example.de> und <https://example.de>'
    expect(slices(text)).toEqual(['mailto:anna@example.de', 'https://example.de'])
  })

  test('frontmatter is not protected', () => {
    const text = '---\nname: Anna Müller\n---\n\nText'
    expect(protectedSpans(text)).toEqual([])
  })

  test('unclosed fence does not protect the content; empty text has no spans', () => {
    const text = '```\nnie geschlossen'
    expect(overlaps(text.indexOf('nie'), text.length, protectedSpans(text))).toBe(false)
    expect(protectedSpans('')).toEqual([])
  })

  test('nested regions are merged into one span; spans come out sorted', () => {
    const nested = '```\n`a` `b`\n```'
    expect(protectedSpans(nested)).toEqual([[0, nested.length]])
    const text = '[a](https://e.de) und `x`'
    const spans = protectedSpans(text)
    expect(spans).toHaveLength(2)
    expect(spans[0]![0]).toBeLessThan(spans[1]![0])
  })
})

describe('overlaps', () => {
  const spans: [number, number][] = [
    [10, 20],
    [30, 40]
  ]
  test('intersection is half-open', () => {
    expect(overlaps(0, 10, spans)).toBe(false)
    expect(overlaps(0, 11, spans)).toBe(true)
    expect(overlaps(19, 25, spans)).toBe(true)
    expect(overlaps(20, 30, spans)).toBe(false)
    expect(overlaps(12, 15, spans)).toBe(true)
    expect(overlaps(35, 50, spans)).toBe(true)
    expect(overlaps(40, 50, spans)).toBe(false)
  })
})
