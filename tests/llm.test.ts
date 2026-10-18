import { describe, expect, test } from 'bun:test'
import {
  assertLocalUrl,
  extractJson,
  findCandidates,
  llmDetector,
  openAiCompatibleChat,
  toRules,
  type NerChat
} from '../src/detectors/llm.ts'
import { semanticStyle } from '../src/placeholders.ts'
import { applyRules, buildRules } from '../src/rules.ts'
import { maskPlaceholders } from '../src/text.ts'
import type { Stammdaten } from '../src/types.ts'

// Ausschließlich erfundene Daten.
const TEXT = [
  '{{vorname}} berichtet von einem Streit mit ihrer Freundin Mia. Mia schrieb ihr danach per Nachricht.',
  'Gespräch mit {{km}}. Die Bezugspflege (Frau Hellwig) war dabei. Therapeutin Ostertag war nicht anwesend.',
  'Die Mutter und der Vater sind getrennt. Die Patientin besucht die Klinik am Montag im Januar.',
  'Hospitation an der Realschule Birkenhain in Rotenfels, Nachsorge bei Dr. Albrecht.'
].join('\n')

const fake = (answer: unknown): { chat: NerChat; calls: { system: string; user: string }[] } => {
  const calls: { system: string; user: string }[] = []
  return {
    calls,
    chat: async p => {
      calls.push(p)
      return answer
    }
  }
}

describe('findCandidates', () => {
  test('findet Namen, zählt Vorkommen, sortiert nach Häufigkeit, Kontexte ≤ 80 Zeichen', async () => {
    const { chat } = fake({
      persons: ['Mia', 'Hellwig', 'Ostertag'],
      places: ['Rotenfels'],
      institutions: ['Realschule Birkenhain'],
      dates: []
    })
    const out = await findCandidates(TEXT, { chat })
    expect(out[0]).toMatchObject({ text: 'Mia', kind: 'NAME_EXT', count: 2 })
    expect(out.map(c => c.text).sort()).toEqual(['Hellwig', 'Mia', 'Ostertag', 'Realschule Birkenhain', 'Rotenfels'])
    expect(out.find(c => c.text === 'Rotenfels')?.kind).toBe('LOCATION_OTHER')
    expect(out.find(c => c.text === 'Realschule Birkenhain')?.kind).toBe('LOCATION_ORGANIZATION')
    for (const c of out) {
      expect(c.contexts.length).toBeGreaterThan(0)
      for (const ctx of c.contexts) {
        expect(ctx.length).toBeLessThanOrEqual(80)
        expect(ctx).toContain(c.text)
      }
    }
  })

  test('Halluzinationsfilter: nicht im Text vorhandene Vorschläge entfallen', async () => {
    const { chat } = fake({
      persons: ['Mia', 'Zorbas', 'Mia Zorbas', 'Mi'],
      places: ['Rotenfels', 'Rotenfeld'],
      institutions: [],
      dates: ['01.01.1999']
    })
    const out = await findCandidates(TEXT, { chat })
    expect(out.map(c => c.text).sort()).toEqual(['Mia', 'Rotenfels'])
  })

  test('Teilwort ist kein Treffer (ganze Wörter), Genitiv-s zählt mit', async () => {
    const { chat } = fake({ persons: ['Mi', 'Mik', 'Mikad'], places: [], institutions: [], dates: [] })
    expect(await findCandidates('Mia kam. Mias Tasche. Mikado.', { chat })).toEqual([])
    const { chat: c2 } = fake({ persons: ['Mia'], places: [], institutions: [], dates: [] })
    const out = await findCandidates('Mia kam. Mias Tasche. Mikado.', { chat: c2 })
    expect(out[0]).toMatchObject({ text: 'Mia', count: 2 })
  })

  test('Dedupe case-insensitiv über Arten und Wiederholungen; Schreibweise aus dem Text', async () => {
    const { chat } = fake({
      persons: ['Mia', 'mia', 'MIA', 'Frau Hellwig', 'Hellwig'],
      places: ['Mia'],
      institutions: [],
      dates: []
    })
    const out = await findCandidates(TEXT, { chat })
    expect(out.filter(c => c.text === 'Mia')).toHaveLength(1)
    expect(out.find(c => c.text === 'Mia')?.kind).toBe('NAME_EXT')
    expect(out.filter(c => /hellwig/i.test(c.text))).toHaveLength(1)
    expect(out.find(c => /hellwig/i.test(c.text))?.text).toBe('Hellwig')
    expect(out).toHaveLength(2)
  })

  test('Stoppliste: Rollen, Gattungswörter, Wochentage, Monate', async () => {
    const { chat } = fake({
      persons: ['Mutter', 'Vater', 'Patientin', 'Bezugspflege', 'Therapeutin', 'Mia'],
      places: ['Klinik', 'Montag', 'Januar'],
      institutions: ['Klinik', 'Schule', 'Realschule'],
      dates: []
    })
    const out = await findCandidates(TEXT, { chat })
    expect(out.map(c => c.text)).toEqual(['Mia'])
  })

  test('Platzhalter werden ignoriert, auch wenn das Modell sie ausgibt', async () => {
    const { chat, calls } = fake({
      persons: ['{{vorname}}', 'vorname', 'km', '{{km}}', 'Mia'],
      places: [],
      institutions: [],
      dates: []
    })
    const out = await findCandidates(TEXT, { chat })
    expect(out.map(c => c.text)).toEqual(['Mia'])
    expect(calls[0]!.system).toContain('{{')
    expect(calls[0]!.system.toLowerCase()).toContain('platzhalter')
    // Platzhalter-Hinweis lässt sich abschalten
    const f2 = fake({ persons: [], places: [], institutions: [], dates: [] })
    await findCandidates('Mia kam.', { knownPlaceholdersText: false, chat: f2.chat })
    expect(f2.calls[0]!.system).not.toContain('Hinweis: Der Text enthält Platzhalter')
  })

  test('placeholderPattern eines anderen Styles maskiert dessen Platzhalter', async () => {
    const text = 'Treffen mit <NAME_EXT_1a2b3c4d> und Mia.'
    const { chat } = fake({ persons: ['NAME_EXT_1a2b3c4d', 'Mia'], places: [], institutions: [], dates: [] })
    const out = await findCandidates(text, { chat, placeholderPattern: /<[A-Z_]+_[0-9a-f]{8}>/g })
    expect(out.map(c => c.text)).toEqual(['Mia'])
  })

  test('Geburtsdatum nur in Datumsform', async () => {
    const { chat } = fake({ persons: [], places: [], institutions: [], dates: ['12.3.2010', 'Mittwoch'] })
    const out = await findCandidates('Geb. 12.3.2010, Termin am Mittwoch.', { chat })
    expect(out).toEqual([{ text: '12.3.2010', kind: 'DATE', count: 1, contexts: ['Geb. 12.3.2010, Termin am Mittwoch.'] }])
  })

  test('Chunking: mehrere Aufrufe, Ergebnisse werden gemergt, Überlappung zählt nicht doppelt', async () => {
    const filler = 'Ein ruhiger Tag ohne Besonderheiten in der Gruppe. '.repeat(8) // ~400 Zeichen
    const text = `Mia kam früh.\n${filler}\nMitte: Hellwig sprach mit Mia.\n${filler}\nEnde: Ostertag und Mia.`
    const seen: string[] = []
    const chat: NerChat = async p => {
      seen.push(p.user)
      // Das Modell sieht nur seinen Chunk.
      const persons = ['Mia', 'Hellwig', 'Ostertag'].filter(n => p.user.includes(n))
      return { persons, places: [], institutions: [], dates: [] }
    }
    const out = await findCandidates(text, { chat, chunkChars: 600, overlapChars: 100 })
    expect(seen.length).toBeGreaterThan(1)
    expect(out.map(c => [c.text, c.count])).toEqual([
      ['Mia', 3],
      ['Hellwig', 1],
      ['Ostertag', 1]
    ])
  })

  test('Chunks decken den gesamten Text ab', async () => {
    const text = Array.from({ length: 50 }, (_, i) => `Zeile ${i} mit etwas Text darin.`).join('\n')
    const seen: string[] = []
    const chat: NerChat = async p => {
      seen.push(p.user)
      return { persons: [], places: [], institutions: [], dates: [] }
    }
    await findCandidates(text, { chat, chunkChars: 300, overlapChars: 60 })
    for (let i = 0; i < 50; i++) expect(seen.some(s => s.includes(`Zeile ${i} mit`))).toBe(true)
  })

  test('fehlende Schlüssel -> leer; ungültige Antwort wirft', async () => {
    const ok = await findCandidates(TEXT, { chat: async () => ({ persons: ['Mia'] }) })
    expect(ok.map(c => c.text)).toEqual(['Mia'])
    await expect(findCandidates(TEXT, { chat: async () => ({ persons: 'Mia' }) })).rejects.toThrow(/ungültig/)
    await expect(
      findCandidates(TEXT, {
        chat: async () => {
          throw new Error('offline')
        }
      })
    ).rejects.toThrow('offline')
  })
})

describe('toRules', () => {
  const S: Stammdaten = { vorname: 'Lena', nachname: 'Sommerfeld', koseformen: ['Leni'] }

  test('angenommene Kandidaten werden Regeln mit eindeutigen Platzhaltern', async () => {
    const base = buildRules(S)
    const { chat } = fake({
      persons: ['Mia', 'Hellwig'],
      places: ['Rotenfels'],
      institutions: ['Realschule Birkenhain'],
      dates: []
    })
    const cands = await findCandidates(TEXT, { chat })
    const rules = toRules(
      cands,
      [
        { text: 'Mia', kind: 'NAME_EXT' },
        { text: 'hellwig', kind: 'NAME_EXT' },
        { text: 'Rotenfels', kind: 'LOCATION_OTHER' },
        { text: 'Realschule Birkenhain', kind: 'LOCATION_ORGANIZATION' }
      ],
      base,
      semanticStyle
    )
    expect(rules.slice(0, base.length)).toEqual(base)
    const added = rules.slice(base.length)
    expect(added.map(r => r.placeholder)).toEqual(['{{person1}}', '{{person2}}', '{{ort1}}', '{{institution1}}'])
    expect(added.every(r => r.source === 'llm')).toBe(true)
    expect(added[1]!.original).toBe('Hellwig')

    const out = applyRules('Mia schrieb Mias Mutter. Frau Hellwig in Rotenfels, Realschule Birkenhain.', rules).text
    expect(out).toBe('{{person1}} schrieb {{person1}}s Mutter. Frau {{person2}} in {{ort1}}, {{institution1}}.')
  })

  test('manuell ergänzte Einträge, doppelte Annahme, Platzhalter-Fortsetzung bei vorhandenen Regeln', () => {
    const base = buildRules(S)
    const first = toRules([], [{ text: 'Mia', kind: 'NAME_EXT' }], base, semanticStyle)
    expect(first[first.length - 1]).toMatchObject({ placeholder: '{{person1}}', source: 'manual' })
    const second = toRules([], [{ text: 'Mia', kind: 'NAME_EXT' }, { text: 'Tom', kind: 'NAME_EXT' }], first, semanticStyle)
    expect(second).toHaveLength(first.length + 1)
    expect(second[second.length - 1]!.placeholder).toBe('{{person2}}')
    // Eingabe unverändert
    expect(first).toHaveLength(base.length + 1)
  })
})

describe('assertLocalUrl', () => {
  test.each([
    'http://localhost:8000/v1',
    'http://127.0.0.1:8000/v1',
    'http://[::1]:8000/v1',
    'http://llm-box.local/v1',
    'http://10.0.0.5:11434/v1',
    'http://192.168.1.2/v1',
    'http://172.20.0.1/v1'
  ])('akzeptiert %s', url => {
    expect(assertLocalUrl(url)).toBeInstanceOf(URL)
  })

  test.each(['https://api.openai.com/v1', 'http://172.32.0.1/v1', 'http://172.15.0.1/v1', 'http://8.8.8.8/v1'])(
    'lehnt %s ab',
    url => {
      expect(() => assertLocalUrl(url)).toThrow(/^LLM-Endpunkt muss lokal sein: /)
    }
  )

  test('Fehlermeldung nennt den Host', () => {
    expect(() => assertLocalUrl('https://api.openai.com/v1')).toThrow('LLM-Endpunkt muss lokal sein: api.openai.com')
  })
})

describe('extractJson', () => {
  test('rohes JSON, Zaun, <think>-Block, Text drumherum', () => {
    expect(extractJson('{"a":1}')).toEqual({ a: 1 })
    expect(extractJson('```json\n{"a":2}\n```')).toEqual({ a: 2 })
    expect(extractJson('<think>{"x":0}</think>{"a":3}')).toEqual({ a: 3 })
    expect(extractJson('Hier: {"a":"}"} Ende')).toEqual({ a: '}' })
    expect(() => extractJson('kein json')).toThrow(/Kein JSON/)
  })
})

describe('openAiCompatibleChat', () => {
  type Call = { url: string; init: RequestInit }
  const stub = (res: Response): { fetch: typeof fetch; calls: Call[] } => {
    const calls: Call[] = []
    const impl = async (url: string | URL | Request, init?: RequestInit) => {
      calls.push({ url: String(url), init: init ?? {} })
      return res
    }
    return { fetch: impl as unknown as typeof fetch, calls }
  }
  const reply = (content: string): Response =>
    new Response(JSON.stringify({ choices: [{ message: { content } }] }), { status: 200 })

  test('sendet JSON-Modus-Anfrage und parst choices[0].message.content', async () => {
    const { fetch: f, calls } = stub(reply('<think>x</think>```json\n{"persons":["Mia"]}\n```'))
    const chat = openAiCompatibleChat({ baseUrl: 'http://127.0.0.1:8000/v1/', model: 'test-model', timeoutMs: 5000, fetch: f })
    const out = await chat({ system: 'SYS', user: 'USR' })
    expect(out).toEqual({ persons: ['Mia'] })
    expect(calls).toHaveLength(1)
    expect(calls[0]!.url).toBe('http://127.0.0.1:8000/v1/chat/completions')
    expect(calls[0]!.init.method).toBe('POST')
    const body = JSON.parse(String(calls[0]!.init.body))
    expect(body).toEqual({
      model: 'test-model',
      messages: [
        { role: 'system', content: 'SYS' },
        { role: 'user', content: 'USR' }
      ],
      response_format: { type: 'json_object' },
      temperature: 0
    })
    expect(calls[0]!.init.signal).toBeInstanceOf(AbortSignal)
    expect((calls[0]!.init.headers as Record<string, string>).authorization).toBeUndefined()
  })

  test('Authorization-Header nur mit apiKey', async () => {
    const { fetch: f, calls } = stub(reply('{}'))
    const chat = openAiCompatibleChat({ baseUrl: 'http://localhost:11434/v1', model: 'm', apiKey: 'sk-test', timeoutMs: 1000, fetch: f })
    await chat({ system: 's', user: 'u' })
    expect((calls[0]!.init.headers as Record<string, string>).authorization).toBe('Bearer sk-test')
  })

  test('nicht-2xx wirft mit Status; fehlender Inhalt wirft', async () => {
    const bad = stub(new Response('boom', { status: 503 }))
    await expect(
      openAiCompatibleChat({ baseUrl: 'http://localhost:1/v1', model: 'm', timeoutMs: 1000, fetch: bad.fetch })({ system: 's', user: 'u' })
    ).rejects.toThrow(/503/)
    const empty = stub(new Response(JSON.stringify({ choices: [] }), { status: 200 }))
    await expect(
      openAiCompatibleChat({ baseUrl: 'http://localhost:1/v1', model: 'm', timeoutMs: 1000, fetch: empty.fetch })({ system: 's', user: 'u' })
    ).rejects.toThrow(/choices/)
  })

  test('lehnt nicht-lokale Endpunkte schon bei der Konstruktion ab', () => {
    expect(() => openAiCompatibleChat({ baseUrl: 'https://api.openai.com/v1', model: 'm', timeoutMs: 1000 })).toThrow(
      'LLM-Endpunkt muss lokal sein: api.openai.com'
    )
  })
})

describe('llmDetector', () => {
  test('ein Span je Vorkommen mit korrekten Offsets (Genitiv-s ausgeschlossen)', async () => {
    const text = 'Mia kam früh. Später sagte {{km}} zu Mia: Mias Tasche, Mikado.'
    const { chat } = fake({ persons: ['Mia'], places: [], institutions: [], dates: [] })
    const det = llmDetector({ chat })
    expect(det.name).toBe('llm')
    const masked = maskPlaceholders(text, semanticStyle.pattern)
    const spans = await det.detect(text, { masked })
    const first = text.indexOf('Mia')
    const second = text.indexOf('Mia', text.indexOf('zu '))
    const third = text.indexOf('Mias')
    expect(spans.map(s => s.start)).toEqual([first, second, third])
    for (const s of spans) {
      expect(s.end - s.start).toBe(3)
      expect(text.slice(s.start, s.end)).toBe('Mia')
      expect(s).toMatchObject({ text: 'Mia', kind: 'NAME_EXT', score: 0.7, detector: 'llm' })
    }
  })

  test('Platzhalter erzeugen keine Spans; Kinds je Kategorie', async () => {
    const text = 'Frau {{km}} aus Rotenfels, Realschule Birkenhain, geb. 12.3.2010.'
    const { chat } = fake({
      persons: ['km'],
      places: ['Rotenfels'],
      institutions: ['Realschule Birkenhain'],
      dates: ['12.3.2010']
    })
    const spans = await llmDetector({ chat }).detect(text, { masked: maskPlaceholders(text, semanticStyle.pattern) })
    expect(spans.map(s => [s.text, s.kind, s.start])).toEqual([
      ['Rotenfels', 'LOCATION_OTHER', text.indexOf('Rotenfels')],
      ['Realschule Birkenhain', 'LOCATION_ORGANIZATION', text.indexOf('Realschule')],
      ['12.3.2010', 'DATE', text.indexOf('12.3.2010')]
    ])
  })
})
