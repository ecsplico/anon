import { describe, expect, test } from 'bun:test'
import { mkdtempSync, mkdirSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import path from 'node:path'
import * as openmed from 'openmed'
import { chunkRanges, ONNX_DEFAULTS, onnxCacheStatus, onnxDetector } from '../src/detectors/onnx.ts'
import type { OpenmedModule, OpenmedSpan } from '../src/detectors/onnx.ts'
import { ENTITY_KINDS } from '../src/types.ts'
import type { EntityKind } from '../src/types.ts'

type Label = { label: string; surface: string; score?: number; all?: boolean }

/** Fake token-classification pipeline: one `B-<label>` token per surface occurrence, with offsets into the chunk it is given. */
function fakePipeline(labels: Label[]) {
  const calls: string[] = []
  const pipeline = async (text: string) => {
    calls.push(text)
    const tokens: { entity: string; score: number; index: number; word: string; start: number; end: number }[] = []
    for (const { label, surface, score, all } of labels) {
      let from = 0
      for (;;) {
        const i = text.indexOf(surface, from)
        if (i < 0) break
        tokens.push({ entity: `B-${label}`, score: score ?? 0.9, index: 0, word: surface.toLowerCase(), start: i, end: i + surface.length })
        from = i + surface.length
        if (!all) break
      }
    }
    tokens.sort((a, b) => a.start - b.start)
    // Non-contiguous indices: the decoder must not glue neighbouring tokens together.
    tokens.forEach((t, k) => { t.index = 2 * k + 1 })
    return tokens
  }
  return { pipeline, calls }
}

const ALL_KINDS: EntityKind[] = [...ENTITY_KINDS]

async function run(text: string, labels: Label[], opts: Parameters<typeof onnxDetector>[0] = {}) {
  const { pipeline } = fakePipeline(labels)
  return onnxDetector({ ...opts, loader: async () => openmed as unknown as OpenmedModule, pipeline }).detect(text, { masked: text })
}

describe('kind mapping', () => {
  const table: [string, string, EntityKind][] = [
    ['FIRSTNAME', 'Anna', 'NAME_EXT'],
    ['PERSON', 'Anna', 'NAME_EXT'],
    ['ACCOUNTNAME', 'Anna', 'NAME_EXT'],
    ['DATEOFBIRTH', '12.03.2011', 'DATE_BIRTH'],
    ['DATE', '12.03.2011', 'DATE'],
    ['AGE', '14 Jahre', 'AGE'],
    ['EMAIL', 'a@example.de', 'CONTACT_EMAIL'],
    ['PHONE', '+49 30 1234567', 'CONTACT_PHONE'],
    ['URL', 'www.example.de', 'CONTACT_URL'],
    ['IPADDRESS', '10.0.0.1', 'CONTACT_IP'],
    ['CITY', 'Berlin', 'LOCATION_CITY'],
    ['STATE', 'Bayern', 'LOCATION_STATE'],
    ['COUNTY', 'Landkreis', 'LOCATION_STATE'],
    ['STREET', 'Hauptstraße', 'LOCATION_STREET'],
    ['BUILDINGNUMBER', '12a', 'LOCATION_STREET'],
    ['ZIPCODE', '10115', 'LOCATION_ZIP'],
    ['GPSCOORDINATES', '52.5200', 'LOCATION_OTHER'],
    ['ORGANIZATION', 'Charité', 'LOCATION_ORGANIZATION'],
    ['JOBDEPARTMENT', 'Pädiatrie', 'LOCATION_ORGANIZATION'],
    ['JOBTITLE', 'Lehrerin', 'PROFESSION'],
    ['OCCUPATION', 'Lehrerin', 'PROFESSION'],
    ['IBAN', 'DE89370400440532013000', 'ID'],
    ['SSN', 'A123456789', 'ID'],
    ['USERNAME', 'amueller', 'ID'],
  ]
  for (const [label, surface, kind] of table) {
    test(`${label} → ${kind}`, async () => {
      const spans = await run(surface, [{ label, surface }], { entities: ALL_KINDS })
      expect(spans).toHaveLength(1)
      expect(spans[0]).toMatchObject({ kind, text: surface, detector: 'onnx', start: 0, end: surface.length })
    })
  }

  for (const label of ['PREFIX', 'GENDER', 'SEX', 'TIME', 'HEIGHT', 'AMOUNT', 'CURRENCY', 'EYECOLOR', 'USERAGENT']) {
    test(`${label} is dropped`, async () => {
      expect(await run('Herr', [{ label, surface: 'Herr' }], { entities: ALL_KINDS })).toEqual([])
    })
  }

  test('canonical_label is the fallback when the raw type is unknown', async () => {
    const fake: OpenmedModule = {
      loadOnnxModel: async () => ({}),
      extractPii: async (): Promise<OpenmedSpan[]> => [
        { start: 0, end: 12, entity_type: 'WEIRD', canonical_label: 'EMAIL', score: 0.9 },
        { start: 13, end: 17, entity_type: 'WEIRD2', canonical_label: 'OTHER', score: 0.9 },
      ],
    }
    const text = 'a@example.de abcd'
    const spans = await onnxDetector({ loader: async () => fake, pipeline: {}, entities: ALL_KINDS }).detect(text, { masked: text })
    expect(spans.map(s => [s.kind, s.text])).toEqual([['CONTACT_EMAIL', 'a@example.de']])
  })
})

describe('post-processing', () => {
  test('FIRSTNAME + LASTNAME separated by whitespace merge into one NAME_EXT span (score = min)', async () => {
    const text = 'Die Patientin Anna Müller kam.'
    const spans = await run(text, [
      { label: 'FIRSTNAME', surface: 'Anna', score: 0.9 },
      { label: 'LASTNAME', surface: 'Müller', score: 0.6 },
    ])
    expect(spans).toHaveLength(1)
    expect(spans[0]).toMatchObject({ kind: 'NAME_EXT', text: 'Anna Müller', start: 14, end: 25 })
    expect(spans[0]!.score).toBeCloseTo(0.6, 5)
  })

  test('first and last name on different lines do not merge', async () => {
    const text = 'Anna\nMüller'
    const spans = await run(text, [
      { label: 'FIRSTNAME', surface: 'Anna' },
      { label: 'LASTNAME', surface: 'Müller' },
    ])
    expect(spans.map(s => s.text)).toEqual(['Anna', 'Müller'])
  })

  test('name spans extend to the surrounding word (truncated surname)', async () => {
    const text = 'Frau Lena Sommerfeld kam.'
    const spans = await run(text, [
      { label: 'FIRSTNAME', surface: 'Lena' },
      { label: 'LASTNAME', surface: 'Somme' },
    ])
    expect(spans).toHaveLength(1)
    expect(spans[0]!.text).toBe('Lena Sommerfeld')
  })

  test('non-name spans are not extended', async () => {
    const text = 'Berlinerstraße'
    const spans = await run(text, [{ label: 'CITY', surface: 'Berlin' }])
    expect(spans[0]!.text).toBe('Berlin')
  })

  test('date fragments separated by "." merge into one DATE span', async () => {
    const text = 'Geboren am 12.03.2011 in Wien.'
    const spans = await run(text, [
      { label: 'DATE', surface: '12' },
      { label: 'DATE', surface: '03' },
      { label: 'DATE', surface: '2011' },
    ])
    expect(spans).toHaveLength(1)
    expect(spans[0]).toMatchObject({ kind: 'DATE', text: '12.03.2011' })
  })

  test('two separate dates stay separate', async () => {
    const text = 'vom 12.03.2011 bis 14.03.2011'
    const spans = await run(text, [
      { label: 'DATE', surface: '12.03.2011' },
      { label: 'DATE', surface: '14.03.2011' },
    ])
    expect(spans.map(s => s.text)).toEqual(['12.03.2011', '14.03.2011'])
  })

  test('trailing punctuation and whitespace are trimmed; Span.text is sliced from the original text', async () => {
    const text = 'Frau Müller, kam.'
    const masked = text
    const { pipeline } = fakePipeline([{ label: 'LASTNAME', surface: 'Müller,' }])
    const spans = await onnxDetector({ loader: async () => openmed as unknown as OpenmedModule, pipeline }).detect(text, { masked })
    expect(spans).toHaveLength(1)
    expect(spans[0]).toMatchObject({ text: 'Müller', start: 5, end: 11 })
  })

  test('placeholders masked to spaces are not reported; offsets index the original text', async () => {
    const text = 'Frau {{km}} und Anna Müller.'
    const masked = text.replace('{{km}}', ' '.repeat('{{km}}'.length))
    const { pipeline } = fakePipeline([
      { label: 'FIRSTNAME', surface: 'Anna' },
      { label: 'LASTNAME', surface: 'Müller' },
    ])
    const spans = await onnxDetector({ loader: async () => openmed as unknown as OpenmedModule, pipeline }).detect(text, { masked })
    expect(spans.map(s => s.text)).toEqual(['Anna Müller'])
    expect(text.slice(spans[0]!.start, spans[0]!.end)).toBe('Anna Müller')
  })
})

describe('filters', () => {
  const text = 'Anna Müller wohnt in Berlin.'
  const labels: Label[] = [
    { label: 'FIRSTNAME', surface: 'Anna', score: 0.9 },
    { label: 'LASTNAME', surface: 'Müller', score: 0.9 },
    { label: 'CITY', surface: 'Berlin', score: 0.4 },
  ]

  test('threshold drops spans whose mean score is below it', async () => {
    expect((await run(text, labels, { threshold: 0.5 })).map(s => s.text)).toEqual(['Anna Müller'])
    expect((await run(text, labels, { threshold: 0.3 })).map(s => s.text)).toEqual(['Anna Müller', 'Berlin'])
  })

  test('entities filter drops kinds that are not configured', async () => {
    const onlyCity = await run(text, labels, { threshold: 0.3, entities: ['LOCATION_CITY'] })
    expect(onlyCity.map(s => [s.kind, s.text])).toEqual([['LOCATION_CITY', 'Berlin']])
    const onlyName = await run(text, labels, { threshold: 0.3, entities: ['NAME_EXT'] })
    expect(onlyName.map(s => s.kind)).toEqual(['NAME_EXT'])
  })

  test('default entity list covers names, dates, locations, contact, ID', () => {
    expect(ONNX_DEFAULTS.entities).toContain('NAME_EXT')
    expect(ONNX_DEFAULTS.entities).toContain('LOCATION_CITY')
    expect(ONNX_DEFAULTS.chunkChars).toBe(900)
  })
})

describe('chunking', () => {
  test('chunkRanges packs paragraphs under the limit and splits oversized ones', () => {
    const text = ['Erster Absatz mit Anna.', 'Zweiter Absatz mit Jane.', 'Dritter Absatz.'].join('\n\n')
    const ranges = chunkRanges(text, 50)
    expect(ranges.length).toBeGreaterThan(1)
    for (const r of ranges) expect(r.end - r.start).toBeLessThanOrEqual(50)

    const long = 'Satz eins ist hier. Satz zwei ist auch hier. Satz drei folgt noch. Satz vier schließt ab.'
    const sentenceChunks = chunkRanges(long, 40)
    expect(sentenceChunks.length).toBeGreaterThan(1)
    for (const r of sentenceChunks) {
      expect(r.end - r.start).toBeLessThanOrEqual(40)
      expect(long.slice(r.start, r.end).trim().endsWith('.')).toBe(true)
    }

    const blob = 'x'.repeat(250)
    const hard = chunkRanges(blob, 100)
    expect(hard.map(r => r.end - r.start)).toEqual([100, 100, 50])
  })

  test('whitespace-only text yields no chunks and no pipeline call', async () => {
    const { pipeline, calls } = fakePipeline([])
    const spans = await onnxDetector({ loader: async () => openmed as unknown as OpenmedModule, pipeline }).detect('   \n\n  ', { masked: '   \n\n  ' })
    expect(spans).toEqual([])
    expect(calls).toHaveLength(0)
  })

  test('multi-paragraph text is chunked and span offsets are shifted back to document positions', async () => {
    const paragraphs = [
      'Die Patientin Anna Müller wurde aufgenommen.',
      'Der Vater Thomas Schneider berichtet vom Verlauf.',
      'Die Schwester Lena Sommerfeld war nicht anwesend.',
      'Zum Schluss sprach Jane Doe mit dem Team.',
    ]
    const text = paragraphs.join('\n\n')
    const { pipeline, calls } = fakePipeline([
      { label: 'FIRSTNAME', surface: 'Anna' }, { label: 'LASTNAME', surface: 'Müller' },
      { label: 'FIRSTNAME', surface: 'Thomas' }, { label: 'LASTNAME', surface: 'Schneider' },
      { label: 'FIRSTNAME', surface: 'Lena' }, { label: 'LASTNAME', surface: 'Sommerfeld' },
      { label: 'FIRSTNAME', surface: 'Jane' }, { label: 'LASTNAME', surface: 'Doe' },
    ])
    const spans = await onnxDetector({
      loader: async () => openmed as unknown as OpenmedModule,
      pipeline,
      chunkChars: 100,
    }).detect(text, { masked: text })

    expect(calls.length).toBeGreaterThan(1)
    for (const c of calls) expect(c.length).toBeLessThanOrEqual(100)
    expect(spans.map(s => s.text)).toEqual(['Anna Müller', 'Thomas Schneider', 'Lena Sommerfeld', 'Jane Doe'])
    for (const s of spans) {
      expect(s.start).toBe(text.indexOf(s.text))
      expect(text.slice(s.start, s.end)).toBe(s.text)
    }
  })
})

describe('loading', () => {
  test('missing packages: rejecting loader yields the German error naming the config switch', async () => {
    const det = onnxDetector({ loader: async () => { throw new Error('Cannot find module') } })
    const err = await det.detect('Anna', { masked: 'Anna' }).catch((e: Error) => e)
    expect(err).toBeInstanceOf(Error)
    expect((err as Error).message).toContain('Pakete "openmed" und "@huggingface/transformers" fehlen')
    expect((err as Error).message).toContain('detectors.onnx.enabled=false')
  })

  test('openmed\'s "Install @huggingface/transformers" error maps to the same message', async () => {
    const fake: OpenmedModule = {
      extractPii: async () => [],
      loadOnnxModel: async () => { throw new Error('Install @huggingface/transformers or pass a token-classification pipeline.') },
    }
    const cacheDir = mkdtempSync(path.join(tmpdir(), 'anon-onnx-'))
    const err = await onnxDetector({ loader: async () => fake, cacheDir }).detect('Anna', { masked: 'Anna' }).catch((e: Error) => e)
    expect((err as Error).message).toContain('detectors.onnx.enabled=false')
  })

  test('localFilesOnly with an uncached model gets a friendly error', async () => {
    const fake: OpenmedModule = {
      extractPii: async () => [],
      loadOnnxModel: async () => { throw new Error('`local_files_only=true` or `env.allowRemoteModels=false` and file was not found locally at "/x/config.json"') },
    }
    const cacheDir = mkdtempSync(path.join(tmpdir(), 'anon-onnx-'))
    const err = await onnxDetector({ loader: async () => fake, cacheDir, localFilesOnly: true }).detect('Anna', { masked: 'Anna' }).catch((e: Error) => e)
    expect((err as Error).message).toBe(`Modell ${ONNX_DEFAULTS.model} nicht im Cache (${cacheDir}); einmal mit localFilesOnly=false laden`)
  })

  test('the model is loaded once and passed to extractPii for every chunk', async () => {
    let loads = 0
    const seen: unknown[] = []
    const model = { id: 'model' }
    const fake: OpenmedModule = {
      loadOnnxModel: async (id, o) => {
        loads++
        expect(id).toBe(ONNX_DEFAULTS.model)
        expect(o).toEqual({ variant: 'int8', device: 'cpu', localFilesOnly: false })
        return model
      },
      extractPii: async (_t, o) => { seen.push(o?.pipeline); return [] },
    }
    const cacheDir = mkdtempSync(path.join(tmpdir(), 'anon-onnx-'))
    const det = onnxDetector({ loader: async () => fake, cacheDir, chunkChars: 20 })
    await det.detect('Erster Absatz.\n\nZweiter Absatz.', { masked: 'Erster Absatz.\n\nZweiter Absatz.' })
    await det.detect('Noch ein Text.', { masked: 'Noch ein Text.' })
    expect(loads).toBe(1)
    expect(seen.length).toBeGreaterThanOrEqual(3)
    expect(seen.every(p => p === model)).toBe(true)
  })
})

describe('onnxCacheStatus', () => {
  test('hit only when config, tokenizer files and the variant file exist', () => {
    const cacheDir = mkdtempSync(path.join(tmpdir(), 'anon-onnx-cache-'))
    const model = 'Org/Some-Model-onnx-android'
    expect(onnxCacheStatus({ model, variant: 'int8', cacheDir })).toBe('missing')

    const dir = path.join(cacheDir, 'Org', 'Some-Model-onnx-android')
    mkdirSync(dir, { recursive: true })
    for (const f of ['config.json', 'tokenizer.json', 'tokenizer_config.json']) writeFileSync(path.join(dir, f), '{}')
    expect(onnxCacheStatus({ model, variant: 'int8', cacheDir })).toBe('missing')

    writeFileSync(path.join(dir, 'model_int8.onnx'), '')
    expect(onnxCacheStatus({ model, variant: 'int8', cacheDir })).toBe('hit')
    expect(onnxCacheStatus({ model, variant: 'fp16', cacheDir })).toBe('missing')

    writeFileSync(path.join(dir, 'model.onnx'), '')
    expect(onnxCacheStatus({ model, variant: 'fp32', cacheDir })).toBe('hit')
  })
})
