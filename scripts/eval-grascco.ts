// Evaluate detectors against GraSCCo_PII_V2 (synthetic German clinical text, CC-BY-4.0).
// Repo-only; not shipped. Prints numbers and document ids only, never corpus text.
//
//   bun scripts/eval-grascco.ts [--corpus <dir>] [--config <path>] [--detectors regex,onnx,llm]
//                               [--limit <n>] [--skip-labels NAME_TITLE,...] [--json]
//
// Corpus: INCEpTION CAS JSON export (Zenodo 15747389), unzipped to eval/corpus/json/grascco_pii_2_json/.
import fs from 'node:fs'
import path from 'node:path'
import { parseArgs } from 'node:util'
import { fileURLToPath } from 'node:url'
import { loadConfig } from '../src/config.ts'
import { llmDetector, openAiCompatibleChat } from '../src/detectors/llm.ts'
import { onnxDetector } from '../src/detectors/onnx.ts'
import { regexDetector } from '../src/detectors/regex.ts'
import type { Detector } from '../src/detectors/types.ts'
import { detect } from '../src/pipeline.ts'
import type { KindGroup, Span } from '../src/types.ts'
import { KIND_GROUPS, parseCasJson, scoreAll, type GraSCCoDoc, type Metrics, type PredSpan, type ScoreResult } from './grascco.ts'

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const DEFAULT_CORPUS = path.join(ROOT, 'eval/corpus/json/grascco_pii_2_json')
const KNOWN_DETECTORS = ['regex', 'onnx', 'llm'] as const
type EvalDetector = (typeof KNOWN_DETECTORS)[number]

const { values: args } = parseArgs({
  options: {
    corpus: { type: 'string' },
    config: { type: 'string' },
    detectors: { type: 'string', default: 'regex,onnx' },
    json: { type: 'boolean', default: false },
    limit: { type: 'string' },
    'skip-labels': { type: 'string', default: '' },
  },
})

function fail(msg: string): never {
  console.error(msg)
  process.exit(1)
}

const wanted = (args.detectors ?? '').split(',').map((s) => s.trim()).filter(Boolean)
const unknown = wanted.filter((d) => !(KNOWN_DETECTORS as readonly string[]).includes(d))
if (unknown.length || wanted.length === 0) {
  fail(`Unbekannte Detektoren: ${unknown.join(', ') || '(leer)'}. Erlaubt: ${KNOWN_DETECTORS.join(', ')}`)
}
const detectorNames = KNOWN_DETECTORS.filter((d) => wanted.includes(d))

const limit = args.limit === undefined ? undefined : Number(args.limit)
if (limit !== undefined && (!Number.isInteger(limit) || limit < 1)) fail(`--limit muss eine positive ganze Zahl sein: ${args.limit}`)

const corpusDir = args.corpus ? path.resolve(args.corpus) : DEFAULT_CORPUS
if (!fs.existsSync(corpusDir)) {
  fail(`Korpus nicht gefunden: ${corpusDir}\nGraSCCo_PII_V2 von Zenodo (Record 15747389) laden und nach eval/corpus/ entpacken.`)
}
const skipLabels = new Set((args['skip-labels'] ?? '').split(',').map((s) => s.trim()).filter(Boolean))

const files = fs.readdirSync(corpusDir).filter((f) => f.endsWith('.json')).sort()
if (files.length === 0) fail(`Keine .json-Dateien in ${corpusDir}`)

const docs: (GraSCCoDoc & { file: string })[] = files.slice(0, limit).map((file) => {
  const doc = parseCasJson(JSON.parse(fs.readFileSync(path.join(corpusDir, file), 'utf8')))
  return { ...doc, id: doc.id || file, file, gold: doc.gold.filter((g) => !skipLabels.has(g.label)) }
})

const { config, sources } = loadConfig({ path: args.config })

/** Memoises one detector per document text, so "alone" and "combined" runs share one inference pass. */
function cached(inner: Detector): Detector {
  const memo = new Map<string, Promise<Span[]>>()
  return {
    name: inner.name,
    detect(text, ctx) {
      let hit = memo.get(text)
      if (!hit) {
        hit = inner.detect(text, ctx)
        memo.set(text, hit)
      }
      return hit
    },
  }
}

function build(name: EvalDetector): Detector {
  switch (name) {
    case 'regex':
      return regexDetector({ entities: config.detectors.regex.entities })
    case 'onnx': {
      const { enabled: _enabled, ...onnx } = config.detectors.onnx
      return onnxDetector(onnx)
    }
    case 'llm': {
      const llm = config.detectors.llm
      const chat = openAiCompatibleChat({
        baseUrl: llm.baseUrl,
        model: llm.model,
        timeoutMs: llm.timeoutMs,
        apiKey: process.env[llm.apiKeyEnv],
      })
      return llmDetector({ chat, chunkChars: llm.chunkChars, overlapChars: llm.overlapChars })
    }
  }
}

const built = detectorNames.map((n) => ({ name: n, detector: cached(build(n)) }))

type Setting = { name: string; detectors: Detector[] }
const settings: Setting[] = built.map((b) => ({ name: b.name, detectors: [b.detector] }))
if (built.length > 1) settings.push({ name: detectorNames.join('+'), detectors: built.map((b) => b.detector) })

type SettingResult = { name: string; strict: ScoreResult; overlap: ScoreResult }
const results: SettingResult[] = []

// Review kinds (AGE, PROFESSION, ...) are detected but not replaced; they still count as predictions.
async function predict(doc: GraSCCoDoc, setting: Setting): Promise<PredSpan[]> {
  const { spans, review } = await detect(doc.text, { config, detectors: setting.detectors, markdown: false })
  return [...spans, ...review].map((s) => ({ start: s.start, end: s.end, kind: s.kind }))
}

for (const setting of settings) {
  const items: { gold: GraSCCoDoc['gold']; predicted: PredSpan[] }[] = []
  try {
    for (const doc of docs) {
      items.push({ gold: doc.gold, predicted: await predict(doc, setting) })
      process.stderr.write(`\r${setting.name}: ${items.length}/${docs.length}`)
    }
    process.stderr.write('\n')
  } catch (e) {
    process.stderr.write('\n')
    console.error(`Setting "${setting.name}" übersprungen: ${e instanceof Error ? e.message : String(e)}`)
    continue
  }
  results.push({
    name: setting.name,
    strict: scoreAll(items, { mode: 'strict' }),
    overlap: scoreAll(items, { mode: 'overlap' }),
  })
}

if (results.length === 0) fail('Kein Setting konnte ausgewertet werden.')

const pct = (x: number) => (x * 100).toFixed(1)
const present = (r: ScoreResult, g: KindGroup) => r.groups[g].gold > 0 || r.groups[g].predicted > 0

function row(label: string, strict: Metrics, overlap: Metrics): string {
  return `| ${label} | ${strict.gold} | ${pct(strict.precision)} | ${pct(strict.recall)} | ${pct(strict.f1)} | ${pct(overlap.recall)} | ${strict.tp} | ${strict.fp} | ${strict.fn} |`
}

const out: string[] = []
out.push(`# GraSCCo_PII_V2 – ${docs.length} Dokumente, ${docs.reduce((n, d) => n + d.gold.length, 0)} Gold-Spans`)
if (skipLabels.size) out.push(`\nAusgeschlossene Gold-Labels: ${[...skipLabels].join(', ')}`)
out.push(`\nPrädiktionen = spans + review. Config-Quellen: ${sources.join(', ')}`)
for (const r of results) {
  out.push(`\n## ${r.name}\n`)
  out.push('| Gruppe | Gold | P strict | R strict | F1 strict | R overlap | TP | FP | FN |')
  out.push('|---|---:|---:|---:|---:|---:|---:|---:|---:|')
  for (const g of KIND_GROUPS) if (present(r.strict, g)) out.push(row(g, r.strict.groups[g], r.overlap.groups[g]))
  out.push(row('**gesamt**', r.strict.total, r.overlap.total))
}
out.push('\n## Recall (overlap) je Gruppe und Setting\n')
out.push(`| Gruppe | Gold | ${results.map((r) => r.name).join(' | ')} |`)
out.push(`|---|---:|${results.map(() => '---:').join('|')}|`)
const first = results[0] as SettingResult
for (const g of KIND_GROUPS) {
  if (!results.some((r) => present(r.strict, g))) continue
  out.push(`| ${g} | ${first.strict.groups[g].gold} | ${results.map((r) => pct(r.overlap.groups[g].recall)).join(' | ')} |`)
}
out.push(`| **gesamt** | ${first.strict.total.gold} | ${results.map((r) => pct(r.overlap.total.recall)).join(' | ')} |`)
console.log(out.join('\n'))

if (args.json) {
  const date = new Date().toISOString().slice(0, 10)
  const file = path.join(ROOT, 'eval', `results-${date}.json`)
  const { enabled: _e, cacheDir: _c, ...onnxInfo } = config.detectors.onnx
  fs.mkdirSync(path.dirname(file), { recursive: true })
  fs.writeFileSync(
    file,
    JSON.stringify(
      {
        date,
        corpus: path.relative(ROOT, corpusDir),
        documents: docs.map((d) => d.id),
        skipLabels: [...skipLabels],
        configSources: sources,
        onnx: { model: onnxInfo.model, variant: onnxInfo.variant, threshold: onnxInfo.threshold },
        settings: results,
      },
      null,
      2,
    ) + '\n',
  )
  console.error(`→ ${path.relative(process.cwd(), file)}`)
}
