// GraSCCo_PII_V2 (INCEpTION UIMA CAS JSON) parser and entity-level scorer. Pure functions, no I/O.
// Repo-only helper for scripts/eval-grascco.ts; not part of the published package.
import type { KindGroup } from '../src/types.ts'

export type GoldSpan = { start: number; end: number; label: string }
export type GraSCCoDoc = { id: string; text: string; gold: GoldSpan[] }

const SOFA = 'uima.cas.Sofa'
const PHI = 'webanno.custom.PHI'
const META_SUFFIX = 'DocumentMetaData'

type FeatureStructure = Record<string, unknown> & { '%TYPE'?: string }

/**
 * Parses one INCEpTION CAS JSON export.
 *
 * Offsets (`begin`/`end`) are UIMA offsets = UTF-16 code units, i.e. JS string indices; no
 * conversion needed. Verified on the real corpus: all 63 documents are BMP-only, so code-point and
 * code-unit counting coincide there, and spans with umlauts are aligned. UIMA JSON omits default
 * values, so a missing `begin` means 0.
 */
export function parseCasJson(json: unknown): GraSCCoDoc {
  const fss = (json as { '%FEATURE_STRUCTURES'?: unknown } | null)?.['%FEATURE_STRUCTURES']
  if (!Array.isArray(fss)) throw new Error('Kein CAS-JSON: %FEATURE_STRUCTURES fehlt')
  const entries = fss as FeatureStructure[]

  const sofa = entries.find((e) => e['%TYPE'] === SOFA)
  if (!sofa || typeof sofa.sofaString !== 'string') throw new Error('Kein CAS-JSON: Sofa/sofaString fehlt')
  const text = sofa.sofaString

  const meta = entries.find((e) => typeof e['%TYPE'] === 'string' && e['%TYPE'].endsWith(META_SUFFIX))
  const id = typeof meta?.documentId === 'string' ? meta.documentId : typeof meta?.documentTitle === 'string' ? meta.documentTitle : ''

  const gold: GoldSpan[] = []
  for (const e of entries) {
    if (e['%TYPE'] !== PHI) continue
    const start = typeof e.begin === 'number' ? e.begin : 0
    const end = typeof e.end === 'number' ? e.end : NaN
    const label = typeof e.kind === 'string' ? e.kind : ''
    if (!Number.isInteger(end) || end <= start || end > text.length) {
      throw new Error(`Ungültiger PHI-Span (${label || '?'}): ${start}-${end} bei Textlänge ${text.length}`)
    }
    gold.push({ start, end, label })
  }
  gold.sort((a, b) => a.start - b.start || a.end - b.end)
  return { id, text, gold }
}

export const KIND_GROUPS: readonly KindGroup[] = ['NAME', 'DATE', 'AGE', 'LOCATION', 'ID', 'CONTACT', 'PROFESSION', 'OTHER']

/** Prefix before `_` (`DATE_BIRTH` -> DATE, `NAME_USERNAME` -> NAME). Unknown prefix -> OTHER; empty label -> null. */
export function labelGroup(label: string): KindGroup | null {
  if (!label) return null
  const i = label.indexOf('_')
  const prefix = i < 0 ? label : label.slice(0, i)
  return (KIND_GROUPS as readonly string[]).includes(prefix) ? (prefix as KindGroup) : 'OTHER'
}

export type PredSpan = { start: number; end: number; kind: string }
export type ScoreMode = 'strict' | 'overlap'

export type Metrics = {
  /** Gold spans in this group. */
  gold: number
  /** Distinct predicted spans in this group. */
  predicted: number
  /** Gold spans matched by a prediction (recall side). */
  tp: number
  /** Predicted spans without any matching gold span. */
  fp: number
  /** Gold spans without any matching prediction. */
  fn: number
  /** Predicted spans matching a gold span (precision side). Equals `tp` in strict mode. */
  predTp: number
  precision: number
  recall: number
  f1: number
}

export type ScoreResult = { groups: Record<KindGroup, Metrics>; total: Metrics }

type Counts = Pick<Metrics, 'gold' | 'predicted' | 'tp' | 'predTp'>

const emptyCounts = (): Counts => ({ gold: 0, predicted: 0, tp: 0, predTp: 0 })

function ratio(n: number, d: number): number {
  return d === 0 ? 0 : n / d
}

function finalize(c: Counts): Metrics {
  const precision = ratio(c.predTp, c.predicted)
  const recall = ratio(c.tp, c.gold)
  return {
    ...c,
    fp: c.predicted - c.predTp,
    fn: c.gold - c.tp,
    precision,
    recall,
    f1: precision + recall === 0 ? 0 : (2 * precision * recall) / (precision + recall),
  }
}

type Group = { start: number; end: number; group: KindGroup }

function toGrouped(spans: { start: number; end: number; kind?: string; label?: string }[]): Group[] {
  const out: Group[] = []
  for (const s of spans) {
    const group = labelGroup(s.label ?? s.kind ?? '')
    if (group) out.push({ start: s.start, end: s.end, group })
  }
  return out
}

/** Identical (group, start, end) predictions count once: several detectors may report the same span. */
function dedupe(spans: Group[]): Group[] {
  const seen = new Set<string>()
  return spans.filter((s) => {
    const k = `${s.group}:${s.start}:${s.end}`
    if (seen.has(k)) return false
    seen.add(k)
    return true
  })
}

function accumulate(into: Record<KindGroup, Counts>, gold: Group[], predicted: Group[], mode: ScoreMode): void {
  const match =
    mode === 'strict'
      ? (a: Group, b: Group) => a.start === b.start && a.end === b.end
      : (a: Group, b: Group) => a.start < b.end && b.start < a.end
  for (const g of KIND_GROUPS) {
    const gs = gold.filter((x) => x.group === g)
    const ps = predicted.filter((x) => x.group === g)
    const c = into[g]
    c.gold += gs.length
    c.predicted += ps.length
    c.tp += gs.filter((x) => ps.some((p) => match(x, p))).length
    c.predTp += ps.filter((p) => gs.some((x) => match(x, p))).length
  }
}

/**
 * Entity-level scoring, per kind group plus micro total.
 *
 * - `strict`: same group and identical offsets.
 * - `overlap`: same group and at least one shared character. `tp`/`fn` are counted on the gold side
 *   (recall), `predTp`/`fp` on the prediction side (precision).
 *
 * Gold labels and predicted kinds are both mapped through {@link labelGroup}.
 */
export function score(
  gold: { start: number; end: number; label: string }[],
  predicted: PredSpan[],
  opts: { mode: ScoreMode },
): ScoreResult {
  return scoreAll([{ gold, predicted }], opts)
}

/** Like {@link score}, accumulated over several documents (counts are summed before ratios are formed). */
export function scoreAll(
  docs: { gold: { start: number; end: number; label: string }[]; predicted: PredSpan[] }[],
  opts: { mode: ScoreMode },
): ScoreResult {
  const counts = Object.fromEntries(KIND_GROUPS.map((g) => [g, emptyCounts()])) as Record<KindGroup, Counts>
  for (const d of docs) accumulate(counts, toGrouped(d.gold), dedupe(toGrouped(d.predicted)), opts.mode)

  const total = emptyCounts()
  for (const g of KIND_GROUPS) {
    total.gold += counts[g].gold
    total.predicted += counts[g].predicted
    total.tp += counts[g].tp
    total.predTp += counts[g].predTp
  }
  const groups = Object.fromEntries(KIND_GROUPS.map((g) => [g, finalize(counts[g])])) as Record<KindGroup, Metrics>
  return { groups, total: finalize(total) }
}
