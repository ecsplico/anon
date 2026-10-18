// Regelbasierte Pseudonymisierung (Stammdaten -> Platzhalter) und Rest-Scan.
//
// Entscheidungen:
// - Namen sind case-sensitiv und werden nur als ganze Wörter ersetzt (Unicode-Grenzen).
//   Genitiv ('Lenas', "Max'", "Lena's") wird erkannt: ersetzt wird nur der Name,
//   das angehängte 's bzw. ' bleibt stehen ('{{vorname}}s'), damit die Rückersetzung exakt ist.
//   'Sommerfelder' (Adjektiv-/Ableitungsform) wird bewusst NICHT ersetzt; der Rest-Scan meldet es dafür.
// - Überlappungen: längster Treffer zuerst; bei Gleichstand gewinnt die frühere Regel.
// - Koseformen -> {{vorname}} (Rückersetzung liefert den Vornamen, nicht die Koseform).

import { candidatePrefix, semanticStyle, type PlaceholderStyle } from './placeholders.ts'
import {
  AFTER,
  BEFORE,
  NOT_AFTER_CAPITALIZED,
  clean,
  datePattern,
  literal,
  maskPlaceholders,
  nameVariants,
  slug,
  wordPattern
} from './text.ts'
import { kindGroup, type DetectorName, type EntityKind, type Hit, type Rule, type Stammdaten } from './types.ts'

// ---------------------------------------------------------------------------
// Regeln aufbauen

type RoleClass = 'km' | 'kv' | 'geschwister' | string

function classifyRole(rolle: string): RoleClass {
  const r = rolle.trim().toLowerCase()
  if (r === 'mutter' || r === 'kindsmutter' || r === 'km') return 'km'
  if (r === 'vater' || r === 'kindsvater' || r === 'kv') return 'kv'
  if (r === 'geschwister' || r === 'schwester' || r === 'bruder') return 'geschwister'
  return slug(r)
}

function inferKind(placeholder: string): EntityKind {
  const p = placeholder.toLowerCase()
  if (/ort|wohnort|stadt/.test(p)) return 'LOCATION_CITY'
  if (/schule|institution|klinik|praxis|kita/.test(p)) return 'LOCATION_ORGANIZATION'
  if (/datum/.test(p)) return 'DATE'
  return 'NAME_EXT'
}

function ph(key: string): string {
  return `{{${key}}}`
}

export function buildRules(s: Stammdaten): Rule[] {
  const rules: Rule[] = []
  const seen = new Set<string>() // placeholder + '\0' + original
  const add = (placeholder: string, original: string, kind: EntityKind, body: string, flags = 'gu') => {
    const key = `${placeholder}\0${original}`
    if (!original || seen.has(key)) return
    seen.add(key)
    rules.push({ placeholder, original, kind, pattern: wordPattern(body, flags), source: 'rules' })
  }

  const vorname = clean(s.vorname)
  const nachname = clean(s.nachname)
  const vornamen = [...new Set([...nameVariants(vorname), ...(s.koseformen ?? []).map(clean).filter(Boolean)])]
  const nachnamen = nameVariants(nachname)

  // --- Patient: Vollname (Vorname Nachname / Nachname, Vorname) -> {{name}}
  if (vorname && nachname) {
    // Komplette Vor-/Nachnamen zuerst, damit der Original-Eintrag der Klarname ist.
    for (const v of vornamen) {
      for (const n of [nachname]) {
        add(ph('name'), `${v} ${n}`, 'NAME_PATIENT', `${literal(v)}\\s+${literal(n)}`)
        add(ph('name'), `${n}, ${v}`, 'NAME_PATIENT', `${NOT_AFTER_CAPITALIZED}${literal(n)}\\s*,\\s*${literal(v)}`)
      }
    }
  }

  // --- Eltern: Frau/Herr <Nachname> -> {{km}}/{{kv}}
  const angeh = (s.angehoerige ?? []).map(a => ({ ...a, vorname: clean(a.vorname), nachname: clean(a.nachname) }))
  const mothers = angeh.filter(a => classifyRole(a.rolle) === 'km')
  const fathers = angeh.filter(a => classifyRole(a.rolle) === 'kv')
  const frauSurnames = [...new Set([nachname, ...mothers.map(m => m.nachname)].filter(Boolean))]
  const herrSurnames = [...new Set([nachname, ...fathers.map(f => f.nachname)].filter(Boolean))]

  // Platzhalter-Basis je Angehörigem: km, kv, km2, geschwister1, oma1, …
  const parentPlaceholders = new Map<(typeof angeh)[number], string>()
  const countRole: Record<string, number> = {}
  for (const a of angeh) {
    const cls = classifyRole(a.rolle)
    const n = (countRole[cls] = (countRole[cls] ?? 0) + 1)
    let base: string
    if (cls === 'km' || cls === 'kv') base = n === 1 ? cls : `${cls}${n}`
    else base = `${cls}${n}` // geschwister1, oma1, …
    parentPlaceholders.set(a, base)
  }

  // Eltern: Vollname zuerst (erster Mapping-Eintrag von {{km}}/{{kv}}). Übrige Angehörige: Vollname erst nach dem
  // Vornamen, damit {{geschwister1}} beim Export zum Vornamen zurückgesetzt wird.
  const addFullNames = (a: (typeof angeh)[number]) => {
    const base = parentPlaceholders.get(a)!
    const eff = a.nachname || nachname
    for (const v of nameVariants(a.vorname)) {
      if (eff) {
        add(ph(base), `${v} ${eff}`, 'NAME_RELATIVE', `${literal(v)}\\s+${literal(eff)}`)
        add(ph(base), `${eff}, ${v}`, 'NAME_RELATIVE', `${NOT_AFTER_CAPITALIZED}${literal(eff)}\\s*,\\s*${literal(v)}`)
      }
    }
  }
  for (const a of angeh) if (['km', 'kv'].includes(classifyRole(a.rolle))) addFullNames(a)

  for (const n of frauSurnames) add(ph('km'), `Frau ${n}`, 'NAME_RELATIVE', `(?:Frau|Fr\\.)\\s+${literal(n)}`)
  for (const n of herrSurnames) add(ph('kv'), `Herr ${n}`, 'NAME_RELATIVE', `(?:Herr|Hr\\.)\\s+${literal(n)}`)

  // Angehörige: nur-Nachname (falls abweichend vom Patienten) und nur-Vorname
  for (const a of angeh) {
    const cls = classifyRole(a.rolle)
    const base = parentPlaceholders.get(a)!
    const isParent = cls === 'km' || cls === 'kv'
    if (a.nachname && a.nachname !== nachname) {
      for (const n of nameVariants(a.nachname)) {
        add(ph(isParent ? `${base}_nachname` : base), n, 'NAME_RELATIVE', literal(n))
      }
    }
    const vp = isParent ? `${base}_vorname` : base
    for (const v of nameVariants(a.vorname)) add(ph(vp), v, 'NAME_RELATIVE', literal(v))
  }

  for (const a of angeh) if (!['km', 'kv'].includes(classifyRole(a.rolle))) addFullNames(a)

  // --- Patient: Vorname / Nachname einzeln
  for (const v of vornamen) add(ph('vorname'), v, 'NAME_PATIENT', literal(v))
  for (const n of nachnamen) add(ph('nachname'), n, 'NAME_PATIENT', literal(n))

  // --- Geburtsdatum
  const geb = clean(s.geburtsdatum)
  if (geb) {
    const key = `${ph('geburtsdatum')}\0${geb}`
    seen.add(key)
    rules.push({ placeholder: ph('geburtsdatum'), original: geb, kind: 'DATE_BIRTH', pattern: datePattern(geb), source: 'rules' })
  }

  // --- Schule / Wohnort
  const schule = clean(s.schule)
  if (schule) add(ph('schule'), schule, 'LOCATION_ORGANIZATION', literal(schule))
  const wohnort = clean(s.wohnort)
  if (wohnort) add(ph('wohnort'), wohnort, 'LOCATION_CITY', literal(wohnort))

  // --- Weitere frei definierte
  for (const w of s.weitere ?? []) {
    const t = clean(w.text)
    if (t) add(w.placeholder, t, inferKind(w.placeholder), literal(t))
  }

  return rules
}

/** Placeholder -> erster Originaltext (für `restore`). */
export function toMapping(rules: Rule[]): { placeholder: string; original: string }[] {
  const out: { placeholder: string; original: string }[] = []
  const done = new Set<string>()
  for (const r of rules) {
    if (done.has(r.placeholder)) continue
    done.add(r.placeholder)
    out.push({ placeholder: r.placeholder, original: r.original })
  }
  return out
}

// ---------------------------------------------------------------------------
// Anwenden

function globalPattern(p: RegExp): RegExp {
  return p.global ? new RegExp(p.source, p.flags) : new RegExp(p.source, p.flags + 'g')
}

export type RuleHit = { start: number; end: number; /** Index into the `rules` array. */ rule: number }

/**
 * Non-overlapping rule matches, sorted by start. Longest match first, then earlier rule, then earlier start
 * (the rule that `applyRules` applies).
 */
export function findRuleHits(text: string, rules: Rule[]): RuleHit[] {
  const cands: RuleHit[] = []
  rules.forEach((r, i) => {
    const re = globalPattern(r.pattern)
    for (const m of text.matchAll(re)) {
      if (m[0].length === 0) continue
      cands.push({ start: m.index!, end: m.index! + m[0].length, rule: i })
    }
  })
  // Längster Treffer zuerst, dann frühere Regel, dann früherer Beginn
  cands.sort((a, b) => (b.end - b.start) - (a.end - a.start) || a.rule - b.rule || a.start - b.start)
  const taken = new Uint8Array(text.length)
  const accepted: RuleHit[] = []
  for (const c of cands) {
    let free = true
    for (let k = c.start; k < c.end; k++) {
      if (taken[k]) { free = false; break }
    }
    if (!free) continue
    taken.fill(1, c.start, c.end)
    accepted.push(c)
  }
  return accepted.sort((a, b) => a.start - b.start)
}

export function applyRules(text: string, rules: Rule[]): { text: string; hits: Hit[] } {
  const accepted = findRuleHits(text, rules)

  let out = ''
  let pos = 0
  const hits: Hit[] = []
  for (const c of accepted) {
    const r = rules[c.rule]!
    out += text.slice(pos, c.start) + r.placeholder
    pos = c.end
    hits.push({ placeholder: r.placeholder, original: text.slice(c.start, c.end), start: c.start, end: c.end })
  }
  out += text.slice(pos)
  return { text: out, hits }
}

// ---------------------------------------------------------------------------
// Kandidaten (Detektor / manuell)

export type Candidate = {
  text: string
  kind: EntityKind
  placeholder?: string
  source?: Extract<DetectorName, 'regex' | 'onnx' | 'llm' | 'manual'>
}

/**
 * Builds a rule for a detected/manual candidate; the caller appends it to `rules`.
 * Same original text and same `kindGroup` as an existing rule reuse that rule's placeholder
 * (a manual `NAME_EXT` equal to a relative's `NAME_RELATIVE` stays the same person).
 * Otherwise `semantic` numbers per `candidatePrefix(kind)` (`{{person1}}`, `{{ort2}}`), token styles
 * draw `style.token(kind)`. An explicit `placeholder` wins. Default source: `'llm'`.
 */
export function addCandidate(rules: Rule[], candidate: Candidate, style: PlaceholderStyle = semanticStyle): Rule {
  const text = clean(candidate.text)
  if (!text) throw new Error('addCandidate: leerer Text')
  let placeholder = candidate.placeholder
  if (!placeholder) {
    const group = kindGroup(candidate.kind)
    const same = rules.find(r => r.original === text && kindGroup(r.kind) === group)
    if (same) placeholder = same.placeholder
  }
  if (!placeholder) {
    if (style.name === 'semantic') {
      const prefix = candidatePrefix(candidate.kind)
      const re = new RegExp(`^\\{\\{${prefix}(\\d+)\\}\\}$`)
      let max = 0
      for (const r of rules) {
        const m = r.placeholder.match(re)
        if (m) max = Math.max(max, Number(m[1]))
      }
      placeholder = style.token(candidate.kind, `${prefix}${max + 1}`)
    } else {
      placeholder = style.token(candidate.kind)
    }
  }
  return {
    placeholder,
    original: text,
    kind: candidate.kind,
    pattern: wordPattern(literal(text)),
    source: candidate.source ?? 'llm'
  }
}

// ---------------------------------------------------------------------------
// Rest-Scan

/**
 * Sucht nach Stammdaten-Resten (case-insensitiv). Platzhalter des Styles werden ignoriert.
 * Strenger als applyRules: Begriffe ≥ 5 Zeichen werden auch als Wortanfang gemeldet
 * ('Sommerfelder'), kürzere nur als ganzes Wort (inkl. Genitiv-s).
 * `term` ist der gefundene Textausschnitt, `index` der Offset im übergebenen Text.
 */
export function residualScan(
  text: string,
  s: Stammdaten,
  style: PlaceholderStyle = semanticStyle
): { term: string; index: number }[] {
  const masked = maskPlaceholders(text, style.pattern)

  const terms = new Set<string>()
  const addName = (n?: string) => {
    for (const v of nameVariants(clean(n))) if (v.length >= 3) terms.add(v)
  }
  addName(s.vorname)
  addName(s.nachname)
  for (const k of s.koseformen ?? []) addName(k)
  for (const a of s.angehoerige ?? []) {
    addName(a.vorname)
    addName(a.nachname)
  }
  for (const w of s.weitere ?? []) {
    const t = clean(w.text)
    if (t) terms.add(t)
  }
  for (const t of [s.schule, s.wohnort]) {
    const v = clean(t)
    if (v) terms.add(v)
  }

  const found: { term: string; index: number }[] = []
  for (const term of terms) {
    const body = literal(term)
    const re = term.length >= 5
      ? new RegExp(`${BEFORE}${body}`, 'giu')
      : new RegExp(`${BEFORE}${body}${AFTER}`, 'giu')
    for (const m of masked.matchAll(re)) {
      // bei längeren Begriffen das ganze Wort melden ('Sommerfelder')
      let end = m.index! + m[0].length
      if (term.length >= 5) {
        const rest = /^[\p{L}\p{N}_]*/u.exec(masked.slice(end))![0]
        end += rest.length
      }
      found.push({ term: text.slice(m.index!, end), index: m.index! })
    }
  }

  const geb = clean(s.geburtsdatum)
  if (geb) {
    for (const m of masked.matchAll(datePattern(geb))) found.push({ term: text.slice(m.index!, m.index! + m[0].length), index: m.index! })
  }

  // Doppelte Positionen: längsten Treffer behalten
  found.sort((a, b) => a.index - b.index || b.term.length - a.term.length)
  const out: { term: string; index: number }[] = []
  for (const f of found) {
    const last = out[out.length - 1]
    if (last && f.index < last.index + last.term.length) continue
    out.push(f)
  }
  return out
}
