// LLM-Stufe der Anonymisierung: ein lokales LLM schlägt Personen, Orte, Institutionen und
// Geburtsdaten vor, die Stammdaten-Regeln und Regex nicht kennen. Eingabe ist Text, der bereits
// teilweise anonymisiert sein darf (enthält dann Platzhalter des aktiven Styles).
//
// Entscheidungen:
// - Das Modell ist nur Vorschlagsgeber. Jeder Vorschlag wird gegen den Text geprüft (Halluzinationsfilter):
//   nur Zeichenketten, die wörtlich (ganzes Wort, Genitiv-s erlaubt) im Text stehen, bleiben übrig.
// - Platzhalter werden vor dem Abgleich maskiert, damit z. B. `{{km}}` nie als Treffer zählt.
// - `chat` ist immer injiziert; die Bibliothek kennt weder Datenbank noch Standard-Endpunkt.
//   `openAiCompatibleChat` liefert einen Client für lokale OpenAI-kompatible Server (oMLX, Ollama).
// - Ein fehlgeschlagener Chunk wirft: stilles Auslassen würde unentdeckt Klarnamen durchlassen.

import { z } from 'zod'
import { semanticStyle } from '../placeholders.ts'
import { addCandidate } from '../rules.ts'
import { AFTER, BEFORE, chunkText, literal, maskPlaceholders } from '../text.ts'
import type { EntityKind, Rule, Span } from '../types.ts'
import type { DetectContext, Detector } from './types.ts'
import type { PlaceholderStyle } from '../placeholders.ts'

export type NerKind = Extract<EntityKind, 'NAME_EXT' | 'LOCATION_OTHER' | 'LOCATION_ORGANIZATION' | 'DATE'>

export type NerCandidate = {
  text: string
  kind: NerKind
  count: number
  /** Kurze Ausschnitte (≤ 80 Zeichen) aus dem Text. */
  contexts: string[]
}

export type NerPrompt = { system: string; user: string }

export type NerChat = (prompt: NerPrompt) => Promise<unknown>

export type FindCandidatesOptions = {
  /** Modellaufruf (Pflicht): liefert das geparste JSON-Objekt der Antwort. */
  chat: NerChat
  /** Eingabe enthält Platzhalter (Standard: true); steuert nur den Hinweis im Prompt. */
  knownPlaceholdersText?: boolean
  /** Muster der Platzhalter des aktiven Styles (Standard: semantic `{{…}}`). */
  placeholderPattern?: RegExp
  /** Zeichen je Chunk (Standard 3000). */
  chunkChars?: number
  /** Überlappung zwischen Chunks (Standard 300). */
  overlapChars?: number
}

// ---------------------------------------------------------------------------
// Schema + Prompt

export const NerResponseSchema = z.object({
  persons: z.array(z.string()).default([]),
  places: z.array(z.string()).default([]),
  institutions: z.array(z.string()).default([]),
  dates: z.array(z.string()).default([])
})
export type NerResponse = z.infer<typeof NerResponseSchema>

const SYSTEM = `Du bist ein Prüfer für Datenschutz in deutschsprachigen Verlaufsnotizen einer kinder- und jugendpsychiatrischen Tagesklinik.
Der Text wurde bereits teilweise anonymisiert. Du findest Reste, die noch auf Personen oder Orte hinweisen.

Suche im Text nach:
- persons: Personennamen (Vornamen allein, Nachnamen, Vor- und Nachnamen, Spitznamen, Kosenamen), z. B. von Freundinnen, Mitpatienten, Verwandten oder Behandelnden. Ohne Anrede ("Frau", "Herr") und ohne Berufsbezeichnung angeben, nur den Namen.
- places: Ortsnamen (Städte, Dörfer, Stadtteile, Straßen, Plätze).
- institutions: Institutionen mit eigenem Namen (Schulen, Kliniken, Praxen, Ämter, Vereine, Arbeitgeber).
- dates: konkrete Geburtsdaten in jeder Schreibweise (z. B. "12.3.2010", "12.03.10", "12. März 2010"). Keine Behandlungs-, Termin- oder Aufnahmedaten.

Regeln:
- Gib jede Zeichenkette exakt so zurück, wie sie im Text steht (gleiche Schreibweise, ohne Anrede, ohne Genitiv-s wenn erkennbar nur angehängt).
- Ignoriere Platzhalter in doppelten geschweiften Klammern wie {{name}}, {{km}} oder {{ort1}}: sie sind bereits anonymisiert und dürfen nie ausgegeben werden.
- Ignoriere Rollen und Gattungswörter ohne Namen (Mutter, Vater, Patientin, Klassenlehrerin, Bezugspflege, Klinik, Schule, Therapeut, Arzt) sowie Berufs-, Wochentags- und Monatsnamen.
- Medikamente, Diagnosen, Testverfahren und Therapiemethoden sind keine Namen.
- Erfinde nichts. Gib nur zurück, was im Text steht. Gibt es nichts, liefere leere Listen.

Antworte ausschließlich mit JSON ohne Erläuterung: {"persons": [], "places": [], "institutions": [], "dates": []}`

const PLACEHOLDER_HINT = '\nHinweis: Der Text enthält Platzhalter wie {{name}}; behandle sie als bereits anonymisiert.'

function buildPrompt(chunk: string, withPlaceholders: boolean): NerPrompt {
  return {
    system: withPlaceholders ? SYSTEM + PLACEHOLDER_HINT : SYSTEM,
    user: `Text:\n"""\n${chunk}\n"""\n\nGib das JSON zurück.`
  }
}

// ---------------------------------------------------------------------------
// Stoppliste: Gattungswörter, Rollen, Berufe, Wochentage, Monate

const STOP: Record<string, true> = Object.fromEntries(
  `mutter vater eltern elternteil geschwister bruder schwester oma opa großmutter großvater tante onkel cousine cousin
  kind kinder tochter sohn patient patientin patienten mitpatient mitpatientin mitpatienten mitschüler mitschülerin freund freundin
  klinik tagesklinik klinikschule station ambulanz praxis schule gesamtschule heimatschule grundschule realschule gymnasium hauptschule kita kindergarten
  jugendamt erziehungsberatung sozialdienst krankenhaus hausarzt kinderarzt kinderärztin
  arzt ärztin oberarzt oberärztin chefarzt chefärztin therapeut therapeutin psychologe psychologin psychotherapeut psychotherapeutin
  pflege pflegekraft bezugspflege bezugstherapeut bezugstherapeutin pflegeteam team sozialarbeiter sozialarbeiterin
  ergotherapeut ergotherapeutin musiktherapeut musiktherapeutin kunsttherapeut kunsttherapeutin lehrer lehrerin klassenlehrer klassenlehrerin
  schulbegleitung frau herr dr prof fr hr
  montag dienstag mittwoch donnerstag freitag samstag sonntag
  januar februar märz april mai juni juli august september oktober november dezember`
    .split(/\s+/)
    .filter(Boolean)
    .map(w => [w, true] as const)
)

function isStopText(text: string): boolean {
  const words = text.toLowerCase().split(/[\s\-.,;:]+/).filter(Boolean)
  return words.length === 0 || words.every(w => Object.hasOwn(STOP, w))
}

// ---------------------------------------------------------------------------
// Text-Hilfen

const TITLE = /^(?:Frau|Herr|Fr\.|Hr\.|Dr\.|Prof\.)\s+/u

function normalize(raw: string, kind: NerKind): string {
  let s = raw.trim().replace(/\s+/g, ' ')
  s = s.replace(/^[\s"'„“”‚‘’(\[]+|[\s"'„“”‚‘’)\],;:]+$/gu, '')
  // Anrede abstreifen (mehrfach, z. B. "Frau Dr. Meier").
  for (let prev = ''; prev !== s;) {
    prev = s
    s = s.replace(TITLE, '')
  }
  // Schlusspunkt nur bei Daten ("12.03.10.") und Initialen ("Lena S.") behalten.
  if (kind !== 'DATE' && /\p{L}{2,}\.$/u.test(s)) s = s.slice(0, -1)
  return s.trim()
}

/** Context of at most 80 characters around `text[index, index+length)`, widened to word boundaries, whitespace collapsed. */
export function contextSnippet(text: string, index: number, length: number): string {
  const max = 80
  const half = Math.max(0, Math.floor((max - length) / 2))
  let from = Math.max(0, index - half)
  let to = Math.min(text.length, index + length + half)
  // Auf Wortgrenzen ziehen, soweit der Rahmen es zulässt.
  while (from > 0 && from < index && /\S/.test(text[from - 1]!)) from++
  while (to < text.length && to > index + length && /\S/.test(text[to]!)) to--
  const raw = text.slice(from, to).replace(/\s+/g, ' ').trim()
  return raw.length > max ? raw.slice(0, max) : raw
}

const DATE_RE = /\d{1,2}\s*\.\s*(?:\d{1,2}|\p{L}+)\s*\.?\s*(?:\d{4}|\d{2})/u

// ---------------------------------------------------------------------------
// Hauptfunktion

const KIND_ORDER: NerKind[] = ['NAME_EXT', 'LOCATION_ORGANIZATION', 'LOCATION_OTHER', 'DATE']

/**
 * Findet Namens-Reste im Text. Sortiert nach Häufigkeit absteigend (bei Gleichstand nach Text).
 */
export async function findCandidates(text: string, opts: FindCandidatesOptions): Promise<NerCandidate[]> {
  const chunkChars = Math.max(200, opts.chunkChars ?? 3000)
  const overlap = Math.min(opts.overlapChars ?? 300, Math.floor(chunkChars / 2))
  const withPlaceholders = opts.knownPlaceholdersText ?? true
  const pattern = opts.placeholderPattern ?? semanticStyle.pattern

  const proposals: { text: string; kind: NerKind }[] = []
  for (const chunk of chunkText(text, chunkChars, overlap)) {
    if (!chunk.trim()) continue
    const raw = await opts.chat(buildPrompt(chunk, withPlaceholders))
    const parsed = NerResponseSchema.safeParse(raw)
    if (!parsed.success) {
      throw new Error(`NER-Antwort ungültig: ${parsed.error.issues.map(i => `${i.path.join('.') || '(Wurzel)'}: ${i.message}`).join('; ')}`)
    }
    const d = parsed.data
    for (const t of d.persons) proposals.push({ text: t, kind: 'NAME_EXT' })
    for (const t of d.institutions) proposals.push({ text: t, kind: 'LOCATION_ORGANIZATION' })
    for (const t of d.places) proposals.push({ text: t, kind: 'LOCATION_OTHER' })
    for (const t of d.dates) proposals.push({ text: t, kind: 'DATE' })
  }

  const masked = maskPlaceholders(text, pattern)
  const byKey = new Map<string, NerCandidate>()

  for (const p of proposals) {
    const norm = normalize(p.text, p.kind)
    if (norm.length < 2) continue
    if (p.kind === 'DATE') {
      if (!DATE_RE.test(norm)) continue
    } else {
      if (!/\p{Lu}/u.test(norm)) continue // Namen sind großgeschrieben
      if (isStopText(norm)) continue
    }
    const key = norm.toLowerCase()
    const existing = byKey.get(key)
    if (existing && KIND_ORDER.indexOf(existing.kind) <= KIND_ORDER.indexOf(p.kind)) continue

    // Halluzinationsfilter: wörtlich im (maskierten) Text? Schreibweise aus dem Text übernehmen.
    const probe = new RegExp(`${BEFORE}(?:${literal(norm)})${AFTER}`, 'giu')
    const first = probe.exec(masked)
    if (!first) continue
    const surface = text.slice(first.index, first.index + first[0].length)
    if (p.kind !== 'DATE' && isStopText(surface)) continue

    const exact = new RegExp(`${BEFORE}(?:${literal(surface)})${AFTER}`, 'gu')
    const contexts: string[] = []
    let count = 0
    for (const m of masked.matchAll(exact)) {
      count++
      if (contexts.length < 3) {
        const c = contextSnippet(text, m.index!, m[0].length)
        if (!contexts.includes(c)) contexts.push(c)
      }
    }
    if (count === 0) continue
    byKey.set(key, { text: surface, kind: p.kind, count, contexts })
  }

  return [...byKey.values()].sort((a, b) => b.count - a.count || a.text.localeCompare(b.text, 'de'))
}

// ---------------------------------------------------------------------------
// Brücke zu rules.ts

export type AcceptedCandidate = { text: string; kind: EntityKind; placeholder?: string }

/**
 * Angenommene Kandidaten -> Regeln. Rückgabe: `existing` gefolgt von den neuen Regeln (eindeutige
 * Platzhalter je Style; derselbe Text + Art teilt den Platzhalter). Kandidaten aus der Liste
 * erhalten `source: 'llm'`, frei ergänzte Einträge `source: 'manual'`. Die Eingaberegeln bleiben
 * unverändert.
 */
export function toRules(
  candidates: NerCandidate[],
  accepted: AcceptedCandidate[],
  existing: Rule[],
  style: PlaceholderStyle = semanticStyle
): Rule[] {
  const known = new Map(candidates.map(c => [`${c.kind}\0${c.text.toLowerCase()}`, c]))
  const rules = [...existing]
  for (const a of accepted) {
    const hit = known.get(`${a.kind}\0${a.text.trim().toLowerCase()}`)
    const text = hit?.text ?? a.text.trim()
    if (!text) continue
    // Schon durch dieselbe Original+Art-Regel abgedeckt: keine zweite Regel mit gleichem Muster.
    if (!a.placeholder && rules.some(r => r.original === text && r.kind === a.kind)) continue
    rules.push(addCandidate(rules, { text, kind: a.kind, placeholder: a.placeholder, source: hit ? 'llm' : 'manual' }, style))
  }
  return rules
}

// ---------------------------------------------------------------------------
// Detektor

export type LlmDetectorOptions = {
  chat: NerChat
  chunkChars?: number
  overlapChars?: number
  placeholderPattern?: RegExp
}

/** Ein Span je ganzem Wortvorkommen jedes Kandidaten (Score 0.7). */
export function llmDetector(opts: LlmDetectorOptions): Detector {
  return {
    name: 'llm',
    async detect(text: string, ctx: DetectContext): Promise<Span[]> {
      // `ctx.masked` enthält keine Platzhalter mehr (Leerzeichen), daher kein Platzhalter-Hinweis.
      const candidates = await findCandidates(ctx.masked, {
        chat: opts.chat,
        chunkChars: opts.chunkChars,
        overlapChars: opts.overlapChars,
        placeholderPattern: opts.placeholderPattern,
        knownPlaceholdersText: false
      })
      const spans: Span[] = []
      for (const c of candidates) {
        const re = new RegExp(`${BEFORE}(?:${literal(c.text)})${AFTER}`, 'gu')
        for (const m of ctx.masked.matchAll(re)) {
          const start = m.index!
          const end = start + m[0].length
          spans.push({ start, end, text: text.slice(start, end), kind: c.kind, score: 0.7, detector: 'llm' })
        }
      }
      return spans.sort((a, b) => a.start - b.start || b.end - a.end)
    }
  }
}

// ---------------------------------------------------------------------------
// Lokaler OpenAI-kompatibler Client

/** Nur lokale oder private Ziele: Patientendaten verlassen den Rechner nicht. Keine DNS-Auflösung. */
export function assertLocalUrl(baseUrl: string): URL {
  const url = new URL(baseUrl)
  const host = url.hostname.replace(/^\[|\]$/g, '').toLowerCase()
  const v4 = host.match(/^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$/)
  let local = host === 'localhost' || host.endsWith('.localhost') || host.endsWith('.local') || host === '::1'
  if (v4) {
    const [a, b] = [Number(v4[1]), Number(v4[2])]
    local = a === 127 || a === 10 || (a === 172 && b >= 16 && b <= 31) || (a === 192 && b === 168) || (a === 169 && b === 254)
  }
  if (host.includes(':') && (host.startsWith('fe80:') || host.startsWith('fc') || host.startsWith('fd'))) local = true
  if (!local || !/^https?:$/.test(url.protocol)) throw new Error(`LLM-Endpunkt muss lokal sein: ${url.hostname}`)
  return url
}

function tryParse(s: string): { ok: true; value: unknown } | { ok: false } {
  try {
    return { ok: true, value: JSON.parse(s) }
  } catch {
    return { ok: false }
  }
}

/** Index der schließenden Klammer zum Öffner an `start` (stringbewusst), sonst -1. */
function balancedEnd(s: string, start: number): number {
  let depth = 0
  let inString = false
  for (let i = start; i < s.length; i++) {
    const c = s[i]
    if (inString) {
      if (c === '\\') i++
      else if (c === '"') inString = false
    } else if (c === '"') inString = true
    else if (c === '{' || c === '[') depth++
    else if (c === '}' || c === ']') {
      depth--
      if (depth === 0) return i
    }
  }
  return -1
}

/** Erstes JSON-Objekt/-Array aus Modelltext: entfernt `<think>`-Blöcke, bevorzugt ```json-Zäune, sonst balancierte Klammern. */
export function extractJson(text: string): unknown {
  const cleaned = text.replace(/<think>[\s\S]*?<\/think>/g, '').trim()
  const candidates: string[] = []
  for (const m of cleaned.matchAll(/```(?:json)?\s*([\s\S]*?)```/gi)) candidates.push(m[1]!.trim())
  candidates.push(cleaned)
  for (const candidate of candidates) {
    const direct = tryParse(candidate)
    if (direct.ok) return direct.value
    for (let i = 0; i < candidate.length; i++) {
      const ch = candidate[i]
      if (ch !== '{' && ch !== '[') continue
      const end = balancedEnd(candidate, i)
      if (end < 0) continue
      const parsed = tryParse(candidate.slice(i, end + 1))
      if (parsed.ok) return parsed.value
    }
  }
  throw new Error('Kein JSON in der Antwort gefunden')
}

export type OpenAiCompatibleChatOptions = {
  baseUrl: string
  model: string
  apiKey?: string
  timeoutMs: number
  /** Für Tests; Standard: globales `fetch`. */
  fetch?: typeof fetch
}

/** Chat-Funktion für einen lokalen OpenAI-kompatiblen Server (`POST <baseUrl>/chat/completions`, JSON-Modus). */
export function openAiCompatibleChat(opts: OpenAiCompatibleChatOptions): NerChat {
  assertLocalUrl(opts.baseUrl)
  const endpoint = `${opts.baseUrl.replace(/\/+$/, '')}/chat/completions`
  const doFetch = opts.fetch ?? fetch
  return async prompt => {
    const headers: Record<string, string> = { 'content-type': 'application/json' }
    if (opts.apiKey) headers.authorization = `Bearer ${opts.apiKey}`
    const res = await doFetch(endpoint, {
      method: 'POST',
      headers,
      body: JSON.stringify({
        model: opts.model,
        messages: [
          { role: 'system', content: prompt.system },
          { role: 'user', content: prompt.user }
        ],
        response_format: { type: 'json_object' },
        temperature: 0
      }),
      signal: AbortSignal.timeout(opts.timeoutMs)
    })
    if (!res.ok) throw new Error(`LLM-Anfrage fehlgeschlagen: HTTP ${res.status}`)
    const body = (await res.json()) as { choices?: { message?: { content?: unknown } }[] }
    const content = body.choices?.[0]?.message?.content
    if (typeof content !== 'string') throw new Error('LLM-Antwort ohne choices[0].message.content')
    return extractJson(content)
  }
}
