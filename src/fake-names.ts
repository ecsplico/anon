// Random, gender-matched German replacement names for NAME_* entities.
//
// `FakeNames` maps each original given name and surname to a fake of the same gender (Herr/Frau cue,
// else lookup in the German first-name lists, else a coin flip). Maps are shared per instance, so
// family members keep a common fake surname and a given name maps to the same fake everywhere.
// Leading titles (`Dr.`, `Frau` …) are kept verbatim. Fakes never occur in the source text, so the
// replacement can be reversed by plain string substitution.
//
// Known limitation: inflected forms (genitive `Müllers`) are not normalised; they get their own fake.

import { fakerDE } from '@faker-js/faker'

type Gender = 'male' | 'female'

/** The slice of a faker instance this module draws names from (stub-able in tests). */
export type FakerLike = {
  person: { firstName(sex: Gender): string; lastName(): string }
}

export type FakeNamesOptions = {
  faker?: FakerLike
  /** Replaces `Math.random` for the coin flip when the gender is unknown. */
  random?: () => number
}

/** Leading tokens kept verbatim; Herr/Frau also fix the gender. */
const TITLES: Record<string, true> = Object.fromEntries(
  ['herr', 'herrn', 'hr.', 'frau', 'fr.', 'dr.', 'prof.', 'med.', 'dent.', 'dipl.-med.', 'pd', 'oa', 'oä'].map(t => [t, true as const])
)
const MALE_CUES: Record<string, true> = { herr: true, herrn: true, 'hr.': true }
const FEMALE_CUES: Record<string, true> = { frau: true, 'fr.': true }
const CUE_BEFORE = /(Herrn?|Hr\.|Frau|Fr\.)\s*$/
const MAX_DRAWS = 100

type FirstNameLists = { female: Set<string>; male: Set<string> }
let firstNames: FirstNameLists | undefined

/** German first names by gender from the faker locale data (lower-cased). */
function germanFirstNames(): FirstNameLists {
  if (!firstNames) {
    const raw = fakerDE.rawDefinitions.person?.first_name
    firstNames = {
      female: new Set((raw?.female ?? []).map(n => n.toLowerCase())),
      male: new Set((raw?.male ?? []).map(n => n.toLowerCase()))
    }
  }
  return firstNames
}

export class FakeNames {
  private readonly text: string
  private readonly faker: FakerLike
  private readonly random: () => number
  private readonly given = new Map<string, string>()
  private readonly family = new Map<string, string>()
  private readonly used = new Set<string>()

  /** `text` is the source: no fake may occur in it. */
  constructor(text: string, opts: FakeNamesOptions = {}) {
    this.text = text
    this.faker = opts.faker ?? fakerDE
    this.random = opts.random ?? Math.random
  }

  /** Fake name for `original`; `before` is the text just ahead of it (salutation cue). Titles-only input is returned unchanged. */
  replacement(original: string, before = ''): string {
    const tokens = original.split(/\s+/).filter(Boolean)
    const prefix: string[] = []
    while (tokens.length > 0 && Object.hasOwn(TITLES, tokens[0]!.toLowerCase())) prefix.push(tokens.shift()!)
    if (tokens.length === 0) return original

    const cue = this.cue(prefix, before)
    const rest = tokens.join(' ')
    let result: string

    if (rest.includes(',')) {
      const comma = rest.indexOf(',')
      const family = rest.slice(0, comma).trim()
      const given = rest.slice(comma + 1).trim().split(/\s+/).filter(Boolean)
      const gender = this.gender(cue, given)
      const fakeGiven = given.map(g => this.fakeGiven(g, gender))
      const fakeFamily = family ? this.fakeFamily(family) : ''
      const body = `${fakeFamily}, ${fakeGiven.join(' ')}`.trim()
      result = [...prefix, body].join(' ')
    } else {
      let given: string[]
      let family: string | undefined
      if (tokens.length === 1) {
        const tok = tokens[0]!
        if (cue !== undefined || this.lookupGender(tok) === undefined) {
          given = []
          family = tok
        } else {
          given = [tok]
          family = undefined
        }
      } else {
        given = tokens.slice(0, -1)
        family = tokens[tokens.length - 1]
      }
      const gender = this.gender(cue, given)
      const parts = given.map(g => this.fakeGiven(g, gender))
      if (family !== undefined) parts.push(this.fakeFamily(family))
      result = [...prefix, ...parts].join(' ')
    }

    const isUpper = original === original.toUpperCase() && original !== original.toLowerCase()
    return isUpper ? result.toUpperCase() : result
  }

  private cue(prefix: string[], before: string): Gender | undefined {
    const lowered = prefix.map(p => p.toLowerCase())
    if (lowered.some(p => Object.hasOwn(MALE_CUES, p))) return 'male'
    if (lowered.some(p => Object.hasOwn(FEMALE_CUES, p))) return 'female'
    const m = CUE_BEFORE.exec(before)
    if (m) return Object.hasOwn(MALE_CUES, m[1]!.toLowerCase()) ? 'male' : 'female'
    return undefined
  }

  private gender(cue: Gender | undefined, given: string[]): Gender {
    if (cue !== undefined) return cue
    if (given.length > 0) {
      const g = this.lookupGender(given[0]!)
      if (g !== undefined) return g
    }
    return this.random() < 0.5 ? 'male' : 'female'
  }

  private lookupGender(name: string): Gender | undefined {
    const first = name.split('-')[0]!.toLowerCase()
    const lists = germanFirstNames()
    const female = lists.female.has(first)
    const male = lists.male.has(first)
    if (female === male) return undefined // unknown or ambiguous
    return female ? 'female' : 'male'
  }

  private fakeGiven(original: string, gender: Gender): string {
    let fake = this.given.get(original)
    if (fake === undefined) {
      fake = this.draw(() => this.faker.person.firstName(gender))
      this.given.set(original, fake)
    }
    return fake
  }

  private fakeFamily(original: string): string {
    let fake = this.family.get(original)
    if (fake === undefined) {
      fake = this.draw(() => this.faker.person.lastName())
      this.family.set(original, fake)
    }
    return fake
  }

  private draw(fn: () => string): string {
    for (let i = 0; i < MAX_DRAWS; i++) {
      const candidate = fn()
      // Plain substring check on purpose: a fake "Weber" must not be reversed inside a source "Webersberg".
      if (!this.used.has(candidate) && !this.text.includes(candidate)) {
        this.used.add(candidate)
        return candidate
      }
    }
    throw new Error('Kein unbenutzter Fake-Name verfügbar')
  }
}
