import { describe, expect, test } from 'bun:test'
import {
  STYLE_NAMES,
  candidatePrefix,
  createStyle,
  gemtexStyle,
  getStyle,
  llmGuardStyle,
  presidioStyle,
  restore,
  semanticStyle
} from '../src/placeholders.ts'
import { addCandidate, applyRules, buildRules } from '../src/rules.ts'
import { ENTITY_KINDS, kindGroup } from '../src/types.ts'
import { maskPlaceholders } from '../src/text.ts'

const matches = (re: RegExp, s: string) => [...s.matchAll(re)].map(m => m[0])

describe('styles', () => {
  test('every token matches its own pattern completely', () => {
    for (const kind of ENTITY_KINDS) {
      const tokens = [
        semanticStyle.token(kind, 'person1'),
        presidioStyle.token(kind),
        llmGuardStyle.token(kind),
        gemtexStyle.token(kind)
      ]
      const styles = [semanticStyle, presidioStyle, llmGuardStyle, gemtexStyle]
      tokens.forEach((t, i) => expect(matches(styles[i]!.pattern, t)).toEqual([t]))
    }
  })

  test('token formats', () => {
    expect(semanticStyle.token('NAME_EXT', 'km')).toBe('{{km}}')
    expect(presidioStyle.token('CONTACT_EMAIL')).toMatch(/^<CONTACT_EMAIL_[0-9a-f]{8}>$/)
    expect(llmGuardStyle.token('DATE_BIRTH')).toMatch(/^\[REDACTED_DATE_BIRTH_[0-9a-f]{8}\]$/)
    expect(gemtexStyle.token('NAME_PATIENT')).toMatch(/^\[\*\* NAME_PATIENT [A-Z0-9]{6} \*\*\]$/)
  })

  test('semantic requires a key', () => {
    expect(() => semanticStyle.token('NAME_EXT')).toThrow()
  })

  test('token styles are random per call', () => {
    expect(presidioStyle.token('ID')).not.toBe(presidioStyle.token('ID'))
    expect(gemtexStyle.token('ID')).not.toBe(gemtexStyle.token('ID'))
  })

  test('patterns are global and do not cross-match', () => {
    const t = `${presidioStyle.token('ID')} ${gemtexStyle.token('ID')} ${llmGuardStyle.token('ID')} {{km}}`
    expect(matches(presidioStyle.pattern, t)).toHaveLength(1)
    expect(matches(gemtexStyle.pattern, t)).toHaveLength(1)
    expect(matches(llmGuardStyle.pattern, t)).toHaveLength(1)
    expect(matches(semanticStyle.pattern, t)).toEqual(['{{km}}'])
  })

  test('createStyle injects the random source', () => {
    const seq = ['aaaaaaaa', 'bbbbbbbb']
    const s = createStyle('presidio', { random: () => seq.shift()! })
    expect(s.token('NAME_EXT')).toBe('<NAME_EXT_aaaaaaaa>')
    expect(s.token('NAME_EXT')).toBe('<NAME_EXT_bbbbbbbb>')
    expect(createStyle('llm-guard', { random: () => 'deadbeef' }).token('ID')).toBe('[REDACTED_ID_deadbeef]')
    expect(createStyle('gemtex', { random: () => 'AB12CD' }).token('ID')).toBe('[** ID AB12CD **]')
    expect(createStyle('semantic', { random: () => 'ignored' }).token('ID', 'x')).toBe('{{x}}')
  })

  test('getStyle returns the named constants', () => {
    expect(STYLE_NAMES.map(n => getStyle(n).name)).toEqual([...STYLE_NAMES])
    expect(getStyle('gemtex')).toBe(gemtexStyle)
  })

  test('maskPlaceholders keeps indices', () => {
    const t = 'a {{km}} b'
    const m = maskPlaceholders(t, semanticStyle.pattern)
    expect(m).toBe('a        b')
    expect(m.length).toBe(t.length)
    // reusable: shared global regex state does not leak between calls
    expect(maskPlaceholders(t, semanticStyle.pattern)).toBe(m)
  })
})

describe('kinds', () => {
  test('kindGroup', () => {
    expect(kindGroup('NAME_RELATIVE')).toBe('NAME')
    expect(kindGroup('DATE_BIRTH')).toBe('DATE')
    expect(kindGroup('DATE')).toBe('DATE')
    expect(kindGroup('LOCATION_CITY')).toBe('LOCATION')
    expect(kindGroup('CONTACT_IP')).toBe('CONTACT')
    expect(kindGroup('ID')).toBe('ID')
    expect(kindGroup('AGE')).toBe('AGE')
    expect(kindGroup('PROFESSION')).toBe('PROFESSION')
    expect(kindGroup('OTHER')).toBe('OTHER')
  })

  test('candidatePrefix', () => {
    expect(candidatePrefix('NAME_EXT')).toBe('person')
    expect(candidatePrefix('NAME_DOCTOR')).toBe('person')
    expect(candidatePrefix('DATE_DEATH')).toBe('datum')
    expect(candidatePrefix('LOCATION_HOSPITAL')).toBe('institution')
    expect(candidatePrefix('LOCATION_ORGANIZATION')).toBe('institution')
    expect(candidatePrefix('LOCATION_STREET')).toBe('ort')
    expect(candidatePrefix('CONTACT_PHONE')).toBe('kontakt')
    expect(candidatePrefix('ID')).toBe('id')
    expect(candidatePrefix('AGE')).toBe('alter')
    expect(candidatePrefix('PROFESSION')).toBe('beruf')
    expect(candidatePrefix('OTHER')).toBe('sonstiges')
  })
})

describe('restore', () => {
  test('longest placeholder first when one is a prefix of another', () => {
    const e = [
      { placeholder: '<NAME_EXT_1234abcd>', original: 'Anna' },
      { placeholder: '<NAME_EXT_1234abcd>X', original: 'Berta' },
      { placeholder: '{{km}}', original: 'Katrin Berger' },
      { placeholder: '{{km}}_vorname', original: 'Katrin' }
    ]
    expect(restore('<NAME_EXT_1234abcd>X und <NAME_EXT_1234abcd>', e)).toBe('Berta und Anna')
    expect(restore('{{km}}_vorname, {{km}}', e)).toBe('Katrin, Katrin Berger')
  })

  test('first mapping per placeholder wins', () => {
    expect(restore('{{km}}', [{ placeholder: '{{km}}', original: 'A' }, { placeholder: '{{km}}', original: 'B' }])).toBe('A')
  })

  test('unknown placeholders stay, empty mapping is a no-op', () => {
    expect(restore('{{x}} text', [])).toBe('{{x}} text')
    expect(restore('{{x}}', [{ placeholder: '', original: 'boom' }])).toBe('{{x}}')
  })
})

describe('addCandidate', () => {
  const S = {
    vorname: 'Lena',
    nachname: 'Sommerfeld',
    angehoerige: [{ rolle: 'mutter', vorname: 'Katrin' }]
  }
  const rules = buildRules(S)

  test('a manual NAME_EXT equal to a relative name shares the relative placeholder', () => {
    const c = addCandidate(rules, { text: 'Katrin', kind: 'NAME_EXT', source: 'manual' }, semanticStyle)
    expect(c.placeholder).toBe('{{km_vorname}}')
    expect(c.kind).toBe('NAME_EXT')
    expect(c.source).toBe('manual')
  })

  test('reuse is limited to the same kindGroup', () => {
    const c = addCandidate(rules, { text: 'Lena', kind: 'LOCATION_CITY' }, semanticStyle)
    expect(c.placeholder).toBe('{{ort1}}')
  })

  test('semantic numbering per prefix', () => {
    const c1 = addCandidate(rules, { text: 'Dr. Winter', kind: 'NAME_DOCTOR' }, semanticStyle)
    const r2 = [...rules, c1]
    expect(c1.placeholder).toBe('{{person1}}')
    expect(addCandidate(r2, { text: 'Herr Kuhn', kind: 'NAME_EXT' }, semanticStyle).placeholder).toBe('{{person2}}')
    expect(addCandidate(r2, { text: 'Hohenstein', kind: 'LOCATION_CITY' }, semanticStyle).placeholder).toBe('{{ort1}}')
    expect(addCandidate(r2, { text: 'Klinik am See', kind: 'LOCATION_ORGANIZATION' }, semanticStyle).placeholder).toBe('{{institution1}}')
    expect(addCandidate(r2, { text: '01.02.2020', kind: 'DATE' }, semanticStyle).placeholder).toBe('{{datum1}}')
  })

  test('token styles draw style.token(kind)', () => {
    const style = createStyle('presidio', { random: () => '0a1b2c3d' })
    const c = addCandidate(rules, { text: 'Dr. Winter', kind: 'NAME_EXT' }, style)
    expect(c.placeholder).toBe('<NAME_EXT_0a1b2c3d>')
    const g = addCandidate(rules, { text: 'Hohenstein', kind: 'LOCATION_CITY' }, gemtexStyle)
    expect(g.placeholder).toMatch(/^\[\*\* LOCATION_CITY [A-Z0-9]{6} \*\*\]$/)
  })

  test('token styles: same text and group share the placeholder', () => {
    const c1 = addCandidate(rules, { text: 'Dr. Winter', kind: 'NAME_EXT' }, presidioStyle)
    const c2 = addCandidate([...rules, c1], { text: 'Dr. Winter', kind: 'NAME_DOCTOR' }, presidioStyle)
    expect(c2.placeholder).toBe(c1.placeholder)
  })

  test('candidate rule applies and restores through a token style', () => {
    const style = createStyle('gemtex', { random: () => 'ZZ99ZZ' })
    const c = addCandidate([], { text: 'Dr. Winter', kind: 'NAME_EXT' }, style)
    const r = applyRules('Bei Dr. Winter.', [c])
    expect(r.text).toBe('Bei [** NAME_EXT ZZ99ZZ **].')
    expect(restore(r.text, [c])).toBe('Bei Dr. Winter.')
  })

  test('empty text throws', () => {
    expect(() => addCandidate(rules, { text: '  ', kind: 'NAME_EXT' }, semanticStyle)).toThrow()
  })
})
