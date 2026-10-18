import { describe, expect, test } from 'bun:test'
import { fakerDE } from '@faker-js/faker'
import { FakeNames, type FakerLike } from '../src/fake-names.ts'

const person = fakerDE.rawDefinitions.person!
const male = new Set([...(person.first_name?.male ?? []), ...(person.first_name?.generic ?? [])])
const female = new Set([...(person.first_name?.female ?? []), ...(person.first_name?.generic ?? [])])
const surnames = new Set(person.last_name?.generic ?? [])

/** Faker stub drawing from fixed queues (cycling the last value once a queue runs dry). */
function stub(first: string[], last: string[]): FakerLike {
  const next = (queue: string[]) => {
    const v = queue.length > 1 ? queue.shift()! : queue[0]!
    return v
  }
  return { person: { firstName: () => next(first), lastName: () => next(last) } }
}

describe('gender and structure', () => {
  test('male full name: male given name, surname from the list', () => {
    const out = new FakeNames('').replacement('Thomas Schneider')
    const [given, ...family] = out.split(' ')
    expect(male.has(given!)).toBe(true)
    expect(surnames.has(family.join(' '))).toBe(true)
    expect(out).not.toBe('Thomas Schneider')
  })

  test('female full name: female given name', () => {
    const out = new FakeNames('').replacement('Anna Müller')
    expect(female.has(out.split(' ')[0]!)).toBe(true)
  })

  test('a salutation cue beats the name lookup', () => {
    const out = new FakeNames('').replacement('Thomas Schneider', 'Frau ')
    expect(female.has(out.split(' ')[0]!)).toBe(true)
    const out2 = new FakeNames('').replacement('Anna Müller', 'bei Herrn ')
    expect(male.has(out2.split(' ')[0]!)).toBe(true)
  })

  test('with a cue, a single token is a surname', () => {
    expect(surnames.has(new FakeNames('').replacement('Müller', 'Frau '))).toBe(true)
  })

  test('a single known given name without cue stays a given name', () => {
    const out = new FakeNames('').replacement('Thomas')
    expect(male.has(out)).toBe(true)
  })

  test('title prefix is preserved and Herr/Frau inside the span fix the gender', () => {
    const herr = new FakeNames('').replacement('Herrn Müller')
    expect(herr.startsWith('Herrn ')).toBe(true)
    expect(surnames.has(herr.slice('Herrn '.length))).toBe(true)
    const frau = new FakeNames('').replacement('Frau Dr. Anna Müller')
    expect(frau.startsWith('Frau Dr. ')).toBe(true)
  })

  test('titles only: returned unchanged', () => {
    expect(new FakeNames('').replacement('Dr. med.')).toBe('Dr. med.')
  })

  test('comma form keeps its shape', () => {
    const n = new FakeNames('')
    const out = n.replacement('Müller, Anna')
    const [family, given] = out.split(', ')
    expect(surnames.has(family!)).toBe(true)
    expect(female.has(given!)).toBe(true)
    // shares the surname with the plain form
    expect(n.replacement('Anna Müller').endsWith(family!)).toBe(true)
  })

  test('uppercase originals give uppercase fakes', () => {
    const out = new FakeNames('').replacement('ANNA MÜLLER')
    expect(out).toBe(out.toUpperCase())
    expect(out).not.toBe('ANNA MÜLLER')
  })

  test('unknown given name falls back to the injected random', () => {
    const lowFirst = new FakeNames('', { random: () => 0.1 }).replacement('Zyxwv Müller')
    const highFirst = new FakeNames('', { random: () => 0.9 }).replacement('Zyxwv Müller')
    expect(male.has(lowFirst.split(' ')[0]!)).toBe(true)
    expect(female.has(highFirst.split(' ')[0]!)).toBe(true)
  })
})

describe('consistency', () => {
  test('family members share the fake surname, given names differ', () => {
    const n = new FakeNames('')
    const anna = n.replacement('Anna Müller').split(' ')
    const peter = n.replacement('Peter Müller').split(' ')
    expect(anna[anna.length - 1]).toBe(peter[peter.length - 1])
    expect(anna[0]).not.toBe(peter[0])
  })

  test('the same original always maps to the same fake', () => {
    const n = new FakeNames('')
    expect(n.replacement('Anna Müller')).toBe(n.replacement('Anna Müller'))
    expect(n.replacement('Müller', 'Frau ')).toBe(n.replacement('Müller', 'Herrn '))
  })
})

describe('collisions', () => {
  test('a candidate contained in the source text is rejected (substring match)', () => {
    const n = new FakeNames('Frau Weber wohnt in Webersberg.', {
      faker: stub(['Hans'], ['Weber', 'Braun'])
    })
    expect(n.replacement('Max Mustermann')).toBe('Hans Braun')
  })

  test('a fake already handed out to another original is rejected', () => {
    const n = new FakeNames('', { faker: stub(['Hans', 'Karl'], ['Braun', 'Braun', 'Koch']) })
    expect(n.replacement('Max Mustermann')).toBe('Hans Braun')
    expect(n.replacement('Eva Beispiel', 'Frau ')).toBe('Karl Koch')
  })

  test('gives up with an error when no unused fake exists', () => {
    const n = new FakeNames('Weber', { faker: stub(['Hans'], ['Weber']) })
    expect(() => n.replacement('Max Mustermann')).toThrow('Fake-Name')
  })
})
