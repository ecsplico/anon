import { describe, expect, test } from 'bun:test'
import { REGEX_ENTITIES, regexDetector } from '../src/detectors/regex.ts'
import type { EntityKind, Span } from '../src/types.ts'

async function run(text: string, entities?: EntityKind[], masked = text): Promise<Span[]> {
  return regexDetector(entities ? { entities } : {}).detect(text, { masked })
}
const found = (spans: Span[], kind: EntityKind) => spans.filter(s => s.kind === kind).map(s => s.text)

describe('phone and fax', () => {
  test('DE numbers with and without +', async () => {
    const spans = await run('Tel. +49 30 1234567, mobil 0171 1234567, Festnetz 030 7654321.')
    expect(found(spans, 'CONTACT_PHONE')).toEqual(['+49 30 1234567', '0171 1234567', '030 7654321'])
  })

  test('AT and CH numbers through their country prefix', async () => {
    const spans = await run('Praxis +43 1 5871234 und +41 44 668 18 00 sowie +43 664 1234567.')
    expect(found(spans, 'CONTACT_PHONE')).toEqual(['+43 1 5871234', '+41 44 668 18 00', '+43 664 1234567'])
  })

  test('fax context gives CONTACT_FAX', async () => {
    const spans = await run('Tel. +49 30 1234567, Fax: 030 7654321.')
    expect(found(spans, 'CONTACT_PHONE')).toEqual(['+49 30 1234567'])
    expect(found(spans, 'CONTACT_FAX')).toEqual(['030 7654321'])
  })

  test('fax label belongs to the number that follows it only', async () => {
    const spans = await run('Fax 030 1234567 Tel 030 7654321')
    expect(found(spans, 'CONTACT_FAX')).toEqual(['030 1234567'])
    expect(found(spans, 'CONTACT_PHONE')).toEqual(['030 7654321'])
  })

  test('a bare year or short number is not a phone number', async () => {
    const spans = await run('Im Jahr 2024 wurde Nr. 12345 vergeben.')
    expect(spans).toEqual([])
  })

  test('digits of a date, IBAN, email or invalid IP are not reported as phone', async () => {
    const spans = await run(
      'geb. 12.03.2011, IBAN DE89370400440532013000, Mail max1234567@example.de, Host 999.1.1.1'
    )
    expect(found(spans, 'CONTACT_PHONE')).toEqual([])
    expect(found(spans, 'CONTACT_FAX')).toEqual([])
  })
})

describe('IBAN and KVNR', () => {
  test('a valid IBAN (with and without spaces) is detected', async () => {
    expect(found(await run('IBAN DE89 3704 0044 0532 0130 00 ok'), 'ID')).toEqual(['DE89 3704 0044 0532 0130 00'])
    expect(found(await run('IBAN DE89370400440532013000 ok'), 'ID')).toEqual(['DE89370400440532013000'])
  })

  test('a checksum-invalid IBAN is not detected', async () => {
    expect(found(await run('Falsch DE89 3704 0044 0532 0130 01 hier.'), 'ID')).toEqual([])
  })

  test('trailing capitals are not swallowed into the IBAN', async () => {
    expect(found(await run('IBAN DE89 3704 0044 0532 0130 00 ABC.'), 'ID')).toEqual(['DE89 3704 0044 0532 0130 00'])
  })

  test('Krankenversichertennummer', async () => {
    const spans = await run('KVNR A123456789 hier')
    expect(spans).toHaveLength(1)
    expect(spans[0]).toMatchObject({ kind: 'ID', text: 'A123456789', score: 0.7, detector: 'regex' })
  })
})

describe('email, URL, IP', () => {
  test('email including umlaut local part and domain', async () => {
    const spans = await run('Mail an anna.müller@bücher-müller.example.de oder anna.mueller@example.de.')
    expect(found(spans, 'CONTACT_EMAIL')).toEqual(['anna.müller@bücher-müller.example.de', 'anna.mueller@example.de'])
  })

  test('URLs lose trailing sentence punctuation', async () => {
    const spans = await run('Siehe http://example.de/a?b=1, und www.example.de.')
    expect(found(spans, 'CONTACT_URL')).toEqual(['http://example.de/a?b=1', 'www.example.de'])
  })

  test('IPv4 with octets up to 255 only', async () => {
    const spans = await run('Server 192.168.1.10 und 999.1.1.1 und Version 1.2.3.4.5')
    expect(found(spans, 'CONTACT_IP')).toEqual(['192.168.1.10'])
  })
})

describe('dates', () => {
  test('numeric, ISO and written-out month forms', async () => {
    const spans = await run('geb. 12.03.2011, 2011-03-12, am 5. März 2011 und 12 Sept. 2011')
    expect(found(spans, 'DATE')).toEqual(['12.03.2011', '2011-03-12', '5. März 2011', '12 Sept. 2011'])
  })

  test('two-digit year form', async () => {
    expect(found(await run('am 3.5.11 gesehen'), 'DATE')).toEqual(['3.5.11'])
  })
})

describe('salutation surname', () => {
  test('Frau Dr. med. X yields only the surname', async () => {
    const spans = await run('Frau Dr. med. Müller-Lüdenscheid sah den Patienten.')
    expect(spans).toHaveLength(1)
    expect(spans[0]).toMatchObject({ kind: 'NAME_EXT', text: 'Müller-Lüdenscheid', score: 0.6 })
  })

  test('Herr Prof. Dr. and a sentence-final surname', async () => {
    const spans = await run('Herrn Prof. Dr. Schmidt. Danach Frau Becker.')
    expect(found(spans, 'NAME_EXT')).toEqual(['Schmidt', 'Becker'])
  })

  test('a salutation without a following capitalised word yields nothing', async () => {
    expect(await run('Die Frau kam und der Herr ging.')).toEqual([])
  })
})

describe('options and masking', () => {
  test('entities option restricts recognizers; default list covers all kinds', async () => {
    const text = 'Mail a@example.de am 12.03.2011'
    expect(found(await run(text, ['DATE']), 'CONTACT_EMAIL')).toEqual([])
    expect(found(await run(text, ['DATE']), 'DATE')).toEqual(['12.03.2011'])
    const kinds = new Set((await run(text)).map(s => s.kind))
    expect([...kinds].every(k => REGEX_ENTITIES.includes(k))).toBe(true)
  })

  test('masked placeholders are skipped; offsets index into the original', async () => {
    const text = 'Mail {{email1}} und b@example.de'
    const masked = text.replace('{{email1}}', ' '.repeat('{{email1}}'.length))
    const spans = await run(text, undefined, masked)
    expect(spans).toHaveLength(1)
    expect(text.slice(spans[0]!.start, spans[0]!.end)).toBe('b@example.de')
    expect(spans[0]!.text).toBe('b@example.de')
  })

  test('spans are sorted by start', async () => {
    const spans = await run('12.03.2011 a@example.de 030 1234567')
    expect(spans.map(s => s.start)).toEqual([...spans.map(s => s.start)].sort((a, b) => a - b))
  })
})
