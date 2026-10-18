import { describe, expect, test } from 'bun:test'
import { labelGroup, parseCasJson, score, scoreAll } from '../scripts/grascco.ts'

// Tiny synthetic CAS JSON in the INCEpTION layout (UIMA JSON omits default values: begin 0 is absent).
const TEXT = 'Lena Sommerfeld, geb. 03.04.2010, wohnt in Mühlhausen. Tel. 030 1234567.'

function casJson(text: string, phi: { begin?: number; end: number; kind: string }[]) {
  return {
    '%TYPES': {},
    '%VIEWS': {},
    '%FEATURE_STRUCTURES': [
      { '%ID': 1, '%TYPE': 'uima.cas.Sofa', sofaString: text },
      { '%ID': 2, '%TYPE': 'de.tudarmstadt.ukp.dkpro.core.api.metadata.type.DocumentMetaData', documentId: 'doc-1', documentTitle: 'Titel' },
      { '%ID': 3, '%TYPE': 'de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token', begin: 0, end: 4 },
      ...phi.map((p, i) => ({ '%ID': 10 + i, '%TYPE': 'webanno.custom.PHI', ...p })),
    ],
  }
}

const span = (needle: string, kind: string) => {
  const begin = TEXT.indexOf(needle)
  return { ...(begin > 0 ? { begin } : {}), end: begin + needle.length, kind }
}

describe('parseCasJson', () => {
  const doc = parseCasJson(
    casJson(TEXT, [
      span('Tel. 030 1234567', 'CONTACT_PHONE'),
      span('Lena Sommerfeld', 'NAME_PATIENT'), // begin 0 omitted
      span('Mühlhausen', 'LOCATION_CITY'),
      span('03.04.2010', 'DATE_BIRTH'),
    ]),
  )

  test('reads id and text, ignores non-PHI structures', () => {
    expect(doc.id).toBe('doc-1')
    expect(doc.text).toBe(TEXT)
    expect(doc.gold).toHaveLength(4)
  })

  test('missing begin means 0; gold is sorted; offsets are string indices (umlauts aligned)', () => {
    expect(doc.gold[0]).toEqual({ start: 0, end: 15, label: 'NAME_PATIENT' })
    expect(doc.gold.map((g) => TEXT.slice(g.start, g.end))).toEqual(['Lena Sommerfeld', '03.04.2010', 'Mühlhausen', 'Tel. 030 1234567'])
  })

  test('rejects JSON without feature structures, without sofa, and spans beyond the text', () => {
    expect(() => parseCasJson({})).toThrow('%FEATURE_STRUCTURES')
    expect(() => parseCasJson({ '%FEATURE_STRUCTURES': [] })).toThrow('sofaString')
    expect(() => parseCasJson(casJson('kurz', [{ begin: 0, end: 99, kind: 'ID' }]))).toThrow('Ungültiger PHI-Span')
  })
})

describe('labelGroup', () => {
  test('prefix before underscore', () => {
    expect(labelGroup('DATE_BIRTH')).toBe('DATE')
    expect(labelGroup('DATE')).toBe('DATE')
    expect(labelGroup('NAME_USERNAME')).toBe('NAME')
    expect(labelGroup('LOCATION_HOSPITAL')).toBe('LOCATION')
    expect(labelGroup('ID')).toBe('ID')
    expect(labelGroup('CONTACT_FAX')).toBe('CONTACT')
  })

  test('unknown prefix is OTHER, empty is null', () => {
    expect(labelGroup('WHATEVER_X')).toBe('OTHER')
    expect(labelGroup('')).toBeNull()
  })
})

describe('score', () => {
  const gold = [
    { start: 0, end: 15, label: 'NAME_PATIENT' },
    { start: 26, end: 36, label: 'DATE_BIRTH' },
    { start: 44, end: 54, label: 'LOCATION_CITY' },
    { start: 56, end: 72, label: 'CONTACT_PHONE' },
  ]
  const predicted = [
    { start: 0, end: 15, kind: 'NAME_EXT' }, // exact, same group (NAME) despite other kind
    { start: 26, end: 36, kind: 'DATE' }, // exact, DATE group
    { start: 44, end: 50, kind: 'LOCATION_CITY' }, // partial overlap only
    { start: 60, end: 72, kind: 'CONTACT_PHONE' }, // partial overlap only
    { start: 16, end: 20, kind: 'NAME_EXT' }, // false positive
  ]

  test('strict: exact offsets and group', () => {
    const { groups, total } = score(gold, predicted, { mode: 'strict' })
    expect(groups.NAME).toMatchObject({ gold: 1, predicted: 2, tp: 1, fp: 1, fn: 0, precision: 0.5, recall: 1 })
    expect(groups.DATE).toMatchObject({ tp: 1, fp: 0, fn: 0, precision: 1, recall: 1, f1: 1 })
    expect(groups.LOCATION).toMatchObject({ tp: 0, fp: 1, fn: 1, precision: 0, recall: 0, f1: 0 })
    expect(groups.CONTACT).toMatchObject({ tp: 0, fp: 1, fn: 1 })
    expect(total).toMatchObject({ gold: 4, predicted: 5, tp: 2, fp: 3, fn: 2, recall: 0.5, precision: 0.4 })
  })

  test('overlap: any shared character in the same group counts', () => {
    const { groups, total } = score(gold, predicted, { mode: 'overlap' })
    expect(groups.LOCATION).toMatchObject({ tp: 1, fp: 0, fn: 0, recall: 1, precision: 1 })
    expect(groups.CONTACT).toMatchObject({ tp: 1, fn: 0, recall: 1 })
    expect(groups.NAME).toMatchObject({ tp: 1, fp: 1 })
    expect(total).toMatchObject({ tp: 4, fn: 0, recall: 1, fp: 1 })
  })

  test('overlap in another group is no match', () => {
    const r = score([{ start: 0, end: 10, label: 'NAME_DOCTOR' }], [{ start: 0, end: 10, kind: 'LOCATION_CITY' }], { mode: 'overlap' })
    expect(r.groups.NAME).toMatchObject({ tp: 0, fn: 1 })
    expect(r.groups.LOCATION).toMatchObject({ predicted: 1, fp: 1 })
  })

  test('identical predictions from several detectors count once; empty groups give zeros', () => {
    const dup = [{ start: 0, end: 15, kind: 'NAME_EXT' }, { start: 0, end: 15, kind: 'NAME_PATIENT' }]
    const r = score([{ start: 0, end: 15, label: 'NAME_PATIENT' }], dup, { mode: 'strict' })
    expect(r.groups.NAME).toMatchObject({ predicted: 1, tp: 1, fp: 0 })
    expect(r.groups.AGE).toMatchObject({ gold: 0, predicted: 0, precision: 0, recall: 0, f1: 0 })
  })

  test('scoreAll sums counts across documents before forming ratios', () => {
    const doc1 = { gold: [{ start: 0, end: 5, label: 'ID' }], predicted: [{ start: 0, end: 5, kind: 'ID' }] }
    const doc2 = { gold: [{ start: 0, end: 5, label: 'ID' }], predicted: [] }
    const r = scoreAll([doc1, doc2], { mode: 'strict' })
    expect(r.groups.ID).toMatchObject({ gold: 2, tp: 1, fn: 1, recall: 0.5, precision: 1 })
    // Same offsets in different documents must not collide.
    expect(r.total).toMatchObject({ gold: 2, predicted: 1, tp: 1 })
  })
})
