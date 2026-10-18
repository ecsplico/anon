/** Entity kinds: GeMTeX PII scheme plus CONTACT_IP. */
export const ENTITY_KINDS = [
  'NAME_PATIENT', 'NAME_RELATIVE', 'NAME_DOCTOR', 'NAME_EXT', 'NAME_TITLE',
  'DATE_BIRTH', 'DATE_DEATH', 'DATE', 'AGE',
  'LOCATION_STREET', 'LOCATION_CITY', 'LOCATION_ZIP', 'LOCATION_COUNTRY', 'LOCATION_STATE',
  'LOCATION_HOSPITAL', 'LOCATION_ORGANIZATION', 'LOCATION_OTHER',
  'ID', 'CONTACT_PHONE', 'CONTACT_EMAIL', 'CONTACT_FAX', 'CONTACT_URL', 'CONTACT_IP',
  'PROFESSION', 'OTHER'
] as const

export type EntityKind = (typeof ENTITY_KINDS)[number]

export type KindGroup = 'NAME' | 'DATE' | 'AGE' | 'LOCATION' | 'ID' | 'CONTACT' | 'PROFESSION' | 'OTHER'

/** Prefix before the first `_` (`DATE_BIRTH` -> `DATE`, `ID` -> `ID`). */
export function kindGroup(k: EntityKind): KindGroup {
  const i = k.indexOf('_')
  return (i < 0 ? k : k.slice(0, i)) as KindGroup
}

export type DetectorName = 'rules' | 'regex' | 'onnx' | 'llm' | 'manual'

export type Span = {
  start: number
  end: number
  text: string
  kind: EntityKind
  score: number
  detector: DetectorName
  placeholder?: string
}

export type Entity = {
  placeholder: string
  original: string
  kind: EntityKind
  detector: DetectorName
  occurrences: number
}

export type Hit = { placeholder: string; original: string; start: number; end: number }

export type Angehoeriger = { rolle: string; vorname?: string; nachname?: string }

export type Stammdaten = {
  vorname: string
  nachname: string
  geburtsdatum?: string // dd.mm.yyyy
  koseformen?: string[]
  angehoerige?: Angehoeriger[]
  schule?: string
  wohnort?: string
  weitere?: { text: string; placeholder: string }[]
}

export type Rule = {
  placeholder: string
  original: string
  kind: EntityKind
  pattern: RegExp
  source: DetectorName
}
