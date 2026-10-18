import type { DetectorName, Span, Stammdaten } from '../types.ts'

export type DetectContext = {
  /** `text` with every placeholder of the active style replaced by spaces of equal length. */
  masked: string
  stammdaten?: Stammdaten
}

/** A detector proposes spans; offsets index into the original text (same length as `ctx.masked`). */
export type Detector = {
  name: DetectorName
  detect(text: string, ctx: DetectContext): Promise<Span[]>
}
