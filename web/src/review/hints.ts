// "Seconda lettura" in the panel: what another model saw in the frame, one row per sign, with "Usa" to turn
// it into a box. Only on the frames the second opinion has looked at. The rows as data for SecondReading.vue.
import {confidenceLabel, familyForHint, hasValue, hintTypeOf, sideLabel} from './catalog';
import {overlap, sameBox} from './geometry';
import {report} from './store';
import type {FrameEdit, Hint} from './types';

export interface HintRow {
  /** The index of the first reading of the group in the frame's hints ("Usa" takes that one). */
  index: number;
  text: string;
  confidence: string;
  confidenceText: string;
  /** "lettura A · a destra", or empty. */
  sub: string;
  note: string;
  used: boolean;
  /** The accessible name of "Usa". */
  useLabel: string;
}

// Readings A and B of the same sign become one row: same type and value, boxes overlapping.
function groupReadings(hints: readonly Hint[]): number[][] {
  const groups: number[][] = [];
  hints.forEach((hint, i) => {
    let found: number[] | null = null;
    for (const group of groups) {
      const first = hints[group[0] ?? -1];
      const same = first && first.type === hint.type && first.value_kph === hint.value_kph;
      if (same && overlap(first.box, hint.box) > 0.3) found = group;
    }
    if (found === null) groups.push([i]);
    else found.push(i);
  });
  return groups;
}

/** True when a box of the frame already is this sign: same box, or same type and value over it. */
function alreadyUsed(work: FrameEdit, h: Hint): boolean {
  return work.detections.some((d) => sameBox(d.bbox_xyxy, h.box)
    || (overlap(d.bbox_xyxy, h.box) > 0.3 && d.family === familyForHint(h.type) && d.value_kph === h.value_kph));
}

function hintRow(hints: readonly Hint[], group: readonly number[], work: FrameEdit): HintRow | null {
  const i = group[0];
  const h = i === undefined ? undefined : hints[i];
  if (i === undefined || h === undefined) return null;
  let text = hintTypeOf(h.type).label;
  if (hasValue(h)) text += ' · ' + h.value_kph + ' km/h';
  const passes: string[] = [];
  for (const k of group) {
    const pass = hints[k]?.pass;
    if (pass && !passes.includes(pass)) passes.push(pass);
  }
  const parts: string[] = [];
  if (passes.length === 1) parts.push('lettura ' + passes[0]);
  if (passes.length > 1) parts.push('letture ' + passes.join(' e '));
  if (h.side) parts.push(sideLabel(h.side));
  return {
    index: i,
    text,
    confidence: h.confidence,
    confidenceText: confidenceLabel(h.confidence),
    sub: parts.join(' · '),
    note: h.note ?? '',
    used: alreadyUsed(work, h),
    useLabel: 'Usa questo: ' + text + (h.pass ? ' (lettura ' + h.pass + ')' : ''),
  };
}

export function hintRows(hints: readonly Hint[], work: FrameEdit): HintRow[] {
  return groupReadings(hints).map((group) => hintRow(hints, group, work)).filter((row) => row !== null);
}

/** "Seconda lettura su 148 fotogrammi di questa route · 2 voci scartate perché non valide." */
export function statsLine(): string {
  const stats = report.hintStats;
  if (!stats) return '';
  let line = 'Seconda lettura su ' + stats.frames + (stats.frames === 1 ? ' fotogramma' : ' fotogrammi')
    + ' di questa route';
  const skipped = (stats.skipped_lines || 0) + (stats.skipped_signs || 0) + (stats.skipped_files || 0);
  if (skipped === 1) line += ' · 1 voce scartata perché non valida';
  if (skipped > 1) line += ' · ' + skipped + ' voci scartate perché non valide';
  return line + '.';
}
