// "Second reading" in the panel: what another model saw in the frame, one row per sign, with "Use" to turn
// it into a box. Only on the frames the second opinion has looked at. The rows as data for SecondReading.vue.
import {confidenceLabel, familyForHint, hasValue, hintTypeOf, sideLabel} from './catalog';
import {overlap, sameBox} from './geometry';
import {report} from './store';
import type {FrameEdit, Hint} from './types';

export interface HintRow {
  /** The index of the first reading of the group in the frame's hints ("Use" takes that one). */
  index: number;
  text: string;
  confidence: string;
  confidenceText: string;
  /** "reading A · on the right", or empty. */
  sub: string;
  note: string;
  used: boolean;
  /** The accessible name of "Use". */
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
  if (passes.length === 1) parts.push('reading ' + passes[0]);
  if (passes.length > 1) parts.push('readings ' + passes.join(' and '));
  if (h.side) parts.push(sideLabel(h.side));
  return {
    index: i,
    text,
    confidence: h.confidence,
    confidenceText: confidenceLabel(h.confidence),
    sub: parts.join(' · '),
    note: h.note ?? '',
    used: alreadyUsed(work, h),
    useLabel: 'Use this: ' + text + (h.pass ? ' (reading ' + h.pass + ')' : ''),
  };
}

export function hintRows(hints: readonly Hint[], work: FrameEdit): HintRow[] {
  return groupReadings(hints).map((group) => hintRow(hints, group, work)).filter((row) => row !== null);
}

/** "Second reading on 148 frames of this route · 2 entries dropped as invalid." */
export function statsLine(): string {
  const stats = report.hintStats;
  if (!stats) return '';
  let line = 'Second reading on ' + stats.frames + (stats.frames === 1 ? ' frame' : ' frames')
    + ' of this route';
  const skipped = (stats.skipped_lines || 0) + (stats.skipped_signs || 0) + (stats.skipped_files || 0);
  if (skipped === 1) line += ' · 1 entry dropped as invalid';
  if (skipped > 1) line += ' · ' + skipped + ' entries dropped as invalid';
  return line + '.';
}
