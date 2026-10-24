// What a box can be (families, values, roads) and what the second reading says, with the English words the
// page shows for them (the same as the dataset section of the Train tab, nnslr_tools.panel.FAMILY_LABELS).
import type {Detection, Hint} from './types';

export interface Family {
  id: string;
  label: string;
  hasValue: boolean;
  /** Shown smaller, under the usual choices. */
  minor: boolean;
}

// The first five are the usual choices; the minor ones are shown smaller under them.
export const FAMILIES: readonly Family[] = [
  {id: 'maximum_speed', label: 'Speed limit', hasValue: true, minor: false},
  {id: 'cancellation', label: 'End of limit', hasValue: true, minor: false},
  {id: 'unreadable', label: 'Unreadable', hasValue: false, minor: false},
  {id: 'variable_display', label: 'Variable display', hasValue: true, minor: false},
  {id: 'zone', label: 'Zone', hasValue: true, minor: false},
  {id: 'road_marking_candidate', label: 'Road marking', hasValue: true, minor: true},
  {id: 'other_sign', label: 'Other sign', hasValue: false, minor: true},
  {id: 'not_a_sign', label: 'Not a sign', hasValue: false, minor: true},
];
export const QUICK_VALUES: readonly number[] = [20, 30, 40, 50, 60, 70, 80, 100, 120];
export const ROADS: readonly {id: string; label: string}[] = [
  {id: 'own_road', label: 'Ours'},
  {id: 'other_road', label: 'Another'},
  {id: 'unresolved', label: 'Unclear'},
];

// The selects and the text field of "Sign details" (their ids are the field names).
export const SELECT_FIELDS = ['visibility', 'occlusion', 'supplementary_panel'] as const;
export const BOOLEAN_FIELDS = ['temporary', 'variable_display'] as const;
export type SelectField = typeof SELECT_FIELDS[number];
export type BooleanField = typeof BOOLEAN_FIELDS[number];

interface HintType {
  id: string;
  label: string;
  short: string;
}
const UNSURE: HintType = {id: 'unsure', label: 'Unsure', short: '?'};
const HINT_TYPES: readonly HintType[] = [
  {id: 'maximum_speed', label: 'Speed limit', short: 'limit'},
  {id: 'cancellation', label: 'End of limit', short: 'end'},
  {id: 'zone', label: 'Zone', short: 'zone'},
  {id: 'zone_end', label: 'End of zone', short: 'zone end'},
  {id: 'variable_display', label: 'Variable display', short: 'display'},
  {id: 'road_marking', label: 'Road marking', short: 'road'},
  UNSURE,
];
const CONFIDENCES: ReadonlyMap<string, string> = new Map([['high', 'sure'], ['medium', 'medium'], ['low', 'low']]);

/** A speed limit with its number: limit, zone, variable display. */
export function isSpeedFamily(family: string): boolean {
  return family === 'maximum_speed' || family === 'zone' || family === 'variable_display';
}

export function familyOf(id: string): Family | null {
  return FAMILIES.find((family) => family.id === id) ?? null;
}

export function familyHasValue(id: string): boolean {
  return familyOf(id)?.hasValue ?? false;
}

export function hasValue(d: {value_kph?: number | null | undefined}): boolean {
  return d.value_kph !== null && d.value_kph !== undefined;
}

/** The full description in the panel: "Speed limit · 60 km/h". */
export function describe(d: Detection): string {
  const family = familyOf(d.family);
  let text = family ? family.label : String(d.family || 'No type');
  if (hasValue(d)) text += ' · ' + d.value_kph + ' km/h';
  return text;
}

// Short label on the photo: the list in the panel has the full description.
export function shortName(d: Detection): string {
  const v = d.value_kph;
  const known = hasValue(d);
  switch (d.family) {
    case 'maximum_speed': return known ? String(v) : 'limit ?';
    case 'zone': return known ? 'zone ' + v : 'zone';
    case 'variable_display': return known ? 'variable ' + v : 'variable';
    case 'road_marking_candidate': return known ? 'road ' + v : 'road';
    case 'cancellation': return known ? 'end ' + v : 'end';
    case 'other_sign': return 'other';
    case 'not_a_sign': return 'no';
    default: return '?';
  }
}

/** The colour of a box on the photo (overlay tokens of shared.css). */
export function boxColor(d: Detection): string {
  if (isSpeedFamily(d.family)) return 'var(--ov-speed)';
  if (d.family === 'cancellation') return 'var(--ov-cancel)';
  if (d.family === 'not_a_sign') return 'var(--ov-none)';
  return 'var(--ov-other)';
}

export function hintTypeOf(id: string): HintType {
  return HINT_TYPES.find((type) => type.id === id) ?? UNSURE;
}

export function confidenceLabel(id: string): string {
  return CONFIDENCES.get(id) ?? String(id);
}

export function sideLabel(side: string): string {
  if (side === 'left') return 'on the left';
  if (side === 'right') return 'on the right';
  if (side === 'center' || side === 'centre') return 'in the middle';
  if (side === 'overhead') return 'over the road';
  return side;
}

// Short label drawn next to a hint box, e.g. "60?": its colour says it is the second opinion.
export function hintLabel(h: Hint): string {
  const type = hintTypeOf(h.type);
  let what = type.short;
  if (hasValue(h)) {
    what = h.type === 'maximum_speed' || h.type === 'unsure' ? String(h.value_kph) : type.short + ' ' + h.value_kph;
  } else if (h.type === 'unsure') {
    what = 'unsure';
  }
  return what + '?';
}

// The family a hint becomes when the reviewer presses "Use".
export function familyForHint(type: string): string {
  switch (type) {
    case 'maximum_speed': return 'maximum_speed';
    case 'cancellation': return 'cancellation';
    case 'zone': return 'zone';
    case 'zone_end': return 'cancellation';
    case 'variable_display': return 'variable_display';
    case 'road_marking': return 'road_marking_candidate';
    default: return 'unreadable';
  }
}
