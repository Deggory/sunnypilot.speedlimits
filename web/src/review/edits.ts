// Changes to the frame on screen: type, value, road and details of the selected box, boxes added (drawn or
// taken from the second reading) and removed, drawing mode and second reading on or off.
import {byId} from '../common/dom';
import {familyForHint, familyHasValue} from './catalog';
import {narrowLayout, writeLocal} from './dom';
import {currentRow, selectedDetection, workOf} from './frames';
import {isBox} from './geometry';
import {hintsOf, render, state, store} from './state';
import type {Box, Detection, FrameEdit, FrameRow} from './types';

/** The frame was changed here: it waits for a new confirmation. */
export function changed(row: FrameRow): void {
  const s = workOf(row);
  s.review_state = 'pending';
  s.saved = false;
  s.error = null;
  s.version = (s.version || 0) + 1;
  state.withdrawArmed = null;
  if (state.notice && state.notice.key === row.frame_key) state.notice = null;
  store();
  render();
}

/** Applies a change to the selected box, if there is one, and marks the frame changed. */
function editSelected(change: (d: Detection) => void): void {
  const d = selectedDetection();
  const row = currentRow();
  if (!d || !row) return;
  change(d);
  changed(row);
}

export function setFamily(id: string): void {
  editSelected((d) => {
    d.family = id;
    if (!familyHasValue(id)) d.value_kph = null;
    d.value_state = d.value_kph === null || d.value_kph === undefined ? 'unknown' : 'reviewed';
    byId('valueError').hidden = true;
  });
}

export function setValue(value: number | null): void {
  editSelected((d) => {
    if (!familyHasValue(d.family)) d.family = 'maximum_speed';
    d.value_kph = value;
    d.value_state = value === null ? 'unknown' : 'reviewed';
    byId('valueError').hidden = true;
  });
}

export function setField<K extends keyof Detection>(key: K, value: Detection[K]): void {
  editSelected((d) => {
    d[key] = value;
  });
}

export function removeBox(index: number): void {
  const row = currentRow();
  if (!row) return;
  const work = workOf(row);
  if (index < 0 || index >= work.detections.length) return;
  work.detections.splice(index, 1);
  state.selected = -1;
  changed(row);
}

export function removeSelected(): void {
  removeBox(state.selected);
}

/** A detection id no other box of the frame has: manual-<time>[-n]. */
function manualId(work: FrameEdit): string {
  const stamp = Date.now();
  const taken = new Set(work.detections.map((d) => d.detection_id));
  let id = 'manual-' + stamp;
  for (let n = 1; taken.has(id); n++) id = 'manual-' + stamp + '-' + n;
  return id;
}

// A box added by the reviewer: drawn by hand or taken from the second reading. Same fields either way.
export function addManualBox(row: FrameRow, box: Box, family: string, value: number | null): void {
  const work = workOf(row);
  work.detections.push({
    bbox_xyxy: box,
    family,
    value_kph: value,
    value_state: value === null ? 'unknown' : 'reviewed',
    score: null,
    model_label: null,
    class_id: null,
    review_state: 'pending',
    applicability: 'unresolved',
    supplementary_panel: 'unknown',
    detection_id: manualId(work),
  });
  state.selected = work.detections.length - 1;
  state.drawing = false;
  changed(row);
}

/** "Usa": the sign of the second reading becomes a box of the frame. */
export function useHint(index: number): void {
  const row = currentRow();
  if (!row) return;
  const hint = hintsOf(row.frame_key)[index];
  if (!hint || !isBox(hint.box)) return;
  const family = familyForHint(hint.type);
  const value = familyHasValue(family) && Number.isInteger(hint.value_kph) ? hint.value_kph ?? null : null;
  addManualBox(row, [...hint.box], family, value);
}

// On a computer the panel scrolls so that values, type and road of the chosen box are in sight.
export function revealEditor(): void {
  if (narrowLayout()) return;
  byId('roads').scrollIntoView({block: 'nearest'});
}

export function selectBox(index: number): void {
  state.selected = index;
  state.drawing = false;
  render();
  revealEditor();
}

export function toggleDrawing(): void {
  state.drawing = !state.drawing;
  state.selected = -1;
  render();
}

export function toggleHints(): void {
  state.showHints = !state.showHints;
  byId<HTMLInputElement>('showHints').checked = state.showHints;
  writeLocal('nnslr-show-hints', state.showHints ? '1' : '0');
  render();
}
