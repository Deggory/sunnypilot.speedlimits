// Moving between frames: arrows, the frame after a confirmation, the filters, the frame to start from.
import {nextTick} from 'vue';
import {readLocal, writeLocal} from './dom';
import {confirmedFilter, currentRow, frameStatus, isQualityFilter, matchesFilter, visibleRows, workOf} from './frames';
import {revealStage, scrollTimelineToCurrent} from './scroll';
import {indexOf, report, rowOf, state} from './store';
import type {Filter} from './types';

export function go(step: number): void {
  const list = visibleRows();
  const index = list.findIndex((row) => row.frame_key === state.currentKey);
  const target = list[index + step];
  if (index + step < 0 || target === undefined) return;
  openFrame(target.frame_key);
}

export function openFrame(key: string): void {
  state.currentKey = key;
  state.selected = -1;
  state.drawing = false;
  state.drag = null;
  state.draft = null;
  state.filterNote = '';
  state.withdrawArmed = null;
  const row = rowOf(key);
  if (row !== null) workOf(row);
  writeLocal(report.lastKeyStorage, key);
  // Once the page shows the frame: its tick in sight, and on a phone the photo.
  void nextTick(() => {
    scrollTimelineToCurrent();
    revealStage();
  });
}

// The frame to show after a confirmation: the next one still to do under the current filter.
function nextAfterConfirm(key: string): string | null {
  for (const row of report.rows.slice(indexOf(key) + 1)) {
    if (state.filter === 'all') return row.frame_key;
    if (confirmedFilter(state.filter)) {
      if (matchesFilter(row, state.filter)) return row.frame_key;
      continue;
    }
    if (frameStatus(row.frame_key) === 'done') continue;
    if (matchesFilter(row, state.filter)) return row.frame_key;
  }
  return null;
}

export function advanceAfterConfirm(key: string, savedText: string): void {
  const next = nextAfterConfirm(key);
  if (next === null) {
    state.notice = {key, text: savedText + ' Non restano fotogrammi da rivedere con questo filtro.', error: false};
    return;
  }
  state.notice = {key: next, text: savedText + ' Ecco il prossimo.', error: false};
  openFrame(next);
}

export function setFilter(filter: Filter): void {
  state.filter = filter;
  writeLocal('nnslr-review-filter', filter);
  const row = currentRow();
  if (row === null || matchesFilter(row, filter)) {
    state.filterNote = '';
    return;
  }
  // The next frame of the filter after this one, else the first one before it.
  const start = indexOf(row.frame_key);
  const target = report.rows.slice(start + 1).find((item) => matchesFilter(item, filter))
    ?? report.rows.slice(0, Math.max(start, 0)).find((item) => matchesFilter(item, filter));
  if (target === undefined) {
    if (filter === 'suspicious') state.filterNote = 'Nessun fotogramma sospetto da rivedere.';
    else if (isQualityFilter(filter)) state.filterNote = 'Nessun fotogramma confermato con questo dubbio.';
    else if (filter === 'ai') state.filterNote = 'Nessun fotogramma confermato da un’AI.';
    else state.filterNote = 'Nessun fotogramma corrisponde a questo filtro.';
    return;
  }
  openFrame(target.frame_key);
}

/** A click on a filter of the top bar or on a doubt counter of the timeline card. */
export function pickFilter(filter: Filter): void {
  // A second click on a pressed doubt counter goes back to the frames to review.
  if (isQualityFilter(filter) && state.filter === filter) setFilter('todo');
  else setFilter(filter);
}

/** The frame last open in this browser, else the first of the filter, else the first one. */
export function pickStartFrame(): string | null {
  const last = readLocal(report.lastKeyStorage, null);
  if (last !== null && rowOf(last)) return last;
  const first = report.rows.find((row) => matchesFilter(row, state.filter)) ?? report.rows[0];
  return first ? first.frame_key : null;
}
