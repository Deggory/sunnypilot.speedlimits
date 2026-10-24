// Moving between frames: arrows, the frame after a confirmation, the filters, the frame to start from.
import {byId} from '../common/dom';
import {narrowLayout, one, readLocal, writeLocal} from './dom';
import {confirmedFilter, currentRow, frameStatus, isQualityFilter, matchesFilter, visibleRows} from './frames';
import {indexOf, lastKeyStorage, render, rowOf, rows, state} from './state';
import {scrollTimelineToCurrent} from './timeline';
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
  state.filterNote = '';
  state.withdrawArmed = null;
  writeLocal(lastKeyStorage, key);
  render();
  scrollTimelineToCurrent();
  revealStage();
}

// On a phone the panel is below the photo: bring the photo back into view after a frame change.
function revealStage(): void {
  if (!narrowLayout()) return;
  const header = one('.topbar');
  const top = getComputedStyle(header).position === 'sticky' ? header.getBoundingClientRect().bottom : 0;
  const rect = byId('stage').getBoundingClientRect();
  if (rect.top < top || rect.top > window.innerHeight * 0.6) {
    window.scrollTo(0, window.scrollY + rect.top - top - 8);
  }
}

// The frame to show after a confirmation: the next one still to do under the current filter.
function nextAfterConfirm(key: string): string | null {
  for (const row of rows.slice(indexOf(key) + 1)) {
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
    render();
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
    render();
    return;
  }
  // The next frame of the filter after this one, else the first one before it.
  const start = indexOf(row.frame_key);
  const target = rows.slice(start + 1).find((item) => matchesFilter(item, filter))
    ?? rows.slice(0, Math.max(start, 0)).find((item) => matchesFilter(item, filter));
  if (target === undefined) {
    if (filter === 'suspicious') state.filterNote = 'Nessun fotogramma sospetto da rivedere.';
    else if (isQualityFilter(filter)) state.filterNote = 'Nessun fotogramma confermato con questo dubbio.';
    else if (filter === 'ai') state.filterNote = 'Nessun fotogramma confermato da un’AI.';
    else state.filterNote = 'Nessun fotogramma corrisponde a questo filtro.';
    render();
    return;
  }
  openFrame(target.frame_key);
}

/** The frame last open in this browser, else the first of the filter, else the first one. */
export function pickStartFrame(): string | null {
  const last = readLocal(lastKeyStorage, null);
  if (last !== null && rowOf(last)) return last;
  const first = rows.find((row) => matchesFilter(row, state.filter)) ?? rows[0];
  return first ? first.frame_key : null;
}
