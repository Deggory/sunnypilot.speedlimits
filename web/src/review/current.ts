// The frame on screen, as computed values the components share: its row, its working copy, its state, the
// box selected in it, its place in the filter.
import {computed, watchEffect} from 'vue';
import {aiConfirmed, currentRow, frameStatus, isConfirmed, reasonsOf, visibleRows, workOf} from './frames';
import {report, state} from './store';
import type {Detection, FrameEdit, FrameRow, FrameStatus} from './types';

export const row = computed<FrameRow | null>(() => currentRow());
/** The working copy of the frame on screen (ensureCurrentFrame makes sure there is one). */
export const work = computed<FrameEdit | null>(() => (row.value ? state.edits[row.value.frame_key] ?? null : null));
export const status = computed<FrameStatus>(() => (row.value ? frameStatus(row.value.frame_key) : 'todo'));
export const selectedBox = computed<Detection | null>(() => work.value?.detections[state.selected] ?? null);

/** Position of the frame in its filter (0 based) and the length of the filter. */
export const position = computed(() => {
  const list = visibleRows();
  const key = row.value?.frame_key;
  return {index: Math.max(0, list.findIndex((item) => item.frame_key === key)), length: list.length};
});

/** Confirmed, suspicious and AI-confirmed frames of the route. */
export const routeCounts = computed(() => {
  let done = 0;
  let suspicious = 0;
  let byAi = 0;
  for (const item of report.rows) {
    if (isConfirmed(item.frame_key)) done += 1;
    if (reasonsOf(item.frame_key).length > 0) suspicious += 1;
    if (aiConfirmed(item.frame_key)) byAi += 1;
  }
  return {done, suspicious, byAi, total: report.rows.length};
});

/** Before each drawing: a frame is on screen when the report has frames, it has its working copy (the
 * proposals until the reviewer changes it), and the selection points at one of its boxes. */
export function ensureCurrentFrame(): void {
  watchEffect(() => {
    let shown = currentRow();
    if (shown === null) {
      shown = report.rows[0] ?? null;
      if (shown === null) return;
      state.currentKey = shown.frame_key;
    }
    const copy = workOf(shown);
    if (state.selected >= copy.detections.length) state.selected = -1;
  });
}
