// Frames and their review state: the working copy, the status of a frame, its doubts, the filters.
import {hasValue, isSpeedFamily} from './catalog';
import {clone} from './dom';
import {report, rowOf, state} from './store';
import type {
  ConfirmedCopy, Detection, Filter, FrameEdit, FrameRow, FrameStatus, QualityFilter, SuspiciousReason,
} from './types';

export function currentRow(): FrameRow | null {
  return state.currentKey === null ? null : rowOf(state.currentKey);
}

export function proposalsOf(row: FrameRow): Detection[] {
  return Array.isArray(row.detections) ? row.detections : [];
}

// The page's working copy of a frame: the proposals until the reviewer changes or confirms it.
export function workOf(row: FrameRow): FrameEdit {
  const key = row.frame_key;
  let work = state.edits[key];
  if (!work) {
    const detections = clone(proposalsOf(row));
    // Numbered from left to right, as they appear on the photo.
    detections.sort((a, b) => a.bbox_xyxy[0] - b.bbox_xyxy[0]);
    state.edits[key] = {review_state: 'pending', detections, saved: true, revision: 0};
    // The reactive copy: changes made through it reach the page.
    work = state.edits[key];
  }
  return work as FrameEdit;
}

/** The box selected in the frame on screen. */
export function selectedDetection(): Detection | null {
  const row = currentRow();
  if (!row) return null;
  return workOf(row).detections[state.selected] ?? null;
}

/** The working copy of a review read from the server (or of the last confirmed version, to go back to it). */
export function fromServer(item: ConfirmedCopy): FrameEdit {
  return {
    review_state: item.confirmed ? 'reviewed' : 'pending',
    detections: clone(item.detections ?? []),
    reviewer: item.reviewer,
    reviewer_kind: item.reviewer_kind || 'human',
    channel: item.channel,
    review_timestamp: item.review_timestamp,
    site_group: item.site_group,
    encounter_id: item.encounter_id,
    revision: item.revision,
    saved: true,
    error: null,
    server_confirmed: item.confirmed,
    base: clone(item),
  };
}

// True when the working copy is exactly what the server has confirmed (an agent may edit it without changed()).
export function sameAsServer(s: FrameEdit | undefined): boolean {
  if (!s || !s.base) return false;
  if (JSON.stringify(s.detections) !== JSON.stringify(s.base.detections || [])) return false;
  if ((s.site_group ?? null) !== (s.base.site_group ?? null)) return false;
  return (s.encounter_id ?? null) === (s.base.encounter_id ?? null);
}

export function isConfirmed(key: string): boolean {
  const s = state.edits[key];
  if (!s) return false;
  if (report.apiBase) return s.server_confirmed === true;
  return s.review_state === 'reviewed';
}

export function reasonsOf(key: string): SuspiciousReason[] {
  if (isConfirmed(key)) return [];
  const list = state.suspicious[key];
  return Array.isArray(list) ? list : [];
}

export function frameStatus(key: string): FrameStatus {
  if (report.apiBase && state.inFlight.has(key)) return 'saving';
  const s = state.edits[key];
  if (!s) return 'todo';
  if (s.review_state === 'reviewed') {
    if (!report.apiBase) return 'done';
    if (s.saved === true && s.server_confirmed === true) return 'done';
    return 'error';
  }
  if (s.saved === false) return 'dirty';
  return 'todo';
}

// Confirmed on the server by an AI reviewer (REST API or MCP) and not changed here. The confirmation is final,
// like a person's: this only shows where it comes from (filter, striped tick, save line).
export function aiConfirmed(key: string): boolean {
  const s = state.edits[key];
  return Boolean(report.apiBase && s && s.reviewer_kind === 'ai' && frameStatus(key) === 'done' && sameAsServer(s));
}

export function statusLabel(key: string): string {
  const status = frameStatus(key);
  const s = state.edits[key];
  if (status === 'done' && aiConfirmed(key)) return 'Confermato · AI';
  if (status === 'done') return 'Confermato';
  if (status === 'saving') return 'Salvataggio…';
  if (status === 'error') return s && s.error ? 'Non salvato' : 'In attesa';
  if (status === 'dirty') return s && s.server_confirmed ? 'Da riconfermare' : 'Da confermare';
  if (reasonsOf(key).length > 0) return 'Sospetto';
  return 'Da rivedere';
}

export interface Doubts {
  unknown: number;
  unreadable: number;
  road: number;
}

// Doubts left in a confirmed frame: a speed sign without its number, an unreadable sign, a road not chosen.
export function doubtsOf(key: string): Doubts {
  const out = {unknown: 0, unreadable: 0, road: 0};
  const s = state.edits[key];
  if (!s || !isConfirmed(key)) return out;
  for (const d of s.detections) {
    const speed = isSpeedFamily(d.family);
    if (d.family === 'unreadable') {
      out.unreadable += 1;
    } else if ((speed || d.family === 'road_marking_candidate') && !hasValue(d)) {
      out.unknown += 1;
    }
    if (speed && (!d.applicability || d.applicability === 'unresolved')) out.road += 1;
  }
  return out;
}

export function isQualityFilter(filter: string | undefined): filter is QualityFilter {
  return filter === 'unknown' || filter === 'unreadable' || filter === 'road';
}

// Filters over confirmed frames: a confirmation does not take the frame out of them by itself.
export function confirmedFilter(filter: string): boolean {
  return isQualityFilter(filter) || filter === 'ai';
}

export function isFilter(value: string | null | undefined): value is Filter {
  return value === 'todo' || value === 'suspicious' || value === 'proposals' || value === 'all'
    || (value !== null && value !== undefined && confirmedFilter(value));
}

export function matchesFilter(row: FrameRow, filter: Filter): boolean {
  const key = row.frame_key;
  switch (filter) {
    case 'todo': return frameStatus(key) !== 'done';
    case 'suspicious': return reasonsOf(key).length > 0;
    case 'proposals': return proposalsOf(row).length > 0;
    case 'ai': return aiConfirmed(key);
    case 'unknown': return doubtsOf(key).unknown > 0;
    case 'unreadable': return doubtsOf(key).unreadable > 0;
    case 'road': return doubtsOf(key).road > 0;
    default: return true;
  }
}

// Frames of the current filter, plus the frame on screen so that it never disappears under the reviewer.
export function visibleRows(): FrameRow[] {
  return report.rows.filter((row) => row.frame_key === state.currentKey || matchesFilter(row, state.filter));
}

/** Frames confirmed here and not stored on the server yet. */
export function unsavedKeys(): string[] {
  return report.rows.filter((row) => {
    const s = state.edits[row.frame_key];
    return s && s.saved === false && s.review_state === 'reviewed';
  }).map((row) => row.frame_key);
}
