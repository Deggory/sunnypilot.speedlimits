// The page's data and its one reactive state: the payload embedded by the server (or by write_review for the
// offline report), the working copies of the frames, what is on screen. The modules change this state; the
// components of ReviewApp.vue follow it.
import {reactive} from 'vue';
import {readLocal} from './dom';
import {isFilter} from './frames';
import type {
  Drag, Filter, FrameEdit, FrameRow, Hint, HintStats, LastSave, Notice, Payload, RouteLink, ServerState,
  Suspicious,
} from './types';

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

/** What the payload says, fixed for the life of the page (initStore fills it before the app mounts). */
export interface Report {
  runId: string;
  routeId: string;
  rows: readonly FrameRow[];
  routes: readonly RouteLink[];
  /** Country code: English name, for the country of a frame. */
  countries: Readonly<Record<string, string>>;
  hintStats: HintStats | null;
  routeHasHints: boolean;
  /** With the review server every confirmation is saved in its database; opened as a plain file the page
   * works offline (null). */
  apiBase: string | null;
  storageKey: string;
  lastKeyStorage: string;
}

export const report: Report = {
  runId: '', routeId: '', rows: [], routes: [], countries: {}, hintStats: null, routeHasHints: false, apiBase: null,
  storageKey: '', lastKeyStorage: '',
};

let hintsByFrame: Record<string, Hint[]> = {};
let rowIndex = new Map<string, number>();

/** The export of "Download reviews (JSONL)": its message under the button. */
export interface ExportMessage {
  text: string;
  ok: boolean;
}

export interface ReviewState {
  /** The working copy of every frame touched, by frame key (also what localStorage keeps). */
  edits: Record<string, FrameEdit>;
  suspicious: Suspicious;
  currentKey: string | null;
  /** Index of the selected box in the working copy, -1 for none. */
  selected: number;
  drawing: boolean;
  drag: Drag | null;
  /** The rectangle being drawn on the photo (image pixels), while drawing. */
  draft: {x: number; y: number; width: number; height: number} | null;
  /** The click that follows a drag is ignored until then (ms). */
  ignoreClickUntil: number;
  filter: Filter;
  filterNote: string;
  showHints: boolean;
  /** The timeline card unfolded on a phone. */
  timelineOpen: boolean;
  /** The reviewer field of the top bar, as typed. */
  reviewer: string;
  /** A save needed the reviewer's name: the field is marked until something is typed. */
  reviewerNeeded: boolean;
  /** Counts the requests to put the focus on the reviewer field. */
  reviewerFocus: number;
  /** The "Other value" field holds a number out of range. */
  valueError: boolean;
  exportMessage: ExportMessage | null;
  /** Frames being sent to the server now. */
  inFlight: Set<string>;
  notice: Notice | null;
  storageWarning: string;
  serverWarning: string;
  retrying: boolean;
  /** The frame whose "Withdraw confirmation" was pressed once: the second press withdraws. */
  withdrawArmed: string | null;
  saveCount: number;
  lastSave: LastSave | null;
  server: ServerState;
}

export const state: ReviewState = reactive({
  edits: {},
  suspicious: {},
  currentKey: null,
  selected: -1,
  drawing: false,
  drag: null,
  draft: null,
  ignoreClickUntil: 0,
  filter: 'todo',
  filterNote: '',
  showHints: true,
  timelineOpen: false,
  reviewer: '',
  reviewerNeeded: false,
  reviewerFocus: 0,
  valueError: false,
  exportMessage: null,
  inFlight: new Set<string>(),
  notice: null,
  storageWarning: '',
  serverWarning: '',
  retrying: false,
  withdrawArmed: null,
  saveCount: 0,
  lastSave: null,
  server: 'offline',
});

/** The payload of the page (<script id="payload">), read once before the app mounts. */
export function initStore(raw: Partial<Payload>): void {
  const routeId = raw.route_id ?? '';
  hintsByFrame = isRecord(raw.hints) ? raw.hints as Record<string, Hint[]> : {};
  Object.assign(report, {
    runId: String(raw.run_id),
    routeId,
    rows: Array.isArray(raw.frames) ? raw.frames : [],
    routes: Array.isArray(raw.routes) ? raw.routes : [],
    countries: isRecord(raw.countries) ? raw.countries as Record<string, string> : {},
    hintStats: isRecord(raw.hint_stats) ? raw.hint_stats as unknown as HintStats : null,
    routeHasHints: Object.keys(hintsByFrame).length > 0,
    apiBase: raw.server ? '/api/route/' + encodeURIComponent(routeId) : null,
    storageKey: 'nnslr-review-v1-' + String(raw.run_id),
    lastKeyStorage: 'nnslr-review-last-' + String(raw.run_id),
  } satisfies Report);
  rowIndex = new Map(report.rows.map((row, index) => [row.frame_key, index]));
  state.suspicious = isRecord(raw.suspicious) ? raw.suspicious as Suspicious : {};
}

export function indexOf(key: string): number {
  return rowIndex.get(key) ?? -1;
}

export function rowOf(key: string): FrameRow | null {
  const index = rowIndex.get(key);
  return index === undefined ? null : report.rows[index] ?? null;
}

export function hintsOf(key: string): Hint[] {
  const list = hintsByFrame[key];
  return Array.isArray(list) ? list : [];
}

/** Keeps the working copies in this browser; a failure (storage full or off) becomes a warning. */
export function store(): void {
  try {
    localStorage.setItem(report.storageKey, JSON.stringify(state.edits));
    state.storageWarning = '';
  } catch {
    state.storageWarning = 'Saving in this browser is not available: download the JSONL before closing.';
  }
}

/** What this browser kept: the working copies of the frames, reviewer, filter, timeline fold, hints shown. */
export function restore(): void {
  try {
    const saved: unknown = JSON.parse(localStorage.getItem(report.storageKey) || '{}');
    if (!isRecord(saved)) throw new Error('Invalid review storage');
    state.edits = saved as Record<string, FrameEdit>;
  } catch {
    state.storageWarning = 'Earlier reviews cannot be read back from this browser. Export the JSONL before closing.';
  }
  state.reviewer = readLocal('nnslr-reviewer', '') || '';
  const savedFilter = readLocal('nnslr-review-filter', 'todo');
  if (isFilter(savedFilter)) state.filter = savedFilter;
  state.timelineOpen = readLocal('nnslr-timeline-open', '0') === '1';
  state.showHints = readLocal('nnslr-show-hints', '1') !== '0';
}
