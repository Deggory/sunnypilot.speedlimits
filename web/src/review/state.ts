// The page's data and its one state object: the payload embedded by the server (or by write_review for the
// offline report), the working copies of the frames, what is on screen. Every module reads and changes this
// state, then calls render().
import {byId} from '../common/dom';
import type {
  Drag, Filter, FrameEdit, FrameRow, Hint, HintStats, LastSave, Notice, Payload, RouteLink, ServerState,
  Suspicious,
} from './types';

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

const raw = JSON.parse(byId('payload').textContent || '{}') as Partial<Payload>;

export const runId = String(raw.run_id);
export const routeId = raw.route_id ?? '';
export const rows: readonly FrameRow[] = Array.isArray(raw.frames) ? raw.frames : [];
export const routes: readonly RouteLink[] = Array.isArray(raw.routes) ? raw.routes : [];
const hintsByFrame: Record<string, Hint[]> = isRecord(raw.hints) ? raw.hints : {};
export const hintStats: HintStats | null = isRecord(raw.hint_stats) ? raw.hint_stats : null;
export const routeHasHints = Object.keys(hintsByFrame).length > 0;
// With the review server every confirmation is saved in its database; opened as a plain file the page works offline.
export const apiBase: string | null = raw.server ? '/api/route/' + encodeURIComponent(routeId) : null;
export const storageKey = 'nnslr-review-v1-' + runId;
export const lastKeyStorage = 'nnslr-review-last-' + runId;

export interface ReviewState {
  /** The working copy of every frame touched, by frame key (also what localStorage keeps). */
  edits: Record<string, FrameEdit>;
  suspicious: Suspicious;
  currentKey: string | null;
  /** Index of the selected box in the working copy, -1 for none. */
  selected: number;
  drawing: boolean;
  drag: Drag | null;
  /** The click that follows a drag is ignored until then (ms). */
  ignoreClickUntil: number;
  filter: Filter;
  filterNote: string;
  showHints: boolean;
  /** Frames being sent to the server now. */
  inFlight: Set<string>;
  notice: Notice | null;
  storageWarning: string;
  serverWarning: string;
  retrying: boolean;
  /** The frame whose "Annulla conferma" was pressed once: the second press withdraws. */
  withdrawArmed: string | null;
  saveCount: number;
  lastSave: LastSave | null;
  server: ServerState;
}

export const state: ReviewState = {
  edits: {},
  suspicious: isRecord(raw.suspicious) ? raw.suspicious : {},
  currentKey: null,
  selected: -1,
  drawing: false,
  drag: null,
  ignoreClickUntil: 0,
  filter: 'todo',
  filterNote: '',
  showHints: true,
  inFlight: new Set(),
  notice: null,
  storageWarning: '',
  serverWarning: '',
  retrying: false,
  withdrawArmed: null,
  saveCount: 0,
  lastSave: null,
  server: 'offline',
};

const rowIndex = new Map<string, number>(rows.map((row, index) => [row.frame_key, index]));

export function indexOf(key: string): number {
  return rowIndex.get(key) ?? -1;
}

export function rowOf(key: string): FrameRow | null {
  const index = rowIndex.get(key);
  return index === undefined ? null : rows[index] ?? null;
}

export function hintsOf(key: string): Hint[] {
  const list = hintsByFrame[key];
  return Array.isArray(list) ? list : [];
}

// The whole page is drawn by render.ts; the modules that change the state only ask for it here, so they need
// not import the drawing code.
let renderer: () => void = () => undefined;

export function setRenderer(draw: () => void): void {
  renderer = draw;
}

export function render(): void {
  renderer();
}

/** Keeps the working copies in this browser; a failure (storage full or off) becomes a warning. */
export function store(): void {
  try {
    localStorage.setItem(storageKey, JSON.stringify(state.edits));
    state.storageWarning = '';
  } catch {
    state.storageWarning = 'Salvataggio nel browser non disponibile: scarica il JSONL prima di chiudere.';
  }
}
