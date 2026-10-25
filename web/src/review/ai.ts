// The AI reviewer on the review page: "Ask AI" (I) sends the boxes of the frame on screen to the server, which
// asks the model chosen in Settings (POST …/ai-suggest, nnslr_tools.ai_review.suggest_frame) and answers with a
// suggestion for each box. Nothing is saved: Apply makes the suggestion the frame's unsaved edits, and the
// reviewer's Confirm saves them as their own review. The suggestions live in this page only, by frame: an answer
// that comes after the reviewer moved on is kept for its frame. Not offline (the static report has no server).
import {reactive} from 'vue';
import {messageOf} from '../common/api';
import type {AiAction, AiState, AiSuggestion} from '../common/ai-types';
import {askConfirm} from '../composables/confirm';
import {clone} from './dom';
import {changed} from './edits';
import {currentRow, workOf} from './frames';
import {call} from './server';
import {report, state} from './store';

/** What the page has for a frame: nothing, a question on its way, a suggestion, or the error of the last one. */
export type AiPhase = 'none' | 'asking' | 'ready' | 'failed';

export interface AiAsk {
  phase: Exclude<AiPhase, 'none'>;
  /** Tells this question from a later one for the same frame. */
  ticket: number;
  /** When it was asked (ms since the epoch), for the seconds shown while waiting. */
  since: number;
  /** The boxes of the frame when it was asked (JSON): Apply asks first when they changed since. */
  sent: string;
  suggestion: AiSuggestion | null;
  error: string;
}

interface AiPage {
  /** GET /api/ai/state: the reviewer chosen in Settings and whether this browser may ask it; null until read. */
  state: AiState | null;
  /** Why the state could not be read. */
  error: string;
  /** By frame key; never kept in localStorage. */
  asks: Record<string, AiAsk>;
}

export const ai: AiPage = reactive({state: null, error: '', asks: {}});

let tickets = 0;

/** The pill of each verdict in the list, with its colour (pill classes of shared.css). */
export const VERDICT: Record<AiAction | 'delete', [label: string, kind: string]> = {
  keep: ['Keep', 'ok'], correct: ['Correct', 'rev'], add: ['Add', 'imp'], delete: ['Delete', 'bad'],
  unjudged: ['Not judged', 'muted'],
};

/** Reads (again) which AI reviewer the server has and whether this browser may ask it. */
export async function loadAiState(): Promise<void> {
  if (!report.apiBase) return;
  try {
    ai.state = await call<AiState>('GET', '/api/ai/state');
    ai.error = '';
  } catch (error) {
    ai.error = 'The AI reviewer cannot be read: ' + messageOf(error);
  }
}

/** Why "Ask AI" is off for every frame ('' when it is on). */
export function aiBlocker(): string {
  if (!report.apiBase) return 'Only with the review server.';
  if (state.server === 'stale') return 'The report of this route was made again: reload the page.';
  if (ai.state === null) return ai.error || 'Reading the AI reviewer…';
  if (!ai.state.available) return ai.state.reason || 'Choose an AI reviewer in Settings.';
  if (ai.state.can_ask === false) return ai.state.reason || 'Sign in on the site to use the AI reviewer.';
  return '';
}

export function aiPhase(key: string): AiPhase {
  return ai.asks[key]?.phase ?? 'none';
}

/** Asks the AI about the frame on screen, with its boxes as they are now. */
export async function askAi(): Promise<void> {
  const row = currentRow();
  const apiBase = report.apiBase;
  if (!row || !apiBase || aiBlocker() !== '') return;
  const key = row.frame_key;
  if (aiPhase(key) === 'asking') return;
  const detections = clone(workOf(row).detections);
  const ticket = ++tickets;
  ai.asks[key] = {phase: 'asking', ticket, since: Date.now(), sent: JSON.stringify(detections), suggestion: null, error: ''};
  try {
    const answer = await call<AiSuggestion>('POST', apiBase + '/frame/' + key + '/ai-suggest',
      {run_id: report.runId, detections});
    const ask = ai.asks[key];
    // Dismissed, or asked again, while the model was thinking: that decides.
    if (ask?.ticket !== ticket) return;
    ask.suggestion = answer;
    ask.phase = 'ready';
  } catch (error) {
    const ask = ai.asks[key];
    if (ask?.ticket !== ticket) return;
    ask.error = 'AI: ' + messageOf(error);
    ask.phase = 'failed';
    // The reviewer may have been changed in Settings meanwhile.
    void loadAiState();
  }
}

/** Apply: the suggestion becomes the boxes of the frame on screen, to check and then confirm (nothing is saved). */
export async function applyAi(): Promise<void> {
  const row = currentRow();
  if (!row) return;
  const key = row.frame_key;
  const ask = ai.asks[key];
  const suggestion = ask?.suggestion;
  if (!ask || ask.phase !== 'ready' || !suggestion) return;
  const work = workOf(row);
  if (JSON.stringify(work.detections) !== ask.sent) {
    const ok = await askConfirm({title: 'Apply the AI suggestion?', action: 'Apply',
      text: 'The boxes of this frame changed after you asked the AI: applying replaces them with its suggestion.'});
    if (!ok || state.currentKey !== key || ai.asks[key]?.ticket !== ask.ticket) return;
  }
  work.detections = clone(suggestion.detections);
  delete ai.asks[key];
  state.selected = work.detections.length > 0 ? 0 : -1;
  state.selectedByUser = false;
  state.drawing = false;
  changed(row);
  state.notice = {key, text: 'AI suggestion applied: check each box, then Confirm.', error: false, warn: true};
}

/** Dismiss: the suggestion (or the error) of the frame on screen goes away; the boxes stay as they are. */
export function dismissAi(): void {
  const key = state.currentKey;
  if (key !== null) delete ai.asks[key];
}
