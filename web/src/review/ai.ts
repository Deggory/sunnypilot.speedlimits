// The AI reviewer on the review page: "Ask AI" (I) sends the boxes of the frame on screen to the server, which
// asks the model chosen in Settings (POST …/ai-suggest, nnslr_tools.ai_review.suggest_frame) and answers with a
// suggestion for each box. Nothing is saved: Apply makes the suggestion the frame's unsaved edits, and the
// reviewer's Confirm saves them as their own review. The suggestions live in this page only, by frame: an answer
// that comes after the reviewer moved on is kept for its frame. The route's last "Try without saving" (GET
// …/ai-trial) comes in the same way, one ready suggestion per frame it tried, and fills the "AI trial" filter.
// Not offline (the static report has no server).
import {refuseChange} from './busy';
import {reactive} from 'vue';
import {messageOf} from '../common/api';
import type {AiAction, AiState, AiSuggestion, AiTrial} from '../common/ai-types';
import type {Box} from './types';
import {askConfirm} from '../composables/confirm';
import {clone} from './dom';
import {changed} from './edits';
import {currentRow, startingDetections, workOf} from './frames';
import {call} from './server';
import {report, rowOf, state} from './store';
import {trialKeys} from './trial';

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
  /** From the route's "Try without saving", not asked on this page: the boxes sent were the proposals. */
  trial?: {at: string | null};
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

let trialRead = false;

/** Reads (again) which AI reviewer the server has and whether this browser may ask it; the first time, the trial. */
export async function loadAiState(): Promise<void> {
  if (!report.apiBase) return;
  if (!trialRead) {
    trialRead = true;
    void loadAiTrial();
  }
  try {
    ai.state = await call<AiState>('GET', '/api/ai/state');
    ai.error = '';
  } catch (error) {
    ai.error = 'The AI reviewer cannot be read: ' + messageOf(error);
  }
}

/** The route's last "Try without saving": a ready suggestion for each frame it tried that nobody confirmed, unless
 * this page already asked the AI about it. Nothing is saved; without a trial nothing changes. */
async function loadAiTrial(): Promise<void> {
  const apiBase = report.apiBase;
  if (!apiBase) return;
  let answer: {trial: AiTrial | null};
  try {
    answer = await call<{trial: AiTrial | null}>('GET', apiBase + '/ai-trial');
  } catch {
    // An older server, or no trial to read: the page works as before.
    return;
  }
  const trial = answer.trial;
  if (trial === null || trial.run_id !== report.runId) return;
  for (const frame of trial.frames) {
    const row = rowOf(frame.frame_key);
    if (!row) continue;
    trialKeys.add(frame.frame_key);
    if (ai.asks[frame.frame_key]) continue;
    ai.asks[frame.frame_key] = {
      phase: 'ready', ticket: ++tickets, since: Date.now(), sent: JSON.stringify(startingDetections(row)), error: '',
      suggestion: {...frame, ok: true, route_id: trial.route_id, run_id: trial.run_id, provider: trial.provider,
        reviewer: trial.reviewer},
      trial: {at: trial.finished_at ?? trial.started_at},
    };
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

/** The box being fitted (Fit with AI), by frame key and index; null when none. */
export const fitting = reactive<{key: string | null; index: number}>({key: null, index: -1});

/** Fit with AI (C): the selected box sent with only the picture around it (…/ai-fit, ai_review.fit_box); it comes
 * back tight, as an unsaved change to confirm. A box moved or removed meanwhile is left as it is. */
export async function fitBox(): Promise<void> {
  const row = currentRow();
  const apiBase = report.apiBase;
  const index = state.selected;
  const detection = row ? workOf(row).detections[index] : undefined;
  if (!row || !apiBase || !detection || fitting.key !== null || aiBlocker() !== '' || refuseChange()) return;
  const key = row.frame_key;
  const sent = JSON.stringify(detection.bbox_xyxy);
  fitting.key = key;
  fitting.index = index;
  try {
    const answer = await call<{box: Box | null; message: string}>('POST', apiBase + '/frame/' + key + '/ai-fit',
      {run_id: report.runId, box: detection.bbox_xyxy});
    const unchanged = workOf(row).detections.includes(detection) && JSON.stringify(detection.bbox_xyxy) === sent;
    if (answer.box && unchanged) {
      detection.bbox_xyxy = answer.box;
      changed(row);
    }
    state.notice = {key, text: answer.message + (answer.box && !unchanged ? ': the box changed meanwhile, kept.' : '.'),
      error: false, warn: !answer.box || !unchanged};
  } catch (error) {
    state.notice = {key, text: 'Fit with AI: ' + messageOf(error), error: true};
    void loadAiState();
  } finally {
    fitting.key = null;
    fitting.index = -1;
  }
}

/** Apply: the suggestion becomes the boxes of the frame on screen, to check and then confirm (nothing is saved). */
export async function applyAi(): Promise<void> {
  const row = currentRow();
  if (!row || refuseChange()) return;
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
