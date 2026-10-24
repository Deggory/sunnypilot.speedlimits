// Conferma, Annulla conferma, Annulla modifiche: what the reviewer decides about the frame on screen. With the
// server a confirmation is saved at once; offline it stays in this browser until the JSONL export.
import {byId} from '../common/dom';
import {messageOf} from '../common/api';
import {clone, writeLocal} from './dom';
import {currentRow, frameStatus, fromServer, sameAsServer, workOf} from './frames';
import {advanceAfterConfirm, go} from './navigation';
import {call, reviewerName, send, takeSuspicious} from './server';
import {apiBase, render, runId, state, store} from './state';
import type {SaveAnswer} from './types';

function needReviewer(key: string): void {
  const input = byId('reviewer');
  input.classList.add('need');
  input.focus();
  state.notice = {key, text: 'Scrivi nome o sigla del revisore, poi conferma.', error: true};
  state.lastSave = {key, result: 'need-reviewer', revision: 0};
  render();
}

function timeNow(): string {
  return new Date().toLocaleTimeString('it-CH', {hour: '2-digit', minute: '2-digit'});
}

export async function confirmFrame(): Promise<void> {
  const row = currentRow();
  if (!row) return;
  const key = row.frame_key;
  if (state.inFlight.has(key)) return;
  // Already on the server and not changed since: Conferma moves on, it does not sign the frame again
  // (a frame an AI confirmed too: that confirmation is final).
  const current = state.edits[key];
  if (apiBase && frameStatus(key) === 'done' && current && sameAsServer(current)) {
    state.lastSave = {key, result: 'unchanged', revision: current.revision || 0};
    go(1);
    if (state.currentKey === key) {
      state.notice = {key, text: 'Già confermato: non ci sono altri fotogrammi con questo filtro.', error: false};
    }
    render();
    return;
  }
  const reviewer = reviewerName();
  if (!reviewer) {
    needReviewer(key);
    return;
  }
  writeLocal('nnslr-reviewer', reviewer);
  const s = workOf(row);
  s.review_state = 'reviewed';
  s.reviewer = reviewer;
  s.reviewer_kind = 'human';
  s.review_timestamp = new Date().toISOString();
  s.saved = false;
  s.error = null;
  s.retry = true;
  s.version = (s.version || 0) + 1;
  state.drawing = false;
  if (!apiBase) {
    s.saved = true;
    s.base = {
      detections: clone(s.detections),
      site_group: s.site_group === undefined ? null : s.site_group,
      encounter_id: s.encounter_id === undefined ? null : s.encounter_id,
      reviewer: s.reviewer,
      review_timestamp: s.review_timestamp,
    };
    store();
    advanceAfterConfirm(key, 'Confermato ' + key + ' in questo browser.');
    return;
  }
  state.notice = null;
  store();
  render();
  const ok = await send(key);
  if (ok && state.currentKey === key) advanceAfterConfirm(key, 'Salvato ' + key + ' alle ' + timeNow() + '.');
}

export async function withdrawFrame(): Promise<void> {
  const row = currentRow();
  if (!row || !apiBase) return;
  const key = row.frame_key;
  if (state.inFlight.has(key)) return;
  const reviewer = reviewerName();
  if (!reviewer) {
    needReviewer(key);
    return;
  }
  // Two presses instead of a browser dialog: the second one, on the same frame, withdraws.
  if (state.withdrawArmed !== key) {
    state.withdrawArmed = key;
    const text = 'Premi “Sì, annulla” per togliere la conferma: il fotogramma non verrà importato.';
    state.notice = {key, text, error: false, warn: true};
    render();
    return;
  }
  state.withdrawArmed = null;
  const s = workOf(row);
  state.inFlight.add(key);
  render();
  try {
    const result = await call<SaveAnswer>('POST', apiBase + '/frame/' + key,
      {run_id: runId, reviewer, confirmed: false, base_revision: s.revision || 0});
    const saved = fromServer(result.review);
    state.edits[key] = saved;
    takeSuspicious(result);
    state.notice = {key, text: 'Conferma annullata: il fotogramma non verrà importato.', error: false};
    state.lastSave = {key, result: 'withdrawn', revision: saved.revision || 0};
  } catch (error) {
    state.notice = {key, text: 'Annullamento non riuscito: ' + messageOf(error), error: true};
    state.lastSave = {key, result: 'error', revision: s.revision || 0};
  } finally {
    state.inFlight.delete(key);
    state.saveCount += 1;
    store();
    render();
  }
}

// Back to the last confirmed version of the frame (on the server, or in this browser offline), else to the proposals.
export function discardChanges(): void {
  const row = currentRow();
  if (!row) return;
  const key = row.frame_key;
  const s = state.edits[key];
  if (!s || state.inFlight.has(key)) return;
  if (s.base) {
    if (apiBase) {
      state.edits[key] = fromServer(s.base);
    } else {
      state.edits[key] = {
        review_state: 'reviewed',
        detections: clone(s.base.detections || []),
        site_group: s.base.site_group === undefined ? null : s.base.site_group,
        encounter_id: s.base.encounter_id === undefined ? null : s.base.encounter_id,
        reviewer: s.base.reviewer,
        review_timestamp: s.base.review_timestamp,
        saved: true,
        revision: 0,
        base: s.base,
      };
    }
  } else {
    // Never confirmed: back to the proposals.
    delete state.edits[key];
  }
  state.selected = -1;
  state.notice = null;
  state.withdrawArmed = null;
  byId('valueError').hidden = true;
  store();
  render();
}
