// What the page tells the reviewer, as data for ConfirmArea.vue and MobileBar.vue: warnings above the confirm
// button (server, browser storage, frames not saved), the confirm buttons with what they save, the save line
// of the frame, the message of the phone bar.
import {shortName} from './catalog';
import {aiConfirmed, frameStatus, sameAsServer, unsavedKeys} from './frames';
import {report, state} from './store';
import type {FrameEdit, FrameStatus} from './types';

export interface PageNotice {
  text: string;
  error: boolean;
  /** With "Riprova a salvare". */
  retry: boolean;
}

export function pageNotices(): PageNotice[] {
  const out: PageNotice[] = [];
  if (state.serverWarning) out.push({text: state.serverWarning, error: false, retry: false});
  if (state.storageWarning) out.push({text: state.storageWarning, error: false, retry: false});
  // The first send of a fresh confirmation is not an error: it becomes one only if it fails.
  const unsaved = report.apiBase
    ? unsavedKeys().filter((key) => !(state.inFlight.has(key) && !state.edits[key]?.error)) : [];
  if (unsaved.length === 0) return out;
  let list = unsaved.slice(0, 8).join(', ');
  if (unsaved.length > 8) list += ', …';
  const text = unsaved.length === 1
    ? '1 fotogramma confermato NON salvato sul server (' + list + ').'
    : unsaved.length + ' fotogrammi confermati NON salvati sul server (' + list + ').';
  out.push({text, error: true, retry: true});
  return out;
}

export type LineClass = '' | 'ok' | 'warn' | 'error';

// The save line next to the confirm button: what the server (or this browser) holds for the frame.
function saveText(key: string, s: FrameEdit | undefined): [string, LineClass] {
  const status = frameStatus(key);
  if (!report.apiBase) {
    if (status === 'done') return ['Confermato in questo browser: esporta il JSONL per conservarlo.', 'ok'];
    if (status === 'dirty') return ['Modifiche non confermate (solo in questo browser).', 'warn'];
    return ['', ''];
  }
  if (status === 'saving') return ['Salvataggio in corso…', ''];
  if (!s) return ['', ''];
  if (status === 'done' && aiConfirmed(key)) {
    const via = s.channel === 'mcp' ? 'MCP' : s.channel === 'rest' ? 'REST' : '';
    const who = s.reviewer || 'un’AI';
    return ['Salvato ✓ · revisione ' + s.revision + ' · ' + who + ' (AI' + (via ? ' · ' + via : '') + ')', 'ok'];
  }
  if (status === 'done') {
    return ['Salvato ✓ · revisione ' + s.revision + (s.reviewer ? ' · ' + s.reviewer : ''), 'ok'];
  }
  if (status === 'error' && s.error) return ['NON salvato: ' + s.error, 'error'];
  if (status === 'error') return ['Confermato ma non ancora salvato: nuovo tentativo automatico.', 'error'];
  if (status === 'dirty' && s.server_confirmed) {
    return ['Modificato: premi Conferma per salvarlo. Sul server resta la versione di prima.', 'warn'];
  }
  if (status === 'dirty') return ['Modifiche da confermare.', 'warn'];
  if (s.base && s.base.confirmed === false) {
    return ['Conferma annullata: non verrà importato finché non lo confermi di nuovo.', ''];
  }
  return ['', ''];
}

export interface ConfirmLabels {
  label: string;
  mobileLabel: string;
  /** Already on the server and unchanged: Conferma only moves on. */
  already: boolean;
}

/** Conferma on the panel and on the phone bar: they say what they save ("Conferma · 60 · 40"). */
export function confirmLabels(key: string, work: FrameEdit, status: FrameStatus): ConfirmLabels {
  const names = work.detections.map(shortName);
  // On a phone too: "Conferma · 60", so the list under the photo need not be checked to know what is saved.
  let label = 'Conferma · ' + names.join(' · ');
  let mobileLabel = label;
  if (work.detections.length === 0) {
    label = 'Conferma · nessun cartello';
    mobileLabel = 'Nessun cartello';
  }
  let already = false;
  if (status === 'done' && report.apiBase && sameAsServer(state.edits[key])) {
    label = 'Già confermato · avanti →';
    mobileLabel = 'Avanti →';
    already = true;
  } else if (status === 'done') {
    label = 'Confermato ✓ · conferma di nuovo';
    mobileLabel = 'Confermato ✓';
  }
  if (status === 'saving') {
    label = 'Salvataggio…';
    mobileLabel = 'Salvataggio…';
  }
  return {label, mobileLabel, already};
}

export interface StatusLine {
  text: string;
  cls: LineClass;
  /** Annulla modifiche, Annulla conferma (armed: "Sì, annulla"). */
  discard: boolean;
  withdraw: boolean;
  armed: boolean;
  /** The message of the phone bar, empty for none. */
  mobile: string;
}

/** The line under the inspector: Annulla modifiche / Annulla conferma, the save line, the phone bar message. */
export function statusLine(key: string, status: FrameStatus): StatusLine {
  const s = state.edits[key];
  // One status line of fixed height: the latest message for this frame, else what the server holds.
  const [savedText, savedClass] = saveText(key, s);
  const notice = state.notice && state.notice.key === key ? state.notice : null;
  let text = savedText;
  let cls: LineClass = savedClass;
  if (notice !== null) {
    if (notice.error) {
      text = notice.text;
      cls = 'error';
    } else if (savedClass !== 'error') {
      // A failed save of this frame stays on the line: a note about the frame saved before must not hide it.
      text = notice.text;
      cls = notice.warn ? 'warn' : 'ok';
    }
  }
  let mobile = '';
  const unsaved = report.apiBase ? unsavedKeys().length : 0;
  if (notice && notice.error) mobile = notice.text;
  else if (savedClass === 'error') mobile = savedText;
  else if (state.serverWarning) mobile = state.serverWarning;
  else if (state.storageWarning) mobile = state.storageWarning;
  else if (unsaved > 0) mobile = unsaved + ' fotogrammi confermati NON salvati: riprova dal pannello.';
  return {
    text,
    cls,
    discard: status === 'dirty' || status === 'error',
    withdraw: Boolean(report.apiBase && s && s.saved === true && s.server_confirmed === true && status !== 'saving'),
    armed: state.withdrawArmed === key,
    mobile,
  };
}
