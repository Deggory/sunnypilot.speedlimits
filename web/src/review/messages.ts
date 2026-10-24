// What the page tells the reviewer: warnings above the confirm button (server, browser storage, frames not
// saved), the confirm buttons with what they save, the save line of the frame, the message of the phone bar,
// and the empty state.
import {byId, node} from '../common/dom';
import {shortName} from './catalog';
import {one} from './dom';
import {aiConfirmed, frameStatus, sameAsServer, unsavedKeys} from './frames';
import {retryUnsaved} from './server';
import {apiBase, state} from './state';
import type {FrameEdit, FrameStatus} from './types';

function noticeNode(text: string, className: string): HTMLDivElement {
  const box = node('div', className);
  box.setAttribute('role', 'alert');
  box.append(node('span', '', text));
  return box;
}

export function renderNotices(): void {
  const box = byId('notices');
  box.replaceChildren();
  if (state.serverWarning) box.append(noticeNode(state.serverWarning, 'notice'));
  if (state.storageWarning) box.append(noticeNode(state.storageWarning, 'notice'));
  // The first send of a fresh confirmation is not an error: it becomes one only if it fails.
  const unsaved = apiBase ? unsavedKeys().filter((key) => !(state.inFlight.has(key) && !state.edits[key]?.error)) : [];
  if (unsaved.length === 0) return;
  let list = unsaved.slice(0, 8).join(', ');
  if (unsaved.length > 8) list += ', …';
  const text = unsaved.length === 1
    ? '1 fotogramma confermato NON salvato sul server (' + list + ').'
    : unsaved.length + ' fotogrammi confermati NON salvati sul server (' + list + ').';
  const notice = noticeNode(text, 'notice error');
  const retry = node('button', 'btn small', 'Riprova a salvare');
  retry.type = 'button';
  retry.addEventListener('click', () => void retryUnsaved(true));
  notice.append(retry);
  box.append(notice);
}

/** Instead of the workspace: a report without frames. */
export function showState(title: string, paragraphs: readonly string[]): void {
  const box = byId('stateBox');
  box.replaceChildren(node('h2', '', title), ...paragraphs.map((text) => node('p', '', text)));
  box.hidden = false;
  byId('workspace').hidden = true;
  byId('mobilebar').hidden = true;
}

type LineClass = '' | 'ok' | 'warn' | 'error';

// The save line next to the confirm button: what the server (or this browser) holds for the frame.
function saveText(key: string, s: FrameEdit | undefined): [string, LineClass] {
  const status = frameStatus(key);
  if (!apiBase) {
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

/** Conferma on the panel and on the phone bar: they say what they save ("Conferma · 60 · 40"). */
function renderConfirmButtons(key: string, work: FrameEdit, status: FrameStatus): void {
  const names = work.detections.map(shortName);
  // On a phone too: "Conferma · 60", so the list under the photo need not be checked to know what is saved.
  let label = 'Conferma · ' + names.join(' · ');
  let mobileLabel = label;
  if (work.detections.length === 0) {
    label = 'Conferma · nessun cartello';
    mobileLabel = 'Nessun cartello';
  }
  let already = false;
  if (status === 'done' && apiBase && sameAsServer(state.edits[key])) {
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
  byId('confirmLabel').textContent = label;
  byId('mConfirmLabel').textContent = mobileLabel;
  for (const id of ['confirm', 'mConfirm']) {
    const button = byId<HTMLButtonElement>(id);
    button.disabled = status === 'saving';
    button.dataset['saveState'] = status;
    button.classList.toggle('already', already);
  }
}

/** The confirm block: buttons, Annulla modifiche / Annulla conferma, the save line, the phone bar message. */
export function renderConfirmArea(key: string, work: FrameEdit, status: FrameStatus): void {
  renderConfirmButtons(key, work, status);
  const s = state.edits[key];
  const discard = byId('discard');
  const withdraw = byId('withdraw');
  discard.hidden = !(status === 'dirty' || status === 'error');
  withdraw.hidden = !(apiBase && s && s.saved === true && s.server_confirmed === true && status !== 'saving');
  const armed = state.withdrawArmed === key;
  withdraw.textContent = armed ? 'Sì, annulla' : 'Annulla conferma';
  withdraw.dataset['armed'] = armed ? 'true' : 'false';

  // One status line of fixed height: the latest message for this frame, else what the server holds.
  const [savedText, savedClass] = saveText(key, s);
  const notice = state.notice && state.notice.key === key ? state.notice : null;
  let lineText = savedText;
  let lineClass: LineClass = savedClass;
  if (notice !== null) {
    if (notice.error) {
      lineText = notice.text;
      lineClass = 'error';
    } else if (savedClass !== 'error') {
      // A failed save of this frame stays on the line: a note about the frame saved before must not hide it.
      lineText = notice.text;
      lineClass = notice.warn ? 'warn' : 'ok';
    }
  }
  const line = byId('saveMsg');
  line.textContent = lineText;
  line.title = lineText;
  line.className = 'save-msg' + (lineClass ? ' ' + lineClass : '');

  let mobileText = '';
  const unsaved = apiBase ? unsavedKeys().length : 0;
  if (notice && notice.error) mobileText = notice.text;
  else if (savedClass === 'error') mobileText = savedText;
  else if (state.serverWarning) mobileText = state.serverWarning;
  else if (state.storageWarning) mobileText = state.storageWarning;
  else if (unsaved > 0) mobileText = unsaved + ' fotogrammi confermati NON salvati: riprova dal pannello.';
  const mobile = byId('mobileMsg');
  mobile.textContent = mobileText;
  mobile.hidden = mobileText === '';
  // Nothing to say under the panel: on a phone that section folds away instead of leaving an empty block.
  const quiet = lineText === '' && byId('notices').childElementCount === 0 && discard.hidden && withdraw.hidden;
  one('.confirm-section').classList.toggle('quiet', quiet);
}
