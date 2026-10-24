// Revisione cartelli: one route's frames, the boxes the model proposed, the reviewer's confirmation of each frame.
// Served by the review server (every confirmation saved in its database) or written by write_review as a static
// index.html that works offline (payload "server": false, confirmations kept in this browser, JSONL export).
// The modules of web/src/review: state (data and store), frames (review state, filters), navigation, server and
// confirm (saving), overlay and pointer (the photo), boxes, editor, edits, hints, timeline, topbar, messages,
// export, keyboard, layout, render.
import {byId} from '../common/dom';
import {initThemeSwitch} from '../common/theme';
import {discardChanges, confirmFrame, withdrawFrame} from '../review/confirm';
import {one, readLocal} from '../review/dom';
import {initEditor} from '../review/editor';
import {removeSelected, toggleDrawing, toggleHints} from '../review/edits';
import {exportJsonl} from '../review/export';
import {isFilter} from '../review/frames';
import {initKeyboard} from '../review/keyboard';
import {initLayout} from '../review/layout';
import {go, openFrame, pickStartFrame} from '../review/navigation';
import {initPhoto} from '../review/overlay';
import {initPointer} from '../review/pointer';
import {renderPage} from '../review/render';
import {loadServer, retryUnsaved} from '../review/server';
import {apiBase, render, setRenderer, state, storageKey} from '../review/state';
import {buildTimeline, initTimeline, scrollTimelineToCurrent, setTimelineOpen} from '../review/timeline';
import {initTopbar, renderRoutes} from '../review/topbar';
import type {FrameEdit} from '../review/types';

const SERVER_NOTE = 'Ogni conferma viene salvata nel database del server. Se il salvataggio non riesce, il fotogramma '
  + 'resta segnato “non salvato” e viene ritentato; nel frattempo le modifiche restano in questo browser. Il JSONL '
  + 'resta disponibile come copia di sicurezza. Un frame senza riquadri non è automaticamente un esempio negativo: '
  + 'va controllato e confermato.';
const OFFLINE_NOTE = 'Le modifiche sono salvate in questo browser. Esporta il JSONL per conservarle fuori dal browser; '
  + 'il file originale delle proposte resta invariato. Un frame senza riquadri non è automaticamente un esempio '
  + 'negativo: va controllato e confermato.';

function initControls(): void {
  const click = (id: string, action: () => void): void => byId(id).addEventListener('click', action);
  click('prev', () => go(-1));
  click('next', () => go(1));
  click('mPrev', () => go(-1));
  click('mNext', () => go(1));
  click('confirm', () => void confirmFrame());
  click('mConfirm', () => void confirmFrame());
  click('withdraw', () => void withdrawFrame());
  click('discard', discardChanges);
  click('export', exportJsonl);
  click('draw', toggleDrawing);
  click('remove', removeSelected);
  const showHints = byId<HTMLInputElement>('showHints');
  showHints.addEventListener('change', () => {
    if (showHints.checked !== state.showHints) toggleHints();
  });
}

/** What this browser kept: the working copies of the frames, reviewer, filter, timeline fold, hints shown. */
function restore(): void {
  try {
    const saved: unknown = JSON.parse(localStorage.getItem(storageKey) || '{}');
    if (!saved || typeof saved !== 'object' || Array.isArray(saved)) throw new Error('Invalid review storage');
    state.edits = saved as Record<string, FrameEdit>;
  } catch {
    state.storageWarning = 'Revisioni precedenti non recuperabili dal browser. Esporta il JSONL prima di chiudere.';
  }
  byId<HTMLInputElement>('reviewer').value = readLocal('nnslr-reviewer', '') || '';
  const savedFilter = readLocal('nnslr-review-filter', 'todo');
  if (isFilter(savedFilter)) state.filter = savedFilter;
  if (readLocal('nnslr-timeline-open', '0') === '1') setTimelineOpen(true);
  state.showHints = readLocal('nnslr-show-hints', '1') !== '0';
  byId<HTMLInputElement>('showHints').checked = state.showHints;
}

function start(): void {
  setRenderer(renderPage);
  initThemeSwitch();
  initTopbar();
  initControls();
  initEditor();
  initPointer();
  initPhoto();
  initTimeline(openFrame);
  initKeyboard();
  initLayout();
  restore();
  if (apiBase) {
    byId('storageNote').textContent = SERVER_NOTE;
  } else {
    const brand = byId('brand');
    brand.removeAttribute('href');
    brand.removeAttribute('title');
    // A report opened without the server has no account.
    one('.account-btn').hidden = true;
    byId('storageNote').textContent = OFFLINE_NOTE;
  }
  renderRoutes();
  buildTimeline();
  state.currentKey = pickStartFrame();
  if (apiBase) state.server = 'loading';
  render();
  scrollTimelineToCurrent();
  if (apiBase) {
    void loadServer();
    window.setInterval(() => void retryUnsaved(false), 30000);
    window.addEventListener('online', () => void retryUnsaved(false));
  }
}

start();
