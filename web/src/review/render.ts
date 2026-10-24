// Draws the whole page from the state: called after every change (state.render()).
import {byId} from '../common/dom';
import {renderBoxList} from './boxes';
import {renderEditor, renderFrameFields} from './editor';
import {
  aiConfirmed, currentRow, frameStatus, proposalsOf, reasonsOf, statusLabel, unsavedKeys, visibleRows, workOf,
} from './frames';
import {renderHintList} from './hints';
import {renderConfirmArea, renderNotices, showState} from './messages';
import {renderStage} from './overlay';
import {hintsOf, routeHasHints, rows, state} from './state';
import {renderTimeline} from './timeline';
import {markFilters, renderProgress} from './topbar';
import type {FrameEdit, FrameRow, FrameStatus} from './types';

/** Under the photo: position in the filter, state tag, frame data; in the panel the suspicious reasons. */
function renderFrameBar(row: FrameRow, status: FrameStatus): void {
  const key = row.frame_key;
  const list = visibleRows();
  const pos = Math.max(0, list.findIndex((item) => item.frame_key === key));
  byId('pos').textContent = (pos + 1) + ' / ' + list.length;
  for (const id of ['prev', 'mPrev']) byId<HTMLButtonElement>(id).disabled = pos <= 0;
  for (const id of ['next', 'mNext']) byId<HTMLButtonElement>(id).disabled = pos >= list.length - 1;

  const reasons = reasonsOf(key);
  const pill = byId('statePill');
  pill.className = 'pill state ' + status + (aiConfirmed(key) ? ' ai' : '')
    + (status === 'todo' && reasons.length > 0 ? ' sus' : '');
  pill.textContent = statusLabel(key);

  const proposals = proposalsOf(row).length;
  let info = 'Segmento ' + row.segment_index + ' · immagine ' + row.output_index + ' · ' + proposals
    + (proposals === 1 ? ' proposta' : ' proposte');
  if (routeHasHints) info += ' · ' + hintsOf(key).length + ' in seconda lettura';
  byId('frameData').textContent = info;

  const reasonBox = byId('reasons');
  reasonBox.replaceChildren();
  if (reasons.length > 0) {
    const strong = document.createElement('strong');
    strong.textContent = 'Sospetto: ';
    reasonBox.append(strong, document.createTextNode(reasons.map((reason) => String(reason.text)).join('; ')));
  }
  reasonBox.hidden = reasons.length === 0;
  const note = byId('filterNote');
  note.textContent = state.filterNote;
  note.hidden = state.filterNote === '';

  const draw = byId('draw');
  draw.classList.toggle('on', state.drawing);
  draw.textContent = state.drawing ? 'Annulla' : '+ Riquadro';
}

// State for whoever drives the page (an agent with Playwright): attributes only, the layout does not change.
function publishState(key: string, status: FrameStatus, s: FrameEdit | undefined): void {
  const data = document.body.dataset;
  data['frameKey'] = key;
  data['frameStatus'] = status;
  data['frameReviewerKind'] = s && s.server_confirmed ? (s.reviewer_kind || 'human') : '';
  data['server'] = state.server;
  data['unsaved'] = String(unsavedKeys().length);
  data['saving'] = state.inFlight.size > 0 ? 'true' : 'false';
  data['saveCount'] = String(state.saveCount);
  if (state.lastSave) {
    data['lastSaveKey'] = state.lastSave.key;
    data['lastSaveResult'] = state.lastSave.result;
    data['lastSaveRevision'] = String(state.lastSave.revision);
  }
}

export function renderPage(): void {
  renderNotices();
  renderProgress();
  markFilters();
  let row = currentRow();
  if (row === null) {
    row = rows[0] ?? null;
    if (row === null) {
      showState('Questo report non ha fotogrammi',
        ['Scegli un’altra route in alto, oppure rigenera la pre-annotazione.']);
      return;
    }
    state.currentKey = row.frame_key;
  }
  byId('stateBox').hidden = true;
  byId('workspace').hidden = false;
  byId('mobilebar').hidden = false;

  const key = row.frame_key;
  const work = workOf(row);
  if (state.selected >= work.detections.length) state.selected = -1;
  renderStage(row, work);
  renderBoxList(row, work);
  renderEditor(row, work);
  renderHintList(row, work);
  renderTimeline();
  const status = frameStatus(key);
  renderFrameBar(row, status);
  renderConfirmArea(key, work, status);
  renderFrameFields(row, work);
  publishState(key, status, state.edits[key]);
}
