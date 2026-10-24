// The list of the boxes of the frame in the panel ("Cartelli"): what each one is, where it comes from, what the
// reviewer changed, the boxes that are the same sign twice.
import {byId, node} from '../common/dom';
import {describe, shortName} from './catalog';
import {removeBox, selectBox} from './edits';
import {proposalsOf} from './frames';
import {overlap} from './geometry';
import {state} from './state';
import type {Detection, FrameEdit, FrameRow} from './types';

// The proposal a reviewed box comes from, by detection_id; null for a box added by hand.
function originalOf(row: FrameRow, d: Detection): Detection | null {
  if (!d.detection_id) return null;
  return proposalsOf(row).find((p) => p.detection_id === d.detection_id) ?? null;
}

/** "aggiunto a mano", "modello · 87%", or what the reviewer changed: "modello 40 → 60". */
function sourceLine(row: FrameRow, d: Detection): HTMLSpanElement {
  const src = node('span', 'src');
  const original = originalOf(row, d);
  if (original === null) {
    src.textContent = 'aggiunto a mano';
  } else if (original.family !== d.family || original.value_kph !== d.value_kph) {
    src.append(document.createTextNode('modello ' + shortName(original) + ' '),
      node('span', 'changed', '→ ' + shortName(d)));
  } else if (original.score === null || original.score === undefined) {
    src.textContent = 'modello';
  } else {
    src.textContent = 'modello · ' + Math.round(Number(original.score) * 100) + '%';
  }
  return src;
}

export function renderBoxList(row: FrameRow, work: FrameEdit): void {
  const list = byId('boxList');
  list.replaceChildren();
  work.detections.forEach((d, i) => {
    const selected = i === state.selected;
    const item = node('div', selected ? 'box-row selected' : 'box-row');
    item.dataset['index'] = String(i);
    item.dataset['detectionId'] = String(d.detection_id || '');
    const main = node('button', 'box-main');
    main.type = 'button';
    main.setAttribute('aria-pressed', selected ? 'true' : 'false');
    const src = sourceLine(row, d);
    // A box covering most of an earlier one is the same sign drawn twice.
    const earlier = work.detections.slice(0, i).findIndex((other) => overlap(other.bbox_xyxy, d.bbox_xyxy) > 0.6);
    if (earlier >= 0) {
      src.append(node('span', 'dup', ' · stesso cartello di ' + (earlier + 1)));
      item.dataset['duplicateOf'] = String(earlier);
    }
    main.append(node('span', 'n', String(i + 1)), node('span', 'what', describe(d)), src);
    main.addEventListener('click', () => selectBox(i));
    const remove = node('button', 'box-del', '✕');
    remove.type = 'button';
    remove.title = 'Elimina il riquadro ' + (i + 1);
    remove.setAttribute('aria-label', 'Elimina il riquadro ' + (i + 1));
    remove.addEventListener('click', () => removeBox(i));
    item.append(main, remove);
    list.append(item);
  });
  const count = work.detections.length;
  byId('noBoxes').hidden = count > 0;
  byId('pickHint').hidden = count === 0 || state.selected >= 0;
  byId('boxCount').textContent = count === 0 ? '' : count === 1 ? '1 riquadro' : count + ' riquadri';
}
