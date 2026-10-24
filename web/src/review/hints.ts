// "Seconda lettura" in the panel: what another model saw in the frame, one row per sign, with "Usa" to turn
// it into a box. Only on the frames the second opinion has looked at.
import {byId, node} from '../common/dom';
import {confidenceLabel, familyForHint, hasValue, hintTypeOf, sideLabel} from './catalog';
import {useHint} from './edits';
import {overlap, sameBox} from './geometry';
import {hintStats, hintsOf, routeHasHints} from './state';
import type {FrameEdit, FrameRow, Hint} from './types';

// Readings A and B of the same sign become one row: same type and value, boxes overlapping.
function groupReadings(hints: readonly Hint[]): number[][] {
  const groups: number[][] = [];
  hints.forEach((hint, i) => {
    let found: number[] | null = null;
    for (const group of groups) {
      const first = hints[group[0] ?? -1];
      const same = first && first.type === hint.type && first.value_kph === hint.value_kph;
      if (same && overlap(first.box, hint.box) > 0.3) found = group;
    }
    if (found === null) groups.push([i]);
    else found.push(i);
  });
  return groups;
}

/** True when a box of the frame already is this sign: same box, or same type and value over it. */
function alreadyUsed(work: FrameEdit, h: Hint): boolean {
  return work.detections.some((d) => sameBox(d.bbox_xyxy, h.box)
    || (overlap(d.bbox_xyxy, h.box) > 0.3 && d.family === familyForHint(h.type) && d.value_kph === h.value_kph));
}

function hintRow(hints: readonly Hint[], group: readonly number[], work: FrameEdit): HTMLDivElement | null {
  const i = group[0];
  const h = i === undefined ? undefined : hints[i];
  if (i === undefined || h === undefined) return null;
  const item = node('div', 'hint-row');
  const body = node('div', 'hint-text');
  const head = node('div', 'hint-head');
  let text = hintTypeOf(h.type).label;
  if (hasValue(h)) text += ' · ' + h.value_kph + ' km/h';
  head.append(node('span', 'hint-what', text), node('span', 'conf ' + h.confidence, confidenceLabel(h.confidence)));
  const passes: string[] = [];
  for (const k of group) {
    const pass = hints[k]?.pass;
    if (pass && !passes.includes(pass)) passes.push(pass);
  }
  const parts: string[] = [];
  if (passes.length === 1) parts.push('lettura ' + passes[0]);
  if (passes.length > 1) parts.push('letture ' + passes.join(' e '));
  if (h.side) parts.push(sideLabel(h.side));
  if (parts.length > 0) head.append(node('span', 'hint-sub', parts.join(' · ')));
  body.append(head);
  if (h.note) body.append(node('p', 'hint-note', h.note));
  item.append(body);
  const use = node('button', 'btn small tint');
  use.type = 'button';
  use.dataset['hintIndex'] = String(i);
  if (alreadyUsed(work, h)) {
    use.textContent = 'Già usato';
    use.disabled = true;
  } else {
    use.textContent = 'Usa';
    use.setAttribute('aria-label', 'Usa questo: ' + text + (h.pass ? ' (lettura ' + h.pass + ')' : ''));
    use.addEventListener('click', () => useHint(i));
  }
  item.append(use);
  return item;
}

/** "Seconda lettura su 148 fotogrammi di questa route · 2 voci scartate perché non valide." */
function statsLine(): string {
  if (!hintStats) return '';
  let stats = 'Seconda lettura su ' + hintStats.frames + (hintStats.frames === 1 ? ' fotogramma' : ' fotogrammi')
    + ' di questa route';
  const skipped = (hintStats.skipped_lines || 0) + (hintStats.skipped_signs || 0) + (hintStats.skipped_files || 0);
  if (skipped === 1) stats += ' · 1 voce scartata perché non valida';
  if (skipped > 1) stats += ' · ' + skipped + ' voci scartate perché non valide';
  return stats + '.';
}

export function renderHintList(row: FrameRow, work: FrameEdit): void {
  const hints = hintsOf(row.frame_key);
  byId('hintToggle').hidden = !routeHasHints;
  byId('keyHints').hidden = !routeHasHints;
  byId('hintSection').hidden = hints.length === 0;
  if (hints.length === 0) return;
  const rowsOfHints = groupReadings(hints).map((group) => hintRow(hints, group, work));
  byId('hintList').replaceChildren(...rowsOfHints.filter((item) => item !== null));
  byId('hintStats').textContent = statsLine();
}
