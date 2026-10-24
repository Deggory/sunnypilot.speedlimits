// The timeline card: one row of ticks per segment, coloured by the state of each frame, and the counters of
// the confirmed frames that still carry a doubt (each one a filter). On a phone the card folds under its title.
import {byId, node} from '../common/dom';
import {writeLocal} from './dom';
import {aiConfirmed, doubtsOf, frameStatus, proposalsOf, reasonsOf, statusLabel} from './frames';
import {hintsOf, rowOf, rows, state} from './state';

/** The tick buttons of the timeline, in frame order. */
function ticks(): NodeListOf<HTMLButtonElement> {
  return byId('timeline').querySelectorAll('button');
}

// One row of ticks per segment, with its label.
export function buildTimeline(): void {
  const box = byId('timeline');
  box.replaceChildren();
  let lastSegment: number | null = null;
  let strip: HTMLDivElement | null = null;
  let longest = 1;
  for (const row of rows) {
    if (strip === null || row.segment_index !== lastSegment) {
      const segment = node('div', 'tl-seg');
      strip = node('div', 'tl-ticks');
      segment.append(node('span', 'tl-label', 'seg ' + row.segment_index), strip);
      box.append(segment);
      lastSegment = row.segment_index;
    }
    const tick = node('button');
    tick.type = 'button';
    tick.tabIndex = -1;
    tick.dataset['key'] = row.frame_key;
    strip.append(tick);
    longest = Math.max(longest, strip.childElementCount);
  }
  // The cells fill the width: one column per image of the longest segment.
  box.style.setProperty('--tl-cols', String(longest));
}

export function renderTimeline(): void {
  for (const tick of ticks()) {
    const key = tick.dataset['key'] ?? '';
    const row = rowOf(key);
    if (row === null) continue;
    const status = frameStatus(key);
    let cls = '';
    if (status === 'done') {
      const doubts = doubtsOf(key);
      cls = doubts.unknown + doubts.unreadable > 0 ? 'done unsure' : 'done';
      if (aiConfirmed(key)) cls += ' ai';
    } else if (status === 'error') {
      cls = 'error';
    } else if (status === 'dirty' || status === 'saving') {
      cls = 'dirty';
    } else if (reasonsOf(key).length > 0) {
      cls = 'sus';
    } else if (proposalsOf(row).length > 0 || hintsOf(key).length > 0) {
      cls = 'has';
    }
    if (key === state.currentKey) cls += ' cur';
    if (tick.className !== cls) tick.className = cls;
    const label = 'Segmento ' + row.segment_index + ' · immagine ' + row.output_index + ' · '
      + statusLabel(key).toLowerCase();
    if (tick.title !== label) {
      tick.title = label;
      tick.setAttribute('aria-label', label);
    }
  }
}

// Only a row that scrolls (phones) moves, and only when the current tick is out of sight.
export function scrollTimelineToCurrent(): void {
  for (const tick of ticks()) {
    if (tick.dataset['key'] !== state.currentKey) continue;
    const strip = tick.parentElement;
    if (strip === null || strip.scrollWidth <= strip.clientWidth) return;
    const left = tick.offsetLeft - strip.offsetLeft;
    if (left >= strip.scrollLeft + 40 && left <= strip.scrollLeft + strip.clientWidth - 40) return;
    const target = left - strip.clientWidth / 2;
    strip.scrollLeft = target < 0 ? 0 : target;
  }
}

// Confirmed is not the same as certain: how many confirmed frames still carry a doubt. Each counter is a filter.
export function renderQuality(): void {
  let unknown = 0;
  let unreadable = 0;
  let road = 0;
  for (const row of rows) {
    const doubts = doubtsOf(row.frame_key);
    if (doubts.unknown > 0) unknown += 1;
    if (doubts.unreadable > 0) unreadable += 1;
    if (doubts.road > 0) road += 1;
  }
  setQualityChip(byId('qUnknown'), unknown, 'senza numero', true);
  setQualityChip(byId('qUnreadable'), unreadable, 'con illeggibili', true);
  setQualityChip(byId('qRoad'), road, 'strada non indicata', false);
}

function setQualityChip(chip: HTMLElement, count: number, label: string, warn: boolean): void {
  chip.textContent = count + ' ' + label;
  chip.dataset['count'] = String(count);
  if (count > 0 && warn) chip.className = 'qchip some';
  else if (count > 0) chip.className = 'qchip';
  else chip.className = 'qchip zero';
  chip.setAttribute('aria-pressed', state.filter === chip.dataset['filter'] ? 'true' : 'false');
}

/** The fold of the card on a phone (kept in this browser) and the click on a tick. */
export function initTimeline(open: (key: string) => void): void {
  byId('tlToggle').addEventListener('click', () => {
    const opened = !byId('workspace').classList.contains('tl-open');
    setTimelineOpen(opened);
    writeLocal('nnslr-timeline-open', opened ? '1' : '0');
    if (opened) scrollTimelineToCurrent();
  });
  byId('timeline').addEventListener('click', (event) => {
    const target = event.target;
    const key = target instanceof HTMLElement ? target.dataset['key'] : undefined;
    if (key) open(key);
  });
}

export function setTimelineOpen(opened: boolean): void {
  byId('workspace').classList.toggle('tl-open', opened);
  byId('tlToggle').setAttribute('aria-expanded', opened ? 'true' : 'false');
}
