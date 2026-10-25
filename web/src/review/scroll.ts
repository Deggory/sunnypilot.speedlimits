// Scrolling, the one thing the components leave to the DOM: the tick of the frame on screen in its timeline
// row, the photo back in view on a phone, the editor of the chosen box in view on a computer. Called once Vue
// has drawn the change (nextTick).
import {narrowLayout} from './dom';
import {state} from './store';

// Only a row that scrolls (phones) moves, and only when the current tick is out of sight.
export function scrollTimelineToCurrent(): void {
  const timeline = document.getElementById('timeline');
  if (timeline === null) return;
  for (const tick of timeline.querySelectorAll<HTMLButtonElement>('button')) {
    if (tick.dataset['key'] !== state.currentKey) continue;
    // On a computer the timeline card scrolls on its own (the photo stays): the segment of the frame in view.
    const card = timeline.closest<HTMLElement>('.tl-card');
    if (card !== null && card.scrollHeight > card.clientHeight) {
      const box = card.getBoundingClientRect();
      const line = (tick.closest('.tl-seg') ?? tick).getBoundingClientRect();
      if (line.top < box.top) card.scrollTop -= box.top - line.top + 8;
      else if (line.bottom > box.bottom) card.scrollTop += line.bottom - box.bottom + 8;
    }
    const strip = tick.parentElement;
    if (strip === null || strip.scrollWidth <= strip.clientWidth) return;
    const left = tick.offsetLeft - strip.offsetLeft;
    if (left >= strip.scrollLeft + 40 && left <= strip.scrollLeft + strip.clientWidth - 40) return;
    const target = left - strip.clientWidth / 2;
    strip.scrollLeft = target < 0 ? 0 : target;
  }
}

// On a phone the panel is below the photo: bring the photo back into view after a frame change.
export function revealStage(): void {
  if (!narrowLayout()) return;
  const header = document.querySelector('.topbar');
  const stage = document.getElementById('stage');
  if (header === null || stage === null) return;
  const top = getComputedStyle(header).position === 'sticky' ? header.getBoundingClientRect().bottom : 0;
  const rect = stage.getBoundingClientRect();
  if (rect.top < top || rect.top > window.innerHeight * 0.6) {
    window.scrollTo(0, window.scrollY + rect.top - top - 8);
  }
}

// On a computer the panel scrolls so that values, type and road of the chosen box are in sight.
export function revealEditor(): void {
  if (narrowLayout()) return;
  document.getElementById('roads')?.scrollIntoView({block: 'nearest'});
}
