// Sizes the CSS needs from the script: the height of the top bar (the docked panel starts under it) and of the
// phone's bottom bar (the page ends above it); and the labels and corner handles, which follow the on-screen
// size of the photo and of the zoom.
import {byId} from '../common/dom';
import {one, svgById} from './dom';
import {currentRow, selectedDetection, workOf} from './frames';
import {renderStage, renderZoom} from './overlay';
import {state} from './state';

function renderOverlaysOnly(): void {
  const row = currentRow();
  if (!row || byId('workspace').hidden || state.drag) return;
  renderStage(row, workOf(row));
  const d = selectedDetection();
  if (d) renderZoom(row, d);
}

function measure(element: Element, property: string): void {
  document.documentElement.style.setProperty(property, Math.ceil(element.getBoundingClientRect().height) + 'px');
}

export function initLayout(): void {
  const topbar = one('.topbar');
  const mobilebar = byId('mobilebar');
  if (typeof ResizeObserver === 'function') {
    const overlays = new ResizeObserver(() => renderOverlaysOnly());
    overlays.observe(byId('stage'));
    overlays.observe(svgById('zoom'));
    new ResizeObserver(() => measure(topbar, '--topbar-h')).observe(topbar);
    new ResizeObserver(() => measure(mobilebar, '--mobilebar-h')).observe(mobilebar);
  } else {
    window.addEventListener('resize', () => {
      renderOverlaysOnly();
      measure(topbar, '--topbar-h');
      measure(mobilebar, '--mobilebar-h');
    });
    measure(topbar, '--topbar-h');
    measure(mobilebar, '--mobilebar-h');
  }
}
