// The one-row bar on top: route picker, filters, route progress, reviewer name (the theme switch and the
// Account icon are pages.py's).
import {byId, node} from '../common/dom';
import {one, writeLocal} from './dom';
import {aiConfirmed, isConfirmed, isFilter, isQualityFilter, reasonsOf} from './frames';
import {setFilter} from './navigation';
import {routeHasHints, routeId, routes, rows, state} from './state';
import {renderQuality} from './timeline';

/** The routes of the server, this one chosen; offline only this report. */
export function renderRoutes(): void {
  const select = byId<HTMLSelectElement>('route');
  select.replaceChildren();
  let found = false;
  for (const route of routes) {
    const option = node('option', '', String(route.route_id || ''));
    option.value = String(route.url || '');
    if (route.route_id === routeId) {
      option.selected = true;
      found = true;
    }
    select.append(option);
  }
  if (!found) {
    const option = node('option', '', routeId ? routeId : 'Report locale');
    option.value = '';
    option.selected = true;
    select.prepend(option);
  }
  select.disabled = routes.length <= 1;
}

/** "120 di 720 confermati · 3 sospetti", its bar, the short count of the phones and the timeline title. */
export function renderProgress(): void {
  const total = rows.length;
  let done = 0;
  let suspicious = 0;
  let byAi = 0;
  for (const row of rows) {
    if (isConfirmed(row.frame_key)) done += 1;
    if (reasonsOf(row.frame_key).length > 0) suspicious += 1;
    if (aiConfirmed(row.frame_key)) byAi += 1;
  }
  // The AI filter appears only when an AI confirmed some frames (or while it is the filter in use).
  const aiButton = one('.filters button[data-filter="ai"]');
  aiButton.textContent = 'AI · ' + byAi;
  aiButton.hidden = byAi === 0 && state.filter !== 'ai';
  let summary = done + '/' + total + ' confermati';
  if (byAi > 0) summary += ' · ' + byAi + ' AI';
  if (suspicious > 0) summary += ' · ' + suspicious + (suspicious === 1 ? ' sospetto' : ' sospetti');
  byId('tlSub').textContent = summary;
  byId('progressShort').textContent = done + '/' + total;
  const text = byId('progressText');
  text.replaceChildren();
  if (total === 0) {
    text.textContent = 'Nessun fotogramma';
  } else {
    text.append(node('strong', '', done + ' di ' + total), document.createTextNode(' confermati'));
    if (suspicious > 0 || routeHasHints) {
      const sus = node('span', suspicious > 0 ? 'sus' : '', suspicious === 1 ? '1 sospetto' : suspicious + ' sospetti');
      text.append(document.createTextNode(' · '), sus);
    }
  }
  byId('progressBar').style.width = total === 0 ? '0' : Math.round(done * 100 / total) + '%';
  renderQuality();
}

export function markFilters(): void {
  for (const button of document.querySelectorAll<HTMLElement>('.filters button')) {
    button.setAttribute('aria-pressed', button.dataset['filter'] === state.filter ? 'true' : 'false');
  }
}

export function initTopbar(): void {
  // The filters on top and the doubt counters of the timeline card.
  for (const button of document.querySelectorAll<HTMLElement>('.filters button, .qchip')) {
    button.addEventListener('click', () => {
      const filter = button.dataset['filter'];
      if (!isFilter(filter)) return;
      // A second click on a pressed doubt counter goes back to the frames to review.
      if (isQualityFilter(filter) && state.filter === filter) setFilter('todo');
      else setFilter(filter);
    });
  }
  const select = byId<HTMLSelectElement>('route');
  select.addEventListener('change', () => {
    if (select.value.startsWith('/route/')) window.location.assign(select.value);
  });
  const reviewer = byId<HTMLInputElement>('reviewer');
  reviewer.addEventListener('input', () => {
    reviewer.classList.remove('need');
    writeLocal('nnslr-reviewer', reviewer.value.trim());
  });
}
