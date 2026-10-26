// The pages built by pages.py without a script of their own (route list, error pages): the Account menu and
// the confirm dialog; on the route list also the "Review with AI" sheet of its buttons (button[data-ai-route]) and,
// while drives are proposed again or reviewed by the AI, a live line under each of them (span[data-live]: who, how
// far, the time left).
import {createApp} from 'vue';
import {api, messageOf, post} from '../common/api';
import {askConfirm} from '../composables/confirm';
import type {AiLive, PanelStatus} from '../common/panel-types';
import {durationText} from '../control/jobs';
import {mountSiteParts} from '../common/mount';
import AiRouteSheet from '../components/AiRouteSheet.vue';

mountSiteParts();
if (document.querySelector('button[data-ai-route]') !== null) {
  const host = document.createElement('div');
  host.id = 'ai-route-host';
  document.body.append(host);
  createApp(AiRouteSheet).mount(host);
}

/** The line of a drive the AI reviews: how far, the time left, and a bar. */
function showAi(line: HTMLElement, live: AiLive): void {
  line.hidden = false;
  const percent = live.total ? Math.min(100, Math.round(100 * live.done / live.total)) : 0;
  const text = (live.saving ? 'The AI is reviewing: ' : 'The AI is trying: ') + live.done.toLocaleString('en')
    + (live.total ? ' of ' + live.total.toLocaleString('en') : '') + ' frames'
    + (live.seconds_left ? ' · ' + durationText(live.seconds_left) + ' left' : '');
  line.replaceChildren(document.createTextNode(text));
  const bar = document.createElement('span');
  bar.className = 'bar run';
  const fill = document.createElement('span');
  fill.style.width = percent + '%';
  bar.append(fill);
  line.append(bar);
}

/** The live lines of the route list, from /api/panel/status every 5 s: Propose again and Review with AI. */
async function showProposing(): Promise<void> {
  let preparation: PanelStatus['preparation'] = null;
  let live: AiLive | null = null;
  try {
    const status = await api<PanelStatus>('/api/panel/status');
    preparation = status.preparation;
    live = status.ai_live ?? null;
  } catch {
    return;
  }
  // One AI review at a time: while one runs, every other "Review with AI" is grey and says why; the drive at work
  // has "Stop the AI" in its place.
  for (const button of document.querySelectorAll<HTMLButtonElement>('button[data-ai-route]')) {
    const here = live !== null && live.route === button.dataset['aiRoute'];
    button.disabled = live !== null && !here;
    button.title = live === null || here ? '' : 'The AI is reviewing another drive: one at a time.';
    button.textContent = here ? 'Stop the AI' : 'Review with AI';
    button.classList.toggle('danger', here);
    if (here && live !== null) button.dataset['aiStop'] = live.job;
    else delete button.dataset['aiStop'];
  }
  const routes = new Map((preparation?.again ? preparation.working : []).map((item) => [item.route, item]));
  for (const line of document.querySelectorAll<HTMLElement>('span[data-live]')) {
    const item = routes.get(line.dataset['live'] ?? '');
    if ((!item || item.phase === 'ready') && live !== null && live.route === line.dataset['live']) {
      showAi(line, live);
      continue;
    }
    line.hidden = !item || item.phase === 'ready';
    if (!item || item.phase === 'ready') continue;
    const who = preparation?.proposer ?? 'The proposer';
    const text = item.phase === 'waiting' ? 'Waiting to be proposed again by ' + who
      : who + ' is proposing again: ' + item.done.toLocaleString('en') + ' of ' + item.total.toLocaleString('en') + ' frames'
        + (item.seconds_left ? ' · ' + durationText(item.seconds_left) + ' left' : '');
    line.replaceChildren(document.createTextNode(text));
    if (item.phase !== 'waiting') {
      const bar = document.createElement('span');
      bar.className = 'bar run';
      const fill = document.createElement('span');
      fill.style.width = item.percent + '%';
      bar.append(fill);
      line.append(bar);
    }
  }
}
// "Stop the AI" on the drive the AI reviews: asked first; the frames it saved stay.
document.addEventListener('click', (event) => {
  const button = event.target instanceof Element ? event.target.closest<HTMLButtonElement>('button[data-ai-stop]') : null;
  if (button === null) return;
  event.preventDefault();
  event.stopImmediatePropagation();
  const job = button.dataset['aiStop'] ?? '';
  void (async () => {
    const ok = await askConfirm({title: 'Stop the AI review?', action: 'Stop', danger: true,
      text: 'The frames it saved stay saved. Review with AI again carries on with the frames left.'});
    if (!ok) return;
    button.disabled = true;
    button.textContent = 'Stopping…';
    try {
      await post('/api/panel/job/' + job + '/stop');
    } catch (problem) {
      button.disabled = false;
      button.textContent = 'Stop the AI';
      button.title = messageOf(problem);
    }
  })();
}, true);
if (document.querySelector('span[data-live]') !== null) {
  void showProposing();
  window.setInterval(() => void showProposing(), 5000);
}

// The route list's groups (details[data-group]): the tiles of the overview (button[data-open-group]) choose the one
// shown, one at a time (a closed group is not drawn, .groups-by-tile), and one is always shown: the address
// #group-doing, else the one this browser last chose, else the first with drives (In progress, To start, Done).
// Its title, or its tile again, leaves it open: the list never empties.
const OPEN_KEY = 'nnslr.routes.open';
const groups = [...document.querySelectorAll<HTMLDetailsElement>('details[data-group]')];
if (groups.length) {
  document.body.classList.add('groups-by-tile');
  let remembered = '';
  try {
    const value: unknown = JSON.parse(localStorage.getItem(OPEN_KEY) ?? '[]');
    remembered = Array.isArray(value) && typeof value[0] === 'string' ? value[0] : '';
  } catch {
    remembered = '';
  }
  const keyOf = (group: HTMLDetailsElement): string => group.dataset['group'] ?? '';
  const fromHash = location.hash.startsWith('#group-') ? location.hash.slice(7) : '';
  const first = groups.find((group) => keyOf(group) === fromHash) ?? groups.find((group) => keyOf(group) === remembered) ?? groups[0] as HTMLDetailsElement;
  const show = (chosen: HTMLDetailsElement): void => {
    for (const group of groups) {
      group.open = group === chosen;
      document.querySelector(`button[data-open-group="${keyOf(group)}"]`)?.setAttribute('aria-pressed', String(group === chosen));
    }
    try {
      localStorage.setItem(OPEN_KEY, JSON.stringify([keyOf(chosen)]));
    } catch {
      // Private browsing: it is only not remembered.
    }
  };
  show(first);
  if (fromHash) first.scrollIntoView({block: 'start'});
  // The title does not fold its group: only another tile changes it.
  for (const group of groups) group.querySelector('summary')?.addEventListener('click', (event) => event.preventDefault());
  for (const tile of document.querySelectorAll<HTMLButtonElement>('button[data-open-group]')) {
    tile.addEventListener('click', () => {
      const group = groups.find((item) => keyOf(item) === tile.dataset['openGroup']);
      if (group === undefined) return;
      if (!group.open) show(group);
      group.scrollIntoView({behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth', block: 'nearest'});
    });
  }
}
