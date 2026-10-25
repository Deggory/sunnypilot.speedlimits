// The pages built by pages.py without a script of their own (route list, error pages): the Account menu and
// the confirm dialog; on the route list also the "Review with AI" sheet of its buttons (button[data-ai-route]) and,
// while drives are proposed again, a live line under each of them (span[data-live]: who, how far, the time left).
import {createApp} from 'vue';
import {api} from '../common/api';
import type {PanelStatus} from '../common/panel-types';
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

/** The live lines of the route list, from /api/panel/status every 5 s while a Propose again runs. */
async function showProposing(): Promise<void> {
  let preparation: PanelStatus['preparation'] = null;
  try {
    preparation = (await api<PanelStatus>('/api/panel/status')).preparation;
  } catch {
    return;
  }
  const routes = new Map((preparation?.again ? preparation.working : []).map((item) => [item.route, item]));
  for (const line of document.querySelectorAll<HTMLElement>('span[data-live]')) {
    const item = routes.get(line.dataset['live'] ?? '');
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
if (document.querySelector('span[data-live]') !== null) {
  void showProposing();
  window.setInterval(() => void showProposing(), 5000);
}
