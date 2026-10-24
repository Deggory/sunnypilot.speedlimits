// Pannello: the next step, the three steps with their buttons and job cards, the routes. Data from
// /api/panel/status every 4 s; the parts live in web/src/panel/.
import {api, messageOf} from '../common/api';
import {byId, unchanged} from '../common/dom';
import {clock} from '../common/format';
import type {PanelStatus} from '../common/panel-types';
import {initThemeSwitch} from '../common/theme';
import type {ActionContext} from '../panel/actions';
import {renderHero} from '../panel/hero';
import {followJob, renderJobCards} from '../panel/jobs';
import {renderRoutes} from '../panel/routes';
import {renderSteps} from '../panel/steps';

const context: ActionContext = {refresh, started: (job) => followJob(job.id)};

function render(status: PanelStatus): void {
  byId('readonly').classList.toggle('show', !status.actions_enabled);
  // Buttons and table are rebuilt only when their data changes: rebuilding them at every
  // refresh would lose a click landing at that moment and the keyboard focus.
  if (!unchanged('panel', [status.routes, status.actions, status.jobs, status.next_step, status.next_action,
    status.reviewed_routes, status.min_routes, status.actions_enabled])) {
    renderHero(status, context);
    renderSteps(status, context);
    renderRoutes(status.routes);
  }
  // Every time: the running time and the last log line of a job move on their own.
  renderJobCards(status, context);
}

// Live status; on phones only the dot shows, its title keeps the text.
function showLive(text: string, ok: boolean): void {
  document.querySelector('.live')?.classList.toggle('off', !ok);
  const dot = byId('live-dot');
  dot.className = ok ? 'dot' : 'dot off';
  dot.title = text;
  byId('live-text').textContent = text;
}

async function refresh(): Promise<void> {
  try {
    render(await api<PanelStatus>('/api/panel/status'));
    showLive('aggiornato ' + clock(), true);
  } catch (error) {
    showLive('non raggiungibile: ' + messageOf(error), false);
  }
}

initThemeSwitch();
void refresh();
window.setInterval(() => void refresh(), 4000);
