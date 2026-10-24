// Stato: the machine and the jobs, as the Status tab of distill-web. Data from /api/panel/status every 4 s.
import {api, messageOf, post} from '../common/api';
import {byId, node, pill, setBar, unchanged} from '../common/dom';
import {clock} from '../common/format';
import type {ActionName, Job, JobState, LogChunk, PanelStatus, StartedJob} from '../common/panel-types';
import {initThemeSwitch} from '../common/theme';
import {toast} from '../common/toast';

const JOB_STATE: Record<JobState, [string, string]> = {
  running: ['in corso', 'run'], ok: ['finito', 'ok'], failed: ['fallito', 'bad'], interrupted: ['interrotto', 'warn'],
};
// The last few minutes of CPU and GPU use, one point per refresh, kept in the page.
const HISTORY_POINTS = 90;
const usage = {cpu: [] as number[], gpu: [] as number[]};
let status: PanelStatus | null = null;

interface LogView {
  id: string | null;
  offset: number;
  timer: number | undefined;
  follow: boolean;
}
let log: LogView = {id: null, offset: 0, timer: undefined, follow: true};

function pushUsage(key: keyof typeof usage, value: number): void {
  const history = usage[key];
  history.push(value);
  if (history.length > HISTORY_POINTS) history.shift();
}

/** An area chart 0-100% of a history, newest on the right; the colour is a CSS variable (follows the theme). */
function renderSpark(id: string, values: number[], color: string): void {
  const SVG = 'http://www.w3.org/2000/svg';
  const svg = document.createElementNS(SVG, 'svg');
  svg.setAttribute('viewBox', '0 0 ' + HISTORY_POINTS + ' 30');
  svg.setAttribute('preserveAspectRatio', 'none');
  if (values.length > 1) {
    const start = HISTORY_POINTS - values.length;
    const points = values.map((value, index) => (start + index) + ',' + (30 - Math.max(0, Math.min(100, value)) * 0.3));
    const area = document.createElementNS(SVG, 'polygon');
    area.setAttribute('points', start + ',30 ' + points.join(' ') + ' ' + (HISTORY_POINTS - 1) + ',30');
    area.style.fill = color;
    area.setAttribute('fill-opacity', '0.15');
    const line = document.createElementNS(SVG, 'polyline');
    line.setAttribute('points', points.join(' '));
    line.setAttribute('fill', 'none');
    line.style.stroke = color;
    line.setAttribute('stroke-width', '1.5');
    line.setAttribute('vector-effect', 'non-scaling-stroke');
    svg.append(area, line);
  }
  byId(id).replaceChildren(svg);
}

/** One small bar per core, green to orange to red with the load. */
function renderCores(cores: number[]): void {
  const box = byId('cpu-cores');
  if (box.children.length !== cores.length) {
    box.replaceChildren(...cores.map(() => node('span', 'core')));
  }
  cores.forEach((value, index) => {
    const bar = box.children[index] as HTMLElement;
    bar.style.setProperty('--h', Math.max(4, value) + '%');
    bar.className = 'core' + (value >= 85 ? ' hot' : value >= 50 ? ' warm' : '');
    bar.title = 'core ' + index + ': ' + Math.round(value) + '%';
  });
}

/** Label: value lines under a machine card. */
function statRows(id: string, rows: [string, string][]): void {
  if (unchanged('stats:' + id, rows)) return;
  byId(id).replaceChildren(...rows.flatMap(([label, value]) => [
    node('span', 'stat-label', label), node('span', 'stat-value', value)]));
}

function renderMachine(current: PanelStatus): void {
  const system = current.system;
  const gpu = system.gpu;
  if (gpu) {
    const util = gpu.utilization ?? 0;
    byId('gpu-value').textContent = util + '%';
    setBar('gpu-bar', util, util > 85 ? 'warn' : '');
    pushUsage('gpu', util);
    renderSpark('gpu-spark', usage.gpu, 'var(--spark-gpu)');
    const busy = gpu.process_list.length || gpu.processes;
    const rows: [string, string][] = [['Scheda', gpu.name.replace(/^Tesla /, '')]];
    if (gpu.memory_total_mb && gpu.memory_used_mb !== null) {
      rows.push(['Memoria', (gpu.memory_used_mb / 1024).toFixed(1) + ' di ' + Math.round(gpu.memory_total_mb / 1024) + ' GB']);
    }
    if (gpu.temperature !== null) rows.push(['Temperatura', gpu.temperature + ' °C']);
    if (gpu.power_w !== null) rows.push(['Potenza', Math.round(gpu.power_w) + ' W']);
    rows.push(['Lavora a', busy ? busy + (busy === 1 ? ' processo' : ' processi') : 'niente (libera)']);
    statRows('gpu-sub', rows);
  } else {
    byId('gpu-value').textContent = 'n/d';
    setBar('gpu-bar', 0);
    statRows('gpu-sub', [['Scheda', 'nessuna GPU NVIDIA trovata']]);
  }
  if (system.cpus) {
    const cpu = system.cpu;
    if (cpu) {
      byId('cpu-value').textContent = Math.round(cpu.total) + '%';
      pushUsage('cpu', cpu.total);
      renderCores(cpu.cores);
    } else {
      byId('cpu-value').textContent = '…';
    }
    renderSpark('cpu-spark', usage.cpu, 'var(--spark-cpu)');
    const rows: [string, string][] = cpu
      ? [['Core occupati', cpu.cores.filter((value) => value >= 50).length + ' di ' + system.cpus]]
      : [['Core', String(system.cpus)]];
    if (system.loads) {
      // Load average = programs running or waiting: per core it says how loaded the machine was over
      // the last 15 minutes, and comparing 1 and 15 minutes gives the trend.
      const [one, , fifteen] = system.loads;
      const perCore = fifteen / system.cpus;
      rows.push(['Ultimi 15 minuti', perCore < 0.25 ? 'leggero' : perCore < 0.7 ? 'moderato' : 'pesante']);
      rows.push(['Andamento', one > fifteen * 1.3 + 0.5 ? 'in salita' : one < fifteen * 0.7 - 0.5 ? 'in calo' : 'stabile']);
    }
    statRows('cpu-sub', rows);
  }
  const memory = system.memory;
  if (memory) {
    byId('ram-value').textContent = memory.available_gb + ' GB liberi';
    setBar('ram-bar', memory.used_percent, memory.used_percent > 90 ? 'bad' : memory.used_percent > 75 ? 'warn' : '');
    statRows('ram-sub', [
      ['In uso', (memory.total_gb - memory.available_gb).toFixed(1) + ' GB (' + memory.used_percent + '%)'],
      ['Totale', memory.total_gb + ' GB']]);
  } else {
    byId('ram-value').textContent = 'n/d';
  }
  const disk = system.disk;
  if (disk) {
    const used = 100 - disk.free_gb * 100 / disk.total_gb;
    byId('disk-value').textContent = disk.free_gb + ' GB liberi';
    setBar('disk-bar', used, disk.free_gb < 10 ? 'bad' : used > 80 ? 'warn' : '');
    statRows('disk-sub', [
      ['In uso', (disk.total_gb - disk.free_gb).toFixed(1) + ' GB (' + Math.round(used) + '%)'],
      ['Totale', disk.total_gb + ' GB']]);
  }
  const processes = gpu ? gpu.process_list : [];
  if (unchanged('gpu-procs', processes)) return;
  byId('gpu-procs-empty').hidden = processes.length > 0;
  byId('gpu-procs').replaceChildren(...processes.map((process) => {
    const row = node('tr');
    row.append(
      node('td', 'mono', process.pid === null ? '?' : String(process.pid)),
      node('td', '', process.name),
      node('td', 'num', process.memory_mb === null ? '–' : Math.round(process.memory_mb) + ' MiB'));
    return row;
  }));
}

function jobItem(job: Job): HTMLLIElement {
  const item = node('li', 'item');
  const [label, kind] = JOB_STATE[job.state];
  // Name and start time stacked, so on a phone the name keeps the width.
  const text = node('div', 'grow');
  text.append(node('div', 'what', job.label), node('div', 'sub', job.started));
  const view = node('button', 'btn small', 'Dettagli');
  view.type = 'button';
  view.addEventListener('click', () => openLog(job.id, job.label));
  item.append(pill(label, kind), text, view);
  if (job.stoppable) {
    const stop = node('button', 'btn small danger', 'Ferma');
    stop.type = 'button';
    stop.addEventListener('click', () => void stopJob(job));
    item.appendChild(stop);
  }
  return item;
}

/** What runs now: the jobs started from the site and the same work started from a terminal. */
function renderRunning(current: PanelStatus): void {
  const prep = current.preparation;
  const running = current.jobs.filter((job) => job.state === 'running');
  if (unchanged('running', [running, current.activities, prep])) return;
  const fromSite = new Set(running.map((job) => job.label));
  const items: HTMLLIElement[] = running.map(jobItem);
  for (const activity of current.activities) {
    if (fromSite.has(activity.label)) continue;
    const row = node('li', 'item');
    row.append(pill('in corso', 'run'), node('span', 'what grow', activity.label), node('span', 'sub', 'pid ' + activity.pid));
    if (activity.label === 'preparazione' && prep && prep.percent !== null) {
      const bar = node('div', 'bar run');
      const fill = node('span');
      fill.style.width = prep.percent + '%';
      bar.appendChild(fill);
      row.append(bar, node('span', 'sub',
        prep.routes_done + ' di ' + prep.routes_total + ' route pronte · ' + prep.percent + '%'));
    }
    items.push(row);
  }
  byId('running').replaceChildren(...items);
  byId('running-empty').hidden = items.length > 0;
}

function renderJobs(current: PanelStatus): void {
  if (unchanged('jobs', current.jobs)) return;
  byId('jobs-empty').hidden = current.jobs.length > 0;
  byId('jobs').replaceChildren(...current.jobs.map(jobItem));
}

function renderUpdate(current: PanelStatus): void {
  const state = current.actions.update;
  if (unchanged('update', state)) return;
  const wrap = node('div', 'action');
  const button = node('button', 'btn', state.label);
  button.type = 'button';
  button.disabled = !state.allowed;
  button.addEventListener('click', () => void run('update'));
  wrap.appendChild(button);
  if (!state.allowed && state.reason) {
    const reason = node('div', 'reason', state.reason);
    reason.id = 'reason-update';
    button.setAttribute('aria-describedby', reason.id);
    wrap.appendChild(reason);
  }
  byId('update-buttons').replaceChildren(wrap);
}

function render(current: PanelStatus): void {
  byId('readonly').hidden = current.actions_enabled;
  renderMachine(current);
  renderRunning(current);
  renderJobs(current);
  renderUpdate(current);
  if (log.id) {
    const job = current.jobs.find((item) => item.id === log.id);
    byId('sheet-state').hidden = !(job && job.state === 'running');
  }
}

function showLive(text: string, ok: boolean): void {
  byId('live-dot').className = ok ? 'dot' : 'dot off';
  byId('live-dot').title = text;
  byId('live-text').textContent = text;
}

async function refresh(): Promise<void> {
  try {
    status = await api<PanelStatus>('/api/panel/status');
    render(status);
    showLive('aggiornato ' + clock(), true);
  } catch (error) {
    showLive('non raggiungibile: ' + messageOf(error), false);
  }
}

async function run(name: ActionName): Promise<void> {
  if (!status) return;
  if (!window.confirm(status.actions[name].label + ': git pull e riavvio del sito (qualche secondo). Avvio?')) return;
  try {
    const job = await post<StartedJob>('/api/panel/run/' + name);
    toast('Partito: ' + job.label);
    await refresh();
    openLog(job.id, job.label);
  } catch (error) {
    toast(messageOf(error), true);
    void refresh();
  }
}

async function stopJob(job: Job): Promise<void> {
  if (!window.confirm('Fermo "' + job.label + '"? Il lavoro già fatto resta, si riprende rilanciando.')) return;
  try {
    await post('/api/panel/job/' + job.id + '/stop');
    toast('Fermato: ' + job.label);
  } catch (error) {
    toast(messageOf(error), true);
  }
  void refresh();
}

// -- the details of a job (its log) in the bottom sheet --------------------------------------------------
function openLog(id: string, label: string): void {
  window.clearTimeout(log.timer);
  log = {id, offset: 0, timer: undefined, follow: true};
  byId('sheet-title').textContent = label;
  byId('sheet-text').textContent = '';
  byId('sheet-follow').setAttribute('aria-pressed', 'true');
  byId('sheet').classList.add('open');
  byId('sheet-backdrop').classList.add('open');
  document.body.classList.add('sheet-open');
  void pollLog();
}

function closeLog(): void {
  window.clearTimeout(log.timer);
  log.id = null;
  byId('sheet').classList.remove('open');
  byId('sheet-backdrop').classList.remove('open');
  document.body.classList.remove('sheet-open');
}

async function pollLog(): Promise<void> {
  const id = log.id;
  if (!id) return;
  try {
    const data = await api<LogChunk>('/api/panel/job/' + id + '/log?offset=' + log.offset);
    if (log.id !== id) return;
    const pre = byId('sheet-text');
    if (data.text) {
      if (log.offset === 0 && data.offset - data.text.length > 0) pre.textContent = '… (inizio tagliato)\n';
      pre.textContent += data.text;
      if (log.follow) pre.scrollTop = pre.scrollHeight;
    }
    log.offset = data.offset;
  } catch (error) {
    toast(messageOf(error), true);
    return;
  }
  const job = status?.jobs.find((item) => item.id === id);
  const running = !job || job.state === 'running';
  byId('sheet-state').hidden = !running;
  log.timer = window.setTimeout(() => void pollLog(), running ? 1500 : 5000);
}

byId('sheet-close').addEventListener('click', closeLog);
byId('sheet-backdrop').addEventListener('click', closeLog);
byId('sheet-follow').addEventListener('click', (event) => {
  log.follow = !log.follow;
  (event.currentTarget as HTMLElement).setAttribute('aria-pressed', String(log.follow));
  if (log.follow) byId('sheet-text').scrollTop = byId('sheet-text').scrollHeight;
});
document.addEventListener('keydown', (event) => {
  if (event.key === 'Escape' && log.id) closeLog();
});

initThemeSwitch();
void refresh();
window.setInterval(() => void refresh(), 4000);
