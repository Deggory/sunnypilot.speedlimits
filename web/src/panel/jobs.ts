// A job started from a step shows as a card in that step: state, time, progress when known and the
// last line of its log as a sentence. The raw log stays folded in "Dettagli tecnici"; Stato has the rest.
import {api} from '../common/api';
import {byId, node, pill} from '../common/dom';
import {when} from '../common/format';
import type {ActionName, Job, JobState, LogChunk, PanelStatus} from '../common/panel-types';
import {stopJob, type ActionContext} from './actions';

/** The box of a step and the actions whose jobs it shows. */
const BOXES: [box: string, actions: ActionName[]][] = [
  ['prepare-live', ['fetch', 'prepare']],
  ['train-live', ['dataset', 'train', 'train_continue']],
];
const JOB_STATE: Record<JobState, [label: string, kind: string]> = {
  running: ['in corso', 'run'], ok: ['finito', 'ok'], failed: ['fallito', 'bad'], interrupted: ['interrotto', 'warn'],
};
const RECENT_MINUTES = 60;  // a finished job stays this long after its start, or until closed
const TAIL_BYTES = 6000;    // how much of the end of the log a card reads
const CLOSED_KEY = 'nnslr-panel-closed';

/** The end of a job's log, as read at this log size, and its last line that says something. */
interface Tail {
  size: number;
  text: string;
  line: string;
}

/** The parts of a card that change at every refresh. */
interface Card {
  /** Job id, state and stoppable: the card is built again only when this changes. */
  key: string;
  root: HTMLElement;
  sub: HTMLElement;
  bar: HTMLElement;
  fill: HTMLElement;
  line: HTMLElement;
  log: HTMLElement;
  details: HTMLDetailsElement;
}

const cards = {
  /** Jobs seen running here (or started here) stay after they end, until closed. */
  seen: new Set<string>(),
  /** Closed by the user, for this browser tab. */
  dismissed: loadDismissed(),
  tails: new Map<string, Tail>(),
  shown: new Map<string, Card>(),
  /** The last status and context, to draw again when a card is closed. */
  last: null as {status: PanelStatus; context: ActionContext} | null,
};

function loadDismissed(): Set<string> {
  try {
    const stored: unknown = JSON.parse(sessionStorage.getItem(CLOSED_KEY) ?? '[]');
    return new Set(Array.isArray(stored) ? stored.filter((id): id is string => typeof id === 'string') : []);
  } catch {
    return new Set();  // private mode
  }
}

function dismiss(id: string): void {
  cards.dismissed.add(id);
  try {
    sessionStorage.setItem(CLOSED_KEY, JSON.stringify([...cards.dismissed]));
  } catch {
    // Storage off: closed until the page is left.
  }
  if (cards.last) renderJobCards(cards.last.status, cards.last.context);
}

/** A job started from this page keeps its card after it ends. */
export function followJob(id: string): void {
  cards.seen.add(id);
}

function startedAt(job: Job): Date | null {
  const match = /^(\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2})/.exec(job.started);
  if (!match) return null;
  const [year = 0, month = 1, day = 1, hour = 0, minute = 0, second = 0] = match.slice(1).map(Number);
  return new Date(year, month - 1, day, hour, minute, second);
}

function duration(ms: number): string {
  const minutes = Math.floor(ms / 60000);
  if (minutes < 1) return 'meno di un minuto';
  if (minutes < 60) return minutes + ' min';
  return Math.floor(minutes / 60) + ' h ' + (minutes % 60) + ' min';
}

/** The job a box shows: the running one, else the last one if seen running here or recent. */
function liveJob(status: PanelStatus, actions: ActionName[]): Job | null {
  const jobs = status.jobs.filter((job) => actions.includes(job.action) && !cards.dismissed.has(job.id));
  const running = jobs.find((job) => job.state === 'running');
  if (running) {
    cards.seen.add(running.id);
    return running;
  }
  const last = jobs[0];
  if (!last) return null;
  const start = startedAt(last);
  const recent = start !== null && Date.now() - start.getTime() < RECENT_MINUTES * 60000;
  return cards.seen.has(last.id) || recent ? last : null;
}

// The end of the log, read again only when it grew; its last line that says something.
async function readTail(job: Job): Promise<Tail> {
  const known = cards.tails.get(job.id);
  if (known && known.size === job.size) return known;
  const from = Math.max(0, job.size - TAIL_BYTES);
  const data = await api<LogChunk>('/api/panel/job/' + job.id + '/log?offset=' + from);
  let lines = data.text.split('\n');
  // Read from the middle of the log, the first line is a piece of a line.
  if (from > 0) lines = lines.slice(1);
  const useful = lines.map((line) => line.trim()).filter((line) => line && !line.startsWith('=='));
  const tail = {size: job.size, text: (from > 0 ? '… (inizio tagliato)\n' : '') + lines.join('\n'),
    line: useful[useful.length - 1] ?? ''};
  cards.tails.set(job.id, tail);
  return tail;
}

function buildCard(job: Job, key: string, context: ActionContext): Card {
  const [label, kind] = JOB_STATE[job.state];
  const root = node('div', 'jobcard ' + kind);
  const head = node('div', 'jobcard-head');
  if (job.state === 'running') head.appendChild(node('span', 'spinner'));
  head.append(node('strong', 'jobcard-label', job.label), pill(label, kind));
  const bar = node('div', 'jobcard-bar');
  const fill = node('span');
  bar.appendChild(fill);
  const log = node('pre', 'jobcard-log');
  const details = node('details');
  details.append(node('summary', '', 'Dettagli tecnici'), log);
  const link = node('a', '', 'Dettagli in Stato →');
  link.href = '/stato';
  const foot = node('div', 'jobcard-foot');
  foot.append(link, node('span', 'grow'));
  if (job.state === 'running' && job.stoppable) {
    const stop = node('button', 'btn small danger', 'Ferma');
    stop.type = 'button';
    stop.addEventListener('click', () => void stopJob(job, context));
    foot.appendChild(stop);
  } else if (job.state !== 'running') {
    const close = node('button', 'btn small', 'Chiudi');
    close.type = 'button';
    close.addEventListener('click', () => dismiss(job.id));
    foot.appendChild(close);
  }
  foot.appendChild(details);
  const sub = node('div', 'jobcard-sub');
  const line = node('p', 'jobcard-line');
  root.append(head, bar, sub, line, foot);
  return {key, root, sub, bar, fill, line, log, details};
}

/** "In corso da 12 min · 40% · 2 di 5 route pronte", or how and when it ended. */
function summary(job: Job, status: PanelStatus, percent: number | null): string {
  if (job.state === 'running') {
    const start = startedAt(job);
    let text = 'In corso da ' + (start ? duration(Date.now() - start.getTime()) : '–');
    const prep = status.preparation;
    if (percent !== null && prep) text += ' · ' + percent + '% · ' + prep.routes_done + ' di ' + prep.routes_total + ' route pronte';
    return text;
  }
  const at = 'Partito alle ' + when(job.started);
  if (job.state === 'ok') return at + ', finito.';
  if (job.state === 'failed') return at + ', fallito' + (job.code !== null ? ' (codice ' + job.code + ').' : '.');
  return at + ', fermato.';
}

function showTail(card: Card, tail: Tail | undefined): void {
  card.line.textContent = tail ? tail.line : '';
  card.log.textContent = tail ? tail.text : '';
}

function renderBox(boxId: string, actions: ActionName[], status: PanelStatus, context: ActionContext): void {
  const box = byId(boxId);
  const job = liveJob(status, actions);
  if (!job) {
    box.replaceChildren();
    cards.shown.delete(boxId);
    return;
  }
  const key = job.id + ':' + job.state + ':' + job.stoppable;
  let card = cards.shown.get(boxId);
  if (!card || card.key !== key) {
    // A new card for the new state keeps "Dettagli tecnici" open if it was.
    const open = card?.details.open ?? false;
    card = buildCard(job, key, context);
    card.details.open = open;
    cards.shown.set(boxId, card);
    box.replaceChildren(card.root);
  }
  const progress = job.action === 'prepare' ? status.preparation?.percent ?? null : null;
  card.sub.textContent = summary(job, status, progress);
  card.bar.classList.toggle('indeterminate', job.state === 'running' && progress === null);
  card.fill.style.width = (job.state === 'ok' ? 100 : progress ?? 0) + '%';
  showTail(card, cards.tails.get(job.id));
  const shown = card;
  readTail(job).then((tail) => {
    if (cards.shown.get(boxId) === shown) showTail(shown, tail);
  }).catch(() => { /* the card stays without its last line */ });
}

export function renderJobCards(status: PanelStatus, context: ActionContext): void {
  cards.last = {status, context};
  for (const [boxId, actions] of BOXES) renderBox(boxId, actions, status, context);
}
