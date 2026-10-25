// The jobs of the control pages: their state in words, their times, which job a tab shows, the work started
// from a terminal.
import {shortWhen} from '../common/format';
import type {ActionName, Activity, Job, JobState, PanelStatus} from '../common/panel-types';
import type {IconName} from '../components/icons/shapes';

/** State of a job: label and pill class. */
export const JOB_STATE: Record<JobState, [label: string, kind: string]> = {
  running: ['running', 'run'], ok: ['done', 'ok'], failed: ['failed', 'bad'], interrupted: ['interrupted', 'warn'],
};

/** The group of each action (nnslr_tools.panel.ACTIONS): a terminal activity of the same group is that work. */
export const ACTION_GROUP: Record<ActionName, string> = {
  fetch: 'fetch', prepare: 'prepare', prepare_again: 'prepare', measure_proposer: 'train', setup_vlm: 'train', dataset: 'train', train: 'train', train_continue: 'train', train_resume: 'train',
  train_classifier: 'train', train_cutter: 'train', update: 'update', ai_review: 'ai', measure_ai_reviewer: 'ai',
};

/** The icon of each action, on its button. */
export const ACTION_ICON: Record<ActionName, IconName> = {
  fetch: 'download', prepare: 'prepare', prepare_again: 'resume', measure_proposer: 'train', setup_vlm: 'download', dataset: 'import', train: 'train', train_continue: 'train',
  train_resume: 'resume', train_classifier: 'train', train_cutter: 'train', update: 'update', ai_review: 'ai',
  measure_ai_reviewer: 'ai',
};

/** The actions whose jobs the Prepare and Train tabs show. */
export const PREPARE_ACTIONS: ActionName[] = ['fetch', 'prepare', 'prepare_again', 'setup_vlm'];
export const TRAIN_ACTIONS: ActionName[] = ['measure_proposer', 'dataset', 'train', 'train_continue', 'train_resume', 'train_classifier', 'train_cutter',
  'ai_review', 'measure_ai_reviewer'];
/** The trainings whose job shows its curves on the Train tab. */
export const CHARTED_ACTIONS: ActionName[] = ['train', 'train_continue', 'train_classifier', 'train_cutter'];
/** The trainings of the whole pipeline (helpers/3_train.sh): a stopped one can be resumed on its run. */
export const PIPELINE_ACTIONS: ActionName[] = ['train', 'train_continue', 'train_resume'];

/**
 * When the job started. The server writes started_at with its offset when it has it; otherwise "started",
 * the server's clock without a zone, which on this server (the LXC) is UTC.
 */
export function startedAt(job: Job): Date | null {
  if (job.started_at) {
    const parsed = Date.parse(job.started_at);
    if (!Number.isNaN(parsed)) return new Date(parsed);
  }
  const match = /^(\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2})/.exec(job.started);
  if (!match) return null;
  const [year = 0, month = 1, day = 1, hour = 0, minute = 0, second = 0] = match.slice(1).map(Number);
  return new Date(Date.UTC(year, month - 1, day, hour, minute, second));
}

const pad = (value: number): string => String(value).padStart(2, '0');

/** "18:55" today, otherwise "23 Oct 18:55", in the browser's time. */
export function startedWhen(job: Job): string {
  const start = startedAt(job);
  return start === null ? job.started : shortWhen(start);
}

/** "42 s", "12 min 03 s", "2 h 05 min". */
export function elapsedText(ms: number): string {
  const seconds = Math.max(0, Math.floor(ms / 1000));
  if (seconds < 60) return seconds + ' s';
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return minutes + ' min ' + pad(seconds % 60) + ' s';
  return Math.floor(minutes / 60) + ' h ' + pad(minutes % 60) + ' min';
}

/** A rough remaining time, in whole minutes: "less than a minute", "40 min", "1 h 10 min". */
export function roughText(ms: number): string {
  const minutes = Math.round(ms / 60000);
  if (minutes < 1) return 'less than a minute';
  if (minutes < 60) return minutes + ' min';
  return Math.floor(minutes / 60) + ' h' + (minutes % 60 ? ' ' + (minutes % 60) + ' min' : '');
}

/** The job a tab shows for these actions: the running one, else the latest. */
export function jobOf(status: PanelStatus | null, actions: ActionName[]): Job | null {
  const jobs = (status?.jobs ?? []).filter((job) => actions.includes(job.action));
  return jobs.find((job) => job.state === 'running') ?? jobs[0] ?? null;
}

/** The work started from a terminal (or by another program) that is not a job of the site. */
export function terminalActivities(status: PanelStatus | null, groups?: string[]): Activity[] {
  if (status === null) return [];
  const fromSite = new Set(status.jobs.filter((job) => job.state === 'running').map((job) => ACTION_GROUP[job.action]));
  // A process whose command line matches two labels (same pid) is shown once.
  const pids = new Set<number>();
  return status.activities.filter((activity) => {
    if (pids.has(activity.pid)) return false;
    pids.add(activity.pid);
    return !fromSite.has(activity.group) && (groups === undefined || groups.includes(activity.group));
  });
}

/** The step a failed job stopped at, if the log said. */
export function failedStep(job: Job): string {
  return job.progress.steps.find((step) => step.state === 'failed')?.label ?? '';
}
