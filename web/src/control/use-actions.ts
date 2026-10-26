// Starting an action of the control pages (a job in the background) and stopping a running job, each asked
// first in the confirm dialog when it is long, remote or disruptive.
import {messageOf, post} from '../common/api';
import type {ActionName, ActionOptions, ActionState, Job, StartedJob, TrainingStart} from '../common/panel-types';
import {askConfirm, type ConfirmOptions} from '../composables/confirm';
import {toast} from '../composables/toast';
import type {InjectionKey} from 'vue';

/** The resume() of the control app, for the job cards (provided by ControlApp.vue). */
export const RESUME_JOB: InjectionKey<(job: Job) => Promise<void>> = Symbol('resume-job');
/** "Try again" on the card of a failed job (provided by ControlApp.vue): whether it can start now, and start it with
 * the proposer it had. */
export const RETRY_JOB: InjectionKey<{can: (job: Job) => boolean; run: (job: Job) => void}> = Symbol('retry-job');
/** The actions whose failed job can simply run again: no option but the proposer (a training has Resume). */
export const RETRY_ACTIONS: ActionName[] = ['fetch', 'prepare', 'prepare_again', 'setup_vlm', 'dataset', 'measure_proposer',
  'networks_prepare'];

/** The question before an action; null: it starts at the click (preparing, importing the reviews). */
/** The proposer as the Prepare tab names it. */
function proposerName(proposer: string | undefined): string {
  return proposer === 'ours' ? 'our cutter and classifier' : proposer === 'qwen' ? 'Qwen3-VL' : 'the Vietnamese YOLO of Hugging Face';
}

function question(name: ActionName, state: ActionState, start: TrainingStart | null, options: ActionOptions | null): ConfirmOptions | null {
  if (name === 'train' || name === 'train_continue') {
    const from = start ? ' It starts from: ' + start.label + '.' : '';
    return {title: 'Train the whole pipeline?', action: 'Start training',
      text: 'The import, the split, then the detector and the reader on the GPU: it takes hours.' + from};
  }
  if (name === 'train_cutter') {
    const what = options?.families === 'speed' ? 'the speed signs only' : 'the speed signs and the other signs';
    return {title: 'Train the cutter?', action: 'Start training',
      text: 'Trains a detector at ' + (options?.size ?? 'the default size').replace('x', ' × ') + ' that finds ' + what
        + ' on the GPU, then measures it on test and validation: it takes hours. Its run is a new one, named by the time it starts.'};
  }
  if (name === 'train_decider') {
    return {title: 'Train the decider?', action: 'Start training',
      text: 'Trains three deciders on the GPU (the crop alone; with position and speed; with the scene) and compares them: '
        + 'about 15 minutes. Its run is a new one, named by the time it starts.'};
  }
  if (name === 'train_classifier') {
    return {title: 'Train the classifier?', action: 'Start training',
      text: 'Mines the false boxes of the cutter in use, trains the classifier with “not a sign” on the GPU, then measures it '
        + 'alone and in the chain: it takes hours. Its run is a new one, named by the time it starts.'};
  }
  if (name === 'networks_publish') {
    return {title: 'Publish ' + (options?.package ?? 'the package') + ' on Hugging Face?', action: 'Publish', danger: true,
      text: 'Uploads the cutter, the classifier and the decider of this package, their model card and the catalog to '
        + 'Hugging Face, public: anyone can download them, and a copy may stay even if you delete it later. No frame '
        + 'or dataset is uploaded.'};
  }
  if (name === 'setup_vlm') {
    return {title: 'Install Qwen3-VL?', action: 'Install',
      text: 'Downloads the model (17.5 GB, the files already copied here are kept), installs what it needs and tries it on the GPU. '
        + 'It can take an hour on a slow line.'};
  }
  if (name === 'measure_proposer') {
    return {title: 'Measure ' + proposerName(options?.proposer) + '?', action: 'Measure',
      text: 'It proposes the signs of the reviewed test frames, as Prepare would, and compares them with the boxes you kept. '
        + 'Nothing changes in the reviews or the drives. Our networks and Qwen3-VL use the GPU.'};
  }
  if (name === 'ai_review' && options?.save === 'yes') {
    return {title: 'Review and save with the AI?', action: 'Review and save', danger: true,
      text: 'AI saves are final: nobody approves them again, and the next import puts them into training. Frames a '
        + 'person confirmed are never touched.'};
  }
  if (name === 'ai_review') {
    return {title: 'Try the AI reviewer?', action: 'Try',
      text: 'It suggests the boxes of ' + (options?.limit ? 'up to ' + options.limit + ' ' : 'the ')
        + 'frames still to review; the review page shows them on each frame. Nothing is saved.'};
  }
  if (name === 'measure_ai_reviewer') {
    return {title: 'Measure ' + (options?.provider ?? 'the AI reviewer') + '?', action: 'Measure',
      text: 'It asks the AI about up to ' + (options?.limit ?? '60') + ' frames a person reviewed and compares its '
        + 'answers with theirs. Nothing is saved; it takes minutes to an hour.'};
  }
  if (name === 'prepare_again') {
    return {title: 'Propose the signs again?', action: 'Propose again',
      text: 'The frames not confirmed yet get new proposals from ' + proposerName(options?.proposer)
        + ', on every drive. The frames already confirmed stay as they are. Each drive is read only until its new '
        + 'proposals are in.'};
  }
  if (name === 'fetch') {
    return {title: state.label + '?', action: 'Download',
      text: 'Copies the new drives from the comma into the video folder, segment by segment.'};
  }
  if (name === 'update') {
    return {title: 'Update the site?', action: 'Update',
      text: 'git pull, the libraries, then the site restarts: it is away for a few seconds.'};
  }
  return null;
}

/** What stopping a job keeps, and what starting it again does: only a download and the install carry on from where
 * they stopped; a training of the whole pipeline has Resume; a measure starts over. */
const STOP_TEXT: Partial<Record<ActionName, string>> = {
  fetch: 'The drives already copied stay: downloading again carries on from there.',
  prepare: 'The drives already prepared stay. The one being prepared now starts again from its first picture next time.',
  prepare_again: 'The drives already done keep their new signs; the one at work keeps its old ones. Propose again '
    + 'starts over on every drive with frames not confirmed, those done now included.',
  measure_proposer: 'The measurement is lost: measuring again starts from the first frame.',
  measure_ai_reviewer: 'The measurement is lost: measuring again starts from the first frame.',
  setup_vlm: 'The files already downloaded stay: installing again carries on from there.',
  train: 'The steps already done stay: Resume runs the missing ones on the same run.',
  train_continue: 'The steps already done stay: Resume runs the missing ones on the same run.',
  train_resume: 'The steps already done stay: Resume runs the missing ones on the same run.',
  train_classifier: 'This training is lost: training again starts a new run from the beginning.',
  train_cutter: 'This training is lost: training again starts a new run from the beginning.',
  train_decider: 'The deciders already trained stay; training again starts a new run.',
  ai_review: 'The frames already saved stay saved; the others are left to review.',
  networks_prepare: 'Nothing is kept: preparing again starts over.',
  networks_publish: 'What Hugging Face already received may stay there.',
};

/** run() and stop() for the buttons; refresh reads /api/panel/status again. */
export function usePanelActions(refresh: () => Promise<void>) {
  /** start: where a training starts (the selector of the Train tab); the server's default without it. options: the
   * cutter's size and signs. */
  async function run(name: ActionName, state: ActionState, start: TrainingStart | null = null,
    options: ActionOptions | null = null): Promise<void> {
    const ask = question(name, state, start, options);
    if (ask !== null && !await askConfirm(ask)) return;
    try {
      const job = await post<StartedJob>('/api/panel/run/' + name, start ? {start: start.key} : options ? {...options} : {});
      toast('Started: ' + job.label);
      await refresh();
    } catch (error) {
      toast(messageOf(error), true);
      void refresh();
    }
  }

  async function stop(job: Job): Promise<void> {
    const ok = await askConfirm({title: 'Stop “' + job.label + '”?', action: 'Stop', danger: true,
      text: STOP_TEXT[job.action] ?? 'It stops where it is now.'});
    if (!ok) return;
    try {
      await post('/api/panel/job/' + job.id + '/stop');
      toast('Stopped: ' + job.label);
    } catch (error) {
      toast(messageOf(error), true);
    }
    void refresh();
  }

  /** "Resume": the steps a stopped training still misses, on the same run (helpers/3_train.sh --resume). */
  async function resume(job: Job): Promise<void> {
    if (!job.run) return;
    const ok = await askConfirm({title: 'Resume “' + job.run + '”?', action: 'Resume',
      text: 'Only the steps still missing run again, on the same run: what is done stays.'});
    if (!ok) return;
    try {
      const started = await post<StartedJob>('/api/panel/run/train_resume', {run: job.run});
      toast('Started: ' + started.label);
      await refresh();
    } catch (error) {
      toast(messageOf(error), true);
      void refresh();
    }
  }

  /** "Try again": the same action, with the proposer it had (asked first as the action's own button asks). */
  async function retry(job: Job, state: ActionState): Promise<void> {
    await run(job.action, state, null, job.proposer ? {proposer: job.proposer} : null);
  }

  return {run, stop, resume, retry};
}
