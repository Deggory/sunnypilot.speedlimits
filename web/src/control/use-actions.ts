// Starting an action of the control pages (a job in the background) and stopping a running job, each asked
// first in the confirm dialog when it is long, remote or disruptive.
import {messageOf, post} from '../common/api';
import type {ActionName, ActionOptions, ActionState, Job, StartedJob, TrainingStart} from '../common/panel-types';
import {askConfirm, type ConfirmOptions} from '../composables/confirm';
import {toast} from '../composables/toast';
import type {InjectionKey} from 'vue';

/** The resume() of the control app, for the job cards (provided by ControlApp.vue). */
export const RESUME_JOB: InjectionKey<(job: Job) => Promise<void>> = Symbol('resume-job');

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
  if (name === 'train_classifier') {
    return {title: 'Train the classifier?', action: 'Start training',
      text: 'Mines the false boxes of the cutter in use, trains the classifier with “not a sign” on the GPU, then measures it '
        + 'alone and in the chain: it takes hours. Its run is a new one, named by the time it starts.'};
  }
  if (name === 'setup_vlm') {
    return {title: 'Install Qwen3-VL?', action: 'Install',
      text: 'Downloads the model (17.5 GB, the files already copied here are kept), installs what it needs and tries it on the GPU. '
        + 'It can take an hour on a slow line.'};
  }
  if (name === 'measure_proposer') {
    return {title: 'Measure ' + proposerName(options?.proposer) + '?', action: 'Measure',
      text: 'It proposes the signs of the reviewed test frames, as Prepare would, and compares them with the boxes you kept. '
        + 'Nothing changes in the reviews or the routes. Our networks and Qwen3-VL use the GPU.'};
  }
  if (name === 'ai_review' && options?.save === 'yes') {
    return {title: 'Review and save with the AI?', action: 'Review and save', danger: true,
      text: 'AI saves are final: nobody approves them again, and the next import puts them into training. Frames a '
        + 'person confirmed are never touched.'};
  }
  if (name === 'ai_review') {
    return {title: 'Try the AI reviewer?', action: 'Try',
      text: 'It suggests the boxes of ' + (options?.limit ? 'up to ' + options.limit + ' ' : 'the ')
        + 'frames still to review and writes them to a file. Nothing is saved.'};
  }
  if (name === 'measure_ai_reviewer') {
    return {title: 'Measure ' + (options?.provider ?? 'the AI reviewer') + '?', action: 'Measure',
      text: 'It asks the AI about up to ' + (options?.limit ?? '60') + ' frames a person reviewed and compares its '
        + 'answers with theirs. Nothing is saved; it takes minutes to an hour.'};
  }
  if (name === 'prepare_again') {
    return {title: 'Propose the signs again?', action: 'Propose again',
      text: 'The routes nobody has reviewed a frame of get new proposals from ' + proposerName(options?.proposer)
        + '. Routes already in review are left as they are.'};
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
      text: 'The work already done stays: starting it again carries on from there.'});
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

  return {run, stop, resume};
}
