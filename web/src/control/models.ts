// The Models tab: the three networks of the pipeline (cutter, classifier, decider; docs/NEURAL_NETWORK.md), every
// version trained on this server (GET /api/panel/models every 10 s while the tab is open), the one in use for each
// role (POST /api/panel/models/active, asked first in the confirm dialog), and the words the tab shows.
import {onBeforeUnmount, onMounted, ref, shallowRef} from 'vue';
import {api, messageOf, post} from '../common/api';
import {isoWhen} from '../common/format';
import type {ActiveChosen, ActiveModel, ModelRole, ModelVersion, ModelsPayload} from '../common/panel-types';
import {askConfirm} from '../composables/confirm';
import {toast} from '../composables/toast';
import {chooseRun} from './evaluate';
import {goTo} from './tabs';

const INTERVAL_MS = 10000;

/** What each role does, what plays it today, and the words when none is trained yet. */
export const ROLE_TEXT: Record<ModelRole, {what: string; today: string; empty: string; emptyMore: string}> = {
  cutter: {
    what: 'Finds every box that could be a sign. It must miss nothing: the classifier drops its false boxes.',
    today: 'Today: the detector, judged on the signs it misses by size.',
    empty: 'No cutter trained yet: the detector of a training run plays this role.',
    emptyMore: 'Start a training on the Train tab; its detector appears here.',
  },
  classifier: {
    what: 'Says what each box is: a speed value, end of limit, another sign, or not a sign.',
    today: 'Today: the reader, judged on right answers per sign; the classifier with “not a sign” also on the false boxes '
      + 'it rejects and on the chain end to end.',
    empty: 'No classifier trained yet: the reader of a training run plays this role.',
    emptyMore: 'Start a training, or “Train the classifier”, on the Train tab; it appears here.',
  },
  decider: {
    what: 'Says whom each sign applies to: our road, another road, or abstain.',
    today: 'Today: only the prototypes of experiment B.',
    empty: 'No decider trained yet: experiment B is the first prototype.',
    emptyMore: 'Its variants appear here once scripts/run_experiment_b.sh has trained them.',
  },
};

/** The models, asked every 10 s while the tab is shown. */
export function useModels() {
  const payload = shallowRef<ModelsPayload | null>(null);
  const error = ref('');
  let timer: number | undefined;
  async function refresh(): Promise<void> {
    try {
      payload.value = await api<ModelsPayload>('/api/panel/models');
      error.value = '';
    } catch (problem) {
      error.value = messageOf(problem);
    }
  }
  onMounted(() => {
    void refresh();
    timer = window.setInterval(() => void refresh(), INTERVAL_MS);
  });
  onBeforeUnmount(() => window.clearInterval(timer));
  return {payload, error, refresh};
}

/** "2026-10-24_1452 · Detector 320 × 320": a version in one line. */
export function versionName(version: ModelVersion): string {
  return version.run + ' · ' + version.label;
}

/** Asks, then makes a version the one in use for its role; true when it is. */
export async function putInUse(version: ModelVersion, roleLabel: string): Promise<boolean> {
  const role = roleLabel.toLowerCase();
  const ok = await askConfirm({
    title: 'Use this ' + role + '?', action: 'Use this one',
    text: versionName(version) + ' becomes the ' + role + ' in use on this server. It is only recorded '
      + '(models/active.json): nothing is copied, exported or sent to the comma.',
  });
  if (!ok) return false;
  try {
    await post<ActiveChosen>('/api/panel/models/active', {role: version.role, id: version.id});
    toast(roleLabel + ' in use: ' + versionName(version));
    return true;
  } catch (problem) {
    toast(messageOf(problem), true);
    return false;
  }
}

/** Who chose the model in use, and when: "chosen 24 Oct 16:40 by admin". */
export function chosenText(active: ActiveModel): string {
  const when = isoWhen(active.at);
  const who = active.via === 'sign-in' && active.who ? 'by ' + active.who
    : active.via === 'token' ? 'by an agent with the access token'
      : active.via === 'local' ? 'on this computer' : '';
  return ['chosen', when, who].filter(Boolean).join(' ');
}

/** Opens a run on the Evaluate tab (a click with a modifier key opens the link as usual). */
export function openRun(event: MouseEvent, run: string): void {
  if (event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
  event.preventDefault();
  goTo('evaluate');
  chooseRun(run);
}

/** "320x320" -> "320 × 320". */
export function sizeText(size: string): string {
  return size.replace('x', ' × ');
}

/** 1.234 -> "1.23"; null -> "–". */
export function perFrame(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return '–';
  return value >= 100 ? value.toFixed(0) : value >= 10 ? value.toFixed(1) : value.toFixed(2);
}

/** "8 of 20 epochs", "8", "–". */
export function epochText(best: number | null, epochs: number | null): string {
  if (best === null) return '–';
  return epochs ? best + ' of ' + epochs : String(best);
}

/** Where the weights started. */
export function initText(initialization: string | null): string {
  if (initialization === 'imagenet1k_v1') return 'ImageNet';
  if (initialization === 'random') return 'random weights';
  return initialization ?? '–';
}
