// The Models tab: the three networks of the pipeline (cutter, classifier, decider; docs/NEURAL_NETWORK.md), every
// version trained on this server (GET /api/panel/models every 10 s while the tab is open), the one in use for each
// role (POST /api/panel/models/active, asked first in the confirm dialog), and the words the tab shows.
import {onBeforeUnmount, onMounted, ref, shallowRef} from 'vue';
import {api, messageOf, post} from '../common/api';
import {isoWhen, thousands} from '../common/format';
import type {ActionState, ActiveChosen, ActiveModel, ModelRole, ModelVersion, ModelsPayload} from '../common/panel-types';
import {askConfirm, type ConfirmOptions} from '../composables/confirm';
import {toast} from '../composables/toast';
import {chooseRun} from './evaluate';
import {goTo} from './tabs';

const INTERVAL_MS = 10000;

/** What each role does, what plays it today, the words when none is trained yet, and what uses the one in use
 * (Prepare's “Our networks” proposer, nnslr_tools.preannotate.our_networks; the classifier's training, which learns
 * from the false boxes of the cutter in use and is measured with it, scripts/run_classifier.sh). */
export const ROLE_TEXT: Record<ModelRole, {what: string; today: string; empty: string; emptyMore: string; uses: string}> = {
  cutter: {
    what: 'Finds every box that could be a sign. It must miss nothing: the classifier drops its false boxes. '
      + 'A detector does it here; SignNet, made small for the comma’s GPU, does it in the car.',
    today: 'Today: the detector, judged on the signs it misses by size.',
    empty: 'No cutter trained yet: the detector of a training run plays this role.',
    emptyMore: 'Start a training on the Train tab; its detector appears here.',
    uses: 'Prepare’s “Our networks” finds the signs with it. The next “Train the classifier” learns from its '
      + 'false boxes and is measured with it.',
  },
  classifier: {
    what: 'Says what each box is: a speed value, end of limit, another sign, or not a sign.',
    today: 'Today: the reader, judged on right answers per sign; the classifier with “not a sign” also on the false boxes '
      + 'it rejects and on the chain end to end.',
    empty: 'No classifier trained yet: the reader of a training run plays this role.',
    emptyMore: 'Start a training, or “Train the classifier”, on the Train tab; it appears here.',
    uses: 'Prepare’s “Our networks” reads the signs with it.',
  },
  decider: {
    what: 'Says whom each sign applies to: our road, another road, or abstain.',
    today: 'Today: only the prototypes of experiment B.',
    empty: 'No decider trained yet: experiment B is the first prototype.',
    emptyMore: 'Its variants appear here once scripts/run_experiment_b.sh has trained them.',
    uses: 'Nothing uses the decider yet: the choice is only noted for later.',
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
    text: versionName(version) + ' becomes the ' + role + ' in use on this server. ' + ROLE_TEXT[version.role].uses
      + ' Nothing is copied or sent to the comma.',
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

/** “Send to the car” for the SignNet versions: the action's state, the run of the SignNet being sent ('' when the
 * run is not known yet; null when none is sent), and whether a build of the testing branch is still owed. */
export interface CarState {
  state: ActionState | null;
  sending: string | null;
  pending: boolean;
  /** “Try on the comma now”: its state, the run being copied ('' while not known; null: none), the run on trial. */
  tryState?: ActionState | null;
  trying?: string | null;
  onTrial?: string | null;
}

/** The question before "Send to the car" (scripts/send_to_car.sh): which SignNet and which classifier (the one in
 * use) the comma gets, what the job publishes and builds, and what is left to do on the comma. */
export function sendQuestion(signnet: ModelVersion, classifier: ModelVersion | null): ConfirmOptions {
  const reader = classifier ? classifier.label + ' ' + (classifier.trained_at ? isoWhen(classifier.trained_at) : classifier.run)
    : 'the classifier in use';
  return {
    title: 'Send this SignNet to the car?', action: 'Send to the car',
    text: 'The comma gets ' + signnet.label + ' of ' + signnet.run + ' to find the signs and ' + reader
      + ' (the classifier in use) to read them. They are published in speed-limits-models, the sunnypilot speed-limits '
      + 'branch moves to them and psa-torque-sunny-testing is built again: a few minutes. Then, on the comma: '
      + 'Settings, Software, Check for updates, and install.',
  };
}

/** The question before "Try on the comma now" (scripts/try_on_comma.sh): the same two networks copied straight onto the
 * comma, checked first with the comma's own code; no publication, no build. */
export function tryQuestion(signnet: ModelVersion, classifier: ModelVersion | null): ConfirmOptions {
  const reader = classifier ? classifier.label + ' ' + (classifier.trained_at ? isoWhen(classifier.trained_at) : classifier.run)
    : 'the classifier in use';
  return {
    title: 'Try this SignNet on the comma now?', action: 'Try on the comma',
    text: 'The networks go onto the comma now: ' + signnet.label + ' of ' + signnet.run + ' and ' + reader
      + ' (the classifier in use), checked first with the comma\'s own code. Parked, the comma prepares them within a '
      + 'few minutes; they are used from the next drive. Nothing is published or built: the comma\'s next software '
      + 'update puts back the build\'s networks.',
  };
}

/** The question before "Back to the build's networks" (scripts/try_on_comma.sh --restore). */
export function restoreQuestion(): ConfirmOptions {
  return {
    title: 'Back to the build\'s networks?', action: 'Put them back',
    text: 'The comma gets back the networks of the build it runs, from the next drive. Nothing else on the comma is touched.',
  };
}

/** Who chose the model in use, and when: "chosen 24 Oct 16:40 by admin". */
export function chosenText(active: ActiveModel): string {
  const when = isoWhen(active.at);
  const who = active.via === 'sign-in' && active.who ? 'by ' + active.who
    : active.via === 'token' ? 'by an agent with the access token'
      : active.via === 'local' ? 'on this computer' : '';
  return ['chosen', when, who].filter(Boolean).join(' ');
}

/** The address of a version on the Evaluate tab: its run and its id. */
export function evaluateHref(version: ModelVersion): string {
  return '/evaluate?run=' + encodeURIComponent(version.run) + '&model=' + encodeURIComponent(version.id);
}

/** Opens a version on the Evaluate tab, chosen in the view of its role (a click with a modifier key opens the link as
 * usual). */
export function openRun(event: MouseEvent, version: ModelVersion): void {
  if (event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
  event.preventDefault();
  goTo('evaluate');
  chooseRun(version.run, version.id);
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

/** A headline number of a version: what it counts (and a short name for a table header), the share (0-1, null: not
 * measured or not trained for it), the count it is of ("145 of 208") and a line under it. */
export interface Metric {
  key: string;
  label: string;
  short?: string;
  value: number | null;
  count: string;
  detail: string;
}

/** The numbers that judge a version, side by side (MetricTiles, the Models table): the cutter: the speed signs and
 * every sign found on the test at its score, and the speed signs among its 5 best boxes a frame (what the car reads:
 * a detector passes on a couple of hundred boxes a frame at that score, SignNet a few); the classifier with “not a sign”: the speed-sign crops and every sign
 * crop read right (the kind of an other sign aside), and the chain with the cutter; an old reader: its crops read
 * right; the decider: the “Another” signs found on validation and on the test. */
export function metricsOf(version: ModelVersion): Metric[] {
  const of = (part: number | null | undefined, whole: number | null | undefined): string =>
    whole ? thousands(part ?? 0) + ' of ' + thousands(whole) : '';
  if (version.role === 'cutter') {
    const cut = version.test ?? version.validation ?? null;
    if (cut === null) return [];
    const every: Metric = {key: 'every', label: 'Every sign', value: cut.recall, count: of(cut.found, cut.truth_boxes),
      detail: cut.truth_boxes === null ? '' : of(cut.found, cut.truth_boxes) + ' found'};
    const boxes = cut.budget?.boxes ?? 5;
    const kept = cut.budget ? (version.families === 'all' ? cut.budget.by_family?.['speed'] ?? null
      : {truth: cut.budget.truth_boxes, found: cut.budget.found, recall: cut.budget.recall}) : null;
    const budget: Metric = kept
      ? {key: 'budget', label: 'Speed signs in ' + boxes + ' boxes', short: 'In ' + boxes + ' boxes', value: kept.recall,
        count: of(kept.found, kept.truth), detail: of(kept.found, kept.truth) + ' in the ' + boxes + ' best boxes of each frame'}
      : {key: 'budget', label: 'Speed signs in ' + boxes + ' boxes', short: 'In ' + boxes + ' boxes', value: null,
        count: 'not measured', detail: 'Its report did not keep the boxes of each frame'};
    const speed = version.families === 'all' ? cut.by_family?.['speed'] ?? null : null;
    if (speed) {
      return [{key: 'speed', label: 'Speed signs', value: speed.recall, count: of(speed.found, speed.truth),
        detail: of(speed.found, speed.truth) + ' found'}, every, budget];
    }
    if (version.families === 'all') return [every, budget];
    return [{...every, key: 'speed', label: 'Speed signs'},
      {key: 'every', label: 'Every sign', value: null, count: 'speed only', detail: 'Trained on speed signs only'}, budget];
  }
  if (version.role === 'classifier') {
    const test = version.test;
    const metrics: Metric[] = [];
    if (version.class_scheme === undefined) {
      if (test) metrics.push({key: 'every', label: 'Read right', value: test.accuracy, count: of(test.accuracy === null || test.samples === null
        ? null : Math.round(test.accuracy * test.samples), test.samples), detail: 'right answers on ' + thousands(test.samples ?? 0) + ' test crops'});
      return metrics;
    }
    if (test) {
      metrics.push({key: 'speed', label: 'Speed signs', value: test.speed_accuracy ?? null, count: of(test.speed_correct, test.speed_samples),
        detail: test.speed_samples ? of(test.speed_correct, test.speed_samples) + ' read right' : 'Not counted apart in this report'});
      const every = test.family_accuracy ?? test.accuracy;
      const right = every === null || test.samples === null ? null : Math.round(every * test.samples);
      metrics.push({key: 'every', label: 'Every sign', value: every, count: of(right, test.samples), detail: of(right, test.samples) + ' read right'});
    }
    const chain = version.chain;
    if (chain) {
      metrics.push({key: 'chain', label: 'With the cutter' + (chain.cutter_size ? ' ' + sizeText(chain.cutter_size) : ''), value: chain.recall,
        count: of(chain.correct, chain.truth), detail: chain.truth ? of(chain.correct, chain.truth) + ' speed signs, end to end' : 'speed signs, end to end'});
    }
    return metrics;
  }
  const another = (numbers: typeof version.validation): Metric['count'] =>
    numbers?.other_road_support ? of(numbers.other_road_found, numbers.other_road_support) : '';
  return [
    {key: 'validation', label: '“Another” found', value: version.validation?.other_road_recall ?? null, count: another(version.validation),
      detail: another(version.validation) + ' on validation'},
    {key: 'test', label: 'On the test', value: version.test?.other_road_support ? version.test.other_road_recall : null,
      count: version.test?.other_road_support ? another(version.test) : 'no case', detail: version.test?.other_road_support ? another(version.test) + ' on the test' : 'The test has no “Another” sign yet'},
  ];
}
