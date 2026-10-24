<script setup lang="ts">
// Train: from the reviews to a model. The numbers (reviewed routes against what the split needs, confirmed
// frames still to import, what the dataset holds); then the networks of the pipeline by role (docs/NEURAL_NETWORK.md),
// each with what it does, its own training and the latest version trained here with its headline numbers (GET
// /api/panel/models, as the Models tab): the cutter (scripts/run_detector_sizes.sh, one input size and the signs it
// finds), the classifier with “not a sign” (scripts/run_classifier.sh, on the false boxes of the cutter in use), the
// decider (in preparation: experiment B is its prototype); the whole pipeline with the choice of where it starts
// (helpers/3_train.sh: import, split, detector, reader, evaluation, export); the job running (or the last one) with all
// its steps and its curves, and what the dataset holds.
import {computed, ref} from 'vue';
import {plural, thousands} from '../common/format';
import type {ActionName, ActionState, CutterFamilies, CutterOptions, Job, JobProgress as Progress, ModelRole, ModelVersion, PanelStatus,
  TrainingStart} from '../common/panel-types';
import ActionList from './ActionList.vue';
import DatasetSection from './DatasetSection.vue';
import JobProgress from './JobProgress.vue';
import {CHARTED_ACTIONS, jobOf, terminalActivities, TRAIN_ACTIONS} from './jobs';
import ModelItem from './ModelItem.vue';
import {putInUse, sizeText, useModels} from './models';
import Segmented from './Segmented.vue';
import TerminalActivity from './TerminalActivity.vue';
import TrainingCharts from './TrainingCharts.vue';
import TrainStart from './TrainStart.vue';

const props = defineProps<{status: PanelStatus | null}>();
const emit = defineEmits<{run: [name: ActionName, state: ActionState, start?: TrainingStart, options?: CutterOptions]; stop: [job: Job];
  details: [job: Job]}>();

const job = computed(() => jobOf(props.status, TRAIN_ACTIONS));

/** The networks by role: every version trained here, newest first (the Models tab's data). */
const {payload: models, refresh: refreshModels} = useModels();
const roleOf = (role: ModelRole) => models.value?.roles.find((item) => item.role === role) ?? null;
/** The latest version of a role (newest weights), or null. */
const latest = (role: ModelRole): ModelVersion | null => roleOf(role)?.versions[0] ?? null;
const latestCutter = computed(() => latest('cutter'));
const latestClassifier = computed(() => latest('classifier'));
const latestDecider = computed(() => latest('decider'));
const roleLabel = (role: ModelRole): string => roleOf(role)?.label ?? role;
const readOnly = computed(() => (props.status !== null && !props.status.actions_enabled) || Boolean(models.value?.active_problem));
const busy = ref<string | null>(null);
async function choose(version: ModelVersion): Promise<void> {
  if (busy.value !== null) return;
  busy.value = version.id;
  try {
    if (await putInUse(version, roleLabel(version.role))) await refreshModels();
  } finally {
    busy.value = null;
  }
}

/** The cutter: its input size and the signs it learns to find (status.cutter_options; these when the server is older). */
const sizeOptions = computed(() => (props.status?.cutter_options?.sizes ?? ['640x360', '960x544', '1344x760'])
  .map((size) => ({key: size, label: sizeText(size)})));
const FAMILY_OPTIONS: {key: CutterFamilies; label: string}[] = [{key: 'all', label: 'Every sign'}, {key: 'speed', label: 'Speed signs only'}];
const pickedSize = ref<string | null>(null);
const pickedFamilies = ref<CutterFamilies | null>(null);
const cutterSize = computed({
  get: (): string => pickedSize.value ?? props.status?.cutter_options?.size ?? '960x544',
  set: (value: string) => { pickedSize.value = value; },
});
const cutterFamilies = computed({
  get: (): CutterFamilies => pickedFamilies.value ?? props.status?.cutter_options?.family ?? 'all',
  set: (value: CutterFamilies) => { pickedFamilies.value = value; },
});
const CUTTER_ACTIONS = computed((): [ActionName, string][] => [
  ['train_cutter', 'Trains a detector at ' + sizeText(cutterSize.value) + ' that finds '
    + (cutterFamilies.value === 'all' ? 'the speed signs and the other signs' : 'the speed signs only')
    + ' on the GPU, then measures it on test and validation by the size of the sign. A new run; it takes hours.'],
]);
/** Reviews into the training data, under the count of frames not yet in it. */
const IMPORT_ACTIONS: [ActionName, string][] = [
  ['dataset', 'Adds the frames confirmed since the last import to the training data and splits it again by place. '
    + 'No GPU, a few minutes.'],
];
/** The classifier's training, in its card. */
const CLASSIFIER_ACTIONS: [ActionName, string][] = [
  ['train_classifier', 'Mines the false boxes of the cutter in use, trains the classifier with “not a sign” and the kinds of '
    + 'other signs on the GPU, then measures it alone and the whole chain (cutter, then classifier). A new run; it takes hours.'],
];
const activities = computed(() => terminalActivities(props.status, ['train', 'other']));
/** The curves of a training: while it runs (a placeholder before the first epoch), or of the last one if it has any. */
const charted = computed((): {progress: Progress; running: boolean} | null => {
  // A training started from a terminal: the server reads its progress from data/logs/train-<run>.log.
  const terminal = activities.value.find((activity) => activity.group === 'train' && activity.progress);
  if (terminal?.progress && !(job.value?.state === 'running')) return {progress: terminal.progress, running: true};
  const value = job.value;
  if (value === null || !CHARTED_ACTIONS.includes(value.action)) return null;
  const curves = value.progress.curves ?? {};
  const shown = value.state === 'running'
    || Boolean(curves.detector?.points.length || curves.reader?.points.length || curves.classifier?.points.length);
  return shown ? {progress: value.progress, running: value.state === 'running'} : null;
});
const numbers = computed(() => {
  const status = props.status;
  if (status === null) return null;
  const routes = status.routes;
  const dataset = status.dataset;
  return {
    reviewed: status.reviewed_routes,
    needed: status.min_routes,
    fresh: routes.reduce((total, row) => total + row.new_reviews, 0),
    freshRoutes: routes.filter((row) => row.new_reviews > 0).length,
    // What the dataset file holds (status.dataset), else what the routes say they put there.
    objects: dataset ? dataset.objects : routes.reduce((total, row) => total + row.objects, 0),
    frames: dataset ? dataset.frames : null,
    inDataset: dataset ? dataset.routes : routes.filter((row) => row.objects > 0).length,
  };
});
</script>

<template>
  <header class="page-head">
    <h1 class="page-title">Train</h1>
    <p class="page-sub">From the reviews to the dataset, then to a model on the GPU.</p>
  </header>

  <section class="numbers" aria-label="The numbers">
    <div class="card number">
      <h2>Routes with confirmed frames</h2>
      <div class="figure" id="train-big">{{ numbers ? (numbers.reviewed >= numbers.needed ? numbers.reviewed : numbers.reviewed + ' of ' + numbers.needed) : '–' }}</div>
      <div class="goal" aria-hidden="true"><span v-for="index in numbers?.needed ?? 0" :key="index" :class="{on: index <= (numbers?.reviewed ?? 0)}"></span></div>
      <p class="figure-note">{{ numbers === null ? '' : numbers.reviewed >= numbers.needed ? 'Enough to train: whole routes are kept apart to check the model on roads it never saw (at least ' + numbers.needed + ').'
        : 'Training needs at least ' + numbers.needed + ' routes: whole routes are kept apart to check the model on roads it never saw.' }}</p>
    </div>
    <div class="card number">
      <h2>Not yet in the training data</h2>
      <div class="figure" id="train-fresh">{{ numbers ? thousands(numbers.fresh) : '–' }}<span class="figure-unit">{{ numbers?.fresh === 1 ? 'frame' : 'frames' }}</span></div>
      <p class="figure-note">{{ numbers === null ? '' : numbers.fresh ? 'Confirmed or corrected after the last import, in ' + plural(numbers.freshRoutes, 'route') + '. Only Import reviews and the whole pipeline add them: the cutter and the classifier train on the data of the last import.'
        : 'Every confirmed frame is already in the training data.' }}</p>
      <ActionList v-if="numbers?.fresh" :items="IMPORT_ACTIONS" :actions="status?.actions ?? null"
                  @run="(name, state) => emit('run', name, state)"/>
    </div>
    <div class="card number">
      <h2>What the model learns from</h2>
      <div class="figure" id="train-objects">{{ numbers ? thousands(numbers.objects) : '–' }}<span class="figure-unit">boxes</span></div>
      <p class="figure-note">{{ numbers === null ? '' : 'Signs and “not a sign” boxes on ' + (numbers.frames === null ? '' : thousands(numbers.frames) + ' frames of ')
        + plural(numbers.inDataset, 'route') + ', from the last import.' }} <a v-if="status?.dataset" class="figure-link" href="#dataset">Details ↓</a></p>
    </div>
  </section>

  <div class="roles" id="train-roles">
    <section class="card role-card" id="train-cutter" data-role="cutter" aria-labelledby="train-cutter-title">
      <div class="ds-head"><h2 id="train-cutter-title">Cutter</h2></div>
      <p class="role-what">Finds every box that could be a sign. It must miss nothing: the classifier drops its false boxes.</p>
      <div class="seg-row">
        <Segmented v-model="cutterSize" :options="sizeOptions" label="Input size of the cutter"/>
        <Segmented v-model="cutterFamilies" :options="FAMILY_OPTIONS" label="Signs the cutter finds"/>
      </div>
      <ActionList :items="CUTTER_ACTIONS" :actions="status?.actions ?? null"
                  @run="(name, state) => emit('run', name, state, undefined, {size: cutterSize, families: cutterFamilies})"/>
      <h3 class="ds-title">Latest version</h3>
      <ul v-if="latestCutter" class="model-list">
        <ModelItem :version="latestCutter" :role-label="roleLabel('cutter')" :score="models?.cutter_score_threshold ?? 0.1"
                   :read-only="readOnly" :busy="busy" @use="choose"/>
      </ul>
      <div v-else class="role-empty"><strong>{{ models === null ? 'Loading…' : 'No cutter trained yet.' }}</strong>
        <span v-if="models">The detector of a whole pipeline plays this role too.</span></div>
    </section>

    <section class="card role-card" id="train-classifier" data-role="classifier" aria-labelledby="train-classifier-title">
      <div class="ds-head"><h2 id="train-classifier-title">Classifier</h2></div>
      <p class="role-what">Says what each box of the cutter is: a speed value, end of limit, another sign and its kind, or not a sign.</p>
      <ActionList :items="CLASSIFIER_ACTIONS" :actions="status?.actions ?? null" @run="(name, state) => emit('run', name, state)"/>
      <h3 class="ds-title">Latest version</h3>
      <ul v-if="latestClassifier" class="model-list">
        <ModelItem :version="latestClassifier" :role-label="roleLabel('classifier')" :score="models?.cutter_score_threshold ?? 0.1"
                   :read-only="readOnly" :busy="busy" @use="choose"/>
      </ul>
      <div v-else class="role-empty"><strong>{{ models === null ? 'Loading…' : 'No classifier trained yet.' }}</strong>
        <span v-if="models">The reader of a whole pipeline plays this role too.</span></div>
    </section>

    <section class="card role-card" id="train-decider" data-role="decider" aria-labelledby="train-decider-title">
      <div class="ds-head"><h2 id="train-decider-title">Decider</h2></div>
      <p class="role-what">Says whom each sign applies to: our road, another road, or abstain.</p>
      <div class="role-empty" id="train-decider-note"><strong>In preparation: experiment B is the prototype.</strong>
        <span>No training from here yet: its variants are trained by scripts/run_experiment_b.sh.</span></div>
      <template v-if="latestDecider">
        <h3 class="ds-title">Latest prototype</h3>
        <ul class="model-list">
          <ModelItem :version="latestDecider" :role-label="roleLabel('decider')" :score="models?.cutter_score_threshold ?? 0.1"
                     :read-only="readOnly" :busy="busy" @use="choose"/>
        </ul>
      </template>
    </section>
  </div>

  <section class="card tab-card" id="train-pipeline" aria-labelledby="train-pipeline-title">
    <h2 id="train-pipeline-title">Whole pipeline</h2>
    <p class="role-what">The detector and the reader together, from the reviews to a package: the way every model was trained so far.</p>
    <TrainStart :starts="status ? status.training_starts ?? [] : null" :state="status?.actions.train ?? null"
                @start="(start) => status && emit('run', 'train', status.actions.train, start)"/>
  </section>

  <section v-if="job || activities.length" class="live-section" :class="{now: job?.state === 'running' || activities.length > 0}" aria-labelledby="train-job-title">
    <h2 class="section-title" id="train-job-title">{{ job?.state === 'running' || activities.length ? 'Running' : 'Last job' }}</h2>
    <div class="jp-list" aria-live="polite">
      <JobProgress v-if="job" :job="job" :heading="job.state !== 'running' && activities.length ? 'Last job' : undefined"
                   @details="emit('details', $event)" @stop="emit('stop', $event)"/>
      <TrainingCharts v-if="charted" :progress="charted.progress" :running="charted.running"/>
      <TerminalActivity v-for="activity in activities" :key="activity.pid + activity.label" :activity="activity" :preparation="null"/>
    </div>
  </section>

  <DatasetSection :dataset="status?.dataset ?? null" :loading="status === null"/>
</template>
