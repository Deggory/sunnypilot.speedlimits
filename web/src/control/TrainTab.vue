<script setup lang="ts">
// Train: from the reviews to a model, as an app on the phone. One line says what the networks learn from (boxes, frames,
// drives; the reviews not added yet, with Add: helpers/3_train.sh --dataset-only). Then one network at a time, chosen
// in a switcher with a dot on the one training: the cutter (scripts/run_detector_sizes.sh, its input size and the
// signs it finds), the classifier with "not a sign" (scripts/run_classifier.sh), the decider (in preparation:
// experiment B), the whole pipeline (helpers/3_train.sh, with where it starts). Each has the same place for its work:
// the Train button, or its training running (or failed) with its steps, then its curves and its latest version
// (GET /api/panel/models, as the Models tab). The page does not move while work runs. What the dataset holds is below.
import {computed, ref} from 'vue';
import {plural, thousands} from '../common/format';
import type {ActionName, ActionState, Activity, CutterFamilies, CutterOptions, Job, JobProgress as Progress, ModelRole, ModelVersion,
  PanelStatus, TrainingStart} from '../common/panel-types';
import ActionList from './ActionList.vue';
import DatasetSection from './DatasetSection.vue';
import JobProgress from './JobProgress.vue';
import {CHARTED_ACTIONS, JOB_STATE, jobOf, PIPELINE_ACTIONS, startedWhen, terminalActivities} from './jobs';
import ModelItem from './ModelItem.vue';
import {putInUse, sizeText, useModels} from './models';
import Segmented from './Segmented.vue';
import TerminalActivity from './TerminalActivity.vue';
import TrainingCharts from './TrainingCharts.vue';
import TrainStart from './TrainStart.vue';

const props = defineProps<{status: PanelStatus | null}>();
const emit = defineEmits<{run: [name: ActionName, state: ActionState, start?: TrainingStart, options?: CutterOptions]; stop: [job: Job];
  details: [job: Job]}>();

/** The page as an app: one line of data, then one network at a time (Cutter, Classifier, Decider, the whole Pipeline),
 * whose Train button becomes its progress, in the same place, while it trains. */
type Section = 'cutter' | 'classifier' | 'decider' | 'pipeline';
const SECTIONS: {key: Section; label: string}[] = [{key: 'cutter', label: 'Cutter'}, {key: 'classifier', label: 'Classifier'},
  {key: 'decider', label: 'Decider'}, {key: 'pipeline', label: 'Pipeline'}];
function sectionOf(action: ActionName): Section | null {
  if (action === 'train_cutter') return 'cutter';
  if (action === 'train_classifier') return 'classifier';
  return PIPELINE_ACTIONS.includes(action) ? 'pipeline' : null;
}
/** The newest job of a section (running or the last one). */
function jobFor(section: Section): Job | null {
  return props.status?.jobs.find((item) => sectionOf(item.action) === section) ?? null;
}
/** The work started from a terminal, by section (its label names the network; else the pipeline). */
function activitiesFor(section: Section): Activity[] {
  return activities.value.filter((activity) => {
    const named: Section = activity.label.includes('cutter') ? 'cutter' : activity.label.includes('classifier') ? 'classifier' : 'pipeline';
    return named === section;
  });
}
const runningSections = computed(() => SECTIONS.map((item) => item.key)
  .filter((key) => jobFor(key)?.state === 'running' || activitiesFor(key).length > 0));
const picked = ref<Section | null>(null);
/** The section shown: the one picked, else the one training, else the cutter. */
const section = computed({
  get: (): Section => picked.value ?? runningSections.value[0] ?? 'cutter',
  set: (value: Section) => { picked.value = value; },
});
const job = computed(() => jobFor(section.value));
/** The job of the section in its slot: running or failed; one done or stopped by hand is a line. */
const bigJob = computed(() => job.value && (job.value.state === 'running' || job.value.state === 'failed') ? job.value : null);
/** Adding the reviews (Import reviews): shown in the line of data. */
const importJob = computed(() => jobOf(props.status, ['dataset']));
const importing = computed(() => importJob.value?.state === 'running');

/** The networks by role: every version trained here, newest first (the Models tab's data). */
const {payload: models, refresh: refreshModels} = useModels();
const roleOf = (role: ModelRole) => models.value?.roles.find((item) => item.role === role) ?? null;
/** The latest version of a role (newest weights), or null. */
const latest = (role: ModelRole): ModelVersion | null => roleOf(role)?.versions[0] ?? null;
const latestCutter = computed(() => latest('cutter'));
const latestClassifier = computed(() => latest('classifier'));
const latestDecider = computed(() => latest('decider'));
const roleLabel = (role: ModelRole): string => roleOf(role)?.label ?? role;
const sectionLatest = computed(() => section.value === 'cutter' ? latestCutter.value : section.value === 'classifier'
  ? latestClassifier.value : section.value === 'decider' ? latestDecider.value : null);
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
  ['train_cutter', 'A new cutter at ' + sizeText(cutterSize.value) + ' for '
    + (cutterFamilies.value === 'all' ? 'every sign' : 'speed signs only') + '. Takes a few hours on the GPU.'],
]);
/** The classifier's training, in its card. */
const CLASSIFIER_ACTIONS: [ActionName, string][] = [
  ['train_classifier', 'A new classifier, taught on the signs and on the cutter’s mistakes. Takes a few hours on the GPU.'],
];
const activities = computed(() => terminalActivities(props.status, ['train']));
const sectionActivities = computed(() => activitiesFor(section.value));
/** The curves of a training: while it runs (a placeholder before the first epoch), or of the last one if it has any. */
const charted = computed((): {progress: Progress; running: boolean} | null => {
  // A training started from a terminal: the server reads its progress from data/logs/train-<run>.log.
  const terminal = sectionActivities.value.find((activity) => activity.progress);
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
  <h1 class="tab-title">Train</h1>

  <!-- What the networks learn from, in one line; the reviews not in yet, with their button. -->
  <section class="card data-strip" aria-label="What the networks learn from">
    <p class="data-line"><strong>{{ numbers ? thousands(numbers.objects) : '–' }}</strong> boxes on
      <strong>{{ numbers?.frames === null || !numbers ? '–' : thousands(numbers.frames) }}</strong> frames of
      {{ numbers ? plural(numbers.inDataset, 'drive') : '–' }}<a v-if="status?.dataset" class="figure-link" href="#dataset">Details ›</a></p>
    <p v-if="importing" class="data-new">Adding your reviews… <strong>{{ importJob?.progress.percent === null || importJob?.progress.percent === undefined
      ? '' : Math.floor(importJob.progress.percent) + '%' }}</strong></p>
    <p v-else-if="numbers?.fresh" class="data-new"><span><strong>{{ thousands(numbers.fresh) }}</strong> new
      {{ numbers.fresh === 1 ? 'review' : 'reviews' }} to add</span>
      <button type="button" class="btn small primary" :disabled="!status?.actions.dataset?.allowed"
              :title="status?.actions.dataset?.allowed ? 'A few minutes' : status?.actions.dataset?.reason"
              @click="status && emit('run', 'dataset', status.actions.dataset)">Add</button></p>
    <p v-else-if="numbers" class="data-new muted">All your reviews are in.</p>
    <p v-if="numbers && numbers.reviewed < numbers.needed" class="data-new warn">Training needs {{ numbers.needed }} reviewed drives
      (now {{ numbers.reviewed }}): some are kept apart to test on roads the network never saw.</p>
  </section>

  <Segmented v-model="section" class="train-sections" :options="SECTIONS" :marked="runningSections" label="Network"/>

  <section class="card role-card train-section" :id="'train-' + section" :data-role="section" aria-live="polite">
    <p v-if="section === 'cutter'" class="role-what">Finds every box that could be a sign. It must miss nothing: the classifier drops its false boxes.</p>
    <p v-else-if="section === 'classifier'" class="role-what">Says what each box of the cutter is: a speed value, end of limit, another sign and its kind, or not a sign.</p>
    <p v-else-if="section === 'decider'" class="role-what">Says whom each sign applies to: our road, another road, or abstain.</p>
    <p v-else class="role-what">The detector and the reader together, from the reviews to a package: the way every model was trained so far.</p>

    <!-- The slot of the work: the training running (or failed) in place of its button. -->
    <div class="train-slot">
      <template v-if="bigJob || sectionActivities.length">
        <JobProgress v-if="bigJob" :job="bigJob" @details="emit('details', $event)" @stop="emit('stop', $event)"/>
        <TerminalActivity v-for="activity in sectionActivities" :key="activity.pid + activity.label" :activity="activity" :preparation="null"/>
      </template>
      <template v-else-if="section === 'cutter'">
        <div class="seg-row">
          <Segmented v-model="cutterSize" :options="sizeOptions" label="Input size of the cutter"/>
          <Segmented v-model="cutterFamilies" :options="FAMILY_OPTIONS" label="Signs the cutter finds"/>
        </div>
        <ActionList :items="CUTTER_ACTIONS" :actions="status?.actions ?? null"
                    @run="(name, state) => emit('run', name, state, undefined, {size: cutterSize, families: cutterFamilies})"/>
      </template>
      <ActionList v-else-if="section === 'classifier'" :items="CLASSIFIER_ACTIONS" :actions="status?.actions ?? null"
                  @run="(name, state) => emit('run', name, state)"/>
      <div v-else-if="section === 'decider'" class="role-empty" id="train-decider-note"><strong>In preparation: experiment B is the prototype.</strong>
        <span>No training from here yet: its variants are trained by scripts/run_experiment_b.sh.</span></div>
      <TrainStart v-else :starts="status ? status.training_starts ?? [] : null" :state="status?.actions.train ?? null"
                  @start="(start) => status && emit('run', 'train', status.actions.train, start)"/>
      <p v-if="job && !bigJob && !sectionActivities.length" class="last-line">Last: {{ job.label }}, {{ startedWhen(job) }} ·
        {{ job.state === 'interrupted' ? 'stopped' : JOB_STATE[job.state][0] }} ·
        <button type="button" class="link-button" @click="emit('details', job)">Details</button></p>
    </div>

    <TrainingCharts v-if="charted" :progress="charted.progress" :running="charted.running"/>

    <template v-if="section !== 'pipeline'">
      <h3 class="ds-title">{{ section === 'decider' ? 'Latest prototype' : 'Latest version' }}</h3>
      <ul v-if="sectionLatest" class="model-list">
        <ModelItem :version="sectionLatest" :role-label="roleLabel(section)" :score="models?.cutter_score_threshold ?? 0.1"
                   :read-only="readOnly" :busy="busy" @use="choose"/>
      </ul>
      <div v-else-if="section !== 'decider'" class="role-empty"><strong>{{ models === null ? 'Loading…' : 'None trained yet.' }}</strong>
        <span v-if="models">The {{ section === 'cutter' ? 'detector' : 'reader' }} of a whole pipeline plays this role too.</span></div>
    </template>
  </section>

  <DatasetSection :dataset="status?.dataset ?? null" :loading="status === null"/>
</template>
