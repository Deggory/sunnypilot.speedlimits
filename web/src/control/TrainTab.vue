<script setup lang="ts">
// Training: from the reviews to a model, as an app on the phone. One line says what the networks learn from (objects,
// frames, drives; the reviews not imported yet), a link to Data › Training dataset, where they are imported. Then one network at a time, chosen
// in a switcher with a dot on the one training: the cutter (scripts/run_detector_sizes.sh, its input size and the
// signs it finds), the classifier with "not a sign" (scripts/run_classifier.sh), the decider (in preparation:
// experiment B), the whole pipeline (helpers/3_train.sh, with where it starts). Each has the same place for its work:
// the Train button, or its training running (or failed) with its steps, then its curves and its latest version
// (GET /api/panel/models, a link to its versions in Models). The page does not move while work runs.
import {computed, ref} from 'vue';
import {isoWhen, plural, thousands} from '../common/format';
import type {ActionName, ActionOptions, ActionState, Activity, CutterFamilies, Job, JobProgress as Progress, ModelRole, ModelVersion,
  PanelStatus, TrainingStart} from '../common/panel-types';
import ActionList from './ActionList.vue';
import JobProgress from './JobProgress.vue';
import {CHARTED_ACTIONS, JOB_STATE, jobOf, PIPELINE_ACTIONS, startedWhen, terminalActivities} from './jobs';
import {pct} from './evaluate';
import {sizeText, useModels} from './models';
import Segmented from './Segmented.vue';
import TerminalActivity from './TerminalActivity.vue';
import TrainingCharts from './TrainingCharts.vue';
import TrainStart from './TrainStart.vue';
import SectionHead from './SectionHead.vue';

const props = defineProps<{status: PanelStatus | null}>();
const emit = defineEmits<{run: [name: ActionName, state: ActionState, start?: TrainingStart, options?: ActionOptions]; stop: [job: Job];
  details: [job: Job]}>();

/** The page as an app: one line of data, then one network at a time (Cutter, Classifier, Decider, the whole Pipeline),
 * whose Train button becomes its progress, in the same place, while it trains. */
type Section = 'cutter' | 'classifier' | 'decider' | 'pipeline';
const SECTIONS: {key: Section; label: string}[] = [{key: 'cutter', label: 'Cutter'}, {key: 'classifier', label: 'Classifier'},
  {key: 'decider', label: 'Decider'}, {key: 'pipeline', label: 'Pipeline'}];
function sectionOf(action: ActionName): Section | null {
  if (action === 'train_cutter') return 'cutter';
  if (action === 'train_classifier') return 'classifier';
  if (action === 'train_decider') return 'decider';
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
/** Importing the reviews (Data › Training dataset): said in the line of data. */
const importing = computed(() => jobOf(props.status, ['dataset'])?.state === 'running');

/** The networks by role: every version trained here, newest first (the Models tab's data). */
const {payload: models} = useModels();
const roleOf = (role: ModelRole) => models.value?.roles.find((item) => item.role === role) ?? null;
/** The latest version of a role (newest weights), or null. */
const latest = (role: ModelRole): ModelVersion | null => roleOf(role)?.versions[0] ?? null;
const latestCutter = computed(() => latest('cutter'));
const latestClassifier = computed(() => latest('classifier'));
const latestDecider = computed(() => latest('decider'));
const roleLabel = (role: ModelRole): string => roleOf(role)?.label ?? role;
const sectionLatest = computed(() => section.value === 'cutter' ? latestCutter.value : section.value === 'classifier'
  ? latestClassifier.value : section.value === 'decider' ? latestDecider.value : null);
/** The headline of a version in one line: its main number and what it counts (the numbers of ModelItem). */
function headlineOf(version: ModelVersion): {value: string; text: string} | null {
  if (version.role === 'cutter') {
    const cut = version.test ?? version.validation ?? null;
    if (cut === null) return null;
    const speed = version.families === 'all' ? cut.by_family?.['speed'] ?? null : null;
    if (speed) return {value: pct(speed.recall), text: 'speed signs found, ' + thousands(speed.found) + ' of ' + thousands(speed.truth ?? 0)
      + ' · every sign ' + pct(cut.recall)};
    return {value: pct(cut.recall), text: 'signs found'};
  }
  if (version.role === 'classifier') {
    if (version.chain) return {value: pct(version.chain.recall), text: 'speed signs found and read right'};
    if (version.test) return {value: pct(version.test.accuracy), text: 'test signs read right'};
    return null;
  }
  return version.validation ? {value: pct(version.validation.other_road_recall), text: '“Another” found on validation'} : null;
}
const latestHeadline = computed(() => sectionLatest.value ? headlineOf(sectionLatest.value) : null);
/** How many versions the section's network has: Models lists them, to compare and to choose. */
const sectionVersions = computed(() => section.value === 'cutter' || section.value === 'classifier' || section.value === 'decider'
  ? roleOf(section.value)?.versions.length ?? 0 : 0);

/** The cutter: its input size and the signs it learns to find (status.cutter_options; these when the server is older). */
const sizeOptions = computed(() => (props.status?.cutter_options?.sizes ?? ['640x360', '960x544', '1344x760'])
  .map((size) => ({key: size, label: sizeText(size)})));
const FAMILY_OPTIONS: {key: CutterFamilies; label: string}[] = [{key: 'all', label: 'Every sign'}, {key: 'speed', label: 'Speed signs only'}];
const pickedSize = ref<string | null>(null);
const pickedFamilies = ref<CutterFamilies | null>(null);
const cutterSize = computed({
  get: (): string => pickedSize.value ?? props.status?.cutter_options?.size ?? '1344x760',
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
/** The decider's training, in its card: experiment B, its three variants. */
const DECIDER_ACTIONS: [ActionName, string][] = [
  ['train_decider', 'Three deciders: the crop alone, with position and speed, with the scene. About 15 minutes on the GPU.'],
];
/** The classifier: our signs only, or with the signs of the Mapillary dataset in train (status.classifier_options). */
const mapillaryState = computed(() => props.status?.classifier_options?.mapillary ?? null);
const MAPILLARY_OPTIONS: {key: 'off' | 'on'; label: string}[] = [{key: 'off', label: 'Our signs'}, {key: 'on', label: 'With Mapillary'}];
const pickedMapillary = ref<'off' | 'on' | null>(null);
const mapillary = computed({
  get: (): 'off' | 'on' => mapillaryState.value?.available ? pickedMapillary.value ?? 'off' : 'off',
  set: (value: 'off' | 'on') => { pickedMapillary.value = value; },
});
const CLASSIFIER_ACTIONS = computed((): [ActionName, string][] => [
  ['train_classifier', 'A new classifier, taught on the signs and on the cutter’s mistakes'
    + (mapillary.value === 'on' ? ', with ' + thousands(mapillaryState.value?.crops ?? 0) + ' signs of Mapillary' : '')
    + '. Takes a few hours on the GPU.'],
]);
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
    // What the dataset file holds (status.dataset), else what the routes say they put there.
    objects: dataset ? dataset.objects : routes.reduce((total, row) => total + row.objects, 0),
    frames: dataset ? dataset.frames : null,
    inDataset: dataset ? dataset.routes : routes.filter((row) => row.objects > 0).length,
  };
});
</script>

<template>
  <SectionHead tab="train"/>
  <!-- Which network: first, under the title, as the segments of an iOS page. -->
  <Segmented v-model="section" class="train-sections" :options="SECTIONS" :marked="runningSections" label="Network"/>


  <!-- What the networks learn from, in one line; the reviews not imported yet lead to Data › Training dataset, where they are
       imported. -->
  <a class="card data-strip" href="/dataset" aria-label="What the networks learn from: open the dataset">
    <span class="data-texts"><span class="data-line"><strong>{{ numbers ? thousands(numbers.objects) : '–' }}</strong> objects on
      <strong>{{ numbers?.frames === null || !numbers ? '–' : thousands(numbers.frames) }}</strong> frames of
      {{ numbers ? plural(numbers.inDataset, 'drive') : '–' }}</span>
    <span v-if="importing" class="data-new">Importing your reviews…</span>
    <span v-else-if="numbers?.fresh" class="data-new"><strong>{{ plural(numbers.fresh, 'review') }}</strong> not imported</span>
    <span v-else-if="numbers" class="data-new muted">Dataset is up to date</span>
    <span v-if="numbers && numbers.reviewed < numbers.needed" class="data-new warn">Training needs {{ numbers.needed }} reviewed
      drives (now {{ numbers.reviewed }})</span></span>
    <span class="data-chev" aria-hidden="true">›</span>
  </a>

  <section class="card role-card train-section" :id="'train-' + section" :data-role="section" aria-live="polite">
    <p v-if="section === 'cutter'" class="role-what">Finds every box that could be a sign. It must miss nothing: the classifier drops its false boxes.</p>
    <p v-else-if="section === 'classifier'" class="role-what">Says what each box of the cutter is: a speed value, end of limit, another sign and its kind, or not a sign.</p>
    <p v-else-if="section === 'decider'" class="role-what">Says whom each sign applies to: our road, another road, or abstain.</p>
    <p v-else class="role-what">The detector and the reader together, from the reviews to a package: the way every model was trained so far.</p>

    <!-- The slot of the work: the training running (or failed) in place of its button, first: what runs and what to start come before the versions. -->
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
      <template v-else-if="section === 'classifier'">
        <div v-if="mapillaryState?.available" class="seg-row">
          <Segmented v-model="mapillary" :options="MAPILLARY_OPTIONS" label="Signs the classifier learns from"/>
        </div>
        <ActionList :items="CLASSIFIER_ACTIONS" :actions="status?.actions ?? null"
                    @run="(name, state) => emit('run', name, state, undefined, {mapillary})"/>
        <p v-if="mapillaryState?.available" class="latest-none" id="train-mapillary-note">Mapillary: {{ thousands(mapillaryState.crops) }}
          street-photo signs of {{ plural(mapillaryState.labels, 'kind') }}, added to train only: every measure stays on your drives.</p>
        <p v-else-if="mapillaryState" class="latest-none" id="train-mapillary-note">Mapillary signs: not prepared yet
          (helpers/mapillary.sh).</p>
      </template>
      <template v-else-if="section === 'decider'">
        <ActionList :items="DECIDER_ACTIONS" :actions="status?.actions ?? null" @run="(name, state) => emit('run', name, state)"/>
        <p class="latest-none" id="train-decider-note">The one in use (Models) sets OURS or another on the speed signs Our networks propose.</p>
      </template>
      <TrainStart v-else :starts="status ? status.training_starts ?? [] : null" :state="status?.actions.train ?? null"
                  @start="(start) => status && emit('run', 'train', status.actions.train, start)"/>
      <p v-if="job && !bigJob && !sectionActivities.length" class="last-line">Last: {{ job.label }}, {{ startedWhen(job) }} ·
        {{ job.state === 'interrupted' ? 'stopped' : JOB_STATE[job.state][0] }} ·
        <button type="button" class="link-button" @click="emit('details', job)">Details</button></p>
    </div>

    <!-- The latest version in one row, a link to Models: its versions there, to compare and to choose (the choice
         is made only there). -->
    <a v-if="sectionLatest" class="latest-row latest-link" :href="'/models#role-' + sectionLatest.role" id="latest-version">
      <span class="latest-what"><span class="latest-kicker">{{ section === 'decider' ? 'Latest prototype' : 'Latest' }} ·
        {{ sectionLatest.trained_at ? isoWhen(sectionLatest.trained_at) : sectionLatest.run }}<span
        v-if="sectionLatest.in_use" class="latest-use">in use</span><span v-else-if="!sectionLatest.finished" class="latest-wait">not finished</span></span>
        <span class="latest-name">{{ sectionLatest.label }}</span>
        <span class="latest-text">{{ latestHeadline ? latestHeadline.text + ' · ' : '' }}{{ plural(sectionVersions, 'version', 'versions') }} in Models</span></span>
      <strong v-if="latestHeadline" class="latest-value">{{ latestHeadline.value }}</strong>
      <span class="latest-chev" aria-hidden="true">›</span>
    </a>
    <p v-else-if="section === 'cutter' || section === 'classifier'" class="latest-none">{{ models === null ? 'Loading…'
      : 'None trained yet: the ' + (section === 'cutter' ? 'detector' : 'reader') + ' of a whole pipeline plays this role.' }}</p>

    <TrainingCharts v-if="charted?.running" :progress="charted.progress" :running="true"/>
    <details v-else-if="charted" class="latest-row curves-row">
      <summary><span class="latest-what"><span class="latest-name">Curves of the last training</span>
        <span class="latest-text">recall and loss, epoch by epoch</span></span><span class="latest-chev" aria-hidden="true">›</span></summary>
      <TrainingCharts :progress="charted.progress" :running="false"/>
    </details>

  </section>
</template>
