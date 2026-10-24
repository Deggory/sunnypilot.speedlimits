<script setup lang="ts">
// Train: from the reviews to a model. The numbers (reviewed routes against what the split needs, confirmed
// frames still to import, what the dataset holds), the action (Train a model with the choice of where it
// starts: it imports the reviews and fixes the split itself, helpers/3_train.sh steps a-b), the import or training running (or the last one) with all its steps and its
// curves, and what the dataset holds.
import {computed} from 'vue';
import {plural, thousands} from '../common/format';
import type {ActionName, ActionState, Job, JobProgress as Progress, PanelStatus, TrainingStart} from '../common/panel-types';
import DatasetSection from './DatasetSection.vue';
import JobProgress from './JobProgress.vue';
import {jobOf, terminalActivities, TRAIN_ACTIONS} from './jobs';
import TerminalActivity from './TerminalActivity.vue';
import TrainingCharts from './TrainingCharts.vue';
import TrainStart from './TrainStart.vue';

const props = defineProps<{status: PanelStatus | null}>();
const emit = defineEmits<{run: [name: ActionName, state: ActionState, start?: TrainingStart]; stop: [job: Job]; details: [job: Job]}>();

const job = computed(() => jobOf(props.status, TRAIN_ACTIONS));
const activities = computed(() => terminalActivities(props.status, ['train', 'other']));
/** The curves of a training: while it runs (a placeholder before the first epoch), or of the last one if it has any. */
const charted = computed((): {progress: Progress; running: boolean} | null => {
  // A training started from a terminal: the server reads its progress from data/logs/train-<run>.log.
  const terminal = activities.value.find((activity) => activity.group === 'train' && activity.progress);
  if (terminal?.progress && !(job.value?.state === 'running')) return {progress: terminal.progress, running: true};
  const value = job.value;
  if (value === null || (value.action !== 'train' && value.action !== 'train_continue')) return null;
  const curves = value.progress.curves ?? {};
  const shown = value.state === 'running' || Boolean(curves.detector?.points.length || curves.reader?.points.length);
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
      <h2>New since the last training</h2>
      <div class="figure" id="train-fresh">{{ numbers ? thousands(numbers.fresh) : '–' }}<span class="figure-unit">{{ numbers?.fresh === 1 ? 'frame' : 'frames' }}</span></div>
      <p class="figure-note">{{ numbers === null ? '' : numbers.fresh ? 'Frames confirmed or corrected after the last import, in ' + plural(numbers.freshRoutes, 'route') + '. The next training takes them in by itself.'
        : 'Every confirmed frame is already in the dataset.' }}</p>
    </div>
    <div class="card number">
      <h2>What the model learns from</h2>
      <div class="figure" id="train-objects">{{ numbers ? thousands(numbers.objects) : '–' }}<span class="figure-unit">boxes</span></div>
      <p class="figure-note">{{ numbers === null ? '' : 'Signs and “not a sign” boxes on ' + (numbers.frames === null ? '' : thousands(numbers.frames) + ' frames of ')
        + plural(numbers.inDataset, 'route') + ', from the last import.' }} <a v-if="status?.dataset" class="figure-link" href="#dataset">Details ↓</a></p>
    </div>
  </section>

  <section class="card tab-card" aria-labelledby="train-actions-title">
    <h2 id="train-actions-title">Actions</h2>
    <TrainStart :starts="status ? status.training_starts ?? [] : null" :state="status?.actions.train ?? null"
                @start="(start) => status && emit('run', 'train', status.actions.train, start)"/>
  </section>

  <section v-if="job || activities.length" class="live-section" :class="{now: job?.state === 'running' || activities.length > 0}" aria-labelledby="train-job-title">
    <h2 class="section-title" id="train-job-title">{{ job?.state === 'running' || activities.length ? 'Running' : 'Last job' }}</h2>
    <div class="jp-list" aria-live="polite">
      <JobProgress v-if="job" :job="job" :heading="job.state !== 'running' && activities.length ? 'Last job' : undefined" :system="status?.system"
                   @details="emit('details', $event)" @stop="emit('stop', $event)"/>
      <TrainingCharts v-if="charted" :progress="charted.progress" :running="charted.running"/>
      <TerminalActivity v-for="activity in activities" :key="activity.pid + activity.label" :activity="activity" :preparation="null" :system="status?.system"/>
    </div>
  </section>

  <DatasetSection :dataset="status?.dataset ?? null" :loading="status === null"/>
</template>
