<script setup lang="ts">
// Allena: from the reviews to a model. The numbers (reviewed routes against what the split needs, confirmed
// frames still to import, what the dataset holds), the three actions, the import or training running (or
// the last one) with all its steps.
import {computed} from 'vue';
import {thousands} from '../common/format';
import type {ActionName, ActionState, Job, JobProgress as Progress, PanelStatus} from '../common/panel-types';
import ActionList from './ActionList.vue';
import JobProgress from './JobProgress.vue';
import {jobOf, terminalActivities, TRAIN_ACTIONS} from './jobs';
import TerminalActivity from './TerminalActivity.vue';
import TrainingCharts from './TrainingCharts.vue';

const props = defineProps<{status: PanelStatus | null}>();
defineEmits<{run: [name: ActionName, state: ActionState]; stop: [job: Job]; details: [job: Job]}>();

const ACTIONS: [ActionName, string][] = [
  ['dataset', 'Porta nel dataset i fotogrammi confermati e fissa lo split train/validation/test, senza GPU. Pochi minuti.'],
  ['train', 'Dataset, poi detector e reader sulla GPU partendo da una rete pre-allenata su ImageNet; li valuta e prepara il pacchetto. Ore.'],
  ['train_continue', 'Riparte dall’ultimo modello pubblicato su Hugging Face come stable: serve averne pubblicato uno.'],
];

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
  return {
    reviewed: status.reviewed_routes,
    needed: status.min_routes,
    fresh: routes.reduce((total, row) => total + row.new_reviews, 0),
    freshRoutes: routes.filter((row) => row.new_reviews > 0).length,
    objects: routes.reduce((total, row) => total + row.objects, 0),
    inDataset: routes.filter((row) => row.objects > 0).length,
  };
});
</script>

<template>
  <header class="page-head">
    <h1 class="page-title">Allena</h1>
    <p class="page-sub">Dalle revisioni al dataset, poi al modello sulla GPU.</p>
  </header>

  <section class="numbers" aria-label="I numeri">
    <div class="card number">
      <h2>Route revisionate</h2>
      <div class="figure" id="train-big">{{ numbers ? (numbers.reviewed >= numbers.needed ? numbers.reviewed : numbers.reviewed + ' di ' + numbers.needed) : '–' }}</div>
      <div class="goal" aria-hidden="true"><span v-for="index in numbers?.needed ?? 0" :key="index" :class="{on: index <= (numbers?.reviewed ?? 0)}"></span></div>
      <p class="figure-note">{{ numbers === null ? '' : numbers.reviewed >= numbers.needed ? 'Ne bastano ' + numbers.needed + ' per lo split: si può allenare.'
        : 'Lo split train/validation/test ne vuole almeno ' + numbers.needed + ' indipendenti.' }}</p>
    </div>
    <div class="card number">
      <h2>Da importare</h2>
      <div class="figure" id="train-fresh">{{ numbers ? thousands(numbers.fresh) : '–' }}<span class="figure-unit">fotogrammi</span></div>
      <p class="figure-note">{{ numbers === null ? '' : numbers.fresh ? 'Confermati dopo l’ultimo import, in ' + numbers.freshRoutes + ' route.'
        : 'Ogni conferma è già nel dataset.' }}</p>
    </div>
    <div class="card number">
      <h2>Nel dataset</h2>
      <div class="figure">{{ numbers ? thousands(numbers.objects) : '–' }}<span class="figure-unit">oggetti</span></div>
      <p class="figure-note">{{ numbers === null ? '' : 'Da ' + numbers.inDataset + ' route.' }}</p>
    </div>
  </section>

  <section class="card tab-card" aria-labelledby="train-actions-title">
    <h2 id="train-actions-title">Azioni</h2>
    <ActionList :items="ACTIONS" :actions="status?.actions ?? null" @run="(name, state) => $emit('run', name, state)"/>
  </section>

  <section v-if="job || activities.length" class="live-section" :class="{now: job?.state === 'running' || activities.length > 0}" aria-labelledby="train-job-title">
    <h2 class="section-title" id="train-job-title">{{ job?.state === 'running' || activities.length ? 'In corso' : 'Ultimo lavoro' }}</h2>
    <div class="jp-list" aria-live="polite">
      <JobProgress v-if="job" :job="job" :heading="job.state !== 'running' && activities.length ? 'Ultimo lavoro' : undefined"
                   @details="$emit('details', $event)" @stop="$emit('stop', $event)"/>
      <TrainingCharts v-if="charted" :progress="charted.progress" :running="charted.running"/>
      <TerminalActivity v-for="activity in activities" :key="activity.label" :activity="activity" :preparation="null"/>
    </div>
  </section>
</template>
