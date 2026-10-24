<script setup lang="ts">
// Prepara: from the drives on the comma to frames with the proposed signs. The two actions, the download or
// preparation running (or the last one), the routes.
import {computed} from 'vue';
import type {ActionName, ActionState, Job, PanelStatus, RouteState} from '../common/panel-types';
import ActionList from './ActionList.vue';
import JobProgress from './JobProgress.vue';
import {jobOf, PREPARE_ACTIONS, terminalActivities} from './jobs';
import PrepRoutes from './PrepRoutes.vue';
import RouteTable from './RouteTable.vue';
import TerminalActivity from './TerminalActivity.vue';

const props = defineProps<{status: PanelStatus | null}>();
defineEmits<{run: [name: ActionName, state: ActionState]; stop: [job: Job]; details: [job: Job]}>();

const ACTIONS: [ActionName, string][] = [
  ['fetch', 'Copia dal comma i giri nuovi nella cartella dei video, segmento per segmento; le route già preparate restano come sono.'],
  ['prepare', 'Per ogni route nuova collega i video, estrae i fotogrammi e fa proporre i cartelli al modello: poi la route è pronta da revisionare.'],
];
const COUNTS: [RouteState, string][] = [['collegare', 'da collegare'], ['estrarre', 'da estrarre'], ['preannotare', 'da pre-annotare']];

const job = computed(() => jobOf(props.status, PREPARE_ACTIONS));
const activities = computed(() => terminalActivities(props.status, ['fetch', 'prepare']));
const counts = computed(() => COUNTS.map(([state, label]) => ({state, label,
  count: (props.status?.routes ?? []).filter((row) => row.state === state).length})));
const total = computed(() => counts.value.reduce((sum, item) => sum + item.count, 0));
</script>

<template>
  <header class="page-head">
    <h1 class="page-title">Prepara</h1>
    <p class="page-sub">Dai giri del comma ai fotogrammi con i cartelli proposti, tutto automatico.</p>
  </header>

  <section class="card tab-card" aria-labelledby="prepare-title">
    <div class="tab-card-head">
      <div>
        <h2 id="prepare-title">Route nuove</h2>
        <div class="figure" id="prepare-big">{{ status === null ? '–' : total ? total + ' route da preparare' : 'Tutto preparato' }}</div>
        <p class="figure-note">{{ status === null ? '' : total ? 'Video arrivati dal comma, non ancora fotogrammi.' : 'Nessun video in attesa: per altre route, scarica dal comma.' }}</p>
      </div>
      <ul v-if="total" class="chips">
        <li v-for="item in counts.filter((entry) => entry.count)" :key="item.state"><strong>{{ item.count }}</strong> {{ item.label }}</li>
      </ul>
    </div>
    <ActionList :items="ACTIONS" :actions="status?.actions ?? null" @run="(name, state) => $emit('run', name, state)"/>
  </section>

  <section v-if="job || activities.length" class="live-section" :class="{now: job?.state === 'running' || activities.length > 0}" aria-labelledby="prepare-job-title">
    <h2 class="section-title" id="prepare-job-title">{{ job?.state === 'running' || activities.length ? 'In corso' : 'Ultimo lavoro' }}</h2>
    <div class="jp-list" aria-live="polite">
      <JobProgress v-if="job" :job="job" :heading="job.state !== 'running' && activities.length ? 'Ultimo lavoro' : undefined" @details="$emit('details', $event)" @stop="$emit('stop', $event)">
        <PrepRoutes v-if="job.action === 'prepare' && job.state === 'running' && status?.preparation" :routes="status.preparation.working"/>
      </JobProgress>
      <TerminalActivity v-for="activity in activities" :key="activity.label" :activity="activity" :preparation="status?.preparation ?? null"/>
    </div>
  </section>

  <RouteTable :routes="status?.routes ?? null"/>
</template>
