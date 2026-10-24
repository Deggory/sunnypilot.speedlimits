<script setup lang="ts">
// Prepare: from the drives on the comma to frames with the proposed signs. The two actions, the download or
// preparation running (or the last one), the routes.
import {computed} from 'vue';
import {plural} from '../common/format';
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
  ['fetch', 'Copies the new drives from the comma into the video folder, segment by segment; the routes already prepared stay as they are.'],
  ['prepare', 'For every new route it links the videos, extracts the frames and has the model propose the signs: then the route is ready to review.'],
];
const COUNTS: [RouteState, string][] = [['collegare', 'to link'], ['estrarre', 'to extract'], ['preannotare', 'to pre-annotate']];

const job = computed(() => jobOf(props.status, PREPARE_ACTIONS));
const activities = computed(() => terminalActivities(props.status, ['fetch', 'prepare']));
const counts = computed(() => COUNTS.map(([state, label]) => ({state, label,
  count: (props.status?.routes ?? []).filter((row) => row.state === state).length})));
const total = computed(() => counts.value.reduce((sum, item) => sum + item.count, 0));
</script>

<template>
  <header class="page-head">
    <h1 class="page-title">Prepare</h1>
    <p class="page-sub">From the drives of the comma to frames with the proposed signs, all automatic.</p>
  </header>

  <section class="card tab-card" aria-labelledby="prepare-title">
    <div class="tab-card-head">
      <div>
        <h2 id="prepare-title">New routes</h2>
        <div class="figure" id="prepare-big">{{ status === null ? '–' : total ? plural(total, 'route') + ' to prepare' : 'All prepared' }}</div>
        <p class="figure-note">{{ status === null ? '' : total ? 'Videos arrived from the comma, not frames yet.' : 'No video waiting: for more routes, download from the comma.' }}</p>
      </div>
      <ul v-if="total" class="chips">
        <li v-for="item in counts.filter((entry) => entry.count)" :key="item.state"><strong>{{ item.count }}</strong> {{ item.label }}</li>
      </ul>
    </div>
    <ActionList :items="ACTIONS" :actions="status?.actions ?? null" @run="(name, state) => $emit('run', name, state)"/>
  </section>

  <section v-if="job || activities.length" class="live-section" :class="{now: job?.state === 'running' || activities.length > 0}" aria-labelledby="prepare-job-title">
    <h2 class="section-title" id="prepare-job-title">{{ job?.state === 'running' || activities.length ? 'Running' : 'Last job' }}</h2>
    <div class="jp-list" aria-live="polite">
      <JobProgress v-if="job" :job="job" :heading="job.state !== 'running' && activities.length ? 'Last job' : undefined" :system="status?.system" @details="$emit('details', $event)" @stop="$emit('stop', $event)">
        <PrepRoutes v-if="job.action === 'prepare' && job.state === 'running' && status?.preparation" :routes="status.preparation.working"/>
      </JobProgress>
      <TerminalActivity v-for="activity in activities" :key="activity.pid + activity.label" :activity="activity" :preparation="status?.preparation ?? null" :system="status?.system"/>
    </div>
  </section>

  <RouteTable :routes="status?.routes ?? null"/>
</template>
