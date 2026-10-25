<script setup lang="ts">
// Prepare: from the drives on the comma to frames with the proposed signs. The two actions, the download or
// preparation running (or the last one), the routes.
import {computed, ref, watch} from 'vue';
import {plural} from '../common/format';
import type {ActionName, ActionOptions, ActionState, Job, PanelStatus, Proposer, RouteState} from '../common/panel-types';
import Segmented from './Segmented.vue';
import ActionList from './ActionList.vue';
import JobProgress from './JobProgress.vue';
import {jobOf, PREPARE_ACTIONS, terminalActivities} from './jobs';
import PrepRoutes from './PrepRoutes.vue';
import RouteTable from './RouteTable.vue';
import TerminalActivity from './TerminalActivity.vue';

const props = defineProps<{status: PanelStatus | null}>();
const emit = defineEmits<{run: [name: ActionName, state: ActionState, start?: undefined, options?: ActionOptions]; stop: [job: Job];
  details: [job: Job]}>();

/** Who proposes the signs, kept in this browser. */
const PROPOSERS: {key: Proposer; label: string}[] = [{key: 'vietnam', label: 'Vietnamese YOLO'}, {key: 'ours', label: 'Our networks'},
  {key: 'qwen', label: 'Qwen3-VL'}];
const PROPOSER_TEXT: Record<Proposer, string> = {
  vietnam: 'A ready-made model from Hugging Face, trained on Vietnamese signs, on the CPU. It proposed about 31% of the signs reviewers kept.',
  ours: 'Our cutter and classifier, trained on your reviews, on the GPU. On the test roads they find 81% of the speed signs; the signs they still miss stay missed in the proposals.',
  qwen: 'Qwen3-VL-8B (Alibaba), a model that sees images and is asked for every sign of the frame, on the GPU: a few seconds per frame, '
    + 'no score on its boxes, never measured on our signs yet.',
};
const stored = localStorage.getItem('nnslr-proposer');
const proposer = ref<Proposer>(stored === 'ours' || stored === 'qwen' ? stored : 'vietnam');
/** Why the chosen proposer cannot run here, if it cannot. */
const unavailable = computed(() => proposer.value === 'qwen' ? props.status?.proposers?.qwen ?? null : null);
watch(proposer, (value) => localStorage.setItem('nnslr-proposer', value));
/** Routes with proposals and nothing confirmed yet: those "Propose again" can redo. */
const unstarted = computed(() => (props.status?.routes ?? []).filter((row) => row.state === 'revisionare' && row.confirmed === 0).length);
const PROPOSE_ACTIONS = computed((): [ActionName, string][] => unstarted.value ? [['prepare_again',
  'Gives new proposals to the ' + plural(unstarted.value, 'route') + ' nobody has started reviewing, with the proposer chosen above. Routes in review stay as they are.']] : []);

/** Step 1, the drives from the comma; step 2, the frames and the proposals, with the proposer chosen above them. */
const FETCH_ACTIONS: [ActionName, string][] = [
  ['fetch', 'Copies the new drives from the comma into the video folder, segment by segment; the routes already prepared stay as they are.'],
];
const PREPARE_ITEMS: [ActionName, string][] = [
  ['prepare', 'For every new route it links the videos, extracts the frames and has the proposer chosen above propose the signs: then the route is ready to review.'],
];
const COUNTS: [RouteState, string][] = [['collegare', 'to link'], ['estrarre', 'to extract'], ['preannotare', 'to pre-annotate']];

const job = computed(() => jobOf(props.status, PREPARE_ACTIONS));
const activities = computed(() => terminalActivities(props.status, ['fetch', 'prepare']));
const counts = computed(() => COUNTS.map(([state, label]) => ({state, label,
  count: (props.status?.routes ?? []).filter((row) => row.state === state).length})));
const total = computed(() => counts.value.reduce((sum, item) => sum + item.count, 0));
</script>

<template>
  <h1 class="sr-only">Prepare</h1>

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
    <h3 class="prep-step"><span>1</span> Get the drives</h3>
    <ActionList :items="FETCH_ACTIONS" :actions="status?.actions ?? null" @run="(name, state) => emit('run', name, state)"/>
    <h3 class="prep-step"><span>2</span> Propose the signs</h3>
    <div class="proposer">
      <span class="proposer-label">Signs proposed by</span>
      <Segmented v-model="proposer" :options="PROPOSERS" label="Who proposes the signs"/>
      <p class="figure-note">{{ PROPOSER_TEXT[proposer] }}</p>
      <p v-if="unavailable" class="figure-note proposer-missing">{{ unavailable }}</p>
    </div>
    <ActionList :items="[...PREPARE_ITEMS, ...PROPOSE_ACTIONS]" :actions="status?.actions ?? null"
                @run="(name, state) => emit('run', name, state, undefined, {proposer})"/>
  </section>

  <section v-if="job || activities.length" class="live-section" :class="{now: job?.state === 'running' || activities.length > 0}" aria-labelledby="prepare-job-title">
    <h2 class="section-title" id="prepare-job-title">{{ job?.state === 'running' || activities.length ? 'Running' : 'Last job' }}</h2>
    <div class="jp-list" aria-live="polite">
      <JobProgress v-if="job" :job="job" :heading="job.state !== 'running' && activities.length ? 'Last job' : undefined" @details="$emit('details', $event)" @stop="$emit('stop', $event)">
        <PrepRoutes v-if="job.action === 'prepare' && job.state === 'running' && status?.preparation" :routes="status.preparation.working"/>
      </JobProgress>
      <TerminalActivity v-for="activity in activities" :key="activity.pid + activity.label" :activity="activity" :preparation="status?.preparation ?? null"/>
    </div>
  </section>

  <RouteTable :routes="status?.routes ?? null"/>
</template>
