<script setup lang="ts">
// Prepare: from the drives on the comma to frames with the proposed signs. The two actions, the download or
// preparation running (or the last one), the routes.
import {computed, ref, watch} from 'vue';
import {plural} from '../common/format';
import type {ActionName, ActionOptions, ActionState, Job, PanelStatus, Proposer, RouteState} from '../common/panel-types';
import Segmented from './Segmented.vue';
import ActionList from './ActionList.vue';
import JobProgress from './JobProgress.vue';
import {JOB_STATE, jobOf, PREPARE_ACTIONS, startedWhen, terminalActivities} from './jobs';
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
  vietnam: 'A ready-made model trained on Vietnamese signs: it boxes and reads them by itself. It finds about a third of the speed signs.',
  ours: 'Our cutter draws the boxes, our classifier reads them; both trained on your reviews. They find 3 speed signs out of 4.',
  qwen: 'A model that looks at the whole picture, boxes every sign and reads it in one go. A few seconds per picture; not measured yet.',
};
const stored = localStorage.getItem('nnslr-proposer');
const proposer = ref<Proposer>(stored === 'ours' || stored === 'qwen' ? stored : 'vietnam');
/** Why the chosen proposer cannot run here, if it cannot. */
const unavailable = computed(() => proposer.value === 'qwen' ? props.status?.proposers?.qwen ?? null : null);
watch(proposer, (value) => localStorage.setItem('nnslr-proposer', value));
/** Routes with proposals and nothing confirmed yet: those "Propose again" can redo. */
const unstarted = computed(() => (props.status?.routes ?? []).filter((row) => row.state === 'revisionare' && row.confirmed === 0).length);
const PROPOSE_ACTIONS = computed((): [ActionName, string][] => unstarted.value ? [['prepare_again',
  'New proposals for the ' + plural(unstarted.value, 'drive') + ' nobody has started to review.']] : []);

/** Step 1, the drives from the comma; step 2, the frames and the proposals, with the proposer chosen above them. */
const FETCH_ACTIONS: [ActionName, string][] = [
  ['fetch', 'Copies the new drives from the comma.'],
];
const PREPARE_ITEMS: [ActionName, string][] = [
  ['prepare', 'Takes the pictures out of the new drives and proposes the signs, ready to review.'],
];
const COUNTS: [RouteState, string][] = [['collegare', 'to link'], ['estrarre', 'to extract'], ['preannotare', 'to pre-annotate']];

const job = computed(() => jobOf(props.status, PREPARE_ACTIONS));
const activities = computed(() => terminalActivities(props.status, ['fetch', 'prepare']));
const counts = computed(() => COUNTS.map(([state, label]) => ({state, label,
  count: (props.status?.routes ?? []).filter((row) => row.state === state).length})));
const total = computed(() => counts.value.reduce((sum, item) => sum + item.count, 0));
const running = computed(() => job.value?.state === 'running' || activities.value.length > 0);
/** The job shown as a card: running, or failed (its error must be read); one done or stopped by hand is a line. */
const bigJob = computed(() => job.value && (job.value.state === 'running' || job.value.state === 'failed') ? job.value : null);
const WAITING: RouteState[] = ['collegare', 'estrarre', 'preannotare'];
const notReady = computed(() => (props.status?.routes ?? []).filter((row) => WAITING.includes(row.state)));
const ready = computed(() => (props.status?.routes ?? []).filter((row) => !WAITING.includes(row.state)).length);
</script>

<template>
  <h1 class="sr-only">Prepare</h1>

  <section class="card tab-card" aria-labelledby="prepare-title">
    <h2 class="sr-only" id="prepare-title">New routes</h2>
    <div class="prep-row">
      <h3 class="prep-step"><span>1</span> Get the drives</h3>
      <ActionList class="compact" :items="FETCH_ACTIONS" :actions="status?.actions ?? null" @run="(name, state) => emit('run', name, state)"/>
    </div>
    <p class="figure-note prep-state" id="prepare-big">{{ status === null ? '' : total
      ? plural(total, 'drive') + ' copied from the comma, waiting to be prepared.'
      : 'Every drive copied from the comma is prepared.' }}</p>
    <ul v-if="total" class="chips">
      <li v-for="item in counts.filter((entry) => entry.count)" :key="item.state"><strong>{{ item.count }}</strong> {{ item.label }}</li>
    </ul>
    <h3 class="prep-step"><span>2</span> Find the signs</h3>
    <p class="figure-note prep-state">Draws a box around every sign in every picture and reads it, for you to confirm in Review.</p>
    <div class="proposer">
      <span class="proposer-label">Found by</span>
      <Segmented v-model="proposer" :options="PROPOSERS" label="Who proposes the signs"/>
      <p class="figure-note">{{ PROPOSER_TEXT[proposer] }}</p>
      <p v-if="unavailable" class="figure-note proposer-missing">{{ unavailable }}</p>
    </div>
    <ActionList :items="[...PREPARE_ITEMS, ...PROPOSE_ACTIONS]" :actions="status?.actions ?? null"
                @run="(name, state) => emit('run', name, state, undefined, {proposer})"/>
  </section>

  <!-- Only what is happening now, or a job that failed: a finished job is one line. -->
  <section v-if="bigJob || activities.length" class="live-section" :class="{now: running}" aria-labelledby="prepare-job-title">
    <h2 class="section-title" id="prepare-job-title">{{ running ? 'Running' : 'Last job failed' }}</h2>
    <div class="jp-list" aria-live="polite">
      <JobProgress v-if="bigJob" :job="bigJob" @details="$emit('details', $event)" @stop="$emit('stop', $event)">
        <PrepRoutes v-if="bigJob.action === 'prepare' && bigJob.state === 'running' && status?.preparation" :routes="status.preparation.working"/>
      </JobProgress>
      <TerminalActivity v-for="activity in activities" :key="activity.pid + activity.label" :activity="activity" :preparation="status?.preparation ?? null"/>
    </div>
  </section>
  <p v-else-if="job" class="last-line">Last: {{ job.label }}, {{ startedWhen(job) }} · {{ job.state === 'interrupted' ? 'stopped' : JOB_STATE[job.state][0] }} ·
    <button type="button" class="link-button" @click="$emit('details', job)">Details</button></p>

  <!-- The routes still on their way; the ready ones are on the Review tab, as drives. -->
  <RouteTable v-if="notReady.length" :routes="notReady"/>
  <p v-if="ready" class="last-line"><a href="/routes">{{ plural(ready, 'drive') }} ready to review ›</a></p>
</template>
