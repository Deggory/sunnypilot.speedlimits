<script setup lang="ts">
// Overview, the home page (the server is the pill of the top bar, TopMachine.vue): the next step, the three phases
// (the review with Resume review and the route list), what is running now (live, with Stop) or a last job that
// failed; the latest jobs (all of them on a click), the processes on the GPU when there are. Update the site is in
// the server's panel of the top bar.
import {computed, ref} from 'vue';
import type {ActionName, ActionState, Job, PanelStatus} from '../common/panel-types';
import FlowTiles from './FlowTiles.vue';
import JobItem from './JobItem.vue';
import JobProgress from './JobProgress.vue';
import {gpuProcessNames, terminalActivities} from './jobs';
import NextStepCard from './NextStepCard.vue';
import {nextStep, WAITING} from './next-step';
import PrepRoutes from './PrepRoutes.vue';
import {reviewSummary} from './review-routes';
import type {ControlTab} from './tabs';
import TerminalActivity from './TerminalActivity.vue';
import SectionHead from './SectionHead.vue';

const props = defineProps<{status: PanelStatus | null; busy: ReadonlySet<ControlTab>}>();
defineEmits<{run: [name: ActionName, state: ActionState]; stop: [job: Job]; details: [job: Job]}>();

/** Jobs shown before "Show the last 12" (the server sends the latest 12 only, panel.HISTORY). */
const FEW = 4;
const showAll = ref(false);
const next = computed(() => props.status ? nextStep(props.status) : WAITING);
const reviewHref = computed(() => reviewSummary(props.status).resume?.href ?? '/routes');
const jobs = computed(() => props.status?.jobs ?? []);
const running = computed(() => jobs.value.filter((job) => job.state === 'running'));
const activities = computed(() => terminalActivities(props.status));
/** The last job, as a card only when it failed: one done or stopped by hand is in the list below. */
const failed = computed(() => jobs.value[0]?.state === 'failed' ? jobs.value[0] : null);
const shown = computed(() => showAll.value ? jobs.value : jobs.value.slice(0, FEW));
const gpuProcesses = computed(() => props.status?.system.gpu?.process_list ?? []);
/** What each of them is, when the server can tell (jobs.ts gpuProcessNames). */
const gpuNames = computed(() => gpuProcessNames(props.status));
</script>

<template>
  <SectionHead tab="overview"/>
  <NextStepCard :next="next" :actions="status?.actions ?? null" :review-href="reviewHref" @run="(name, state) => $emit('run', name, state)"/>
  <FlowTiles :status="status" :attention="next.attention" :busy="busy"/>

  <section v-if="running.length + activities.length || failed" class="live-section" aria-labelledby="running-title">
    <h2 class="section-title" id="running-title">{{ running.length + activities.length ? 'Running' : 'Last job failed' }}</h2>
    <div class="jp-list" aria-live="polite">
      <JobProgress v-for="job in running" :key="job.id" :job="job" @details="$emit('details', $event)" @stop="$emit('stop', $event)">
        <PrepRoutes v-if="(job.action === 'prepare' || job.action === 'prepare_again') && status?.preparation" :routes="status.preparation.working"/>
      </JobProgress>
      <TerminalActivity v-for="activity in activities" :key="activity.pid + activity.label" :activity="activity"
                        :preparation="status?.preparation ?? null"/>
      <JobProgress v-if="!running.length && !activities.length && failed" :job="failed" @details="$emit('details', $event)"
                   @stop="$emit('stop', $event)"/>
    </div>
  </section>

  <div :class="{grid2: gpuProcesses.length}">
    <div class="card">
      <h2>Latest jobs</h2>
      <ul class="list">
        <JobItem v-for="job in shown" :key="job.id" :job="job" @details="$emit('details', $event)" @stop="$emit('stop', $event)"/>
      </ul>
      <button v-if="jobs.length > FEW" type="button" class="link-button more" @click="showAll = !showAll">
        {{ showAll ? 'Show fewer' : 'Show the last ' + jobs.length }}</button>
      <p v-if="status && jobs.length === 0" class="empty">No jobs yet: the buttons of Prepare and Train start them.</p>
    </div>
    <div v-if="gpuProcesses.length" class="card">
      <h2>On the GPU</h2>
      <ul class="gpu-procs">
        <li v-for="(process, index) in gpuProcesses" :key="index">
          <span class="grow">{{ gpuNames[index] }}</span>
          <span class="num">{{ process.memory_mb === null ? '–' : Math.round(process.memory_mb) + ' MiB' }}</span>
        </li>
      </ul>
    </div>
  </div>
</template>
