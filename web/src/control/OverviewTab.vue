<script setup lang="ts">
// Overview, the home page: the next step, the three phases (the review with Resume review and the route
// list), what is running now (live, with Stop and the machine in small), or how the last job ended; the
// machine, the recent jobs, the site.
import {computed} from 'vue';
import type {ActionName, ActionState, Job, PanelStatus} from '../common/panel-types';
import ActionList from './ActionList.vue';
import FlowTiles from './FlowTiles.vue';
import JobItem from './JobItem.vue';
import JobProgress from './JobProgress.vue';
import {terminalActivities} from './jobs';
import MachineCards from './MachineCards.vue';
import NextStepCard from './NextStepCard.vue';
import {nextStep, WAITING} from './next-step';
import PrepRoutes from './PrepRoutes.vue';
import {reviewSummary} from './review-routes';
import type {ControlTab} from './tabs';
import TerminalActivity from './TerminalActivity.vue';

const props = defineProps<{status: PanelStatus | null; busy: ReadonlySet<ControlTab>}>();
defineEmits<{run: [name: ActionName, state: ActionState]; stop: [job: Job]; details: [job: Job]}>();

const SITE_ACTIONS: [ActionName, string][] = [
  ['update', 'Downloads the latest code (git pull), updates the libraries and restarts the site; only when nothing else is running.'],
];
const next = computed(() => props.status ? nextStep(props.status) : WAITING);
const reviewHref = computed(() => reviewSummary(props.status).resume?.href ?? '/routes');
const jobs = computed(() => props.status?.jobs ?? []);
const running = computed(() => jobs.value.filter((job) => job.state === 'running'));
const activities = computed(() => terminalActivities(props.status));
const last = computed(() => jobs.value[0] ?? null);
const gpuProcesses = computed(() => props.status?.system.gpu?.process_list ?? []);
</script>

<template>
  <header class="page-head">
    <h1 class="page-title">Overview</h1>
    <p class="page-sub">What the server is doing, how it is, and the next step.</p>
  </header>

  <NextStepCard :next="next" :actions="status?.actions ?? null" :review-href="reviewHref" @run="(name, state) => $emit('run', name, state)"/>
  <FlowTiles :status="status" :attention="next.attention" :busy="busy"/>

  <section class="live-section" aria-labelledby="running-title">
    <h2 class="section-title" id="running-title">Running<span v-if="running.length + activities.length" class="count-badge">{{ running.length + activities.length }}</span></h2>
    <div v-if="running.length + activities.length" class="jp-list" aria-live="polite">
      <JobProgress v-for="job in running" :key="job.id" :job="job" :system="status?.system" @details="$emit('details', $event)" @stop="$emit('stop', $event)">
        <PrepRoutes v-if="job.action === 'prepare' && status?.preparation" :routes="status.preparation.working"/>
      </JobProgress>
      <TerminalActivity v-for="activity in activities" :key="activity.pid + activity.label" :activity="activity"
                        :preparation="status?.preparation ?? null" :system="status?.system"/>
    </div>
    <div v-else class="card idle" :class="{waiting: status === null}">
      <div class="idle-icon" aria-hidden="true"><svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="8.5" fill="none" stroke="currentColor"
        stroke-width="1.8"/><path d="M8.3 12.3l2.5 2.5 4.9-5.2" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"
        stroke-linejoin="round"/></svg></div>
      <div class="idle-text">
        <strong>{{ status === null ? 'Loading…' : 'Nothing running' }}</strong>
        <span v-if="status">Neither from the site nor from a terminal. The buttons of Prepare and Train start jobs in the background.</span>
      </div>
    </div>
    <JobProgress v-if="!running.length && last" :job="last" heading="Last job" @details="$emit('details', $event)" @stop="$emit('stop', $event)"/>
  </section>

  <section aria-labelledby="machine-title">
    <h2 class="section-title" id="machine-title">The server</h2>
    <MachineCards v-if="status" :system="status.system"/>
  </section>

  <div class="grid2">
    <div class="card">
      <h2>Recent jobs</h2>
      <ul class="list">
        <JobItem v-for="job in jobs" :key="job.id" :job="job" @details="$emit('details', $event)" @stop="$emit('stop', $event)"/>
      </ul>
      <p v-if="status && jobs.length === 0" class="empty">No jobs yet: the buttons of Prepare and Train start them in the background.</p>
    </div>
    <div class="side">
      <div class="card">
        <h2>Site</h2>
        <ActionList :items="SITE_ACTIONS" :actions="status?.actions ?? null" @run="(name, state) => $emit('run', name, state)"/>
      </div>
      <div class="card">
        <h2>Processes on the GPU</h2>
        <ul v-if="gpuProcesses.length" class="gpu-procs">
          <li v-for="(process, index) in gpuProcesses" :key="index">
            <span class="mono">{{ process.pid ?? '?' }}</span><span class="grow">{{ process.name }}</span>
            <span class="num">{{ process.memory_mb === null ? '–' : Math.round(process.memory_mb) + ' MiB' }}</span>
          </li>
        </ul>
        <p v-else class="empty">No process on the GPU.</p>
        <p class="note">Inside the LXC container nvidia-smi often cannot see the process names: the known jobs are under “Running”.</p>
      </div>
    </div>
  </div>
</template>
