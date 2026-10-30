<script setup lang="ts">
// Overview, the home page, in the order you look at it: what is running now, one row each with Stop (a tap opens its
// sheet with the log), or the last job when it failed; the loop of the work (LoopRows.vue: Drives, Review, Train, Car,
// each with how much waits and its one button); the last jobs that ended (three, the others on a click). The server,
// its gauges and what is on the GPU are in the server's panel of the top bar (TopMachine.vue), with Update the site.
import {computed, ref} from 'vue';
import type {ActionName, ActionState, Job, PanelStatus} from '../common/panel-types';
import JobItem from './JobItem.vue';
import {terminalActivities} from './jobs';
import LoopRows from './LoopRows.vue';
import RunRow from './RunRow.vue';
import SectionHead from './SectionHead.vue';

const props = defineProps<{status: PanelStatus | null}>();
defineEmits<{run: [name: ActionName, state: ActionState]; stop: [job: Job]; details: [job: Job]}>();

/** Jobs that ended shown before "Show all" (the server sends the latest 12 only, panel.HISTORY). */
const FEW = 3;
const showAll = ref(false);
const jobs = computed(() => props.status?.jobs ?? []);
const running = computed(() => jobs.value.filter((job) => job.state === 'running'));
const activities = computed(() => terminalActivities(props.status));
/** The last job, on top only when it failed and nothing runs. */
const failed = computed(() => !running.value.length && jobs.value[0]?.state === 'failed' ? jobs.value[0] : null);
/** The jobs that ended, newest first; a failed one on top is not repeated. */
const ended = computed(() => jobs.value.filter((job) => job.state !== 'running' && job !== failed.value));
const shown = computed(() => showAll.value ? ended.value : ended.value.slice(0, FEW));
</script>

<template>
  <SectionHead tab="overview"/>

  <section v-if="running.length + activities.length || failed" class="card run-card" aria-labelledby="running-title" aria-live="polite">
    <h2 class="run-title" id="running-title">{{ running.length + activities.length ? 'Running' : 'Last job failed' }}</h2>
    <RunRow v-for="job in running" :key="job.id" :job="job" @open="$emit('details', $event)" @stop="$emit('stop', $event)"/>
    <RunRow v-for="activity in activities" :key="activity.pid + activity.label" :activity="activity"/>
    <RunRow v-if="failed" :job="failed" @open="$emit('details', $event)"/>
  </section>

  <LoopRows :status="status" @run="(name, state) => $emit('run', name, state)"/>

  <section class="card recent" aria-labelledby="recent-title">
    <h2 id="recent-title">Recent activity</h2>
    <ul class="list">
      <JobItem v-for="job in shown" :key="job.id" :job="job" @details="$emit('details', $event)" @stop="$emit('stop', $event)"/>
    </ul>
    <button v-if="ended.length > FEW" type="button" class="link-button more" @click="showAll = !showAll">
      {{ showAll ? 'Show fewer' : 'Show all ' + ended.length }}</button>
    <p v-if="status && ended.length === 0" class="empty">Nothing yet: the buttons of Data and Training start the work.</p>
  </section>
</template>
