<script setup lang="ts">
// Overview, Prepare, Train, Evaluate, Models: one app, the tab shown follows the address (tabs.ts). One status for all
// tabs (/api/panel/status every 4 s, which also feeds the machine's history), the actions, and the sheet with the
// details of a job. Evaluate reads the training runs itself (/api/panel/runs), Models the networks by role
// (/api/panel/models).
import {computed, provide, ref, watch} from 'vue';
import type {ActionName, Job} from '../common/panel-types';
import ToastHost from '../components/ToastHost.vue';
import {usePanelStatus} from '../composables/panel-status';
import {PREPARE_ACTIONS, TRAIN_ACTIONS} from './jobs';
import LogSheet from './LogSheet.vue';
import {recordMachine} from './machine-history';
import EvaluateTab from './EvaluateTab.vue';
import ModelsTab from './ModelsTab.vue';
import OverviewTab from './OverviewTab.vue';
import PrepareTab from './PrepareTab.vue';
import {currentTab, markBusy, useTabRouting, type ControlTab} from './tabs';
import TrainTab from './TrainTab.vue';
import {RESUME_JOB, usePanelActions} from './use-actions';

useTabRouting();
const {status, refresh} = usePanelStatus((value) => recordMachine(value.system));
const {run, stop, resume} = usePanelActions(refresh);
// "Resume" on the card of a stopped training, wherever the card is shown.
provide(RESUME_JOB, resume);

/** The job in the sheet, by id: the sheet shows its latest copy from the status. */
const shownId = ref<string | null>(null);
const shownJob = computed(() => status.value?.jobs.find((job) => job.id === shownId.value) ?? null);
function details(job: Job): void {
  shownId.value = job.id;
}

/** The measures shown on Evaluate. */
const EVALUATE_ACTIONS: ActionName[] = ['measure_proposer', 'measure_ai_reviewer'];
/** The tabs with work running: a dot on their link in the top bar and on the tiles of the Overview. */
const busy = computed((): ReadonlySet<ControlTab> => {
  const tabs = new Set<ControlTab>();
  const value = status.value;
  if (value === null) return tabs;
  // A job lights the tab it belongs to (a measure is Evaluate's even if it uses the GPU as a training does); work
  // started from a terminal is known only by its group.
  for (const job of value.jobs.filter((item) => item.state === 'running')) {
    if (EVALUATE_ACTIONS.includes(job.action)) tabs.add('evaluate');
    else if (PREPARE_ACTIONS.includes(job.action)) tabs.add('prepare');
    else if (TRAIN_ACTIONS.includes(job.action)) tabs.add('train');
  }
  for (const activity of value.activities) {
    if (activity.group === 'fetch' || activity.group === 'prepare') tabs.add('prepare');
    else if (activity.group === 'train') tabs.add('train');
  }
  return tabs;
});
watch(busy, markBusy);
</script>

<template>
  <main class="page" :data-tab="currentTab">
    <div v-if="status && !status.actions_enabled" class="banner" id="readonly">Read-only: the buttons are turned off on this server.</div>
    <OverviewTab v-if="currentTab === 'overview'" :status="status" :busy="busy" @run="run" @stop="stop" @details="details"/>
    <PrepareTab v-else-if="currentTab === 'prepare'" :status="status" @run="run" @stop="stop" @details="details"/>
    <TrainTab v-else-if="currentTab === 'train'" :status="status" @run="run" @stop="stop" @details="details"/>
    <EvaluateTab v-else-if="currentTab === 'evaluate'" :status="status" @run="run"/>
    <ModelsTab v-else :status="status"/>
  </main>
  <LogSheet :job="shownJob" @close="shownId = null"/>
  <ToastHost/>
</template>
