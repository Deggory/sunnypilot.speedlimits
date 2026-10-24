<script setup lang="ts">
// Panoramica, Prepara, Allena: one app, the tab shown follows the address (tabs.ts). One status for all
// tabs (/api/panel/status every 4 s), the actions, and the sheet with the details of a job.
import {computed, ref, watch} from 'vue';
import type {Job} from '../common/panel-types';
import ToastHost from '../components/ToastHost.vue';
import {usePanelStatus} from '../composables/panel-status';
import {ACTION_GROUP, PREPARE_ACTIONS, TRAIN_ACTIONS} from './jobs';
import LogSheet from './LogSheet.vue';
import OverviewTab from './OverviewTab.vue';
import PrepareTab from './PrepareTab.vue';
import {currentTab, markBusy, useTabRouting, type ControlTab} from './tabs';
import TrainTab from './TrainTab.vue';
import {usePanelActions} from './use-actions';

useTabRouting();
const {status, refresh} = usePanelStatus();
const {run, stop} = usePanelActions(refresh);

/** The job in the sheet, by id: the sheet shows its latest copy from the status. */
const shownId = ref<string | null>(null);
const shownJob = computed(() => status.value?.jobs.find((job) => job.id === shownId.value) ?? null);
function details(job: Job): void {
  shownId.value = job.id;
}

/** The tabs with work running: a dot on their link in the top bar and on the tiles of Panoramica. */
const busy = computed((): ReadonlySet<ControlTab> => {
  const tabs = new Set<ControlTab>();
  const value = status.value;
  if (value === null) return tabs;
  const groups = new Set([...value.jobs.filter((job) => job.state === 'running').map((job) => ACTION_GROUP[job.action]),
    ...value.activities.map((activity) => activity.group)]);
  if (PREPARE_ACTIONS.some((action) => groups.has(ACTION_GROUP[action]))) tabs.add('prepare');
  if (TRAIN_ACTIONS.some((action) => groups.has(ACTION_GROUP[action]))) tabs.add('train');
  return tabs;
});
watch(busy, markBusy);
</script>

<template>
  <main class="page" :data-tab="currentTab">
    <div v-if="status && !status.actions_enabled" class="banner" id="readonly">Sola lettura: su questo server i pulsanti sono disattivati.</div>
    <OverviewTab v-if="currentTab === 'overview'" :status="status" :busy="busy" @run="run" @stop="stop" @details="details"/>
    <PrepareTab v-else-if="currentTab === 'prepare'" :status="status" @run="run" @stop="stop" @details="details"/>
    <TrainTab v-else :status="status" @run="run" @stop="stop" @details="details"/>
  </main>
  <LogSheet :job="shownJob" @close="shownId = null"/>
  <ToastHost/>
</template>
