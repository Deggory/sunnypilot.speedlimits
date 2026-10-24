<script setup lang="ts">
// Evaluate: the training runs of this server (data/runs/<run>/, newest first) with how they started, how far they
// got and their headline numbers on the test; the run chosen, in detail (the reader per label, the detector per
// sign size and route, the dataset and split it used, its curves, experiment B); two runs side by side.
import {computed} from 'vue';
import {plural} from '../common/format';
import {chooseRun, selectedRun, useRunDetail, useRuns} from './evaluate';
import RunCompare from './RunCompare.vue';
import RunDetailView from './RunDetail.vue';
import RunList from './RunList.vue';

const {payload, error} = useRuns();
const runs = computed(() => payload.value?.runs ?? []);
/** The run shown: the one chosen (also from ?run= of the address), else the newest evaluated one, else the newest. */
const shown = computed(() => {
  const list = runs.value;
  return list.find((run) => run.name === selectedRun.value) ?? list.find((run) => run.evaluated) ?? list[0] ?? null;
});
const {detail, loading, error: detailError} = useRunDetail(computed(() => shown.value?.name ?? null),
  computed(() => shown.value?.version ?? null));
const evaluated = computed(() => runs.value.filter((run) => run.evaluated).length);
</script>

<template>
  <header class="page-head">
    <h1 class="page-title">Evaluate</h1>
    <p class="page-sub">How every training did on the test: places kept apart, never seen while learning.</p>
  </header>

  <section class="card runs-card" aria-labelledby="runs-title">
    <div class="ds-head">
      <h2 id="runs-title">Training runs<span v-if="runs.length" class="count-badge">{{ runs.length }}</span></h2>
      <span v-if="runs.length" class="ds-when">{{ evaluated ? plural(evaluated, 'run') + ' evaluated' : 'none evaluated yet' }}</span>
    </div>
    <p v-if="payload === null" class="empty">{{ error ? 'The runs cannot be read: ' + error : 'Loading…' }}</p>
    <div v-else-if="runs.length === 0" class="run-empty" id="runs-empty">
      <div class="idle-icon" aria-hidden="true"><svg viewBox="0 0 24 24"><path d="M4 19.5h16M6.5 16V11M11 16V7M15.5 16v-3.5M20 16V9" fill="none"
        stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/></svg></div>
      <div class="idle-text">
        <strong>No training finished yet: start one on the Train tab.</strong>
        <span>Every training is measured on the test at its end; its numbers appear here, run by run.</span>
        <a class="flow-link" href="/train">Go to Train →</a>
      </div>
    </div>
    <RunList v-else :runs="runs" :selected="shown?.name ?? null" @select="chooseRun"/>
    <p v-if="payload && error" class="note">Not updated: {{ error }}</p>
  </section>

  <RunDetailView v-if="shown" :key="shown.name" :summary="shown" :detail="detail" :loading="loading" :error="detailError"/>
  <RunCompare v-if="runs.length >= 2" :runs="runs"/>
</template>
