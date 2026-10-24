<script setup lang="ts">
// The machine in small at the right of the tab row, on every page with the tabs (#tabs-machine of pages.py): GPU,
// its memory and temperature, CPU, RAM and the free disk, each its value and a small sparkline. The control pages
// feed it from their status (feed "status"); the others ask /api/panel/system every 4 s (feed "system").
import {computed, onBeforeUnmount, onMounted} from 'vue';
import {api} from '../common/api';
import type {SystemInfo} from '../common/panel-types';
import {machineCells, machineNow, recordMachine} from '../control/machine-history';
import Spark from './Spark.vue';

const props = defineProps<{feed: string}>();
/** 30 points of 4 s: the last two minutes. */
const POINTS = 30;
const INTERVAL_MS = 4000;

const cells = computed(() => machineNow.value ? machineCells(machineNow.value, POINTS, true, true) : []);

let timer: number | undefined;
async function ask(): Promise<void> {
  try {
    recordMachine((await api<{system: SystemInfo}>('/api/panel/system')).system);
  } catch {
    // An old server without the address, or the network: the strip stays as it was.
  }
}
onMounted(() => {
  if (props.feed !== 'system') return;
  void ask();
  timer = window.setInterval(() => void ask(), INTERVAL_MS);
});
onBeforeUnmount(() => window.clearInterval(timer));
</script>

<template>
  <div v-if="cells.length" class="tm" role="group" aria-label="The server now">
    <span v-for="cell in cells" :key="cell.key" class="tm-cell" :class="{hot: cell.hot}" :data-gauge="cell.key"
          :title="cell.label + ': ' + cell.title">
      <span class="tm-label">{{ cell.label }}</span><span class="tm-value">{{ cell.value }}</span>
      <Spark v-if="cell.values.length" :values="cell.values" :color="cell.color" :points="POINTS"/>
    </span>
  </div>
</template>
