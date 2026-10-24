<script setup lang="ts">
// The machine in small at the right of the tab row, on every page with the tabs (#tabs-machine of pages.py): GPU,
// its memory and temperature, CPU, RAM and the free disk, each its value and a stack of 5 notches filled to its level
// now (a history would start again at every reload). The control pages
// feed it from their status (feed "status"); the others ask /api/panel/system every 4 s (feed "system").
import {computed, onBeforeUnmount, onMounted} from 'vue';
import {api} from '../common/api';
import type {SystemInfo} from '../common/panel-types';
import {machineCells, machineNow, recordMachine} from '../control/machine-history';

const props = defineProps<{feed: string}>();
/** 30 points of 4 s: the last two minutes. */
const POINTS = 30;
const INTERVAL_MS = 4000;

const cells = computed(() => machineNow.value ? machineCells(machineNow.value, POINTS, true, true) : []);
const NOTCHES = [1, 2, 3, 4, 5];

/** Notches lit for a level 0-100: one per 20%, at least one above zero. */
function lit(level: number): number {
  return level <= 0 ? 0 : Math.min(5, Math.max(1, Math.ceil(level / 20)));
}
/** The stack's colour: calm, then warm at 4 notches, hot at 5 (or a hot GPU). */
function heat(level: number, hot: boolean | undefined): string {
  if (hot || lit(level) >= 5) return 'hot';
  return lit(level) >= 4 ? 'warm' : '';
}

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
      <span class="tm-stack" :class="heat(cell.level, cell.hot)" aria-hidden="true"><i v-for="n in NOTCHES" :key="n"
        :class="{on: n <= lit(cell.level)}"></i></span>
    </span>
  </div>
</template>
