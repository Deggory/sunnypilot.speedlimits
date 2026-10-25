<script setup lang="ts">
// The server at the right of the top bar, on every page with the tabs (#tabs-machine of pages.py): a small pill
// with the GPU and CPU load ("GPU 85% · CPU 12%"), and under it, on a click, a panel with the GPU, its memory and
// temperature, the CPU, the RAM and the free disk, each its value and a thin bar, what runs now (the jobs of the panel,
// asked while the panel is open) and Update the site at the bottom.
// The dot: green idle, blue working, red when the GPU is hot. The control pages feed it from their status
// (feed "status"); the others ask /api/panel/system every 4 s (feed "system").
import {computed, onBeforeUnmount, onMounted, ref, watch} from 'vue';
import {api, messageOf, post} from '../common/api';
import {askConfirm} from '../composables/confirm';
import type {Job, PanelStatus, SystemInfo} from '../common/panel-types';
import {machineCells, machineNow, recordMachine} from '../control/machine-history';

const props = defineProps<{feed: string}>();
const INTERVAL_MS = 4000;
/** Names of the gauges in the panel, by key. */
const NAMES: Record<string, string> = {gpu: 'GPU', 'gpu-memory': 'GPU memory', 'gpu-temp': 'GPU temperature', cpu: 'CPU',
  ram: 'Memory', disk: 'Disk'};

const cells = computed(() => machineNow.value ? machineCells(machineNow.value, 1, true, true) : []);
const open = ref(false);
/** The jobs running now, read from /api/panel/status while the panel is open. */
const running = ref<Job[]>([]);
/** The tab where a job shows, by its action. */
function tabOf(job: Job): string {
  return job.action === 'measure_proposer' || job.action.startsWith('measure_ai') ? '/evaluate'
    : ['fetch', 'prepare', 'prepare_again', 'setup_vlm'].includes(job.action) ? '/prepare'
      : job.action === 'ai_review' ? '/routes' : '/train';
}
async function askRunning(): Promise<void> {
  try {
    running.value = (await api<PanelStatus>('/api/panel/status')).jobs.filter((job) => job.state === 'running');
  } catch {
    // The list stays as it was.
  }
}
let runningTimer: number | undefined;
watch(open, (value) => {
  window.clearInterval(runningTimer);
  if (!value) return;
  void askRunning();
  runningTimer = window.setInterval(() => void askRunning(), INTERVAL_MS);
});
const root = ref<HTMLElement | null>(null);

function heat(level: number, hot: boolean | undefined): string {
  if (hot || level >= 90) return 'hot';
  return level >= 75 ? 'warm' : '';
}
/** The pill: GPU and CPU load (the memory is in the panel); the dot blue when one works, red when the GPU is hot. */
const summary = computed(() => {
  const byKey = new Map(cells.value.map((cell) => [cell.key, cell]));
  const gpu = byKey.get('gpu');
  const cpu = byKey.get('cpu');
  const hot = byKey.get('gpu-temp')?.hot === true;
  const busy = (gpu?.level ?? 0) >= 15 || (cpu?.level ?? 0) >= 40;
  return {dot: hot ? 'hot' : busy ? 'busy' : '', gpu: gpu?.value ?? null, cpu: cpu?.value ?? null};
});

/** Update the site (panel action "update"): asked first, then the site restarts and this page reloads. */
const updating = ref('');
async function update(): Promise<void> {
  const ok = await askConfirm({title: 'Update the site?', action: 'Update',
    text: 'Gets the latest version and restarts the site: it is away for a few seconds, then this page reloads.'});
  if (!ok) return;
  updating.value = 'Updating…';
  try {
    await post('/api/panel/run/update', {});
    window.setTimeout(() => window.location.reload(), 15000);
  } catch (error) {
    updating.value = messageOf(error);
  }
}

function onDocument(event: Event): void {
  if (open.value && root.value && !root.value.contains(event.target as Node)) open.value = false;
}
function onKey(event: KeyboardEvent): void {
  if (event.key === 'Escape') open.value = false;
}

let timer: number | undefined;
async function ask(): Promise<void> {
  try {
    recordMachine((await api<{system: SystemInfo}>('/api/panel/system')).system);
  } catch {
    // An old server without the address, or the network: the pill stays as it was.
  }
}
onMounted(() => {
  document.addEventListener('pointerdown', onDocument);
  document.addEventListener('keydown', onKey);
  if (props.feed !== 'system') return;
  void ask();
  timer = window.setInterval(() => void ask(), INTERVAL_MS);
});
onBeforeUnmount(() => {
  window.clearInterval(timer);
  window.clearInterval(runningTimer);
  document.removeEventListener('pointerdown', onDocument);
  document.removeEventListener('keydown', onKey);
});
</script>

<template>
  <div v-if="cells.length" ref="root" class="tm">
    <button type="button" class="tm-pill" :aria-expanded="open" aria-controls="tm-pop" aria-label="The server: GPU and CPU load"
            @click="open = !open">
      <span class="dot" :class="summary.dot" aria-hidden="true"></span>
      <span v-if="summary.gpu !== null"><small>GPU</small> {{ summary.gpu }}</span>
      <span v-if="summary.cpu !== null"><small>CPU</small> {{ summary.cpu }}</span>
    </button>
    <div v-if="open" id="tm-pop" class="tm-pop" role="dialog" aria-label="The server now">
      <h2>Server</h2>
      <p class="tm-sub">Now, every 4 seconds</p>
      <div v-for="cell in cells" :key="cell.key" class="tm-row" :class="{hot: cell.hot}" :data-gauge="cell.key">
        <span class="tm-name">{{ NAMES[cell.key] ?? cell.label }}<small>{{ cell.title }}</small></span>
        <span class="tm-value">{{ cell.value }}</span>
        <span class="tm-meter" :class="heat(cell.level, cell.hot)" aria-hidden="true">
          <span :style="{width: Math.max(2, Math.min(100, cell.level)) + '%'}"></span></span>
      </div>
      <div class="tm-running">
        <h3>Running now</h3>
        <p v-if="running.length === 0" class="tm-sub">Nothing from the site.</p>
        <a v-for="job in running" :key="job.id" class="tm-job" :href="tabOf(job)">
          <span class="tm-name">{{ job.label }}<small>{{ job.progress.detail || job.progress.last_line || 'running' }}</small></span>
          <span class="tm-value">{{ job.progress.percent === null ? '' : Math.floor(job.progress.percent) + '%' }}</span>
        </a>
      </div>
      <div class="tm-update">
        <button type="button" class="btn small" :disabled="updating === 'Updating…'" @click="update">Update the site</button>
        <span v-if="updating" class="tm-sub" role="status">{{ updating }}</span>
      </div>
    </div>
  </div>
</template>
