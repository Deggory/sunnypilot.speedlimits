<script setup lang="ts">
// The server at the right of the top bar, on every page with the tabs (#tabs-machine of pages.py): a small pill
// with the GPU and CPU load ("GPU 85% · CPU 12%"), and under it, on a click, a panel with the GPU, its memory and
// temperature, the CPU, the RAM and the free disk, each its value and a thin bar, what runs now (the jobs of the panel,
// asked while the panel is open) and Update the site at the bottom.
// The dot: green idle, blue working, red when the GPU is hot. The control pages feed it from their status
// (feed "status"); the others ask /api/panel/system every 4 s (feed "system"), which also says what runs: the dots
// of the busy tabs (composables/busy.ts) and the connection state of the top bar (composables/connection.ts).
import {computed, onBeforeUnmount, onMounted, ref, watch} from 'vue';
import {messageOf, poll, post, signedOut} from '../common/api';
import {sentence} from '../common/format';
import {markBusy} from '../composables/busy';
import {askConfirm} from '../composables/confirm';
import {showConnection} from '../composables/connection';
import {toast} from '../composables/toast';
import type {ActionState, Activity, Job, PanelStatus, SystemInfo} from '../common/panel-types';
import {ACTION_TAB, busyTabs, gpuProcessNames} from '../control/jobs';
import {machineCells, machineNow, recordMachine, type MachineCell} from '../control/machine-history';

const props = defineProps<{feed: string}>();
const INTERVAL_MS = 4000;
/** Names of the gauges in the panel, by key. */
const NAMES: Record<string, string> = {gpu: 'GPU', 'gpu-memory': 'GPU memory', 'gpu-temp': 'GPU temperature', cpu: 'CPU',
  ram: 'Memory', disk: 'Disk'};

const cells = computed(() => machineNow.value ? machineCells(machineNow.value, 1, true, true) : []);
const open = ref(false);
/** The jobs running now and whether Update can start, read from /api/panel/status while the panel is open. */
const running = ref<Job[]>([]);
/** The processes on the GPU, named when the server can tell (jobs.ts gpuProcessNames), with their memory. */
const gpuProcesses = ref<{name: string; memory: string}[]>([]);
const updateState = ref<ActionState | null>(null);
/** The tab where a job shows (jobs.ts ACTION_TAB, as its dot). */
const tabOf = (job: Job): string => ACTION_TAB[job.action] ?? '/';
async function askRunning(): Promise<void> {
  try {
    const status = await poll<PanelStatus>('/api/panel/status');
    running.value = status.jobs.filter((job) => job.state === 'running');
    const names = gpuProcessNames(status);
    gpuProcesses.value = (status.system.gpu?.process_list ?? []).map((process, index) => ({name: names[index] ?? process.name,
      memory: process.memory_mb === null ? '–' : (process.memory_mb / 1024).toFixed(1) + ' GB'}));
    updateState.value = status.actions.update ?? null;
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

/** The colour of a gauge's bar: how busy the GPU or the CPU is in the accent (work, not a fault); a warm GPU, a full
 * memory or disk orange then red. */
function heat(cell: MachineCell): string {
  if (cell.hot) return 'hot';
  if (cell.key === 'gpu' || cell.key === 'cpu') return 'use';
  return cell.level >= 90 ? 'hot' : cell.level >= 75 ? 'warm' : '';
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

/**
 * Update the site (panel action "update"): asked first, refused with its reason while something else runs; then its job
 * is followed (the site restarts in the middle, so a few asks get no answer) and this page reloads only when it
 * worked; otherwise the panel and a toast say why not.
 */
const updating = ref('');
const following = ref(false);
const FOLLOW_MS = 2000;
const FOLLOW_LIMIT_MS = 5 * 60 * 1000;
/** Why Update cannot start now, as a sentence; '' when it can (or before the status came). */
const updateBlocked = computed(() => updateState.value && !updateState.value.allowed ? sentence(updateState.value.reason) : '');
async function update(): Promise<void> {
  const ok = await askConfirm({title: 'Update the site?', action: 'Update',
    text: 'Gets the latest version and restarts the site: it is away for a few seconds, then this page reloads.'});
  if (!ok) return;
  updating.value = 'Updating…';
  following.value = true;
  try {
    const job = await post<{id: string}>('/api/panel/run/update', {});
    follow(job.id, Date.now(), 0);
  } catch (error) {
    notUpdated(messageOf(error));
  }
}
function notUpdated(why: string): void {
  following.value = false;
  updating.value = 'Not updated. ' + sentence(why.replace(/^ERROR:\s*/, '') || 'it stopped with an error');
  toast(updating.value, true);
}
/** Asks how the update job is doing every 2 s; interrupted: how many asks in a row saw it neither running nor ended. */
function follow(id: string, since: number, interrupted: number): void {
  window.setTimeout(() => void (async () => {
    let job: Job | undefined;
    try {
      job = (await poll<PanelStatus>('/api/panel/status')).jobs.find((item) => item.id === id);
    } catch (error) {
      if (signedOut(error)) {
        following.value = false;
        updating.value = 'Signed out: reload the page to see whether it updated.';
        return;
      }
      updating.value = 'Restarting the site…';
    }
    if (job?.state === 'ok') {
      updating.value = 'Updated. Reloading…';
      window.location.reload();
      return;
    }
    // A moment between the end of the helper and its exit code in the log reads as interrupted: only a few in a row count.
    const stopped = job?.state === 'interrupted' ? interrupted + 1 : 0;
    if (job?.state === 'running') updating.value = 'Updating…';
    if (job?.state === 'failed') return notUpdated(job.progress.last_line);
    if (stopped >= 3) return notUpdated('it stopped before the end');
    if (Date.now() - since > FOLLOW_LIMIT_MS) {
      following.value = false;
      updating.value = 'The site has not come back yet: reload this page in a minute.';
      return;
    }
    follow(id, since, stopped);
  })(), FOLLOW_MS);
}

function onDocument(event: Event): void {
  const target = event.target as Node;
  // The confirm dialog of Update is outside the panel: answering it keeps the panel open, to show how it goes.
  if (document.getElementById('dialog-host')?.contains(target)) return;
  if (open.value && root.value && !root.value.contains(target)) open.value = false;
}
function onKey(event: KeyboardEvent): void {
  if (event.key === 'Escape') open.value = false;
}

let timer: number | undefined;
async function ask(): Promise<void> {
  try {
    const answer = await poll<{system: SystemInfo; running?: {jobs: Pick<Job, 'action' | 'state'>[]; activities: Activity[]}}>(
      '/api/panel/system');
    recordMachine(answer.system);
    if (answer.running) markBusy(busyTabs(answer.running.jobs, answer.running.activities));
    showConnection(null);
  } catch (error) {
    // The network, or the site restarting: the pills stay as they were, greyed out. Signed out: no more asking (the
    // page stays, the review page keeps its unsaved confirmations).
    showConnection(error);
    if (signedOut(error)) window.clearInterval(timer);
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
        <span class="tm-meter" :class="heat(cell)" aria-hidden="true">
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
      <div v-if="gpuProcesses.length" class="tm-running" id="tm-gpu">
        <h3>On the GPU</h3>
        <div v-for="(process, index) in gpuProcesses" :key="index" class="tm-job">
          <span class="tm-name">{{ process.name }}</span><span class="tm-value">{{ process.memory }}</span>
        </div>
      </div>
      <div class="tm-update">
        <button type="button" class="btn small" :disabled="following || updateBlocked !== ''" @click="update">Update the site</button>
        <span v-if="updating || updateBlocked" class="tm-sub" role="status">{{ following || !updateBlocked ? updating : updateBlocked }}</span>
      </div>
    </div>
  </div>
</template>
