<script setup lang="ts">
// Panoramica: the next step, the three phases, what is running now (live, with Ferma), or how the last job
// ended; the machine, the recent jobs, the site.
import {computed} from 'vue';
import type {ActionName, ActionState, Job, PanelStatus} from '../common/panel-types';
import FlowTiles from './FlowTiles.vue';
import JobItem from './JobItem.vue';
import JobProgress from './JobProgress.vue';
import {terminalActivities} from './jobs';
import MachineCards from './MachineCards.vue';
import NextStepCard from './NextStepCard.vue';
import {nextStep, WAITING} from './next-step';
import type {ControlTab} from './tabs';
import TerminalActivity from './TerminalActivity.vue';
import PrepRoutes from './PrepRoutes.vue';

const props = defineProps<{status: PanelStatus | null; busy: ReadonlySet<ControlTab>}>();
defineEmits<{run: [name: ActionName, state: ActionState]; stop: [job: Job]; details: [job: Job]}>();

const next = computed(() => props.status ? nextStep(props.status) : WAITING);
const jobs = computed(() => props.status?.jobs ?? []);
const running = computed(() => jobs.value.filter((job) => job.state === 'running'));
const activities = computed(() => terminalActivities(props.status));
const last = computed(() => jobs.value[0] ?? null);
const update = computed(() => props.status?.actions.update ?? null);
const gpuProcesses = computed(() => props.status?.system.gpu?.process_list ?? []);
</script>

<template>
  <header class="page-head">
    <h1 class="page-title">Panoramica</h1>
    <p class="page-sub">Cosa sta facendo il server, come sta e qual è il prossimo passo.</p>
  </header>

  <NextStepCard :next="next" :actions="status?.actions ?? null" @run="(name, state) => $emit('run', name, state)"/>
  <FlowTiles :status="status" :attention="next.attention" :busy="busy"/>

  <section class="live-section" aria-labelledby="running-title">
    <h2 class="section-title" id="running-title">In corso<span v-if="running.length + activities.length" class="count-badge">{{ running.length + activities.length }}</span></h2>
    <div v-if="running.length + activities.length" class="jp-list" aria-live="polite">
      <JobProgress v-for="job in running" :key="job.id" :job="job" @details="$emit('details', $event)" @stop="$emit('stop', $event)">
        <PrepRoutes v-if="job.action === 'prepare' && status?.preparation" :routes="status.preparation.working"/>
      </JobProgress>
      <TerminalActivity v-for="activity in activities" :key="activity.label" :activity="activity" :preparation="status?.preparation ?? null"/>
    </div>
    <div v-else class="card idle" :class="{waiting: status === null}">
      <div class="idle-icon" aria-hidden="true"><svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="8.5" fill="none" stroke="currentColor"
        stroke-width="1.8"/><path d="M8.3 12.3l2.5 2.5 4.9-5.2" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"
        stroke-linejoin="round"/></svg></div>
      <div class="idle-text">
        <strong>{{ status === null ? 'Caricamento…' : 'Nessun lavoro in corso' }}</strong>
        <span v-if="status">Né dal sito né da un terminale. I pulsanti di Prepara e Allena avviano i lavori in background.</span>
      </div>
    </div>
    <JobProgress v-if="!running.length && last" :job="last" heading="Ultimo lavoro" @details="$emit('details', $event)" @stop="$emit('stop', $event)"/>
  </section>

  <section aria-labelledby="machine-title">
    <h2 class="section-title" id="machine-title">Il server</h2>
    <MachineCards v-if="status" :system="status.system"/>
  </section>

  <div class="grid2">
    <div class="card">
      <h2>Lavori recenti</h2>
      <ul class="list">
        <JobItem v-for="job in jobs" :key="job.id" :job="job" @details="$emit('details', $event)" @stop="$emit('stop', $event)"/>
      </ul>
      <p v-if="status && jobs.length === 0" class="empty">Nessun lavoro ancora: i pulsanti di Prepara e Allena li avviano in background.</p>
    </div>
    <div class="side">
      <div class="card">
        <h2>Sito</h2>
        <p class="lead">Aggiornare scarica l’ultima versione del codice (git pull), aggiorna le librerie e riavvia il sito; parte solo quando non c’è altro in corso.</p>
        <div v-if="update" class="action">
          <button class="btn" type="button" id="run-update" :disabled="!update.allowed"
                  :aria-describedby="!update.allowed && update.reason ? 'reason-update' : undefined"
                  @click="$emit('run', 'update', update)">{{ update.label }}</button>
          <div v-if="!update.allowed && update.reason" id="reason-update" class="reason">{{ update.reason.charAt(0).toUpperCase() + update.reason.slice(1) }}.</div>
        </div>
      </div>
      <div class="card">
        <h2>Processi sulla GPU</h2>
        <ul v-if="gpuProcesses.length" class="gpu-procs">
          <li v-for="(process, index) in gpuProcesses" :key="index">
            <span class="mono">{{ process.pid ?? '?' }}</span><span class="grow">{{ process.name }}</span>
            <span class="num">{{ process.memory_mb === null ? '–' : Math.round(process.memory_mb) + ' MiB' }}</span>
          </li>
        </ul>
        <p v-else class="empty">Nessun processo sulla GPU.</p>
        <p class="note">Dentro il container LXC nvidia-smi spesso non vede i nomi dei processi: i lavori noti sono in “In corso”.</p>
      </div>
    </div>
  </div>
</template>
