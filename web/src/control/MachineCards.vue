<script setup lang="ts">
// GPU, CPU (one bar per core), RAM and disk, with the last few minutes of GPU and CPU use.
import {computed, ref, watch} from 'vue';
import Bar from '../components/Bar.vue';
import Spark from '../components/Spark.vue';
import type {SystemInfo} from '../common/panel-types';

const props = defineProps<{system: SystemInfo}>();
// One point per refresh (every 4 s), kept in the page.
const HISTORY_POINTS = 90;
const history = ref({cpu: [] as number[], gpu: [] as number[]});
function push(list: number[], value: number): number[] {
  return [...list, value].slice(-HISTORY_POINTS);
}
watch(() => props.system, (system) => {
  history.value = {
    gpu: system.gpu ? push(history.value.gpu, system.gpu.utilization ?? 0) : history.value.gpu,
    cpu: system.cpu ? push(history.value.cpu, system.cpu.total) : history.value.cpu,
  };
}, {immediate: true});

type Rows = [string, string][];
const gpu = computed(() => {
  const value = props.system.gpu;
  if (!value) return null;
  const busy = value.process_list.length || value.processes;
  const rows: Rows = [['Scheda', value.name.replace(/^Tesla /, '')]];
  if (value.memory_total_mb && value.memory_used_mb !== null) {
    rows.push(['Memoria', (value.memory_used_mb / 1024).toFixed(1) + ' di ' + Math.round(value.memory_total_mb / 1024) + ' GB']);
  }
  if (value.temperature !== null) rows.push(['Temperatura', value.temperature + ' °C']);
  if (value.power_w !== null) rows.push(['Potenza', Math.round(value.power_w) + ' W']);
  rows.push(['Lavora a', busy ? busy + (busy === 1 ? ' processo' : ' processi') : 'niente (libera)']);
  return {util: value.utilization ?? 0, rows};
});
const cpuRows = computed((): Rows => {
  const {cpu, cpus, loads} = props.system;
  const rows: Rows = cpu
    ? [['Core occupati', cpu.cores.filter((value) => value >= 50).length + ' di ' + cpus]]
    : [['Core', String(cpus)]];
  if (loads && cpus) {
    // Load average = programs running or waiting: per core it says how loaded the machine was over the
    // last 15 minutes, and comparing 1 and 15 minutes gives the trend.
    const [one, , fifteen] = loads;
    const perCore = fifteen / cpus;
    rows.push(['Ultimi 15 minuti', perCore < 0.25 ? 'leggero' : perCore < 0.7 ? 'moderato' : 'pesante']);
    rows.push(['Andamento', one > fifteen * 1.3 + 0.5 ? 'in salita' : one < fifteen * 0.7 - 0.5 ? 'in calo' : 'stabile']);
  }
  return rows;
});
const memory = computed(() => {
  const value = props.system.memory;
  if (!value) return null;
  return {value, kind: value.used_percent > 90 ? 'bad' : value.used_percent > 75 ? 'warn' : '',
    rows: [['In uso', (value.total_gb - value.available_gb).toFixed(1) + ' GB (' + value.used_percent + '%)'],
      ['Totale', value.total_gb + ' GB']] as Rows};
});
const disk = computed(() => {
  const value = props.system.disk;
  if (!value) return null;
  const used = 100 - value.free_gb * 100 / value.total_gb;
  return {value, used, kind: value.free_gb < 10 ? 'bad' : used > 80 ? 'warn' : '',
    rows: [['In uso', (value.total_gb - value.free_gb).toFixed(1) + ' GB (' + Math.round(used) + '%)'],
      ['Totale', value.total_gb + ' GB']] as Rows};
});
const coreClass = (value: number): string => 'core' + (value >= 85 ? ' hot' : value >= 50 ? ' warm' : '');
</script>

<template>
  <div class="metrics">
    <div class="card metric">
      <h2>GPU</h2>
      <template v-if="gpu">
        <div class="value">{{ gpu.util }}%</div>
        <Bar :percent="gpu.util" :kind="gpu.util > 85 ? 'warn' : ''"/>
        <Spark :values="history.gpu" color="var(--spark-gpu)" :points="HISTORY_POINTS"/>
        <div class="stats"><template v-for="[label, value] in gpu.rows" :key="label">
          <span class="stat-label">{{ label }}</span><span class="stat-value">{{ value }}</span></template></div>
      </template>
      <template v-else>
        <div class="value">n/d</div>
        <div class="stats"><span class="stat-label">Scheda</span><span class="stat-value">nessuna GPU NVIDIA trovata</span></div>
      </template>
    </div>
    <div class="card metric">
      <h2>CPU</h2>
      <div class="value">{{ system.cpu ? Math.round(system.cpu.total) + '%' : '…' }}</div>
      <Spark :values="history.cpu" color="var(--spark-cpu)" :points="HISTORY_POINTS"/>
      <div class="cores" aria-hidden="true">
        <span v-for="(value, index) in system.cpu?.cores ?? []" :key="index" :class="coreClass(value)"
              :style="{'--h': Math.max(4, value) + '%'}" :title="'core ' + index + ': ' + Math.round(value) + '%'"></span>
      </div>
      <div class="stats"><template v-for="[label, value] in cpuRows" :key="label">
        <span class="stat-label">{{ label }}</span><span class="stat-value">{{ value }}</span></template></div>
    </div>
    <div class="card metric">
      <h2>RAM</h2>
      <template v-if="memory">
        <div class="value">{{ memory.value.available_gb }} GB liberi</div>
        <Bar :percent="memory.value.used_percent" :kind="memory.kind"/>
        <div class="stats"><template v-for="[label, value] in memory.rows" :key="label">
          <span class="stat-label">{{ label }}</span><span class="stat-value">{{ value }}</span></template></div>
      </template>
      <div v-else class="value">n/d</div>
    </div>
    <div class="card metric">
      <h2>Disco</h2>
      <template v-if="disk">
        <div class="value">{{ disk.value.free_gb }} GB liberi</div>
        <Bar :percent="disk.used" :kind="disk.kind"/>
        <div class="stats"><template v-for="[label, value] in disk.rows" :key="label">
          <span class="stat-label">{{ label }}</span><span class="stat-value">{{ value }}</span></template></div>
      </template>
      <div v-else class="value">–</div>
    </div>
  </div>
</template>
