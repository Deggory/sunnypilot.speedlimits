<script setup lang="ts">
// GPU, CPU (one bar per core), RAM and disk, with the last few minutes of GPU and CPU use.
import {computed} from 'vue';
import Bar from '../components/Bar.vue';
import Spark from '../components/Spark.vue';
import type {SystemInfo} from '../common/panel-types';
import {HISTORY_POINTS, machineHistory as history} from './machine-history';

const props = defineProps<{system: SystemInfo}>();

type Rows = [string, string][];
const gpu = computed(() => {
  const value = props.system.gpu;
  if (!value) return null;
  const busy = value.process_list.length || value.processes;
  const rows: Rows = [['Card', value.name.replace(/^Tesla /, '')]];
  if (value.memory_total_mb && value.memory_used_mb !== null) {
    rows.push(['Memory', (value.memory_used_mb / 1024).toFixed(1) + ' of ' + Math.round(value.memory_total_mb / 1024) + ' GB']);
  }
  if (value.temperature !== null) rows.push(['Temperature', value.temperature + ' °C']);
  if (value.power_w !== null) rows.push(['Power', Math.round(value.power_w) + ' W']);
  rows.push(['Running', busy ? busy + (busy === 1 ? ' process' : ' processes') : 'nothing (free)']);
  return {util: value.utilization ?? 0, rows};
});
const cpuRows = computed((): Rows => {
  const {cpu, cpus, loads} = props.system;
  const rows: Rows = cpu
    ? [['Busy cores', cpu.cores.filter((value) => value >= 50).length + ' of ' + cpus]]
    : [['Cores', String(cpus)]];
  if (loads && cpus) {
    // Load average = programs running or waiting: per core it says how loaded the machine was over the
    // last 15 minutes, and comparing 1 and 15 minutes gives the trend.
    const [one, , fifteen] = loads;
    const perCore = fifteen / cpus;
    rows.push(['Last 15 minutes', perCore < 0.25 ? 'light' : perCore < 0.7 ? 'moderate' : 'heavy']);
    rows.push(['Trend', one > fifteen * 1.3 + 0.5 ? 'rising' : one < fifteen * 0.7 - 0.5 ? 'falling' : 'steady']);
  }
  return rows;
});
const memory = computed(() => {
  const value = props.system.memory;
  if (!value) return null;
  return {value, kind: value.used_percent > 90 ? 'bad' : value.used_percent > 75 ? 'warn' : '',
    rows: [['In use', (value.total_gb - value.available_gb).toFixed(1) + ' GB (' + value.used_percent + '%)'],
      ['Total', value.total_gb + ' GB']] as Rows};
});
const disk = computed(() => {
  const value = props.system.disk;
  if (!value) return null;
  const used = 100 - value.free_gb * 100 / value.total_gb;
  return {value, used, kind: value.free_gb < 10 ? 'bad' : used > 80 ? 'warn' : '',
    rows: [['In use', (value.total_gb - value.free_gb).toFixed(1) + ' GB (' + Math.round(used) + '%)'],
      ['Total', value.total_gb + ' GB']] as Rows};
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
        <div class="value">n/a</div>
        <div class="stats"><span class="stat-label">Card</span><span class="stat-value">no NVIDIA GPU found</span></div>
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
        <div class="value">{{ memory.value.available_gb }} GB free</div>
        <Bar :percent="memory.value.used_percent" :kind="memory.kind"/>
        <div class="stats"><template v-for="[label, value] in memory.rows" :key="label">
          <span class="stat-label">{{ label }}</span><span class="stat-value">{{ value }}</span></template></div>
      </template>
      <div v-else class="value">n/a</div>
    </div>
    <div class="card metric">
      <h2>Disk</h2>
      <template v-if="disk">
        <div class="value">{{ disk.value.free_gb }} GB free</div>
        <Bar :percent="disk.used" :kind="disk.kind"/>
        <div class="stats"><template v-for="[label, value] in disk.rows" :key="label">
          <span class="stat-label">{{ label }}</span><span class="stat-value">{{ value }}</span></template></div>
      </template>
      <div v-else class="value">–</div>
    </div>
  </div>
</template>
