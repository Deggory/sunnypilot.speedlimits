<script setup lang="ts">
// The machine in small: GPU, GPU memory, CPU and RAM (and the free disk, on the Overview), each its value now
// and the last three minutes as a sparkline. Under the buttons of a running job, and at the top of the Overview.
import {computed} from 'vue';
import type {SystemInfo} from '../common/panel-types';
import Spark from '../components/Spark.vue';
import {gpuMemoryPercent, machineHistory} from './machine-history';

const props = defineProps<{system: SystemInfo; disk?: boolean}>();
/** 45 points of 4 s. */
const POINTS = 45;

const cells = computed(() => {
  const {gpu, cpu, memory} = props.system;
  const history = machineHistory.value;
  const recent = (values: number[]): number[] => values.slice(-POINTS);
  const list: {key: string; label: string; value: string; title: string; values: number[]; color: string}[] = [];
  if (gpu) {
    list.push({key: 'gpu', label: 'GPU', value: Math.round(gpu.utilization ?? 0) + '%', title: gpu.name,
      values: recent(history.gpu), color: 'var(--spark-gpu)'});
    const used = gpuMemoryPercent(props.system);
    if (used !== null && gpu.memory_used_mb !== null && gpu.memory_total_mb) {
      list.push({key: 'gpu-memory', label: 'GPU memory', value: (gpu.memory_used_mb / 1024).toFixed(1) + ' GB',
        title: Math.round(used) + '% of ' + Math.round(gpu.memory_total_mb / 1024) + ' GB',
        values: recent(history.gpuMemory), color: 'var(--spark-gpu)'});
    }
  }
  if (cpu) {
    list.push({key: 'cpu', label: 'CPU', value: Math.round(cpu.total) + '%', title: 'all cores',
      values: recent(history.cpu), color: 'var(--spark-cpu)'});
  }
  if (memory) {
    list.push({key: 'ram', label: 'RAM', value: Math.round(memory.used_percent) + '%',
      title: (memory.total_gb - memory.available_gb).toFixed(1) + ' of ' + memory.total_gb + ' GB in use',
      values: recent(history.ram), color: 'var(--spark-cpu)'});
  }
  const disk = props.system.disk;
  if (props.disk && disk) {
    const used = 100 - disk.free_gb * 100 / disk.total_gb;
    list.push({key: 'disk', label: 'Disk', value: Math.round(disk.free_gb) + ' GB free',
      title: Math.round(used) + '% of ' + Math.round(disk.total_gb) + ' GB in use', values: [], color: ''});
  }
  return list;
});
</script>

<template>
  <div v-if="cells.length" class="jp-machine" role="group" aria-label="The server now">
    <div v-for="cell in cells" :key="cell.key" class="mini" :data-gauge="cell.key" :title="cell.label + ': ' + cell.title">
      <span class="mini-label">{{ cell.label }}</span>
      <span class="mini-value">{{ cell.value }}</span>
      <Spark v-if="cell.values.length" :values="cell.values" :color="cell.color" :points="POINTS"/>
    </div>
  </div>
</template>
