// The last minutes of the machine (GPU and its memory, CPU, RAM), one point per status (every 4 s), kept in
// the page for the big cards of the Overview and the mini gauges of a running job.
import {shallowRef} from 'vue';
import type {SystemInfo} from '../common/panel-types';

/** 90 points: 6 minutes. */
export const HISTORY_POINTS = 90;

export interface MachineHistory {
  gpu: number[];
  gpuMemory: number[];
  cpu: number[];
  ram: number[];
}

export const machineHistory = shallowRef<MachineHistory>({gpu: [], gpuMemory: [], cpu: [], ram: []});

function push(list: number[], value: number | null): number[] {
  return value === null ? list : [...list, value].slice(-HISTORY_POINTS);
}

/** GPU memory in use, 0-100, or null when unknown. */
export function gpuMemoryPercent(system: SystemInfo): number | null {
  const gpu = system.gpu;
  return gpu && gpu.memory_total_mb && gpu.memory_used_mb !== null ? gpu.memory_used_mb * 100 / gpu.memory_total_mb : null;
}

/** One more point, from a status. */
export function recordMachine(system: SystemInfo): void {
  const before = machineHistory.value;
  machineHistory.value = {
    gpu: push(before.gpu, system.gpu ? system.gpu.utilization ?? 0 : null),
    gpuMemory: push(before.gpuMemory, gpuMemoryPercent(system)),
    cpu: push(before.cpu, system.cpu ? system.cpu.total : null),
    ram: push(before.ram, system.memory ? system.memory.used_percent : null),
  };
}
