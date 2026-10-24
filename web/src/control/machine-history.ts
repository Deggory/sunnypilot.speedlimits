// The last minutes of the machine (GPU, its memory and temperature, CPU, RAM), one point per status (every 4 s),
// kept in the page for the strip in the tab row and the mini gauges of a running job.
import {shallowRef} from 'vue';
import type {SystemInfo} from '../common/panel-types';

/** 90 points: 6 minutes. */
export const HISTORY_POINTS = 90;

export interface MachineHistory {
  gpu: number[];
  gpuMemory: number[];
  gpuTemp: number[];
  cpu: number[];
  ram: number[];
}

export const machineHistory = shallowRef<MachineHistory>({gpu: [], gpuMemory: [], gpuTemp: [], cpu: [], ram: []});
/** The machine at the latest status, null before the first. */
export const machineNow = shallowRef<SystemInfo | null>(null);

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
  machineNow.value = system;
  machineHistory.value = {
    gpu: push(before.gpu, system.gpu ? system.gpu.utilization ?? 0 : null),
    gpuMemory: push(before.gpuMemory, gpuMemoryPercent(system)),
    gpuTemp: push(before.gpuTemp, system.gpu?.temperature ?? null),
    cpu: push(before.cpu, system.cpu ? system.cpu.total : null),
    ram: push(before.ram, system.memory ? system.memory.used_percent : null),
  };
}

export interface MachineCell {
  key: string;
  label: string;
  value: string;
  /** The tooltip. */
  title: string;
  /** The last points, 0-100 (the temperature in °C), for the sparkline; empty for the disk. */
  values: number[];
  color: string;
  /** A hot GPU (80 °C and more). */
  hot?: boolean;
  /** How full it is now, 0-100: what the 5-notch stack of the tab row shows (the temperature from 30 to 90 °C). */
  level: number;
}

/** GPU, GPU memory and temperature, CPU, RAM and (with disk) the free disk: each its value now and its last points.
 * short: the short labels of the tab row (VRAM, Temp). */
export function machineCells(system: SystemInfo, points: number, disk = false, short = false): MachineCell[] {
  const {gpu, cpu, memory} = system;
  const history = machineHistory.value;
  const recent = (values: number[]): number[] => values.slice(-points);
  const list: MachineCell[] = [];
  if (gpu) {
    list.push({key: 'gpu', label: 'GPU', value: Math.round(gpu.utilization ?? 0) + '%', title: gpu.name,
      values: recent(history.gpu), color: 'var(--spark-gpu)', level: gpu.utilization ?? 0});
    const used = gpuMemoryPercent(system);
    if (used !== null && gpu.memory_used_mb !== null && gpu.memory_total_mb) {
      list.push({key: 'gpu-memory', label: short ? 'VRAM' : 'GPU memory', value: (gpu.memory_used_mb / 1024).toFixed(1) + ' GB',
        title: Math.round(used) + '% of ' + Math.round(gpu.memory_total_mb / 1024) + ' GB',
        values: recent(history.gpuMemory), color: 'var(--spark-gpu)', level: used});
    }
    if (gpu.temperature !== null) {
      list.push({key: 'gpu-temp', label: short ? 'Temp' : 'GPU temp', value: Math.round(gpu.temperature) + ' °C',
        title: gpu.power_w === null ? gpu.name : Math.round(gpu.power_w) + ' W', values: recent(history.gpuTemp),
        color: 'var(--spark-gpu)', hot: gpu.temperature >= 80, level: (gpu.temperature - 30) * 100 / 60});
    }
  }
  if (cpu) {
    list.push({key: 'cpu', label: 'CPU', value: Math.round(cpu.total) + '%', title: 'all cores',
      values: recent(history.cpu), color: 'var(--spark-cpu)', level: cpu.total});
  }
  if (memory) {
    list.push({key: 'ram', label: 'RAM', value: Math.round(memory.used_percent) + '%',
      title: (memory.total_gb - memory.available_gb).toFixed(1) + ' of ' + memory.total_gb + ' GB in use',
      values: recent(history.ram), color: 'var(--spark-cpu)', level: memory.used_percent});
  }
  if (disk && system.disk) {
    const used = 100 - system.disk.free_gb * 100 / system.disk.total_gb;
    list.push({key: 'disk', label: 'Disk', value: Math.round(system.disk.free_gb) + ' GB free',
      title: Math.round(used) + '% of ' + Math.round(system.disk.total_gb) + ' GB in use', values: [], color: '',
      level: used});
  }
  return list;
}
