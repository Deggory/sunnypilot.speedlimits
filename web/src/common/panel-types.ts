// GET /api/panel/status (nnslr_tools.panel.Panel.status): the Pannello and the Stato page.

export type ActionName = 'fetch' | 'prepare' | 'dataset' | 'train' | 'train_continue' | 'update';
export type JobState = 'running' | 'ok' | 'failed' | 'interrupted';
export type RouteState = 'collegare' | 'estrarre' | 'preannotare' | 'revisionare' | 'importare' | 'pronta';

export interface ActionState {
  label: string;
  allowed: boolean;
  reason: string;
}

export interface Job {
  id: string;
  action: ActionName;
  label: string;
  /** "2026-10-23 18:55:09" */
  started: string;
  state: JobState;
  code: number | null;
  stoppable: boolean;
  /** Bytes of the log so far. */
  size: number;
}

export interface Activity {
  group: string;
  label: string;
  pid: number;
}

export interface GpuProcess {
  pid: number | null;
  name: string;
  memory_mb: number | null;
}

export interface Gpu {
  name: string;
  utilization: number | null;
  memory_used_mb: number | null;
  memory_total_mb: number | null;
  temperature: number | null;
  power_w: number | null;
  /** How many processes use the GPU. */
  processes: number;
  process_list: GpuProcess[];
}

export interface SystemInfo {
  gpu: Gpu | null;
  disk: {free_gb: number; total_gb: number} | null;
  cpus: number | null;
  load: number | null;
  /** Load average of 1, 5 and 15 minutes. */
  loads: [number, number, number] | null;
  /** Use since the previous status call, whole CPU and per core, in percent. */
  cpu: {total: number; cores: number[]} | null;
  memory: {total_gb: number; available_gb: number; used_percent: number} | null;
}

export interface RouteRow {
  route: string;
  segments: number;
  unlinked: number;
  linked: number;
  extracted: number;
  frames: number;
  run_id: string | null;
  report_frames: number;
  confirmed: number;
  new_reviews: number;
  objects: number;
  state: RouteState;
}

export interface RouteProgress {
  route: string;
  phase: string;
  done: number;
  total: number;
  unit: string;
  percent: number;
}

export interface Preparation {
  percent: number | null;
  routes_done: number;
  routes_total: number;
  working: RouteProgress[];
}

export interface PanelStatus {
  available: boolean;
  actions_enabled: boolean;
  routes: RouteRow[];
  next_step: string;
  next_action: 'prepare' | 'review' | 'train' | 'fetch' | null;
  reviewed_routes: number;
  min_routes: number;
  activities: Activity[];
  jobs: Job[];
  system: SystemInfo;
  actions: Record<ActionName, ActionState>;
  preparation: Preparation | null;
}

/** POST /api/panel/run/<action> */
export interface StartedJob {
  id: string;
  action: ActionName;
  label: string;
}

/** GET /api/panel/job/<id>/log?offset= */
export interface LogChunk {
  id: string;
  text: string;
  offset: number;
  size: number;
}
