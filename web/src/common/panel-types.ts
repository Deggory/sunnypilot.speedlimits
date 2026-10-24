// GET /api/panel/status (nnslr_tools.panel.Panel.status): the control pages (Overview, Prepare, Train).

export type ActionName = 'fetch' | 'prepare' | 'dataset' | 'train' | 'train_continue' | 'update';
export type JobState = 'running' | 'ok' | 'failed' | 'interrupted';
export type RouteState = 'collegare' | 'estrarre' | 'preannotare' | 'revisionare' | 'importare' | 'pronta';

export interface ActionState {
  label: string;
  allowed: boolean;
  reason: string;
}

/** How far a job is, read by the server from its log (nnslr_tools.job_progress). */
export interface JobStep {
  key: string;
  label: string;
  state: 'done' | 'current' | 'todo' | 'failed';
}

/** One epoch of a training, from its log line. */
export interface EpochPoint {
  epoch: number;
  train_loss?: number;
  val_loss?: number;
  train_accuracy?: number;
  val_accuracy?: number;
  seconds?: number;
}

export interface TrainingCurve {
  /** Epochs planned. */
  epochs: number;
  points: EpochPoint[];
}

export interface StepProgress {
  done: number;
  total: number;
  /** 0-100, or null when the step only says what it is doing. */
  percent: number | null;
  text: string;
}

export interface JobProgress {
  steps: JobStep[];
  /** 0-100 when known from steps, epochs or finished routes; null otherwise (never a guess). */
  percent: number | null;
  /** e.g. "detector · epoch 5 of 20", "2 of 5 routes ready". */
  detail: string;
  last_line: string;
  /** A running job: how far its current step is (images checked, batches, the file being downloaded). */
  step_progress?: StepProgress | null;
  /** Training only: the curves of the detector and of the reader, one point per epoch done. */
  curves: Partial<Record<'detector' | 'reader', TrainingCurve>>;
}

export interface Job {
  id: string;
  action: ActionName;
  label: string;
  /** "2026-10-23 18:55:09", the server's clock without a zone. */
  started: string;
  /** The same instant with its offset ("2026-10-23T18:55:09+00:00"), when the server sends it. */
  started_at?: string;
  state: JobState;
  code: number | null;
  stoppable: boolean;
  /** Bytes of the log so far. */
  size: number;
  progress: JobProgress;
}

export interface Activity {
  group: string;
  label: string;
  pid: number;
  /** A training started from a terminal: its progress, read from data/logs/train-<run>.log. */
  progress?: JobProgress;
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

/** What is in the gold dataset (data/annotations/objects.jsonl), its split and the last import. */
export interface DatasetFamily {
  /** sign_family of the objects: maximum_speed, variable_display, cancellation, zone, other_sign, ... */
  key: string;
  /** English label for the page. */
  label: string;
  count: number;
}

export interface DatasetRoute {
  route: string;
  frames: number;
  objects: number;
  /** Objects that are speed signs (maximum_speed, variable_display, zone, cancellation). */
  speed_signs: number;
  human: number;
  ai: number;
}

export interface DatasetSummary {
  frames: number;
  objects: number;
  routes: number;
  families: DatasetFamily[];
  /** Objects confirmed by a person / by an AI reviewer. */
  human: number;
  ai: number;
  per_route: DatasetRoute[];
  /** ISO time of the last change of the dataset file, or null. */
  changed_at: string | null;
  split: {
    /** train, validation, test, buffer (older splits: route-held-out, hard-negative): objects in each. */
    counts: Record<string, number>;
    made_at: string | null;
    warnings: string[];
  } | null;
  last_import: {
    at: string | null;
    new_frames: number;
    already: number;
    /** Confirmed frames changed after their import, not applied (the dataset is append-only). */
    changed_after_import: number;
    warnings: string[];
  } | null;
}

/** A choice of where a training starts (POST /api/panel/run/train with {"start": key}). */
export interface TrainingStart {
  /** 'imagenet' | 'random' | 'stable' | 'run:<name>' */
  key: string;
  label: string;
  description: string;
  available: boolean;
  /** Why it is not available ('' when it is). */
  reason: string;
  /** The suggested choice. */
  default: boolean;
}

export interface PanelStatus {
  available: boolean;
  actions_enabled: boolean;
  routes: RouteRow[];
  /** The next step in Italian, for the terminal (route_status); the page shows next_step_text. */
  next_step: string;
  /** The next step in one English sentence, for the hero card of the Overview. */
  next_step_text: string;
  /** The dataset is current and a training can start. */
  all_ready: boolean;
  next_action: 'prepare' | 'review' | 'train' | 'fetch' | null;
  reviewed_routes: number;
  min_routes: number;
  activities: Activity[];
  jobs: Job[];
  system: SystemInfo;
  actions: Record<ActionName, ActionState>;
  preparation: Preparation | null;
  dataset: DatasetSummary | null;
  training_starts: TrainingStart[];
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
