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

// GET /api/panel/runs and /api/panel/runs/<run> (nnslr_tools.runs.RunsIndex): the Evaluate tab.

/** running: its pipeline is running now; finished: bundle packaged; incomplete: stopped before the end. */
export type RunState = 'running' | 'finished' | 'incomplete';
/** The step a run reached (scripts/run_v100_baseline.sh), 'done' when packaged; 'experiment' for a folder of
 * scripts/run_experiment_b.sh alone, 'done' when its three variants have a result. */
export type RunStage = 'checks' | 'detector' | 'reader' | 'evaluate' | 'export' | 'bundle' | 'experiment' | 'done';

export interface RunStart {
  /** Where the weights came from; null when not recorded (a fresh run before its detector finished). */
  kind: 'imagenet' | 'random' | 'stable' | 'run' | 'weights' | null;
  label: string;
  /** kind 'run': the earlier run it continued from. */
  run: string | null;
  /** The weights file or Hugging Face path, from the log. */
  source: string | null;
}

export interface RunSummary {
  name: string;
  /** training: the pipeline of helpers/3_train.sh; experiment: a folder with experiment B only. */
  kind: 'training' | 'experiment';
  state: RunState;
  stage: RunStage;
  /** "training the detector", ... */
  stage_label: string;
  /** A running run: "detector · epoch 5 of 20". */
  progress: string | null;
  /** An incomplete run: the last error line of its log. */
  error: string | null;
  /** Experiment B is running in this folder now, and how far ("experiment B · crop_geom · epoch 3 of 20"). */
  experiment_running: boolean;
  experiment_progress: string | null;
  started_at: string | null;
  /** The last change of any of its files. */
  updated_at: string | null;
  /** When the bundle was packaged. */
  finished_at: string | null;
  /** The pipeline's mode (fresh, continue, resume), from the log; null without a log. */
  mode: string | null;
  /** Resumed after an interruption at least once. */
  resumed: boolean;
  start: RunStart;
  dataset_sha256: string | null;
  split_sha256: string | null;
  evaluated: boolean;
  /** Reader on the test partition. */
  reader: {samples: number; accuracy: number | null; human_samples: number | null; human_accuracy: number | null} | null;
  /** Detector on the test partition (IoU matching). */
  detector: {truth_boxes: number; found: number; missed: number; false_boxes: number; recall: number | null;
    precision: number | null; f1: number | null; human_recall: number | null; human_precision: number | null} | null;
  /** Best validation values of the training (result.json). */
  best: {detector_val_loss: number | null; reader_val_accuracy: number | null};
  /** Experiment B variants with a result.json. */
  experiment_b: string[];
  /** Files that exist but cannot be read. */
  problems: string[];
  /** Changes when anything of the run changes: the page asks the detail again. */
  version: string;
}

export interface RunsPayload {
  runs: RunSummary[];
  runs_dir_exists: boolean;
}

export interface LabelRow {
  /** maximum_speed:60, cancellation, ... */
  label: string;
  /** "Speed limit 60". */
  name: string;
  count: number;
  correct: number;
  accuracy: number | null;
  /** The wrong answer given most often, or null when never wrong. */
  wrong: {label: string; name: string; count: number} | null;
}

export interface ReaderEvaluation {
  partition: string;
  samples: number;
  correct: number;
  accuracy: number | null;
  human: {samples: number; accuracy: number | null; correct: number | null} | null;
  counts_by_label_basis: Record<string, number>;
  labels: LabelRow[];
  labels_human: LabelRow[] | null;
  /** Labels of the test the network never learnt. */
  unseen_labels: string[];
  classes: number | null;
}

export interface BoxCounts {
  images: number | null;
  truth_boxes: number;
  found: number;
  missed: number;
  false_boxes: number;
  recall: number | null;
  precision: number | null;
  f1: number | null;
}

export interface SizeRow {
  key: string;
  /** "< 12 px", "12–20 px", ... (width of the box in the frame). */
  label: string;
  truth: number;
  found: number;
  missed: number;
  recall: number | null;
  /** False boxes of this width. */
  false_boxes: number;
}

export interface DetectorEvaluation extends BoxCounts {
  partition: string;
  false_per_image: number | null;
  score_threshold: number | null;
  iou_threshold: number | null;
  human: BoxCounts | null;
  counts_by_label_basis: Record<string, number>;
  /** null when the report has no boxes. */
  sizes: SizeRow[] | null;
  sizes_human: SizeRow[] | null;
  routes: (BoxCounts & {route: string})[];
}

export interface TrainingResult {
  architecture: string;
  initialization: string | null;
  epochs: number | null;
  best_validation_loss: number | null;
  best_validation_accuracy: number | null;
  best_epoch: number | null;
  classes: number | null;
  device: string | null;
  gpu: string | null;
}

export interface SplitUsed {
  policy: string | null;
  seed: number | null;
  grouping: string[];
  /** The routes reviewed entirely by hand whose places are the test. */
  test_routes: string[];
  /** Routes with rows in each partition. */
  routes: Record<string, string[]>;
  counts: Record<string, number>;
  human_reviewed_counts: Record<string, number>;
  warnings: string[];
}

export interface DatasetUsed {
  dataset_sha256: string | null;
  split_sha256: string | null;
  dataset_kind: string | null;
  /** splits/<sha>.json, relative to the data root. */
  split_file: string | null;
  split: SplitUsed | null;
  /** The split is the one data/splits/latest.json points to now. */
  current_split: boolean;
  detector: {frames: Record<string, number>; boxes: Record<string, number>; negative_frames: Record<string, number>} | null;
  reader: {crops: Record<string, number>; classes: number | null} | null;
}

export interface Matrix {
  labels: string[];
  /** "Ours", "Another", "Unclear". */
  names: string[];
  /** Rows: the truth; columns: the answer. */
  matrix: number[][];
}

export interface ExperimentCell {
  all: number | null;
  human: number | null;
}

export interface ExperimentVariant {
  key: string;
  label: string;
  found: boolean;
  unreadable: boolean;
  at: string | null;
  inputs: string[] | null;
  best_epoch: number | null;
  epochs: number | null;
  selection_score: number | null;
  initialization: string | null;
  split_sha256: string | null;
  /** Frames per partition kept (a smoke run, not a measure), or null. */
  limit: number | null;
}

export interface ExperimentB {
  variants: ExperimentVariant[];
  selection_metric: string | null;
  /** test, validation (or 'result' for a report of another shape). */
  partitions: string[];
  metrics: {key: string; label: string; unit: 'percent' | 'count' | 'number'; better: 'higher' | 'lower' | null;
    values: Record<string, Record<string, ExperimentCell>>}[];
  confusions: {key: string; label: string; variants: Record<string, Record<string, {all: Matrix | null; human: Matrix | null}>>}[];
}

export interface RunDetail extends RunSummary {
  files: Record<string, boolean>;
  training: {detector: TrainingResult | null; reader: TrainingResult | null};
  reader_eval: ReaderEvaluation | null;
  detector_eval: DetectorEvaluation | null;
  dataset: DatasetUsed | null;
  curves: Partial<Record<'detector' | 'reader', TrainingCurve>>;
  experiment: ExperimentB | null;
}
