// GET /api/panel/status (nnslr_tools.panel.Panel.status): the control pages (Overview, Prepare, Train).
import type {AiState} from './ai-types';

export type ActionName = 'fetch' | 'prepare' | 'prepare_again' | 'measure_proposer' | 'setup_vlm' | 'dataset' | 'train' | 'train_continue' | 'train_resume' | 'train_classifier'
  | 'train_cutter' | 'update' | 'ai_review' | 'measure_ai_reviewer';
/** What a cutter learns to find: every sign (the speed signs and the other signs) or the speed signs only. */
export type CutterFamilies = 'all' | 'speed';
/** The options of “Train the cutter” (POST /api/panel/run/train_cutter {"size", "families"}). */
export interface CutterOptions {
  size: string;
  families: CutterFamilies;
}
/** Who proposes the signs of a route (nnslr preannotate --proposer). */
export type Proposer = 'vietnam' | 'ours' | 'qwen';
/** GET /api/panel/proposers: a proposer of Prepare measured on the test frames (nnslr_tools.proposer_eval). */
export interface ProposerCounts {
  truth: number;
  found: number;
  read_right: number;
  found_share: number | null;
  read_right_share: number | null;
}
export interface ProposerReport {
  proposer: Proposer;
  label: string;
  partition: string;
  frames: number;
  limited: boolean;
  seconds_per_frame: number;
  measured_at: string;
  speed_signs: ProposerCounts;
  speed_signs_person: ProposerCounts;
  every_sign: {truth: number; found: number; found_share: number | null};
  by_width: (ProposerCounts & {key: string; label: string})[];
  proposals_per_frame: number | null;
  false_per_frame: number | null;
}
export interface ProposersPayload {
  proposers: {key: Proposer; label: string; unavailable: string | null}[];
  reports: ProposerReport[];
}
/** The options of an action: the cutter's size and signs, the proposer of a preparation, or what an AI review
 * reviews (the route, the provider, save yes or no, how many frames at most, as text). */
export interface ActionOptions {
  size?: string;
  families?: CutterFamilies;
  proposer?: Proposer;
  route?: string;
  provider?: string;
  save?: 'yes' | 'no';
  limit?: string;
}
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
  /** The detector: its validation recall at score 0.1. */
  val_recall?: number;
  /** The classifier with “not a sign”: the sign crops read right and the “not a sign” crops rejected on validation, and
   * their mean, which chooses its best epoch. */
  val_sign_accuracy?: number;
  val_rejection?: number;
  val_balanced_accuracy?: number;
  seconds?: number;
}

export interface TrainingCurve {
  /** Epochs planned. */
  epochs: number;
  /** It stopped after this epoch, before the planned ones (validation not better for a while). */
  stopped_early?: number;
  points: EpochPoint[];
}

/** The networks with curves: the pipeline's detector and reader, the classifier with “not a sign”. */
export type CurveNet = 'detector' | 'reader' | 'classifier';

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
  /** Training only: the curves of the detector and of the reader (or of the classifier), one point per epoch done. */
  curves: Partial<Record<CurveNet, TrainingCurve>>;
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
  /** The training run the job wrote to (data/runs/<run>), when it got that far; null otherwise. */
  run?: string | null;
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
  /** The drive ("Basel → Mumpf"), when known. */
  title?: string | null;
  phase: string;
  done: number;
  total: number;
  unit: string;
  percent: number;
  /** Propose again: the time left for this route at its pace. */
  seconds_left?: number;
}

export interface Preparation {
  percent: number | null;
  routes_done: number;
  routes_total: number;
  working: RouteProgress[];
  /** Propose again (routes that had proposals): every route of the batch is listed, with who proposes. */
  again?: boolean;
  seconds_left?: number | null;
  proposer?: string | null;
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
  /** The choices of “Train the cutter”: its input sizes and the default one, what it finds and the default. */
  cutter_options?: {sizes: string[]; size: string; families: CutterFamilies[]; family: CutterFamilies};
  /** Why a proposer cannot run on this server (null: it can). */
  proposers?: {qwen: string | null};
  /** The AI reviewer chosen in Settings (llm_providers.ai_state: a file read, no network). */
  ai_reviewer: AiState;
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
 * scripts/run_experiment_b.sh alone, 'done' when its three variants have a result; mine, classifier, chain for a run of
 * scripts/run_classifier.sh, 'done' when its chain is measured. */
export type RunStage = 'checks' | 'detector' | 'reader' | 'evaluate' | 'export' | 'bundle' | 'experiment' | 'mine' | 'classifier'
  | 'chain' | 'done';

export interface RunStart {
  /** Where the weights came from; null when not recorded (a fresh run before its detector finished). */
  kind: 'imagenet' | 'random' | 'stable' | 'run' | 'weights' | null;
  label: string;
  /** kind 'run': the earlier run it continued from. */
  run: string | null;
  /** The weights file or Hugging Face path, from the log. */
  source: string | null;
}

/** The classifier judged on the test crops: the sign crops (every crop for an old reader) read right, also with the kinds
 * of other signs folded (comparable with the old reader), the “not a sign” crops it rejects, the sign crops it throws away. */
export interface ClassifierNumbers {
  samples: number | null;
  accuracy: number | null;
  human_samples: number | null;
  human_accuracy: number | null;
  /** Null for an old reader (no “not a sign”). */
  family_accuracy?: number | null;
  not_a_sign_crops?: number | null;
  not_a_sign_rejected?: number | null;
  rejection_rate?: number | null;
  signs_rejected?: number | null;
  signs_rejected_rate?: number | null;
  /** 'v2': with “not a sign” and the kinds of other signs. */
  class_scheme?: string | null;
}

/** The chain end to end (cutter -> classifier) at the cutter's score 0.1: the speed signs found and read right. */
export interface ChainNumbers {
  score_threshold: number;
  /** Measured at 0.1 (else at the report's main score). */
  at_chain_score: boolean;
  truth: number | null;
  found: number | null;
  correct: number | null;
  /** End to end: found by the cutter and read right by the classifier. */
  recall: number | null;
  /** The cutter alone. */
  recall_found: number | null;
  read_right_when_found: number | null;
  human_truth: number | null;
  human_recall: number | null;
  boxes_per_frame: number | null;
  false_per_frame: number | null;
  /** False boxes the classifier lets through, per frame. */
  surviving_false_per_frame: number | null;
  /** The classifier can say “not a sign” (the old reader cannot). */
  rejects: boolean | null;
  images: number | null;
}

/** One size of the cutter (scripts/run_detector_sizes.sh) in a line: at score 0.1 on test (else validation). */
export interface CutterLine {
  size: string;
  families: CutterFamilies;
  finished: boolean;
  evaluated: boolean;
  partition: string | null;
  score_threshold: number | null;
  recall: number | null;
  found: number | null;
  truth_boxes: number | null;
  /** A cutter of every sign: its recall on the speed signs alone (comparable with a speed-only cutter). */
  speed_recall: number | null;
  boxes_per_frame: number | null;
  false_per_frame: number | null;
  ms_per_frame: number | null;
}

export interface RunSummary {
  name: string;
  /** training: the pipeline of helpers/3_train.sh; experiment: a folder with experiment B only; classifier: a run of
   * scripts/run_classifier.sh (the classifier with “not a sign”); cutter: a run of scripts/run_detector_sizes.sh alone. */
  kind: 'training' | 'experiment' | 'classifier' | 'cutter';
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
  /** The classifier with “not a sign” on the test crops, and its best validation epoch. */
  classifier: (ClassifierNumbers & {best_validation_balanced_accuracy: number | null; best_validation_accuracy: number | null;
    best_epoch: number | null}) | null;
  /** The chain at the cutter's score 0.1 with this classifier, and with the old reader. */
  chain: ChainNumbers | null;
  chain_old_reader: ChainNumbers | null;
  /** The detectors of scripts/run_detector_sizes.sh in this run, smallest first. */
  cutters: CutterLine[] | null;
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

export interface HumanShare {
  samples: number;
  accuracy: number | null;
  correct: number | null;
}

/** “Not a sign” crops: how many, and how many the classifier rejects (answers “not a sign”). */
export interface Rejection {
  crops: number;
  rejected: number;
  rejection_rate: number | null;
}

export interface ReaderEvaluation {
  partition: string;
  samples: number;
  correct: number;
  accuracy: number | null;
  human: HumanShare | null;
  counts_by_label_basis: Record<string, number>;
  labels: LabelRow[];
  labels_human: LabelRow[] | null;
  /** Labels of the test the network never learnt. */
  unseen_labels: string[];
  classes: number | null;
  /** 'v2' for the classifier with “not a sign” (every crop counted above, “not a sign” included); null for a reader. */
  class_scheme?: string | null;
  /** The sign crops alone (null for a reader). */
  signs?: {
    samples: number;
    correct: number;
    accuracy: number | null;
    /** With other_sign:<kind> folded into other_sign on both sides: comparable with the old reader. */
    family_accuracy: number | null;
    /** Sign crops it answered “not a sign”. */
    rejected: number | null;
    rejected_rate: number | null;
    human: HumanShare | null;
    labels: LabelRow[];
    labels_human: LabelRow[] | null;
    unseen_labels: string[];
  } | null;
  /** The “not a sign” crops (null for a reader): rejected overall, by where they come from, on human-reviewed frames,
   * and what it calls the ones it lets through. */
  not_a_sign?: (Rejection & {
    by_source: (Rejection & {source: string; label: string})[];
    human: Rejection | null;
    survivors: LabelCount[];
  }) | null;
}

export interface LabelCount {
  label: string;
  /** “Speed limit 60”, “Other sign: give way”, “Not a sign”. */
  name: string;
  count: number;
}

/** Reviewed boxes at one score of the cutter: found by the cutter, read right by the classifier, rejected. */
export interface ChainCounts {
  truth: number;
  found: number;
  correct: number;
  rejected: number | null;
  /** End to end: found and read right, of the reviewed boxes. */
  recall: number | null;
  /** The cutter alone. */
  recall_found: number | null;
  read_right_when_found: number | null;
  human: {truth: number | null; found: number | null; correct: number | null; recall: number | null; recall_found: number | null} | null;
  /** Speed signs only: by the width of the reviewed box (lt12, 12-20, 20-40, ge40). */
  widths?: {key: string; label: string; truth: number | null; found: number | null; correct: number | null; recall: number | null;
    recall_found: number | null}[] | null;
}

export interface ChainThreshold {
  score: number;
  speed_signs: ChainCounts | null;
  /** Every reviewed box, other signs included. */
  all_boxes: ChainCounts | null;
  boxes_per_frame: number | null;
  false_boxes: number | null;
  false_per_frame: number | null;
  surviving_false_boxes: number | null;
  surviving_false_per_frame: number | null;
  /** Share of the cutter's false boxes the classifier rejects. */
  false_rejected_rate: number | null;
  human: {images: number | null; false_per_frame: number | null; surviving_false_per_frame: number | null} | null;
  surviving_false_by_label: LabelCount[];
  /** The main score only: the classifier's answers by the review of the box (“not a sign” for a false box). */
  labels: LabelRow[] | null;
}

/** The chain (cutter -> classifier) on a partition (evaluation/chain-<partition>.json). */
export interface ChainEvaluation {
  partition: string;
  cutter: {checkpoint: string | null; id: string | null; same_split: boolean | null; input_size: string | null};
  classifier: {checkpoint: string | null; class_scheme: string | null; classes: number | null; rejects: boolean | null;
    same_split: boolean | null};
  iou_threshold: number | null;
  image_count: number | null;
  human_image_count: number | null;
  speed_sign_count: number | null;
  reviewed_box_count: number | null;
  main_score_threshold: number | null;
  thresholds: ChainThreshold[];
  device: string | null;
  seconds: number | null;
  limitations: string[];
}

/** The training of the classifier with “not a sign” (classifier/result.json). */
export interface ClassifierTraining extends TrainingResult {
  epochs_run: number | null;
  stopped_early: number | null;
  class_scheme: string | null;
  best_validation_sign_accuracy: number | null;
  best_validation_rejection: number | null;
  best_validation_balanced_accuracy: number | null;
  /** What chose the best epoch ('balanced_accuracy'). */
  selection: string | null;
  /** The weight of “not a sign” in the loss. */
  not_a_sign_weight: number | null;
  augmentation: boolean | null;
  negatives: {file: string | null; sha256: string | null; by_split: Record<string, Record<string, number> | number>} | null;
}

/** The header of negatives.jsonl: the cutter's false boxes mined for the classifier. */
export interface NegativesMined {
  created_at: string | null;
  dataset_kind: string | null;
  dataset_sha256: string | null;
  split_sha256: string | null;
  cutter: {checkpoint: string | null; id: string | null; input_size: string | null; same_split: boolean | null};
  settings: {score_threshold: number | null; max_iou: number | null; per_frame: number | null; total_factor: number | null;
    seed: number | null; rule_change_at: string | null};
  splits: {split: string; frames: number | null; frames_with_candidates: number | null; candidates: number | null;
    after_frame_cap: number | null; positives: number | null; cap: number | null; selected: number | null;
    /** human, ai_after_rule, ai_before_rule, other. */
    by_tier: Record<string, number>}[];
}

/** What scripts/run_classifier.sh wrote in a run. */
export interface ClassifierRun {
  files: Record<string, boolean>;
  training: ClassifierTraining | null;
  negatives: NegativesMined | null;
  evaluations: {test: ReaderEvaluation | null; validation: ReaderEvaluation | null};
  chains: {test: ChainEvaluation | null; validation: ChainEvaluation | null; old_reader: ChainEvaluation | null};
  /** comparison.txt, the new classifier against the old reader. */
  comparison: string | null;
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
  /** The classifier with “not a sign”: its crops, and its “not a sign” crops per split by where they come from. */
  classifier?: {crops: Record<string, number>; classes: number | null; class_scheme: string | null;
    negatives: Record<string, Record<string, number> | number>;
    folded_labels: {label: string; name: string; into: string | null; into_name: string | null; count?: number}[] | null} | null;
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
  curves: Partial<Record<CurveNet, TrainingCurve>>;
  experiment: ExperimentB | null;
  /** What scripts/run_classifier.sh wrote here (a run of its own, or into a run of the pipeline). */
  classifier_run: ClassifierRun | null;
  /** What scripts/run_detector_sizes.sh wrote here: each size as the Models tab judges a cutter, the comparison. */
  cutter_run: {sizes: CutterSize[]; comparison: string | null} | null;
  /** The reports of this run the Watch panel can play (only which files exist). */
  watch: WatchModel[];
}

// GET /api/panel/runs/<run>/frames?model=&partition=&route= (nnslr_tools.runs.RunsIndex.frames): the Watch panel.

export type WatchPartition = 'test' | 'validation';

/** A report of a run the Watch panel plays. */
export interface WatchModel {
  /** detector (of the pipeline), detector-sizes/<W>x<H> (a cutter of the sizes), chain, chain-old-reader. */
  model: string;
  /** "Cutter 960 × 544", "Chain", ... */
  label: string;
  kind: 'cutter' | 'chain';
  partitions: WatchPartition[];
}

/** What a report kept of the cutter's boxes frame by frame: every box from its floor (at most max_boxes a frame, with
 * the counts at each step), its boxes at its own threshold only (an older detector report), the signs found and what
 * the classifier said at its main score (an older chain report), nothing (the reviewed boxes alone). */
export type WatchLevel = 'all' | 'above' | 'verdicts' | 'none';

/** x1, y1, x2, y2 in pixels of the frame. */
export type PixelBox = [number, number, number, number];

/** A reviewed box: the truth. */
export interface WatchTruth {
  box: PixelBox;
  family: string | null;
  /** The classifier's label of the review (a chain report); null in a cutter report. */
  label: string | null;
  /** An older chain report: found by the cutter at its main score, and what the classifier called it. */
  found?: boolean;
  verdict?: string | null;
  confidence?: number | null;
}

/** A box of the cutter. */
export interface WatchBox {
  box: PixelBox;
  score: number;
  /** The reviewed box it found (an index of truth), -1 for none: a false box. */
  match: number;
  /** A chain report: what the classifier called it, and how sure it was. */
  label?: string | null;
  confidence?: number | null;
}

/** A box of the cutter as the server sends it: x1, y1, x2, y2, score, truth (a chain: then label, confidence). */
export type WatchBoxRow = [number, number, number, number, number, number]
  | [number, number, number, number, number, number, string | null, number | null];

/** A frame as the server sends it: its boxes as rows (watch.ts makes them WatchBox). */
export interface WatchFrameRows extends Omit<WatchFrame, 'boxes'> {
  boxes: WatchBoxRow[];
}

export interface WatchFrame {
  /** "<segment>/<image>", as the review page names a frame. */
  key: string | null;
  segment: number | null;
  index: number | null;
  /** The preview of the review page, /route/<route>/images/NNNNNNNN.jpg; null without one. */
  image: string | null;
  width: number | null;
  height: number | null;
  /** human_review, ai_review, ... */
  basis: string | null;
  truth: WatchTruth[];
  /** Best score first. */
  boxes: WatchBox[];
  /** Every box of the cutter at or above each step (level all), drawn or not. */
  counts: number[] | null;
  /** A chain: the false boxes the classifier let through at each step. */
  surviving: number[] | null;
}

export interface WatchRoute {
  route: string;
  frames: number;
  signs: number;
  frames_with_signs: number;
  /** At the score the panel starts at; null when the report does not say. */
  missed: number | null;
  frames_with_misses: number | null;
}

export interface WatchPayload {
  run: string;
  model: string;
  label: string;
  partition: WatchPartition;
  kind: 'cutter' | 'chain';
  level: WatchLevel;
  /** The lowest score of the boxes kept. */
  min_score: number | null;
  /** The report's own threshold (a cutter) or main score (a chain). */
  main_score: number | null;
  default_score: number | null;
  max_boxes: number | null;
  /** The scores at which counts and surviving are given (level all). */
  steps: number[] | null;
  iou_threshold: number;
  /** A chain: the classifier has “not a sign”. */
  rejects: boolean | null;
  image_count: number;
  routes: WatchRoute[];
  /** The route shown: the one asked for, else the one with the most signs. */
  route: string | null;
  /** The route asked for when this report has no frame of it. */
  route_not_here: string | null;
  /** Frames of the route with a preview to show. */
  with_image: number;
  frames: WatchFrame[];
}

/** WatchPayload as the server sends it. */
export interface WatchPayloadRows extends Omit<WatchPayload, 'frames'> {
  frames: WatchFrameRows[];
}

export interface CutterSize {
  size: string;
  /** "960 × 544". */
  label: string;
  families: CutterFamilies;
  finished: boolean;
  training: {architecture: string | null; initialization: string | null; best_epoch: number | null; epochs: number | null;
    epochs_run: number | null; best_validation_loss: number | null; input_size: string | null; anchor_min_ratio: number | null;
    augmentation: boolean | null; gpu: string | null; families: CutterFamilies | null} | null;
  test: CutterNumbers | null;
  validation: CutterNumbers | null;
  files: Record<string, boolean>;
}

// GET /api/panel/models and POST /api/panel/models/active (nnslr_tools.model_roles.ModelIndex): the Models tab.

/** The three networks of the pipeline (docs/NEURAL_NETWORK.md). */
export type ModelRole = 'cutter' | 'classifier' | 'decider';

/** Signs of one width found by the cutter. */
export interface WidthRecall {
  /** lt12, 12-20, 20-40, ge40. */
  key: string;
  /** "< 12 px", ... (width of the reviewed box in the frame). */
  label: string;
  truth: number;
  found: number;
  recall: number | null;
}

/** A detector evaluation as the cutter is judged. */
export interface CutterNumbers {
  partition: string;
  /** The score the numbers are at: 0.1 when the report measured it, else the report's own threshold. */
  score_threshold: number | null;
  at_cutter_score: boolean;
  images: number | null;
  truth_boxes: number | null;
  found: number | null;
  false_boxes: number | null;
  recall: number | null;
  precision: number | null;
  /** Boxes passed on to the classifier per frame, and the false ones among them. */
  boxes_per_frame: number | null;
  false_per_frame: number | null;
  widths: WidthRecall[] | null;
  human: {images: number | null; truth_boxes: number | null; found: number | null; recall: number | null} | null;
  /** The model alone, one frame at a time (newer reports only). */
  ms_per_frame: number | null;
  device: string | null;
  input_size: string | null;
  anchor_min_ratio: number | null;
  checkpoint_epoch: number | null;
  /** What the report says the cutter finds (null in older reports). */
  families: CutterFamilies | null;
  /** Recall by family at this score (newer reports): the speed signs, and the other signs for a cutter of every sign. */
  by_family: Record<string, {truth: number; found: number; recall: number | null}> | null;
}

/** An experiment B variant on one partition: whom the sign applies to, and the reading. */
export interface DeciderNumbers {
  boxes: number | null;
  reading_accuracy: number | null;
  ownership_samples: number | null;
  macro_f1: number | null;
  abstain_rate: number | null;
  /** The "Another" road: found (recall) and right when answered (precision); support 0 = no such case. */
  other_road_recall: number | null;
  other_road_precision: number | null;
  other_road_support: number | null;
  other_road_found: number | null;
}

interface ModelBase {
  /** "cutter:2026-10-24_1452/detector", "decider:<run>/experiment-b/crop_geom": stable. */
  id: string;
  run: string;
  part: string;
  /** "Detector 640 × 360", "Reader", "Crop + geometry". */
  label: string;
  /** "training pipeline", "input-size comparison", "experiment B prototype". */
  detail: string;
  /** The time of its weights (*-best.pt). */
  trained_at: string | null;
  /** Its training wrote result.json: the weights no longer change. */
  finished: boolean;
  in_use: boolean;
  /** Files that exist but cannot be read. */
  problems: string[];
  architecture: string | null;
  initialization: string | null;
  best_epoch: number | null;
  epochs: number | null;
  epochs_run: number | null;
}

export interface CutterModel extends ModelBase {
  role: 'cutter';
  /** "320x320"; recorded false: the default, from before the size was recorded. */
  input_size: string;
  input_size_recorded: boolean;
  anchor_min_ratio: number;
  anchor_recorded: boolean;
  augmentation: boolean | null;
  /** What it finds; recorded false: from before it was recorded (the speed signs only). */
  families: CutterFamilies;
  families_recorded: boolean;
  test: CutterNumbers | null;
  validation: CutterNumbers | null;
}

export interface ClassifierModel extends ModelBase {
  role: 'classifier';
  classes: number | null;
  best_validation_accuracy: number | null;
  test: ClassifierNumbers | null;
  /** The classifier with “not a sign” (classifier:<run>/classifier) only; absent for a reader. */
  class_scheme?: string;
  /** The mean of the two below, which chose its best epoch. */
  best_validation_balanced_accuracy?: number | null;
  best_validation_sign_accuracy?: number | null;
  best_validation_rejection?: number | null;
  augmentation?: boolean | null;
  chain?: ChainNumbers | null;
  chain_old_reader?: ChainNumbers | null;
}

export interface DeciderModel extends ModelBase {
  role: 'decider';
  variant: string;
  /** "crop", "box position", "car speed", "scene". */
  inputs: string[] | null;
  /** A smoke run on this many frames per partition, not a measure; null for a real one. */
  smoke_limit: number | null;
  validation: DeciderNumbers | null;
  test: DeciderNumbers | null;
}

export type ModelVersion = CutterModel | ClassifierModel | DeciderModel;

/** The model in use for a role (models/active.json of the data root). */
export interface ActiveModel {
  id: string;
  /** The name signed in; null for an agent with the token or 127.0.0.1 without sign-in. */
  who: string | null;
  via: 'sign-in' | 'token' | 'local' | null;
  at: string | null;
  /** Its files are still on this server. */
  found: boolean;
}

export interface RoleModels {
  role: ModelRole;
  label: string;
  /** Newest first. */
  versions: ModelVersion[];
  /** null: nobody picked one yet (there is no default). */
  active: ActiveModel | null;
}

export interface ModelsPayload {
  roles: RoleModels[];
  /** Why models/active.json cannot be used, if it cannot. */
  active_problem: string | null;
  active_file: string;
  runs_dir_exists: boolean;
  cutter_score_threshold: number;
}

/** POST /api/panel/models/active */
export interface ActiveChosen {
  ok: true;
  role: ModelRole;
  active: Omit<ActiveModel, 'found'>;
}
