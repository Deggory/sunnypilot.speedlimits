// The data of the review page: the payload pages.render_review embeds in the page, the answers of the review
// API (nnslr_tools.review: GET /api/route/<id>/reviews, POST /api/route/<id>/frame/<segment>/<image>) and
// the page's own working copy of each frame, kept in localStorage.

/** x1, y1, x2, y2 in pixels of the full image. */
export type Box = [number, number, number, number];

/** One box: a proposal of the model (report row) or the reviewer's version of it. Proposals carry more
 * fields (evidence_source...): they are kept as they are and sent back unchanged. */
export interface Detection {
  bbox_xyxy: Box;
  family: string;
  value_kph?: number | null | undefined;
  value_state?: 'unknown' | 'reviewed';
  score?: number | null;
  model_label?: string | null;
  class_id?: number | null;
  review_state?: string;
  applicability?: string;
  supplementary_panel?: string;
  visibility?: string;
  occlusion?: string;
  temporary?: boolean | null;
  variable_display?: boolean | null;
  notes?: string | null;
  detection_id?: string;
}

/** One frame of the pre-annotation report. The rows carry more fields (hashes, models...): the JSONL export
 * copies them as they are. */
export interface FrameRow {
  frame_key: string;
  segment_index: number;
  output_index: number;
  width: number;
  height: number;
  /** Relative to the page: images/<n>.jpg */
  preview_path: string;
  detections?: Detection[];
  site_group?: string | null;
  encounter_id?: string | null;
}

/** One sign seen by the second reading (nnslr_tools.review_hints). */
export interface Hint {
  type: string;
  value_kph?: number | null;
  box: Box;
  confidence: string;
  /** "A" or "B": the two passes of the second reading. */
  pass?: string;
  side?: string;
  note?: string;
  source?: string;
}

export interface HintStats {
  files: number;
  lines: number;
  frames: number;
  signs: number;
  skipped_files: number;
  skipped_lines: number;
  skipped_signs: number;
  unknown_frames: number;
}

/** Why a frame is suspicious: rule a..d of review_hints.suspicious_frames, and the Italian text. */
export interface SuspiciousReason {
  rule: string;
  text: string;
}

export type Suspicious = Record<string, SuspiciousReason[]>;

export interface RouteLink {
  route_id: string;
  url: string;
}

/** The JSON in <script id="payload">. server is false in the static index.html of write_review (offline). */
export interface Payload {
  run_id: string;
  route_id: string;
  server: boolean;
  frames: FrameRow[];
  hints: Record<string, Hint[]>;
  hint_stats: HintStats;
  suspicious: Suspicious;
  routes: RouteLink[];
}

/** One saved frame as the server sends it (review_db.public_review). */
export interface ServerReview {
  route_id: string;
  run_id: string;
  frame_key: string;
  confirmed: boolean;
  revision: number;
  reviewer: string | null;
  reviewer_kind: 'human' | 'ai';
  /** web, rest or mcp. */
  channel: string;
  updated_at: string;
  review_timestamp: string | null;
  detections: Detection[];
  site_group: string | null;
  encounter_id: string | null;
}

/** GET /api/route/<id>/reviews */
export interface ReviewsAnswer {
  route_id: string;
  run_id: string;
  frame_count: number;
  confirmed_count: number;
  reviews: ServerReview[];
  suspicious: Suspicious;
}

/** POST /api/route/<id>/frame/<key>: a confirmation or a withdrawal stored. */
export interface SaveAnswer {
  ok: true;
  review: ServerReview;
  confirmed_count: number;
  suspicious: Suspicious;
}

/** An error answer of the API (400, 403, 409 revision_conflict or stale_report...). */
export interface Problem {
  ok?: false;
  error?: string;
  message?: string;
}

/** The last confirmed version of a frame: on the server (the whole review), or offline the confirmation made
 * in this browser (detections, groups, reviewer and time only). */
export type ConfirmedCopy = Partial<ServerReview>;

/** The page's working copy of a frame, kept in localStorage: the proposals until the reviewer changes or
 * confirms it. Old copies in a browser may miss any of the optional fields. */
export interface FrameEdit {
  review_state: 'pending' | 'reviewed';
  detections: Detection[];
  /** false: changed here, or confirmed here and not yet on the server. */
  saved?: boolean | undefined;
  /** The server revision these edits are based on (0: not on the server). */
  revision?: number | undefined;
  /** Counts the edits, to tell whether the frame changed while it was being saved. */
  version?: number | undefined;
  error?: string | null | undefined;
  /** false after an error the automatic retry should not repeat (4xx). */
  retry?: boolean | undefined;
  reviewer?: string | null | undefined;
  reviewer_kind?: 'human' | 'ai' | undefined;
  channel?: string | undefined;
  review_timestamp?: string | null | undefined;
  site_group?: string | null | undefined;
  encounter_id?: string | null | undefined;
  server_confirmed?: boolean | undefined;
  base?: ConfirmedCopy | null | undefined;
}

/** Which frames the arrows walk through: the top filters, the AI one and the doubt counters. */
export type Filter = 'todo' | 'suspicious' | 'proposals' | 'all' | 'ai' | 'unknown' | 'unreadable' | 'road';
export type QualityFilter = 'unknown' | 'unreadable' | 'road';

/** todo, dirty (edited, not confirmed), saving, error (confirmed, not on the server yet) or done. */
export type FrameStatus = 'todo' | 'dirty' | 'saving' | 'error' | 'done';

/** body[data-server] for whoever drives the page. */
export type ServerState = 'offline' | 'loading' | 'ready' | 'stale' | 'error';

export type SaveResult = 'saving' | 'ok' | 'error' | 'unchanged' | 'need-reviewer' | 'withdrawn';

/** The outcome of the last save, as body data attributes. */
export interface LastSave {
  key: string;
  result: SaveResult;
  revision: number;
}

/** A message for one frame on the status line (and the phone bar when it is an error). */
export interface Notice {
  key: string;
  text: string;
  error: boolean;
  warn?: boolean;
}

export type Point = [number, number];
export type Corner = 'nw' | 'ne' | 'sw' | 'se';

/** The two pictures a pointer can work on: the photo and the zoom of the selected box. */
export type Surface = 'scene' | 'zoom';

/** A pointer drag on the photo or on the zoom: drawing a new box, or moving a corner of the selected one. */
export type Drag =
  | {kind: 'draw'; svg: Surface; start: Point}
  | {kind: 'resize'; svg: Surface; corner: Corner; moved: boolean};
