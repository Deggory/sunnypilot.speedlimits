// The AI reviewer: the providers of Settings (GET /api/settings/ai, nnslr_tools.llm_providers), whether this
// browser may ask one (GET /api/ai/state), a suggestion for one frame (POST …/ai-suggest, nnslr_tools.ai_review)
// and the measures of Evaluate (GET /api/panel/ai-reviewers). No answer ever holds a key: only has_key.
import type {Box, Detection} from '../review/types';

/** POST /api/settings/ai/test: one frame asked of the model. ok = answered in JSON and not blind. */
export interface AiTestResult {
  ok: boolean;
  answered_json: boolean;
  /** null: no reviewed speed sign to prove it. */
  sees_images: boolean | null;
  seconds: number | null;
  message: string;
  frame: {route_id: string; frame_key: string} | null;
  at: string;
}

/** A provider as the browser sees it: never its key. */
export interface AiProvider {
  id: string;
  name: string;
  base_url: string;
  model: string;
  has_key: boolean;
  is_reviewer: boolean;
  /** What its saves are recorded as: ai/<id>/<model>. */
  reviewer_name: string;
  updated_at: string;
  last_test: AiTestResult | null;
}

export interface AiSettings {
  providers: AiProvider[];
  reviewer: string | null;
  /** A signed-in browser or the site on 127.0.0.1: the review token only reads. */
  can_edit: boolean;
}

/** POST /api/settings/ai/providers. key: absent keeps the saved one, "" removes it, any other text sets it. */
export interface AiProviderBody {
  id?: string;
  name: string;
  base_url: string;
  model: string;
  key?: string;
}

/** POST /api/settings/ai/models: what the server lists at <base_url>/models. */
export interface AiModel {
  id: string;
  /** null: the server does not say. */
  sees_images: boolean | null;
}

/** GET /api/ai/state */
export interface AiState {
  available: boolean;
  /** This browser may ask (signed in, or the site on 127.0.0.1). */
  can_ask?: boolean;
  /** Why it cannot be asked ('' when it can). */
  reason: string;
  provider: {id: string; name: string; model: string; reviewer_name: string} | null;
}

/** What the AI did with a box: kept, corrected, added; unjudged: a box sent that got no verdict. */
export type AiAction = 'keep' | 'correct' | 'add' | 'unjudged';

/** POST /api/route/<r>/frame/<s>/<i>/ai-suggest: nothing is saved; the boxes in pixels of the full frame. */
export interface AiSuggestion {
  ok: true;
  route_id: string;
  frame_key: string;
  run_id: string;
  provider: {id: string; name: string; model: string};
  reviewer: string;
  seconds: number;
  /** false: some boxes sent were not judged. */
  complete: boolean;
  no_sign: boolean;
  detections: (Detection & {ai_action: AiAction})[];
  /** The boxes sent that the AI would delete; number is 1-based, as in the list sent. */
  deleted: {number: number; detection_id: string; family: string; value_kph: number | null; bbox_xyxy: Box; notes: string}[];
  comment: string;
  warnings: string[];
  message: string;
}

export interface AiCounts {
  truth: number;
  found: number;
  found_share: number | null;
}

/** One measure of an AI reviewer on the frames a person reviewed (nnslr measure-ai-reviewer). */
export interface AiReviewerReport {
  task: 'ai_reviewer';
  provider_id: string;
  provider_name: string;
  model: string;
  reviewer: string;
  limit: number;
  frames: number;
  frames_measured: number;
  frames_failed: number;
  frames_incomplete: number;
  failures: {route_id: string; frame_key: string; reason: string; message: string}[];
  sample_sha256: string;
  seconds_per_frame: number | null;
  measured_at: string;
  no_sign: {truth: number; agreed: number; agreed_share: number | null; said_no_sign_wrongly: number};
  speed_signs: AiCounts & {read_right: number; kind_right: number; road_right: number; read_right_share: number | null;
    kind_right_share: number | null; road_right_share: number | null};
  other_signs: AiCounts & {kind_right: number; kind_right_share: number | null};
  every_sign: AiCounts;
  ai_boxes_per_frame: number | null;
  false_per_frame: number | null;
}

/** GET /api/panel/ai-reviewers */
export interface AiReviewersPayload extends AiSettings {
  reports: AiReviewerReport[];
}
