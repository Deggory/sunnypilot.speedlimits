// The list of the boxes of the frame in the panel ("Signs"), as data for BoxList.vue: what each one is,
// where it comes from, what the reviewer changed, the boxes that are the same sign twice.
import type {AiAction} from '../common/ai-types';
import {describe, shortName} from './catalog';
import {proposalsOf} from './frames';
import {overlap} from './geometry';
import type {Detection, FrameEdit, FrameRow} from './types';

/** "added by hand", "model · 87%", or what the reviewer changed: "model 40 " + "→ 60". */
export interface Source {
  text: string;
  changed: string | null;
}

export interface BoxItem {
  index: number;
  detectionId: string;
  what: string;
  source: Source;
  /** Placed by "Apply previous" where the sign should be: the reviewer must look at it. */
  check: boolean;
  /** The earlier box this one repeats (same sign drawn twice), or null. */
  duplicateOf: number | null;
}

// The proposal a reviewed box comes from, by detection_id; null for a box added by hand.
function originalOf(row: FrameRow, d: Detection): Detection | null {
  if (!d.detection_id) return null;
  return proposalsOf(row).find((p) => p.detection_id === d.detection_id) ?? null;
}

/** A box of an AI suggestion applied: what the AI did with it. */
const AI_SOURCE: Record<AiAction, string> = {
  keep: 'AI suggestion · kept', correct: 'AI suggestion · corrected', add: 'AI suggestion · added',
  unjudged: 'AI suggestion · not judged',
};

function sourceOf(row: FrameRow, d: Detection): Source {
  if (d.check_position) return {text: 'from ' + d.carried_from + ' · check it', changed: null};
  if (d.carried_from) return {text: 'as in ' + d.carried_from, changed: null};
  if (d.ai_action) return {text: AI_SOURCE[d.ai_action] ?? 'AI suggestion', changed: null};
  const original = originalOf(row, d);
  if (original === null) return {text: 'added by hand', changed: null};
  if (original.family !== d.family || original.value_kph !== d.value_kph) {
    return {text: 'model ' + shortName(original) + ' ', changed: '→ ' + shortName(d)};
  }
  if (original.score === null || original.score === undefined) return {text: 'model', changed: null};
  return {text: 'model · ' + Math.round(Number(original.score) * 100) + '%', changed: null};
}

export function boxItems(row: FrameRow, work: FrameEdit): BoxItem[] {
  return work.detections.map((d, i) => {
    // A box covering most of an earlier one is the same sign drawn twice.
    const earlier = work.detections.slice(0, i).findIndex((other) => overlap(other.bbox_xyxy, d.bbox_xyxy) > 0.6);
    return {
      index: i,
      detectionId: String(d.detection_id || ''),
      what: describe(d),
      source: sourceOf(row, d),
      check: d.check_position === true,
      duplicateOf: earlier >= 0 ? earlier : null,
    };
  });
}

export function boxCountText(count: number): string {
  if (count === 0) return '';
  return count === 1 ? '1 box' : count + ' boxes';
}
