// "Apply previous": the boxes of the last frame confirmed before this one, carried onto the frame on screen by the server
// (review_service.apply_previous). A box of this frame that is the same sign takes the previous labels; a sign
// not found here gets a box where it should be, marked "check": Confirm refuses the frame until the reviewer
// has looked at each one (moved it onto the sign, pressed "Here", or deleted it). Pressing A is that look: once the fit
// has run on the carried boxes without an error, they count as checked and the frame stays on screen, not saved, to be
// looked at, its first sign selected so that its zoom opens inside the photo; A again on it saves it and goes on, as
// Enter does. The boxes of the frame before show at once, the first one zoomed, while the server carries them.
// Shift+A, "Apply next" (#105), is the same the other way: the boxes of the first frame confirmed after this one,
// carried back onto it (direction "next"), and Shift+A again saves it and goes to the frame before, through the frames
// of the video as ← does (confirmFrameBack, video.ts).
import {fitLocal} from './ai';
import {confirmFrame, confirmFrameBack} from './confirm';
import {refuseChange} from './busy';
import {nextTick, ref} from 'vue';
import {messageOf} from '../common/api';
import {clone} from './dom';
import {changed, isPlacedByHand, markPlacedByHand} from './edits';
import {currentRow, isConfirmed, workOf} from './frames';
import {revealEditor} from './scroll';
import {call} from './server';
import {anchorIndex, indexOf, isVirtual, report, state} from './store';
import type {CarryAnswer, FrameEdit, FrameRow} from './types';

/** A: from the frame before; Shift+A: from the frame after, going back. */
export type CarryWay = 'previous' | 'next';

/** The frame just before in the timeline (this segment or the end of the one before), or null. For a frame of the
 * video not saved (video.ts): the frame of the report just before it. */
function rowBefore(row: FrameRow): FrameRow | null {
  const key = row.frame_key;
  const before = report.rows[isVirtual(key) ? anchorIndex(key) : indexOf(key) - 1];
  if (!before || before.segment_index < row.segment_index - 1) return null;
  return before;
}

/** The index in the report of the frame just after (a frame of the video not saved: of the report just after it). */
function indexAfter(row: FrameRow): number {
  const key = row.frame_key;
  return isVirtual(key) ? anchorIndex(key) + 1 : indexOf(key) + 1;
}

/** The frame just after in the timeline (this segment or the start of the one after), or null. */
function rowAfter(row: FrameRow): FrameRow | null {
  const after = report.rows[indexAfter(row)];
  if (!after || after.segment_index > row.segment_index + 1) return null;
  return after;
}

/** The frame Apply next (Shift+A) takes the boxes from: the first one confirmed after this frame (or confirmed here
 * and still on its way to the server), in this segment or the one after. Null when none is. As
 * review_service.apply_previous with direction "next". */
export function nextRow(row: FrameRow): FrameRow | null {
  for (let index = indexAfter(row); index < report.rows.length; index += 1) {
    const after = report.rows[index];
    if (!after || after.segment_index > row.segment_index + 1) return null;
    if (isConfirmed(after.frame_key) || savingConfirmed(after.frame_key)) return after;
  }
  return null;
}

function sourceRow(row: FrameRow, way: CarryWay): FrameRow | null {
  return way === 'next' ? nextRow(row) : previousRow(row);
}

/** "Apply previous: …" / "Apply next: …", before a reason or an error. */
function nameOf(way: CarryWay): string {
  return way === 'next' ? 'Apply next: ' : 'Apply previous: ';
}

/** The frame Apply previous takes the boxes from: the last one confirmed before this frame (or confirmed here and still
 * on its way to the server), in this segment or the one before; the reviewer may have gone some frames on without
 * confirming those between (Cristian, 14 Nov 2026). Null when none is. As review_service.apply_previous. */
export function previousRow(row: FrameRow): FrameRow | null {
  const key = row.frame_key;
  for (let index = isVirtual(key) ? anchorIndex(key) : indexOf(key) - 1; index >= 0; index -= 1) {
    const before = report.rows[index];
    if (!before || before.segment_index < row.segment_index - 1) return null;
    if (isConfirmed(before.frame_key) || savingConfirmed(before.frame_key)) return before;
  }
  return null;
}

export interface CarryState {
  /** Shown at all: only with the review server. */
  shown: boolean;
  enabled: boolean;
  /** Why it is off, or what it does (the button's tooltip). */
  title: string;
}

/** The frame on which A carried the boxes: A again on it saves it and goes on (confirmFrame). With carriedWay
 * "next" Shift+A carried them: Shift+A again saves it and goes back. */
export const carried = ref<string | null>(null);
export const carriedWay = ref<CarryWay>('previous');

/** The frame on screen carried by this key, waiting for the same key again to be saved. */
function carriedHere(key: string, way: CarryWay): boolean {
  return carried.value === key && carriedWay.value === way;
}

export function carryState(row: FrameRow, way: CarryWay = 'previous'): CarryState {
  if (!report.apiBase) return {shown: false, enabled: false, title: ''};
  if (state.inFlight.has(row.frame_key)) return {shown: true, enabled: false, title: 'Saving…'};
  if (way === 'next') {
    if (rowAfter(row) === null) return {shown: true, enabled: false, title: 'No frame just after this one.'};
    const after = nextRow(row);
    if (after === null) return {shown: true, enabled: false, title: 'No frame after this one is confirmed yet.'};
    return {
      shown: true,
      enabled: true,
      title: carriedHere(row.frame_key, way) ? 'Save this frame and go back to the frame before (Shift+A)'
        : 'The signs confirmed on ' + after.frame_key + ', carried back onto this frame and fitted; Shift+A again '
          + 'saves and goes back (Shift+A)',
    };
  }
  if (rowBefore(row) === null) return {shown: true, enabled: false, title: 'No frame just before this one.'};
  const before = previousRow(row);
  if (before === null) return {shown: true, enabled: false, title: 'No frame before this one is confirmed yet.'};
  return {
    shown: true,
    enabled: true,
    title: carriedHere(row.frame_key, way) ? 'Save this frame and go on, as Enter (A)'
      : 'The signs confirmed on ' + before.frame_key + ', carried onto this frame and fitted; A again saves and goes on '
        + '(A; Shift+A: from the frame after, going back)',
  };
}

/** Confirmed on this page and still on its way to the server: Apply previous sends its boxes along (applyPrevious). */
function savingConfirmed(key: string): boolean {
  return state.inFlight.has(key) && state.edits[key]?.review_state === 'reviewed';
}

// The frame whose boxes Apply previous is carrying (and which way), and A pressed again meanwhile: it saves once they
// are there.
let carrying: string | null = null;
let carryingWay: CarryWay = 'previous';
let saveAfterCarry = false;
// The frame whose boxes were carried last, and from where: the boxes to check say it (blockUnchecked).
let lastCarry: {key: string; way: CarryWay} | null = null;
// A or Shift+A held down that saved a frame with no frame after it (A) or before it (Shift+A) to go to (the end or the
// start of the drive or of the filter): the repeats of the key stop there, instead of carrying and saving that frame
// again and again; a new press works.
let stoppedAt: string | null = null;

/** The boxes waiting for the reviewer to check their position, by index. */
export function uncheckedIndices(work: FrameEdit): number[] {
  const found: number[] = [];
  work.detections.forEach((d, i) => {
    if (d.check_position) found.push(i);
  });
  return found;
}

/** A: the boxes of the frame before; A again saves and goes on. ``repeat``: a repeat of the key held down. */
export function applyPrevious(repeat = false): Promise<void> {
  return applyFrom('previous', repeat);
}

/** Shift+A (#105): the boxes of the frame after, carried back; Shift+A again saves and goes to the frame before.
 * ``repeat``: a repeat of the key held down. */
export function applyNext(repeat = false): Promise<void> {
  return applyFrom('next', repeat);
}

/** Shift+A again: saved, and back to the frame before; none (the page stays here): the key's repeats stop here. */
async function saveAndGoBack(key: string): Promise<void> {
  await confirmFrameBack();
  if (state.currentKey === key) stoppedAt = key;
}

/** A again: saved, and on to the next frame; none (the page stays here): the key's repeats stop here. */
async function saveAndGoOn(key: string): Promise<void> {
  await confirmFrame();
  if (state.currentKey === key) stoppedAt = key;
}

async function applyFrom(way: CarryWay, repeat = false): Promise<void> {
  const row = currentRow();
  const apiBase = report.apiBase;
  if (!row || !apiBase) return;
  if (!repeat) stoppedAt = null;
  else if (stoppedAt === row.frame_key) return;
  // A, A pressed quickly: the second one saves once the boxes are carried. The other key meanwhile does nothing.
  if (carrying === row.frame_key) {
    if (carryingWay === way) saveAfterCarry = true;
    return;
  }
  if (refuseChange()) return;
  const carry = carryState(row, way);
  if (!carry.enabled) {
    // Why not, said where it is seen (a tooltip never shows on a touch screen).
    state.notice = {key: row.frame_key, text: nameOf(way) + carry.title, error: false, warn: true};
    return;
  }
  const key = row.frame_key;
  // Carried by this same key: saved, and on (A) or back (Shift+A). Carried by the other key, it carries anew.
  if (carriedHere(key, way)) {
    carried.value = null;
    await (way === 'next' ? saveAndGoBack(key) : saveAndGoOn(key));
    return;
  }
  const work = workOf(row);
  const byHand = new Set(work.detections.filter(isPlacedByHand).map((d) => JSON.stringify(d.bbox_xyxy)));
  // The frame before (after, with Shift+A), confirmed and still on its way to the server: its boxes go with the
  // request, nothing waits.
  const before = sourceRow(row, way);
  const previous = before !== null && savingConfirmed(before.frame_key)
    ? {frame_key: before.frame_key, detections: workOf(before).detections} : undefined;
  carrying = key;
  carryingWay = way;
  saveAfterCarry = false;
  if (carried.value === key) carried.value = null;
  const sent = work.detections;
  let answered = false;
  // At once, before the server answers: the boxes of the frame before shown here, the first selected so that its zoom
  // opens; the server's carried boxes take their place.
  const shown = before !== null ? workOf(before).detections : [];
  if (shown.length > 0) {
    work.detections = shown.map((d) => ({...clone(d), check_position: true}));
    state.selected = 0;
    state.selectedByUser = false;
    state.drawing = false;
  }
  try {
    // Plain A sends what it always sent; Shift+A adds the direction.
    const request = way === 'next' ? {run_id: report.runId, detections: sent, previous, direction: 'next'}
      : {run_id: report.runId, detections: sent, previous};
    const answer = await call<CarryAnswer>('POST', apiBase + '/frame/' + key + '/apply-previous', request);
    // The reviewer may have moved on while the server answered.
    if (state.currentKey !== key) {
      work.detections = sent;
      return;
    }
    // A server not restarted since Shift+A came ignores the direction and carries from the frame before: none of
    // those boxes is taken, and nothing can be saved from them.
    if (way === 'next' && answer.direction !== 'next') {
      throw new Error('the server does not know it yet (it carried from the frame before): restart it first.');
    }
    // Every box comes back anew: one placed by hand that the carry kept where it was is still placed by hand.
    for (const d of answer.detections) if (byHand.has(JSON.stringify(d.bbox_xyxy))) markPlacedByHand(d, true);
    work.detections = answer.detections;
    answered = true;
    lastCarry = {key, way};
    // The first box to check is selected, else the first sign carried: its zoom opens inside the photo.
    const first = uncheckedIndices(work)[0];
    state.selected = first ?? (work.detections.length > 0 ? 0 : -1);
    state.selectedByUser = first !== undefined;
    state.drawing = false;
    changed(row);
    if (first !== undefined) void nextTick(revealEditor);
    const message = {key, text: answer.message, error: false, warn: answer.to_check.length > 0};
    state.notice = message;
    // The carried boxes go straight onto their signs, as C does, but those placed by hand stay where they were put;
    // its message after the one of the carry.
    await fitLocal('carried');
    const fit = state.notice;
    if (fit && fit !== message && fit.key === key) {
      state.notice = {...fit, text: message.text + ' ' + fit.text, warn: message.warn || Boolean(fit.warn)};
    }
    // The fit without error: every carried box counts as checked (a box the fit kept stays where the carry put it) and
    // the frame stays on screen, not saved, with no message: A again (or Enter) saves it and goes on.
    if (state.currentKey !== key || state.notice?.error) return;
    // Except a sign with a plate of distance found here without its plate (carry_over._carry, #98): no fit moves a box
    // with a plate, so it stays to check, and A again stops on it (confirmFrame, blockUnchecked).
    for (const d of work.detections) if (d.supplementary_panel !== 'distance') delete d.check_position;
    const plateLeft = uncheckedIndices(work);
    if (plateLeft.length > 0) {
      state.selected = plateLeft[0]!;
      state.selectedByUser = true;
      state.notice = {key, text: (plateLeft.length === 1 ? 'Box ' : 'Boxes ') + plateLeft.map((i) => i + 1).join(', ')
        + ', a sign with a distance plate, to check: draw ' + (plateLeft.length === 1 ? 'it' : 'each')
        + ' over the sign and the plate, then press “Here”.', error: false, warn: true};
      void nextTick(revealEditor);
    } else {
      if (state.selected < 0 || state.selected >= work.detections.length) state.selected = work.detections.length > 0 ? 0 : -1;
      state.notice = null;
    }
    carried.value = key;
    carriedWay.value = way;
    if (saveAfterCarry) {
      carrying = null;
      carried.value = null;
      await (way === 'next' ? saveAndGoBack(key) : saveAndGoOn(key));
    }
  } catch (error) {
    if (!answered) work.detections = sent;
    state.notice = {key, text: nameOf(way) + messageOf(error), error: true};
  } finally {
    if (carrying === key) carrying = null;
    saveAfterCarry = false;
  }
}

/** "Here": the reviewer has looked, the carried box is on its sign. */
export function positionChecked(index: number): void {
  const row = currentRow();
  if (!row || refuseChange()) return;
  const d = workOf(row).detections[index];
  if (!d || !d.check_position) return;
  delete d.check_position;
  changed(row);
}

/** Confirm with boxes still to check: the first one is selected and the reason shown. False: none left. */
export function blockUnchecked(row: FrameRow): boolean {
  const work = workOf(row);
  const left = uncheckedIndices(work);
  const first = left[0];
  if (first === undefined) return false;
  const numbers = left.map((i) => i + 1).join(', ');
  const from = lastCarry !== null && lastCarry.key === row.frame_key && lastCarry.way === 'next' ? 'next' : 'previous';
  const text = (left.length === 1 ? 'Check box ' + numbers : 'Check boxes ' + numbers)
    + ', carried from the ' + from + ' frame: move ' + (left.length === 1 ? 'it' : 'each')
    + ' onto the sign and press “Here”, or delete it if the sign is gone.';
  state.selected = first;
  state.selectedByUser = true;
  state.notice = {key: row.frame_key, text, error: true};
  state.lastSave = {key: row.frame_key, result: 'need-check', revision: work.revision || 0};
  void nextTick(revealEditor);
  return true;
}

