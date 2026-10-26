// The Drive view of Results (DriveView.vue): a drive of the test (or validation) replayed as in the car. Frame by
// frame, in the order of the drive, the speed limit the networks would show next to the limit of the signs a person
// reviewed, each worked out the same way, by the Swiss rules:
// - a speed sign (also a zone or a display showing a number) sets its number until the next one;
// - an end of limit (the number crossed out) brings back the general limit of the road: 50 in a town, 80 outside,
//   100 on an expressway, 120 on a motorway (80 when the road is not known yet: a road outside towns);
// - the signs of the road set its general limit: town entry 50, town exit 80, motorway 120, expressway 100, and their
//   ends 80.
// Only what the chain report keeps is used: the cutter's boxes from a score and what the classifier called each, and the
// reviewed signs with their labels. Which road a sign applies to (the decider) is not in the report: every sign read
// counts, as if it were ours.
import type {PixelBox, WatchFrame} from '../common/panel-types';
import {NOT_A_SIGN} from './watch';

/** A limit in km/h; null before any sign. */
export type Limit = number | null;
export type Road = 'town' | 'rural' | 'expressway' | 'motorway';
/** The general limit of each road (Swiss traffic rules, art. 4a VRV). */
export const GENERAL: Record<Road, number> = {town: 50, rural: 80, expressway: 100, motorway: 120};
/** The signs of the road, by their kind of other sign. */
const ROAD_SIGNS: Record<string, Road> = {town_entry: 'town', town_exit: 'rural', motorway_start: 'motorway',
  motorway_end: 'rural', expressway_start: 'expressway', expressway_end: 'rural'};
/** What a sign does to the limit: sets a number, ends the limit (the general one again), or changes the road. */
export type Effect = {kind: 'limit'; value: number} | {kind: 'end'} | {kind: 'road'; road: Road};
/** A frame against the signs: the limit right, wrong, or nothing to judge yet (neither has a limit). */
export type DriveState = 'right' | 'wrong' | 'none';

/** A box drawn on the frame: a sign the networks read (right, wrong, or on nothing), or a reviewed sign they missed. */
export interface DriveMark {
  box: PixelBox;
  text: string;
  tone: 'right' | 'wrong' | 'false' | 'missed';
}

export interface DriveFrame {
  model: Limit;
  truth: Limit;
  /** The limit shown is the general one of the road (after an end of limit or a sign of the road), not a number read. */
  modelGeneral: boolean;
  truthGeneral: boolean;
  state: DriveState;
  /** A sign that changes the limit read on this frame (the networks), or reviewed on it (the truth). */
  modelSign: boolean;
  truthSign: boolean;
  marks: DriveMark[];
}

export interface DriveSummary {
  /** Frames with a limit to judge, and those where the networks show the right one. */
  judged: number;
  right: number;
  /** Signs that change the limit, reviewed on the drive: read right, read as another, missed; and read on nothing. */
  signs: number;
  read: number;
  misread: number;
  missed: number;
  invented: number;
}

/** What a label does to the limit, or null when it does nothing (another sign, a speed sign whose number was not read,
 * "not a sign"). */
export function effectOf(label: string | null | undefined): Effect | null {
  if (!label || label === NOT_A_SIGN) return null;
  const [family = '', value = ''] = label.split(':');
  if (family === 'cancellation') return {kind: 'end'};
  if (family === 'other_sign') {
    const road = ROAD_SIGNS[value];
    return road ? {kind: 'road', road} : null;
  }
  if (family !== 'maximum_speed' && family !== 'zone' && family !== 'variable_display') return null;
  const number = Number(value);
  return Number.isInteger(number) && number > 0 ? {kind: 'limit', value: number} : null;
}

const same = (a: Effect | null, b: Effect | null): boolean => JSON.stringify(a) === JSON.stringify(b);

/** The words on a box: "60", "END", "town", "end of road", "motorway"… */
export function effectText(effect: Effect): string {
  if (effect.kind === 'limit') return String(effect.value);
  if (effect.kind === 'end') return 'END';
  return {town: 'town', rural: 'end of road', expressway: 'expressway', motorway: 'motorway'}[effect.road];
}

/** "80", "–". */
export function limitText(limit: Limit): string {
  return limit === null ? '–' : String(limit);
}

const area = (box: PixelBox): number => Math.max(0, box[2] - box[0]) * Math.max(0, box[3] - box[1]);
/** How much of the smaller of two boxes the other covers (0-1). */
function cover(a: PixelBox, b: PixelBox): number {
  const width = Math.min(a[2], b[2]) - Math.max(a[0], b[0]);
  const height = Math.min(a[3], b[3]) - Math.max(a[1], b[1]);
  const small = Math.min(area(a), area(b));
  return width <= 0 || height <= 0 || small <= 0 ? 0 : width * height / small;
}

/** The limit as a car keeps it: the road it is on, the limit shown, and whether that is the road's general limit. */
interface Keeper {road: Road | null; limit: Limit; general: boolean}
interface Seen {box: PixelBox; effect: Effect}

/** A frame's signs applied to the limit kept: first a sign of the road (the largest), then the nearest (largest) of a
 * number and an end of limit. Whether there was any. */
function apply(keeper: Keeper, signs: Seen[]): boolean {
  const largest = (kinds: Effect['kind'][]): Seen | null =>
    signs.filter((item) => kinds.includes(item.effect.kind)).sort((a, b) => area(b.box) - area(a.box))[0] ?? null;
  const road = largest(['road'])?.effect;
  if (road?.kind === 'road') {
    keeper.road = road.road;
    keeper.limit = GENERAL[road.road];
    keeper.general = true;
  }
  const sign = largest(['limit', 'end'])?.effect;
  if (sign?.kind === 'limit') {
    keeper.limit = sign.value;
    keeper.general = false;
  } else if (sign?.kind === 'end') {
    keeper.limit = GENERAL[keeper.road ?? 'rural'];
    keeper.general = true;
  }
  return signs.length > 0;
}

/** The drive replayed at a score of the cutter: every frame's limits and marks, and the drive at a glance. */
export function replayDrive(frames: readonly WatchFrame[], score: number): {frames: DriveFrame[]; summary: DriveSummary} {
  const model: Keeper = {road: null, limit: null, general: false};
  const truth: Keeper = {road: null, limit: null, general: false};
  const summary: DriveSummary = {judged: 0, right: 0, signs: 0, read: 0, misread: 0, missed: 0, invented: 0};
  const out = frames.map((frame): DriveFrame => {
    const kept = frame.boxes.filter((item) => item.score >= score && item.label && item.label !== NOT_A_SIGN);
    const marks: DriveMark[] = [];
    const read: Seen[] = [];
    for (const item of kept) {
      const effect = effectOf(item.label);
      // Only what changes the limit is drawn and counted (another sign never does).
      if (effect === null) continue;
      // A second box of the cutter on a reviewed sign (the report matches one box per sign) reads that sign too.
      const on = item.match >= 0 ? item.match : frame.truth.findIndex((sign) => cover(sign.box, item.box) >= 0.5);
      const right = on >= 0 && same(effect, effectOf(frame.truth[on]?.label));
      read.push({box: item.box, effect});
      if (item.match < 0 && on >= 0 && right) continue;
      marks.push({box: item.box, text: effectText(effect), tone: on < 0 ? 'false' : right ? 'right' : 'wrong'});
      if (on < 0) summary.invented += 1;
    }
    const reviewed: Seen[] = [];
    frame.truth.forEach((item, index) => {
      const effect = effectOf(item.label);
      if (effect === null) return;
      reviewed.push({box: item.box, effect});
      summary.signs += 1;
      const answer = kept.find((box) => box.match === index);
      if (!answer) {
        summary.missed += 1;
        marks.push({box: item.box, text: effectText(effect), tone: 'missed'});
      } else if (same(effectOf(answer.label), effect)) {
        summary.read += 1;
      } else {
        summary.misread += 1;
      }
    });
    const modelSign = apply(model, read);
    const truthSign = apply(truth, reviewed);
    const state: DriveState = model.limit === null && truth.limit === null ? 'none'
      : model.limit === truth.limit ? 'right' : 'wrong';
    if (state !== 'none') {
      summary.judged += 1;
      if (state === 'right') summary.right += 1;
    }
    return {model: model.limit, truth: truth.limit, modelGeneral: model.general, truthGeneral: truth.general, state,
            modelSign, truthSign, marks};
  });
  return {frames: out, summary};
}

/** The timeline: runs of frames in the same state, as [first, count, state]. */
export function stateRuns(frames: readonly DriveFrame[]): [number, number, DriveState][] {
  const runs: [number, number, DriveState][] = [];
  frames.forEach((frame, index) => {
    const last = runs[runs.length - 1];
    if (last && last[2] === frame.state) last[1] += 1;
    else runs.push([index, 1, frame.state]);
  });
  return runs;
}
