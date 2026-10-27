// Data › Roads: the kinds of road with their words, and a drive opened (RoadDrive.vue) as runs of frames, the frames
// still to check and the time of a frame.
import type {RoadFrame, RoadType} from '../common/panel-types';

/** The kinds of road in the order of the legend, each with its colour (control.html .rt-town ...). */
export const ROAD_TYPES: [key: RoadType | 'unknown', label: string][] = [['town', 'Town'], ['rural', 'Rural'],
  ['expressway', 'Expressway'], ['motorway', 'Motorway'], ['unknown', 'Unknown']];
export const ROAD_LABEL = Object.fromEntries(ROAD_TYPES) as Record<RoadType | 'unknown', string>;

/** A run of frames of one kind, the map's or a person's: what the strip draws and what a correction covers. */
export interface RoadRun {
  first: number;
  count: number;
  kind: RoadType | 'unknown';
  reviewed: boolean;
}
export function roadRuns(frames: readonly RoadFrame[]): RoadRun[] {
  const runs: RoadRun[] = [];
  frames.forEach((frame, index) => {
    const kind = frame.road_type ?? 'unknown';
    const last = runs[runs.length - 1];
    if (last && last.kind === kind && last.reviewed === frame.reviewed) last.count += 1;
    else runs.push({first: index, count: 1, kind, reviewed: frame.reviewed});
  });
  return runs;
}

/** A frame a person should look at: no person set it, and the map says nothing or is often wrong there. */
export function toCheck(frame: RoadFrame): boolean {
  return !frame.reviewed && (frame.road_type === null || !frame.sure);
}

/** The first frame of the next stretch to check after the one at `from` (from the start again after the last); null when
 * none is left. */
export function nextToCheck(frames: readonly RoadFrame[], from: number): number | null {
  let first: number | null = null;
  for (let index = 0; index < frames.length; index++) {
    const frame = frames[index];
    if (!frame || !toCheck(frame)) continue;
    const start = index === 0 || !toCheck(frames[index - 1] as RoadFrame);
    if (!start) continue;
    if (index > from) return index;
    first ??= index;
  }
  return first !== null && first !== from ? first : null;
}

/** Seconds from the start of the drive of a frame "segment/image": a segment a minute, a frame a second. */
export function frameSeconds(key: string): number | null {
  const [segment, image] = key.split('/').map(Number);
  return segment === undefined || image === undefined || Number.isNaN(segment) || Number.isNaN(image) ? null : segment * 60 + image;
}

/** The time of a frame: the clock ("08:14:32", the browser's time zone) when the start of the drive is known, else how
 * far into the drive ("12:05", "1:02:05"). */
export function frameTime(key: string, start: string | null): string {
  const seconds = frameSeconds(key);
  if (seconds === null) return '';
  const pad = (value: number): string => String(value).padStart(2, '0');
  const begun = start ? Date.parse(start) : NaN;
  if (!Number.isNaN(begun)) {
    const date = new Date(begun + seconds * 1000);
    return pad(date.getHours()) + ':' + pad(date.getMinutes()) + ':' + pad(date.getSeconds());
  }
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor(seconds / 60) % 60;
  return (hours ? hours + ':' + pad(minutes) : String(minutes)) + ':' + pad(seconds % 60);
}
