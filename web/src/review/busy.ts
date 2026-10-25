// A drive Propose again works on (nnslr_tools.reproposing) is read only: nothing on it is confirmed or changed until
// its new proposals are in (the server refuses a save too). The drive at work shows the new proposals made so far,
// live: every LIVE_MS the page asks for the frames proposed since, adds them to the timeline and, following, shows
// the newest. A drive waiting its turn shows its current report and asks every INTERVAL_MS. When the drive is done
// the page says so: a reload shows the new proposals, to review.
import {ref} from 'vue';
import {currentRow} from './frames';
import {openFrame} from './navigation';
import {call} from './server';
import {appendRows, report, state} from './store';
import type {Busy, FrameRow} from './types';

const INTERVAL_MS = 15000;
const LIVE_MS = 4000;
/** Following: each frame proposed shows as it comes; going to another frame stops it (followed: the one shown). */
export const following = ref(false);
let followed: string | null = null;
const READ_ONLY_TEXT = 'Read only while Propose again works on this drive.';
let timer: number | undefined;

/** True (and says why under the photo) when the drive is read only: the change asked for is not made. */
export function refuseChange(): boolean {
  if (!report.readOnly) return false;
  const row = currentRow();
  if (row) state.notice = {key: row.frame_key, text: READ_ONLY_TEXT, error: true};
  return true;
}

function plural(count: number, word: string): string {
  return count.toLocaleString('en-GB') + ' ' + word + (count === 1 ? '' : 's');
}

/** The banner of a read-only page: {title, text, reload (the label of its button, '' for none)}. */
export function busyBanner(): {title: string; text: string; reload: string} | null {
  if (!report.readOnly) return null;
  const busy = state.busy;
  if (busy === null) {
    return {title: 'Propose again is done', text: 'This drive has its new proposals now.', reload: 'Review them'};
  }
  const who = busy.proposer ?? 'Propose again';
  if (busy.phase === 'waiting') {
    const ahead = busy.ahead ?? 0;
    return {title: 'Read only · waiting for ' + who,
      text: who + ' will propose the signs of this drive again' + (ahead > 0 ? ' after ' + plural(ahead, 'drive')
        + ' before it' : '') + '. These proposals will be replaced, so nothing can be confirmed until that is done.',
      reload: ''};
  }
  const done = busy.done ?? 0;
  const of = busy.total ? ' of ' + busy.total.toLocaleString('en-GB') : '';
  if (!report.atWork) {
    return {title: 'Read only · ' + who + ' at work',
      text: who + ' is proposing the signs of this drive again: ' + done.toLocaleString('en-GB') + of + ' frames so far.',
      reload: 'Show its proposals'};
  }
  return {title: 'Live · ' + who + ' at work',
    text: 'Its new proposals appear here as it makes them: ' + plural(report.rows.length, 'frame') + of + ' so far. '
      + 'Read only until it has done them all.', reload: ''};
}

/** Follow on: the newest frame proposed now, then each one as it comes. */
export function toggleFollow(): void {
  following.value = !following.value;
  const last = report.rows[report.rows.length - 1];
  if (following.value && last) {
    openFrame(last.frame_key);
    followed = last.frame_key;
  }
}

let asking = false;
async function ask(): Promise<void> {
  if (!report.apiBase || asking) return;
  asking = true;
  try {
    const url = report.apiBase + '/busy' + (report.atWork ? '/' + report.rows.length : '');
    const answer = await call<{busy: Busy | null; run_id?: string; rows?: FrameRow[]}>('GET', url);
    state.busy = answer.busy;
    if (answer.busy === null) window.clearInterval(timer);
    if (answer.rows && answer.run_id === report.runId && answer.rows.length > 0) {
      appendRows(answer.rows);
      const last = answer.rows[answer.rows.length - 1];
      if (following.value && state.currentKey !== followed) following.value = false;
      if (following.value && last) {
        openFrame(last.frame_key);
        followed = last.frame_key;
      }
    }
  } catch {
    // The network or a restart of the site: asked again at the next round.
  } finally {
    asking = false;
  }
}

/** Follows a read-only drive until Propose again is done with it. */
export function watchBusy(): void {
  if (!report.readOnly || !report.apiBase || timer !== undefined) return;
  timer = window.setInterval(() => void ask(), report.atWork ? LIVE_MS : INTERVAL_MS);
}
