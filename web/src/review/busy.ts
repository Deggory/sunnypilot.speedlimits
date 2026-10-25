// A drive Propose again works on (nnslr_tools.reproposing) is read only: nothing on it is confirmed or changed until
// its new proposals are in (the server refuses a save too). The drive at work shows the new proposals made so far;
// a drive waiting its turn shows its current report. While the page is read only it asks the server every
// INTERVAL_MS how far it is, and says when it is done: then a reload shows the new proposals, to review.
import {currentRow} from './frames';
import {call} from './server';
import {report, state} from './store';
import type {Busy} from './types';

const INTERVAL_MS = 15000;
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
  const shown = report.rows.length;
  return {title: 'Read only · ' + who + ' at work',
    text: 'You see its new proposals: ' + plural(shown, 'frame') + of + ' so far' + (done > shown ? ' (' + done.toLocaleString('en-GB')
      + ' now)' : '') + '. Nothing can be confirmed until it has done them all.',
    reload: done > shown ? 'Show the newest' : ''};
}

async function ask(): Promise<void> {
  if (!report.apiBase) return;
  try {
    const answer = await call<{busy: Busy | null}>('GET', report.apiBase + '/busy');
    state.busy = answer.busy;
    if (answer.busy === null) window.clearInterval(timer);
  } catch {
    // The network or a restart of the site: asked again at the next round.
  }
}

/** Follows a read-only drive until Propose again is done with it. */
export function watchBusy(): void {
  if (!report.readOnly || !report.apiBase || timer !== undefined) return;
  timer = window.setInterval(() => void ask(), INTERVAL_MS);
}
