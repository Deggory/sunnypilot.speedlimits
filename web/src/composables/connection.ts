// The connection of the page with the site, in the top bar of every page (#live-dot, #live-text, rendered by pages.py
// outside the Vue apps): nothing while the site answers; a red dot and one sentence when it does not, and the pills of
// the top bar (the comma, the server) greyed out, since what they show is old. Fed by the one poller of the page: the
// status of the control pages (panel-status.ts), the server pill elsewhere (TopMachine.vue).
import {UNREACHABLE, signedOut} from '../common/api';

/** The sentence of the top bar for an error of the poller, or '' for an answer. */
export function connectionText(error: unknown): string {
  if (error === null) return '';
  if (signedOut(error)) return 'Signed out. Reload the page to sign in.';
  const message = error instanceof Error ? error.message : '';
  return message === UNREACHABLE ? 'Can\'t reach the site. Trying again…' : 'The site has a problem. Trying again…';
}

/** Shows the state of the connection: null when the site answered, else the error of the poller. */
export function showConnection(error: unknown): void {
  const text = connectionText(error);
  const live = document.querySelector('.live');
  live?.classList.toggle('off', text !== '');
  document.querySelector('.bar-end')?.classList.toggle('stale', text !== '');
  const dot = document.getElementById('live-dot');
  const label = document.getElementById('live-text');
  if (dot) {
    dot.className = text ? 'dot off' : 'dot';
    dot.title = text;
  }
  if (label) label.textContent = text;
}
