// The tabs with work running (control/jobs.ts busyTabs, by address): a pulsing dot on their link in the top bar
// (data-busy, shared.css) and on their row of the phone menu (AccountMenu.vue, which reads busyPaths). Fed on every
// page: by the status of the control pages (ControlApp.vue), by the server pill's poll elsewhere (TopMachine.vue).
import {shallowRef} from 'vue';

/** The addresses of the busy tabs ("/prepare", "/train"...). */
export const busyPaths = shallowRef<ReadonlySet<string>>(new Set());

/** Marks the busy tabs: the links of the tab row, and the rows of the phone menu. */
export function markBusy(paths: ReadonlySet<string>): void {
  busyPaths.value = paths;
  for (const link of document.querySelectorAll<HTMLAnchorElement>('nav.tabs a')) {
    if (paths.has(link.pathname)) link.dataset['busy'] = 'true';
    else delete link.dataset['busy'];
  }
}
