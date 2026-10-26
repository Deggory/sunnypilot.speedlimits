// The pages with work running (control/jobs.ts busyTabs, by address): a pulsing dot on their section's link in the
// top bar and the phone's tab bar (data-paths: the addresses of the section, pages.top_bar), on their link under the
// section's title (SectionHead.vue, nav.subnav), and on their row of the phone menu (AccountMenu.vue, which reads
// busyPaths). Fed on every page: by the status of the control pages (ControlApp.vue), by the server pill's poll
// elsewhere (TopMachine.vue).
import {shallowRef} from 'vue';

/** The addresses of the busy tabs ("/prepare", "/train"...). */
export const busyPaths = shallowRef<ReadonlySet<string>>(new Set());

/** Whether a link leads to busy work: its own address, or one of its section's (data-paths). */
export function linkBusy(link: HTMLAnchorElement, paths: ReadonlySet<string>): boolean {
  return (link.dataset['paths'] ?? link.pathname).split(' ').some((path) => paths.has(path));
}

/** Marks the busy sections and pages: the section links of the top bar and the tab bar, the pages under a title. */
export function markBusy(paths: ReadonlySet<string>): void {
  busyPaths.value = paths;
  for (const link of document.querySelectorAll<HTMLAnchorElement>('nav.tabs a, nav.tabbar a, nav.subnav a')) {
    if (linkBusy(link, paths)) link.dataset['busy'] = 'true';
    else delete link.dataset['busy'];
  }
}
