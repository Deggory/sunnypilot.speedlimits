// Panoramica, Prepara, Allena: one page, the tab shown follows the address. The tab links of the top bar
// (rendered by pages.py) switch tab here without a reload; Route and the brand stay normal links.
import {onBeforeUnmount, onMounted, ref} from 'vue';

export type ControlTab = 'overview' | 'prepare' | 'train';

export const TAB_PATH: Record<ControlTab, string> = {overview: '/panoramica', prepare: '/prepara', train: '/allena'};
const TAB_TITLE: Record<ControlTab, string> = {overview: 'Panoramica', prepare: 'Prepara', train: 'Allena'};

export function tabOf(path: string): ControlTab | null {
  const clean = path.replace(/\/+$/, '');
  return (Object.keys(TAB_PATH) as ControlTab[]).find((tab) => TAB_PATH[tab] === clean) ?? null;
}

/** The tab shown. */
export const currentTab = ref<ControlTab>(tabOf(location.pathname) ?? 'overview');

function tabLinks(): [HTMLAnchorElement, ControlTab][] {
  return [...document.querySelectorAll<HTMLAnchorElement>('.tabs a')]
    .map((link): [HTMLAnchorElement, ControlTab | null] => [link, tabOf(link.pathname)])
    .filter((pair): pair is [HTMLAnchorElement, ControlTab] => pair[1] !== null);
}

/** aria-current on the server's tab link of the tab shown, and the page title. */
function markCurrent(tab: ControlTab): void {
  for (const [link, linkTab] of tabLinks()) {
    if (linkTab === tab) link.setAttribute('aria-current', 'page');
    else link.removeAttribute('aria-current');
  }
  document.title = 'Speed limits · ' + TAB_TITLE[tab];
}

/** A dot on the tab links of the tabs with work running (data-busy, styled by control.html). */
export function markBusy(busy: ReadonlySet<ControlTab>): void {
  for (const [link, tab] of tabLinks()) {
    if (busy.has(tab)) link.dataset['busy'] = 'true';
    else delete link.dataset['busy'];
  }
}

/** Shows a tab and puts its address in the history. */
export function goTo(tab: ControlTab): void {
  if (tab === currentTab.value) return;
  currentTab.value = tab;
  history.pushState({tab}, '', TAB_PATH[tab]);
  markCurrent(tab);
  window.scrollTo({top: 0});
}

/** Clicks on the tab links and the back button, for the life of the app. */
export function useTabRouting(): void {
  function onClick(event: MouseEvent): void {
    if (event.defaultPrevented || event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    const link = (event.target as Element | null)?.closest<HTMLAnchorElement>('a[href]');
    if (!link || link.origin !== location.origin || link.target) return;
    const tab = tabOf(link.pathname);
    if (tab === null) return;
    event.preventDefault();
    goTo(tab);
  }
  function onPop(): void {
    const tab = tabOf(location.pathname);
    if (tab === null) return;
    currentTab.value = tab;
    markCurrent(tab);
  }
  onMounted(() => {
    markCurrent(currentTab.value);
    document.addEventListener('click', onClick);
    window.addEventListener('popstate', onPop);
  });
  onBeforeUnmount(() => {
    document.removeEventListener('click', onClick);
    window.removeEventListener('popstate', onPop);
  });
}
