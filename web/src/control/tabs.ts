// Overview, Prepare, Train, Evaluate, Models: one page, the tab shown follows the address ("/", "/prepare", "/train",
// "/evaluate", "/models"). The tab links of the top bar (rendered by pages.py) and the brand switch tab here without a
// reload; the route list (/routes) and the review pages stay normal links.
import {nextTick, onBeforeUnmount, onMounted, ref} from 'vue';

export type ControlTab = 'overview' | 'prepare' | 'train' | 'evaluate' | 'models';

export const TAB_PATH: Record<ControlTab, string> = {
  overview: '/', prepare: '/prepare', train: '/train', evaluate: '/evaluate', models: '/models',
};
const TAB_TITLE: Record<ControlTab, string> = {
  overview: 'Overview', prepare: 'Data', train: 'Train', evaluate: 'Evaluate', models: 'Models',
};

export function tabOf(path: string): ControlTab | null {
  const clean = path.replace(/\/+$/, '') || '/';
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

/** Shows a tab and puts its address in the history; with a hash ("#dataset"), scrolls to it once the tab is drawn. */
export function goTo(tab: ControlTab, hash = ''): void {
  if (tab === currentTab.value && !hash) return;
  currentTab.value = tab;
  history.pushState({tab}, '', TAB_PATH[tab] + hash);
  markCurrent(tab);
  if (!hash) {
    window.scrollTo({top: 0});
    return;
  }
  void nextTick(() => document.getElementById(hash.slice(1))?.scrollIntoView({block: 'start'}));
}

/** Clicks on the tab links and the back button, for the life of the app. */
export function useTabRouting(): void {
  function onClick(event: MouseEvent): void {
    if (event.defaultPrevented || event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    const link = (event.target as Element | null)?.closest<HTMLAnchorElement>('a[href]');
    if (!link || link.origin !== location.origin || link.target) return;
    // An anchor on this tab ("#dataset") scrolls as usual.
    if (link.hash && link.pathname === location.pathname) return;
    const tab = tabOf(link.pathname);
    if (tab === null) return;
    event.preventDefault();
    goTo(tab, link.hash);
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
