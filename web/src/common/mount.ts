// Mounting the Vue parts of a page.
import {createApp, type Component} from 'vue';
import AccountMenu from '../components/AccountMenu.vue';

/** The Account menu in the top bar rendered by pages.py (#account-menu); data-current marks the Account page. */
export function mountAccountMenu(): void {
  const host = document.getElementById('account-menu');
  if (host !== null) createApp(AccountMenu, {current: host.dataset['current'] === 'true'}).mount(host);
}

/** The page's own app in #app, plus the Account menu. */
export function mountPage(root: Component, props: Record<string, unknown> = {}): void {
  mountAccountMenu();
  const host = document.getElementById('app');
  if (host === null) throw new Error('Elemento mancante nella pagina: #app');
  createApp(root, props).mount(host);
}
