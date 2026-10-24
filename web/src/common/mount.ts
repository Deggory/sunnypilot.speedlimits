// Mounting the Vue parts of a page.
import {createApp, type Component} from 'vue';
import AccountMenu from '../components/AccountMenu.vue';
import ConfirmDialog from '../components/ConfirmDialog.vue';

/** The confirm dialog of the page (composables/confirm.ts), in its own element at the end of <body>. */
function mountConfirmDialog(): void {
  if (document.getElementById('dialog-host') !== null) return;
  const host = document.createElement('div');
  host.id = 'dialog-host';
  document.body.append(host);
  createApp(ConfirmDialog).mount(host);
}

/**
 * The parts every signed-in page shares: the Account menu in the top bar rendered by pages.py (#account-menu;
 * data-current marks the Account page) and the confirm dialog.
 */
export function mountSiteParts(): void {
  mountConfirmDialog();
  const host = document.getElementById('account-menu');
  if (host !== null) createApp(AccountMenu, {current: host.dataset['current'] === 'true'}).mount(host);
}

/** The page's own app in #app, plus the Account menu and the confirm dialog. */
export function mountPage(root: Component, props: Record<string, unknown> = {}): void {
  mountSiteParts();
  const host = document.getElementById('app');
  if (host === null) throw new Error('Missing element in the page: #app');
  createApp(root, props).mount(host);
}
