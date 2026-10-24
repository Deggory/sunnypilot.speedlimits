// The confirm dialog of the site (components/ConfirmDialog.vue), instead of the browser's window.confirm:
// askConfirm() shows it and resolves true (the action) or false (Cancel, Esc, a click outside). One dialog
// per page, mounted by common/mount.ts next to the Account menu, so every app of the page can ask.
import {readonly, shallowRef} from 'vue';

export interface ConfirmOptions {
  title: string;
  /** One sentence: what happens. */
  text: string;
  /** The label of the button that goes ahead, e.g. "Start training". */
  action: string;
  /** Red action (Stop, Sign out everywhere); Cancel then has the focus. */
  danger?: boolean;
}

interface Request extends ConfirmOptions {
  resolve: (ok: boolean) => void;
}

const current = shallowRef<Request | null>(null);

/** The question shown now, or null. */
export const confirmRequest = readonly(current);

/** Shows the dialog; a question still open is answered "no" first. */
export function askConfirm(options: ConfirmOptions): Promise<boolean> {
  current.value?.resolve(false);
  return new Promise((resolve) => {
    current.value = {...options, resolve};
  });
}

/** The answer of the dialog. */
export function answerConfirm(ok: boolean): void {
  const request = current.value;
  current.value = null;
  request?.resolve(ok);
}
