// The toast at the bottom right: one per page, shown by <ToastHost/>.
import {ref} from 'vue';

export const toastState = ref<{message: string; error: boolean} | null>(null);
let timer: number | undefined;

export function toast(message: string, error = false): void {
  toastState.value = {message, error};
  window.clearTimeout(timer);
  timer = window.setTimeout(() => { toastState.value = null; }, error ? 7000 : 3500);
}
