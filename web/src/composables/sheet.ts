// A bottom sheet of a form (the AI provider of Settings, "Review with AI" of the route list): while it is open the
// page under it does not scroll, Esc closes it and the focus goes to its first field; closed, the focus goes back
// to where it was. The look is shared.css's .sheet.
import {nextTick, onBeforeUnmount, watch, type Ref} from 'vue';

export function useSheet(open: Ref<boolean>, panel: Readonly<Ref<HTMLElement | null>>, close: () => void): void {
  let returnTo: HTMLElement | null = null;
  function onKey(event: KeyboardEvent): void {
    if (event.key === 'Escape' && open.value) {
      event.preventDefault();
      close();
    }
  }
  watch(open, async (value) => {
    document.body.classList.toggle('sheet-open', value);
    if (value) {
      returnTo = document.activeElement instanceof HTMLElement ? document.activeElement : null;
      document.addEventListener('keydown', onKey);
      await nextTick();
      panel.value?.querySelector<HTMLElement>('input:not([disabled]), button.primary:not([disabled]), button:not([disabled])')?.focus();
      return;
    }
    document.removeEventListener('keydown', onKey);
    const target = returnTo;
    returnTo = null;
    if (target?.isConnected) target.focus();
  });
  onBeforeUnmount(() => {
    document.removeEventListener('keydown', onKey);
    document.body.classList.remove('sheet-open');
  });
}
