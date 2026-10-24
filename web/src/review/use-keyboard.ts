// Keyboard shortcuts: ← → frame, Invio conferma, N new box, H second reading, Canc remove, Esc cancel.
import {onBeforeUnmount, onMounted} from 'vue';
import {confirmFrame} from './confirm';
import {cancelAll, removeSelected, toggleDrawing, toggleHints} from './edits';
import {go} from './navigation';
import {report, state} from './store';

// Enter on these buttons just pressed with the mouse still confirms, as the shortcut list says.
const CONFIRM_ON_ENTER = new Set(['prev', 'next', 'mPrev', 'mNext', 'draw']);

export function useKeyboard(): void {
  // The button last pressed with the mouse or a finger; Tab moves the focus elsewhere and forgets it.
  let pointerButton: Element | null = null;
  const onPointerDown = (event: PointerEvent): void => {
    pointerButton = event.target instanceof Element ? event.target.closest('button') : null;
  };
  const onFocusIn = (event: FocusEvent): void => {
    if (event.target !== pointerButton) pointerButton = null;
  };
  const onKeyDown = (event: KeyboardEvent): void => {
    const target = event.target instanceof HTMLElement ? event.target : null;
    const tag = target ? target.tagName.toLowerCase() : '';
    if (tag === 'input' || tag === 'select' || tag === 'textarea') {
      if (event.key === 'Escape') target?.blur();
      return;
    }
    if (event.metaKey || event.ctrlKey || event.altKey) return;
    // The same buttons reached with Tab keep their own Enter.
    const byPointer = target !== null && target === pointerButton;
    if (event.key === 'Enter' && byPointer && CONFIRM_ON_ENTER.has(target.id)) {
      event.preventDefault();
      void confirmFrame();
      return;
    }
    // Enter on any other focused button or summary belongs to that control.
    if (event.key === 'Enter' && (tag === 'button' || tag === 'summary' || tag === 'a')) return;
    switch (event.key) {
      case 'ArrowLeft':
        event.preventDefault();
        go(-1);
        break;
      case 'ArrowRight':
        event.preventDefault();
        go(1);
        break;
      case 'Enter':
        event.preventDefault();
        void confirmFrame();
        break;
      case 'n':
      case 'N':
        event.preventDefault();
        toggleDrawing();
        break;
      case 'h':
      case 'H':
        if (report.routeHasHints) {
          event.preventDefault();
          toggleHints();
        }
        break;
      case 'Delete':
      case 'Backspace':
        if (state.selected >= 0) {
          event.preventDefault();
          removeSelected();
        }
        break;
      case 'Escape':
        cancelAll();
        break;
    }
  };
  onMounted(() => {
    document.addEventListener('pointerdown', onPointerDown, true);
    document.addEventListener('focusin', onFocusIn);
    document.addEventListener('keydown', onKeyDown);
  });
  onBeforeUnmount(() => {
    document.removeEventListener('pointerdown', onPointerDown, true);
    document.removeEventListener('focusin', onFocusIn);
    document.removeEventListener('keydown', onKeyDown);
  });
}
