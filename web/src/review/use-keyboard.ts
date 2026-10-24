// Keyboard shortcuts: ← → frame (with Shift: the frame before or after in the filter), ↑ ↓ box, 1-9 the speeds
// of the sign buttons and 0 "not readable", O A U the road, Enter confirm, N new box, P apply previous,
// H second reading, Del (or Backspace) remove, Esc cancel.
import {onBeforeUnmount, onMounted} from 'vue';
import {applyPrevious} from './carry';
import {confirmFrame} from './confirm';
import {QUICK_VALUES} from './catalog';
import {cancelAll, removeSelected, selectBox, setField, setValue, toggleDrawing, toggleHints} from './edits';
import {currentRow, workOf} from './frames';
import {go, step} from './navigation';
import {report, state} from './store';

// O A U: the road of the selected box.
const ROAD_KEYS: Record<string, string> = {o: 'own_road', a: 'other_road', u: 'unresolved'};

/** ↑ ↓: the box before or after the selected one (from none: the first or the last). */
function stepBox(delta: number): void {
  const row = currentRow();
  if (!row) return;
  const count = workOf(row).detections.length;
  if (count === 0) return;
  const next = state.selected < 0 ? (delta > 0 ? 0 : count - 1) : (state.selected + delta + count) % count;
  selectBox(next);
}

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
    // With a box selected: its speed (1-9 the sign buttons, 0 not readable) and its road (O A U).
    if (state.selected >= 0 && !event.shiftKey) {
      const digit = /^[0-9]$/.test(event.key) ? Number(event.key) : null;
      if (digit !== null && (digit === 0 || QUICK_VALUES[digit - 1] !== undefined)) {
        event.preventDefault();
        setValue(digit === 0 ? null : QUICK_VALUES[digit - 1] ?? null);
        return;
      }
      const road = ROAD_KEYS[event.key.toLowerCase()];
      if (road) {
        event.preventDefault();
        setField('applicability', road);
        return;
      }
    }
    switch (event.key) {
      case 'ArrowUp':
        event.preventDefault();
        stepBox(-1);
        break;
      case 'ArrowDown':
        event.preventDefault();
        stepBox(1);
        break;
      case 'ArrowLeft':
        event.preventDefault();
        if (event.shiftKey) go(-1);
        else step(-1);
        break;
      case 'ArrowRight':
        event.preventDefault();
        if (event.shiftKey) go(1);
        else step(1);
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
      case 'p':
      case 'P':
        if (report.apiBase) {
          event.preventDefault();
          void applyPrevious();
        }
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
