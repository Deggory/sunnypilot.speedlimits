// Keyboard shortcuts: ← → frame (with Shift: the frame before or after in the filter), ↑ ↓ box, the speed typed
// as it reads ("5 0" = 50, "1 2 0" = 120, "0" alone = not readable), O A U the road, Enter confirm, N new box,
// P apply previous, H second reading, Del removes a box the reviewer selected (never one the page opened on),
// Esc cancel.
import {onBeforeUnmount, onMounted} from 'vue';
import {applyPrevious} from './carry';
import {confirmFrame} from './confirm';
import {cancelAll, removeSelected, selectBox, setField, setValue, toggleDrawing, toggleHints} from './edits';
import {currentRow, workOf} from './frames';
import {go, step} from './navigation';
import {report, state} from './store';

// A speed is typed as it reads: the digits wait this long for the next one, then the value is set.
const TYPING_MS = 900;

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
  let typingTimer: number | undefined;
  /** The digits typed so far become the value of the selected box: "0" alone = not readable; 10-250 a speed. */
  const flushTyping = (): void => {
    window.clearTimeout(typingTimer);
    const typed = state.typing;
    state.typing = '';
    if (!typed || state.selected < 0) return;
    if (typed === '0') {
      setValue(null);
      return;
    }
    const value = Number(typed);
    if (value >= 10 && value <= 250) {
      setValue(value);
      return;
    }
    const key = state.currentKey ?? '';
    state.notice = {key, text: 'Type the whole number: 5 0 for 50, 1 2 0 for 120 (0 alone: not readable).', error: true};
  };
  const onKeyDown = (event: KeyboardEvent): void => {
    const target = event.target instanceof HTMLElement ? event.target : null;
    const tag = target ? target.tagName.toLowerCase() : '';
    if (tag === 'input' || tag === 'select' || tag === 'textarea') {
      if (event.key === 'Escape') target?.blur();
      return;
    }
    if (event.metaKey || event.ctrlKey || event.altKey) return;
    // Any key but a digit ends the number being typed first: Enter then confirms the value just typed.
    if (state.typing && !/^[0-9]$/.test(event.key)) flushTyping();
    // The same buttons reached with Tab keep their own Enter.
    const byPointer = target !== null && target === pointerButton;
    if (event.key === 'Enter' && byPointer && CONFIRM_ON_ENTER.has(target.id)) {
      event.preventDefault();
      void confirmFrame();
      return;
    }
    // Enter on any other focused button or summary belongs to that control.
    if (event.key === 'Enter' && (tag === 'button' || tag === 'summary' || tag === 'a')) return;
    // With a box selected: its speed typed as it reads, and its road (O A U).
    if (state.selected >= 0 && !event.shiftKey && /^[0-9]$/.test(event.key)) {
      event.preventDefault();
      state.typing += event.key;
      window.clearTimeout(typingTimer);
      if (state.typing.length >= 3) flushTyping();
      else typingTimer = window.setTimeout(flushTyping, TYPING_MS);
      return;
    }
    if (state.selected >= 0 && !event.shiftKey) {
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
        // Only a box the reviewer selected: never the one the page opened on, never with Backspace.
        if (state.selected >= 0 && state.selectedByUser) {
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
