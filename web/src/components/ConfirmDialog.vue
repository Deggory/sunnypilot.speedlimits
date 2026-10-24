<script setup lang="ts">
// The confirm dialog (composables/confirm.ts): a card over a dimmed page with a title, one sentence, Cancel
// and the action (red when it stops or signs out). Esc and a click outside cancel; Tab stays in the card;
// the page's own shortcuts wait while it is open; the focus goes back where it was.
import {nextTick, onBeforeUnmount, useTemplateRef, watch} from 'vue';
import {answerConfirm, confirmRequest} from '../composables/confirm';

const card = useTemplateRef<HTMLDivElement>('card');
let returnTo: HTMLElement | null = null;
/** The press started on the backdrop: only then its click cancels (a drag out of the card does not). */
let downOutside = false;

function buttons(): HTMLButtonElement[] {
  return [...(card.value?.querySelectorAll<HTMLButtonElement>('button') ?? [])];
}

function onKey(event: KeyboardEvent): void {
  // Nothing else on the page reacts while the dialog is open (the review page's arrows, Enter, Esc of a sheet).
  event.stopPropagation();
  if (event.key === 'Escape') {
    event.preventDefault();
    answerConfirm(false);
    return;
  }
  if (event.key !== 'Tab') return;
  const list = buttons();
  const first = list[0];
  const last = list[list.length - 1];
  if (!first || !last) return;
  const inside = card.value?.contains(document.activeElement) ?? false;
  if (event.shiftKey && (!inside || document.activeElement === first)) {
    event.preventDefault();
    last.focus();
  } else if (!event.shiftKey && (!inside || document.activeElement === last)) {
    event.preventDefault();
    first.focus();
  }
}

watch(confirmRequest, async (request, before) => {
  if (request !== null && before === null) {
    returnTo = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    window.addEventListener('keydown', onKey, true);
    document.body.classList.add('dialog-open');
  } else if (request === null) {
    window.removeEventListener('keydown', onKey, true);
    document.body.classList.remove('dialog-open');
    const target = returnTo;
    returnTo = null;
    if (target?.isConnected) target.focus();
    return;
  }
  await nextTick();
  // A red action is never one Enter away: Cancel has the focus.
  const [cancel, action] = buttons();
  (request?.danger ? cancel : action)?.focus();
});
onBeforeUnmount(() => window.removeEventListener('keydown', onKey, true));

function pressed(event: PointerEvent): void {
  downOutside = event.target === event.currentTarget;
}
function clickedOutside(): void {
  if (downOutside) answerConfirm(false);
  downOutside = false;
}
</script>

<template>
  <div v-if="confirmRequest" class="dialog-backdrop" @pointerdown="pressed" @click.self="clickedOutside">
    <div ref="card" class="dialog" :class="{danger: confirmRequest.danger}" role="alertdialog" aria-modal="true"
         aria-labelledby="dialog-title" aria-describedby="dialog-text">
      <h2 id="dialog-title" class="dialog-title">{{ confirmRequest.title }}</h2>
      <p id="dialog-text" class="dialog-text">{{ confirmRequest.text }}</p>
      <div class="dialog-actions">
        <button class="btn" type="button" @click="answerConfirm(false)">Cancel</button>
        <button class="btn" :class="confirmRequest.danger ? 'destructive' : 'primary'" type="button"
                @click="answerConfirm(true)">{{ confirmRequest.action }}</button>
      </div>
    </div>
  </div>
</template>
