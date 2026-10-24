<script setup lang="ts">
// "Prossimo passo": the card on top, coloured as the step it points to, with its one obvious action.
import {computed} from 'vue';
import type {ActionName, ActionState, PanelStatus} from '../common/panel-types';
import type {NextStep} from './next-step';

const props = defineProps<{next: NextStep; actions: PanelStatus['actions'] | null}>();
defineEmits<{run: [name: ActionName, state: ActionState]}>();

/** The button of an action other than review (review is a link to the routes). */
const button = computed(() => {
  const action = props.next.action;
  if (action === null || action === 'review' || props.actions === null) return null;
  const state = props.actions[action];
  return {name: action, state, reason: !state.allowed && state.reason ? state.reason : ''};
});
</script>

<template>
  <section class="card hero" :class="next.kind === null ? undefined : 'k-' + next.kind" id="hero" aria-labelledby="hero-title">
    <div>
      <h2 class="kicker">Prossimo passo</h2>
      <div class="title" id="hero-title">{{ next.title }}</div>
      <div class="text" id="hero-text">{{ next.text }}</div>
    </div>
    <div class="actions" id="hero-actions"><a v-if="next.action === 'review'" class="btn primary" href="/">Apri la revisione →</a><template
      v-else-if="button"><button class="btn primary" type="button" :disabled="!button.state.allowed"
        :aria-describedby="button.reason ? 'hero-reason' : undefined" @click="$emit('run', button.name, button.state)">{{ button.state.label }}</button><p
        v-if="button.reason" class="reason" id="hero-reason">{{ button.reason }}</p></template></div>
  </section>
</template>
