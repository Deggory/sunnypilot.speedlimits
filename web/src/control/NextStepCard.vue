<script setup lang="ts">
// "Next step": the card on top of the Overview, coloured as the phase it points to, with its one obvious
// action (reviewing is a link: the route in progress, else the route list).
import {computed} from 'vue';
import {sentence} from '../common/format';
import type {ActionName, ActionState, PanelStatus} from '../common/panel-types';
import Icon from '../components/Icon.vue';
import {ACTION_ICON} from './jobs';
import type {NextStep} from './next-step';

const props = defineProps<{next: NextStep; actions: PanelStatus['actions'] | null; reviewHref: string}>();
defineEmits<{run: [name: ActionName, state: ActionState]}>();

/** The button of an action other than review. */
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
      <h2 class="kicker">Next step</h2>
      <div class="title" id="hero-title">{{ next.title }}</div>
      <div class="text" id="hero-text">{{ next.text }}</div>
    </div>
    <div class="actions" id="hero-actions"><a v-if="next.action === 'review'" class="btn primary" :href="reviewHref"><Icon name="resume"/>Resume review →</a><template
      v-else-if="button"><button class="btn primary" type="button" :disabled="!button.state.allowed"
        :aria-describedby="button.reason ? 'hero-reason' : undefined" @click="$emit('run', button.name, button.state)"><Icon :name="ACTION_ICON[button.name]"/>{{ button.state.label }}</button><p
        v-if="button.reason" class="reason" id="hero-reason">{{ sentence(button.reason) }}</p></template></div>
  </section>
</template>
