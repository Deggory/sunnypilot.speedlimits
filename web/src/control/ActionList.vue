<script setup lang="ts">
// The actions of a tab, one per row: the button, what it does in one sentence and, when it cannot start
// now, why (the reason line under the sentence, tied to the button for screen readers).
import {computed} from 'vue';
import type {ActionName, ActionState, PanelStatus} from '../common/panel-types';

const props = defineProps<{items: [name: ActionName, text: string][]; actions: PanelStatus['actions'] | null}>();
defineEmits<{run: [name: ActionName, state: ActionState]}>();

const rows = computed(() => {
  const actions = props.actions;
  return props.items.map(([name, text]) => {
    const state = actions?.[name] ?? null;
    const reason = state && !state.allowed && state.reason ? state.reason.charAt(0).toUpperCase() + state.reason.slice(1) + '.' : '';
    return {name, text, state, reason};
  });
});
</script>

<template>
  <ul class="acts">
    <li v-for="row in rows" :key="row.name" class="act" :class="{blocked: row.state !== null && !row.state.allowed}" :data-action="row.name">
      <button class="btn" type="button" :id="'run-' + row.name" :disabled="row.state === null || !row.state.allowed"
              :aria-describedby="'act-text-' + row.name + (row.reason ? ' act-reason-' + row.name : '')"
              @click="row.state && $emit('run', row.name, row.state)">{{ row.state?.label ?? '…' }}</button>
      <div class="act-text">
        <p :id="'act-text-' + row.name">{{ row.text }}</p>
        <p v-if="row.reason" class="reason" :id="'act-reason-' + row.name">{{ row.reason }}</p>
      </div>
    </li>
  </ul>
</template>
