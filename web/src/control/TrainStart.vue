<script setup lang="ts">
// "Train a model": where the networks start from (status.training_starts, one card each: ImageNet, random
// weights, the stable model, an earlier run of this server; unavailable ones disabled with their reason, the
// suggested one chosen at first), then one Start training button (POST /api/panel/run/train {"start": key}).
import {computed, ref} from 'vue';
import {sentence} from '../common/format';
import type {ActionState, TrainingStart} from '../common/panel-types';
import Icon from '../components/Icon.vue';
import type {IconName} from '../components/icons/shapes';

const props = defineProps<{starts: TrainingStart[] | null; state: ActionState | null}>();
defineEmits<{start: [start: TrainingStart]}>();

/** The card chosen here; it stays chosen across the status refreshes while it is available. */
const chosen = ref<string | null>(null);
const selected = computed((): TrainingStart | null => {
  const starts = (props.starts ?? []).filter((start) => start.available);
  return starts.find((start) => start.key === chosen.value) ?? starts.find((start) => start.default) ?? starts[0] ?? null;
});
const iconOf = (start: TrainingStart): IconName =>
  start.key === 'imagenet' ? 'imagenet' : start.key === 'random' ? 'random' : start.key === 'stable' ? 'stable' : 'history';
const reason = computed(() => props.state && !props.state.allowed && props.state.reason ? sentence(props.state.reason) : '');
const disabled = computed(() => props.state === null || !props.state.allowed || selected.value === null);
</script>

<template>
  <div class="train-start" id="train-start">
    <div class="ts-head">
      <span class="act-icon" aria-hidden="true"><Icon name="train"/></span>
      <div>
        <h3 class="ts-title" id="train-start-title">Train a model</h3>
        <p class="ts-text" id="train-start-text">Imports the new reviews, splits the dataset, then trains the detector and the reader on the GPU,
          evaluates them and builds the package. It takes hours.</p>
      </div>
    </div>
    <fieldset class="starts" :disabled="starts === null">
      <legend class="ts-legend">Start from</legend>
      <label v-for="start in starts ?? []" :key="start.key" class="start" :data-start="start.key"
             :class="{on: selected?.key === start.key, off: !start.available}">
        <input type="radio" name="train-start" :value="start.key" :checked="selected?.key === start.key" :disabled="!start.available"
               :aria-describedby="'start-text-' + start.key" @change="chosen = start.key">
        <span class="start-art" aria-hidden="true"><Icon :name="iconOf(start)"/></span>
        <span class="start-body">
          <span class="start-label">{{ start.label }}<span v-if="start.default" class="start-tag">Recommended</span></span>
          <span class="start-text" :id="'start-text-' + start.key">{{ start.description }}<template v-if="!start.available && start.reason"><br><span
            class="start-reason">{{ sentence(start.reason) }}</span></template></span>
        </span>
        <span class="start-check" aria-hidden="true"><svg viewBox="0 0 20 20"><path d="m5 10.5 3.2 3L15 6.5" fill="none" stroke="currentColor"
          stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/></svg></span>
      </label>
      <p v-if="starts !== null && starts.length === 0" class="empty">The server lists no starting point.</p>
    </fieldset>
    <div class="ts-go">
      <button class="btn primary" type="button" id="run-train" :disabled="disabled"
              :aria-describedby="'train-start-text' + (reason ? ' train-start-reason' : '')"
              @click="selected && $emit('start', selected)"><Icon name="train"/>Start training</button>
      <p v-if="reason" class="reason act-reason" id="train-start-reason">{{ reason }}</p>
      <p v-else-if="selected" class="ts-from">From: <strong>{{ selected.label }}</strong></p>
    </div>
  </div>
</template>
