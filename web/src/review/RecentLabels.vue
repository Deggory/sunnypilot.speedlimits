<script setup lang="ts">
// The labels given last on this drive, one tap each on the box selected (recent.ts), at the top of the review's side:
// always in the same place, within reach (Cristian, 14 Nov 2026). Without a box selected they are shown and off.
import {computed} from 'vue';
import {releaseFocus} from './dom';
import {applyLabel} from './edits';
import {selectedDetection} from './frames';
import {labelKey, labelOf, recentLabels} from './recent';
import {state} from './store';
import type {FrameRow} from './types';

const props = defineProps<{row: FrameRow}>();

const recent = computed(() => recentLabels(props.row, state.selected));
// Pressed: the label the selected box has.
const labelNow = computed(() => {
  const detection = selectedDetection();
  return detection === null ? '' : labelKey(labelOf(detection));
});
</script>

<template>
  <div v-if="recent.length" class="recent-row recent-top" id="recentLabels" role="group" aria-label="Labels used last on this drive">
    <span class="recent-label">Recent</span>
    <button v-for="item in recent" :key="item.key" type="button" class="chip recent" :aria-pressed="item.key === labelNow"
            :disabled="state.selected < 0" :title="state.selected < 0 ? item.title + ' · select a box first' : item.title"
            @click="applyLabel(item.label); releaseFocus($event)"><span v-if="item.disc !== null"
            class="mini-sign">{{ item.disc }}</span>{{ item.text }}</button>
  </div>
</template>
