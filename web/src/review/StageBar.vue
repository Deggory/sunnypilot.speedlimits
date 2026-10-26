<script setup lang="ts">
// The bar under the photo: position in the route, which frames to show (ReviewFilters.vue), state tag, frame data,
// drawing and removing boxes, the second reading shown or not.
import {computed} from 'vue';
import {position, selectedBox, status} from './current';
import {removeSelected, toggleDrawing, toggleHints} from './edits';
import {thousands} from '../common/format';
import {aiConfirmed, framePlace, proposalsOf, reasonsOf, statusLabel} from './frames';
import {step} from './navigation';
import {hintsOf, report, state} from './store';
import ReviewFilters from './ReviewFilters.vue';
import type {FrameRow} from './types';

const props = defineProps<{row: FrameRow}>();

const pillClass = computed(() => {
  const key = props.row.frame_key;
  return 'pill state ' + status.value + (aiConfirmed(key) ? ' ai' : '')
    + (status.value === 'todo' && reasonsOf(key).length > 0 ? ' sus' : '');
});

const frameData = computed(() => {
  const row = props.row;
  const proposals = proposalsOf(row).length;
  let info = framePlace(row) + ' · ' + proposals + (proposals === 1 ? ' proposal' : ' proposals');
  if (report.routeHasHints) info += ' · ' + hintsOf(row.frame_key).length + ' in the second reading';
  return info;
});

function hintsSwitched(event: Event): void {
  if ((event.target as HTMLInputElement).checked !== state.showHints) toggleHints();
}
</script>

<template>
  <div class="stage-bar">
    <div class="nav">
      <button type="button" class="btn icon" id="prev" aria-label="Previous frame" title="Previous frame (Shift ←)"
              :disabled="position.index <= 0" @click="step(-1)"><span class="nav-glyph">‹</span><kbd class="nav-key">⇧ ←</kbd></button>
      <span class="pos" id="pos">{{ thousands(position.index + 1) }} / {{ thousands(position.length) }}</span>
      <button type="button" class="btn icon" id="next" aria-label="Next frame" title="Next frame (Shift →)"
              :disabled="position.index >= position.length - 1" @click="step(1)"><span class="nav-glyph">›</span><kbd class="nav-key">⇧ →</kbd></button>
    </div>
    <ReviewFilters/>
    <span id="statePill" :class="pillClass">{{ statusLabel(row.frame_key) }}</span>
    <span class="data" id="frameData" :title="'Segment ' + row.segment_index + ', image ' + row.output_index">{{ frameData }}</span>
    <div class="stage-tools">
      <button type="button" class="btn" :class="{on: state.drawing}" id="draw" :disabled="report.readOnly" @click="toggleDrawing">{{ state.drawing ? 'Cancel' : '+ Box' }}</button>
      <button type="button" class="btn danger" id="remove" :disabled="selectedBox === null" @click="removeSelected">Delete</button>
    </div>
    <label class="switch" id="hintToggle" :hidden="!report.routeHasHints"><input type="checkbox" id="showHints" :checked="state.showHints" @change="hintsSwitched"><span class="track" aria-hidden="true"></span>Second reading</label>
  </div>
</template>
