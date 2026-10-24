<script setup lang="ts">
// The bar under the photo: position in the filter, state tag, frame data, drawing and removing boxes, the
// second reading shown or not.
import {computed} from 'vue';
import {position, selectedBox, status} from './current';
import {removeSelected, toggleDrawing, toggleHints} from './edits';
import {aiConfirmed, proposalsOf, reasonsOf, statusLabel} from './frames';
import {go} from './navigation';
import {hintsOf, report, state} from './store';
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
  let info = 'Segmento ' + row.segment_index + ' · immagine ' + row.output_index + ' · ' + proposals
    + (proposals === 1 ? ' proposta' : ' proposte');
  if (report.routeHasHints) info += ' · ' + hintsOf(row.frame_key).length + ' in seconda lettura';
  return info;
});

function hintsSwitched(event: Event): void {
  if ((event.target as HTMLInputElement).checked !== state.showHints) toggleHints();
}
</script>

<template>
  <div class="stage-bar">
    <div class="nav">
      <button type="button" class="btn icon" id="prev" aria-label="Fotogramma precedente" :disabled="position.index <= 0"
              @click="go(-1)">‹</button>
      <span class="pos" id="pos">{{ position.index + 1 }} / {{ position.length }}</span>
      <button type="button" class="btn icon" id="next" aria-label="Fotogramma successivo"
              :disabled="position.index >= position.length - 1" @click="go(1)">›</button>
    </div>
    <span id="statePill" :class="pillClass">{{ statusLabel(row.frame_key) }}</span>
    <span class="data" id="frameData">{{ frameData }}</span>
    <div class="stage-tools">
      <button type="button" class="btn" :class="{on: state.drawing}" id="draw" @click="toggleDrawing">{{ state.drawing ? 'Annulla' : '+ Riquadro' }}</button>
      <button type="button" class="btn danger" id="remove" :disabled="selectedBox === null" @click="removeSelected">Elimina</button>
    </div>
    <label class="switch" id="hintToggle" :hidden="!report.routeHasHints"><input type="checkbox" id="showHints" :checked="state.showHints" @change="hintsSwitched"><span class="track" aria-hidden="true"></span>Seconda lettura</label>
  </div>
</template>
