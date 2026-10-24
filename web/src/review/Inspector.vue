<script setup lang="ts">
// The inspector, docked on the right on a computer and under the photo on a phone: the boxes of the frame,
// the editor of the selected one, the second reading, "Altro", the confirm block.
import {computed} from 'vue';
import BoxEditor from './BoxEditor.vue';
import BoxList from './BoxList.vue';
import {boxCountText} from './boxes';
import ConfirmArea from './ConfirmArea.vue';
import {selectedBox} from './current';
import {reasonsOf} from './frames';
import OtherFields from './OtherFields.vue';
import SecondReading from './SecondReading.vue';
import {state} from './store';
import type {FrameEdit, FrameRow} from './types';

const props = defineProps<{row: FrameRow; work: FrameEdit}>();
const reasons = computed(() => reasonsOf(props.row.frame_key));
const count = computed(() => props.work.detections.length);
</script>

<template>
  <aside class="panel" :class="{editing: selectedBox !== null}" aria-label="Revisione del fotogramma">
    <section>
      <div class="section-head"><h2>Cartelli</h2><span class="count" id="boxCount">{{ boxCountText(count) }}</span></div>
      <p class="reasons" id="reasons" :hidden="reasons.length === 0"><template v-if="reasons.length > 0"><strong>Sospetto: </strong>{{ reasons.map((reason) => String(reason.text)).join('; ') }}</template></p>
      <p class="filter-note" id="filterNote" :hidden="state.filterNote === ''">{{ state.filterNote }}</p>
      <BoxList :row="row" :work="work"/>
      <p class="pick-hint" id="pickHint" :hidden="count === 0 || state.selected >= 0">Seleziona un riquadro per correggerne valore, tipo e strada.</p>
      <p class="empty-note" id="noBoxes" :hidden="count > 0"><svg viewBox="0 0 34 34" aria-hidden="true"><circle cx="17" cy="17" r="14" fill="none" stroke="currentColor" stroke-width="2" stroke-dasharray="4 3.3"/><path d="M12 17h10" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg><span>Nessun riquadro. Se non ci sono cartelli di velocità, conferma “nessun cartello”: serve anche questo esempio.</span></p>
    </section>
    <BoxEditor :row="row" :detection="selectedBox"/>
    <SecondReading :row="row" :work="work"/>
    <OtherFields :row="row" :work="work"/>
    <ConfirmArea :row="row" :work="work"/>
  </aside>
</template>
