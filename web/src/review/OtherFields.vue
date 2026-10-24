<script setup lang="ts">
// "More": physical site and encounter of the frame, the JSONL export, where the reviews are kept, what the
// proposals are.
import {setFrameField} from './edits';
import {exportJsonl} from './export';
import {vFill} from './fill';
import {report, state} from './store';
import type {FrameEdit, FrameRow} from './types';

defineProps<{row: FrameRow; work: FrameEdit}>();

const SERVER_NOTE = 'Every confirmation is saved in the server’s database. If a save fails, the frame stays marked '
  + '“not saved” and is tried again; meanwhile the changes stay in this browser. The JSONL stays available as a backup '
  + 'copy. A frame without boxes is not a negative example by itself: it has to be checked and confirmed.';
const OFFLINE_NOTE = 'The changes are saved in this browser. Export the JSONL to keep them outside the browser; the '
  + 'original file of the proposals stays as it is. A frame without boxes is not a negative example by itself: it has '
  + 'to be checked and confirmed.';

</script>

<template>
  <section>
    <details>
      <summary>More</summary>
      <div class="stack">
        <!-- The frame's own physical site and encounter, else the report's. -->
        <label>Physical site
          <input v-fill="work.site_group ?? row.site_group ?? ''" type="text" id="site_group"
                 placeholder="Same identifier in the other routes" autocomplete="off" @change="setFrameField(row, 'site_group', ($event.target as HTMLInputElement).value)">
        </label>
        <label>Encounter
          <input v-fill="work.encounter_id ?? row.encounter_id ?? ''" type="text" id="encounter_id"
                 placeholder="Same identifier in the nearby frames" autocomplete="off" @change="setFrameField(row, 'encounter_id', ($event.target as HTMLInputElement).value)">
        </label>
        <div class="row">
          <button type="button" class="btn small" id="export" @click="exportJsonl">Download reviews (JSONL)</button>
          <span :class="state.exportMessage ? 'save-msg ' + (state.exportMessage.ok ? 'ok' : 'error') : 'save-msg'"
                id="exportMsg" role="status">{{ state.exportMessage?.text }}</span>
        </div>
        <p id="storageNote">{{ report.apiBase ? SERVER_NOTE : OFFLINE_NOTE }}</p>
        <p>Sign proposals come from a model trained in Vietnam, and the numbers on the road from OCR checked with road segmentation. The models are not validated on Swiss roads. Confirm only what you can see. The proposed number is not the limit that applies to the car.</p>
        <p>Leave unknown what you cannot check. The moment of passing stays unannotated. Every review is checked against the rules of import-annotations before it enters the dataset.</p>
      </div>
    </details>
  </section>
</template>
