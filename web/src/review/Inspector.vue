<script setup lang="ts">
// The inspector, docked on the right on a computer and under the photo on a phone: the boxes of the frame,
// the editor of the selected one, the second reading, "More", the confirm block.
import {computed} from 'vue';
import BoxEditor from './BoxEditor.vue';
import BoxList from './BoxList.vue';
import {boxCountText} from './boxes';
import {applyPrevious, carryState} from './carry';
import ConfirmArea from './ConfirmArea.vue';
import {releaseFocus} from './dom';
import Icon from '../components/Icon.vue';
import {selectedBox} from './current';
import {countryOf, reasonsOf} from './frames';
import {setCountry} from './edits';
import {vFill} from './fill';
import OtherFields from './OtherFields.vue';
import SecondReading from './SecondReading.vue';
import {report, state} from './store';
import type {FrameEdit, FrameRow} from './types';

const props = defineProps<{row: FrameRow; work: FrameEdit}>();
const reasons = computed(() => reasonsOf(props.row.frame_key));
const count = computed(() => props.work.detections.length);
const carry = computed(() => carryState(props.row));
const country = computed(() => countryOf(props.row, props.work));
// The countries of the select by name, with the one of the frame even if the borders do not know it.
const countryOptions = computed(() => {
  const options = Object.entries(report.countries).sort((a, b) => a[1].localeCompare(b[1]));
  for (const code of [props.row.country_proposed, country.value]) {
    if (code && !(code in report.countries)) options.unshift([code, code]);
  }
  return options;
});
// Picked by the reviewer: a country other than the GPS one.
const picked = computed(() => props.work.country !== undefined && props.work.country !== props.row.country_proposed);
const countrySource = computed(() => {
  if (picked.value) return 'picked here';
  if (country.value) return props.row.country_from === 'nearest_fix' ? 'from a nearby GPS fix' : 'from the GPS';
  if (props.row.gps === undefined) return 'GPS not read for this route';
  return 'no GPS position';
});
</script>

<template>
  <aside class="panel" :class="{editing: selectedBox !== null}" aria-label="Review of the frame">
    <section>
      <div class="section-head"><h2>Signs</h2><span class="count" id="boxCount">{{ boxCountText(count) }}</span>
        <button v-if="carry.shown" type="button" class="btn small carry" id="applyPrevious" :disabled="!carry.enabled"
                :title="carry.title" @click="applyPrevious(); releaseFocus($event)"><Icon name="previous"/>Apply previous</button></div>
      <div class="country-line">
        <label for="country">Country</label>
        <select id="country" v-fill="country ?? ''" @change="setCountry(row, ($event.target as HTMLSelectElement).value)">
          <option value="">Not known</option>
          <option v-for="[code, name] in countryOptions" :key="code" :value="code">{{ code }} · {{ name }}</option>
        </select>
        <span class="country-src" :class="{picked}">{{ countrySource }}</span>
      </div>
      <p class="reasons" id="reasons" :hidden="reasons.length === 0"><template v-if="reasons.length > 0"><strong>Suspicious: </strong>{{ reasons.map((reason) => String(reason.text)).join('; ') }}</template></p>
      <p class="filter-note" id="filterNote" :hidden="state.filterNote === ''">{{ state.filterNote }}</p>
      <BoxList :row="row" :work="work"/>
      <p class="pick-hint" id="pickHint" :hidden="count === 0 || state.selected >= 0">Select a box to correct its value, type and road.</p>
      <p class="empty-note" id="noBoxes" :hidden="count > 0"><svg viewBox="0 0 34 34" aria-hidden="true"><circle cx="17" cy="17" r="14" fill="none" stroke="currentColor" stroke-width="2" stroke-dasharray="4 3.3"/><path d="M12 17h10" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg><span>No boxes. If there is no speed sign, confirm “no sign”: this example is needed too.</span></p>
    </section>
    <BoxEditor :row="row" :detection="selectedBox"/>
    <SecondReading :row="row" :work="work"/>
    <OtherFields :row="row" :work="work"/>
    <ConfirmArea :row="row" :work="work"/>
  </aside>
</template>
