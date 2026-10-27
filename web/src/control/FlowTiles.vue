<script setup lang="ts">
// Three summaries under the next step, each a link to where its work is done: the drives left to prepare
// (Data › Drives), the frames left to review (Data › Review), and what the dataset holds with the reviews not imported
// yet (Data › Training dataset). The one the next step points to is marked; a dot on one with work running.
import {computed} from 'vue';
import {plural, reviewPercent, thousands} from '../common/format';
import type {PanelStatus, RouteState} from '../common/panel-types';
import Bar from '../components/Bar.vue';
import type {StepName} from './next-step';
import {reviewSummary} from './review-routes';
import type {ControlTab} from './tabs';

const props = defineProps<{status: PanelStatus | null; attention: StepName | null; busy: ReadonlySet<ControlTab>}>();
const TO_PREPARE: RouteState[] = ['collegare', 'estrarre', 'preannotare'];

const prepare = computed((): [string, string] => {
  if (!props.status) return ['–', ''];
  const count = props.status.routes.filter((row) => TO_PREPARE.includes(row.state)).length;
  return count ? [plural(count, 'drive'), 'waiting to be prepared'] : ['All prepared', 'no drive waiting'];
});
const review = computed(() => reviewSummary(props.status));
const reviewNote = computed(() => {
  const value = review.value;
  if (!props.status) return '';
  if (!value.frames) return 'nothing to review yet';
  const left = value.frames - value.confirmed;
  return left ? thousands(left) + ' frames left' : 'every frame confirmed';
});
const dataset = computed((): [string, string] => {
  const status = props.status;
  if (!status) return ['–', ''];
  const fresh = status.routes.reduce((total, row) => total + row.new_reviews, 0);
  const objects = status.dataset?.objects ?? status.routes.reduce((total, row) => total + row.objects, 0);
  return [thousands(objects) + ' objects', fresh ? plural(fresh, 'review') + ' to import'
    : status.reviewed_routes < status.min_routes ? status.reviewed_routes + ' of ' + status.min_routes + ' drives needed to train'
    : 'up to date'];
});
</script>

<template>
  <nav class="flow" aria-label="At a glance">
    <a class="flow-tile k-prepare" :class="{attention: attention === 'prepare'}" href="/prepare" id="glance-drives">
      <span class="flow-head">Drives<span v-if="busy.has('prepare')" class="flow-busy">running</span></span>
      <span class="flow-big">{{ prepare[0] }}</span><span class="flow-note">{{ prepare[1] }}</span>
    </a>
    <a class="flow-tile k-review" :class="{attention: attention === 'review'}" href="/routes" id="glance-review">
      <span class="flow-head">Review</span>
      <span class="flow-big">{{ status && review.frames ? reviewPercent(review.confirmed, review.frames) + '% confirmed' : '–' }}</span>
      <Bar :percent="review.percent" kind="ok"/>
      <span class="flow-note">{{ reviewNote }}</span>
    </a>
    <a class="flow-tile k-train" :class="{attention: attention === 'train'}" href="/dataset" id="glance-dataset">
      <span class="flow-head">Training dataset<span v-if="busy.has('dataset')" class="flow-busy">running</span></span>
      <span class="flow-big">{{ dataset[0] }}</span><span class="flow-note">{{ dataset[1] }}</span>
    </a>
  </nav>
</template>
