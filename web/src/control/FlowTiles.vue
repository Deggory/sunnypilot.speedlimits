<script setup lang="ts">
// The three phases at a glance, each a link to its tab: what is left to prepare, how much is reviewed,
// whether there are enough reviewed routes to train. The one the next step points to is marked.
import {computed} from 'vue';
import {thousands} from '../common/format';
import type {PanelStatus, RouteState} from '../common/panel-types';
import Bar from '../components/Bar.vue';
import type {StepName} from './next-step';
import type {ControlTab} from './tabs';

const props = defineProps<{status: PanelStatus | null; attention: StepName | null; busy: ReadonlySet<ControlTab>}>();
const TO_PREPARE: RouteState[] = ['collegare', 'estrarre', 'preannotare'];

const prepare = computed((): [string, string] => {
  if (!props.status) return ['–', ''];
  const count = props.status.routes.filter((row) => TO_PREPARE.includes(row.state)).length;
  return count ? [count + (count === 1 ? ' route nuova' : ' route nuove'), 'da trasformare in fotogrammi']
    : ['Tutto preparato', 'nessun video in attesa'];
});
const review = computed(() => {
  const routes = props.status?.routes ?? [];
  const reported = routes.filter((row) => row.run_id);
  const frames = reported.reduce((total, row) => total + row.report_frames, 0);
  const confirmed = reported.reduce((total, row) => total + Math.min(row.confirmed, row.report_frames), 0);
  const left = routes.filter((row) => row.state === 'revisionare').length;
  return {frames, confirmed, percent: frames ? confirmed * 100 / frames : 0,
    note: !frames ? 'niente da revisionare' : left ? left + ' route da revisionare' : 'tutte revisionate'};
});
const train = computed((): [string, string] => {
  if (!props.status) return ['–', ''];
  const {reviewed_routes: reviewed, min_routes: needed, routes} = props.status;
  const fresh = routes.reduce((total, row) => total + row.new_reviews, 0);
  const figure = reviewed >= needed ? reviewed + ' route pronte' : reviewed + ' di ' + needed + ' route';
  return [figure, fresh ? thousands(fresh) + ' fotogrammi da importare' : reviewed >= needed ? 'si può allenare' : 'ne servono ' + needed];
});
</script>

<template>
  <nav class="flow" aria-label="Le tre fasi">
    <a class="flow-tile k-prepare" :class="{attention: attention === 'prepare'}" href="/prepara">
      <span class="flow-head"><span class="flow-num">1</span>Prepara<span v-if="busy.has('prepare')" class="flow-busy">in corso</span></span>
      <span class="flow-big">{{ prepare[0] }}</span><span class="flow-note">{{ prepare[1] }}</span>
    </a>
    <a class="flow-tile k-review" :class="{attention: attention === 'review'}" href="/">
      <span class="flow-head"><span class="flow-num">2</span>Revisiona</span>
      <span class="flow-big">{{ status ? (review.frames ? Math.floor(review.percent) + '%' : '–') : '–' }}</span>
      <Bar :percent="review.percent" kind="ok"/>
      <span class="flow-note">{{ status && review.frames ? thousands(review.confirmed) + ' di ' + thousands(review.frames) + ' fotogrammi · ' + review.note : review.note }}</span>
    </a>
    <a class="flow-tile k-train" :class="{attention: attention === 'train'}" href="/allena">
      <span class="flow-head"><span class="flow-num">3</span>Allena<span v-if="busy.has('train')" class="flow-busy">in corso</span></span>
      <span class="flow-big">{{ train[0] }}</span><span class="flow-note">{{ train[1] }}</span>
    </a>
  </nav>
</template>
