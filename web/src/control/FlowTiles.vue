<script setup lang="ts">
// The three phases at a glance: what is left to prepare (a link to Prepare), the review (a card of its own:
// how much is confirmed, Resume review and the route list) and whether there are enough reviewed routes to
// train (a link to Train). The one the next step points to is marked.
import {computed} from 'vue';
import {plural, thousands} from '../common/format';
import type {PanelStatus, RouteState} from '../common/panel-types';
import Bar from '../components/Bar.vue';
import Icon from '../components/Icon.vue';
import type {StepName} from './next-step';
import {reviewSummary} from './review-routes';
import type {ControlTab} from './tabs';

const props = defineProps<{status: PanelStatus | null; attention: StepName | null; busy: ReadonlySet<ControlTab>}>();
const TO_PREPARE: RouteState[] = ['collegare', 'estrarre', 'preannotare'];

const prepare = computed((): [string, string] => {
  if (!props.status) return ['–', ''];
  const count = props.status.routes.filter((row) => TO_PREPARE.includes(row.state)).length;
  return count ? [plural(count, 'new drive'), 'to turn into frames'] : ['All prepared', 'no video waiting'];
});
const review = computed(() => reviewSummary(props.status));
/** "2 drives in progress · 4 to start", or how the review stands when nothing is left. */
const reviewLine = computed(() => {
  const value = review.value;
  if (!props.status) return '';
  if (!value.frames) return 'nothing to review yet';
  const parts: string[] = [];
  if (value.inProgress) parts.push(plural(value.inProgress, 'drive') + ' in progress');
  if (value.toStart) parts.push((parts.length ? thousands(value.toStart) : plural(value.toStart, 'drive')) + ' to start');
  return parts.length ? parts.join(' · ') : 'every drive reviewed';
});
const train = computed((): [string, string] => {
  if (!props.status) return ['–', ''];
  const {reviewed_routes: reviewed, min_routes: needed, routes} = props.status;
  const fresh = routes.reduce((total, row) => total + row.new_reviews, 0);
  const figure = reviewed >= needed ? plural(reviewed, 'drive') + ' ready' : reviewed + ' of ' + needed + ' drives';
  return [figure, fresh ? thousands(fresh) + ' frames to import' : reviewed >= needed ? 'ready to train' : needed + ' are needed to train'];
});
</script>

<template>
  <nav class="flow" aria-label="The three phases">
    <a class="flow-tile k-prepare" :class="{attention: attention === 'prepare'}" href="/prepare">
      <span class="flow-head"><span class="flow-num">1</span>Prepare<span v-if="busy.has('prepare')" class="flow-busy">running</span></span>
      <span class="flow-big">{{ prepare[0] }}</span><span class="flow-note">{{ prepare[1] }}</span>
    </a>
    <section class="flow-tile flow-review k-review" :class="{attention: attention === 'review'}" id="review-card" aria-labelledby="review-card-title">
      <h2 class="flow-head" id="review-card-title"><span class="flow-num">2</span>Review</h2>
      <span class="flow-big">{{ status && review.frames ? Math.floor(review.percent) + '%' : '–' }}</span>
      <Bar :percent="review.percent" kind="ok"/>
      <span class="flow-note">{{ status && review.frames ? thousands(review.confirmed) + ' of ' + thousands(review.frames) + ' frames confirmed' : '' }}</span>
      <span class="flow-note" id="review-routes-line">{{ reviewLine }}</span>
      <div class="flow-actions">
        <a v-if="review.resume" class="btn small primary" id="resume-review" :href="review.resume.href"
           :title="review.resume.route + ' · ' + thousands(review.resume.left) + ' frames left'"><Icon name="resume"/>Resume review</a>
        <a class="flow-link" id="all-routes" href="/routes"><Icon name="routes"/>All drives →</a>
      </div>
    </section>
    <a class="flow-tile k-train" :class="{attention: attention === 'train'}" href="/train">
      <span class="flow-head"><span class="flow-num">3</span>Train<span v-if="busy.has('train')" class="flow-busy">running</span></span>
      <span class="flow-big">{{ train[0] }}</span><span class="flow-note">{{ train[1] }}</span>
    </a>
  </nav>
</template>
