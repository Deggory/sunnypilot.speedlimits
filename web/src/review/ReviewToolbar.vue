<script setup lang="ts">
// The row of the review under the site's top bar (pages.top_bar, the same on every page): filters, route progress
// and reviewer name (the drive is chosen at the top of the inspector, RoutePick.vue). On a computer it stays under the top bar while the page scrolls.
import {computed, onMounted, ref, useTemplateRef, watch} from 'vue';
import {api} from '../common/api';
import type {AuthState} from '../common/auth-types';
import {thousands} from '../common/format';
import {writeLocal} from './dom';
import {routeCounts} from './current';
import {pickFilter} from './navigation';
import {report, state} from './store';
import type {Filter} from './types';
import {useHeightVar} from './use-layout';

interface FilterButton {
  filter: Filter;
  label: string;
  title?: string;
}
const FILTERS: readonly FilterButton[] = [
  {filter: 'todo', label: 'To review', title: 'Not confirmed yet'},
  {filter: 'suspicious', label: 'Suspicious', title: 'The model and the second reading disagree'},
  {filter: 'proposals', label: 'With proposals', title: 'With proposals of the model'},
  {filter: 'all', label: 'All'},
];

const header = useTemplateRef<HTMLElement>('header');
const reviewerInput = useTemplateRef<HTMLInputElement>('reviewerInput');
/** The site's top bar, rendered by pages.py above the app: the photo and the inspector are sized under both. */
const siteBar = ref<Element | null>(document.querySelector('body > header.topbar'));
useHeightVar(siteBar, '--sitebar-h');
useHeightVar(header, '--reviewbar-h');
const online = report.apiBase !== null;

const counts = routeCounts;
const barWidth = computed(() => {
  const {done, total} = counts.value;
  return total === 0 ? '0' : Math.round(done * 100 / total) + '%';
});

function typed(event: Event): void {
  state.reviewer = (event.target as HTMLInputElement).value;
  state.reviewerNeeded = false;
  writeLocal('nnslr-reviewer', state.reviewer.trim());
}

// Signed in with an account: the reviews carry its name, and there is no field to type another one.
const signedUser = ref<string | null>(null);
onMounted(() => {
  if (!online) return;
  api<AuthState>('/api/auth/state').then((auth) => {
    if (!auth.user) return;
    signedUser.value = auth.user;
    state.reviewer = auth.user;
    state.reviewerNeeded = false;
  }).catch(() => undefined);
});

// A save needed the name: the field takes the focus.
watch(() => state.reviewerFocus, () => reviewerInput.value?.focus(), {flush: 'sync'});
</script>

<template>
  <div ref="header" class="review-bar">
    <div class="review-bar-inner">
      <div class="filters" role="group" aria-label="Which frames to show">
        <button v-for="item in FILTERS" :key="item.filter" type="button" :data-filter="item.filter"
                :aria-pressed="state.filter === item.filter" :title="item.title"
                @click="pickFilter(item.filter)">{{ item.label }}</button>
        <!-- Only when an AI confirmed some frames (or while it is the filter in use). -->
        <button type="button" data-filter="ai" :aria-pressed="state.filter === 'ai'"
                title="Confirmed by an AI reviewer (REST API or MCP)"
                :hidden="counts.byAi === 0 && state.filter !== 'ai'" @click="pickFilter('ai')">AI · {{ counts.byAi }}</button>
      </div>
      <div class="progress" aria-live="polite">
        <div class="progress-text" id="progressText">
          <template v-if="counts.total === 0">No frames</template>
          <template v-else><strong>{{ thousands(counts.done) }} of {{ thousands(counts.total) }}</strong> confirmed<template v-if="counts.suspicious > 0 || report.routeHasHints"> · <span :class="{sus: counts.suspicious > 0}">{{ thousands(counts.suspicious) }} suspicious</span></template></template>
        </div>
        <div class="progress-bar"><span id="progressBar" :style="{width: barWidth}"></span></div>
      </div>
      <span class="progress-short" id="progressShort" aria-hidden="true">{{ counts.done }}/{{ counts.total }}</span>
      <div v-if="signedUser" class="reviewer as" :title="'Your confirmations are saved as ' + signedUser">
        Reviewing as <strong>{{ signedUser }}</strong>
      </div>
      <div v-else class="reviewer">
        <label for="reviewer" class="sr-only">Reviewer</label>
        <input ref="reviewerInput" type="text" id="reviewer" :class="{need: state.reviewerNeeded}" maxlength="100"
               placeholder="Reviewer" autocomplete="name" spellcheck="false" :value="state.reviewer" @input="typed">
      </div>
    </div>
  </div>
</template>
