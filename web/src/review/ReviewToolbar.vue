<script setup lang="ts">
// The one-row bar on top: brand, route picker, filters, route progress, reviewer name and the Account menu
// (none in the offline report, which has no account).
import {computed, onMounted, ref, useTemplateRef, watch} from 'vue';
import {api} from '../common/api';
import type {AuthState} from '../common/auth-types';
import {thousands} from '../common/format';
import AccountMenu from '../components/AccountMenu.vue';
import BrandMark from '../components/BrandMark.vue';
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
useHeightVar(header, '--topbar-h');
const online = report.apiBase !== null;

/** The routes of the server, this one chosen; offline only this report. */
const options = computed(() => {
  const list = report.routes.map((route) => ({
    value: String(route.url || ''), text: String(route.route_id || ''), selected: route.route_id === report.routeId,
  }));
  if (!list.some((option) => option.selected)) {
    list.unshift({value: '', text: report.routeId ? report.routeId : 'Local report', selected: true});
  }
  return list;
});

const counts = routeCounts;
const barWidth = computed(() => {
  const {done, total} = counts.value;
  return total === 0 ? '0' : Math.round(done * 100 / total) + '%';
});

function changeRoute(event: Event): void {
  const value = (event.target as HTMLSelectElement).value;
  if (value.startsWith('/route/')) window.location.assign(value);
}

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
  <header ref="header" class="topbar">
    <div class="topbar-inner">
      <a class="brand" id="brand" :href="online ? '/' : undefined" :title="online ? 'Overview' : undefined"><BrandMark/><span class="brand-name">Speed limits</span></a>
      <div class="route-pick">
        <label for="route" class="sr-only">Route</label>
        <select id="route" :disabled="report.routes.length <= 1" @change="changeRoute">
          <option v-for="option in options" :key="option.value + option.text" :value="option.value"
                  :selected="option.selected">{{ option.text }}</option>
        </select>
      </div>
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
      <div class="account-menu" id="account-menu" data-current="false" :hidden="!online"><AccountMenu v-if="online"/></div>
    </div>
  </header>
</template>
