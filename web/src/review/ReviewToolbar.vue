<script setup lang="ts">
// The one-row bar on top: brand, route picker, filters, route progress, reviewer name and the Account menu
// (none in the offline report, which has no account).
import {computed, useTemplateRef, watch} from 'vue';
import AccountMenu from '../components/AccountMenu.vue';
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
  {filter: 'todo', label: 'Da rivedere', title: 'Non ancora confermati'},
  {filter: 'suspicious', label: 'Sospetti', title: 'Il modello e la seconda lettura non sono d’accordo'},
  {filter: 'proposals', label: 'Con proposte', title: 'Con proposte del modello'},
  {filter: 'all', label: 'Tutti'},
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
    list.unshift({value: '', text: report.routeId ? report.routeId : 'Report locale', selected: true});
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

// A save needed the name: the field takes the focus.
watch(() => state.reviewerFocus, () => reviewerInput.value?.focus(), {flush: 'sync'});
</script>

<template>
  <header ref="header" class="topbar">
    <div class="topbar-inner">
      <a class="brand" id="brand" :href="online ? '/' : undefined" :title="online ? 'Elenco delle route' : undefined"><span class="brand-mark" aria-hidden="true">50</span><span class="brand-name">Speed limits</span></a>
      <div class="route-pick">
        <label for="route" class="sr-only">Route</label>
        <select id="route" :disabled="report.routes.length <= 1" @change="changeRoute">
          <option v-for="option in options" :key="option.value + option.text" :value="option.value"
                  :selected="option.selected">{{ option.text }}</option>
        </select>
      </div>
      <div class="filters" role="group" aria-label="Quali fotogrammi mostrare">
        <button v-for="item in FILTERS" :key="item.filter" type="button" :data-filter="item.filter"
                :aria-pressed="state.filter === item.filter" :title="item.title"
                @click="pickFilter(item.filter)">{{ item.label }}</button>
        <!-- Only when an AI confirmed some frames (or while it is the filter in use). -->
        <button type="button" data-filter="ai" :aria-pressed="state.filter === 'ai'"
                title="Confermati da un revisore AI (REST API o MCP)"
                :hidden="counts.byAi === 0 && state.filter !== 'ai'" @click="pickFilter('ai')">AI · {{ counts.byAi }}</button>
      </div>
      <div class="progress" aria-live="polite">
        <div class="progress-text" id="progressText">
          <template v-if="counts.total === 0">Nessun fotogramma</template>
          <template v-else><strong>{{ counts.done }} di {{ counts.total }}</strong> confermati<template v-if="counts.suspicious > 0 || report.routeHasHints"> · <span :class="{sus: counts.suspicious > 0}">{{ counts.suspicious === 1 ? '1 sospetto' : counts.suspicious + ' sospetti' }}</span></template></template>
        </div>
        <div class="progress-bar"><span id="progressBar" :style="{width: barWidth}"></span></div>
      </div>
      <span class="progress-short" id="progressShort" aria-hidden="true">{{ counts.done }}/{{ counts.total }}</span>
      <div class="reviewer">
        <label for="reviewer" class="sr-only">Revisore</label>
        <input ref="reviewerInput" type="text" id="reviewer" :class="{need: state.reviewerNeeded}" maxlength="100"
               placeholder="Revisore" autocomplete="name" spellcheck="false" :value="state.reviewer" @input="typed">
      </div>
      <div class="account-menu" id="account-menu" data-current="false" :hidden="!online"><AccountMenu v-if="online"/></div>
    </div>
  </header>
</template>
