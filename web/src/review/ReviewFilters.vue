<script setup lang="ts">
// Which frames to show (To review, Suspicious, With proposals, All, and AI when an AI confirmed some), next to the
// frame navigation under the photo.
import {routeCounts} from './current';
import {pickFilter} from './navigation';
import {state} from './store';
import type {Filter} from './types';

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
const counts = routeCounts;
</script>

<template>
  <div class="filters" role="group" aria-label="Which frames to show">
    <button v-for="item in FILTERS" :key="item.filter" type="button" :data-filter="item.filter"
            :aria-pressed="state.filter === item.filter" :title="item.title"
            @click="pickFilter(item.filter)">{{ item.label }}</button>
    <!-- Only when an AI confirmed some frames (or while it is the filter in use). -->
    <button type="button" data-filter="ai" :aria-pressed="state.filter === 'ai'"
            title="Confirmed by an AI reviewer (REST API or MCP)"
            :hidden="counts.byAi === 0 && state.filter !== 'ai'" @click="pickFilter('ai')">AI · {{ counts.byAi }}</button>
  </div>
</template>
