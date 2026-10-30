<script setup lang="ts">
// The headline numbers of a trained version side by side (ModelItem.vue): the speed signs, every sign, and for a
// classifier the chain with the cutter. Each tile: what it counts, the percentage, a bar that fills, how many of how
// many, and a smaller note when it has one. A tile without a number (a cutter trained on speed signs only has no "every sign") says why instead.
import {pct} from './evaluate';
import type {Metric} from './models';

defineProps<{metrics: Metric[]; caption?: string}>();
</script>

<template>
  <div class="metrics" :class="{three: metrics.length > 2}">
    <div v-for="(metric, index) in metrics" :key="metric.key" class="metric" :class="{main: index === 0, none: metric.value === null}"
         :data-metric="metric.key" :data-value="metric.value ?? undefined">
      <span class="metric-label">{{ metric.label }}</span>
      <strong class="metric-value">{{ metric.value === null ? '–' : pct(metric.value) }}</strong>
      <span class="metric-bar" aria-hidden="true"><span :style="{width: Math.max(0, Math.min(1, metric.value ?? 0)) * 100 + '%'}"></span></span>
      <span class="metric-detail">{{ metric.detail }}</span>
      <span v-if="metric.note" class="metric-note">{{ metric.note }}</span>
    </div>
  </div>
  <p v-if="caption" class="metric-caption">{{ caption }}</p>
</template>
