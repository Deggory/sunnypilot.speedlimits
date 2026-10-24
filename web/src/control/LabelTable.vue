<script setup lang="ts">
// A table label by label (the most frequent first), as the reader's: how many crops or boxes, how many answered right,
// the accuracy as a bar, and the wrong answer given most often. The classifier's sign crops and the chain's answers on
// the cutter's boxes use it.
import {computed} from 'vue';
import {thousands} from '../common/format';
import type {LabelRow} from '../common/panel-types';
import {pct} from './evaluate';

const props = defineProps<{rows: LabelRow[]; id: string; unit?: string; right?: string}>();
const total = computed(() => props.rows.reduce((sum, row) => ({count: sum.count + row.count, correct: sum.correct + row.correct}),
  {count: 0, correct: 0}));
</script>

<template>
  <div class="table-wrap">
    <table class="eval-table labels" :id="id">
      <thead><tr><th>Label</th><th class="num">{{ unit ?? 'Crops' }}</th><th class="num">{{ right ?? 'Right' }}</th><th class="acc-col">Accuracy</th>
        <th>Most frequent wrong answer</th></tr></thead>
      <tbody>
        <tr v-for="row in rows" :key="row.label" :data-label="row.label">
          <td :title="row.label">{{ row.name }}</td>
          <td class="num">{{ thousands(row.count) }}</td>
          <td class="num">{{ thousands(row.correct) }}</td>
          <td class="acc-col"><span class="acc"><span class="acc-bar" aria-hidden="true"><span :style="{width: (row.accuracy ?? 0) * 100 + '%'}"></span></span>
            <span class="num">{{ pct(row.accuracy) }}</span></span></td>
          <td :class="{muted: !row.wrong}" :title="row.wrong?.label">{{ row.wrong ? row.wrong.name + ' · ' + row.wrong.count + '×' : 'never wrong' }}</td>
        </tr>
        <tr v-if="rows.length === 0"><td colspan="5" class="muted">None.</td></tr>
      </tbody>
      <tfoot v-if="rows.length > 1"><tr><th scope="row">Total</th><td class="num">{{ thousands(total.count) }}</td><td class="num">{{ thousands(total.correct) }}</td>
        <td class="acc-col num">{{ pct(total.count ? total.correct / total.count : null) }}</td><td></td></tr></tfoot>
    </table>
  </div>
</template>
