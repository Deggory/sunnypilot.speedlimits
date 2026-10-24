<script setup lang="ts">
// The reader on the test: its accuracy on every crop and on the crops a person confirmed, then label by label (the
// most frequent first): how many crops, how many read right, and the wrong answer it gives most often. The table
// shows every crop or only the human-reviewed ones.
import {computed, ref} from 'vue';
import {plural, thousands} from '../common/format';
import type {ReaderEvaluation} from '../common/panel-types';
import {pct} from './evaluate';
import Segmented from './Segmented.vue';

const props = defineProps<{reader: ReaderEvaluation}>();
type Scope = 'all' | 'human';
const SCOPES: {key: Scope; label: string}[] = [{key: 'all', label: 'All crops'}, {key: 'human', label: 'Human-reviewed'}];
const scope = ref<Scope>('all');
const rows = computed(() => scope.value === 'human' && props.reader.labels_human ? props.reader.labels_human : props.reader.labels);
const total = computed(() => rows.value.reduce((sum, row) => ({count: sum.count + row.count, correct: sum.correct + row.correct}),
  {count: 0, correct: 0}));
</script>

<template>
  <section class="card eval-card" id="reader-eval" aria-labelledby="reader-eval-title">
    <div class="ds-head">
      <h2 id="reader-eval-title">Reader<span class="eval-what">reads the limit on a reviewed box</span></h2>
      <Segmented v-if="reader.labels_human" v-model="scope" :options="SCOPES" label="Crops in the table"/>
    </div>
    <div class="ds-figures">
      <div class="ds-figure"><span class="ds-value" id="reader-accuracy">{{ pct(reader.accuracy) }}</span>
        <span class="ds-unit">right, of {{ plural(reader.samples, 'test crop') }}</span></div>
      <div v-if="reader.human" class="ds-figure"><span class="ds-value" id="reader-human-accuracy">{{ pct(reader.human.accuracy) }}</span>
        <span class="ds-unit">of the {{ thousands(reader.human.samples) }} a person confirmed</span></div>
      <div v-else class="ds-figure"><span class="ds-unit">No test crop confirmed by a person.</span></div>
    </div>

    <div class="table-wrap">
      <table class="eval-table labels" id="reader-labels">
        <thead><tr><th>Label</th><th class="num">Crops</th><th class="num">Right</th><th class="acc-col">Accuracy</th><th>Most frequent wrong answer</th></tr></thead>
        <tbody>
          <tr v-for="row in rows" :key="row.label" :data-label="row.label">
            <td :title="row.label">{{ row.name }}</td>
            <td class="num">{{ thousands(row.count) }}</td>
            <td class="num">{{ thousands(row.correct) }}</td>
            <td class="acc-col"><span class="acc"><span class="acc-bar" aria-hidden="true"><span :style="{width: (row.accuracy ?? 0) * 100 + '%'}"></span></span>
              <span class="num">{{ pct(row.accuracy) }}</span></span></td>
            <td :class="{muted: !row.wrong}" :title="row.wrong?.label">{{ row.wrong ? row.wrong.name + ' · ' + row.wrong.count + '×' : 'never wrong' }}</td>
          </tr>
          <tr v-if="rows.length === 0"><td colspan="5" class="muted">No crop.</td></tr>
        </tbody>
        <tfoot v-if="rows.length > 1"><tr><th scope="row">Total</th><td class="num">{{ thousands(total.count) }}</td><td class="num">{{ thousands(total.correct) }}</td>
          <td class="acc-col num">{{ pct(total.count ? total.correct / total.count : null) }}</td><td></td></tr></tfoot>
      </table>
    </div>
    <p v-if="reader.unseen_labels.length" class="note">In the test but never learnt (no training crop): {{ reader.unseen_labels.join(', ') }}.</p>
  </section>
</template>
