<script setup lang="ts">
// Experiment B of a run (nnslr_tools.context_experiment, <run>/experiment-b/<variant>/result.json): the ways of
// reading a reviewed box side by side (the crop alone; plus where the box is and the car's speed; plus the scene),
// measure by measure, on test or validation, on every box or only on those a person confirmed, the best value of a
// row in green. Then whom the sign applies to (Ours, Another, Unclear) as a confusion matrix per variant.
import {computed, ref} from 'vue';
import {isoWhen, thousands} from '../common/format';
import type {ExperimentB, Matrix} from '../common/panel-types';
import {PARTITION_LABEL, pct, plain} from './evaluate';
import Segmented from './Segmented.vue';

const props = defineProps<{experiment: ExperimentB; split: string | null}>();
type Scope = 'all' | 'human';
type Row = ExperimentB['metrics'][number];
const SCOPES: {key: Scope; label: string}[] = [{key: 'all', label: 'All boxes'}, {key: 'human', label: 'Human-reviewed'}];
const scope = ref<Scope>('all');
const chosen = ref('test');
const partitions = computed(() => props.experiment.partitions.map((key) => ({key, label: PARTITION_LABEL[key] ?? key})));
/** The partition shown: the one chosen when the report has it, else its first. */
const partition = computed(() => props.experiment.partitions.includes(chosen.value) ? chosen.value : props.experiment.partitions[0] ?? '');
const variants = computed(() => props.experiment.variants);

function value(row: Row, variant: string): number | null {
  return row.values[variant]?.[partition.value]?.[scope.value] ?? null;
}
function text(row: Row, number: number | null): string {
  if (number === null) return '–';
  if (row.unit === 'percent') return pct(number);
  if (row.unit === 'count') return thousands(Math.round(number));
  return plain(number);
}
/** The variants with the best value of a row: only with a direction, at least two values and not all equal. */
function best(row: Row): Set<string> {
  if (row.better === null) return new Set();
  const values = variants.value.map((variant) => [variant.key, value(row, variant.key)] as const)
    .filter((pair): pair is readonly [string, number] => pair[1] !== null);
  if (values.length < 2) return new Set();
  const target = row.better === 'higher' ? Math.max(...values.map((pair) => pair[1])) : Math.min(...values.map((pair) => pair[1]));
  const winners = values.filter((pair) => pair[1] === target).map((pair) => pair[0]);
  return winners.length === values.length ? new Set() : new Set(winners);
}
const rows = computed(() => props.experiment.metrics.map((row) => ({row, best: best(row)})));
const matrices = computed(() => props.experiment.confusions.map((confusion) => ({
  key: confusion.key, label: confusion.label,
  variants: variants.value.map((variant) => ({variant, matrix: confusion.variants[variant.key]?.[partition.value]?.[scope.value] ?? null})),
})));
/** Each cell's share of its row (the truth), for the shading. */
function share(matrix: Matrix, row: number, column: number): number {
  const cells = matrix.matrix[row] ?? [];
  const total = cells.reduce((sum, item) => sum + item, 0);
  return total ? (cells[column] ?? 0) / total : 0;
}
const notes = computed(() => variants.value.flatMap((variant) => {
  const lines: string[] = [];
  if (!variant.found) lines.push(variant.label + ': no result yet.');
  else if (variant.unreadable) lines.push(variant.label + ': its result.json cannot be read.');
  if (variant.limit !== null) lines.push(variant.label + ': a smoke run on ' + variant.limit + ' frames per partition, not a measure.');
  if (variant.split_sha256 && props.split && variant.split_sha256 !== props.split) {
    lines.push(variant.label + ': measured on another split than the run’s networks.');
  }
  return lines;
}));
</script>

<template>
  <section class="card eval-card" id="experiment-b" aria-labelledby="experiment-b-title">
    <div class="ds-head">
      <h2 id="experiment-b-title">Experiment B<span class="eval-what">does context teach whom a sign applies to, without hurting the reading?</span></h2>
    </div>
    <div class="seg-row">
      <Segmented v-if="partitions.length > 1" v-model="chosen" :options="partitions" label="Partition"/>
      <Segmented v-model="scope" :options="SCOPES" label="Boxes counted"/>
    </div>

    <div class="table-wrap"><table class="eval-table experiment" id="experiment-table">
      <thead>
        <tr><th>Measure</th>
          <th v-for="variant in variants" :key="variant.key" class="num variant-head" :data-variant="variant.key">
            <span class="variant-name">{{ variant.label }}</span>
            <span v-if="variant.inputs" class="variant-inputs">{{ variant.inputs.join(' + ') }}</span>
            <span v-if="variant.best_epoch !== null" class="variant-inputs">epoch {{ variant.best_epoch }}{{ variant.epochs ? ' of ' + variant.epochs : '' }}</span>
          </th></tr>
      </thead>
      <tbody>
        <tr v-for="{row, best: winners} in rows" :key="row.key" :data-metric="row.key">
          <th scope="row">{{ row.label }}<span v-if="row.better" class="better">{{ row.better === 'higher' ? 'higher is better' : 'lower is better' }}</span></th>
          <td v-for="variant in variants" :key="variant.key" class="num" :class="{best: winners.has(variant.key)}">{{ text(row, value(row, variant.key)) }}</td>
        </tr>
        <tr v-if="rows.length === 0"><td :colspan="variants.length + 1" class="muted">No measure in the results.</td></tr>
      </tbody>
    </table></div>
    <p v-if="scope === 'human'" class="note">Human-reviewed: only the boxes a person confirmed (most labels are AI reviews).</p>

    <div v-for="confusion in matrices" :key="confusion.key" class="ds-block">
      <h3 class="ds-title">{{ confusion.label }}<span class="ds-when">rows: the review · columns: the network’s answer</span></h3>
      <div class="matrices">
        <figure v-for="{variant, matrix} in confusion.variants" :key="variant.key" class="matrix">
          <figcaption>{{ variant.label }}</figcaption>
          <table v-if="matrix">
            <thead><tr><th></th><th v-for="name in matrix.names" :key="name" class="num">{{ name }}</th></tr></thead>
            <tbody>
              <tr v-for="(name, row) in matrix.names" :key="name">
                <th scope="row">{{ name }}</th>
                <td v-for="(count, column) in matrix.matrix[row]" :key="column" class="num" :class="{diag: row === column}"
                    :style="{'--share': share(matrix, row, column)}">{{ thousands(count) }}</td>
              </tr>
            </tbody>
          </table>
          <p v-else class="muted">–</p>
        </figure>
      </div>
    </div>

    <p v-if="experiment.selection_metric" class="note">Each variant keeps its best epoch by the {{ experiment.selection_metric }}.</p>
    <p v-for="(line, index) in notes" :key="index" class="note ds-note">{{ line }}</p>
    <p class="note">Results: <template v-for="(variant, index) in variants" :key="variant.key">{{ index ? ', ' : '' }}<code>{{ variant.key }}</code>
      {{ variant.at ? isoWhen(variant.at) : 'missing' }}</template>.</p>
  </section>
</template>
