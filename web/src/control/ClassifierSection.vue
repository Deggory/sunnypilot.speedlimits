<script setup lang="ts">
// The classifier with “not a sign” alone, on the crops of the test or of validation (evaluation/classifier-<partition>.json):
// the sign crops it reads right (also with the kinds of other signs folded, as the old reader reads them), on every
// crop or on those a person confirmed, the sign crops it throws away as “not a sign”; the “not a sign” crops it rejects,
// by where they come from (boxes reviewed as not a sign, proposals a reviewer rejected, false boxes of the cutter), and
// what it calls the ones it lets through; then the sign crops label by label.
import {computed, ref} from 'vue';
import {plural, thousands} from '../common/format';
import type {ReaderEvaluation} from '../common/panel-types';
import {PARTITION_LABEL, pct} from './evaluate';
import LabelTable from './LabelTable.vue';
import Segmented from './Segmented.vue';

const props = defineProps<{evaluations: {test: ReaderEvaluation | null; validation: ReaderEvaluation | null}}>();
type Partition = 'test' | 'validation';
type Scope = 'all' | 'human';
const SCOPES: {key: Scope; label: string}[] = [{key: 'all', label: 'All crops'}, {key: 'human', label: 'Human-reviewed'}];
const chosen = ref<Partition>('test');
const scope = ref<Scope>('all');
const partitions = computed(() => (['test', 'validation'] as Partition[]).filter((key) => props.evaluations[key] !== null)
  .map((key) => ({key, label: PARTITION_LABEL[key] ?? key})));
/** The partition shown: the one chosen when measured, else the other. */
const report = computed((): ReaderEvaluation | null => props.evaluations[chosen.value] ?? props.evaluations.test ?? props.evaluations.validation);
const partition = computed(() => props.evaluations[chosen.value] ? chosen.value : props.evaluations.test ? 'test' : 'validation');
const signs = computed(() => report.value?.signs ?? null);
const rejection = computed(() => report.value?.not_a_sign ?? null);
/** The sign crops label by label (every crop of an older report, which has no “not a sign”). */
const rows = computed(() => {
  const value = report.value;
  if (value === null) return [];
  const own = signs.value;
  if (own) return scope.value === 'human' && own.labels_human ? own.labels_human : own.labels;
  return scope.value === 'human' && value.labels_human ? value.labels_human : value.labels;
});
const hasHuman = computed(() => Boolean(signs.value ? signs.value.labels_human : report.value?.labels_human));
const unseen = computed(() => signs.value?.unseen_labels ?? report.value?.unseen_labels ?? []);
const topSurvivor = computed(() => Math.max(1, ...(rejection.value?.survivors ?? []).map((row) => row.count)));
</script>

<template>
  <section v-if="report" class="card eval-card" id="classifier-eval" aria-labelledby="classifier-eval-title">
    <div class="ds-head">
      <h2 id="classifier-eval-title">Classifier alone<span class="eval-what">says what each reviewed box is, “not a sign” included</span></h2>
      <Segmented v-if="partitions.length > 1" v-model="chosen" :options="partitions" label="Partition"/>
    </div>
    <div class="ds-figures">
      <div class="ds-figure"><span class="ds-value" id="classifier-accuracy">{{ pct(signs ? signs.accuracy : report.accuracy) }}</span>
        <span class="ds-unit">right, of {{ plural(signs ? signs.samples : report.samples, 'sign crop') }}</span></div>
      <div v-if="signs?.family_accuracy != null" class="ds-figure"><span class="ds-value" id="classifier-family">{{ pct(signs.family_accuracy) }}</span>
        <span class="ds-unit">with the kinds of other signs folded</span></div>
      <div v-if="(signs ?? report).human" class="ds-figure"><span class="ds-value" id="classifier-human">{{ pct((signs ?? report).human?.accuracy) }}</span>
        <span class="ds-unit">of the {{ thousands((signs ?? report).human?.samples ?? 0) }} a person confirmed</span></div>
    </div>

    <ul v-if="signs || rejection" class="box-counts" aria-label="“Not a sign”">
      <li v-if="rejection" class="found" id="classifier-rejection"><strong>{{ pct(rejection.rejection_rate) }}</strong>“not a sign” rejected<span>{{
        thousands(rejection.rejected) }} of {{ thousands(rejection.crops) }} crops that are no sign</span></li>
      <li v-if="signs" class="missed" id="classifier-thrown"><strong>{{ pct(signs.rejected_rate) }}</strong>signs thrown away<span>{{
        signs.rejected === null ? '' : thousands(signs.rejected) + ' sign crops answered “not a sign”' }}</span></li>
      <li v-if="rejection?.human"><strong>{{ pct(rejection.human.rejection_rate) }}</strong>rejected, human-reviewed<span>{{
        thousands(rejection.human.rejected) }} of {{ thousands(rejection.human.crops) }} on frames a person confirmed</span></li>
    </ul>

    <div v-if="rejection?.by_source.length" class="ds-block">
      <h3 class="ds-title">“Not a sign” rejected, by where the crop comes from</h3>
      <ul class="barlist share" id="classifier-sources" aria-label="“Not a sign” rejected by source">
        <li v-for="row in rejection.by_source" :key="row.source" :data-source="row.source"
            :title="row.label + ': ' + row.rejected + ' rejected of ' + row.crops">
          <span class="bl-label">{{ row.label }}</span>
          <span class="bl-track" aria-hidden="true"><span :style="{width: Math.max((row.rejection_rate ?? 0) * 100, row.crops ? 0.6 : 0) + '%'}"></span></span>
          <span class="bl-value">{{ row.crops ? pct(row.rejection_rate) : '–' }}<small>{{ thousands(row.rejected) }}/{{ thousands(row.crops) }}</small></span>
        </li>
      </ul>
    </div>

    <div v-if="rejection?.survivors.length" class="ds-block">
      <h3 class="ds-title">What it calls the “not a sign” crops it lets through</h3>
      <ul class="barlist" id="classifier-survivors">
        <li v-for="row in rejection.survivors" :key="row.label" :data-label="row.label" :title="row.label">
          <span class="bl-label">{{ row.name }}</span>
          <span class="bl-track" aria-hidden="true"><span :style="{width: row.count * 100 / topSurvivor + '%'}"></span></span>
          <span class="bl-value">{{ thousands(row.count) }}</span>
        </li>
      </ul>
    </div>

    <div class="ds-block">
      <h3 class="ds-title">{{ signs ? 'Sign crops label by label' : 'Label by label' }}<span class="ds-when">{{
        (PARTITION_LABEL[partition] ?? partition).toLowerCase() }}</span></h3>
      <Segmented v-if="hasHuman" v-model="scope" :options="SCOPES" label="Crops in the table"/>
      <LabelTable :rows="rows" id="classifier-labels"/>
    </div>
    <p v-if="unseen.length" class="note">In the {{ partition }} but never learnt (no training crop): {{ unseen.join(', ') }}.</p>
  </section>
</template>
