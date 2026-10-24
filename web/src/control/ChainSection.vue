<script setup lang="ts">
// The chain end to end (evaluation/chain-<partition>.json): the cutter draws its boxes on the frames, the classifier says
// what each is. At a score of the cutter (0.1 by default, the cutter's own): the speed signs found and read right
// (end to end) next to the share the cutter alone finds, overall, on the frames a person confirmed and by the width of
// the sign; the boxes and false boxes per frame the cutter passes on against the false ones the classifier lets
// through, and what it calls them; the old reader in the same chain side by side (it has no “not a sign”: every box
// survives); the classifier's answers label by label at the main score.
import {computed, ref} from 'vue';
import {thousands} from '../common/format';
import type {ChainEvaluation, ChainThreshold} from '../common/panel-types';
import {PARTITION_LABEL, pct, plain, points} from './evaluate';
import LabelTable from './LabelTable.vue';
import {perFrame} from './models';
import Segmented from './Segmented.vue';

const props = defineProps<{chains: {test: ChainEvaluation | null; validation: ChainEvaluation | null; old_reader: ChainEvaluation | null}}>();
type Partition = 'test' | 'validation';
const CHAIN_SCORE = 0.1;
const chosen = ref<Partition>('test');
const partitions = computed(() => (['test', 'validation'] as Partition[]).filter((key) => props.chains[key] !== null)
  .map((key) => ({key, label: PARTITION_LABEL[key] ?? key})));
const report = computed((): ChainEvaluation | null => props.chains[chosen.value] ?? props.chains.test ?? props.chains.validation);
const partition = computed(() => report.value?.partition ?? 'test');

/** The score of the cutter shown: the one chosen when the report has it, else 0.1, else the report's main one. */
const pickedScore = ref<string | null>(null);
const scores = computed(() => (report.value?.thresholds ?? []).map((item) => ({key: String(item.score), label: 'score ' + item.score})));
const near = (a: number, b: number | null | undefined): boolean => b !== null && b !== undefined && Math.abs(a - b) < 1e-9;
const entry = computed((): ChainThreshold | null => {
  const list = report.value?.thresholds ?? [];
  return list.find((item) => String(item.score) === pickedScore.value) ?? list.find((item) => near(item.score, CHAIN_SCORE))
    ?? list.find((item) => near(item.score, report.value?.main_score_threshold)) ?? list[0] ?? null;
});
const score = computed({
  get: (): string => entry.value === null ? '' : String(entry.value.score),
  set: (value: string) => { pickedScore.value = value; },
});
const signs = computed(() => entry.value?.speed_signs ?? null);
const every = computed(() => entry.value?.all_boxes ?? null);
/** The old reader in the same chain, at the same score: only on the test. */
const old = computed((): ChainThreshold | null => {
  if (partition.value !== 'test' || entry.value === null) return null;
  return props.chains.old_reader?.thresholds.find((item) => near(item.score, entry.value?.score)) ?? null;
});
const topSurvivor = computed(() => Math.max(1, ...(entry.value?.surviving_false_by_label ?? []).map((row) => row.count)));

type Unit = 'percent' | 'number';
const ROWS: {key: string; label: string; unit: Unit; better: 'higher' | 'lower' | null; pick: (item: ChainThreshold) => number | null | undefined}[] = [
  {key: 'recall', label: 'Speed signs, end to end', unit: 'percent', better: 'higher', pick: (item) => item.speed_signs?.recall},
  {key: 'recall_found', label: 'Speed signs found by the cutter', unit: 'percent', better: null, pick: (item) => item.speed_signs?.recall_found},
  {key: 'read_right', label: 'Read right once found', unit: 'percent', better: 'higher', pick: (item) => item.speed_signs?.read_right_when_found},
  {key: 'human', label: 'Speed signs end to end, human-reviewed', unit: 'percent', better: 'higher', pick: (item) => item.speed_signs?.human?.recall},
  {key: 'all', label: 'Every reviewed box, end to end', unit: 'percent', better: 'higher', pick: (item) => item.all_boxes?.recall},
  {key: 'false', label: 'False boxes per frame from the cutter', unit: 'number', better: null, pick: (item) => item.false_per_frame},
  {key: 'surviving', label: 'False boxes per frame left', unit: 'number', better: 'lower', pick: (item) => item.surviving_false_per_frame},
  {key: 'rejected', label: 'False boxes rejected', unit: 'percent', better: 'higher', pick: (item) => item.false_rejected_rate},
];
const text = (unit: Unit, value: number | null | undefined): string => unit === 'percent' ? pct(value) : perFrame(value);
const compared = computed(() => {
  const mine = entry.value;
  const theirs = old.value;
  if (mine === null || theirs === null) return [];
  return ROWS.map((row) => {
    const first = row.pick(theirs) ?? null;
    const second = row.pick(mine) ?? null;
    let change = '';
    let tone = '';
    if (first !== null && second !== null) {
      const difference = second - first;
      change = row.unit === 'percent' ? points(difference)
        : Math.round(difference * 100) === 0 ? '±0' : (difference > 0 ? '+' : '−') + plain(Math.abs(difference));
      const rounded = Math.round(difference * (row.unit === 'percent' ? 1000 : 100));
      if (rounded !== 0 && row.better !== null) tone = (rounded > 0) === (row.better === 'higher') ? 'better' : 'worse';
    }
    return {...row, old: text(row.unit, first), now: text(row.unit, second), change, tone};
  }).filter((row) => row.old !== '–' || row.now !== '–');
});
const warnings = computed(() => {
  const value = report.value;
  if (value === null) return [];
  const lines: string[] = [];
  if (value.cutter.same_split === false) lines.push('The cutter learnt on another split: some of these frames may be ones it was trained on.');
  if (value.classifier.same_split === false) lines.push('The classifier learnt on another split: some of these boxes may be ones it was trained on.');
  return lines;
});
</script>

<template>
  <section v-if="report" class="card eval-card" id="chain-eval" aria-labelledby="chain-eval-title">
    <div class="ds-head">
      <h2 id="chain-eval-title">Chain end to end<span class="eval-what">the cutter finds the boxes, the classifier says what they are</span></h2>
      <span class="ds-when">{{ thousands(report.image_count ?? 0) }} frames · overlap (IoU) ≥ {{ report.iou_threshold ?? '–' }}</span>
    </div>
    <div class="seg-row">
      <Segmented v-if="partitions.length > 1" v-model="chosen" :options="partitions" label="Partition"/>
      <Segmented v-if="scores.length > 1" v-model="score" :options="scores" label="Score of the cutter"/>
    </div>
    <p v-for="line in warnings" :key="line" class="compare-warn">{{ line }}</p>

    <template v-if="entry">
      <div class="ds-figures">
        <div class="ds-figure"><span class="ds-value" id="chain-recall">{{ pct(signs?.recall) }}</span>
          <span class="ds-unit">of {{ thousands(signs?.truth ?? 0) }} speed signs found and read right</span></div>
        <div class="ds-figure"><span class="ds-value" id="chain-found">{{ pct(signs?.recall_found) }}</span>
          <span class="ds-unit">found by the cutter alone</span></div>
        <div class="ds-figure"><span class="ds-value">{{ pct(signs?.read_right_when_found) }}</span>
          <span class="ds-unit">read right once found</span></div>
      </div>
      <p v-if="signs?.human" class="note">On the {{ thousands(report.human_image_count ?? 0) }} frames a person confirmed: {{ pct(signs.human.recall) }} end to end
        ({{ thousands(signs.human.correct ?? 0) }} of {{ thousands(signs.human.truth ?? 0) }}), the cutter alone {{ pct(signs.human.recall_found) }}.</p>

      <ul class="box-counts" aria-label="Boxes per frame">
        <li><strong>{{ perFrame(entry.boxes_per_frame) }}</strong>boxes per frame<span>passed on by the cutter at score {{ entry.score }}</span></li>
        <li class="false"><strong>{{ perFrame(entry.false_per_frame) }}</strong>false per frame<span>{{ entry.false_boxes === null ? 'from the cutter'
          : thousands(entry.false_boxes) + ' false boxes from the cutter' }}</span></li>
        <li class="found" id="chain-surviving"><strong>{{ perFrame(entry.surviving_false_per_frame) }}</strong>false per frame left<span>after the
          classifier{{ entry.false_rejected_rate === null ? '' : ', which rejects ' + pct(entry.false_rejected_rate) + ' of them' }}</span></li>
      </ul>
      <p v-if="entry.human" class="note">On the frames a person confirmed: {{ perFrame(entry.human.false_per_frame) }} false boxes per frame from the cutter,
        {{ perFrame(entry.human.surviving_false_per_frame) }} left after the classifier.</p>

      <div v-if="compared.length" class="ds-block">
        <h3 class="ds-title">Against the old reader<span class="ds-when">the same chain, the reader of the training pipeline as classifier</span></h3>
        <div class="table-wrap"><table class="eval-table compare" id="chain-compare">
          <thead><tr><th>Measure</th><th class="num">Old reader</th><th class="num">This classifier</th><th class="num">Change</th></tr></thead>
          <tbody>
            <tr v-for="row in compared" :key="row.key" :data-measure="row.key">
              <th scope="row">{{ row.label }}</th><td class="num">{{ row.old }}</td><td class="num">{{ row.now }}</td>
              <td class="num change" :class="row.tone">{{ row.change || '–' }}<span v-if="row.tone" class="sr-only">{{ row.tone }}</span></td>
            </tr>
          </tbody>
        </table></div>
        <p class="note">The old reader has no “not a sign”: every box of the cutter reaches the next step.</p>
      </div>

      <div v-if="signs?.widths" class="ds-block">
        <h3 class="ds-title">Speed signs by size<span class="ds-when">end to end · on the right, found by the cutter alone</span></h3>
        <ul class="barlist sizes" id="chain-sizes" aria-label="Speed signs found and read right by size">
          <li v-for="row in signs.widths" :key="row.key" :data-size="row.key"
              :title="row.label + ': ' + (row.correct ?? 0) + ' found and read right of ' + (row.truth ?? 0) + ', ' + (row.found ?? 0) + ' found by the cutter'">
            <span class="bl-label">{{ row.label }}</span>
            <span class="bl-track" aria-hidden="true"><span :style="{width: Math.max((row.recall ?? 0) * 100, row.truth ? 0.6 : 0) + '%'}"></span></span>
            <span class="bl-value">{{ row.truth ? pct(row.recall) : '–' }}<small>{{ thousands(row.correct ?? 0) }}/{{ thousands(row.truth ?? 0) }}</small></span>
            <span class="bl-extra">{{ row.truth ? pct(row.recall_found) : '' }}</span>
          </li>
        </ul>
      </div>
      <p v-if="every" class="note">Every reviewed box, other signs included ({{ thousands(every.truth) }}): {{ pct(every.recall) }} found and read right,
        {{ pct(every.recall_found) }} found by the cutter.</p>

      <div v-if="entry.surviving_false_by_label.length" class="ds-block">
        <h3 class="ds-title">False boxes left, by what the classifier calls them</h3>
        <ul class="barlist" id="chain-survivors">
          <li v-for="row in entry.surviving_false_by_label" :key="row.label" :data-label="row.label" :title="row.label">
            <span class="bl-label">{{ row.name }}</span>
            <span class="bl-track" aria-hidden="true"><span :style="{width: row.count * 100 / topSurvivor + '%'}"></span></span>
            <span class="bl-value">{{ thousands(row.count) }}</span>
          </li>
        </ul>
      </div>

      <div v-if="entry.labels" class="ds-block">
        <h3 class="ds-title">The classifier’s answers on the cutter’s boxes<span class="ds-when">by the review of the box; a false box is “not a sign”</span></h3>
        <LabelTable :rows="entry.labels" id="chain-labels" unit="Boxes"/>
      </div>
    </template>
    <p v-else class="empty">The report has no score of the cutter.</p>

    <p class="note">Cutter: <code :title="report.cutter.checkpoint ?? ''">{{ report.cutter.id ?? report.cutter.checkpoint ?? '–' }}</code><template
      v-if="report.cutter.input_size"> at {{ report.cutter.input_size.replace('x', ' × ') }}</template> · classifier:
      {{ report.classifier.rejects ? 'with “not a sign”' : 'without “not a sign”' }}{{ report.classifier.classes ? ', ' + report.classifier.classes + ' classes' : '' }}<template
      v-if="report.seconds !== null"> · measured in {{ Math.round(report.seconds) }} s{{ report.device ? ' on ' + report.device : '' }}</template>.</p>
    <p v-for="(line, index) in report.limitations" :key="index" class="note ds-note">{{ line }}</p>
  </section>
</template>
