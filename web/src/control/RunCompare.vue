<script setup lang="ts">
// Two runs side by side: their headline numbers on the test and at their best validation epoch, and how the second
// differs from the first, green where it is better and red where it is worse. Numbers measured on different test
// sets (another split) are flagged: their difference mixes the model and the data.
import {computed, ref} from 'vue';
import {thousands} from '../common/format';
import type {RunSummary} from '../common/panel-types';
import {pct, plain, points, signed} from './evaluate';

const props = defineProps<{runs: RunSummary[]}>();
/** The last (largest) size of the cutter in a run, if it has one. */
const lastCutter = (run: RunSummary) => run.cutters?.[run.cutters.length - 1] ?? null;
type Unit = 'percent' | 'count' | 'number';
interface Measure {
  key: string;
  label: string;
  unit: Unit;
  better: 'higher' | 'lower';
  pick: (run: RunSummary) => number | null | undefined;
}
const MEASURES: Measure[] = [
  {key: 'reader', label: 'Reader accuracy', unit: 'percent', better: 'higher', pick: (run) => run.reader?.accuracy},
  {key: 'reader_human', label: 'Reader accuracy, human-reviewed', unit: 'percent', better: 'higher', pick: (run) => run.reader?.human_accuracy},
  {key: 'recall', label: 'Detector recall', unit: 'percent', better: 'higher', pick: (run) => run.detector?.recall},
  {key: 'precision', label: 'Detector precision', unit: 'percent', better: 'higher', pick: (run) => run.detector?.precision},
  {key: 'f1', label: 'Detector F1', unit: 'percent', better: 'higher', pick: (run) => run.detector?.f1},
  {key: 'recall_human', label: 'Detector recall, human-reviewed', unit: 'percent', better: 'higher', pick: (run) => run.detector?.human_recall},
  {key: 'precision_human', label: 'Detector precision, human-reviewed', unit: 'percent', better: 'higher', pick: (run) => run.detector?.human_precision},
  {key: 'found', label: 'Signs found', unit: 'count', better: 'higher', pick: (run) => run.detector?.found},
  {key: 'missed', label: 'Signs missed', unit: 'count', better: 'lower', pick: (run) => run.detector?.missed},
  {key: 'false', label: 'False boxes', unit: 'count', better: 'lower', pick: (run) => run.detector?.false_boxes},
  {key: 'val_accuracy', label: 'Reader, best validation accuracy', unit: 'percent', better: 'higher', pick: (run) => run.best.reader_val_accuracy},
  {key: 'val_loss', label: 'Detector, best validation loss', unit: 'number', better: 'lower', pick: (run) => run.best.detector_val_loss},
  // The classifier with “not a sign” (scripts/run_classifier.sh) and the chain at the cutter's score 0.1.
  {key: 'classifier', label: 'Classifier, signs read right', unit: 'percent', better: 'higher', pick: (run) => run.classifier?.accuracy},
  {key: 'classifier_human', label: 'Classifier, human-reviewed', unit: 'percent', better: 'higher', pick: (run) => run.classifier?.human_accuracy},
  {key: 'classifier_family', label: 'Classifier, other signs folded', unit: 'percent', better: 'higher', pick: (run) => run.classifier?.family_accuracy},
  {key: 'rejection', label: '“Not a sign” rejected', unit: 'percent', better: 'higher', pick: (run) => run.classifier?.rejection_rate},
  {key: 'signs_rejected', label: 'Signs thrown away', unit: 'percent', better: 'lower', pick: (run) => run.classifier?.signs_rejected_rate},
  {key: 'chain_recall', label: 'Chain, speed signs end to end', unit: 'percent', better: 'higher', pick: (run) => run.chain?.recall},
  {key: 'chain_surviving', label: 'Chain, false boxes left per frame', unit: 'number', better: 'lower', pick: (run) => run.chain?.surviving_false_per_frame},
  // The cutter of scripts/run_detector_sizes.sh (its last size) at score 0.1: the speed signs alone stay comparable
  // between a cutter of every sign and one of the speed signs only.
  {key: 'cutter_recall', label: 'Cutter, signs found at 0.1', unit: 'percent', better: 'higher', pick: (run) => lastCutter(run)?.recall},
  {key: 'cutter_speed', label: 'Cutter, speed signs found at 0.1', unit: 'percent', better: 'higher',
    pick: (run) => { const line = lastCutter(run); return line?.families === 'all' ? line.speed_recall : line?.recall; }},
  {key: 'cutter_false', label: 'Cutter, false boxes per frame', unit: 'number', better: 'lower', pick: (run) => lastCutter(run)?.false_per_frame},
];

/** By default the newest evaluated run against the one before it. */
const evaluated = computed(() => props.runs.filter((run) => run.evaluated));
const pickA = ref<string | null>(null);
const pickB = ref<string | null>(null);
const nameA = computed(() => pickA.value ?? evaluated.value[1]?.name ?? props.runs[1]?.name ?? null);
const nameB = computed(() => pickB.value ?? evaluated.value[0]?.name ?? props.runs[0]?.name ?? null);
const runA = computed(() => props.runs.find((run) => run.name === nameA.value) ?? null);
const runB = computed(() => props.runs.find((run) => run.name === nameB.value) ?? null);

function text(unit: Unit, value: number | null | undefined): string {
  if (value === null || value === undefined) return '–';
  return unit === 'percent' ? pct(value) : unit === 'count' ? thousands(value) : plain(value);
}
const rows = computed(() => {
  const a = runA.value;
  const b = runB.value;
  if (a === null || b === null || a.name === b.name) return [];
  return MEASURES.map((measure) => {
    const first = measure.pick(a) ?? null;
    const second = measure.pick(b) ?? null;
    let change = '';
    let tone = '';
    if (first !== null && second !== null) {
      const difference = second - first;
      change = measure.unit === 'percent' ? points(difference) : measure.unit === 'count' ? signed(difference)
        : difference === 0 ? '±0' : (difference > 0 ? '+' : '−') + plain(Math.abs(difference));
      const rounded = measure.unit === 'percent' ? Math.round(difference * 1000) : measure.unit === 'count' ? difference : Math.round(difference * 1000);
      if (rounded !== 0) tone = (rounded > 0) === (measure.better === 'higher') ? 'better' : 'worse';
    }
    return {...measure, first: text(measure.unit, first), second: text(measure.unit, second), change, tone};
  }).filter((row) => row.first !== '–' || row.second !== '–');
});
const sameTest = computed(() => !runA.value || !runB.value || !runA.value.split_sha256 || !runB.value.split_sha256
  || runA.value.split_sha256 === runB.value.split_sha256);
</script>

<template>
  <section class="card eval-card" id="run-compare" aria-labelledby="run-compare-title">
    <div class="ds-head"><h2 id="run-compare-title">Compare two runs</h2></div>
    <div class="compare-pick">
      <label>First<select :value="nameA ?? ''" @change="pickA = ($event.target as HTMLSelectElement).value">
        <option v-for="run in runs" :key="run.name" :value="run.name">{{ run.name }}{{ run.evaluated ? '' : ' (not evaluated)' }}</option></select></label>
      <span class="compare-arrow" aria-hidden="true">→</span>
      <label>Second<select :value="nameB ?? ''" @change="pickB = ($event.target as HTMLSelectElement).value">
        <option v-for="run in runs" :key="run.name" :value="run.name">{{ run.name }}{{ run.evaluated ? '' : ' (not evaluated)' }}</option></select></label>
    </div>
    <p v-if="!sameTest" class="ds-warn compare-warn">The two runs were measured on different test sets (another split): the difference mixes the
      model and the data.</p>
    <div v-if="rows.length" class="table-wrap"><table class="eval-table compare" id="compare-table">
      <thead><tr><th>Measure</th><th class="num">{{ nameA }}</th><th class="num">{{ nameB }}</th><th class="num">Change</th></tr></thead>
      <tbody>
        <tr v-for="row in rows" :key="row.key" :data-measure="row.key">
          <th scope="row">{{ row.label }}</th><td class="num">{{ row.first }}</td><td class="num">{{ row.second }}</td>
          <td class="num change" :class="row.tone">{{ row.change || '–' }}<span v-if="row.tone" class="sr-only">{{ row.tone }}</span></td>
        </tr>
      </tbody>
    </table></div>
    <p v-else class="empty">{{ nameA === nameB ? 'Choose two different runs.' : 'Neither run has numbers to compare yet.' }}</p>
    <p class="note">Change: the second run against the first, in percentage points (pp) for the shares.</p>
  </section>
</template>
