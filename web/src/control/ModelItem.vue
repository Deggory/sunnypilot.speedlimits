<script setup lang="ts">
// One trained version of a role: its run, what it is and when its weights were saved, the headline numbers that judge
// that role (the cutter: signs found overall and by width at a low score, boxes per frame, time per frame; the
// classifier: right answers on the test; the decider: the "Another" road found, on validation and test), the button
// that puts it in use and the link to its run on Evaluate.
import {computed} from 'vue';
import {isoWhen, thousands} from '../common/format';
import type {DeciderNumbers, ModelVersion} from '../common/panel-types';
import {PARTITION_LABEL, pct} from './evaluate';
import {epochText, initText, openRun, perFrame, sizeText} from './models';

const props = defineProps<{version: ModelVersion; roleLabel: string; score: number; readOnly: boolean; busy: string | null}>();
const emit = defineEmits<{use: [version: ModelVersion]}>();

const cutter = computed(() => props.version.role === 'cutter' ? props.version : null);
const classifier = computed(() => props.version.role === 'classifier' ? props.version : null);
const decider = computed(() => props.version.role === 'decider' ? props.version : null);
/** The cutter's numbers: on the test, else on validation (a size compared before its test was measured). */
const cut = computed(() => cutter.value?.test ?? cutter.value?.validation ?? null);
const evaluated = computed(() => cut.value !== null || (classifier.value?.test ?? null) !== null
  || decider.value?.validation != null || decider.value?.test != null);
const epoch = computed(() => epochText(props.version.best_epoch, props.version.epochs_run ?? props.version.epochs));

/** Why the button is off, as its title. */
const blocked = computed((): string => {
  if (props.readOnly) return 'Read-only: the choice is turned off on this server.';
  if (!props.version.finished) return 'Its training has not finished: its weights may still change.';
  return '';
});

const DECIDER_ROWS: {key: keyof DeciderNumbers; label: string; title: string}[] = [
  {key: 'other_road_recall', label: '“Another” found', title: 'Share of the signs for another road it calls “Another” (recall)'},
  {key: 'other_road_precision', label: '“Another” right', title: 'Share of its “Another” answers that are right (precision)'},
  {key: 'macro_f1', label: 'Macro-F1', title: 'Ours, Another and Unclear weighted alike'},
  {key: 'abstain_rate', label: 'Abstains', title: 'Share of the signs it leaves undecided'},
  {key: 'reading_accuracy', label: 'Reading', title: 'Share of the boxes it reads right'},
];
/** A cell of the decider's table; "Another" found on a partition without such a sign is "no case" (its precision
 * there still counts the wrong "Another" answers). */
function deciderCell(numbers: DeciderNumbers | null, key: keyof DeciderNumbers): string {
  if (numbers === null) return '–';
  if (key === 'other_road_recall' && numbers.other_road_support === 0) return 'no case';
  return pct(numbers[key]);
}
const noOtherOnTest = computed(() => decider.value?.test?.other_road_support === 0);
</script>

<template>
  <li class="model" :class="{'in-use': version.in_use}" :data-model="version.id">
    <div class="model-head">
      <div class="model-title">
        <span class="model-name" :title="version.id">{{ version.run }}</span>
        <span class="model-sub">{{ version.label }} · {{ version.detail }}</span>
        <span v-if="version.trained_at" class="model-sub">Weights of {{ isoWhen(version.trained_at) }}</span>
      </div>
      <span v-if="version.in_use" class="pill ok">In use</span>
      <span v-else-if="!version.finished" class="pill warn">Not finished</span>
    </div>

    <template v-if="cutter">
      <template v-if="cut">
        <div class="model-big">
          <strong>{{ pct(cut.recall) }}</strong>
          <span>of the signs found<template v-if="cut.truth_boxes !== null">: {{ thousands(cut.found ?? 0) }} of {{ thousands(cut.truth_boxes) }}</template>
            · {{ (PARTITION_LABEL[cut.partition] ?? cut.partition).toLowerCase() }}, score {{ cut.score_threshold ?? '–' }}</span>
        </div>
        <ul v-if="cut.widths" class="model-widths" aria-label="Signs found by width in the frame">
          <li v-for="row in cut.widths" :key="row.key" :title="row.found + ' of ' + row.truth + ' signs ' + row.label + ' wide'">
            <span>{{ row.label }}</span><b>{{ row.truth ? pct(row.recall) : '–' }}</b><span>{{ row.found }} of {{ row.truth }}</span>
          </li>
        </ul>
        <dl class="model-facts">
          <div><dt>Boxes per frame</dt><dd>{{ perFrame(cut.boxes_per_frame) }}</dd></div>
          <div><dt>False per frame</dt><dd>{{ perFrame(cut.false_per_frame) }}</dd></div>
          <div><dt>Time per frame</dt><dd :title="cut.device ?? ''">{{ cut.ms_per_frame === null ? '–' : perFrame(cut.ms_per_frame) + ' ms' }}</dd></div>
          <div><dt>Input</dt><dd>{{ sizeText(cutter.input_size) }}<span v-if="!cutter.input_size_recorded" class="muted"> (default)</span></dd></div>
          <div><dt>Default box</dt><dd>{{ cutter.anchor_min_ratio }}<span v-if="!cutter.anchor_recorded" class="muted"> (default)</span></dd></div>
          <div><dt>Best epoch</dt><dd>{{ epoch }}</dd></div>
        </dl>
        <p v-if="!cut.at_cutter_score" class="note">At the report’s score {{ cut.score_threshold ?? '–' }}: it was evaluated before the cutter’s {{ score }} was measured.</p>
        <p v-if="cut.human" class="note">Frames a person confirmed: {{ pct(cut.human.recall) }} found ({{ cut.human.found ?? 0 }} of {{ cut.human.truth_boxes ?? 0 }}).</p>
      </template>
    </template>

    <template v-else-if="classifier">
      <template v-if="classifier.test">
        <div class="model-big">
          <strong>{{ pct(classifier.test.accuracy) }}</strong>
          <span>right answers on {{ thousands(classifier.test.samples) }} test crops</span>
        </div>
        <dl class="model-facts">
          <div><dt>Human-reviewed</dt><dd>{{ pct(classifier.test.human_accuracy) }}<span v-if="classifier.test.human_samples" class="muted"> of {{ thousands(classifier.test.human_samples) }}</span></dd></div>
          <div><dt>Classes</dt><dd>{{ classifier.classes ?? '–' }}</dd></div>
          <div><dt>Best validation</dt><dd>{{ pct(classifier.best_validation_accuracy) }}</dd></div>
          <div><dt>Best epoch</dt><dd>{{ epoch }}</dd></div>
        </dl>
      </template>
    </template>

    <template v-else-if="decider">
      <template v-if="decider.validation || decider.test">
        <div class="model-big">
          <strong>{{ pct(decider.validation?.other_road_recall) }}</strong>
          <span>of the “Another” signs found on validation<template v-if="decider.validation?.other_road_support">:
            {{ decider.validation.other_road_found ?? 0 }} of {{ decider.validation.other_road_support }}</template></span>
        </div>
        <div class="table-wrap"><table class="model-table">
          <thead><tr><th></th><th class="num">Validation</th><th class="num">Test</th></tr></thead>
          <tbody>
            <tr v-for="row in DECIDER_ROWS" :key="row.key" :data-metric="row.key">
              <th scope="row" :title="row.title">{{ row.label }}</th>
              <td class="num">{{ deciderCell(decider.validation, row.key) }}</td>
              <td class="num">{{ deciderCell(decider.test, row.key) }}</td>
            </tr>
          </tbody>
        </table></div>
        <p v-if="decider.inputs" class="note">Reads: {{ decider.inputs.join(' + ') }} · best epoch {{ epoch }}.</p>
        <p v-if="noOtherOnTest" class="note">The test has no “Another” sign yet: whom a sign applies to is measured on validation only.</p>
        <p v-if="decider.smoke_limit !== null" class="note ds-note">A smoke run on {{ decider.smoke_limit }} frames per partition, not a measure.</p>
      </template>
    </template>

    <p v-if="!evaluated" class="model-wait">{{ version.finished ? 'Not evaluated: no numbers on the test yet.' : 'Still training, or stopped before its end.' }}</p>
    <p v-for="problem in version.problems" :key="problem" class="note ds-note">{{ problem }}.</p>
    <p v-if="version.initialization" class="note">Started from {{ initText(version.initialization) }}.</p>

    <div class="model-actions">
      <a class="flow-link" :href="'/evaluate?run=' + encodeURIComponent(version.run)" @click="openRun($event, version.run)">Open the run →</a>
      <button v-if="!version.in_use" class="btn small" type="button" :disabled="blocked !== '' || busy !== null" :title="blocked"
              :aria-label="'Use ' + version.run + ' · ' + version.label + ' as the ' + roleLabel.toLowerCase()"
              @click="emit('use', version)">{{ busy === version.id ? 'Saving…' : 'Use this one' }}</button>
    </div>
  </li>
</template>
