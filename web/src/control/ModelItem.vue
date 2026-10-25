<script setup lang="ts">
// One trained version of a role: its run, what it is and when its weights were saved, the headline numbers that judge
// that role (the cutter: the speed signs found at a low score, as on Evaluate, then every sign and by width, boxes per
// frame, time per frame; the
// classifier: right answers on the test, and for the classifier with “not a sign” the “not a sign” crops it rejects and
// the chain end to end at the cutter's score, the speed signs found and read right and the false boxes per frame that
// survive it; the decider: the "Another" road found, on validation and test), the button that puts it in use and the
// link to its run on Evaluate.
import {computed} from 'vue';
import {isoWhen, thousands} from '../common/format';
import type {DeciderNumbers, ModelVersion} from '../common/panel-types';
import {PARTITION_LABEL, pct} from './evaluate';
import {ROLE_TEXT, epochText, evaluateHref, openRun, perFrame, sizeText} from './models';

const props = defineProps<{version: ModelVersion; roleLabel: string; score: number; readOnly: boolean; busy: string | null}>();
const emit = defineEmits<{use: [version: ModelVersion]}>();

const cutter = computed(() => props.version.role === 'cutter' ? props.version : null);
const classifier = computed(() => props.version.role === 'classifier' ? props.version : null);
const decider = computed(() => props.version.role === 'decider' ? props.version : null);
/** The cutter's numbers: on the test, else on validation (a size compared before its test was measured). */
const cut = computed(() => cutter.value?.test ?? cutter.value?.validation ?? null);
/** A cutter of every sign: the speed signs alone, comparable with a cutter of the speed signs only. */
const speedSigns = computed(() => cutter.value?.families === 'all' ? cut.value?.by_family?.['speed'] ?? null : null);
/** The cutter's headline, as on Evaluate: the speed signs found at its score (every sign below it, for a cutter of
 * every sign); every sign only when its report does not count the speed signs apart. */
const headline = computed(() => {
  const numbers = cut.value;
  if (numbers === null) return null;
  if (speedSigns.value) return {speed: true, recall: speedSigns.value.recall, found: speedSigns.value.found, truth: speedSigns.value.truth as number | null};
  return {speed: cutter.value?.families !== 'all', recall: numbers.recall, found: numbers.found ?? 0, truth: numbers.truth_boxes};
});
const evaluated = computed(() => cut.value !== null || (classifier.value?.test ?? null) !== null || (classifier.value?.chain ?? null) !== null
  || decider.value?.validation != null || decider.value?.test != null);
const epoch = computed(() => epochText(props.version.best_epoch, props.version.epochs_run ?? props.version.epochs));
/** The classifier with “not a sign” (classifier:<run>/classifier); an old reader has none of its numbers. */
const rejects = computed(() => classifier.value?.class_scheme !== undefined);
const chain = computed(() => classifier.value?.chain ?? null);
const oldChain = computed(() => classifier.value?.chain_old_reader ?? null);

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
        <span class="model-name" :title="version.id">{{ version.label }}</span>
        <span class="model-sub">{{ version.trained_at ? isoWhen(version.trained_at) : version.run }} · {{ version.detail }}</span>
      </div>
      <span v-if="version.in_use" class="pill ok">In use</span>
      <span v-else-if="!version.finished" class="pill warn">Not finished</span>
    </div>

    <template v-if="cutter">
      <template v-if="cut">
        <div v-if="headline" class="model-big" :data-speed-recall="headline.speed ? headline.recall ?? undefined : undefined">
          <strong>{{ pct(headline.recall) }}</strong>
          <span>of the {{ headline.speed ? 'speed signs' : 'signs' }} found<template v-if="headline.truth !== null">: {{ thousands(headline.found) }}
            of {{ thousands(headline.truth) }}</template> · {{ (PARTITION_LABEL[cut.partition] ?? cut.partition).toLowerCase() }}, score {{ cut.score_threshold ?? '–' }}</span>
        </div>
        <p v-if="speedSigns" class="note">Every sign, speed and other: <strong>{{ pct(cut.recall) }}</strong><template
          v-if="cut.truth_boxes !== null"> ({{ thousands(cut.found ?? 0) }} of {{ thousands(cut.truth_boxes) }})</template>.</p>
        <details class="model-more"><summary>Details</summary>
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
          <div><dt>Finds</dt><dd>{{ cutter.families === 'all' ? 'speed and other signs' : 'speed signs' }}<span v-if="!cutter.families_recorded"
            class="muted"> (not recorded)</span></dd></div>
        </dl>
        <p v-if="!cut.at_cutter_score" class="note">At the report’s score {{ cut.score_threshold ?? '–' }}: it was evaluated before the cutter’s {{ score }} was measured.</p>
        <p v-if="cut.human" class="note">Frames a person confirmed: {{ pct(cut.human.recall) }} found ({{ cut.human.found ?? 0 }} of {{ cut.human.truth_boxes ?? 0 }}).</p>
        </details>
      </template>
    </template>

    <template v-else-if="classifier && rejects">
      <template v-if="classifier.test || chain">
        <div v-if="classifier.test" class="model-big">
          <strong>{{ pct(classifier.test.accuracy) }}</strong>
          <span>of the test signs read right<template v-if="classifier.test.samples !== null">: {{ thousands(classifier.test.samples) }} sign crops</template></span>
        </div>
        <p v-if="chain" class="model-line">With the cutter{{ chain.cutter_size ? ' ' + sizeText(chain.cutter_size) : '' }}:
          <strong>{{ pct(chain.recall) }}</strong> of the speed signs found and read right.</p>
        <details class="model-more"><summary>Details</summary>
        <dl class="model-facts">
          <div v-if="classifier.test"><dt>Human-reviewed</dt><dd>{{ pct(classifier.test.human_accuracy) }}<span v-if="classifier.test.human_samples"
            class="muted"> of {{ thousands(classifier.test.human_samples) }}</span></dd></div>
          <div v-if="classifier.test"><dt title="Share of the “not a sign” test crops it answers “not a sign”">“Not a sign” rejected</dt><dd>{{
            pct(classifier.test.rejection_rate) }}<span v-if="classifier.test.not_a_sign_crops" class="muted"> of {{ thousands(classifier.test.not_a_sign_crops) }}</span></dd></div>
          <div v-if="classifier.test"><dt title="Sign crops it throws away as “not a sign”">Signs thrown away</dt><dd>{{
            pct(classifier.test.signs_rejected_rate) }}<span v-if="classifier.test.signs_rejected !== null && classifier.test.signs_rejected !== undefined"
            class="muted"> ({{ thousands(classifier.test.signs_rejected) }})</span></dd></div>
          <div v-if="chain"><dt :title="'Speed signs of the test found by the cutter at score ' + chain.score_threshold + ' and read right by this classifier'">Chain, end to end</dt><dd>{{
            pct(chain.recall) }}<span v-if="chain.truth" class="muted"> ({{ thousands(chain.correct ?? 0) }} of {{ thousands(chain.truth) }})</span></dd></div>
          <div v-if="chain"><dt :title="'False boxes of the cutter at score ' + chain.score_threshold + ' that this classifier lets through, per frame'">Surviving false per frame</dt><dd>{{
            perFrame(chain.surviving_false_per_frame) }}<span class="muted"> of {{ perFrame(chain.false_per_frame) }}</span></dd></div>
          <div><dt title="Mean of the sign crops read right and the “not a sign” crops rejected, on validation: it chose the best epoch">Best validation</dt><dd>{{
            pct(classifier.best_validation_balanced_accuracy ?? classifier.best_validation_accuracy) }}</dd></div>
          <div><dt>Best epoch</dt><dd>{{ epoch }}</dd></div>
          <div><dt>Classes</dt><dd>{{ classifier.classes ?? '–' }}</dd></div>
        </dl>
        <p v-if="chain" class="note">Chain at the cutter’s score {{ chain.score_threshold }}: the cutter alone finds {{ pct(chain.recall_found) }}
          of the speed signs<template v-if="chain.human_recall !== null">; on the frames a person confirmed, {{ pct(chain.human_recall) }} end to end</template>.</p>
        <p v-if="chain && oldChain" class="note">The old reader in the same chain: {{ pct(oldChain.recall) }} end to end, every false box
          let through ({{ perFrame(oldChain.surviving_false_per_frame) }} per frame).</p>
        <p v-if="classifier.test?.family_accuracy != null" class="note">With the kinds of other signs folded, as the old reader reads them:
          {{ pct(classifier.test.family_accuracy) }} right.</p>
        </details>
      </template>
    </template>

    <template v-else-if="classifier">
      <template v-if="classifier.test">
        <div class="model-big">
          <strong>{{ pct(classifier.test.accuracy) }}</strong>
          <span>right answers on {{ thousands(classifier.test.samples ?? 0) }} test crops</span>
        </div>
        <details class="model-more"><summary>Details</summary>
        <dl class="model-facts">
          <div><dt>Human-reviewed</dt><dd>{{ pct(classifier.test.human_accuracy) }}<span v-if="classifier.test.human_samples" class="muted"> of {{ thousands(classifier.test.human_samples) }}</span></dd></div>
          <div><dt>Classes</dt><dd>{{ classifier.classes ?? '–' }}</dd></div>
          <div><dt>Best validation</dt><dd>{{ pct(classifier.best_validation_accuracy) }}</dd></div>
          <div><dt>Best epoch</dt><dd>{{ epoch }}</dd></div>
        </dl>
        </details>
      </template>
    </template>

    <template v-else-if="decider">
      <template v-if="decider.validation || decider.test">
        <div class="model-big">
          <strong>{{ pct(decider.validation?.other_road_recall) }}</strong>
          <span>of the “Another” signs found on validation<template v-if="decider.validation?.other_road_support">:
            {{ decider.validation.other_road_found ?? 0 }} of {{ decider.validation.other_road_support }}</template></span>
        </div>
        <details class="model-more"><summary>Details</summary>
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
        </details>
      </template>
    </template>

    <p v-if="!evaluated" class="model-wait">{{ version.finished ? 'Not evaluated: no numbers on the test yet.' : 'Still training, or stopped before its end.' }}</p>
    <p v-for="problem in version.problems" :key="problem" class="note ds-note">{{ problem }}.</p>

    <div class="model-actions">
      <a class="flow-link" :href="evaluateHref(version)" @click="openRun($event, version)">Open the run ›</a>
      <button v-if="!version.in_use" class="btn small" type="button" :disabled="blocked !== '' || busy !== null" :title="blocked || ROLE_TEXT[version.role].uses"
              :aria-label="'Use ' + version.run + ' · ' + version.label + ' as the ' + roleLabel.toLowerCase()"
              @click="emit('use', version)">{{ busy === version.id ? 'Saving…' : 'Use this one' }}</button>
    </div>
  </li>
</template>
