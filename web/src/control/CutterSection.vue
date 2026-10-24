<script setup lang="ts">
// The cutters of scripts/run_detector_sizes.sh in a run (detector-sizes/<size>/), as the Models tab judges a cutter: at
// the cutter's score 0.1, the signs found overall, the speed signs alone for a cutter that also finds the other signs
// (comparable with one of the speed signs only), by the width of the sign; the boxes and false boxes per frame it
// passes on to the classifier and its time per frame; on test or validation; how it was trained; the comparison of the
// sizes in plain text.
import {computed, ref} from 'vue';
import {thousands} from '../common/format';
import type {CutterSize} from '../common/panel-types';
import {PARTITION_LABEL, pct} from './evaluate';
import {epochText, perFrame} from './models';
import Segmented from './Segmented.vue';

const props = defineProps<{sizes: CutterSize[]; comparison: string | null}>();
type Partition = 'test' | 'validation';
const pickedSize = ref<string | null>(null);
const chosen = ref<Partition>('test');
const sizeOptions = computed(() => props.sizes.map((item) => ({key: item.size, label: item.label})));
/** The size shown: the one chosen, else the last (the largest). */
const item = computed((): CutterSize | null => props.sizes.find((size) => size.size === pickedSize.value) ?? props.sizes[props.sizes.length - 1] ?? null);
const size = computed({
  get: (): string => item.value?.size ?? '',
  set: (value: string) => { pickedSize.value = value; },
});
const partitions = computed(() => (['test', 'validation'] as Partition[]).filter((key) => item.value?.[key])
  .map((key) => ({key, label: PARTITION_LABEL[key] ?? key})));
const cut = computed(() => item.value?.[chosen.value] ?? item.value?.test ?? item.value?.validation ?? null);
const speedSigns = computed(() => item.value?.families === 'all' ? cut.value?.by_family?.['speed'] ?? null : null);
const otherSigns = computed(() => item.value?.families === 'all' ? cut.value?.by_family?.['other_sign'] ?? null : null);
const training = computed(() => item.value?.training ?? null);
</script>

<template>
  <section v-if="item" class="card eval-card" id="cutter-eval" aria-labelledby="cutter-eval-title">
    <div class="ds-head">
      <h2 id="cutter-eval-title">Cutter<span class="eval-what">finds every box that could be a sign: it must miss nothing</span></h2>
      <span v-if="cut" class="ds-when">score ≥ {{ cut.score_threshold ?? '–' }}{{ cut.at_cutter_score ? '' : ' (the report’s own)' }}</span>
    </div>
    <div class="seg-row">
      <Segmented v-if="sizeOptions.length > 1" v-model="size" :options="sizeOptions" label="Input size"/>
      <Segmented v-if="partitions.length > 1" v-model="chosen" :options="partitions" label="Partition"/>
    </div>
    <template v-if="cut">
      <div class="ds-figures">
        <div class="ds-figure"><span class="ds-value" id="cutter-recall">{{ pct(cut.recall) }}</span>
          <span class="ds-unit">of {{ thousands(cut.truth_boxes ?? 0) }} {{ item.families === 'all' ? 'signs' : 'speed signs' }} found</span></div>
        <div v-if="speedSigns" class="ds-figure"><span class="ds-value" id="cutter-speed-recall">{{ pct(speedSigns.recall) }}</span>
          <span class="ds-unit">of the {{ thousands(speedSigns.truth) }} speed signs alone</span></div>
        <div v-if="otherSigns" class="ds-figure"><span class="ds-value">{{ pct(otherSigns.recall) }}</span>
          <span class="ds-unit">of the {{ thousands(otherSigns.truth) }} other signs</span></div>
      </div>
      <ul class="box-counts" aria-label="Boxes per frame">
        <li><strong>{{ perFrame(cut.boxes_per_frame) }}</strong>boxes per frame<span>passed on to the classifier</span></li>
        <li class="false"><strong>{{ perFrame(cut.false_per_frame) }}</strong>false per frame<span>{{ cut.false_boxes === null ? ''
          : thousands(cut.false_boxes) + ' boxes where there is no reviewed sign' }}</span></li>
        <li><strong>{{ cut.ms_per_frame === null ? '–' : perFrame(cut.ms_per_frame) + ' ms' }}</strong>per frame<span>{{ cut.device
          ? 'the model alone, on ' + cut.device : 'the model alone' }}</span></li>
      </ul>
      <p v-if="cut.human" class="note">On the {{ thousands(cut.human.images ?? 0) }} frames a person confirmed: {{ pct(cut.human.recall) }} found
        ({{ thousands(cut.human.found ?? 0) }} of {{ thousands(cut.human.truth_boxes ?? 0) }}).</p>
      <div v-if="cut.widths" class="ds-block">
        <h3 class="ds-title">Found by sign size<span class="ds-when">width of the reviewed box in the frame</span></h3>
        <ul class="barlist sizes" id="cutter-sizes" aria-label="Signs found by size">
          <li v-for="row in cut.widths" :key="row.key" :data-size="row.key" :title="row.label + ': ' + row.found + ' found of ' + row.truth">
            <span class="bl-label">{{ row.label }}</span>
            <span class="bl-track" aria-hidden="true"><span :style="{width: Math.max((row.recall ?? 0) * 100, row.truth ? 0.6 : 0) + '%'}"></span></span>
            <span class="bl-value">{{ row.truth ? pct(row.recall) : '–' }}<small>{{ thousands(row.found) }}/{{ thousands(row.truth) }}</small></span>
            <span class="bl-extra"></span>
          </li>
        </ul>
      </div>
    </template>
    <p v-else class="net-wait">{{ item.finished ? 'Not evaluated yet: test and validation come after the training.' : 'Still training, or stopped before its end.' }}</p>

    <dl class="facts">
      <div><dt>Input</dt><dd>{{ item.label }}</dd></div>
      <div><dt>Finds</dt><dd>{{ item.families === 'all' ? 'the speed signs and the other signs' : 'the speed signs' }}</dd></div>
      <div v-if="training"><dt>Best epoch</dt><dd>{{ epochText(training.best_epoch, training.epochs_run ?? training.epochs) }}</dd></div>
      <div v-if="training?.anchor_min_ratio != null"><dt>Default box</dt><dd>{{ training.anchor_min_ratio }}</dd></div>
      <div v-if="training?.augmentation != null"><dt>Augmentation</dt><dd>{{ training.augmentation ? 'yes' : 'no' }}</dd></div>
      <div v-if="training?.gpu"><dt>GPU</dt><dd>{{ training.gpu }}</dd></div>
    </dl>
    <details v-if="comparison" class="chart-table">
      <summary>The sizes compared (comparison.txt)</summary>
      <pre class="term">{{ comparison }}</pre>
    </details>
  </section>
</template>
