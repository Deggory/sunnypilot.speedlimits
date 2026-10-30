<script setup lang="ts">
// A SignNet run (detail.signnet_run, nnslr_tools.runs): the cutter the comma runs, judged as the car reads it: of the
// reviewed signs, those among its 5 best boxes of each frame of score 0.3 or more (the car drops the others), and the
// false boxes per frame it passes on, on the test (validation in a line below; validation alone when the test is
// missing); then the same among its 5 best boxes at any score (what its training picks the best epoch by; the only
// numbers of an older server); then its shape (the band at the top of the frame, the grid step, the weights), its best
// epoch and what it finds, and whether the comma's vision.py runs it as it is (band 896 × 288, grid 8) or needs a build
// for it.
import {computed} from 'vue';
import {thousands} from '../common/format';
import type {SignNetCar, SignNetRun} from '../common/panel-types';
import {pct, shortSha} from './evaluate';
import {epochText, perFrame} from './models';

const props = defineProps<{signnet: SignNetRun}>();
/** The test's numbers, else validation's; the other partition in a line. */
const main = computed(() => props.signnet.test ? {name: 'test', numbers: props.signnet.test}
  : props.signnet.validation ? {name: 'validation', numbers: props.signnet.validation} : null);
const other = computed(() => props.signnet.test && props.signnet.validation ? props.signnet.validation : null);
const car = computed((): SignNetCar | null => main.value?.numbers.car ?? null);
const signs = computed(() => props.signnet.families === 'speed' ? 'speed signs' : 'signs');
/** "best 5 at 0.3": how the car reads it. */
const carRule = (found: SignNetCar | null | undefined): string => 'best ' + (found?.max_boxes ?? 5) + ' at ' + (found?.score_threshold ?? 0.3);
const band = computed(() => props.signnet.band ? props.signnet.band[0] + ' × ' + props.signnet.band[1] : '–');
</script>

<template>
  <section class="card eval-card" id="signnet-eval" aria-labelledby="signnet-eval-title">
    <div class="ds-head">
      <h2 id="signnet-eval-title">SignNet<span class="eval-what">the cutter the comma runs: it finds every box that could be a sign</span></h2>
      <span class="pill" :class="signnet.fits_the_car ? 'ok' : 'warn'" id="signnet-fits">{{ signnet.fits_the_car ? 'Fits the car'
        : 'Needs a build for the car' }}</span>
    </div>
    <template v-if="main">
      <h3 class="ds-title" id="signnet-car-title">{{ car ? 'As the car reads them (' + carRule(car) + ')' : 'Best 5 boxes, any score' }}</h3>
      <div v-if="car" class="ds-figures" id="signnet-car">
        <div class="ds-figure"><span class="ds-value" id="signnet-recall">{{ pct(car.recall) }}</span>
          <span class="ds-unit">of the {{ main.name }} {{ signs }} found<template v-if="signnet.families !== 'speed' && car.speed_recall != null">
            (speed signs {{ pct(car.speed_recall) }})</template>: its {{ car.max_boxes }} best boxes of each frame of score
            {{ car.score_threshold }} or more, those the car keeps</span></div>
        <div class="ds-figure"><span class="ds-value" id="signnet-false">{{ perFrame(car.false_per_frame) }}</span>
          <span class="ds-unit">false boxes per frame the car passes on: the classifier drops them</span></div>
      </div>
      <div v-else class="ds-figures">
        <div class="ds-figure"><span class="ds-value" id="signnet-recall">{{ pct(main.numbers.recall) }}</span>
          <span class="ds-unit">of the {{ main.name }} {{ signs }} found in its 5 best boxes of each frame, whatever their score
            (the car also drops those below 0.3: not measured here)</span></div>
        <div class="ds-figure"><span class="ds-value" id="signnet-false">{{ perFrame(main.numbers.false_per_frame) }}</span>
          <span class="ds-unit">false boxes per frame among those 5</span></div>
      </div>
      <p v-if="car" class="note" id="signnet-any">Best 5 boxes, any score: {{ pct(main.numbers.recall) }} found,
        {{ perFrame(main.numbers.false_per_frame) }} false boxes per frame (what its training picks the best epoch by).</p>
      <p v-if="other" class="note" id="signnet-validation">On validation<template v-if="other.car">, as the car reads them
        ({{ carRule(other.car) }}): {{ pct(other.car.recall) }} found, {{ perFrame(other.car.false_per_frame) }} false boxes per
        frame; best 5 boxes, any score</template>: {{ pct(other.recall) }} found, {{ perFrame(other.false_per_frame) }} false boxes
        per frame.</p>
    </template>
    <p v-else class="net-wait">Not measured yet: validation and test come after the training.</p>

    <dl class="facts">
      <div><dt title="The top rows of the frame, scaled to this width">Band</dt><dd>{{ band }}</dd></div>
      <div><dt title="The step of its output grid, in pixels of the band">Grid step</dt><dd>{{ signnet.stride ? signnet.stride + ' px' : '–' }}</dd></div>
      <div><dt>Weights</dt><dd>{{ signnet.parameters ? thousands(signnet.parameters) : '–' }}</dd></div>
      <div><dt>Best epoch</dt><dd>{{ epochText(signnet.best_epoch, signnet.epochs) }}</dd></div>
      <div><dt>Finds</dt><dd>{{ signnet.families === 'speed' ? 'the speed signs' : signnet.families === 'all'
        ? 'the speed signs and the other signs' : '–' }}</dd></div>
      <div v-if="signnet.split_sha256"><dt>Split</dt><dd><code :title="signnet.split_sha256">{{ shortSha(signnet.split_sha256) }}</code></dd></div>
    </dl>
    <p v-if="!signnet.fits_the_car" class="note">The comma runs a SignNet on an 896 × 288 band with a grid of 8 px: this one
      can be compared here, but not put in use or sent to the car.</p>
  </section>
</template>

