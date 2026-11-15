<script setup lang="ts">
// The choices of the brain's training (brain-choices.ts) in one short row, in Train › Brain above its flow (which follows
// them) and in its step of Train › Pipeline: the heads only the training has and the map's limit as an input (#85) as
// switches that light up when on, the
// picture's features as Mean or Grid, its seeds as Three or One, and its drives (every dashcam drive or those of an
// earlier brain run, runs: BrainLabels.same_drives; no choice while there is none).
import {POOL_OPTIONS, SEEDS_OPTIONS, type BrainChoice} from './brain-choices';
import type {BrainRouteRun} from '../common/panel-types';
import Segmented from './Segmented.vue';

const props = defineProps<{choices: BrainChoice; id: string; runs?: readonly BrainRouteRun[]}>();
type Switch = 'works' | 'road' | 'types' | 'users' | 'applicability' | 'map';
const SWITCHES: {key: Switch; label: string; title: string}[] = [
  {key: 'works', label: 'Roadworks', title: 'The brain also learns the roadworks on the road, a head only the training has'},
  {key: 'road', label: 'Road conditions', title: 'The brain also learns the road conditions you mark in Speed review: narrow, country, gravel, potholes, speed bumps…'},
  {key: 'types', label: 'Road type (map)', title: 'The brain also learns the road type of the map (motorway, rural, town…), corrected in Speed review; only the seconds that have one count'},
  {key: 'users', label: 'Road users', title: 'The brain also learns where the persons, the two-wheelers and the vehicles that cross our path are (the crossings you confirm), a head only the training has; read first with a detector on the GPU'},
  {key: 'applicability', label: 'Applicability', title: 'The brain also learns whose road a speed sign is of, ours or another, from the boxes you reviewed in Sign review: a head only the training has for now, measured only against your reviews. Off: no such head'},
  // An input, not a head: off unless turned on (#85). The one brain that reads the map.
  {key: 'map', label: 'Map’s limit', title: 'The limit the comma’s map had goes into the brain as one more input (80 % of your drives’ time has one). Off: the brain never reads the map, as until now. On the car such a brain reads the map’s limit too'},
];
function flip(key: Switch): void {
  props.choices[key] = props.choices[key] === 'on' ? 'off' : 'on';
}
</script>

<template>
  <div class="bc" :id="id + '-choices'" role="group" aria-label="What the next training of the brain learns">
    <button v-for="item in SWITCHES" :key="item.key" type="button" class="bc-switch" :aria-pressed="choices[item.key] === 'on'"
            :title="item.title" @click="flip(item.key)">{{ item.label }}</button>
    <span class="bc-pick" :title="choices.note"><span>Picture</span>
      <Segmented v-model="choices.pool" :options="POOL_OPTIONS" label="The picture’s features: Grid keeps where things are, Mean only what is in it"/></span>
    <span class="bc-pick" :title="choices.seedsNote"><span>Seeds</span>
      <Segmented v-model="choices.seeds" :options="SEEDS_OPTIONS" label="The seeds: seed 0 alone, or three to compare in Models"/></span>
    <label v-if="runs?.length" class="bc-pick" :title="choices.drivesNote"><span>Drives</span>
      <select v-model="choices.routesFrom" :id="id + '-drives'">
        <option value="">All dashcam drives</option>
        <option v-for="run in runs" :key="run.run" :value="run.run">Same drives as {{ run.run }} ({{ run.drives }})</option>
      </select></label>
  </div>
</template>
