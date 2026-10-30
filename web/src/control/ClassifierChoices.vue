<script setup lang="ts">
// The choices of the classifier's next training (classifier-choices.ts), in Train › Classifier and in the classifier's
// step of Train › Pipeline: its signs, the places and the share of the Mapillary signs, where it starts.
import {thousands} from '../common/format';
import type {PanelStatus} from '../common/panel-types';
import PickerSheet from '../components/PickerSheet.vue';
import {MAPILLARY_OPTIONS, type ClassifierChoices} from './classifier-choices';
import RegionPicker from './RegionPicker.vue';
import Segmented from './Segmented.vue';

defineProps<{choices: ClassifierChoices; status: PanelStatus | null}>();
defineEmits<{locate: []}>();
</script>

<template>
  <div v-if="choices.mapillaryState?.available" class="seg-row">
    <Segmented v-model="choices.mapillary" class="mly-seg" :options="MAPILLARY_OPTIONS" label="Signs the classifier learns from"/>
  </div>
  <div v-if="choices.mapillaryState?.available && choices.withMapillary" class="mly-choice" id="train-mapillary-choice">
    <RegionPicker v-model="choices.region" :regions="choices.mapillaryRegions" :locate="status?.actions.mapillary_locations ?? null"
                  :locating="choices.locating" @locate="$emit('locate')"/>
    <label class="picker mly-share" :class="{bad: choices.share === null}">
      <span class="picker-label">Share</span>
      <span class="mly-share-field"><input v-model="choices.shareText" id="mapillary-share" type="text" inputmode="decimal"
             maxlength="6" autocomplete="off" aria-describedby="train-mapillary-note">%</span>
      <span class="picker-sub">{{ choices.share === null ? 'from 0.1 to 100' : 'about ' + thousands(choices.mapillarySigns) + ' signs' }}</span>
    </label>
  </div>
  <div v-if="choices.mapillary !== 'pretrain'" class="cls-start">
    <PickerSheet v-model="choices.startKey" :options="choices.startOptions" label="Start from" id="train-classifier-start"/>
  </div>
  <p v-if="choices.mapillary === 'pretrain'" class="latest-none" id="train-start-note">Start: ImageNet, then its own Mapillary
    stage; another classifier cannot be the start of “Mapillary, then ours”.</p>
  <p v-else-if="choices.startVersion && choices.startSplit(choices.startVersion) === 'another'" class="latest-none warn-text" id="train-start-note">Trained
    on another split: it may have seen some of today's test signs, so the test of the new one is not fair.</p>
  <p v-else-if="choices.startVersion && choices.startSplit(choices.startVersion) === 'unknown'" class="latest-none warn-text" id="train-start-note">Its
    split is not recorded: it may have seen some of today's test signs, so the test of the new one may not be fair.</p>
  <p v-else-if="choices.startVersion" class="latest-none" id="train-start-note">It goes on from that classifier's weights, learnt on
    today's split: the test stays fair.</p>
</template>
