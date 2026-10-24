<script setup lang="ts">
// The runs, newest first: name and when, where the weights started from, the state, and the headline numbers on
// the test (the reader's accuracy on every crop and on the crops a person confirmed, the detector's recall and
// precision). A row opens the run below; the name is the button, for the keyboard.
import {thousands} from '../common/format';
import type {RunSummary} from '../common/panel-types';
import {RUN_STATE, modeText, notEvaluatedText, pct, runWhen, stageText} from './evaluate';

defineProps<{runs: RunSummary[]; selected: string | null}>();
const emit = defineEmits<{select: [name: string]}>();
</script>

<template>
  <div class="table-wrap">
    <table class="runs" id="runs">
      <thead>
        <tr><th>Run</th><th>Started from</th><th>State</th>
          <th class="num" title="Share of the test crops the reader reads right">Reader</th>
          <th class="num" title="The same on the crops a person confirmed">Reader, human</th>
          <th class="num" title="Share of the reviewed signs of the test the detector finds">Det. recall</th>
          <th class="num" title="Share of the detector's boxes that are a reviewed sign">Det. precision</th></tr>
      </thead>
      <tbody>
        <tr v-for="run in runs" :key="run.name" :class="{on: run.name === selected}" :data-run="run.name" @click="emit('select', run.name)">
          <td class="run-cell">
            <button type="button" class="run-name" :aria-pressed="run.name === selected" @click.stop="emit('select', run.name)">{{ run.name }}</button>
            <span class="run-sub">{{ runWhen(run) }}</span>
          </td>
          <td>{{ run.start.label }}<span v-if="modeText(run)" class="run-sub">{{ modeText(run) }}</span></td>
          <td><span class="pill" :class="RUN_STATE[run.state][1]">{{ RUN_STATE[run.state][0] }}</span><span class="run-sub">{{ stageText(run) }}</span></td>
          <template v-if="run.evaluated">
            <td class="num">{{ pct(run.reader?.accuracy) }}<span v-if="run.reader" class="run-sub">{{ thousands(run.reader.samples) }} crops</span></td>
            <td class="num">{{ pct(run.reader?.human_accuracy) }}<span v-if="run.reader?.human_samples" class="run-sub">{{ thousands(run.reader.human_samples) }} crops</span></td>
            <td class="num">{{ pct(run.detector?.recall) }}<span v-if="run.detector" class="run-sub">{{ thousands(run.detector.found) }} of {{ thousands(run.detector.truth_boxes) }}</span></td>
            <td class="num">{{ pct(run.detector?.precision) }}<span v-if="run.detector" class="run-sub">{{ thousands(run.detector.false_boxes) }} false</span></td>
          </template>
          <td v-else colspan="4" class="run-none">{{ notEvaluatedText(run) }}</td>
        </tr>
      </tbody>
    </table>
  </div>
</template>
