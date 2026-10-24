<script setup lang="ts">
// One job of the list of recent jobs: state, name, when it started and how it went; Details, Stop.
import {computed} from 'vue';
import type {Job} from '../common/panel-types';
import Icon from '../components/Icon.vue';
import {failedStep, JOB_STATE, startedWhen} from './jobs';

const props = defineProps<{job: Job}>();
defineEmits<{details: [job: Job]; stop: [job: Job]}>();
const sub = computed(() => {
  const job = props.job;
  const parts = [startedWhen(job)];
  if (job.state === 'running') {
    if (job.progress.percent !== null) parts.push(Math.floor(job.progress.percent) + '%');
    if (job.progress.detail) parts.push(job.progress.detail);
  } else if (job.state === 'failed') {
    const step = failedStep(job);
    parts.push(step ? 'stopped at ' + step : job.code !== null ? 'exit code ' + job.code : 'failed');
  } else if (job.progress.detail) {
    parts.push(job.progress.detail);
  }
  return parts.join(' · ');
});
</script>

<template>
  <li class="item job-item" :data-job="job.id">
    <span class="pill" :class="JOB_STATE[job.state][1]">{{ JOB_STATE[job.state][0] }}</span>
    <div class="grow"><div class="what">{{ job.label }}</div><div class="sub">{{ sub }}</div></div>
    <button class="btn small" type="button" @click="$emit('details', job)"><Icon name="details"/>Details</button>
    <button v-if="job.stoppable" class="btn small danger" type="button" @click="$emit('stop', job)"><Icon name="stop"/>Stop</button>
  </li>
</template>
