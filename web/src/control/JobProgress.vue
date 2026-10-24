<script setup lang="ts">
// A job, live: name and state, the big percent (or "running" with a moving bar when the server does not
// know it), the steps, what it is doing, the time it has been running (ticking) and, when the percent is
// known, about how long is left; Stop and Details (the log in the sheet); while it runs, the machine in small
// under the buttons. A finished job shows how it ended.
import {computed, inject} from 'vue';
import type {Job, SystemInfo} from '../common/panel-types';
import Icon from '../components/Icon.vue';
import {useNow} from '../composables/now';
import JobSteps from './JobSteps.vue';
import {elapsedText, failedStep, JOB_STATE, roughText, startedAt, startedWhen} from './jobs';
import MiniMachine from './MiniMachine.vue';
import {RESUME_JOB} from './use-actions';

const props = defineProps<{job: Job; heading?: string | undefined; system?: SystemInfo | null | undefined}>();
defineEmits<{details: [job: Job]; stop: [job: Job]}>();
const now = useNow();

const running = computed(() => props.job.state === 'running');
const resumeJob = inject(RESUME_JOB, null);
// A training that stopped after writing its run: the missing steps can run again on that run.
const resumable = computed(() => resumeJob !== null && Boolean(props.job.run)
  && (props.job.state === 'failed' || props.job.state === 'interrupted') && props.job.action.startsWith('train'));
const state = computed(() => JOB_STATE[props.job.state]);
const percent = computed(() => {
  const value = props.job.state === 'ok' ? 100 : props.job.progress.percent;
  return value === null ? null : Math.max(0, Math.min(100, value));
});
/** Never rounded up: 99.6 is still 99 until the job says it is done. */
const percentText = computed(() => percent.value === null ? '' : percent.value > 0 && percent.value < 1 ? '<1' : String(Math.floor(percent.value)));
const elapsed = computed(() => {
  const start = startedAt(props.job);
  return start === null ? null : now.value - start.getTime();
});
/** From the time so far and the percent: only when the percent is known and past 3%, and only "about". */
const eta = computed(() => {
  const value = percent.value;
  if (!running.value || value === null || value <= 3 || value >= 100 || elapsed.value === null || elapsed.value <= 0) return '';
  return 'about ' + roughText(elapsed.value * (100 - value) / value) + ' left';
});
const when = computed(() => {
  const at = 'Started ' + startedWhen(props.job);
  if (running.value) return elapsed.value === null ? at : at + ' · for ' + elapsedText(elapsed.value);
  if (props.job.state === 'failed' && props.job.code !== null) return at + ' · exit code ' + props.job.code;
  return at;
});
const ended = computed((): string => {
  if (props.job.state === 'ok') return 'Done';
  if (props.job.state === 'failed') return 'Failed';
  if (props.job.state === 'interrupted') return 'Interrupted';
  return '';
});
const stoppedAt = computed(() => props.job.state === 'failed' ? failedStep(props.job) : '');
const steps = computed(() => props.job.progress.steps.length > 1 ? props.job.progress.steps : []);
const barKind = computed(() => ({ok: 'ok', failed: 'bad', interrupted: 'warn', running: ''})[props.job.state]);
</script>

<template>
  <article class="jp" :class="['jp-' + job.state, {'has-steps': steps.length > 0}]" :data-job="job.id" :aria-label="job.label">
    <header class="jp-head">
      <span class="jp-icon" :class="state[1]" aria-hidden="true"><span v-if="running" class="spinner"></span><svg
        v-else-if="job.state === 'ok'" viewBox="0 0 16 16"><path d="M3.5 8.4l3 3 6-6.4" fill="none" stroke="currentColor" stroke-width="2.2"
        stroke-linecap="round" stroke-linejoin="round"/></svg><svg v-else viewBox="0 0 16 16"><path d="M5 5l6 6M11 5l-6 6"
        fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"/></svg></span>
      <div class="jp-title">
        <span v-if="heading" class="jp-kicker">{{ heading }}</span>
        <h3>{{ job.label }}</h3>
        <span class="jp-when">{{ when }}</span>
      </div>
      <span class="pill" :class="state[1]">{{ state[0] }}</span>
    </header>

    <div class="jp-body">
      <div class="jp-main">
        <div class="jp-figure">
          <span v-if="running && percent === null" class="jp-big jp-big-word">running</span>
          <span v-else-if="running" class="jp-big">{{ percentText }}<small>%</small></span>
          <span v-else class="jp-big jp-big-word" :class="state[1]">{{ ended }}<template v-if="job.state === 'ok'"> ✓</template></span>
          <span v-if="eta" class="jp-eta">{{ eta }}</span>
        </div>
        <div class="jp-bar" :class="[barKind, {indeterminate: running && percent === null}]" role="progressbar"
             :aria-label="job.label" aria-valuemin="0" aria-valuemax="100"
             :aria-valuenow="percent === null ? undefined : Math.floor(percent)"
             :aria-valuetext="percent === null ? 'running, progress unknown' : undefined"><span
          :style="percent === null ? undefined : {width: percent + '%'}"></span></div>
        <p v-if="job.progress.detail" class="jp-detail">{{ job.progress.detail }}</p>
        <p v-if="stoppedAt" class="jp-stopped">It stopped at the step <strong>{{ stoppedAt }}</strong>.</p>
        <p v-if="job.progress.last_line" class="jp-line" :class="{bad: job.state === 'failed'}"
           :title="job.progress.last_line">{{ job.progress.last_line }}</p>
        <slot></slot>
        <div class="jp-actions">
          <button class="btn small" type="button" @click="$emit('details', job)"><Icon name="details"/>Details</button>
          <button v-if="running && job.stoppable" class="btn small danger" type="button" @click="$emit('stop', job)"><Icon name="stop"/>Stop</button>
          <button v-if="resumable" class="btn small primary" type="button" @click="resumeJob?.(job)"><Icon name="resume"/>Resume</button>
        </div>
        <MiniMachine v-if="running && system" :system="system"/>
      </div>
      <JobSteps v-if="steps.length" :steps="steps" :current="job.state === 'running' ? job.progress.step_progress : null"/>
    </div>
  </article>
</template>
