<script setup lang="ts">
// The details of a job in the bottom sheet: how far it is (percent, what it is doing, the steps), Stop while it
// runs, how it ended and after how long, and its log as plain text, followed live while the job runs (Follow keeps
// the end in view). job is the latest copy from the status, so it updates.
import {computed, nextTick, onBeforeUnmount, onMounted, ref, useTemplateRef, watch} from 'vue';
import {api, messageOf} from '../common/api';
import type {Job, LogChunk} from '../common/panel-types';
import Icon from '../components/Icon.vue';
import {useNow} from '../composables/now';
import {toast} from '../composables/toast';
import JobSteps from './JobSteps.vue';
import {elapsedText, failedStep, JOB_STATE, startedAt, startedWhen} from './jobs';

const props = defineProps<{job: Job | null}>();
const running = computed(() => props.job?.state === 'running');
const now = useNow();
/** How long it ran (until now while it runs), or null when a time is unknown. */
const duration = computed((): string | null => {
  const job = props.job;
  const start = job === null ? null : startedAt(job);
  if (job === null || start === null) return null;
  const end = job.state === 'running' ? now.value : job.ended_at ? Date.parse(job.ended_at) : NaN;
  return Number.isNaN(end) ? null : elapsedText(end - start.getTime());
});
const summary = computed(() => {
  const job = props.job;
  if (job === null) return '';
  const parts = ['Started ' + startedWhen(job)];
  if (job.state === 'running' && duration.value) parts.push('for ' + duration.value);
  if (job.state === 'running' && job.progress.percent !== null) parts.push(Math.floor(job.progress.percent) + '%');
  if (job.progress.detail) parts.push(job.progress.detail);
  return parts.join(' · ');
});
/** How it ended, in a sentence: after how long, where it stopped and why when the server knows. */
const outcome = computed(() => {
  const job = props.job;
  if (job === null || job.state === 'running') return '';
  const after = duration.value ? ' after ' + duration.value : '';
  if (job.state === 'ok') return 'Done' + (duration.value ? ' in ' + duration.value : '') + '.';
  if (job.state === 'interrupted') return 'Stopped' + after + '.';
  const step = failedStep(job);
  return 'Failed' + after + (step ? ' at the step “' + step + '”' : '') + '.' + (job.progress.failure ? ' ' + job.progress.failure : '');
});
const emit = defineEmits<{close: []; stop: [job: Job]}>();
const text = ref('');
const follow = ref(true);
const pre = useTemplateRef<HTMLPreElement>('pre');
let offset = 0;
let timer: number | undefined;

async function poll(id: string): Promise<void> {
  try {
    const data = await api<LogChunk>('/api/panel/job/' + id + '/log?offset=' + offset);
    if (props.job?.id !== id) return;
    if (data.text) {
      if (offset === 0 && data.offset - data.text.length > 0) text.value = '… (start cut)\n';
      text.value += data.text;
      if (follow.value) {
        await nextTick();
        if (pre.value) pre.value.scrollTop = pre.value.scrollHeight;
      }
    }
    offset = data.offset;
  } catch (error) {
    toast(messageOf(error), true);
    return;
  }
  timer = window.setTimeout(() => void poll(id), running.value ? 1500 : 5000);
}

watch(() => props.job?.id, (id) => {
  window.clearTimeout(timer);
  text.value = '';
  offset = 0;
  follow.value = true;
  document.body.classList.toggle('sheet-open', id !== undefined);
  if (id !== undefined) void poll(id);
}, {immediate: true});

function onKey(event: KeyboardEvent): void {
  if (event.key === 'Escape' && props.job) emit('close');
}
onMounted(() => document.addEventListener('keydown', onKey));
onBeforeUnmount(() => {
  window.clearTimeout(timer);
  document.removeEventListener('keydown', onKey);
});
</script>

<template>
  <div class="sheet-backdrop" :class="{open: job !== null}" @click="emit('close')"></div>
  <section class="sheet" :class="{open: job !== null}" role="dialog" aria-modal="true" aria-labelledby="sheet-title">
    <div class="sheet-grabber" aria-hidden="true"></div>
    <div class="sheet-head">
      <div class="grow"><h2 id="sheet-title">{{ job?.label ?? 'Details' }}</h2><p class="sheet-sub">{{ summary }}</p></div>
      <span v-if="job" class="pill" :class="JOB_STATE[job.state][1]">{{ JOB_STATE[job.state][0] }}</span>
      <button v-if="job && running && job.stoppable" class="btn small danger" type="button" @click="emit('stop', job)"><Icon
        name="stop"/>Stop</button>
      <button v-if="running" class="btn small" type="button" :aria-pressed="follow"
              title="Keep the end of the log in view as it grows" @click="follow = !follow">Follow</button>
      <button class="btn small" type="button" @click="emit('close')">Close</button>
    </div>
    <div class="sheet-body">
      <p v-if="job && outcome" class="sheet-outcome">{{ outcome }}<small v-if="job.state === 'failed' && job.code !== null">Exit
        code {{ job.code }}.</small></p>
      <JobSteps v-if="job && job.progress.steps.length > 1" :steps="job.progress.steps"/>
      <h3 class="sheet-section">Log</h3>
      <pre ref="pre" class="term" tabindex="0" aria-label="Log of the job">{{ text }}</pre>
    </div>
  </section>
</template>
