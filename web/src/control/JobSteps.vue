<script setup lang="ts">
// The steps of a job as a timeline: done (check), current (pulsing), still to do, failed (red).
import type {JobStep, StepProgress} from '../common/panel-types';

// The current step also shows how far it is (images checked, batches, the file being downloaded).
defineProps<{steps: JobStep[]; current?: StepProgress | null | undefined}>();
const STATE_TEXT: Record<JobStep['state'], string> = {done: 'done', current: 'running', todo: 'to do', failed: 'failed'};
</script>

<template>
  <ol class="jp-steps" aria-label="Steps">
    <li v-for="(step, index) in steps" :key="step.key" :class="step.state" :aria-current="step.state === 'current' ? 'step' : undefined">
      <span class="jp-mark" aria-hidden="true"><svg v-if="step.state === 'done'" viewBox="0 0 12 12"><path d="M2.5 6.2l2.3 2.3 4.7-5"
        fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"/></svg><svg
        v-else-if="step.state === 'failed'" viewBox="0 0 12 12"><path d="M3.5 3.5l5 5M8.5 3.5l-5 5" fill="none" stroke="currentColor"
        stroke-width="1.9" stroke-linecap="round"/></svg><template v-else>{{ index + 1 }}</template></span>
      <span class="jp-step-body">
        <span class="jp-step-label">{{ step.label }}<span class="sr-only"> ({{ STATE_TEXT[step.state] }})</span></span>
        <span v-if="step.state === 'done'" class="jp-step-pct">100%</span>
        <span v-else-if="step.state === 'current' && current?.percent != null" class="jp-step-pct">{{ Math.floor(current.percent) }}%</span>
        <template v-if="step.state === 'current' && current">
          <span class="jp-step-bar" :class="{indeterminate: current.percent == null}" role="progressbar" :aria-label="step.label"
                aria-valuemin="0" aria-valuemax="100" :aria-valuenow="current.percent ?? undefined"><span
            :style="current.percent == null ? undefined : {width: current.percent + '%'}"></span></span>
          <span class="jp-step-text">{{ current.text }}</span>
        </template>
      </span>
    </li>
  </ol>
</template>
