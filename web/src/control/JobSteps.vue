<script setup lang="ts">
// The steps of a job as a timeline: done (check), current (pulsing), still to do, failed (red).
import type {JobStep} from '../common/panel-types';

defineProps<{steps: JobStep[]}>();
const STATE_TEXT: Record<JobStep['state'], string> = {done: 'fatto', current: 'in corso', todo: 'da fare', failed: 'fallito'};
</script>

<template>
  <ol class="jp-steps" aria-label="Passi">
    <li v-for="(step, index) in steps" :key="step.key" :class="step.state" :aria-current="step.state === 'current' ? 'step' : undefined">
      <span class="jp-mark" aria-hidden="true"><svg v-if="step.state === 'done'" viewBox="0 0 12 12"><path d="M2.5 6.2l2.3 2.3 4.7-5"
        fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"/></svg><svg
        v-else-if="step.state === 'failed'" viewBox="0 0 12 12"><path d="M3.5 3.5l5 5M8.5 3.5l-5 5" fill="none" stroke="currentColor"
        stroke-width="1.9" stroke-linecap="round"/></svg><template v-else>{{ index + 1 }}</template></span>
      <span class="jp-step-label">{{ step.label }}<span class="sr-only"> ({{ STATE_TEXT[step.state] }})</span></span>
    </li>
  </ol>
</template>
