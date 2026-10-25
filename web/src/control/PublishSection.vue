<script setup lang="ts">
// Publish (Models tab): our three networks in use as one package, for other people (scripts/publish_networks.sh).
// Prepare exports the cutter, classifier and decider to ONNX, checks each against PyTorch and writes the model card:
// nothing leaves the server. Publish uploads a prepared package to Hugging Face, public, after a question.
import {computed} from 'vue';
import {isoWhen} from '../common/format';
import type {ActionName, ActionOptions, ActionState, Job, NetworkPackage, PanelStatus} from '../common/panel-types';
import ActionList from './ActionList.vue';
import JobProgress from './JobProgress.vue';
import {JOB_STATE, jobOf, startedWhen} from './jobs';

const props = defineProps<{status: PanelStatus | null; inUse: Record<string, string | null>}>();
const emit = defineEmits<{run: [name: ActionName, state: ActionState, start?: undefined, options?: ActionOptions];
  stop: [job: Job]; details: [job: Job]}>();

const PREPARE: [ActionName, string][] = [['networks_prepare',
  'Exports the cutter, classifier and decider in use to ONNX, checks each against PyTorch and writes the model card. '
  + 'Nothing is uploaded.']];
const ROLES = ['cutter', 'classifier', 'decider'] as const;

const job = computed(() => jobOf(props.status, ['networks_prepare', 'networks_publish']));
/** Running or failed: in place of the buttons; one done is a line. */
const bigJob = computed(() => job.value && (job.value.state === 'running' || job.value.state === 'failed') ? job.value : null);
const repo = computed(() => props.status?.networks?.repo ?? '');
const latest = computed((): NetworkPackage | null => props.status?.networks?.packages[0] ?? null);
const publishState = computed(() => props.status?.actions.networks_publish ?? null);
/** The package holds other networks than those in use now: a new one is needed to publish these. */
const outdated = computed(() => latest.value !== null && ROLES.some((role) => props.inUse[role] && latest.value!.networks[role] !== props.inUse[role]));

/** "2026-10-25_1941-cutter" out of "cutter:2026-10-25_1941-cutter/detector-sizes/1344x760". */
function runOf(id: string | null): string {
  return id ? id.slice(id.indexOf(':') + 1).split('/')[0] ?? id : '–';
}

function publish(): void {
  if (latest.value && publishState.value) emit('run', 'networks_publish', publishState.value, undefined, {package: latest.value.name});
}
</script>

<template>
  <section class="card publish" id="publish" aria-labelledby="publish-title">
    <h2 id="publish-title">Publish</h2>
    <p class="note">The three networks in use, as one package, on Hugging Face for other people
      (<a :href="'https://huggingface.co/' + repo" target="_blank" rel="noopener">{{ repo }}</a>). Every package is checked here
      before it can be published.</p>
    <div class="train-slot">
      <JobProgress v-if="bigJob" :job="bigJob" @details="emit('details', $event)" @stop="emit('stop', $event)"/>
      <template v-else>
        <ActionList :items="PREPARE" :actions="status?.actions ?? null" @run="(name, state) => emit('run', name, state)"/>
        <div v-if="latest" class="package-row" id="latest-package">
          <div class="latest-what">
            <span class="latest-kicker">Latest package · {{ isoWhen(latest.created) }} · {{ (latest.bytes / 1e6).toFixed(1) }} MB
              <span v-if="latest.published" class="latest-use">published {{ isoWhen(latest.published.at) }}</span></span>
            <span class="latest-name">{{ latest.name }}</span>
            <span class="latest-text">{{ ROLES.map((role) => role + ' ' + runOf(latest!.networks[role])).join(' · ') }}</span>
            <span v-if="outdated" class="latest-text warn-text">The models in use changed since: prepare a new package to publish them.</span>
          </div>
          <a v-if="latest.published" class="btn small" :href="latest.published.url" target="_blank" rel="noopener">Open</a>
          <button v-else class="btn primary small" type="button" id="run-networks_publish"
                  :disabled="!publishState?.allowed" :title="publishState?.allowed ? '' : publishState?.reason"
                  @click="publish">Publish on Hugging Face</button>
        </div>
      </template>
      <p v-if="job && !bigJob" class="last-line">Last: {{ job.label }}, {{ startedWhen(job) }} ·
        {{ job.state === 'interrupted' ? 'stopped' : JOB_STATE[job.state][0] }} ·
        <button type="button" class="link-button" @click="emit('details', job)">Details</button></p>
    </div>
  </section>
</template>
