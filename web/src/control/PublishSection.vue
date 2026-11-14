<script setup lang="ts">
// Publish › Hugging Face, optional and apart from the car: our three networks in use as one package, for other people
// (scripts/publish_networks.sh). One button, Publish on Hugging Face: it exports the cutter, classifier and decider in
// use to ONNX, checks each against PyTorch, writes the model card and uploads the package, public, after a question
// (a package of these networks prepared before and not published yet is uploaded as it is). "Prepare only" makes the
// package and uploads nothing, to read it first. The comma never reads it: its networks come with Publish, above
// (CarPublishSection.vue).
import {computed} from 'vue';
import {isoWhen} from '../common/format';
import type {ActionName, ActionOptions, ActionState, Job, NetworkPackage, PanelStatus} from '../common/panel-types';
import JobProgress from './JobProgress.vue';
import {JOB_STATE, jobOf, startedWhen} from './jobs';

const props = defineProps<{status: PanelStatus | null; inUse: Record<string, string | null>}>();
const emit = defineEmits<{run: [name: ActionName, state: ActionState, start?: undefined, options?: ActionOptions];
  stop: [job: Job]; details: [job: Job]}>();

const ROLES = ['cutter', 'classifier', 'decider'] as const;
/** A job that ended well, in words. */
const DONE: Partial<Record<ActionName, string>> = {networks_prepare: 'Package ready, nothing uploaded',
  networks_publish: 'Published on Hugging Face'};

const job = computed(() => jobOf(props.status, ['networks_prepare', 'networks_publish']));
/** Running or failed: in place of the buttons; one done is a line. */
const bigJob = computed(() => job.value && (job.value.state === 'running' || job.value.state === 'failed') ? job.value : null);
const repo = computed(() => props.status?.networks?.repo ?? '');
const latest = computed((): NetworkPackage | null => props.status?.networks?.packages[0] ?? null);
const publishState = computed(() => props.status?.actions.networks_publish ?? null);
const prepareState = computed(() => props.status?.actions.networks_prepare ?? null);
const account = computed(() => props.status?.networks?.account ?? null);
/** The package holds other networks than those in use now: a new one is needed to publish these. */
const outdated = computed(() => latest.value !== null && ROLES.some((role) => props.inUse[role] && latest.value!.networks[role] !== props.inUse[role]));

/** "2026-10-25_1941-cutter" out of "cutter:2026-10-25_1941-cutter/detector-sizes/1344x760". */
function runOf(id: string | null): string {
  return id ? id.slice(id.indexOf(':') + 1).split('/')[0] ?? id : '–';
}

const ROLE_WORDS = {cutter: 'the cutter', classifier: 'the classifier', decider: 'the applicability network'} as const;
type Licensed = {license?: string | null; mapillary?: (keyof typeof ROLE_WORDS)[]} | null | undefined;
/** "Licence CC BY-NC-SA 4.0: the cutter and the classifier learnt from Mapillary's pictures, non-commercial use only",
 * "Licence MIT"; '' when the server does not say. */
function licenceOf(of: Licensed): string {
  if (!of?.license) return '';
  const learnt = (of.mapillary ?? []).map((role) => ROLE_WORDS[role]);
  return 'Licence ' + of.license + (learnt.length ? ': ' + learnt.join(' and ') + ' learnt from Mapillary’s pictures, non-commercial use only' : '');
}
/** The package the button uploads as it is: the latest one, when it holds the networks in use and is not published. */
const ready = computed(() => latest.value !== null && !outdated.value && !latest.value.published ? latest.value : null);
/** The networks in use are on Hugging Face already: nothing to publish. */
const onHub = computed(() => latest.value !== null && !outdated.value && latest.value.published !== null);
/** What is published at the click: that package, or a new one of the networks in use (the server's in_use). */
const pending = computed((): Licensed => ready.value ?? props.status?.networks?.in_use);
const licence = computed(() => licenceOf(onHub.value ? latest.value : pending.value));

function publish(): void {
  if (!publishState.value) return;
  const license = pending.value?.license;
  emit('run', 'networks_publish', publishState.value, undefined,
    {...(ready.value ? {package: ready.value.name} : {}), ...(license ? {license} : {})});
}

function prepare(): void {
  if (prepareState.value) emit('run', 'networks_prepare', prepareState.value);
}
</script>

<template>
  <section class="card publish" id="publish" aria-labelledby="publish-title">
    <h2 id="publish-title">Hugging Face <small class="h2-aside">optional</small></h2>
    <p class="note">Not needed for the car: the three networks in use as one package for other people
      (<a :href="'https://huggingface.co/' + repo" target="_blank" rel="noopener">{{ repo }}</a>). Every package is checked here
      before it can be uploaded.</p>
    <p class="hf-line" id="hf-line">
      <template v-if="account?.source === 'portal'">Hugging Face: connected as <strong>{{ account.user }}</strong> ·
        <a href="/account#huggingface">Settings ›</a></template>
      <template v-else-if="account?.source === 'server'">Hugging Face: this server's own login ·
        <a href="/account#huggingface">Connect a token in Settings ›</a></template>
      <template v-else-if="account">Hugging Face: not connected ·
        <a href="/account#huggingface">Connect it in Settings ›</a></template>
    </p>
    <div class="train-slot">
      <JobProgress v-if="bigJob" :job="bigJob" @details="emit('details', $event)" @stop="emit('stop', $event)"/>
      <template v-else>
        <div class="package-row" id="publish-row">
          <div class="latest-what">
            <template v-if="onHub && latest">
              <span class="latest-kicker pub-done">✓ The networks in use are on Hugging Face · {{ isoWhen(latest.published!.at) }}</span>
              <span class="latest-name">{{ latest.name }}</span>
            </template>
            <template v-else>
              <span class="latest-kicker">The networks in use, for other people</span>
              <span class="latest-name">{{ ready ? ready.name : 'A new package' }}</span>
            </template>
            <span class="latest-text">{{ ROLES.map((role) => role + ' ' + runOf(inUse[role] ?? null)).join(' · ') }}</span>
            <span v-if="licence" class="latest-text whole" id="package-licence">{{ licence }}</span>
            <span v-if="!onHub" class="latest-text whole">{{ ready
              ? 'Prepared ' + isoWhen(ready.created) + ' and checked: the button uploads it.'
              : 'The button exports them to ONNX, checks each against PyTorch, writes the model card and uploads the package: about two minutes.' }}</span>
          </div>
          <a v-if="onHub && latest" class="btn small" :href="latest.published!.url" target="_blank" rel="noopener">Open</a>
          <button v-else class="btn primary small" type="button" id="run-networks_publish"
                  :disabled="!publishState?.allowed" :title="publishState?.allowed ? '' : publishState?.reason"
                  @click="publish">Publish on Hugging Face</button>
        </div>
        <!-- How the last one ended, and the package alone: a preparation takes less than a minute. -->
        <p class="last-line" id="publish-last"><template v-if="job"><span :class="{'pub-done': job.state === 'ok'}"><template
          v-if="job.state === 'ok'">✓ {{ DONE[job.action] ?? job.label }}, {{ job.ended_at ? isoWhen(job.ended_at) : startedWhen(job) }}</template><template
          v-else>Last: {{ job.label }}, {{ startedWhen(job) }} · {{ job.state === 'interrupted' ? 'stopped' : JOB_STATE[job.state][0] }}</template></span> ·
          <button type="button" class="link-button" @click="emit('details', job)">Details</button> · </template><button
          type="button" class="link-button" id="run-networks_prepare" :disabled="!prepareState?.allowed"
          :title="prepareState?.allowed ? 'Makes the package and its model card to read them first: nothing is uploaded' : prepareState?.reason"
          @click="prepare">Prepare only, upload nothing</button></p>
      </template>
    </div>
  </section>
</template>
