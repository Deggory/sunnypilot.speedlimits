<script setup lang="ts">
// Applicability on Models, in Review helpers (#91): the network that says whose road each speed sign is of, ours or
// another, only to help Sign review (the one in use proposes OURS or another on the speed signs of Prepare's Our
// networks). Never on the car nor in the brain since #90: the brain learns whose road a sign is of from the reviews. Its
// training is here, in the look of Fit box: the one in use and what the car has (carLine, from ModelsTab), Train
// (scripts/run_experiment_b.sh, the panel action "train_decider": its three versions, about 15 minutes on the GPU),
// where it starts (network-start.ts, the versions of ModelsTab's payload), its job while it runs with its curves, its
// Settings (TrainSettingsSection, its epochs), the last training, the curves of the last one folded, and its versions
// folded (the default slot: ModelsTab's VersionsCard, with How it works), opened by the address
// /models#role-decider[:<id>], their row naming the newest version when it is not the one in use. A choice whose files
// are gone says so, as the cards at the top of Models do. Train › Pipeline keeps its step Applicability, with a Start
// from of its own.
import {computed, ref, toRef, watch} from 'vue';
import {isoWhen, plural, sentence} from '../common/format';
import type {ActionName, ActionOptions, ActionState, Job, ModelRole, ModelVersion, PanelStatus, RoleModels} from '../common/panel-types';
import Icon from '../components/Icon.vue';
import {pct} from './evaluate';
import JobProgress from './JobProgress.vue';
import {JOB_STATE, jobOf, startedWhen, twoGpusTip} from './jobs';
import {chosenText, inUseOf, metricsOf} from './models';
import {useNetworkStart} from './network-start';
import NetworkStartPicker from './NetworkStartPicker.vue';
import TrainingCharts from './TrainingCharts.vue';
import TrainSettingsSection from './TrainSettingsSection.vue';

const props = defineProps<{status: PanelStatus | null; role: RoleModels | null;
  versionsOf: (role: ModelRole) => readonly ModelVersion[] | undefined;
  carLine: {tone: 'same' | 'other' | 'none'; text: string; more?: string} | null; versionsOpen?: boolean}>();
const emit = defineEmits<{run: [name: ActionName, state: ActionState, start?: undefined, options?: ActionOptions];
  stop: [job: Job]; details: [job: Job]}>();

const state = computed((): ActionState | null => props.status?.actions?.train_decider ?? null);
const job = computed(() => jobOf(props.status, ['train_decider']));
const running = computed(() => job.value?.state === 'running');
const bigJob = computed(() => job.value && (running.value || job.value.state === 'failed') ? job.value : null);
/** Where its next training starts: the run in use, until another is chosen. */
const start = useNetworkStart('decider', toRef(props, 'status'), props.versionsOf);

const inUse = computed(() => props.role ? inUseOf(props.role) : null);
/** The one chosen whose files are no longer on this server (its id), else null. */
const missing = computed(() => props.role?.active && !props.role.active.found ? props.role.active.id : null);
/** The one in use in one line: when, who chose it and when, its “Another” found on validation and on the test, what the
 * car has. */
const inUseText = computed(() => {
  const version = inUse.value;
  const numbers = version ? metricsOf(version).filter((metric) => metric.value !== null)
    .map((metric) => metric.label + ' ' + pct(metric.value)) : [];
  return [version ? (version.trained_at ? isoWhen(version.trained_at) : version.run) : '',
    version && props.role?.active ? chosenText(props.role.active) : '', ...numbers,
    props.carLine?.text ?? ''].filter(Boolean).join(' · ');
});
const what = 'Says whose road each speed sign is of: ours or another. Only for Sign review: the one in use proposes OURS '
  + 'or another on the speed signs Our networks propose. Never on the car nor in the brain, which learns whose road a sign '
  + 'is of from your reviews.';
const trainText = computed(() => 'Three versions of applicability: the crop alone, with position and speed, with the scene'
  + start.words + '. About 15 minutes on the GPU.');
const trainTitle = computed(() => [trainText.value, twoGpusTip(props.status?.system)].filter(Boolean).join('\n'));
function train(): void {
  if (state.value?.allowed) emit('run', 'train_decider', state.value, undefined, start.option);
}

/** The curves of the last training, while its job is still listed (they are not kept elsewhere). */
const curved = computed(() => {
  const curves = job.value?.progress.curves ?? {};
  return !running.value && Object.values(curves).some((curve) => Boolean(curve?.points.length));
});
/** Its versions, folded: opened by the address, or by a tap. */
const showVersions = ref(Boolean(props.versionsOpen));
watch(() => props.versionsOpen, (value) => { if (value) showVersions.value = true; });
const count = computed(() => props.role?.versions.length ?? 0);
/** The newest version, when it is not the one in use: named in the row of the versions (Train's old Latest row). */
const newest = computed(() => {
  const version = props.role?.versions[0] ?? null;
  return version && !version.in_use ? version : null;
});
const versionsText = computed(() => {
  const version = newest.value;
  if (version === null) return 'to compare and to choose the one Sign review uses';
  const other = version.role === 'decider' ? version.validation?.other_road_recall ?? null : null;
  return ['newest ' + (version.trained_at ? isoWhen(version.trained_at) : version.run) + (version.finished ? '' : ', not finished'),
    typeof other === 'number' ? '“Another” found on validation ' + pct(other) : '', 'not in use'].filter(Boolean).join(' · ');
});
</script>

<template>
  <section class="card import-card labeler-card" id="role-decider" aria-labelledby="ap-title">
    <h3 id="ap-title" class="rl-title" :title="what">Applicability</h3>
    <JobProgress v-if="bigJob" :job="bigJob" @details="emit('details', $event)" @stop="emit('stop', $event)"/>
    <TrainingCharts v-if="running && job" :progress="job.progress" :running="true"/>
    <template v-else>
      <div class="import-row">
        <span class="import-icon" :class="{fresh: !inUse}" aria-hidden="true"><Icon name="train"/></span>
        <div class="import-what" :title="carLine?.more ?? ''">
          <strong v-if="missing"><code>{{ missing }}</code></strong>
          <strong v-else>{{ inUse ? inUse.label : count ? 'None chosen' : 'Not trained yet' }}</strong>
          <span v-if="missing" id="applicability-missing">Its files are no longer on this server: pick another one.</span>
          <span v-else-if="inUseText" id="applicability-in-use">{{ inUseText }}</span>
        </div>
        <div class="rl-buttons">
          <button type="button" class="btn primary" id="run-train_decider" :disabled="!state?.allowed" :title="trainTitle"
                  @click="train">Train</button>
        </div>
      </div>
      <p v-if="carLine?.tone === 'other' && carLine.more" class="import-why warn">{{ carLine.more }}</p>
      <p v-if="state && !state.allowed" class="import-why">{{ sentence(state.reason) }}</p>
      <NetworkStartPicker :start="start" id="applicability-start"/>
    </template>
    <TrainSettingsSection training="train_decider" :running="running"/>
    <p v-if="job && !bigJob" class="import-last">Last training {{ job.run ? job.run + ', ' : '' }}{{ startedWhen(job) }} ·
      {{ job.state === 'interrupted' ? 'stopped' : JOB_STATE[job.state][0] }} ·
      <button type="button" class="link-button" @click="emit('details', job)">Details</button></p>
    <details v-if="curved && job" class="latest-row curves-row">
      <summary><span class="latest-what"><span class="latest-name">Curves of the last training</span>
        <span class="latest-text">the chosen measure and loss, epoch by epoch</span></span><span class="latest-chev" aria-hidden="true">›</span></summary>
      <TrainingCharts :progress="job.progress" :running="false"/>
    </details>
    <details v-if="count" class="latest-row ap-versions" id="applicability-versions" :open="showVersions"
             @toggle="showVersions = ($event.target as HTMLDetailsElement).open">
      <summary><span class="latest-what"><span class="latest-name">{{ plural(count, 'version', 'versions') }}</span>
        <span class="latest-text">{{ versionsText }}</span></span><span class="latest-chev" aria-hidden="true">›</span></summary>
      <slot/>
    </details>
  </section>
</template>
