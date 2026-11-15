<script setup lang="ts">
// Models: first the networks in use (the cutter, the classifier, the brain and UserNet; docs/NEURAL_NETWORK.md), one
// card each with its numbers; then the versions of one of them (the switch, or a tap on its card) as a table to
// compare (VersionsCard.vue: the numbers that judge that role in columns, the one in use marked; only the recent versions
// of recentOf, the older ones behind “Show older versions”, nothing is deleted; a row opens its whole card, ModelItem:
// details, Use this one, the run). A role has no model in use until someone picks one: there is no default. The
// networks the comma downloads (models/on-car.json, written by nnslr on-car) are marked “on the car”. The cutter is
// SignNet, the same network the comma runs: one at the car's band can be put in use; the old Detectors are listed to
// compare only (an old one still in use says so on its card). A SignNet's card also has “Try on the comma now”
// (scripts/try_on_comma.sh: it and the classifier in use copied straight onto the comma, checked first with its own
// code, until its next software update; models/on-comma-trial.json), whose job shows here while it runs, with “Back to
// the build's networks”. The networks in use reach the car with Publish (/publish: scripts/send_to_car.sh). The runs
// themselves are on Results (/evaluate). The settings of the comma's vision limit are no longer shown here (8 Nov 2026,
// his call): Publish sends them as they are kept (nnslr_tools.car_settings). At the end, Review helpers (#91): the
// networks that help only the review work, each with its training: applicability (ApplicabilityCard.vue, its versions
// in a VersionsCard of their own: it proposes OURS or another in Sign review), the road labeler (RoadLabelerCard.vue: an
// offline network that prepares Speed review, trained and run from there) and Fit box (BoxFitCard.vue: key C of the
// review page trained again on a V100, with its last measures).
import {computed, nextTick, ref, watch} from 'vue';
import {isoWhen} from '../common/format';
import type {ActionName, ActionOptions, ActionState, Job, ModelRole, ModelVersion, PanelStatus, RoleModels} from '../common/panel-types';
import {askConfirm} from '../composables/confirm';
import ApplicabilityCard from './ApplicabilityCard.vue';
import {pct} from './evaluate';
import JobProgress from './JobProgress.vue';
import {JOB_STATE, jobOf, startedWhen} from './jobs';
import BoxFitCard from './BoxFitCard.vue';
import RoadLabelerCard from './RoadLabelerCard.vue';
import Segmented from './Segmented.vue';
import {ROLE_TEXT, chosenText, inUseOf, metricsOf, putInUse, restoreQuestion, tryQuestion, useModels,
  type CarState} from './models';
import SectionHead from './SectionHead.vue';
import VersionsCard from './VersionsCard.vue';

const props = defineProps<{status: PanelStatus | null}>();
const emit = defineEmits<{run: [name: ActionName, state: ActionState, start?: undefined, options?: ActionOptions];
  stop: [job: Job]; details: [job: Job]}>();
const {payload, error, refresh} = useModels();
const roles = computed((): RoleModels[] => payload.value?.roles ?? []);
/** Read-only server: the choice is off, as the buttons of the other tabs. */
const readOnly = computed(() => (props.status !== null && !props.status.actions_enabled) || Boolean(payload.value?.active_problem));
/** The version being put in use, so its button waits for the answer. */
const busy = ref<string | null>(null);

/** The networks of the grid at the top. Applicability, which helps only Sign review, is in Review helpers at the bottom
 * (#91). */
const ROLES: readonly ModelRole[] = ['cutter', 'classifier', 'brain', 'users'];
const topRoles = computed(() => roles.value.filter((role) => ROLES.includes(role.role)));
const deciderRole = computed(() => roles.value.find((role) => role.role === 'decider') ?? null);
/** The role whose versions are listed: the address's (/models#role-classifier, or /models#role-brain:<version id>,
 * Train's link to its latest version), else the cutter. /models#role-decider[:<id>] opens applicability's versions in
 * Review helpers. */
const hashRole = location.hash.split(':')[0];
const fromHash = [...ROLES, 'decider' as const].find((role) => hashRole === '#role-' + role) ?? null;
/** The version the address names, opened and scrolled to once drawn (VersionsCard's openId), or ''. */
const fromHashVersion = fromHash && location.hash.includes(':') ? decodeURIComponent(location.hash.slice(location.hash.indexOf(':') + 1)) : '';
const picked = ref<ModelRole>(fromHash && fromHash !== 'decider' ? fromHash : 'cutter');
/** The version of the address still to open in the top table: only on arrival, not again after another network was
 * picked and this one picked back. */
const linkedVersion = ref(fromHash && fromHash !== 'decider' ? fromHashVersion : '');
watch(picked, () => { linkedVersion.value = ''; });
const pickedRole = computed(() => topRoles.value.find((role) => role.role === picked.value) ?? null);
/** A tap on a card in use: its versions, scrolled to (below the cards on a phone). */
function pick(role: ModelRole): void {
  picked.value = role;
  void nextTick(() => document.querySelector('.versions-card[data-role="' + role + '"]')
    ?.scrollIntoView({behavior: 'smooth', block: 'nearest'}));
}
// Arrived from a link to a role: its versions, once they are drawn (VersionsCard opens the version named, if any);
// applicability's in Review helpers, unfolded.
let arrived = fromHash === null;
watch(payload, (value) => {
  if (arrived || value === null || fromHash === null) return;
  arrived = true;
  if (fromHash === 'decider') {
    if (!fromHashVersion) void nextTick(() => document.getElementById('review-helpers')?.scrollIntoView({behavior: 'smooth', block: 'start'}));
  } else if (!fromHashVersion) pick(fromHash);
}, {immediate: true});
const roleOptions = computed(() => topRoles.value.map((role) => ({key: role.role, label: role.label})));
/** The versions of applicability, from the same payload: the start of its next training (network-start.ts). */
const versionsOf = (role: ModelRole) => roles.value.find((item) => item.role === role)?.versions;
/** The cutter in use is an old Detector, chosen before SignNet could be, or the brain in use takes the driver's pedals and
 * the car's speed, chosen before 10 Nov 2026 (the server says so; an older one: a cutter not a SignNet). */
function oldInUse(role: RoleModels): boolean {
  if (role.old_kind_in_use !== undefined) return role.old_kind_in_use;
  if (role.role !== 'cutter') return false;
  const used = inUseOf(role);
  return used?.role === 'cutter' && used.kind !== 'signnet';
}
/** The name of a training whatever its seed: the seeds of a brain are runs of their own, NAME-s0, NAME-s1, NAME-s2. */
const seedsOf = (run: string): string => run.replace(/-s\d+$/, '');
/** The versions of a role worth a look, newest first: the one in use and the others of its run or of its training, the newer ones that can
 * be put in use, the one on the car or tried on the comma, a training running now; with none in use, every one that can
 * be. The others (the old Detectors, the SignNets that do not fit the car, stopped trainings, the older versions) wait
 * behind “Show older versions”: nothing is deleted. */
function recentOf(role: RoleModels): ModelVersion[] {
  const used = inUseOf(role);
  const since = used?.trained_at ? Date.parse(used.trained_at) : null;
  const running = new Set((props.status?.jobs ?? []).filter((job) => job.state === 'running').map((job) => job.run));
  const tried = role.role === 'cutter' ? trial.value?.signnet.run : role.role === 'classifier' ? trial.value?.classifier.run : null;
  const recent = role.versions.filter((version) => version.in_use || version.on_car || version.run === used?.run
    || (used !== null && seedsOf(version.run) === seedsOf(used.run))
    || version.run === tried || running.has(version.run)
    || (version.finished && version.selectable !== false && (used === null
      || (since !== null && version.trained_at !== null && Date.parse(version.trained_at) > since))));
  return recent.length ? recent : role.versions;
}
const recent = computed(() => Object.fromEntries(roles.value.map((role) => [role.role, recentOf(role)])) as Record<ModelRole, ModelVersion[]>);
/** The version of a role that runs on the comma, or null. */
const carOf = (role: RoleModels): ModelVersion | null => role.versions.find((version) => version.on_car) ?? null;
/** Whether the comma runs the version in use of a role, in words. Two versions often have the same name: the time
 * of the one on the car tells them apart. The comma runs the cutter, the classifier and the brain (Publish puts the ones
 * in use there); a trial replaces the first two until its next software update. UserNet is not on the car: no line. The
 * applicability network never goes since #90: one an older publication left is taken away by the next. */
interface CarLine {tone: 'same' | 'other' | 'none'; text: string; more?: string}
function carLine(role: RoleModels): CarLine | null {
  if (payload.value?.on_car_problem || role.role === 'users') return null;
  const used = inUseOf(role);
  const car = carOf(role);
  const tried = role.role === 'cutter' ? trial.value?.signnet.run : role.role === 'classifier' ? trial.value?.classifier.run : null;
  if (tried) return used?.run === tried ? {tone: 'same', text: 'On the comma: this one, as a trial'}
    : {tone: 'other', text: 'On the comma: a trial of ' + tried};
  if (role.role === 'decider') return car === null ? {tone: 'none', text: 'Not on the car: only for Sign review'}
    : {tone: 'other', text: 'In the branch: an old one, ' + (car.trained_at ? isoWhen(car.trained_at) : car.run),
      more: 'The next Publish takes it away: the car takes every sign.'};
  const more = 'Publish, on the Publish tab, puts it there.';
  if (car === null) return used ? {tone: 'other', text: 'In the branch: none yet', more} : {tone: 'none', text: 'In the branch: none'};
  const when = car.trained_at ? isoWhen(car.trained_at) : car.run;
  if (used === null) return {tone: 'none', text: 'In the branch: ' + car.label + ', ' + when};
  if (car.id === used.id) return payload.value?.car_build_pending
    ? {tone: 'other', text: 'In the branch: this one, once its build is finished'} : {tone: 'same', text: 'In the branch: this one'};
  const age = car.trained_at && used.trained_at ? (Date.parse(car.trained_at) < Date.parse(used.trained_at) ? 'an older one' : 'a newer one')
    : 'another one';
  return {tone: 'other', text: 'In the branch: ' + age + ', ' + when, more};
}
const carLines = computed(() => Object.fromEntries(roles.value.map((role) => [role.role, carLine(role)])) as Record<ModelRole, CarLine | null>);
/** The published models on the car, in a few words: "speed-limits-models 62dca08, 28 Oct". */
const carText = computed((): string => {
  const car = payload.value?.on_car;
  if (!car) return '';
  const name = car.repository?.split('/').pop() ?? 'the published models';
  return [name + (car.commit ? ' ' + car.commit.slice(0, 7) : ''), car.version ?? ''].filter(Boolean).join(', ');
});
/** What Prepare uses while no classifier is in use: the newest classifier with “not a sign”
 * (nnslr_tools.preannotate.our_networks); the versions are newest first. */
function fallbackOf(role: RoleModels): ModelVersion | null {
  return role.role === 'classifier' && inUseOf(role) === null ? role.versions.find((version) => version.part === 'classifier') ?? null : null;
}

/** The last trial on the comma or return to the build's networks; running or failed, it is shown in full above the
 * versions (a publication to the car shows on the Publish tab). */
const carJob = computed(() => jobOf(props.status, ['try_on_comma', 'comma_restore']));
const bigCarJob = computed(() => carJob.value && (carJob.value.state === 'running' || carJob.value.state === 'failed') ? carJob.value : null);
const CAR_JOB_TEXT: Partial<Record<ActionName, string>> = {try_on_comma: 'Last trial on the comma',
  comma_restore: 'Last return to the build’s networks'};
const publishJob = computed(() => jobOf(props.status, ['send_to_car']));
const carHeading = (job: Job): string | undefined => !job.run ? undefined
  : 'SignNet ' + job.run + (job.action === 'try_on_comma' ? ' on the comma' : '');
const runningOf = (action: ActionName): string | null => {
  const job = action === 'send_to_car' ? publishJob.value : carJob.value;
  return job?.state === 'running' && job.action === action ? job.run ?? '' : null;
};
/** The networks being tried on the comma now (models/on-comma-trial.json), null when none. */
const trial = computed(() => payload.value?.comma_trial ?? null);
const car = computed((): CarState => ({
  sending: runningOf('send_to_car'),
  tryState: props.status?.actions.try_on_comma ?? null,
  trying: runningOf('try_on_comma') ?? runningOf('comma_restore'),
  onTrial: trial.value?.signnet.run ?? null,
}));
/** The trial in a few words: "copied 28 Oct 14:20 · ready: used from the next drive". */
const trialText = computed((): string => {
  const value = trial.value;
  if (value === null) return '';
  const classifier = value.classifier.run ? ', classifier ' + value.classifier.run : '';
  const state = value.state === 'ready' ? 'ready: used from the next drive' : 'being prepared on the comma';
  return [classifier ? classifier.slice(2) : '', value.copied_at ? 'copied ' + isoWhen(value.copied_at) : '', state]
    .filter(Boolean).join(' · ');
});
const restoreState = computed(() => props.status?.actions.comma_restore ?? null);
const restoreBlocked = computed((): string => {
  if (readOnly.value) return 'Read-only: the buttons are turned off on this server.';
  if (carJob.value?.state === 'running' || publishJob.value?.state === 'running') return 'Wait for the job running now.';
  const state = restoreState.value;
  if (state === null) return 'Not available on this server.';
  return state.allowed ? '' : state.reason.charAt(0).toUpperCase() + state.reason.slice(1) + '.';
});
/** Asks in plain words what a trial on the comma does, then starts the job. */
async function tryOnComma(version: ModelVersion): Promise<void> {
  const state = car.value.tryState ?? null;
  if (state === null || !state.allowed) return;
  const classifiers = roles.value.find((role) => role.role === 'classifier');
  if (!await askConfirm(tryQuestion(version, classifiers ? inUseOf(classifiers) : null))) return;
  emit('run', 'try_on_comma', state, undefined, {signnet: version.run});
}
async function restore(): Promise<void> {
  const state = restoreState.value;
  if (state === null || !state.allowed) return;
  if (!await askConfirm(restoreQuestion())) return;
  emit('run', 'comma_restore', state);
}

async function choose(version: ModelVersion): Promise<void> {
  if (busy.value !== null) return;
  busy.value = version.id;
  try {
    if (await putInUse(version, roles.value.find((role) => role.role === version.role)?.label ?? version.role)) await refresh();
  } finally {
    busy.value = null;
  }
}
</script>

<template>
  <SectionHead tab="models"/>

  <p v-if="payload === null" class="card empty">{{ error ? 'The models cannot be read: ' + error : 'Loading…' }}</p>
  <template v-else>
    <p v-if="payload.active_problem" class="banner" id="active-problem">{{ payload.active_problem }}: the choices cannot be shown or changed until it is fixed or removed.</p>

    <!-- The ones in use: tap one to list its versions below. -->
    <div class="inuse-grid" id="roles">
      <button v-for="role in topRoles" :key="role.role" type="button" class="inuse-card" :class="{picked: role.role === picked}"
              :id="'role-' + role.role" :data-role="role.role" :aria-pressed="role.role === picked" @click="pick(role.role)">
        <span class="inuse-role">{{ role.label }}<span class="inuse-tag" :class="{none: inUseOf(role) === null}">{{ inUseOf(role) ? 'In use' : 'None chosen' }}</span></span>
        <template v-if="inUseOf(role)">
          <strong class="inuse-name">{{ inUseOf(role)?.label }}</strong>
          <span class="inuse-when">{{ inUseOf(role)?.trained_at ? isoWhen(inUseOf(role)?.trained_at ?? null) : inUseOf(role)?.run }}<span
            v-if="role.active" class="inuse-chosen"> · {{ chosenText(role.active) }}</span></span>
          <span class="inuse-figs">
            <span v-for="metric in metricsOf(inUseOf(role) as ModelVersion).slice(0, 2)" :key="metric.key" class="inuse-fig">
              <b>{{ metric.value === null ? '–' : pct(metric.value) }}</b><small v-if="metric.short"><span class="fig-long">{{ metric.label }}</span><span
                class="fig-short">{{ metric.short }}</span></small><small v-else>{{ metric.label }}</small></span>
          </span>
        </template>
        <template v-else-if="role.active && !role.active.found">
          <strong class="inuse-name"><code>{{ role.active.id }}</code></strong>
          <span class="inuse-when">Its files are no longer on this server: pick another one.</span>
        </template>
        <template v-else>
          <strong class="inuse-name muted">{{ role.versions.length ? 'Pick one below' : 'Nothing trained yet' }}</strong>
          <span class="inuse-when">{{ fallbackOf(role) ? 'Prepare uses the newest, ' + fallbackOf(role)?.label + ' ' + isoWhen(fallbackOf(role)?.trained_at ?? null) + ', until you pick one.'
            : role.versions.length ? 'There is no default.' : ROLE_TEXT[role.role].emptyMore }}</span>
        </template>
        <span v-if="oldInUse(role)" class="inuse-old">{{ role.notice ?? 'The cutter in use is an old Detector: choose a SignNet.' }}</span>
        <span v-if="carLines[role.role]" class="inuse-car" :class="carLines[role.role]!.tone">{{ carLines[role.role]!.text }}<span
          v-if="carLines[role.role]!.more" class="inuse-car-more">{{ carLines[role.role]!.more }}</span></span>
        <span class="inuse-more">{{ (recent[role.role] ?? []).length }} {{ (recent[role.role] ?? []).length === 1 ? 'version' : 'versions' }} ›</span>
      </button>
    </div>

    <section v-if="bigCarJob" class="card car-job" id="car-job" :aria-label="bigCarJob.label">
      <JobProgress :job="bigCarJob" :heading="carHeading(bigCarJob)"
                   @details="emit('details', $event)" @stop="emit('stop', $event)"/>
    </section>

    <!-- The networks tried on the comma now, and the way back to the build's. -->
    <section v-if="trial" class="card trial-card" id="comma-trial" aria-label="Trying on the comma">
      <p class="trial-text"><strong>Trying on the comma: SignNet {{ trial.signnet.run }}</strong><br><span>{{ trialText }}.
        Until the comma's next software update.</span></p>
      <button type="button" class="link-button" :disabled="restoreBlocked !== ''" :title="restoreBlocked"
              id="comma-restore" @click="restore">Back to the build’s networks</button>
    </section>

    <!-- The versions of one network, to compare and to choose. -->
    <VersionsCard v-if="pickedRole" :key="pickedRole.role" :role="pickedRole" :recent="recent[pickedRole.role] ?? pickedRole.versions"
                  :payload="payload" :read-only="readOnly" :busy="busy" :car="car" :trial="trial"
                  :open-id="linkedVersion" @use="choose" @try="tryOnComma">
      <Segmented v-model="picked" :options="roleOptions" label="Network" class="versions-seg"/>
      <template #foot>
        <p v-if="pickedRole.role === 'cutter' && carJob && !bigCarJob" class="last-line" id="car-last">{{ CAR_JOB_TEXT[carJob.action] }}:
          {{ carJob.run ? 'SignNet ' + carJob.run + ', ' : '' }}{{ startedWhen(carJob) }} ·
          {{ carJob.state === 'interrupted' ? 'stopped' : JOB_STATE[carJob.state][0] }} ·
          <button type="button" class="link-button" @click="emit('details', carJob)">Details</button></p>
        <p v-if="pickedRole.role === 'cutter'" class="note models-foot">As the car reads: a SignNet’s speed signs among its
          {{ payload.box_budget ?? 5 }} best boxes of each frame of score 0.3 or more, those the comma keeps (best 5 at 0.3). Speed signs and
          Every sign: found at score {{ payload.cutter_score_threshold }}, however many boxes it passes on (an old Detector a couple of hundred a
          frame, SignNet a few). Best {{ payload.box_budget ?? 5 }}, any score: the speed signs among its {{ payload.box_budget ?? 5 }} best boxes
          of each frame, whatever their score.</p>
        <p class="note models-foot">The numbers compare only on the same test: the older versions were measured on fewer signs (the count
          under each number). The choice is kept in <code>{{ payload.active_file }}</code> of the data root; it copies, exports and
          deploys nothing: nothing on the comma changes. Only Publish, on the Publish tab, changes what the comma
          downloads; Try on the comma now copies a SignNet onto it until its next software update.<template v-if="carText"> In the branch: the networks published in {{ carText }}.</template></p>
        <p v-if="payload.on_car_problem" class="note">What is on the car cannot be shown: {{ payload.on_car_problem }}.</p>
      </template>
    </VersionsCard>

    <!-- Review helpers: the networks that help only the review work, each with its training (#91). -->
    <section class="live-section" id="review-helpers" aria-labelledby="review-helpers-title">
      <h2 class="section-title" id="review-helpers-title">Review helpers</h2>
      <!-- Applicability: proposes OURS or another in Sign review; its versions folded under it. -->
      <ApplicabilityCard :status="status" :role="deciderRole" :versions-of="versionsOf" :car-line="carLines.decider ?? null"
                         :versions-open="fromHash === 'decider'"
                         @run="(...args) => emit('run', ...args)" @stop="emit('stop', $event)" @details="emit('details', $event)">
        <VersionsCard v-if="deciderRole" bare :role="deciderRole" :recent="recent.decider ?? deciderRole.versions" :payload="payload"
                      :read-only="readOnly" :busy="busy" :car="car" :trial="trial"
                      :open-id="fromHash === 'decider' ? fromHashVersion : ''" @use="choose" @try="tryOnComma"/>
      </ApplicabilityCard>
      <!-- The road labeler: an offline network that prepares Speed review, never on the car. -->
      <RoadLabelerCard :status="status" @run="(...args) => emit('run', ...args)" @stop="emit('stop', $event)"
                       @details="emit('details', $event)"/>
      <!-- Fit box: key C of the review page, trained again on a V100 on the boxes people drew. -->
      <BoxFitCard :status="status" @run="(...args) => emit('run', ...args)" @stop="emit('stop', $event)"
                  @details="emit('details', $event)"/>
    </section>

    <p v-if="error" class="note">Not updated: {{ error }}</p>
  </template>
</template>
