<script setup lang="ts">
// Models: first the networks in use (the cutter, classifier and decider of the pipeline, the road type; docs/NEURAL_NETWORK.md), one
// card each with its numbers; then the versions of one of them (the switch, or a tap on its card) as a table to
// compare: the numbers that judge that role in columns, the one in use marked; a row opens its whole card (ModelItem:
// details, Use this one, the run). A role has no model in use until someone picks one: there is no default. The
// networks the comma downloads (models/on-car.json, written by nnslr on-car) are marked “on the car”. The cutter is
// SignNet, the same network the comma runs: one at the car's band can be put in use; the old Detectors are listed to
// compare only (an old one still in use says so on its card). A SignNet's card also has “Send to the car”
// (scripts/send_to_car.sh: it and the classifier in use published for the comma, the sunnypilot speed-limits
// branch moved, psa-torque-sunny-testing built), whose job shows here while it runs, and “Try on the comma now”
// (scripts/try_on_comma.sh: the same two networks copied straight onto the comma, checked first with its own code, until
// its next software update; models/on-comma-trial.json), with “Back to the build's networks”. The runs themselves are on
// Results (/evaluate); the package of the three in use is on Publish (/publish). Last, On the car: the settings of the
// comma's vision limit, sent with the networks (CarSettingsSection).
import {computed, nextTick, ref, watch} from 'vue';
import {isoWhen} from '../common/format';
import type {ActionName, ActionOptions, ActionState, Job, ModelRole, ModelVersion, PanelStatus, RoleModels} from '../common/panel-types';
import {askConfirm} from '../composables/confirm';
import {pct} from './evaluate';
import CarSettingsSection from './CarSettingsSection.vue';
import JobProgress from './JobProgress.vue';
import {JOB_STATE, jobOf, startedWhen} from './jobs';
import ModelItem from './ModelItem.vue';
import Segmented from './Segmented.vue';
import {ROLE_TEXT, chosenText, metricsOf, putInUse, restoreQuestion, sendQuestion, tryQuestion, useModels,
  type CarState} from './models';
import SectionHead from './SectionHead.vue';

const props = defineProps<{status: PanelStatus | null}>();
const emit = defineEmits<{run: [name: ActionName, state: ActionState, start?: undefined, options?: ActionOptions];
  stop: [job: Job]; details: [job: Job]}>();
const {payload, error, refresh} = useModels();
const roles = computed((): RoleModels[] => payload.value?.roles ?? []);
/** Read-only server: the choice is off, as the buttons of the other tabs. */
const readOnly = computed(() => (props.status !== null && !props.status.actions_enabled) || Boolean(payload.value?.active_problem));
/** The version being put in use, so its button waits for the answer. */
const busy = ref<string | null>(null);

const ROLES: readonly ModelRole[] = ['cutter', 'classifier', 'decider', 'road_type'];
/** The role whose versions are listed: the address's (/models#role-classifier, or /models#role-decider:<version id>,
 * Train's link to its latest version), else the cutter. */
const hashRole = location.hash.split(':')[0];
const fromHash = ROLES.find((role) => hashRole === '#role-' + role) ?? null;
/** The version the address names, opened and scrolled to once drawn, or ''. */
const fromHashVersion = fromHash && location.hash.includes(':') ? decodeURIComponent(location.hash.slice(location.hash.indexOf(':') + 1)) : '';
const picked = ref<ModelRole>(fromHash ?? 'cutter');
const pickedRole = computed(() => roles.value.find((role) => role.role === picked.value) ?? null);
/** A tap on a card in use: its versions, scrolled to (below the cards on a phone). */
function pick(role: ModelRole): void {
  picked.value = role;
  void nextTick(() => document.querySelector('.versions-card')?.scrollIntoView({behavior: 'smooth', block: 'nearest'}));
}
/** A version of the list opened and scrolled to (Train's link to the latest version). */
function openVersion(id: string): void {
  const row = document.querySelector<HTMLDetailsElement>('.versions-card details[data-model="' + CSS.escape(id) + '"]');
  if (row === null) return;
  row.open = true;
  row.scrollIntoView({behavior: 'smooth', block: 'center'});
}
// Arrived from a link to a role: its versions, once they are drawn (opened on the version named, if any).
let arrived = fromHash === null;
watch(payload, (value) => {
  if (arrived || value === null || fromHash === null) return;
  arrived = true;
  if (fromHashVersion) {
    picked.value = fromHash;
    void nextTick(() => openVersion(fromHashVersion));
  } else pick(fromHash);
}, {immediate: true});
const roleOptions = computed(() => roles.value.map((role) => ({key: role.role, label: role.label})));

/** The version in use of a role, or null (none chosen, or its files are gone). */
const inUseOf = (role: RoleModels): ModelVersion | null => role.versions.find((version) => version.in_use) ?? null;
/** The cutter in use is an old Detector, chosen before SignNet could be (the server says so; an older one: not a SignNet). */
function oldInUse(role: RoleModels): boolean {
  if (role.role !== 'cutter') return false;
  const used = inUseOf(role);
  return role.old_kind_in_use ?? (used?.role === 'cutter' && used.kind !== 'signnet');
}
/** The version of a role that runs on the comma, or null. */
const carOf = (role: RoleModels): ModelVersion | null => role.versions.find((version) => version.on_car) ?? null;
/** A SignNet in a few words, to tell its versions apart in the table: "grid 8 px", "grid 4 px · 25 % of the drives". */
function signnetText(version: ModelVersion): string {
  if (version.role !== 'cutter' || version.kind !== 'signnet') return '';
  return [version.stride ? 'grid ' + version.stride + ' px' : '',
    version.train_share ? Math.round(version.train_share * 100) + ' % of the drives' : ''].filter(Boolean).join(' · ');
}
/** Whether the comma runs the version in use of a role, in words. Two versions often have the same name: the time
 * of the one on the car tells them apart. The comma runs the cutter, the classifier and the road type (Send to the
 * car sends the SignNet chosen with the classifier and the road type in use); a trial replaces the first two until
 * its next software update; the decider runs only on this server. */
interface CarLine {tone: 'same' | 'other' | 'none'; text: string; more?: string}
function carLine(role: RoleModels): CarLine | null {
  if (payload.value?.on_car_problem) return null;
  if (role.role === 'decider') return {tone: 'none', text: 'On the car: not used there, only on this server'};
  const used = inUseOf(role);
  const car = carOf(role);
  const tried = role.role === 'cutter' ? trial.value?.signnet.run : role.role === 'classifier' ? trial.value?.classifier.run : null;
  if (tried) return used?.run === tried ? {tone: 'same', text: 'On the comma: this one, as a trial'}
    : {tone: 'other', text: 'On the comma: a trial of ' + tried};
  const more = role.role === 'cutter' ? 'Send to the car, on its version below, puts this one there.'
    : 'It goes with the next Send to the car.';
  if (car === null) return used ? {tone: 'other', text: 'On the car: none yet', more} : {tone: 'none', text: 'On the car: none'};
  const when = car.trained_at ? isoWhen(car.trained_at) : car.run;
  if (used === null) return {tone: 'none', text: 'On the car: ' + car.label + ', ' + when};
  if (car.id === used.id) return payload.value?.car_build_pending
    ? {tone: 'other', text: 'On the car: this one, once its build is finished'} : {tone: 'same', text: 'On the car: this one'};
  const age = car.trained_at && used.trained_at ? (Date.parse(car.trained_at) < Date.parse(used.trained_at) ? 'an older one' : 'a newer one')
    : 'another one';
  return {tone: 'other', text: 'On the car: ' + age + ', ' + when, more};
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

/** The columns of the table: the numbers of the role's newest measured version (they are the same for every version
 * of a role, a cutter of the speed signs only has “–” under every sign). */
const columns = computed(() => {
  const versions = pickedRole.value?.versions ?? [];
  let widest: ReturnType<typeof metricsOf> = [];
  for (const version of versions) {
    const metrics = metricsOf(version);
    if (metrics.length > widest.length) widest = metrics;
  }
  return widest.map((metric) => ({key: metric.key, label: metric.short ?? metric.label.replace(/^With the cutter.*/, 'With the cutter')}));
});
function cell(version: ModelVersion, key: string) {
  return metricsOf(version).find((metric) => metric.key === key) ?? null;
}

/** The last Send to the car, trial on the comma or return to the build's networks; running or failed, it is shown in
 * full above the versions. */
const carJob = computed(() => jobOf(props.status, ['send_to_car', 'try_on_comma', 'comma_restore']));
const bigCarJob = computed(() => carJob.value && (carJob.value.state === 'running' || carJob.value.state === 'failed') ? carJob.value : null);
const CAR_JOB_TEXT: Partial<Record<ActionName, string>> = {send_to_car: 'Last send to the car',
  try_on_comma: 'Last trial on the comma', comma_restore: 'Last return to the build’s networks'};
const carHeading = (job: Job): string | undefined => !job.run ? undefined
  : 'SignNet ' + job.run + (job.action === 'try_on_comma' ? ' on the comma' : '');
const runningOf = (action: ActionName): string | null =>
  carJob.value?.state === 'running' && carJob.value.action === action ? carJob.value.run ?? '' : null;
/** The networks being tried on the comma now (models/on-comma-trial.json), null when none. */
const trial = computed(() => payload.value?.comma_trial ?? null);
const car = computed((): CarState => ({
  state: props.status?.actions.send_to_car ?? null,
  sending: runningOf('send_to_car'),
  pending: Boolean(payload.value?.car_build_pending),
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
  if (carJob.value?.state === 'running') return 'Wait for the job running now.';
  const state = restoreState.value;
  if (state === null) return 'Not available on this server.';
  return state.allowed ? '' : state.reason.charAt(0).toUpperCase() + state.reason.slice(1) + '.';
});
/** Asks, naming the SignNet, the classifier and the road type in use, then starts the job. */
async function send(version: ModelVersion): Promise<void> {
  const state = car.value.state;
  if (state === null || !state.allowed) return;
  const classifiers = roles.value.find((role) => role.role === 'classifier');
  const roadTypes = roles.value.find((role) => role.role === 'road_type');
  if (!await askConfirm(sendQuestion(version, classifiers ? inUseOf(classifiers) : null,
                                     roadTypes ? inUseOf(roadTypes) : null))) return;
  emit('run', 'send_to_car', state, undefined, {signnet: version.run});
}
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
    if (await putInUse(version, pickedRole.value?.label ?? version.role)) await refresh();
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
      <button v-for="role in roles" :key="role.role" type="button" class="inuse-card" :class="{picked: role.role === picked}"
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
        <span class="inuse-more">{{ role.versions.length }} {{ role.versions.length === 1 ? 'version' : 'versions' }} ›</span>
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
    <section v-if="pickedRole" class="card versions-card" :data-role="pickedRole.role" aria-label="Versions">
      <Segmented v-model="picked" :options="roleOptions" label="Network" class="versions-seg"/>
      <p class="role-what">{{ ROLE_TEXT[pickedRole.role].what }}</p>
      <p v-if="pickedRole.versions.length === 0" class="role-what muted">{{ ROLE_TEXT[pickedRole.role].empty }}
        <a v-if="pickedRole.role !== 'decider'" href="/train">Go to Train ›</a></p>
      <div v-else class="vt" :style="{'--vt-cols': columns.length}">
        <div class="vt-head" aria-hidden="true"><span>Version</span><span v-for="column in columns" :key="column.key"
          :class="'vt-c-' + column.key">{{ column.label }}</span><span></span></div>
        <details v-for="version in pickedRole.versions" :key="version.id" class="vt-row" :class="{'in-use': version.in_use}"
                 :data-model="version.id">
          <summary>
            <span class="vt-name"><b>{{ version.label }}</b><small>{{ version.trained_at ? isoWhen(version.trained_at) : version.run }}<template
              v-if="signnetText(version)"> · {{ signnetText(version) }}</template><template
              v-if="version.in_use"> · <em class="vt-use">in use</em></template><template v-else-if="!version.finished"> · <em
              class="vt-wait">not finished</em></template><template v-if="version.on_car"> · <em class="vt-car">on the car</em></template><template
              v-if="trial && version.role === 'cutter' && trial.signnet.run === version.run"> · <em class="vt-trial">trying on the comma</em></template></small></span>
            <span v-for="column in columns" :key="column.key" class="vt-num" :class="['vt-c-' + column.key, {none: cell(version, column.key)?.value == null}]"
                  :title="cell(version, column.key)?.detail ?? ''">
              <b>{{ cell(version, column.key)?.value == null ? '–' : pct(cell(version, column.key)?.value) }}</b>
              <span class="vt-bar" aria-hidden="true"><span :style="{width: Math.min(1, cell(version, column.key)?.value ?? 0) * 100 + '%'}"></span></span>
              <small>{{ cell(version, column.key)?.count ?? '' }}</small>
            </span>
            <span class="vt-chev" aria-hidden="true">›</span>
          </summary>
          <ul class="model-list">
            <ModelItem :version="version" :role-label="pickedRole.label" :score="payload.cutter_score_threshold"
                       :read-only="readOnly" :busy="busy" :car="car" @use="choose" @send="send" @try="tryOnComma"/>
          </ul>
        </details>
      </div>
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
        deploys nothing: nothing on the comma changes. Only Send to the car, on a SignNet, changes what the comma
        downloads; Try on the comma now copies a SignNet onto it until its next software update.<template v-if="carText"> On the car: the networks published in {{ carText }}.</template></p>
      <p v-if="payload.on_car_problem" class="note">What is on the car cannot be shown: {{ payload.on_car_problem }}.</p>
    </section>

    <!-- The settings of the comma's vision limit, sent with the networks. -->
    <CarSettingsSection :read-only="readOnly"/>
    <p v-if="error" class="note">Not updated: {{ error }}</p>
  </template>
</template>
