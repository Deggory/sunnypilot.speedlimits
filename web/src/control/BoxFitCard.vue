<script setup lang="ts">
// Fit box on Models: the review page's key C, which fits a rough box to its sign on this server's CPU
// (nnslr_tools.box_fit): two rankers (colour trees and edge networks), then the box refiner, a small network, on the
// small boxes. Every training is a version of its own (#94, nnslr_tools.box_fit.versions): its run folder, its three
// files and its measure on the drives left out, the new version against the one in use then, each as the review page
// runs it, on the same boxes. Nothing is replaced by a training: a person chooses the version in use here ("Use this
// one", POST /api/panel/box-fit/use), and the review page fits with it from its next fit, without a restart. The code's
// files are a version too, in use until someone chooses.
// Train again on the boxes people drew (scripts/train_box_fit.sh, the panel action "box_fit_train"), the refiner too or
// not (without it the new version carries the refiner in use), a dry run first if wanted (it only counts the boxes); its
// job while it runs. Trained on a V100 (#65); its dry run takes no card. Beside each version, whether GitHub has it as it
// is here (#92, the status's networks.github.freshness). Hidden on a server without the action, and the versions left
// out when the endpoint is not there yet.
import {computed, onMounted, ref, shallowRef, watch} from 'vue';
import {api, messageOf, post} from '../common/api';
import {isoWhen, plural, sentence} from '../common/format';
import type {ActionName, ActionOptions, ActionState, BoxFitMeasure, BoxFitPayload, BoxFitVersion, GitHubFileState, Job,
  PanelStatus} from '../common/panel-types';
import Icon from '../components/Icon.vue';
import {askConfirm} from '../composables/confirm';
import {toast} from '../composables/toast';
import JobProgress from './JobProgress.vue';
import {JOB_STATE, jobOf, startedWhen, terminalActivities} from './jobs';
import TerminalActivity from './TerminalActivity.vue';

const props = defineProps<{status: PanelStatus | null}>();
const emit = defineEmits<{run: [name: ActionName, state: ActionState, start?: undefined, options?: ActionOptions];
  stop: [job: Job]; details: [job: Job]}>();

const state = computed((): ActionState | null => props.status?.actions?.box_fit_train ?? null);
/** A server from before the action: no card. */
const shown = computed(() => state.value !== null);
const job = computed(() => jobOf(props.status, ['box_fit_train']));
const bigJob = computed(() => job.value && (job.value.state === 'running' || job.value.state === 'failed') ? job.value : null);
const activities = computed(() => terminalActivities(props.status, ['box_fit']));
const idle = computed(() => !bigJob.value && !activities.value.length);
const readOnly = computed(() => props.status !== null && !props.status.actions_enabled);

const payload = shallowRef<BoxFitPayload | null>(null);
async function load(): Promise<void> {
  try {
    const data = await api<BoxFitPayload>('/api/panel/box-fit');
    // A server of before #94 answers the old summaries: no versions to show until its restart.
    payload.value = Array.isArray(data?.versions) ? data : null;
  } catch {
    // Not on this server yet (it comes with a restart), or not readable now: the card shows no versions.
    payload.value = null;
  }
}
onMounted(() => void load());
const working = computed(() => job.value?.state === 'running' || activities.value.length > 0);
watch(working, (now, before) => {
  if (before && !now) void load();
});

/** The rankers only, without the hour of the refiner: the new version carries the refiner in use. */
const withoutRefiner = ref(false);
function start(dry: boolean): void {
  if (!state.value) return;
  const options: ActionOptions = {...(dry ? {dry: 'on'} : {}), ...(withoutRefiner.value ? {refiner: 'off'} : {})};
  emit('run', 'box_fit_train', state.value, undefined, Object.keys(options).length ? options : undefined);
}

const versions = computed((): BoxFitVersion[] => payload.value?.versions ?? []);
const used = computed(() => versions.value.find((version) => version.in_use) ?? null);
/** The newest versions, the older ones behind "Show older versions" (nothing is deleted). */
const RECENT = 4;
const older = ref(false);
const listed = computed(() => {
  if (older.value || versions.value.length <= RECENT + 1) return versions.value;
  const recent = versions.value.slice(0, RECENT);
  // The one in use and the code's files always shown.
  for (const version of versions.value.slice(RECENT)) {
    if (version.in_use || version.run === null) recent.push(version);
  }
  return recent;
});
const olderCount = computed(() => versions.value.length - listed.value.length);

/** A mean IoU as written in the records, 0 to 1. */
const iou = (value: number | null | undefined): string => typeof value === 'number' ? value.toFixed(3) : '–';
const nameOf = (version: BoxFitVersion): string => version.run === null ? 'The code’s files'
  : version.run.replace(/-box-fit$/, '').replace('_', ' ');
const nameOfId = (id: string | null): string => {
  if (!id) return '–';
  const version = versions.value.find((item) => item.id === id);
  if (version) return nameOf(version);
  return id === 'box_fit:code' ? 'The code’s files' : id.replace(/^box_fit:/, '').replace(/-box-fit$/, '').replace('_', ' ');
};
const REFINER: Record<BoxFitVersion['refiner'], string> = {trained: 'trained by it', carried: 'carried', none: 'none',
  present: 'there'};
function subOf(version: BoxFitVersion): string {
  if (version.run === null) return 'In the checkout, no record';
  return [isoWhen(version.at) || version.run, version.boxes !== null ? plural(version.boxes, 'box', 'boxes') : '',
    'refiner ' + REFINER[version.refiner]].filter(Boolean).join(' · ');
}
/** The columns: this version on the drives left out, the version in use then on the same boxes, and this version on
 * the small signs (under 20 px), where the refiner works. */
const COLUMNS = [
  {key: 'version', label: 'This version', title: 'Mean IoU on the drives left out, as the review page runs it'},
  {key: 'in_use', label: 'In use then', title: 'The version in use when it was trained, on the same boxes'},
  {key: 'small', label: 'Small signs', title: 'This version on the signs under 20 px'},
] as const;
function cell(version: BoxFitVersion, key: typeof COLUMNS[number]['key']): {value: number | null; count: string} {
  if (key === 'small') return {value: version.small?.version ?? null, count: countOf(version.small)};
  return {value: version.all?.[key] ?? null, count: key === 'version' ? countOf(version.all) : ''};
}
const countOf = (measure: BoxFitMeasure | null): string => typeof measure?.cases === 'number'
  ? plural(measure.cases, 'box', 'boxes') : '';
const measureText = (measure: BoxFitMeasure): string => 'rough ' + iou(measure.rough) + ' · its rankers '
  + iou(measure.rankers) + ' · this version ' + iou(measure.version) + ' · in use then ' + iou(measure.in_use);
/** Against the version in use when it was trained, in words. */
function verdictOf(version: BoxFitVersion): string {
  const all = version.all;
  if (!all || typeof all.version !== 'number') return '';
  if (typeof all.in_use !== 'number') return 'Not measured against the version in use then.';
  const word = all.version > all.in_use ? 'Better than' : all.version === all.in_use ? 'As good as' : 'Worse than';
  return word + ' ' + nameOfId(version.in_use_then) + ', in use when it was trained: ' + iou(all.version) + ' against '
    + iou(all.in_use) + ' on the same boxes.' + leakOf(version);
}
/** Which of the two had seen the drives of the measure in its training: its figure is a little in its favour. */
function leakOf(version: BoxFitVersion): string {
  const seen = [version.measured_clean?.version === false ? 'this version' : '',
    version.measured_clean?.in_use === false ? 'the version in use then' : ''].filter(Boolean);
  return seen.length ? ' ' + seen.join(' and ').replace(/^./, c => c.toUpperCase()) + (seen.length > 1 ? ' were' : ' was')
    + ' trained on these drives too: a little in ' + (seen.length > 1 ? 'their' : 'its') + ' favour.' : '';
}

/** Whether GitHub has each version as it is here (#92); absent on a server that does not tell. */
const github = computed(() => props.status?.networks?.github?.freshness?.versions ?? null);
const GITHUB: Record<GitHubFileState, [string, string]> = {saved: ['On GitHub', 'ok'],
  changed: ['Changed since the save', 'warn'], missing: ['Not on GitHub', 'warn']};
const githubOf = (version: BoxFitVersion): GitHubFileState | null => github.value?.[version.id] ?? null;

const busy = ref<string | null>(null);
function blockedWhy(version: BoxFitVersion): string {
  if (readOnly.value) return 'Read-only: the buttons are turned off on this server.';
  if (version.in_use) return '';
  if (!version.files.includes('colour.npz') || !version.files.includes('edges.npz')) return 'Its rankers are missing.';
  return '';
}
async function use(version: BoxFitVersion): Promise<void> {
  if (busy.value !== null || blockedWhy(version)) return;
  const ok = await askConfirm({
    title: 'Use this version of Fit box?', action: 'Use this one',
    text: nameOf(version) + ' fits the boxes of the review page (key C) from its next fit, without a restart. '
      + 'Nothing is copied or deleted: the version in use now stays here, to choose again.',
  });
  if (!ok) return;
  busy.value = version.id;
  try {
    await post('/api/panel/box-fit/use', {id: version.id});
    toast('Fit box in use: ' + nameOf(version));
    await load();
  } catch (problem) {
    toast(messageOf(problem), true);
  } finally {
    busy.value = null;
  }
}

/** What key C does, in the title's tooltip. */
const what = 'Key C of the review page fits a rough box to its sign on this server: two rankers (colour trees and edge '
  + 'networks), then the box refiner on the small boxes. Every training is a version of its own, measured against the '
  + 'one in use; nothing is replaced until you choose a version here. Trained on a V100.';
const usedLine = computed(() => {
  const version = used.value;
  if (!version) return '';
  const all = version.all;
  return 'In use: ' + nameOf(version) + (all && typeof all.version === 'number' ? ' · IoU ' + iou(all.version)
    + ' on drives left out' : '');
});
const newest = computed(() => versions.value[0]?.run ? versions.value[0] : null);
/** The newest version when it is not the one in use: its measure against it, to choose. */
const newestLine = computed(() => {
  const version = newest.value;
  if (!version || version.in_use) return '';
  return 'Newest: ' + nameOf(version) + (verdictOf(version) ? ' · ' + verdictOf(version) : '');
});
</script>

<template>
  <section v-if="shown" class="card import-card labeler-card" id="box-fit" aria-labelledby="bf-title">
    <h3 id="bf-title" class="rl-title" :title="what">Fit box</h3>
    <JobProgress v-if="bigJob" :job="bigJob" @details="emit('details', $event)" @stop="emit('stop', $event)"/>
    <TerminalActivity v-for="activity in activities" :key="activity.pid + activity.label" :activity="activity" :preparation="null"/>
    <div v-if="idle" class="import-row">
      <span class="import-icon" :class="{fresh: versions.length <= 1}" aria-hidden="true"><Icon name="train"/></span>
      <div class="import-what">
        <strong>{{ payload ? plural(versions.length, 'version') : 'Fit box' }}</strong>
        <span v-if="usedLine" id="box-fit-in-use">{{ usedLine }}</span>
        <span v-if="newestLine" id="box-fit-newest">{{ newestLine }}</span>
      </div>
      <div class="rl-buttons">
        <button type="button" class="btn" id="box-fit-dry" :disabled="!state?.allowed" title="Only counts the boxes"
                @click="start(true)">Dry run</button>
        <button type="button" class="btn primary" id="box-fit-train" :disabled="!state?.allowed"
                title="A new version, measured against the one in use; nothing is replaced"
                @click="start(false)">Train</button>
      </div>
    </div>
    <p v-if="idle && state && !state.allowed" class="import-why">{{ sentence(state.reason) }}</p>
    <p v-if="payload?.in_use.problem" class="import-why warn" id="box-fit-problem">{{ sentence(payload.in_use.problem) }}</p>
    <label v-if="idle" class="rl-marks" title="The new version carries the refiner in use"><input type="checkbox"
      id="box-fit-no-refiner" v-model="withoutRefiner"> Rankers only</label>
    <p v-if="job && !bigJob" class="import-last">Last run {{ startedWhen(job) }} · {{ job.state === 'interrupted' ? 'stopped' : JOB_STATE[job.state][0] }} ·
      <button type="button" class="link-button" @click="emit('details', job)">Details</button></p>

    <div v-if="versions.length" class="vt" id="box-fit-versions" :style="{'--vt-cols': COLUMNS.length}">
      <div class="vt-head" aria-hidden="true"><span>Version</span><span v-for="column in COLUMNS" :key="column.key"
        :title="column.title">{{ column.label }}</span><span></span></div>
      <details v-for="version in listed" :key="version.id" class="vt-row" :class="{'in-use': version.in_use}"
               :data-model="version.id">
        <summary>
          <span class="vt-name"><b>{{ nameOf(version) }}</b><small>{{ subOf(version) }}<template
            v-if="version.in_use"> · <em class="vt-use">in use</em></template><template
            v-if="githubOf(version) && githubOf(version) !== 'saved'"> · <em class="vt-wait">{{
              GITHUB[githubOf(version)!][0] }}</em></template></small></span>
          <span v-for="column in COLUMNS" :key="column.key" class="vt-num" :class="{none: cell(version, column.key).value === null}"
                :title="column.title">
            <b>{{ iou(cell(version, column.key).value) }}</b>
            <span class="vt-bar" aria-hidden="true"><span :style="{width: Math.min(1, cell(version, column.key).value ?? 0) * 100 + '%'}"></span></span>
            <small>{{ cell(version, column.key).count }}</small>
          </span>
          <span class="vt-chev" aria-hidden="true">›</span>
        </summary>
        <ul class="model-list">
          <li class="model" :class="{'in-use': version.in_use}">
            <div class="model-head">
              <div class="model-title">
                <span class="model-name" :title="version.id">{{ nameOf(version) }}</span>
                <span class="model-sub">{{ version.run ? 'runs/' + version.run : 'src/nnslr_tools/box_fit/' }} · {{ version.files.join(', ') }}</span>
              </div>
              <span class="model-pills">
                <span v-if="version.in_use" class="pill ok">In use</span>
                <span v-if="githubOf(version)" class="pill" :class="GITHUB[githubOf(version)!][1]">{{
                  GITHUB[githubOf(version)!][0] }}</span>
              </span>
            </div>
            <p v-if="verdictOf(version)" class="import-why">{{ verdictOf(version) }}</p>
            <dl class="model-facts">
              <div v-if="version.boxes !== null"><dt>Boxes</dt><dd>{{ plural(version.boxes, 'box', 'boxes') }}<template
                v-if="version.routes !== null"> from {{ plural(version.routes, 'drive') }}</template></dd></div>
              <div v-if="version.rankers_fit !== null"><dt title="On the groups of drives the rankers' training left out">Rankers</dt>
                <dd>{{ iou(version.rankers_rough) }} → {{ iou(version.rankers_fit) }}</dd></div>
              <div><dt>Refiner</dt><dd>{{ version.refiner === 'carried' ? 'carried from ' + nameOfId(version.refiner_from)
                : version.refiner === 'trained' ? 'trained' + (version.refiner_boxes !== null ? ' on ' + plural(version.refiner_boxes, 'box', 'boxes') : '')
                : version.refiner === 'present' ? 'refiner.onnx' : 'none' }}</dd></div>
              <div v-if="version.all"><dt>Drives left out</dt><dd>{{ measureText(version.all) }}</dd></div>
              <div v-if="version.small"><dt>Small signs</dt><dd>{{ measureText(version.small) }}</dd></div>
              <div v-if="version.fair"><dt title="The boxes Fit box did not place: a measure that does not favour the version in use">Not placed by Fit box</dt>
                <dd>{{ measureText(version.fair) }}</dd></div>
            </dl>
            <div class="model-actions">
              <span class="import-why">{{ blockedWhy(version) || (version.in_use ? 'The review page fits with this one.' : '') }}</span>
              <button v-if="!version.in_use" class="btn small" type="button" :disabled="blockedWhy(version) !== '' || busy !== null"
                      :data-use="version.id" @click="use(version)">{{ busy === version.id ? 'Saving…' : 'Use this one' }}</button>
            </div>
          </li>
        </ul>
      </details>
      <button v-if="olderCount > 0 || older" type="button" class="link-button vt-older" :aria-expanded="older"
              @click="older = !older">{{ older ? 'Hide the older versions'
                : 'Show ' + olderCount + ' older ' + (olderCount === 1 ? 'version' : 'versions') }}</button>
    </div>
  </section>
</template>
