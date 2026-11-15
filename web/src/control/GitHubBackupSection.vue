<script setup lang="ts">
// Publish › GitHub, a private backup (#92): every network in use and on the car, the sign networks each saved brain
// read its signs with, the road labeler and Fit box, saved in the private repository cristianku/smartspeed-networks
// (scripts/save_networks_github.sh), and put back from there (scripts/restore_networks_github.sh) after a lost data
// root or on a new server: Models then shows the networks in use and on the car of the last save. Save adds only what
// changed and never drops a version saved before; Restore never overwrites a file of the data root. Whether GitHub is
// up to date comes from the status (networks.github.freshness): every version and record a save would copy, saved as it
// is here, changed since, or not on GitHub; a server of before it compares the ids in use and on the car only.
import {computed} from 'vue';
import {isoWhen} from '../common/format';
import type {ActionName, ActionOptions, ActionState, Job, PanelStatus} from '../common/panel-types';
import JobProgress from './JobProgress.vue';
import {JOB_STATE, jobOf, startedWhen} from './jobs';

const props = defineProps<{status: PanelStatus | null}>();
const emit = defineEmits<{run: [name: ActionName, state: ActionState, start?: undefined, options?: ActionOptions];
  stop: [job: Job]; details: [job: Job]}>();

const ROLES = ['cutter', 'classifier', 'decider', 'brain'] as const;
/** A job that ended well, in words. */
const DONE: Partial<Record<ActionName, string>> = {networks_github_save: 'Saved on GitHub',
  networks_github_restore: 'Restored from GitHub'};

const github = computed(() => props.status?.networks?.github ?? null);
const job = computed(() => jobOf(props.status, ['networks_github_save', 'networks_github_restore']));
/** Running: in place of the buttons; one ended (done, failed or stopped) is a line under them, with its Details. */
const bigJob = computed(() => job.value?.state === 'running' ? job.value : null);
const saveState = computed(() => props.status?.actions.networks_github_save ?? null);
const restoreState = computed(() => props.status?.actions.networks_github_restore ?? null);
const saved = computed(() => github.value?.saved ?? null);
const restored = computed(() => github.value?.restored ?? null);

/** The networks in use and on the car now (models/active.json, on-car.json), by role. */
const inUse = computed(() => props.status?.car?.in_use ?? null);
const onCar = computed(() => props.status?.car?.on_car ?? null);
const freshness = computed(() => github.value?.freshness ?? null);
/** The save GitHub is compared with: the last one from here, or the one a restore put back (no saved.json then). */
const lastAt = computed(() => saved.value?.at ?? freshness.value?.saved_at ?? null);
const commitUrl = computed(() => saved.value?.commit_url
  ?? (freshness.value?.commit && github.value ? github.value.url + '/commit/' + freshness.value.commit : null));
/** What a save leaves out, and why: now (the plan), else what the last save said. */
const warnings = computed(() => freshness.value?.warnings ?? saved.value?.warnings ?? []);
/** GitHub has everything a save would copy now; a server without freshness: the last save holds the networks in use
 * and on the car now. */
const current = computed(() => {
  if (freshness.value) return freshness.value.up_to_date === true;
  const last = saved.value;
  if (!last || !inUse.value) return false;
  return ROLES.every((role) => !inUse.value![role] || last.in_use?.[role] === inUse.value![role])
    && ROLES.every((role) => !onCar.value?.[role] || last.on_car?.[role] === onCar.value![role]);
});

/** "2026-10-25_1941-cutter" out of "cutter:2026-10-25_1941-cutter/detector-sizes/1344x760". */
function runOf(id: string | null | undefined): string {
  return id ? id.slice(id.indexOf(':') + 1).split('/')[0] ?? id : '–';
}

/** A version or a record not on GitHub as it is here, in words: "brain 2026-11-15_0805-brain-s0", "Fit box (the code)",
 * "models/active.json", with "(changed)" when GitHub has an older one. */
function unsavedName(item: string): string {
  const fresh = freshness.value;
  const state = fresh?.versions[item] ?? fresh?.records[item];
  const name = item === 'box_fit:code' ? 'Fit box (the code)' : item.startsWith('box_fit:') ? 'Fit box ' + runOf(item)
    : item.includes(':') ? item.slice(0, item.indexOf(':')).replace('decider', 'applicability').replace('_', ' ') + ' '
      + runOf(item) : item;
  return name + (state === 'changed' ? ' (changed)' : '');
}
const UNSAVED_LISTED = 6;
const unsavedLine = computed(() => {
  const fresh = freshness.value;
  if (!fresh || fresh.up_to_date !== false) return '';
  const names = fresh.unsaved.slice(0, UNSAVED_LISTED).map(unsavedName);
  const more = fresh.unsaved_count - names.length;
  return 'Not on GitHub: ' + names.join(', ') + (more > 0 ? ' and ' + more + ' more' : '');
});

function megabytes(bytes: number): string {
  return bytes >= 1e9 ? (bytes / 1e9).toFixed(1) + ' GB' : Math.max(1, Math.round(bytes / 1e6)) + ' MB';
}

/** "12 files new, 3 changed" or "nothing new". */
function whatChanged(last: NonNullable<typeof saved.value>): string {
  if (!last.pushed) return 'nothing new';
  return [last.new ? last.new + ' files new' : '', last.updated ? last.updated + ' changed' : ''].filter(Boolean).join(', ')
    || 'the records';
}

/** "4 files put back, 120 already here, 2 here and different (kept)". */
function whatRestored(last: NonNullable<typeof restored.value>): string {
  return last.copied + ' files put back, ' + last.same + ' already here'
    + (last.kept_count ? ', ' + last.kept_count + ' here and different (kept as they are)' : '');
}

function save(): void {
  if (saveState.value) emit('run', 'networks_github_save', saveState.value);
}

function restore(): void {
  if (restoreState.value) emit('run', 'networks_github_restore', restoreState.value);
}
</script>

<template>
  <section v-if="github" class="card publish" id="github" aria-labelledby="github-title">
    <h2 id="github-title">GitHub <small class="h2-aside">private backup</small></h2>
    <p class="note">The networks in use and on the car, the sign networks of each brain, the road labeler and Fit box, in
      the private repository <a :href="github.url" target="_blank" rel="noopener">{{ github.repo }}</a>. Restore puts them
      back after a lost data root, or on a new server: Models then shows them as at the last save. A save adds only what
      changed and keeps every version saved before.</p>
    <p v-if="!github.token" class="hf-line" id="github-token">This server has no GitHub token: a save works only once the
      repository exists, and cannot create it.</p>
    <div class="train-slot">
      <JobProgress v-if="bigJob" :job="bigJob" @details="emit('details', $event)" @stop="emit('stop', $event)"/>
      <template v-else>
        <div class="package-row" id="github-row">
          <div class="latest-what">
            <template v-if="saved || freshness?.commit">
              <span class="latest-kicker" id="github-fresh" :class="{'pub-done': current, 'pub-warn': !current}">{{ current
                ? (freshness ? '✓ GitHub is up to date' : '✓ The networks in use and on the car are on GitHub')
                : freshness ? 'GitHub is not up to date: ' + freshness.unsaved_count + ' not saved'
                  : 'Saved before the networks in use or on the car changed' }}<template
                  v-if="lastAt"> · {{ saved ? 'last save' : 'restored from the save of' }} {{ isoWhen(lastAt) }}</template></span>
              <span v-if="unsavedLine" class="latest-text whole" id="github-unsaved" :title="freshness?.unsaved.map(unsavedName).join('\n')">{{ unsavedLine }}</span>
              <template v-if="saved">
                <span class="latest-text">{{ ROLES.map((role) => role + ' ' + runOf(saved!.in_use?.[role])).join(' · ') }}</span>
                <span class="latest-text whole">{{ saved.versions }} versions, {{ saved.files }} files, {{ megabytes(saved.bytes) }}
                  · last save: {{ whatChanged(saved) }}</span>
              </template>
            </template>
            <template v-else>
              <span class="latest-kicker">Not saved on GitHub yet</span>
              <span class="latest-text whole">The first save creates the private repository and pushes every network in
                use: a few minutes.</span>
            </template>
          </div>
          <button class="btn primary small" type="button" id="run-networks_github_save"
                  :disabled="!saveState?.allowed" :title="saveState?.allowed ? '' : saveState?.reason"
                  @click="save">Save on GitHub</button>
        </div>
        <p class="last-line" id="github-last"><template v-if="job"><span :class="{'pub-done': job.state === 'ok'}"><template
          v-if="job.state === 'ok'">✓ {{ DONE[job.action] ?? job.label }}, {{ job.ended_at ? isoWhen(job.ended_at) : startedWhen(job) }}</template><template
          v-else>Last: {{ job.label }}, {{ startedWhen(job) }} · {{ job.state === 'interrupted' ? 'stopped' : JOB_STATE[job.state][0] }}</template></span> ·
          <button type="button" class="link-button" @click="emit('details', job)">Details</button> · </template><template
          v-if="commitUrl"><a :href="commitUrl" target="_blank" rel="noopener">Open the commit</a> · </template><button
          type="button" class="link-button" id="run-networks_github_restore" :disabled="!restoreState?.allowed"
          :title="restoreState?.allowed ? 'Puts back every saved file missing here; never overwrites one' : restoreState?.reason"
          @click="restore">Restore from GitHub</button></p>
        <p v-if="restored" class="last-line" id="github-restored">Last restore {{ isoWhen(restored.at) }}:
          {{ whatRestored(restored) }}.</p>
        <p v-if="warnings.length" class="last-line pub-warn" id="github-warnings" :title="warnings.join('\n')">Left out of
          a save: {{ warnings.slice(0, 3).join(' · ') }}{{ warnings.length > 3 ? ' and ' + (warnings.length - 3) + ' more' : '' }}</p>
      </template>
    </div>
  </section>
</template>
