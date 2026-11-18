<script setup lang="ts">
// The versions of one network on Models, to compare and to choose: the numbers that judge its role in columns, the one in
// use marked; only the recent versions (ModelsTab's recentOf), the older ones behind “Show older versions” (nothing is
// deleted); a row opens its whole card (ModelItem: details, Use this one, the run). Drawn twice by ModelsTab: at the top for
// the network picked in the grid (its switch in the default slot), and in Review helpers for applicability, which helps
// only Sign review (bare: no card of its own, no role text, which is its card title's tooltip; How it works stays).
// openId: a version to open and scroll to once drawn (the address /models#role-<role>:<id>). The lines under the table
// that belong to one role come in the foot slot.
import {computed, nextTick, ref, useTemplateRef, watch} from 'vue';
import {isoWhen} from '../common/format';
import type {CommaTrial, GitHubFileState, ModelVersion, ModelsPayload, RoleModels} from '../common/panel-types';
import {pct} from './evaluate';
import ModelItem from './ModelItem.vue';
import ModelFlow from './ModelFlow.vue';
import {ROLE_TEXT, inUseOf, metricsOf, type CarState} from './models';

const props = defineProps<{role: RoleModels; recent: ModelVersion[]; payload: ModelsPayload; readOnly: boolean;
  busy: string | null; car: CarState; trial: CommaTrial | null; openId?: string; bare?: boolean;
  /** Whether GitHub has each version a save copies (#92): marked on the row in use, and on any other not saved. */
  github?: Record<string, GitHubFileState> | null}>();
const emit = defineEmits<{use: [version: ModelVersion]; try: [version: ModelVersion]; replay: [version: ModelVersion]}>();
const root = useTemplateRef<HTMLElement>('root');

/** The older versions shown too. */
const older = ref(false);
const listed = computed((): ModelVersion[] => older.value ? props.role.versions : props.recent);
const olderCount = computed(() => props.role.versions.length - props.recent.length);
/** The version drawn under the network: the one in use, else the newest. */
const drawn = computed(() => inUseOf(props.role) ?? props.role.versions[0] ?? null);

/** A SignNet in a few words, to tell its versions apart in the table: "grid 8 px", "grid 4 px · 25 % of the drives". */
function signnetText(version: ModelVersion): string {
  if (version.role !== 'cutter' || version.kind !== 'signnet') return '';
  return [version.stride ? 'grid ' + version.stride + ' px' : '',
    version.train_share ? Math.round(version.train_share * 100) + ' % of the drives' : ''].filter(Boolean).join(' · ');
}
/** The columns of the table: the numbers of the role's newest measured version (they are the same for every version
 * of a role, a cutter of the speed signs only has “–” under every sign). */
const columns = computed(() => {
  let widest: ReturnType<typeof metricsOf> = [];
  for (const version of listed.value) {
    const metrics = metricsOf(version);
    if (metrics.length > widest.length) widest = metrics;
  }
  return widest.map((metric) => ({key: metric.key, label: metric.short ?? metric.label.replace(/^With the cutter.*/, 'With the cutter')}));
});
/** The buttons on the row itself (#116): Use this one (off on the one in use), and for a brain Replay, without opening it. */
const canUse = (version: ModelVersion): boolean => !version.in_use
  && (version.selectable ?? (version.finished && !(version.role === 'cutter' && version.kind === 'signnet')));
const replayable = (version: ModelVersion): boolean => version.role === 'brain' && version.finished && !props.readOnly
  && props.car?.replayState != null;
const replaying = (version: ModelVersion): boolean => props.car?.replaying != null && props.car.replaying === version.run;
function replayBlocked(): string {
  if (props.car?.replaying != null) return 'A brain is being replayed now.';
  const state = props.car?.replayState;
  if (state && !state.allowed) return state.reason.charAt(0).toUpperCase() + state.reason.slice(1) + '.';
  return '';
}
const hasReplay = (version: ModelVersion): boolean => version.role === 'brain' && version.replay != null;
const cell = (version: ModelVersion, key: string) => metricsOf(version).find((metric) => metric.key === key) ?? null;
/** The GitHub mark of a row: "on GitHub" on the one in use, "not on GitHub" or "changed since the GitHub save" on any. */
function githubMark(version: ModelVersion): {text: string; tone: string} | null {
  const state = props.github?.[version.id];
  if (!state || (state === 'saved' && !version.in_use)) return null;
  return state === 'saved' ? {text: 'on GitHub', tone: 'vt-use'}
    : {text: state === 'changed' ? 'changed since the GitHub save' : 'not on GitHub', tone: 'vt-wait'};
}

/** The version the address names, opened and scrolled to once (shown among the older ones when it is one of them). */
let opened = '';
watch(() => [props.openId, props.role.versions.length] as const, ([id]) => {
  if (!id || opened === id || !props.role.versions.some((version) => version.id === id)) return;
  opened = id;
  if (!props.recent.some((version) => version.id === id)) older.value = true;
  void nextTick(() => {
    const row = root.value?.querySelector<HTMLDetailsElement>('details[data-model="' + CSS.escape(id) + '"]');
    if (!row) return;
    row.open = true;
    row.scrollIntoView({behavior: 'smooth', block: 'center'});
  });
}, {immediate: true});
</script>

<template>
  <section ref="root" class="versions-card" :class="{card: !bare}" :data-role="role.role" aria-label="Versions">
    <slot/>
    <p v-if="!bare" class="role-what">{{ ROLE_TEXT[role.role].what }}</p>
    <template v-if="drawn">
      <h3 class="flow-head">How it works{{ drawn.in_use ? '' : ' (no version in use: the newest)' }}</h3>
      <ModelFlow :version="drawn"/>
    </template>
    <p v-if="role.versions.length === 0" class="role-what muted">{{ ROLE_TEXT[role.role].empty }}
      <a v-if="role.role !== 'decider'" href="/train">Go to Train ›</a></p>
    <div v-else class="vt" :style="{'--vt-cols': columns.length}">
      <div class="vt-head" aria-hidden="true"><span>Version</span><span v-for="column in columns" :key="column.key"
        :class="'vt-c-' + column.key">{{ column.label }}</span><span></span></div>
      <details v-for="version in listed" :key="version.id" class="vt-row" :class="{'in-use': version.in_use}"
               :data-model="version.id">
        <summary>
          <span class="vt-first"><span class="vt-name"><b>{{ version.label }}</b><small>{{ version.trained_at ? isoWhen(version.trained_at) : version.run }}<template
            v-if="signnetText(version)"> · {{ signnetText(version) }}</template><template
            v-if="version.in_use"> · <em class="vt-use">in use</em></template><template v-else-if="!version.finished"> · <em
            class="vt-wait">not finished</em></template><template v-if="version.on_car"> · <em class="vt-car">on the car</em></template><template
            v-if="githubMark(version)"> · <em :class="githubMark(version)!.tone">{{ githubMark(version)!.text }}</em></template><template
            v-if="trial && version.role === 'cutter' && trial.signnet.run === version.run"> · <em class="vt-trial">trying on the comma</em></template></small></span>
          <span v-if="((canUse(version) || version.in_use) && !readOnly) || replayable(version)" class="vt-acts">
            <button v-if="replayable(version)" class="btn small" type="button" :disabled="replayBlocked() !== ''"
                    :title="replayBlocked() || 'Runs it again on its validation and test drives, without training: its error by situation, ramps and merges, the time after an end, how much the sign moves it.'"
                    :aria-label="'Replay ' + version.run" :data-replay="version.run"
                    @click.prevent.stop="emit('replay', version)">{{ replaying(version) ? 'Replaying…' : hasReplay(version) ? 'Replay again' : 'Replay' }}</button>
            <button v-if="(canUse(version) || version.in_use) && !readOnly" class="btn small" type="button"
                    :disabled="version.in_use || busy !== null" :title="version.in_use ? 'This one is in use.' : ROLE_TEXT[version.role].uses" :aria-label="'Use ' + version.run + ' · ' + version.label"
                    @click.prevent.stop="emit('use', version)">{{ busy === version.id ? 'Saving…' : 'Use this one' }}</button>
          </span></span>
          <span v-for="column in columns" :key="column.key" class="vt-num" :class="['vt-c-' + column.key, {none: cell(version, column.key)?.value == null}]"
                :title="cell(version, column.key)?.detail ?? ''">
            <b>{{ cell(version, column.key)?.value == null ? '–' : pct(cell(version, column.key)?.value) }}</b>
            <span class="vt-bar" aria-hidden="true"><span :style="{width: Math.min(1, cell(version, column.key)?.value ?? 0) * 100 + '%'}"></span></span>
            <small>{{ cell(version, column.key)?.count ?? '' }}</small>
          </span>
          <span class="vt-chev" aria-hidden="true">›</span>
        </summary>
        <ul class="model-list">
          <ModelItem :version="version" :role-label="role.label" :score="payload.cutter_score_threshold"
                     :read-only="readOnly" :busy="busy" :car="car" @use="emit('use', $event)" @try="emit('try', $event)"
                     @replay="emit('replay', $event)"/>
        </ul>
      </details>
      <button v-if="olderCount > 0" type="button" class="link-button vt-older" :aria-expanded="older"
              @click="older = !older">{{ older ? 'Hide the older versions'
                : 'Show ' + olderCount + ' older ' + (olderCount === 1 ? 'version' : 'versions') }}</button>
    </div>
    <slot name="foot"/>
  </section>
</template>
