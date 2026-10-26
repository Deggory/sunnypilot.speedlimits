<script setup lang="ts">
// Models: first the three networks of the pipeline in use (cutter, classifier, decider; docs/NEURAL_NETWORK.md), one
// card each with its numbers; then the versions of one of them (the switch, or a tap on its card) as a table to
// compare: the numbers that judge that role in columns, the one in use marked; a row opens its whole card (ModelItem:
// details, Use this one, the run). A role has no model in use until someone picks one: there is no default. The runs
// themselves are on Results (/evaluate); the package of the three in use is on Publish (/publish).
import {computed, nextTick, ref, watch} from 'vue';
import {isoWhen} from '../common/format';
import type {ModelRole, ModelVersion, PanelStatus, RoleModels} from '../common/panel-types';
import {pct} from './evaluate';
import ModelItem from './ModelItem.vue';
import Segmented from './Segmented.vue';
import {ROLE_TEXT, chosenText, metricsOf, putInUse, useModels} from './models';
import SectionHead from './SectionHead.vue';

const props = defineProps<{status: PanelStatus | null}>();
const {payload, error, refresh} = useModels();
const roles = computed((): RoleModels[] => payload.value?.roles ?? []);
/** Read-only server: the choice is off, as the buttons of the other tabs. */
const readOnly = computed(() => (props.status !== null && !props.status.actions_enabled) || Boolean(payload.value?.active_problem));
/** The version being put in use, so its button waits for the answer. */
const busy = ref<string | null>(null);

const ROLES: readonly ModelRole[] = ['cutter', 'classifier', 'decider'];
/** The role whose versions are listed: the address's (/models#role-classifier, Train's link), else the cutter. */
const fromHash = ROLES.find((role) => location.hash === '#role-' + role) ?? null;
const picked = ref<ModelRole>(fromHash ?? 'cutter');
const pickedRole = computed(() => roles.value.find((role) => role.role === picked.value) ?? null);
/** A tap on a card in use: its versions, scrolled to (below the cards on a phone). */
function pick(role: ModelRole): void {
  picked.value = role;
  void nextTick(() => document.querySelector('.versions-card')?.scrollIntoView({behavior: 'smooth', block: 'nearest'}));
}
// Arrived from a link to a role: its versions, once they are drawn.
let arrived = fromHash === null;
watch(payload, (value) => {
  if (arrived || value === null || fromHash === null) return;
  arrived = true;
  pick(fromHash);
}, {immediate: true});
const roleOptions = computed(() => roles.value.map((role) => ({key: role.role, label: role.label})));

/** The version in use of a role, or null (none chosen, or its files are gone). */
const inUseOf = (role: RoleModels): ModelVersion | null => role.versions.find((version) => version.in_use) ?? null;
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
  return widest.map((metric) => ({key: metric.key, label: metric.label.replace(/^With the cutter.*/, 'With the cutter')}));
});
function cell(version: ModelVersion, key: string) {
  return metricsOf(version).find((metric) => metric.key === key) ?? null;
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

    <!-- The three in use: tap one to list its versions below. -->
    <div class="inuse-grid" id="roles">
      <button v-for="role in roles" :key="role.role" type="button" class="inuse-card" :class="{picked: role.role === picked}"
              :id="'role-' + role.role" :data-role="role.role" :aria-pressed="role.role === picked" @click="pick(role.role)">
        <span class="inuse-role">{{ role.label }}<span class="inuse-tag" :class="{none: inUseOf(role) === null}">{{ inUseOf(role) ? 'In use' : 'None chosen' }}</span></span>
        <template v-if="inUseOf(role)">
          <strong class="inuse-name">{{ inUseOf(role)?.label }}</strong>
          <span class="inuse-when">{{ inUseOf(role)?.trained_at ? isoWhen(inUseOf(role)?.trained_at ?? null) : inUseOf(role)?.run }}<template
            v-if="role.active"> · {{ chosenText(role.active) }}</template></span>
          <span class="inuse-figs">
            <span v-for="metric in metricsOf(inUseOf(role) as ModelVersion).slice(0, 2)" :key="metric.key" class="inuse-fig">
              <b>{{ metric.value === null ? '–' : pct(metric.value) }}</b><small>{{ metric.label }}</small></span>
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
        <span class="inuse-more">{{ role.versions.length }} {{ role.versions.length === 1 ? 'version' : 'versions' }} ›</span>
      </button>
    </div>

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
              v-if="version.in_use"> · <em class="vt-use">in use</em></template><template v-else-if="!version.finished"> · <em
              class="vt-wait">not finished</em></template></small></span>
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
                       :read-only="readOnly" :busy="busy" @use="choose"/>
          </ul>
        </details>
      </div>
      <p class="note models-foot">The numbers compare only on the same test: the older versions were measured on fewer signs (the count
        under each number). The choice is kept in <code>{{ payload.active_file }}</code> of the data root; it copies, exports and
        deploys nothing: nothing on the comma changes.</p>
    </section>
    <p v-if="error" class="note">Not updated: {{ error }}</p>
  </template>
</template>
