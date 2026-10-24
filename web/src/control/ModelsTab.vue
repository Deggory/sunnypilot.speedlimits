<script setup lang="ts">
// Models: the three networks of the pipeline side by side (cutter, classifier, decider; docs/NEURAL_NETWORK.md), each
// with every version trained on this server, newest first, its headline numbers on the test, and the one in use. A
// role has no model in use until someone picks one: there is no default. The runs themselves are on Evaluate.
import {computed, ref} from 'vue';
import type {ModelVersion, PanelStatus, RoleModels} from '../common/panel-types';
import ModelColumn from './ModelColumn.vue';
import {putInUse, useModels} from './models';

const props = defineProps<{status: PanelStatus | null}>();
const {payload, error, refresh} = useModels();
const roles = computed((): RoleModels[] => payload.value?.roles ?? []);
/** Read-only server: the choice is off, as the buttons of the other tabs. */
const readOnly = computed(() => props.status !== null && !props.status.actions_enabled);
/** The version being put in use, so its button waits for the answer. */
const busy = ref<string | null>(null);

async function choose(version: ModelVersion, roleLabel: string): Promise<void> {
  if (busy.value !== null) return;
  busy.value = version.id;
  try {
    if (await putInUse(version, roleLabel)) await refresh();
  } finally {
    busy.value = null;
  }
}
</script>

<template>
  <header class="page-head">
    <h1 class="page-title">Models</h1>
    <p class="page-sub">The three networks of the pipeline, every version trained here, and the one in use for each.</p>
  </header>

  <p v-if="payload === null" class="card empty">{{ error ? 'The models cannot be read: ' + error : 'Loading…' }}</p>
  <template v-else>
    <p v-if="payload.active_problem" class="banner" id="active-problem">{{ payload.active_problem }}: the choices cannot be shown or changed until it is fixed or removed.</p>
    <div class="roles" id="roles">
      <ModelColumn v-for="role in roles" :key="role.role" :role="role" :score="payload.cutter_score_threshold"
                   :read-only="readOnly || payload.active_problem !== null" :busy="busy" @use="choose"/>
    </div>
    <p class="note models-foot">The choice is kept in <code>{{ payload.active_file }}</code> of the data root. It copies, exports
      and deploys nothing: nothing on the comma changes.</p>
    <p v-if="error" class="note">Not updated: {{ error }}</p>
  </template>
</template>
