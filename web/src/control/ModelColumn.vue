<script setup lang="ts">
// One role of the pipeline (cutter, classifier or decider): what it does, the version in use (or none yet: there is
// no default), and every version trained here, newest first.
import {computed} from 'vue';
import type {ModelVersion, RoleModels} from '../common/panel-types';
import ModelItem from './ModelItem.vue';
import {ROLE_TEXT, chosenText} from './models';

const props = defineProps<{role: RoleModels; score: number; readOnly: boolean; busy: string | null}>();
const emit = defineEmits<{use: [version: ModelVersion, roleLabel: string]}>();
const text = computed(() => ROLE_TEXT[props.role.role]);
const inUse = computed(() => props.role.versions.find((version) => version.in_use) ?? null);
</script>

<template>
  <section class="card role-card" :id="'role-' + role.role" :data-role="role.role" :aria-labelledby="'role-title-' + role.role">
    <div class="ds-head">
      <h2 :id="'role-title-' + role.role">{{ role.label }}<span v-if="role.versions.length" class="count-badge">{{ role.versions.length }}</span></h2>
    </div>
    <p class="role-what">{{ text.what }} <span class="muted">{{ text.today }}</span></p>

    <div class="role-use" :class="{set: inUse !== null, missing: role.active !== null && !role.active.found}">
      <span class="role-use-label">In use</span>
      <template v-if="role.active === null">
        <strong>None yet</strong>
        <span class="role-use-sub">{{ role.versions.length ? 'Pick one below: there is no default.' : 'Nothing to pick yet.' }}</span>
      </template>
      <template v-else-if="inUse">
        <strong>{{ inUse.run }} · {{ inUse.label }}</strong>
        <span class="role-use-sub">{{ chosenText(role.active) }}</span>
      </template>
      <template v-else>
        <strong><code>{{ role.active.id }}</code></strong>
        <span class="role-use-sub">Its files are no longer on this server: pick another one. {{ chosenText(role.active) }}</span>
      </template>
    </div>

    <div v-if="role.versions.length === 0" class="role-empty" :id="'role-empty-' + role.role">
      <strong>{{ text.empty }}</strong>
      <span>{{ text.emptyMore }}</span>
      <a v-if="role.role !== 'decider'" class="flow-link" href="/train">Go to Train →</a>
    </div>
    <ul v-else class="model-list">
      <ModelItem v-for="version in role.versions" :key="version.id" :version="version" :role-label="role.label" :score="score"
                 :read-only="readOnly" :busy="busy" @use="(item) => emit('use', item, role.label)"/>
    </ul>
  </section>
</template>
