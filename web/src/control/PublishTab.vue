<script setup lang="ts">
// Models › Publish: the three networks in use as one package for other people (PublishSection.vue): prepare it here,
// then upload it to Hugging Face after a question. The versions in use come from the Models data, to tell when the
// package holds others.
import {computed} from 'vue';
import type {ActionName, ActionOptions, ActionState, Job, PanelStatus} from '../common/panel-types';
import {useModels} from './models';
import PublishSection from './PublishSection.vue';
import SectionHead from './SectionHead.vue';

defineProps<{status: PanelStatus | null}>();
const emit = defineEmits<{run: [name: ActionName, state: ActionState, start?: undefined, options?: ActionOptions];
  stop: [job: Job]; details: [job: Job]}>();
const {payload} = useModels();
/** The id in use of each role. */
const inUse = computed(() => Object.fromEntries((payload.value?.roles ?? []).map((role) => [role.role, role.active?.id ?? null])));
</script>

<template>
  <SectionHead tab="publish"/>
  <PublishSection :status="status" :in-use="inUse" @run="(...args) => emit('run', ...args)" @stop="emit('stop', $event)"
                  @details="emit('details', $event)"/>
</template>
