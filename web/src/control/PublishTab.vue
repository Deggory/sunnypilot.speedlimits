<script setup lang="ts">
// Models › Publish: first the networks in use into the comma's software, in one go (CarPublishSection.vue: prepared,
// published in speed-limits-models, psa-torque-sunny-testing built); then, apart and optional, a package of the
// networks in use for other people on Hugging Face (PublishSection.vue), which the car does not need. The versions in
// use come from the Models data.
import {computed} from 'vue';
import type {ActionName, ActionOptions, ActionState, Job, PanelStatus} from '../common/panel-types';
import CarPublishSection from './CarPublishSection.vue';
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
  <CarPublishSection :status="status" :models="payload" @run="(...args) => emit('run', ...args)" @stop="emit('stop', $event)"
                     @details="emit('details', $event)"/>
  <PublishSection :status="status" :in-use="inUse" @run="(...args) => emit('run', ...args)" @stop="emit('stop', $event)"
                  @details="emit('details', $event)"/>
</template>
