<script setup lang="ts">
// The machine in small under the buttons of a running job: GPU, its memory and temperature, CPU and RAM, each its
// value now and the last three minutes as a sparkline.
import {computed} from 'vue';
import type {SystemInfo} from '../common/panel-types';
import Spark from '../components/Spark.vue';
import {machineCells} from './machine-history';

const props = defineProps<{system: SystemInfo}>();
/** 45 points of 4 s. */
const POINTS = 45;

const cells = computed(() => machineCells(props.system, POINTS));
</script>

<template>
  <div v-if="cells.length" class="jp-machine" role="group" aria-label="The server now">
    <div v-for="cell in cells" :key="cell.key" class="mini" :class="{hot: cell.hot}" :data-gauge="cell.key" :title="cell.label + ': ' + cell.title">
      <span class="mini-label">{{ cell.label }}</span>
      <span class="mini-value">{{ cell.value }}</span>
      <Spark v-if="cell.values.length" :values="cell.values" :color="cell.color" :points="POINTS"/>
    </div>
  </div>
</template>
