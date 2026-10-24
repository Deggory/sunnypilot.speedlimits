<script setup lang="ts">
// An area chart 0-100% of a history, newest on the right. The colour is a CSS variable: it follows the theme.
import {computed} from 'vue';

const props = defineProps<{values: number[]; color: string; points: number}>();
const line = computed(() => {
  const start = props.points - props.values.length;
  return props.values.map((value, index) => (start + index) + ',' + (30 - Math.max(0, Math.min(100, value)) * 0.3));
});
const area = computed(() =>
  (props.points - props.values.length) + ',30 ' + line.value.join(' ') + ' ' + (props.points - 1) + ',30');
</script>

<template>
  <div class="spark" aria-hidden="true">
    <svg :viewBox="'0 0 ' + points + ' 30'" preserveAspectRatio="none">
      <template v-if="values.length > 1">
        <polygon :points="area" :style="{fill: color}" fill-opacity="0.15"/>
        <polyline :points="line.join(' ')" fill="none" :style="{stroke: color}" stroke-width="1.5"
                  vector-effect="non-scaling-stroke"/>
      </template>
    </svg>
  </div>
</template>
