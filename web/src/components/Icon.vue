<script setup lang="ts">
// One icon of icons/shapes.ts, sized by the CSS of where it sits (.icon: 1.25em, in a button 18px); decorative,
// so hidden from screen readers: the text next to it says what it is.
import {computed} from 'vue';
import {ICONS, type IconName, type Shape} from './icons/shapes';

const props = defineProps<{name: IconName}>();
const shapes = computed((): readonly Shape[] => ICONS[props.name]);
</script>

<template>
  <svg class="icon" :data-icon="name" viewBox="0 0 24 24" aria-hidden="true" focusable="false">
    <path v-for="(shape, index) in shapes" :key="index" :d="shape.d" :fill="shape.fill ? 'currentColor' : 'none'"
          :stroke="shape.fill ? 'none' : 'currentColor'" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
  </svg>
</template>
