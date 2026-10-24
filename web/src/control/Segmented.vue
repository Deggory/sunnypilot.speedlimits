<script setup lang="ts" generic="Key extends string">
// A small choice of two or three (All rows / Human-reviewed, Test / Validation): buttons in a pill, the chosen one
// marked as the tabs of the top bar are. v-model is the key chosen.
defineProps<{options: readonly {key: Key; label: string}[]; label: string}>();
const model = defineModel<Key>({required: true});
</script>

<template>
  <div class="seg" role="radiogroup" :aria-label="label">
    <button v-for="option in options" :key="option.key" type="button" role="radio" :aria-checked="model === option.key"
            @click="model = option.key">{{ option.label }}</button>
  </div>
</template>
