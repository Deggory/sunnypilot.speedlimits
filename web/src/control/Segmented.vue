<script setup lang="ts" generic="Key extends string">
// A small choice of two or three (All rows / Human-reviewed, Test / Validation): buttons in a pill, the chosen one
// marked as the tabs of the top bar are. v-model is the key chosen; disabled shows it without letting it change.
defineProps<{options: readonly {key: Key; label: string}[]; label: string; disabled?: boolean}>();
const model = defineModel<Key>({required: true});
</script>

<template>
  <div class="seg" role="radiogroup" :aria-label="label" :aria-disabled="disabled || undefined">
    <button v-for="option in options" :key="option.key" type="button" role="radio" :aria-checked="model === option.key"
            :disabled="disabled" @click="model = option.key">{{ option.label }}</button>
  </div>
</template>
