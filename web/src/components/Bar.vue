<script setup lang="ts">
// A progress bar (.bar in shared.css), 0-100, with an optional colour kind (ok, warn, bad, run).
defineProps<{percent: number; kind?: string}>();
</script>

<template>
  <div class="bar" :class="kind" aria-hidden="true"><span :style="{width: Math.max(0, Math.min(100, percent)) + '%'}"></span></div>
</template>
