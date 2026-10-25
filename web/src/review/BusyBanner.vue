<script setup lang="ts">
// At the top of the inspector of a read-only drive (review/busy.ts): Propose again works on it, how far it is, and a
// reload when there is more to see (the newest proposals, or the drive done).
import {computed, onMounted} from 'vue';
import {busyBanner, watchBusy} from './busy';

const banner = computed(() => busyBanner());
onMounted(watchBusy);
function reload(): void {
  window.location.reload();
}
</script>

<template>
  <div v-if="banner" class="busy-banner" role="status" id="busyBanner">
    <p><strong>{{ banner.title }}</strong> {{ banner.text }}</p>
    <button v-if="banner.reload" type="button" class="btn small" @click="reload">{{ banner.reload }}</button>
  </div>
</template>
