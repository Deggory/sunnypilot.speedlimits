<script setup lang="ts">
// At the top of the inspector of a read-only drive (review/busy.ts): Propose again works on it and how far it is;
// on the drive at work Follow (the newest frame shown as it comes), a reload once the drive is done.
import {computed, onMounted} from 'vue';
import {busyBanner, following, toggleFollow, watchBusy} from './busy';
import {report} from './store';

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
    <button v-else-if="report.atWork && report.readOnly" type="button" class="btn small" :class="{on: following}"
            :aria-pressed="following" @click="toggleFollow">{{ following ? 'Following' : 'Follow' }}</button>
  </div>
</template>
