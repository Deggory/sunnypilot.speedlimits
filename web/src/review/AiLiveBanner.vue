<script setup lang="ts">
// At the top of the inspector while Review with AI works on this drive (review/ai-live.ts): how far it is, the
// time left, the frame it did last, and Follow, which opens each frame as the AI does it.
import {computed} from 'vue';
import {thousands} from '../common/format';
import {durationText} from '../control/jobs';
import {aiLive, following, toggleFollow} from './ai-live';

const percent = computed(() => {
  const live = aiLive.value;
  return live && live.total ? Math.min(100, Math.round(100 * live.done / live.total)) : null;
});
const text = computed(() => {
  const live = aiLive.value;
  if (live === null) return '';
  const parts = [thousands(live.done) + (live.total ? ' of ' + thousands(live.total) : '') + ' frames'];
  if (live.seconds_left) parts.push(durationText(live.seconds_left) + ' left');
  if (live.frame_key) parts.push('last ' + live.frame_key);
  return parts.join(' · ');
});
</script>

<template>
  <div v-if="aiLive" class="busy-banner ai-live" id="aiLiveBanner">
    <div class="busy-head">
      <strong>{{ aiLive.saving ? 'The AI is reviewing this drive' : 'The AI is trying this drive' }}</strong>
      <span v-if="percent !== null" class="busy-pct">{{ percent }}%</span>
    </div>
    <div v-if="percent !== null" class="busy-bar" role="progressbar" aria-label="Review with AI"
         :aria-valuenow="percent" aria-valuemin="0" aria-valuemax="100"><span :style="{width: percent + '%'}"></span></div>
    <p>{{ text }}</p>
    <p class="busy-note">{{ aiLive.saving ? 'Its saves appear here as they come.' : 'A trial: nothing is saved.' }}</p>
    <button type="button" class="busy-newest" id="followAi" :aria-pressed="following" @click="toggleFollow">
      {{ following ? 'Stop following' : 'Follow the AI' }}<span class="chev" aria-hidden="true">›</span>
    </button>
  </div>
</template>
