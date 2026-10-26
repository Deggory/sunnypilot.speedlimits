<script setup lang="ts">
// At the top of the inspector while Review with AI works on this drive (review/ai-live.ts): how far it is, the
// time left, the frame it did last, Follow, which opens each frame as the AI does it, and Stop (after a question).
import {computed, ref} from 'vue';
import {messageOf, post} from '../common/api';
import {thousands} from '../common/format';
import {askConfirm} from '../composables/confirm';
import {durationText} from '../control/jobs';
import {aiLive, following, toggleFollow} from './ai-live';

const stopping = ref(false);
const stopError = ref('');

async function stop(): Promise<void> {
  const live = aiLive.value;
  if (live === null || stopping.value) return;
  const ok = await askConfirm({title: 'Stop the AI review?', action: 'Stop', danger: true,
    text: 'The frames it saved stay saved. Review with AI again carries on with the frames left.'});
  if (!ok) return;
  stopping.value = true;
  stopError.value = '';
  try {
    await post('/api/panel/job/' + live.job + '/stop');
  } catch (problem) {
    stopError.value = messageOf(problem);
    stopping.value = false;
  }
}

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
    <div class="ai-live-actions">
      <button type="button" class="busy-newest" id="followAi" :aria-pressed="following" @click="toggleFollow">
        {{ following ? 'Stop following' : 'Follow the AI' }}<span class="chev" aria-hidden="true">›</span>
      </button>
      <button type="button" class="btn small danger" id="stopAi" :disabled="stopping" @click="stop">
        {{ stopping ? 'Stopping…' : 'Stop' }}</button>
    </div>
    <p v-if="stopError" class="busy-note error" role="alert">{{ stopError }}</p>
  </div>
</template>
