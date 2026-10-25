<script setup lang="ts">
// The drive's count ("476 of 2,760 confirmed", its bar) and who reviews, at the top of the inspector under the drive
// (RoutePick.vue); nothing between the site's top bar and the photo. It also measures the site's top bar
// (--sitebar-h): the photo and the inspector are sized under it.
import {computed, onMounted, ref, useTemplateRef, watch} from 'vue';
import {api} from '../common/api';
import type {AuthState} from '../common/auth-types';
import {thousands} from '../common/format';
import {writeLocal} from './dom';
import {routeCounts} from './current';
import {report, state} from './store';
import {useHeightVar} from './use-layout';



const reviewerInput = useTemplateRef<HTMLInputElement>('reviewerInput');
/** The site's top bar, rendered by pages.py above the app: the photo and the inspector are sized under both. */
const siteBar = ref<Element | null>(document.querySelector('body > header.topbar'));
useHeightVar(siteBar, '--sitebar-h');
const online = report.apiBase !== null;

const counts = routeCounts;
const barWidth = computed(() => {
  const {done, total} = counts.value;
  return total === 0 ? '0' : Math.round(done * 100 / total) + '%';
});

function typed(event: Event): void {
  state.reviewer = (event.target as HTMLInputElement).value;
  state.reviewerNeeded = false;
  writeLocal('nnslr-reviewer', state.reviewer.trim());
}

// Signed in with an account: the reviews carry its name, and there is no field to type another one.
const signedUser = ref<string | null>(null);
onMounted(() => {
  if (!online) return;
  api<AuthState>('/api/auth/state').then((auth) => {
    if (!auth.user) return;
    signedUser.value = auth.user;
    state.reviewer = auth.user;
    state.reviewerNeeded = false;
  }).catch(() => undefined);
});

// A save needed the name: the field takes the focus.
watch(() => state.reviewerFocus, () => reviewerInput.value?.focus(), {flush: 'sync'});
</script>

<template>
  <div class="route-status">
    <div class="progress" aria-live="polite">
      <div class="progress-text" id="progressText">
        <template v-if="counts.total === 0">No frames</template>
        <template v-else><strong>{{ thousands(counts.done) }} of {{ thousands(counts.total) }}</strong> confirmed<template v-if="counts.suspicious > 0 || report.routeHasHints"> · <span :class="{sus: counts.suspicious > 0}">{{ thousands(counts.suspicious) }} suspicious</span></template></template>
      </div>
      <div class="progress-bar"><span id="progressBar" :style="{width: barWidth}"></span></div>
    </div>
    <div v-if="signedUser" class="reviewer as" :title="'Your confirmations are saved as ' + signedUser">
      Reviewing as <strong>{{ signedUser }}</strong>
    </div>
    <div v-else class="reviewer">
      <label for="reviewer" class="reviewer-label">Reviewer</label>
      <input ref="reviewerInput" type="text" id="reviewer" :class="{need: state.reviewerNeeded}" maxlength="100"
             placeholder="Your name" autocomplete="name" spellcheck="false" :value="state.reviewer" @input="typed">
    </div>
  </div>
</template>
