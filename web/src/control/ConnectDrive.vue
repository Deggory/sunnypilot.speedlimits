<script setup lang="ts">
// A drive of another car, shared in comma connect (#76, #115): its link pasted here, Check asks the server what
// connect holds of it (POST /api/panel/connect/check: the car, the segments, the files the owner has not uploaded;
// nothing is written), Download starts the panel action fetch_connect with that link. Its job shows in Running below,
// like the comma's download.
import {computed, ref} from 'vue';
import {messageOf, post} from '../common/api';
import {plural, sentence} from '../common/format';
import type {ActionName, ActionOptions, ActionState, PanelStatus} from '../common/panel-types';

interface ConnectCheck {
  link: string;
  route: string;
  dongle_id: string;
  platform: string | null;
  segments: number;
  /** "1/fcamera.hevc": segment and file connect does not hold. */
  missing: string[];
  how_to_share: string | null;
  downloaded_before: boolean;
}

const props = defineProps<{status: PanelStatus | null}>();
const emit = defineEmits<{run: [name: ActionName, state: ActionState, start?: undefined, options?: ActionOptions]}>();

const link = ref('');
const checking = ref(false);
const checked = ref<ConnectCheck | null>(null);
const error = ref('');

/** The action as the server sees it; absent on a server older than #115. */
const action = computed(() => props.status?.actions?.fetch_connect ?? null);

async function check(): Promise<void> {
  checked.value = null;
  error.value = '';
  if (!link.value.trim()) return;
  checking.value = true;
  try {
    checked.value = (await post<{connect: ConnectCheck}>('/api/panel/connect/check', {link: link.value.trim()})).connect;
  } catch (failure) {
    error.value = messageOf(failure);
  } finally {
    checking.value = false;
  }
}

function download(): void {
  if (checked.value === null || action.value === null) return;
  emit('run', 'fetch_connect', action.value, undefined, {link: checked.value.link});
}

/** "Segment 1 has no road camera" for each segment with files missing, in words. */
const missingText = computed(() => {
  const bySegment = new Map<string, string[]>();
  for (const item of checked.value?.missing ?? []) {
    const [segment = '?', file = ''] = item.split('/', 2);
    const word = file === 'fcamera.hevc' ? 'road camera' : file === 'ecamera.hevc' ? 'wide camera' : 'log';
    bySegment.set(segment, [...bySegment.get(segment) ?? [], word]);
  }
  return [...bySegment].map(([segment, words]) => 'segment ' + segment + ' has no ' + words.join(' or ')).join(', ');
});
</script>

<template>
  <h3 class="prep-step">A drive of another car</h3>
  <form class="connect-row" @submit.prevent="check">
    <input v-model="link" type="text" inputmode="url" autocomplete="off" spellcheck="false" aria-label="Link of the drive in comma connect"
           placeholder="https://connect.comma.ai/<dongle>/<route>" @input="checked = null; error = ''">
    <button type="submit" class="btn small" :disabled="checking || !link.trim()"
            title="Asks comma connect what it holds of this drive; nothing is downloaded">{{ checking ? 'Checking…' : 'Check' }}</button>
  </form>
  <p v-if="error" class="figure-note proposer-missing" role="alert">{{ error }}</p>
  <div v-else-if="checked" class="connect-found">
    <p class="figure-note">{{ checked.platform ?? 'Car unknown' }} · {{ plural(checked.segments, 'segment') }} in connect
      · {{ checked.route }}<template v-if="checked.downloaded_before"> · downloaded here before: only what is missing is copied</template>.</p>
    <p v-if="checked.missing.length" class="figure-note proposer-missing" :title="checked.how_to_share ?? ''">Not uploaded to connect:
      {{ missingText }}. The rest can be downloaded now; once the owner uploads it, Download again adds it.</p>
    <span class="connect-actions">
      <button type="button" class="btn small primary" :disabled="!action?.allowed" :title="action?.allowed ? '' : action ? sentence(action.reason) : 'This server cannot download it yet'"
              @click="download">Download</button>
      <span v-if="action && !action.allowed" class="figure-note">{{ sentence(action.reason) }}</span>
    </span>
  </div>
</template>
