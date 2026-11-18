<script setup lang="ts">
// A drive of another car shared in comma connect (#76, #115), in the comma panel of the top bar under Download new
// drives, usable with the comma away: its <dongle>/<route> typed here, Check asks the server what connect holds of it
// (POST /api/panel/connect/check: the car, the segments, the files the owner has not uploaded; nothing is written),
// Download starts the panel action fetch_connect with it. Its progress comes with the comma's state
// (connect_downloading, connect_progress of /api/panel/comma); its job shows in Data › Prepare like the comma's download.
import {computed, ref} from 'vue';
import {messageOf, post} from '../common/api';
import {plural} from '../common/format';
import {askConfirm} from '../composables/confirm';

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

const props = defineProps<{
  downloading: boolean;
  /** What keeps it from starting ("a preparation"); null: nothing. */
  waitsFor: string | null;
  progress: {percent: number | null; detail: string} | null;
}>();
const emit = defineEmits<{started: []}>();

const drive = ref('');
const checking = ref(false);
const checked = ref<ConnectCheck | null>(null);
const message = ref('');

async function check(): Promise<void> {
  checked.value = null;
  message.value = '';
  if (!drive.value.trim()) return;
  checking.value = true;
  try {
    checked.value = (await post<{connect: ConnectCheck}>('/api/panel/connect/check', {link: drive.value.trim()})).connect;
  } catch (failure) {
    message.value = messageOf(failure);
  } finally {
    checking.value = false;
  }
}

async function download(): Promise<void> {
  if (checked.value === null) return;
  const ok = await askConfirm({title: 'Download ' + checked.value.route + ' from comma connect?', action: 'Download',
    text: 'Copies its road camera, wide camera and log, segment by segment, into the video folder (about 100 MB a '
      + 'minute of drive). Then Prepare takes it like a drive of ours.'});
  if (!ok) return;
  message.value = 'Starting…';
  try {
    await post('/api/panel/run/fetch_connect', {link: checked.value.link});
    message.value = 'Downloading: the Prepare tab shows how far it is.';
    checked.value = null;
    drive.value = '';
    emit('started');
  } catch (failure) {
    message.value = messageOf(failure);
  }
}

/** "segment 1 has no road camera" for each segment with files missing. */
const missingText = computed(() => {
  const bySegment = new Map<string, string[]>();
  for (const item of checked.value?.missing ?? []) {
    const [segment = '?', file = ''] = item.split('/', 2);
    const word = file === 'fcamera.hevc' ? 'road camera' : file === 'ecamera.hevc' ? 'wide camera' : 'log';
    bySegment.set(segment, [...bySegment.get(segment) ?? [], word]);
  }
  return [...bySegment].map(([segment, words]) => 'segment ' + segment + ' has no ' + words.join(' or ')).join(', ');
});
const percent = computed(() => {
  const value = props.progress?.percent;
  return value === null || value === undefined ? null : Math.max(0, Math.min(100, value));
});
</script>

<template>
  <div class="tm-connect">
    <span class="tm-name">From comma connect<small>A drive someone shared with you.</small></span>
    <div v-if="downloading" class="tm-progress" role="status">
      <div class="tm-bar" :class="{indeterminate: percent === null}" role="progressbar" aria-label="Download from comma connect"
           aria-valuemin="0" aria-valuemax="100" :aria-valuenow="percent === null ? undefined : Math.floor(percent)"><span
           :style="percent === null ? undefined : {width: percent + '%'}"></span></div>
      <p class="tm-sub">Downloading{{ progress?.detail ? ': ' + progress.detail : '…' }}</p>
    </div>
    <form v-else class="tm-edit" @submit.prevent="check">
      <input v-model="drive" type="text" autocomplete="off" spellcheck="false" aria-label="Dongle id / route id"
             placeholder="dongle id/route id" title="e.g. 35edd31af7e86e74/00000004--3188701194"
             @input="checked = null; message = ''">
      <span class="tm-edit-actions">
        <button type="submit" class="btn small" :disabled="checking || !drive.trim()"
                title="Asks comma connect what it holds of this drive; nothing is downloaded">{{ checking ? 'Checking…' : 'Check' }}</button>
        <button v-if="checked" type="button" class="btn small primary" :disabled="!!waitsFor" @click="download">Download</button>
      </span>
    </form>
    <template v-if="checked && !downloading">
      <p class="tm-sub">{{ checked.platform ?? 'Car unknown' }} · {{ plural(checked.segments, 'segment') }}<template
        v-if="checked.downloaded_before"> · downloaded before: only what is missing is copied</template>.</p>
      <p v-if="checked.missing.length" class="tm-sub tm-warn" :title="checked.how_to_share ?? ''">Not uploaded to connect:
        {{ missingText }}. The rest can be downloaded now; Download again adds it once the owner uploads it.</p>
    </template>
    <p v-if="waitsFor && !downloading && checked" class="tm-sub" role="status">Waits for {{ waitsFor }} to finish.</p>
    <p v-if="message" class="tm-sub" role="status">{{ message }}</p>
  </div>
</template>
