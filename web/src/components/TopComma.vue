<script setup lang="ts">
// The comma at the right of the top bar, next to the server (#comma-pill of pages.py): a pill with a green dot when
// this server reaches it, grey when it does not (the car off, or away from the Wi-Fi), blue while a download from it
// runs; on a click, a panel with that, its address (changed and saved there) and Download new drives (the panel
// action "fetch", to that address). It asks /api/panel/comma every 10 s (nnslr_tools.comma_link: only the ssh port
// is tried, nothing runs on the comma).
import {computed, onBeforeUnmount, onMounted, ref} from 'vue';
import {messageOf, poll, post, signedOut} from '../common/api';
import {isoWhen} from '../common/format';
import {askConfirm} from '../composables/confirm';

interface CommaState {
  online: boolean | null;
  checked: string | null;
  downloading: boolean;
  /** What a download waits for ("a preparation"): Download new drives is off until it ends; null: nothing. */
  waits_for?: string | null;
  /** The download of the panel running now: its percent (null while unknown) and what it is at. */
  download_progress?: {percent: number | null; detail: string} | null;
  /** What is tried: the address set here, or "comma (192.168.200.208)", the ssh host of the server. */
  address: string;
  configured: boolean;
  /** This server's key accepted by the comma (None while it is away). */
  login: boolean | null;
  /** This server's public ssh key, for the GitHub user whose keys the comma takes. */
  server_key: string | null;
  /** ISO time of the last login that worked (nnslr_tools.comma_link.last_login); null: never. */
  last_login?: string | null;
}

const INTERVAL_MS = 10000;
const comma = ref<CommaState | null>(null);
const open = ref(false);
const root = ref<HTMLElement | null>(null);
const message = ref('');
/** The address being edited; null when not editing. */
const editing = ref<string | null>(null);

async function saveAddress(): Promise<void> {
  if (editing.value === null) return;
  try {
    comma.value = (await post<{comma: CommaState}>('/api/panel/comma', {address: editing.value})).comma;
    editing.value = null;
    message.value = '';
  } catch (error) {
    message.value = messageOf(error);
  }
}

const dot = computed(() => comma.value?.downloading ? 'busy' : comma.value?.online && comma.value.login === false ? 'warm'
  : comma.value?.online ? 'on' : 'off');
const copied = ref(false);
/** The setup from zero, open until the comma answers and takes this server's key (or when asked); folded for a comma
 * this server once logged in to, which is only switched off or away now. */
const setupOpen = ref<boolean | null>(null);
const ready = computed(() => comma.value?.online === true && comma.value.login === true);
const known = computed(() => Boolean(comma.value?.last_login) && comma.value?.login !== false);
const showSetup = computed(() => setupOpen.value ?? (!ready.value && !known.value));
/** "Last connected 18:55" (or "24 Oct 18:55") for a known comma that is away. */
const lastSeen = computed(() => comma.value?.online === false && comma.value.last_login
  ? 'Last connected ' + isoWhen(comma.value.last_login) + '.' : '');
const testing = ref(false);
async function testNow(): Promise<void> {
  testing.value = true;
  message.value = '';
  try {
    comma.value = (await post<{comma: CommaState}>('/api/panel/comma/test', {})).comma;
    if (!comma.value.online) message.value = 'It does not answer: is it on, and on the same network as this server?';
    else if (comma.value.login === false) message.value = 'It answers but refuses this server\'s key: step 2.';
    else message.value = 'Connected: this server can log in.';
  } catch (error) {
    message.value = messageOf(error);
  } finally {
    testing.value = false;
  }
}
async function copyKey(): Promise<void> {
  if (!comma.value?.server_key) return;
  try {
    await navigator.clipboard.writeText(comma.value.server_key);
    copied.value = true;
    window.setTimeout(() => { copied.value = false; }, 2000);
  } catch {
    copied.value = false;
  }
}
const downloadPercent = computed(() => {
  const value = comma.value?.download_progress?.percent;
  return value === null || value === undefined ? null : Math.max(0, Math.min(100, value));
});
const word = computed(() => comma.value?.downloading
  ? 'Downloading' + (downloadPercent.value === null ? '' : ' · ' + Math.floor(downloadPercent.value) + '%') : comma.value?.online && comma.value.login === false ? 'Refuses this server'
  : comma.value?.online ? 'Connected'
  : comma.value?.online === false ? 'Not reachable' : 'Checking…');

async function ask(): Promise<void> {
  try {
    comma.value = (await poll<{comma: CommaState}>('/api/panel/comma')).comma;
  } catch (error) {
    // An old server without the address, or the network: the pill stays as it was (greyed out by the connection
    // state of the top bar). Signed out: no more asking, the page stays as it is.
    if (signedOut(error)) window.clearInterval(timer);
  }
}

async function download(): Promise<void> {
  const ok = await askConfirm({title: 'Download from the comma?', action: 'Download',
    text: 'Copies the new drives from the comma, segment by segment; the drives already here stay as they are.'});
  if (!ok) return;
  message.value = 'Starting…';
  try {
    await post('/api/panel/run/fetch', {});
    message.value = 'Downloading: the Prepare tab shows how far it is.';
    void ask();
  } catch (error) {
    message.value = messageOf(error);
  }
}

function onDocument(event: Event): void {
  const target = event.target as Node;
  // The confirm dialog of Download is outside the panel: answering it keeps the panel open, to show how it goes.
  if (document.getElementById('dialog-host')?.contains(target)) return;
  if (open.value && root.value && !root.value.contains(target)) open.value = false;
}
function onKey(event: KeyboardEvent): void {
  if (event.key === 'Escape') open.value = false;
}

let timer: number | undefined;
onMounted(() => {
  document.addEventListener('pointerdown', onDocument);
  document.addEventListener('keydown', onKey);
  void ask();
  timer = window.setInterval(() => void ask(), INTERVAL_MS);
});
onBeforeUnmount(() => {
  window.clearInterval(timer);
  document.removeEventListener('pointerdown', onDocument);
  document.removeEventListener('keydown', onKey);
});
</script>

<template>
  <div v-if="comma" ref="root" class="tm">
    <button type="button" class="tm-pill" :aria-expanded="open" aria-controls="comma-pop" :aria-label="'The comma: ' + word"
            :title="'The comma: ' + word" @click="open = !open">
      <span class="dot" :class="dot" aria-hidden="true"></span><span>comma</span>
    </button>
    <div v-if="open" id="comma-pop" class="tm-pop" role="dialog" aria-label="The comma">
      <h2>comma</h2>
      <p class="tm-sub">{{ word }}<template v-if="comma.online === false && !comma.downloading">: the car is off, or not on the
        Wi-Fi of this server.</template> {{ lastSeen }}</p>
      <div class="tm-row tm-address">
        <template v-if="editing === null">
          <span class="tm-name">Address<small>{{ comma.address }}</small></span>
          <button type="button" class="link-button" @click="editing = comma.configured ? comma.address : ''">Change</button>
        </template>
        <form v-else class="tm-edit" @submit.prevent="saveAddress">
          <label class="tm-name" for="comma-address">Address</label>
          <input id="comma-address" v-model="editing" type="text" inputmode="url" autocomplete="off" spellcheck="false"
                 placeholder="e.g. 192.168.1.20 · empty: the ssh host comma">
          <span class="tm-edit-actions"><button type="button" class="btn small" @click="editing = null">Cancel</button>
            <button type="submit" class="btn small primary">Save</button></span>
        </form>
      </div>
      <p v-if="comma.online && comma.login" class="tm-sub">This server can log in to it.</p>
      <button v-if="!showSetup" type="button" class="link-button tm-setup-open" @click="setupOpen = true">Set up a comma</button>
      <ol v-else class="tm-setup">
        <li><strong>On the comma</strong>: Settings → Network: turn on <em>Enable SSH</em>, and under <em>SSH Keys</em> enter
          a GitHub username. The comma takes that user's public keys.</li>
        <li><strong>On GitHub</strong>, signed in as that user: add this server's key
          (<a href="https://github.com/settings/ssh/new" target="_blank" rel="noopener">Settings → SSH keys → New</a>).
          <span v-if="comma.server_key" class="tm-keyline"><code class="tm-key">{{ comma.server_key }}</code>
            <button type="button" class="link-button" @click="copyKey">{{ copied ? 'Copied' : 'Copy' }}</button></span></li>
        <li><strong>Here</strong>: its address, the IP under Settings → Network → Advanced on the comma (above).</li>
        <li><button type="button" class="btn small" :disabled="testing" @click="testNow">{{ testing ? 'Testing…' : 'Test now' }}</button></li>
      </ol>
      <div v-if="comma.downloading" class="tm-progress" role="status">
        <div class="tm-bar" :class="{indeterminate: downloadPercent === null}" role="progressbar" aria-label="Download"
             aria-valuemin="0" aria-valuemax="100" :aria-valuenow="downloadPercent === null ? undefined : Math.floor(downloadPercent)"><span
             :style="downloadPercent === null ? undefined : {width: downloadPercent + '%'}"></span></div>
        <p v-if="comma.download_progress?.detail" class="tm-sub">{{ comma.download_progress.detail }}</p>
      </div>
      <div class="tm-update">
        <button type="button" class="btn small primary"
                :disabled="!comma.online || comma.login === false || comma.downloading || !!comma.waits_for" @click="download">
          Download new drives</button>
        <span v-if="comma.waits_for && !comma.downloading" class="tm-sub" role="status">Waits for {{ comma.waits_for }} to
          finish (Data › Drives): the two never run together.</span>
        <span v-else-if="message" class="tm-sub" role="status">{{ message }}</span>
      </div>
    </div>
  </div>
</template>
