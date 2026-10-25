<script setup lang="ts">
// "Hugging Face" in Settings: the account our networks are published with (Models › Publish). A wizard in three
// steps: create a token with Write access on Hugging Face, paste it, check and save. The server asks Hugging Face
// whose it is and whether it can write, and keeps it (GET/POST /api/settings/huggingface): the page only ever learns
// the user name and what the token can do, never the token again.
import {computed, nextTick, onMounted, ref} from 'vue';
import {api, messageOf, post} from '../common/api';
import {isoWhen} from '../common/format';
import {askConfirm} from '../composables/confirm';

interface HfState {
  connected: boolean;
  /** portal: a token saved here; server: the server's own `hf auth login`; null: none. */
  source: 'portal' | 'server' | null;
  user: string | null;
  role: string | null;
  can_write: boolean | null;
  checked_at: string | null;
  new_token_url: string;
  can_edit: boolean;
}

const state = ref<HfState | null>(null);
const token = ref('');
const error = ref('');
const done = ref('');
const busy = ref(false);
/** The wizard is open: no token saved yet, or "Replace the token". */
const wizard = ref(false);
const opened = ref(false);
const showWizard = computed(() => state.value !== null && state.value.can_edit && (wizard.value || state.value.source !== 'portal'));

async function load(): Promise<void> {
  try {
    state.value = await api<HfState>('/api/settings/huggingface');
  } catch (problem) {
    error.value = messageOf(problem);
  }
}

async function save(): Promise<void> {
  error.value = '';
  done.value = '';
  if (!token.value.trim()) { error.value = 'Paste the token first.'; return; }
  busy.value = true;
  try {
    state.value = await post<HfState>('/api/settings/huggingface', {token: token.value.trim()});
    token.value = '';
    wizard.value = false;
    done.value = 'Connected as ' + (state.value.user ?? '?') + ': the networks can be published.';
  } catch (problem) {
    error.value = messageOf(problem);
  } finally {
    busy.value = false;
  }
}

async function disconnect(): Promise<void> {
  if (!await askConfirm({title: 'Forget the Hugging Face token?', action: 'Forget', danger: true,
    text: 'The portal can no longer publish with it. The token itself stays valid on Hugging Face: delete it there too if it should not be used again.'})) return;
  try {
    state.value = await post<HfState>('/api/settings/huggingface/remove', {});
    done.value = 'The token is forgotten.';
  } catch (problem) {
    error.value = messageOf(problem);
  }
}

onMounted(async () => {
  await load();
  if (location.hash === '#huggingface') {
    await nextTick();
    document.getElementById('huggingface')?.scrollIntoView({block: 'start'});
  }
});
</script>

<template>
  <section class="card state-box hf-card" id="huggingface" aria-labelledby="hf-title">
    <h2 id="hf-title">Hugging Face</h2>
    <p>The account our networks are published with, for other people (Models › Publish).</p>
    <p v-if="state === null && !error" class="ai-loading">Loading…</p>
    <template v-if="state !== null">
      <div v-if="state.source === 'portal'" class="hf-connected" id="hf-connected">
        <span class="pill ok">Connected</span>
        <span class="hf-who">as <strong>{{ state.user }}</strong> · {{ state.can_write ? 'can publish' : 'read only' }}
          <span v-if="state.checked_at" class="hf-when">· checked {{ isoWhen(state.checked_at) }}</span></span>
        <div v-if="state.can_edit && !wizard" class="buttons">
          <button type="button" class="btn" @click="wizard = true">Replace the token</button>
          <button type="button" class="btn danger" @click="disconnect">Forget it</button>
        </div>
      </div>
      <p v-else-if="state.source === 'server'" class="note" id="hf-server">This server is already logged in to Hugging Face
        (<code>hf auth login</code>), and publishing uses that login. Connect a token here to see whose it is and to change it from the portal.</p>
      <p v-else class="note">Not connected: Publish waits until a token is saved here.</p>

      <ol v-if="showWizard" class="hf-steps" id="hf-wizard">
        <li>
          <span class="hf-step-title">Create a token</span>
          <span class="hf-step-text">On Hugging Face, signed in as yourself: type <strong>Write</strong>, any name (for example
            <em>sunnypilot-speedlimits</em>), then <strong>Create token</strong> and copy it.</span>
          <a class="btn" :href="state.new_token_url" target="_blank" rel="noopener" @click="opened = true">Open Hugging Face ↗</a>
        </li>
        <li>
          <span class="hf-step-title">Paste it here</span>
          <label class="field">Token<input id="hf-token" v-model="token" type="password" autocomplete="off" spellcheck="false"
                 placeholder="hf_…" maxlength="300" @keydown.enter.prevent="save"></label>
        </li>
        <li>
          <span class="hf-step-title">Check and save</span>
          <span class="hf-step-text">The server asks Hugging Face whose token it is and whether it can write, then keeps it. It is never
            shown again, nor written in a log.</span>
          <div class="buttons">
            <button id="hf-save" type="button" class="btn primary" :disabled="busy || !token.trim()" @click="save">{{ busy ? 'Checking…' : 'Check and save' }}</button>
            <button v-if="wizard" type="button" class="btn" @click="wizard = false; token = ''">Cancel</button>
          </div>
        </li>
      </ol>
      <p v-if="!state.can_edit" class="note">Sign in on the site to change it.</p>
    </template>
    <div class="auth-error" role="alert">{{ error }}</div>
    <div class="auth-ok" role="status">{{ done }}</div>
  </section>
</template>
