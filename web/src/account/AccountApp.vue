<script setup lang="ts">
// Account and settings: who is signed in, sign out, change the password (the appearance is in the Account menu),
// the AI reviewer (AiProviders.vue) and the Hugging Face account our networks are published with (HuggingFace.vue). The site on 127.0.0.1 has no sign-in: there only the AI reviewer. Opened
// with the access token (an agent), nobody is signed in: one read-only line instead of the account cards.
import {computed, onMounted, ref} from 'vue';
import {api, messageOf, post} from '../common/api';
import type {AuthAnswer, AuthState} from '../common/auth-types';
import {askConfirm} from '../composables/confirm';
import AiProviders from './AiProviders.vue';
import HuggingFace from './HuggingFace.vue';

/** null until /api/auth/state answers. */
const auth = ref<AuthState | null>(null);
const who = ref('Loading…');
const current = ref('');
const fresh = ref('');
const repeat = ref('');
const error = ref('');
const done = ref('');
const busy = ref(false);
/** A browser signed in on the LAN: the account cards. */
const account = computed(() => auth.value !== null && auth.value.signed_in && !auth.value.local);
/** Opened with the access token: nobody to sign out, no password to change. */
const readOnly = computed(() => auth.value !== null && !auth.value.signed_in && !auth.value.local);

function show(state: AuthState): void {
  auth.value = state;
  const many = state.sessions !== undefined && state.sessions > 1;
  who.value = 'Signed in as ' + (state.user ?? '') + (many ? ', on ' + state.sessions + ' browsers.' : '.');
}

async function signOut(everywhere: boolean): Promise<void> {
  if (everywhere && !await askConfirm({title: 'Sign out everywhere?', action: 'Sign out everywhere', danger: true,
    text: 'Every browser signed in to this site, this one included, will have to sign in again.'})) return;
  try {
    await post(everywhere ? '/api/auth/logout-all' : '/api/auth/logout');
  } catch {
    // Already signed out.
  }
  location.replace('/');
}

async function changePassword(): Promise<void> {
  error.value = '';
  done.value = '';
  if (!current.value) { error.value = 'Type the current password.'; return; }
  if (fresh.value.length < 12) { error.value = 'The new password needs at least 12 characters.'; return; }
  if (fresh.value !== repeat.value) { error.value = 'The two new passwords are different.'; return; }
  busy.value = true;
  try {
    const answer = await post<AuthAnswer>('/api/auth/password', {current: current.value, password: fresh.value});
    current.value = fresh.value = repeat.value = '';
    show(answer.state);
    done.value = 'Password changed: the other browsers are signed out.';
  } catch (problem) {
    error.value = messageOf(problem);
  } finally {
    busy.value = false;
  }
}

onMounted(() => {
  api<AuthState>('/api/auth/state').then(show).catch((problem: unknown) => { who.value = messageOf(problem); });
});
</script>

<template>
  <main class="page settings">
    <h1 class="sr-only">Account and settings</h1>
    <section v-if="readOnly" class="card state-box" id="read-only">
      <h2>Read only</h2>
      <p>Nobody is signed in, so this page can only show the settings. Sign in on the site to change them.</p>
    </section>
    <section v-if="account" class="card state-box">
      <h2>Administrator</h2>
      <p id="admin-sub">{{ who }}</p>
      <div class="buttons">
        <button id="btn-signout" class="btn" type="button" @click="signOut(false)">Sign out</button>
        <button id="btn-signout-all" class="btn danger" type="button" @click="signOut(true)">Sign out everywhere</button>
      </div>
    </section>
    <section v-if="account" class="card state-box">
      <h2>Change the password</h2>
      <form id="password-form" class="auth-form" novalidate @submit.prevent="changePassword">
        <input type="text" name="username" autocomplete="username" hidden>
        <label class="field">Current password<input id="current" v-model="current" type="password" autocomplete="current-password" maxlength="256"></label>
        <label class="field">New password<input id="fresh" v-model="fresh" type="password" autocomplete="new-password" maxlength="256" minlength="12"></label>
        <label class="field">Repeat the new password<input id="repeat" v-model="repeat" type="password" autocomplete="new-password" maxlength="256"></label>
        <div id="password-error" class="auth-error" role="alert">{{ error }}</div>
        <div id="password-ok" class="auth-ok" role="status">{{ done }}</div>
        <button id="btn-password" class="btn primary" type="submit" :disabled="busy">Change the password</button>
      </form>
      <p class="note">After the change every other browser has to sign in again.</p>
    </section>
    <AiProviders/>
    <HuggingFace/>
  </main>
</template>
