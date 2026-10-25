<script setup lang="ts">
// Account and settings: who is signed in, sign out, change the password (the appearance is in the Account menu),
// and the AI reviewer (AiProviders.vue). The site on 127.0.0.1 has no sign-in: there only the AI reviewer.
import {onMounted, ref} from 'vue';
import {api, messageOf, post} from '../common/api';
import type {AuthAnswer, AuthState} from '../common/auth-types';
import {askConfirm} from '../composables/confirm';
import AiProviders from './AiProviders.vue';

/** null until /api/auth/state answers. */
const auth = ref<AuthState | null>(null);
const who = ref('Loading…');
const current = ref('');
const fresh = ref('');
const repeat = ref('');
const error = ref('');
const done = ref('');
const busy = ref(false);

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
    <section v-if="!auth?.local" class="card state-box">
      <h2>Administrator</h2>
      <p id="admin-sub">{{ who }}</p>
      <div class="buttons">
        <button id="btn-signout" class="btn" type="button" @click="signOut(false)">Sign out</button>
        <button id="btn-signout-all" class="btn danger" type="button" @click="signOut(true)">Sign out everywhere</button>
      </div>
    </section>
    <section v-if="!auth?.local" class="card state-box">
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
      <p class="note">After the change every other browser has to sign in again. Forgotten password: someone with access to the server stops the site, moves <code>web/auth/admin.json</code> out of the data folder and starts it again; the next visit asks for the setup code again.</p>
    </section>
    <AiProviders/>
  </main>
</template>
