<script setup lang="ts">
// Account: who is signed in, sign out, change the password (the appearance is in the Account menu).
import {onMounted, ref} from 'vue';
import {api, messageOf, post} from '../common/api';
import type {AuthAnswer, AuthState} from '../common/auth-types';

const who = ref('Caricamento…');
const current = ref('');
const fresh = ref('');
const repeat = ref('');
const error = ref('');
const done = ref('');
const busy = ref(false);

function show(state: AuthState): void {
  let text = 'Connesso come ' + (state.user ?? '') + '.';
  if (state.sessions !== undefined && state.sessions > 1) text += ' Connesso da ' + state.sessions + ' browser.';
  who.value = text;
}

async function signOut(everywhere: boolean): Promise<void> {
  if (everywhere && !window.confirm('Uscire da tutti i browser, compreso questo?')) return;
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
  if (!current.value) { error.value = 'Scrivi la password attuale.'; return; }
  if (fresh.value.length < 12) { error.value = 'La nuova password deve avere almeno 12 caratteri.'; return; }
  if (fresh.value !== repeat.value) { error.value = 'Le due nuove password sono diverse.'; return; }
  busy.value = true;
  try {
    const answer = await post<AuthAnswer>('/api/auth/password', {current: current.value, password: fresh.value});
    current.value = fresh.value = repeat.value = '';
    show(answer.state);
    done.value = 'Password cambiata: gli altri browser sono usciti.';
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
    <header class="page-head"><h1 class="page-title">Account</h1>
      <p class="page-sub">Chi entra in questo sito dal browser. Gli agenti e i client API usano invece il token nell’header, mai il nome e la password.</p></header>
    <section class="card state-box">
      <h2>Amministratore</h2>
      <p id="admin-sub">{{ who }}</p>
      <div class="buttons">
        <button id="btn-signout" class="btn" type="button" @click="signOut(false)">Esci</button>
        <button id="btn-signout-all" class="btn danger" type="button" @click="signOut(true)">Esci da tutti i browser</button>
      </div>
    </section>
    <section class="card state-box">
      <h2>Cambia la password</h2>
      <form id="password-form" class="auth-form" novalidate @submit.prevent="changePassword">
        <input type="text" name="username" autocomplete="username" hidden>
        <label class="field">Password attuale<input id="current" v-model="current" type="password" autocomplete="current-password" maxlength="256"></label>
        <label class="field">Nuova password<input id="fresh" v-model="fresh" type="password" autocomplete="new-password" maxlength="256" minlength="12"></label>
        <label class="field">Ripeti la nuova password<input id="repeat" v-model="repeat" type="password" autocomplete="new-password" maxlength="256"></label>
        <div id="password-error" class="auth-error" role="alert">{{ error }}</div>
        <div id="password-ok" class="auth-ok" role="status">{{ done }}</div>
        <button id="btn-password" class="btn primary" type="submit" :disabled="busy">Cambia la password</button>
      </form>
      <p class="note">Dopo il cambio, tutti gli altri browser devono accedere di nuovo. Password dimenticata: chi ha accesso al server ferma il sito, sposta via <code>web/auth/admin.json</code> della cartella dati e lo riavvia; la visita successiva chiede di nuovo il codice di configurazione.</p>
    </section>
  </main>
</template>
