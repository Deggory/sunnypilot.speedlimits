<script setup lang="ts">
// The sign-in page, also the first visit that creates the administrator (setup code, then name and password).
import {nextTick, onMounted, ref, useTemplateRef} from 'vue';
import {api, messageOf, post} from '../common/api';
import type {AuthState} from '../common/auth-types';

type View = 'loading' | 'login' | 'code' | 'setup';
const view = ref<View>('loading');
const message = ref('Loading…');
const error = ref('');
const busy = ref(false);
const user = ref('');
const password = ref('');
const code = ref('');
const setupUser = ref('admin');
const setupPassword = ref('');
const setupRepeat = ref('');
const plainHttp = location.protocol !== 'https:';
const first = useTemplateRef<HTMLInputElement>('first');

const TITLES: Record<View, [string, string]> = {
  loading: ['Speed limits', ''],
  login: ['Welcome back', 'Sign in to review the signs.'],
  code: ['Set up the site', 'First, the setup code of this server.'],
  setup: ['Create the administrator', 'The name and password you will sign in with from now on.'],
};

async function go(next: View): Promise<void> {
  view.value = next;
  error.value = '';
  await nextTick();
  first.value?.focus();
}

/** A form's action, with its button disabled while it waits and its error shown. */
async function submit(action: () => Promise<void>): Promise<void> {
  error.value = '';
  busy.value = true;
  try {
    await action();
  } catch (problem) {
    error.value = messageOf(problem);
  } finally {
    busy.value = false;
  }
}

// Signed in: the page the browser asked for opens by itself.
const done = (): void => location.reload();

const signIn = () => submit(async () => {
  await post('/api/auth/login', {user: user.value, password: password.value}, true);
  done();
});
const checkCode = () => submit(async () => {
  if (!code.value.trim()) throw new Error('Type the setup code.');
  await post('/api/auth/setup/check', {code: code.value.trim()}, true);
  await go('setup');
});
const createAdmin = () => submit(async () => {
  if (setupPassword.value.length < 12) throw new Error('The password needs at least 12 characters.');
  if (setupPassword.value !== setupRepeat.value) throw new Error('The two passwords are different.');
  await post('/api/auth/setup', {code: code.value.trim(), user: setupUser.value.trim(), password: setupPassword.value}, true);
  done();
});

onMounted(() => {
  api<AuthState>('/api/auth/state', {}, true).then((state) => {
    if (state.signed_in) done();
    else void go(state.configured ? 'login' : 'code');
  }).catch((problem: unknown) => { message.value = messageOf(problem); });
});
</script>

<template>
  <main class="page">
    <section class="auth-card" aria-labelledby="auth-title">
      <span class="brand-mark auth-logo" aria-hidden="true">50</span>
      <div v-if="view === 'code' || view === 'setup'" class="auth-head">
        <button v-if="view === 'setup'" class="auth-back" type="button" aria-label="Back" @click="go('code')">‹</button>
        <div class="auth-steps" aria-hidden="true"><span :class="{on: view === 'code'}"></span><span :class="{on: view === 'setup'}"></span></div>
      </div>
      <h1 class="auth-title" id="auth-title">{{ TITLES[view][0] }}</h1>
      <p class="auth-sub" id="auth-sub">{{ view === 'loading' ? message : TITLES[view][1] }}</p>

      <form v-if="view === 'login'" id="form-login" class="auth-form" novalidate @submit.prevent="signIn">
        <label class="field">Name<input id="login-user" ref="first" v-model="user" type="text" name="username" autocomplete="username"
          autocapitalize="off" autocorrect="off" spellcheck="false" maxlength="64" required></label>
        <label class="field">Password<input id="login-password" v-model="password" type="password" name="password" autocomplete="current-password" maxlength="256" required></label>
        <div class="auth-error" role="alert">{{ error }}</div>
        <button id="login-submit" class="btn primary" type="submit" :disabled="busy">Sign in</button>
      </form>

      <form v-else-if="view === 'code'" id="form-code" class="auth-form" novalidate @submit.prevent="checkCode">
        <label class="field">Setup code<input id="setup-code" ref="first" v-model="code" type="password" autocomplete="off"
          autocapitalize="off" autocorrect="off" spellcheck="false" maxlength="256" required></label>
        <p class="note">It is the access token of this server (the <code>NNSLR_REVIEW_TOKEN</code> line of the repository’s <code>.env</code> file): the installation prints it. It is needed only once.</p>
        <div class="auth-error" role="alert">{{ error }}</div>
        <button id="code-submit" class="btn primary" type="submit" :disabled="busy">Next</button>
      </form>

      <form v-else-if="view === 'setup'" id="form-setup" class="auth-form" novalidate @submit.prevent="createAdmin">
        <label class="field">Name<input id="setup-user" v-model="setupUser" type="text" name="username" autocomplete="username"
          autocapitalize="off" autocorrect="off" spellcheck="false" maxlength="64" required></label>
        <label class="field">Password<input id="setup-password" ref="first" v-model="setupPassword" type="password" name="new-password" autocomplete="new-password" maxlength="256" minlength="12" required></label>
        <label class="field">Repeat the password<input id="setup-repeat" v-model="setupRepeat" type="password" autocomplete="new-password" maxlength="256" required></label>
        <p class="note">At least 12 characters. A password manager can make one for you.</p>
        <div class="auth-error" role="alert">{{ error }}</div>
        <button id="setup-submit" class="btn primary" type="submit" :disabled="busy">Create and sign in</button>
      </form>
      <p v-if="plainHttp && view !== 'loading'" class="note auth-foot">Unencrypted connection (http): use this page only on the home network.</p>
    </section>
  </main>
</template>
