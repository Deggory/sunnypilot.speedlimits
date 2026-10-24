<script setup lang="ts">
// The Account menu at the right of every top bar: who is signed in, the appearance (dark, light,
// automatic), the account settings and sign out. A panel under the icon, a sheet across the width on a phone.
import {nextTick, onBeforeUnmount, onMounted, ref, useTemplateRef} from 'vue';
import {api, messageOf, post} from '../common/api';
import type {AuthState} from '../common/auth-types';
import {onThemeChange, setTheme, storedTheme, type Theme} from '../common/theme';
import {askConfirm} from '../composables/confirm';

const props = defineProps<{current?: boolean}>();
const THEMES: [Theme, string][] = [['dark', 'Dark'], ['light', 'Light'], ['auto', 'Automatic']];

const open = ref(false);
const state = ref<AuthState | null>(null);
const error = ref('');
const theme = ref<Theme>(storedTheme());
const button = useTemplateRef<HTMLButtonElement>('button');
const pop = useTemplateRef<HTMLDivElement>('pop');
onThemeChange(() => { theme.value = storedTheme(); });

async function show(): Promise<void> {
  open.value = true;
  await nextTick();
  pop.value?.querySelector<HTMLElement>('[aria-checked="true"]')?.focus();
  try {
    state.value = await api<AuthState>('/api/auth/state');
  } catch (problem) {
    error.value = messageOf(problem);
  }
}

function close(refocus: boolean): void {
  if (!open.value) return;
  open.value = false;
  if (refocus) button.value?.focus();
}

async function signOut(everywhere: boolean): Promise<void> {
  if (everywhere) {
    // The menu closes first: the dialog gives the focus back to the Account button.
    close(true);
    const ok = await askConfirm({title: 'Sign out everywhere?', action: 'Sign out everywhere', danger: true,
      text: 'Every browser signed in to this site, this one included, will have to sign in again.'});
    if (!ok) return;
  }
  try {
    await post(everywhere ? '/api/auth/logout-all' : '/api/auth/logout');
  } catch {
    // Already signed out.
  }
  location.replace('/');
}

function onDocumentClick(event: MouseEvent): void {
  const target = event.target as Node;
  if (!pop.value?.contains(target) && !button.value?.contains(target)) close(false);
}
function onKey(event: KeyboardEvent): void {
  if (event.key === 'Escape' && open.value) {
    event.stopPropagation();
    close(true);
  }
}
onMounted(() => {
  document.addEventListener('click', onDocumentClick);
  document.addEventListener('keydown', onKey, true);
});
onBeforeUnmount(() => {
  document.removeEventListener('click', onDocumentClick);
  document.removeEventListener('keydown', onKey, true);
});

const sessionsText = (value: AuthState): string =>
  (value.sessions ?? 1) > 1 ? 'Signed in on ' + value.sessions + ' browsers' : 'Signed in on this browser';
</script>

<template>
  <button ref="button" class="icon-btn account-btn" :class="{current: props.current}" type="button" id="account-btn"
          aria-haspopup="dialog" :aria-expanded="open" aria-controls="account-pop" aria-label="Account and settings"
          title="Account" @click="open ? close(false) : show()">
    <svg viewBox="0 0 20 20" aria-hidden="true">
      <circle cx="10" cy="7" r="3.4" fill="none" stroke="currentColor" stroke-width="1.8"/>
      <path d="M3.6 17.2c.9-3.1 3.4-4.8 6.4-4.8s5.5 1.7 6.4 4.8" fill="none" stroke="currentColor" stroke-width="1.8"
            stroke-linecap="round"/>
    </svg>
  </button>
  <div v-show="open" ref="pop" class="account-pop" id="account-pop" role="dialog" aria-label="Account and settings">
    <div class="pop-head">
      <span class="pop-avatar" aria-hidden="true">{{ (state?.user ?? '?').slice(0, 1).toUpperCase() }}</span>
      <div class="pop-who">
        <template v-if="state === null">
          <strong>Account</strong><span>{{ error || 'Loading…' }}</span>
        </template>
        <template v-else-if="state.local">
          <strong>This computer</strong><span>Site on 127.0.0.1, no sign-in</span>
        </template>
        <template v-else>
          <strong>{{ state.user }}</strong><span>{{ sessionsText(state) }}</span>
        </template>
      </div>
    </div>
    <div class="pop-section">
      <div class="pop-label">Appearance</div>
      <div class="segmented" role="radiogroup" aria-label="Appearance">
        <button v-for="[value, label] in THEMES" :key="value" type="button" role="radio"
                :aria-checked="theme === value" @click="setTheme(value)">{{ label }}</button>
      </div>
    </div>
    <div class="pop-section pop-items">
      <a class="pop-item" href="/account">Password and signed-in browsers</a>
      <template v-if="state !== null && !state.local">
        <button class="pop-item" type="button" @click="signOut(false)">Sign out</button>
        <button class="pop-item danger" type="button" @click="signOut(true)">Sign out everywhere</button>
      </template>
    </div>
  </div>
</template>
