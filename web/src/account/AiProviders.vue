<script setup lang="ts">
// "AI reviewer" in Settings: the LLM servers the site may ask to review signs (GET /api/settings/ai), as an iOS
// grouped list. Each row: the name, the model, the host of its address, a dot for its key, what its last Test said,
// and "Reviews" on the one the site asks (on the review page, and for a whole drive from the route list). A row
// opens its sheet (ProviderSheet.vue); "Add a provider" opens an empty one. Test asks the model about one reviewed
// frame and says whether it answered and sees images. "Who reviews" under it is a checkmark list: a provider that
// passed its Test, or "Don't use AI" (the server refuses the others). Read-only when the browser may not change them.
import {computed, nextTick, onMounted, ref} from 'vue';
import {api, messageOf, post} from '../common/api';
import type {AiProvider, AiSettings, AiTestResult} from '../common/ai-types';
import {isoWhen} from '../common/format';
import ProviderSheet from './ProviderSheet.vue';

const MAX_PROVIDERS = 20;

const settings = ref<AiSettings | null>(null);
const error = ref('');
const done = ref('');
/** The provider in the sheet; 'new' for an empty one; null: closed. */
const editing = ref<AiProvider | 'new' | null>(null);
const testing = ref<string | null>(null);
const choosing = ref(false);

const canEdit = computed(() => settings.value?.can_edit === true);
const sheetProvider = computed(() => (editing.value === 'new' ? null : editing.value));

async function load(): Promise<void> {
  try {
    settings.value = await api<AiSettings>('/api/settings/ai');
    error.value = '';
  } catch (problem) {
    error.value = messageOf(problem);
  }
}

function take(value: AiSettings): void {
  settings.value = {providers: value.providers, reviewer: value.reviewer, can_edit: value.can_edit};
}

function saved(value: AiSettings, message: string): void {
  take(value);
  editing.value = null;
  error.value = '';
  done.value = message;
}

/** "strata.farm.14bodhi.com", "10.10.10.50:8080". */
function hostOf(address: string): string {
  try {
    return new URL(address).host;
  } catch {
    return address;
  }
}

function testText(provider: AiProvider): string {
  if (testing.value === provider.id) return 'Testing… it can take two minutes.';
  const test: AiTestResult | null = provider.last_test;
  if (test === null) return 'Not tested yet';
  return test.message + (test.at ? ' · ' + isoWhen(test.at) : '');
}

function testClass(provider: AiProvider): string {
  if (testing.value === provider.id || provider.last_test === null) return '';
  return provider.last_test.ok ? 'ok' : 'bad';
}

async function test(provider: AiProvider): Promise<void> {
  if (testing.value !== null) return;
  testing.value = provider.id;
  done.value = '';
  error.value = '';
  try {
    const answer = await post<{ok: true; result: AiTestResult; provider: AiProvider}>('/api/settings/ai/test', {id: provider.id});
    const list = settings.value?.providers;
    const index = list?.findIndex((item) => item.id === provider.id) ?? -1;
    if (list && index >= 0) list[index] = answer.provider;
  } catch (problem) {
    error.value = provider.name + ': ' + messageOf(problem);
  } finally {
    testing.value = null;
  }
}

/** Only a provider whose last Test passed may review: its saves are final. */
function passed(provider: AiProvider): boolean {
  return provider.last_test?.ok === true;
}

/** Under a provider of "Who reviews": why it cannot be chosen, or why the one chosen cannot review now. */
function pickNote(provider: AiProvider): string {
  if (testing.value === provider.id) return 'Testing…';
  if (passed(provider)) return provider.model;
  if (provider.last_test === null) return 'Test it first';
  return provider.is_reviewer ? 'Didn’t pass its last Test: it can’t review until it does' : 'Didn’t pass its last Test';
}

/** "Who reviews": the site asks this provider (null: none). Choosing the one already chosen does nothing. */
async function choose(id: string | null): Promise<void> {
  if (choosing.value || (settings.value?.reviewer ?? null) === id) return;
  choosing.value = true;
  done.value = '';
  error.value = '';
  try {
    take(await post<AiSettings & {ok: true}>('/api/settings/ai/reviewer', {id}));
  } catch (problem) {
    error.value = messageOf(problem);
  } finally {
    choosing.value = false;
  }
}

function open(provider: AiProvider | 'new'): void {
  done.value = '';
  editing.value = provider;
}

onMounted(async () => {
  await load();
  // The Account menu's "AI reviewer" (/account#ai-reviewer): the card is there only now.
  if (location.hash === '#ai-reviewer') {
    await nextTick();
    document.getElementById('ai-reviewer')?.scrollIntoView({block: 'start'});
  }
});
</script>

<template>
  <section class="card state-box ai-card" id="ai-reviewer" aria-labelledby="ai-title">
    <h2 id="ai-title">AI reviewer</h2>
    <p>A model that reviews the signs: ask it about a frame on the review page, or let it review a whole drive from the
      route list. Any OpenAI-compatible server that sees images works.</p>
    <p v-if="settings === null && !error" class="ai-loading">Loading…</p>
    <template v-if="settings !== null">
      <ul class="ios-list" id="ai-providers">
        <li v-for="provider in settings.providers" :key="provider.id" class="ai-prov" :class="{reviewer: provider.is_reviewer}"
            :data-provider="provider.id">
          <component :is="canEdit ? 'button' : 'div'" class="ai-prov-main" :type="canEdit ? 'button' : undefined"
                     :aria-label="canEdit ? 'Change ' + provider.name : undefined" @click="canEdit && open(provider)">
            <span class="ai-prov-name">{{ provider.name }}
              <span class="ai-key" :class="{on: provider.has_key}"><span class="ai-key-dot" aria-hidden="true"></span>{{ provider.has_key ? 'Key' : 'No key' }}</span></span>
            <span class="ai-prov-sub">{{ provider.model }} · {{ hostOf(provider.base_url) }}</span>
            <span class="ai-prov-test" :class="testClass(provider)" role="status">{{ testText(provider) }}</span>
          </component>
          <div class="ai-prov-actions">
            <span v-if="provider.is_reviewer" class="pill ok">Reviews</span>
            <button v-if="canEdit" type="button" class="btn small" :disabled="testing !== null" @click="test(provider)">{{ testing === provider.id ? 'Testing…' : 'Test' }}</button>
          </div>
        </li>
        <li v-if="canEdit && settings.providers.length < MAX_PROVIDERS" class="ai-prov ai-add">
          <button type="button" class="ai-prov-main" id="ai-add" @click="open('new')"><span class="ai-plus" aria-hidden="true">+</span>Add a provider</button>
        </li>
        <li v-if="settings.providers.length === 0 && !canEdit" class="ai-prov"><span class="ai-prov-main">No AI provider yet.</span></li>
      </ul>
      <template v-if="canEdit && settings.providers.length > 0">
        <h3 class="ai-group-title" id="ai-who-title">Who reviews</h3>
        <div class="ios-list ai-picks" id="ai-who" role="radiogroup" aria-labelledby="ai-who-title">
          <button v-for="provider in settings.providers" :key="provider.id" type="button" role="radio" class="ai-model-row ai-pick"
                  :data-pick="provider.id" :aria-checked="provider.is_reviewer"
                  :disabled="choosing || (!provider.is_reviewer && !passed(provider))" @click="choose(provider.id)">
            <span class="ai-check" aria-hidden="true"></span>
            <span class="ai-pick-text"><span class="ai-model-id">{{ provider.name }}</span>
              <span class="ai-pick-note" :class="{bad: provider.is_reviewer && !passed(provider) && testing !== provider.id}">{{ pickNote(provider) }}</span></span>
          </button>
          <button type="button" role="radio" class="ai-model-row ai-pick" data-pick="none" :aria-checked="settings.reviewer === null"
                  :disabled="choosing" @click="choose(null)">
            <span class="ai-check" aria-hidden="true"></span>
            <span class="ai-pick-text"><span class="ai-model-id">Don’t use AI</span></span>
          </button>
        </div>
        <p class="ai-help">Only a provider that passed its Test can review.</p>
      </template>
      <p v-if="!canEdit && settings.providers.length > 0 && settings.reviewer === null" class="note">No provider reviews the signs.</p>
      <p class="note">What the AI saves is recorded as <code>ai/&lt;provider&gt;/&lt;model&gt;</code> and is final: no person approves it
        again. Keys stay on the server, never sent back to a browser.</p>
    </template>
    <div class="auth-error" role="alert">{{ error }}</div>
    <div class="auth-ok" role="status">{{ done }}</div>
    <ProviderSheet :provider="sheetProvider" :open="editing !== null" @close="editing = null" @saved="saved"/>
  </section>
</template>
