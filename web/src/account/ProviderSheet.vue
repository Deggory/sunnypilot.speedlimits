<script setup lang="ts">
// Adding or changing an AI provider of Settings, in a bottom sheet: its name, the address of its OpenAI-compatible
// server (…/v1), its API key and the model, picked from what the server lists (POST /api/settings/ai/models) or
// typed when it lists nothing. The key goes to the server only: the page never gets it back, it learns only that
// one is saved ("Saved (leave empty to keep)"), and "Remove key" removes it on Save. Delete removes the provider.
import {computed, ref, useTemplateRef, watch} from 'vue';
import {messageOf, post} from '../common/api';
import {sentence} from '../common/format';
import type {AiModel, AiProvider, AiProviderBody, AiSettings} from '../common/ai-types';
import {askConfirm} from '../composables/confirm';
import {useSheet} from '../composables/sheet';

/** provider: the one to change, null to add one; open: the sheet is shown. */
const props = defineProps<{provider: AiProvider | null; open: boolean}>();
const emit = defineEmits<{close: []; saved: [settings: AiSettings, message: string]}>();

const name = ref('');
const baseUrl = ref('');
const key = ref('');
const removeKey = ref(false);
const model = ref('');
const models = ref<AiModel[] | null>(null);
const listing = ref(false);
const listError = ref('');
const error = ref('');
const busy = ref(false);
const panel = useTemplateRef<HTMLElement>('panel');
let listTicket = 0;

const isOpen = computed(() => props.open);
useSheet(isOpen, panel, () => emit('close'));

watch(() => props.open, (open) => {
  if (!open) {
    // The key typed never stays in the page.
    key.value = '';
    return;
  }
  const value = props.provider;
  name.value = value?.name ?? '';
  baseUrl.value = value?.base_url ?? '';
  model.value = value?.model ?? '';
  key.value = '';
  removeKey.value = false;
  models.value = null;
  listError.value = '';
  error.value = '';
  // A saved provider: what its server offers, at once.
  if (value !== null) void listModels();
});

const keyPlaceholder = computed(() => {
  if (removeKey.value) return 'Removed when you save';
  return props.provider?.has_key ? 'Saved (leave empty to keep)' : 'None';
});

/** The models the server lists, the ones that see images first. */
const sortedModels = computed(() => [...(models.value ?? [])]
  .sort((a, b) => Number(b.sees_images === true) - Number(a.sees_images === true)));

function seesText(item: AiModel): string {
  return item.sees_images === true ? 'Sees images' : item.sees_images === false ? 'Text only' : '';
}

async function listModels(): Promise<void> {
  const address = baseUrl.value.trim();
  if (!address && props.provider === null) {
    listError.value = 'Type the address of the server first.';
    return;
  }
  const ticket = ++listTicket;
  listing.value = true;
  listError.value = '';
  // The saved key is used by the server only for the saved address: a new address needs the key typed again.
  const body: {id?: string; base_url?: string; key?: string} = {};
  if (props.provider !== null) body.id = props.provider.id;
  if (address) body.base_url = address;
  if (key.value && !removeKey.value) body.key = key.value;
  try {
    const answer = await post<{ok: true; models: AiModel[]}>('/api/settings/ai/models', body);
    if (ticket !== listTicket) return;
    models.value = answer.models;
    const first = sortedModels.value[0];
    if (!model.value && first) model.value = first.id;
    if (answer.models.length === 0) listError.value = 'The server lists no model: type its name.';
  } catch (problem) {
    if (ticket !== listTicket) return;
    models.value = null;
    listError.value = 'Models not listed. ' + sentence(messageOf(problem)) + ' Type the model name instead.';
  } finally {
    if (ticket === listTicket) listing.value = false;
  }
}

async function save(): Promise<void> {
  error.value = '';
  const body: AiProviderBody = {name: name.value.trim(), base_url: baseUrl.value.trim(), model: model.value.trim()};
  if (!body.name) { error.value = 'Give it a name.'; return; }
  if (!body.base_url) { error.value = 'Type the address of the server, ending with /v1.'; return; }
  if (!body.model) { error.value = 'Choose or type the model.'; return; }
  if (props.provider !== null) body.id = props.provider.id;
  if (removeKey.value) body.key = '';
  else if (key.value) body.key = key.value;
  busy.value = true;
  try {
    const answer = await post<AiSettings & {ok: true; provider: AiProvider}>('/api/settings/ai/providers', body);
    key.value = '';
    emit('saved', {providers: answer.providers, reviewer: answer.reviewer, can_edit: answer.can_edit},
      'Saved ' + answer.provider.name + (props.provider === null ? ': press Test to check it sees the frames.' : '.'));
  } catch (problem) {
    error.value = messageOf(problem);
  } finally {
    busy.value = false;
  }
}

async function remove(): Promise<void> {
  const value = props.provider;
  if (value === null) return;
  const ok = await askConfirm({title: 'Delete ' + value.name + '?', action: 'Delete', danger: true,
    text: 'Its address and key are removed from the server' + (value.is_reviewer ? ', and no AI reviews until you choose another' : '')
      + '. The reviews it saved stay.'});
  if (!ok) return;
  busy.value = true;
  try {
    const answer = await post<AiSettings & {ok: true}>('/api/settings/ai/providers/' + encodeURIComponent(value.id) + '/delete');
    emit('saved', {providers: answer.providers, reviewer: answer.reviewer, can_edit: answer.can_edit}, 'Deleted ' + value.name + '.');
  } catch (problem) {
    error.value = messageOf(problem);
  } finally {
    busy.value = false;
  }
}
</script>

<template>
  <div class="sheet-backdrop" :class="{open}" @click="emit('close')"></div>
  <section ref="panel" class="sheet ai-sheet" :class="{open}" role="dialog" aria-modal="true" aria-labelledby="provider-title"
           :aria-hidden="!open" :inert="!open">
    <div class="sheet-grabber" aria-hidden="true"></div>
    <div class="sheet-head">
      <div class="grow"><h2 id="provider-title">{{ provider ? provider.name : 'Add an AI provider' }}</h2>
        <p class="sheet-sub">Any server that speaks the OpenAI API: llama.cpp, vLLM, OpenAI.</p></div>
      <button class="btn small" type="button" @click="emit('close')">Cancel</button>
    </div>
    <form class="sheet-body ai-form" novalidate @submit.prevent="save">
      <div class="ai-fields">
        <label class="ai-field"><span>Name</span>
          <input id="provider-name" v-model="name" type="text" maxlength="40" autocomplete="off" spellcheck="false" placeholder="strata"></label>
        <label class="ai-field"><span>Address</span>
          <input id="provider-url" v-model="baseUrl" type="url" inputmode="url" maxlength="200" autocomplete="off" autocapitalize="off"
                 spellcheck="false" placeholder="https://host:8080/v1"></label>
        <div class="ai-field">
          <label for="provider-key">API key</label>
          <div class="ai-key-line">
            <input id="provider-key" v-model="key" type="password" maxlength="512" autocomplete="off" autocapitalize="off" spellcheck="false"
                   data-1p-ignore data-lpignore="true" :disabled="removeKey" :placeholder="keyPlaceholder">
            <button v-if="provider?.has_key" type="button" class="ai-link" :class="{danger: !removeKey}" id="provider-remove-key"
                    @click="removeKey = !removeKey; key = ''">{{ removeKey ? 'Keep key' : 'Remove key' }}</button>
          </div>
        </div>
      </div>
      <p class="ai-help">The key stays on the server: this page only learns whether one is saved. Leave it empty for a server
        without one (llama.cpp started without <code>--api-key</code>).</p>

      <h3 class="sheet-section">Model</h3>
      <div class="ai-fields">
        <div class="ai-field">
          <label for="provider-model">Model ID</label>
          <div class="ai-key-line">
            <input id="provider-model" v-model="model" type="text" maxlength="200" autocomplete="off" autocapitalize="off"
                   spellcheck="false" placeholder="qwen3-vl-32b-instruct">
            <button type="button" class="ai-link" id="provider-list" :disabled="listing" @click="listModels">{{ listing ? 'Listing…' : models ? 'List again' : 'List models' }}</button>
          </div>
        </div>
      </div>
      <p v-if="listError" class="ai-help ai-warn-text" role="status">{{ listError }}</p>
      <div v-if="sortedModels.length > 0" class="ai-models" role="radiogroup" aria-label="Models on the server">
        <button v-for="item in sortedModels" :key="item.id" type="button" role="radio" class="ai-model-row"
                :aria-checked="model === item.id" @click="model = item.id">
          <span class="ai-check" aria-hidden="true"></span><span class="ai-model-id">{{ item.id }}</span>
          <span v-if="seesText(item)" class="ai-tag" :class="{text: item.sees_images === false}">{{ seesText(item) }}</span>
        </button>
      </div>
      <p class="ai-help">It must see images: a vision model such as Qwen3-VL (llama.cpp with its <code>--mmproj</code>).</p>

      <div class="auth-error" role="alert">{{ error }}</div>
      <div class="ai-sheet-actions">
        <button v-if="provider" type="button" class="btn danger" id="provider-delete" :disabled="busy" @click="remove">Delete</button>
        <span class="grow"></span>
        <button type="button" class="btn" @click="emit('close')">Cancel</button>
        <button type="submit" class="btn primary" id="provider-save" :disabled="busy">{{ busy ? 'Saving…' : 'Save' }}</button>
      </div>
    </form>
  </section>
</template>
