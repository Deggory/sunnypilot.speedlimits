<script setup lang="ts">
// The phone's bottom bar: previous, Conferma (saying what it saves), next, and the message that must not be
// missed (a save that failed, a warning).
import {computed, useTemplateRef} from 'vue';
import {position, status} from './current';
import {confirmFrame} from './confirm';
import {confirmLabels, statusLine} from './messages';
import {go} from './navigation';
import type {FrameEdit, FrameRow} from './types';
import {useHeightVar} from './use-layout';

const props = defineProps<{row: FrameRow | null; work: FrameEdit | null}>();
const bar = useTemplateRef<HTMLElement>('bar');
useHeightVar(bar, '--mobilebar-h');

const labels = computed(() => (props.row && props.work
  ? confirmLabels(props.row.frame_key, props.work, status.value) : {label: 'Conferma', mobileLabel: 'Conferma', already: false}));
const message = computed(() => (props.row ? statusLine(props.row.frame_key, status.value).mobile : ''));
</script>

<template>
  <nav ref="bar" class="mobilebar" id="mobilebar" aria-label="Navigazione" :hidden="row === null">
    <p class="mobile-msg" id="mobileMsg" role="alert" :hidden="message === ''">{{ message }}</p>
    <div class="mobile-buttons">
      <button type="button" class="btn" id="mPrev" aria-label="Fotogramma precedente" :disabled="position.index <= 0"
              @click="go(-1)">‹</button>
      <button type="button" class="btn primary confirm" :class="{already: labels.already}" id="mConfirm" data-action="confirm"
              :disabled="status === 'saving'" :data-save-state="status" @click="confirmFrame"><span class="btn-label" id="mConfirmLabel">{{ labels.mobileLabel }}</span><kbd class="keycap" aria-hidden="true">↵</kbd></button>
      <button type="button" class="btn" id="mNext" aria-label="Fotogramma successivo"
              :disabled="position.index >= position.length - 1" @click="go(1)">›</button>
    </div>
  </nav>
</template>
