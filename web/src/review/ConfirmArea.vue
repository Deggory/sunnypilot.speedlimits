<script setup lang="ts">
// The confirm block at the bottom of the inspector: warnings (server, browser storage, frames not saved), the
// save line of the frame with Annulla modifiche / Annulla conferma, Conferma and the keyboard shortcuts.
import {computed} from 'vue';
import {status} from './current';
import {confirmFrame, discardChanges, withdrawFrame} from './confirm';
import {releaseFocus} from './dom';
import {confirmLabels, pageNotices, statusLine} from './messages';
import {retryUnsaved} from './server';
import {report} from './store';
import type {FrameEdit, FrameRow} from './types';

const props = defineProps<{row: FrameRow; work: FrameEdit}>();
const notices = computed(() => pageNotices());
const line = computed(() => statusLine(props.row.frame_key, status.value));
const labels = computed(() => confirmLabels(props.row.frame_key, props.work, status.value));
// Nothing to say under the panel: on a phone that section folds away instead of leaving an empty block.
const quiet = computed(() => line.value.text === '' && notices.value.length === 0 && !line.value.discard
  && !line.value.withdraw);
</script>

<template>
  <section class="confirm-section" :class="{quiet}">
    <div class="notices" id="notices">
      <div v-for="(notice, i) in notices" :key="i" :class="notice.error ? 'notice error' : 'notice'" role="alert">
        <span>{{ notice.text }}</span>
        <button v-if="notice.retry" type="button" class="btn small" @click="retryUnsaved(true); releaseFocus($event)">Riprova a salvare</button>
      </div>
    </div>
    <div class="confirm-area">
      <div class="status-line">
        <p :class="line.cls ? 'save-msg ' + line.cls : 'save-msg'" id="saveMsg" role="status" :title="line.text">{{ line.text }}</p>
        <div class="status-actions">
          <button type="button" class="link" id="discard" :hidden="!line.discard" @click="discardChanges">Annulla modifiche</button>
          <button type="button" class="link danger" id="withdraw" :hidden="!line.withdraw" :data-armed="line.armed"
                  @click="withdrawFrame">{{ line.armed ? 'Sì, annulla' : 'Annulla conferma' }}</button>
        </div>
      </div>
      <button type="button" class="btn primary confirm" :class="{already: labels.already}" id="confirm"
              data-action="confirm" :disabled="status === 'saving'" :data-save-state="status"
              @click="confirmFrame"><span class="btn-label" id="confirmLabel">{{ labels.label }}</span><kbd class="keycap" aria-hidden="true">↵</kbd></button>
      <p class="keys" aria-label="Scorciatoie da tastiera"><span><kbd>←</kbd> <kbd>→</kbd> fotogramma</span> <span><kbd>Invio</kbd> conferma</span> <span><kbd>N</kbd> nuovo riquadro</span> <span><kbd>Canc</kbd> elimina</span> <span id="keyHints" :hidden="!report.routeHasHints"><kbd>H</kbd> seconda lettura</span> <span><kbd>Esc</kbd> annulla</span></p>
    </div>
  </section>
</template>
