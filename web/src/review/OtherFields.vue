<script setup lang="ts">
// "Altro": physical site and encounter of the frame, the JSONL export, where the reviews are kept, what the
// proposals are.
import {setFrameField} from './edits';
import {exportJsonl} from './export';
import {vFill} from './fill';
import {report, state} from './store';
import type {FrameEdit, FrameRow} from './types';

defineProps<{row: FrameRow; work: FrameEdit}>();

const SERVER_NOTE = 'Ogni conferma viene salvata nel database del server. Se il salvataggio non riesce, il fotogramma '
  + 'resta segnato “non salvato” e viene ritentato; nel frattempo le modifiche restano in questo browser. Il JSONL '
  + 'resta disponibile come copia di sicurezza. Un frame senza riquadri non è automaticamente un esempio negativo: '
  + 'va controllato e confermato.';
const OFFLINE_NOTE = 'Le modifiche sono salvate in questo browser. Esporta il JSONL per conservarle fuori dal browser; '
  + 'il file originale delle proposte resta invariato. Un frame senza riquadri non è automaticamente un esempio '
  + 'negativo: va controllato e confermato.';

</script>

<template>
  <section>
    <details>
      <summary>Altro</summary>
      <div class="stack">
        <!-- The frame's own physical site and encounter, else the report's. -->
        <label>Sito fisico
          <input v-fill="work.site_group ?? row.site_group ?? ''" type="text" id="site_group"
                 placeholder="Stesso identificativo nelle altre route" autocomplete="off" @change="setFrameField(row, 'site_group', ($event.target as HTMLInputElement).value)">
        </label>
        <label>Incontro
          <input v-fill="work.encounter_id ?? row.encounter_id ?? ''" type="text" id="encounter_id"
                 placeholder="Stesso identificativo nei frame vicini" autocomplete="off" @change="setFrameField(row, 'encounter_id', ($event.target as HTMLInputElement).value)">
        </label>
        <div class="row">
          <button type="button" class="btn small" id="export" @click="exportJsonl">Scarica revisioni JSONL</button>
          <span :class="state.exportMessage ? 'save-msg ' + (state.exportMessage.ok ? 'ok' : 'error') : 'save-msg'"
                id="exportMsg" role="status">{{ state.exportMessage?.text }}</span>
        </div>
        <p id="storageNote">{{ report.apiBase ? SERVER_NOTE : OFFLINE_NOTE }}</p>
        <p>Proposte di cartelli da un modello addestrato in Vietnam e numeri sull’asfalto da OCR verificato con segmentazione della strada. Modelli non validati sulle strade svizzere. Conferma solo ciò che riesci a vedere. Il numero proposto non è il limite applicabile alla vettura.</p>
        <p>Lascia sconosciuto ciò che non puoi verificare. L’istante di passaggio resta non annotato. Ogni revisione viene controllata con le regole di import-annotations prima di entrare nel dataset.</p>
      </div>
    </details>
  </section>
</template>
