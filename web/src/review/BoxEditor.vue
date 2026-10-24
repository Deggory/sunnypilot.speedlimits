<script setup lang="ts">
// The editor of the selected box: zoom (its corners move the box too), speed signs, other value, type, road,
// details.
import {computed, ref, useTemplateRef, watchEffect} from 'vue';
import {BOOLEAN_FIELDS, FAMILIES, QUICK_VALUES, ROADS, SELECT_FIELDS, familyHasValue, type BooleanField,
  type SelectField} from './catalog';
import {coarsePointer, releaseFocus} from './dom';
import {setFamily, setField, setValue} from './edits';
import {vFill} from './fill';
import {handlesOf, sceneScale, zoomView, type Rect} from './overlay';
import {finishDrag, onPointerDown, onPointerMove, onTouchStart} from './pointer';
import {state} from './store';
import type {Detection, FrameRow} from './types';
import {useScreenSize} from './use-layout';

const props = defineProps<{row: FrameRow; detection: Detection | null}>();

const zoom = useTemplateRef<SVGSVGElement>('zoom');
const zoomSize = useScreenSize(zoom);

// While a corner of the zoom is dragged its view stays still, so the corner stays under the finger.
const view = ref<Rect | null>(null);
watchEffect(() => {
  const d = props.detection;
  if (d === null || (state.drag !== null && state.drag.svg === 'zoom')) return;
  view.value = zoomView(d.bbox_xyxy, props.row.width, props.row.height);
});
const viewBox = computed(() => {
  const v = view.value;
  return v === null ? undefined : v.x + ' ' + v.y + ' ' + v.width + ' ' + v.height;
});
const handles = computed(() => {
  const d = props.detection;
  const v = view.value;
  if (d === null) return [];
  // The SVG keeps its view box in single precision.
  const kz = v && v.width && v.height ? sceneScale(zoomSize.value, Math.fround(v.width), Math.fround(v.height)) : 1;
  return handlesOf(d.bbox_xyxy, 12 / kz, (coarsePointer() ? 36 : 22) / kz, false);
});
const box = computed(() => {
  const b = props.detection?.bbox_xyxy;
  return b ? {x: b[0], y: b[1], width: b[2] - b[0], height: b[3] - b[1]} : null;
});
const resizing = computed(() => state.drag?.kind === 'resize' && state.drag.svg === 'zoom');

const usual = FAMILIES.filter((family) => !family.minor);
const minor = FAMILIES.filter((family) => family.minor);
const hasValueField = computed(() => props.detection !== null && familyHasValue(props.detection.family));
const noValue = computed(() => hasValueField.value
  && (props.detection?.value_kph === null || props.detection?.value_kph === undefined));
const road = computed(() => props.detection?.applicability || 'unresolved');

const SELECT_CHOICES: Record<SelectField, [string, string][]> = {
  visibility: [['unknown', 'Da valutare'], ['clear', 'Chiara'], ['partial', 'Parziale'], ['poor', 'Scarsa']],
  occlusion: [['unknown', 'Da valutare'], ['none', 'Assente'], ['partial', 'Parziale'], ['heavy', 'Forte']],
  supplementary_panel: [['unknown', 'Da valutare'], ['none', 'Assente'], ['present', 'Presente'],
    ['unresolved', 'Non risolto']],
};
const SELECT_LABELS: Record<SelectField, string> = {
  visibility: 'Visibilità', occlusion: 'Occlusione', supplementary_panel: 'Pannello sotto',
};
const BOOLEAN_LABELS: Record<BooleanField, string> = {temporary: 'Temporaneo', variable_display: 'Display variabile'};

function selectValue(key: SelectField): string {
  return props.detection?.[key] ?? 'unknown';
}
function booleanValue(key: BooleanField): string {
  const value = props.detection?.[key];
  return value === true ? 'true' : value === false ? 'false' : 'unknown';
}

function valueChanged(event: Event): void {
  const raw = (event.target as HTMLInputElement).value.trim();
  if (raw === '') {
    setValue(null);
    return;
  }
  const value = Number(raw);
  if (!Number.isInteger(value) || value < 5 || value > 250) {
    state.valueError = true;
    return;
  }
  setValue(value);
}
function selectChanged(key: SelectField, event: Event): void {
  setField(key, (event.target as HTMLSelectElement).value);
}
function booleanChanged(key: BooleanField, event: Event): void {
  const value = (event.target as HTMLSelectElement).value;
  setField(key, value === 'true' ? true : value === 'false' ? false : null);
}
</script>

<template>
  <section id="editor" :hidden="detection === null">
    <div class="section-head"><h2 id="editorTitle">Riquadro {{ state.selected + 1 }}</h2></div>
    <div class="zoom-wrap">
      <svg ref="zoom" id="zoom" :class="{dragging: resizing}" xmlns="http://www.w3.org/2000/svg"
           preserveAspectRatio="xMidYMid meet" aria-label="Ingrandimento del riquadro selezionato" :viewBox="viewBox"
           @pointerdown="onPointerDown($event, 'zoom')" @pointermove="onPointerMove($event, 'zoom')"
           @pointerup="finishDrag($event, 'zoom')" @pointercancel="finishDrag($event, 'zoom')"
           @touchstart="onTouchStart($event, 'zoom')">
        <image id="zoomPhoto" x="0" y="0" preserveAspectRatio="none" :href="row.preview_path" :width="row.width"
               :height="row.height"></image>
        <rect id="zoomBox" :x="box?.x" :y="box?.y" :width="box?.width" :height="box?.height"></rect>
        <g id="zoomHandles">
          <rect v-for="handle in handles" :key="'mark-' + handle.corner" :x="handle.mark.x" :y="handle.mark.y"
                :width="handle.mark.width" :height="handle.mark.height" class="handle"/>
          <rect v-for="handle in handles" :key="'hit-' + handle.corner" :x="handle.hit.x" :y="handle.hit.y"
                :width="handle.hit.width" :height="handle.hit.height" :class="'handle-hit ' + handle.corner"
                :data-corner="handle.corner"/>
        </g>
      </svg>
    </div>
    <p class="evidence" id="evidence">{{ detection?.model_label ? 'Proposta del modello: ' + detection.model_label : 'Riquadro aggiunto a mano.' }}</p>
    <!-- The signs are always there: a number makes the box a speed limit, "?" means the number cannot be read. -->
    <div class="signs" id="quickValues" role="group" aria-label="Velocità">
      <button v-for="value in QUICK_VALUES" :key="value" type="button" :class="value >= 100 ? 'sign wide' : 'sign'"
              :aria-pressed="detection?.value_kph === value" :data-value="value" :aria-label="value + ' km/h'"
              @click="setValue(value); releaseFocus($event)">{{ value }}</button>
      <button type="button" class="sign unknown" :aria-pressed="noValue" data-value="unknown" title="Numero non leggibile"
              aria-label="Numero non leggibile" @click="setValue(null); releaseFocus($event)">?</button>
    </div>
    <div id="valueBlock" class="value-line" :hidden="!hasValueField">
      <label for="value">Altro valore</label>
      <input v-fill="detection?.value_kph === null || detection?.value_kph === undefined ? '' : String(detection.value_kph)"
             type="number" id="value" min="5" max="250" step="1" inputmode="numeric" placeholder="–" @change="valueChanged">
      <span>km/h</span>
    </div>
    <p class="field-error" id="valueError" :hidden="!state.valueError">Scrivi un numero intero tra 5 e 250, oppure lascia vuoto se non si legge.</p>
    <p class="field-label">Tipo</p>
    <div class="chips" id="families" role="group" aria-label="Tipo di cartello">
      <button v-for="family in usual" :key="family.id" type="button" class="chip"
              :aria-pressed="detection?.family === family.id" :data-family="family.id"
              @click="setFamily(family.id); releaseFocus($event)">{{ family.label }}</button>
    </div>
    <div class="chips minor" id="familiesMore" role="group" aria-label="Altri tipi">
      <button v-for="family in minor" :key="family.id" type="button" class="chip"
              :aria-pressed="detection?.family === family.id" :data-family="family.id"
              :title="family.id === 'not_a_sign' ? 'Resta come esempio negativo; per togliere un riquadro sbagliato usa ✕' : undefined"
              @click="setFamily(family.id); releaseFocus($event)">{{ family.label }}</button>
    </div>
    <p class="field-label">Strada</p>
    <div class="segmented" id="roads" role="group" aria-label="Su quale strada vale">
      <button v-for="item in ROADS" :key="item.id" type="button" :aria-pressed="road === item.id"
              :data-applicability="item.id" @click="setField('applicability', item.id); releaseFocus($event)">{{ item.label }}</button>
    </div>
    <details>
      <summary>Dettagli del cartello</summary>
      <div class="grid2">
        <label v-for="key in SELECT_FIELDS" :key="key">{{ SELECT_LABELS[key] }}
          <select :id="key" v-fill="selectValue(key)" @change="selectChanged(key, $event)">
            <option v-for="[value, text] in SELECT_CHOICES[key]" :key="value" :value="value">{{ text }}</option>
          </select>
        </label>
        <label v-for="key in BOOLEAN_FIELDS" :key="key">{{ BOOLEAN_LABELS[key] }}
          <select :id="key" v-fill="booleanValue(key)" @change="booleanChanged(key, $event)">
            <option value="unknown">Non determinato</option>
            <option value="true">Sì</option>
            <option value="false">No</option>
          </select>
        </label>
        <label>Note
          <input v-fill="detection?.notes ?? ''" type="text" id="notes" autocomplete="off"
                 @change="setField('notes', ($event.target as HTMLInputElement).value)">
        </label>
      </div>
    </details>
  </section>
</template>
