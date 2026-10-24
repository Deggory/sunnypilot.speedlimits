// The editor of the selected box: zoom, speed signs, other value, type, road, details; and the fields of the
// frame under "Altro" (physical site, encounter).
import {byId, node} from '../common/dom';
import {BOOLEAN_FIELDS, FAMILIES, QUICK_VALUES, ROADS, SELECT_FIELDS, familyHasValue} from './catalog';
import {one} from './dom';
import {changed, setFamily, setField, setValue} from './edits';
import {currentRow, workOf} from './frames';
import {renderZoom} from './overlay';
import {state} from './state';
import type {Detection, FrameEdit, FrameRow} from './types';

type Field = HTMLInputElement | HTMLSelectElement;

/** A field the reviewer is typing in keeps what is typed: only the others follow the data. */
function fill(input: Field, value: string): void {
  if (document.activeElement !== input) input.value = value;
}

function choice(className: string, text: string, pressed: boolean, onClick: () => void): HTMLButtonElement {
  const button = node('button', className, text);
  button.type = 'button';
  button.setAttribute('aria-pressed', pressed ? 'true' : 'false');
  button.addEventListener('click', onClick);
  return button;
}

function renderFamilies(d: Detection): void {
  const usual = byId('families');
  const more = byId('familiesMore');
  usual.replaceChildren();
  more.replaceChildren();
  for (const family of FAMILIES) {
    const chip = choice('chip', family.label, d.family === family.id, () => setFamily(family.id));
    chip.dataset['family'] = family.id;
    if (family.id === 'not_a_sign') {
      chip.title = 'Resta come esempio negativo; per togliere un riquadro sbagliato usa ✕';
    }
    (family.minor ? more : usual).append(chip);
  }
}

// The signs are always there: a number makes the box a speed limit, "?" means the number cannot be read.
function renderValues(d: Detection): void {
  byId('valueBlock').hidden = !familyHasValue(d.family);
  const quick = byId('quickValues');
  quick.replaceChildren();
  for (const value of QUICK_VALUES) {
    const pressed = d.value_kph === value;
    const sign = choice(value >= 100 ? 'sign wide' : 'sign', String(value), pressed, () => setValue(value));
    sign.dataset['value'] = String(value);
    sign.setAttribute('aria-label', value + ' km/h');
    quick.append(sign);
  }
  const noValue = familyHasValue(d.family) && (d.value_kph === null || d.value_kph === undefined);
  const unknown = choice('sign unknown', '?', noValue, () => setValue(null));
  unknown.dataset['value'] = 'unknown';
  unknown.title = 'Numero non leggibile';
  unknown.setAttribute('aria-label', 'Numero non leggibile');
  quick.append(unknown);
  fill(byId<HTMLInputElement>('value'), d.value_kph === null || d.value_kph === undefined ? '' : String(d.value_kph));
}

function renderRoads(d: Detection): void {
  const roads = byId('roads');
  roads.replaceChildren();
  const current = d.applicability ? d.applicability : 'unresolved';
  for (const road of ROADS) {
    const chip = choice('', road.label, current === road.id, () => setField('applicability', road.id));
    chip.dataset['applicability'] = road.id;
    roads.append(chip);
  }
}

function renderDetails(d: Detection): void {
  for (const key of SELECT_FIELDS) fill(byId<HTMLSelectElement>(key), d[key] ?? 'unknown');
  for (const key of BOOLEAN_FIELDS) {
    const value = d[key];
    fill(byId<HTMLSelectElement>(key), value === true ? 'true' : value === false ? 'false' : 'unknown');
  }
  fill(byId<HTMLInputElement>('notes'), d.notes ?? '');
}

export function renderEditor(row: FrameRow, work: FrameEdit): void {
  const d = work.detections[state.selected];
  one('.panel').classList.toggle('editing', d !== undefined);
  byId('editor').hidden = d === undefined;
  byId<HTMLButtonElement>('remove').disabled = d === undefined;
  if (d === undefined) return;
  byId('editorTitle').textContent = 'Riquadro ' + (state.selected + 1);
  renderZoom(row, d);
  renderFamilies(d);
  renderValues(d);
  renderRoads(d);
  renderDetails(d);
  byId('evidence').textContent = d.model_label ? 'Proposta del modello: ' + d.model_label : 'Riquadro aggiunto a mano.';
}

/** Physical site and encounter of the frame (under "Altro"): the frame's own, else the report's. */
export function renderFrameFields(row: FrameRow, work: FrameEdit): void {
  fill(byId<HTMLInputElement>('site_group'), work.site_group ?? row.site_group ?? '');
  fill(byId<HTMLInputElement>('encounter_id'), work.encounter_id ?? row.encounter_id ?? '');
}

export function initEditor(): void {
  const valueInput = byId<HTMLInputElement>('value');
  valueInput.addEventListener('change', () => {
    const raw = valueInput.value.trim();
    if (raw === '') {
      setValue(null);
      return;
    }
    const value = Number(raw);
    if (!Number.isInteger(value) || value < 5 || value > 250) {
      const error = byId('valueError');
      error.textContent = 'Scrivi un numero intero tra 5 e 250, oppure lascia vuoto se non si legge.';
      error.hidden = false;
      return;
    }
    setValue(value);
  });
  for (const key of SELECT_FIELDS) {
    const select = byId<HTMLSelectElement>(key);
    select.addEventListener('change', () => setField(key, select.value));
  }
  for (const key of BOOLEAN_FIELDS) {
    const select = byId<HTMLSelectElement>(key);
    select.addEventListener('change', () => {
      setField(key, select.value === 'true' ? true : select.value === 'false' ? false : null);
    });
  }
  const notes = byId<HTMLInputElement>('notes');
  notes.addEventListener('change', () => setField('notes', notes.value));
  for (const key of ['site_group', 'encounter_id'] as const) {
    const input = byId<HTMLInputElement>(key);
    input.addEventListener('change', () => {
      const row = currentRow();
      if (!row) return;
      const value = input.value.trim();
      workOf(row)[key] = value === '' ? null : value;
      changed(row);
    });
  }
}
