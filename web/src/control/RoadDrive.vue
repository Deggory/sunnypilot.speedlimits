<script setup lang="ts">
// Data › Roads, a drive opened ("/roads?drive=<route>"): its pictures one at a time with the type of road there, the
// map's or a person's (GET /api/panel/road-types/<route>), and the strip of the whole drive under it, coloured as the
// list, a dark line under the stretches a person set; a tap or a drag goes there. Play, the arrows and space step as in
// Results › Drive (DriveView.vue). A person corrects the map: the type of the stretch shown (or from a marked start to
// here), saved at once (POST, the drive comes back); Next to check goes to the next stretch the map is unsure about.
import {computed, onBeforeUnmount, onMounted, ref, shallowRef, useTemplateRef, watch} from 'vue';
import {api, messageOf, post} from '../common/api';
import {plural, thousands} from '../common/format';
import type {RoadDriveView, RoadType} from '../common/panel-types';
import {toast} from '../composables/toast';
import {frameTime, nextToCheck, ROAD_LABEL, roadRuns, toCheck} from './roads';
import Segmented from './Segmented.vue';
import {goTo} from './tabs';
import {SPEEDS, type Speed} from './watch';

const props = defineProps<{route: string; readOnly: boolean}>();

const MARKS = {
  play: 'M8 5.6v12.8a.8.8 0 0 0 1.2.7l10.3-6.4a.8.8 0 0 0 0-1.4L9.2 4.9A.8.8 0 0 0 8 5.6z',
  pause: 'M7 5h3.2v14H7zM13.8 5H17v14h-3.2z',
  back: 'M14.5 5.5 8 12l6.5 6.5',
  forward: 'M9.5 5.5 16 12l-6.5 6.5',
} as const;
/** The choices of a correction, null: the person cannot tell. */
const CHOICES: [key: RoadType | null, label: string][] = [['town', 'Town'], ['rural', 'Rural'], ['expressway', 'Expressway'],
  ['motorway', 'Motorway'], [null, 'Can\'t tell']];

// -- the drive -----------------------------------------------------------------------------------------------------------
const payload = shallowRef<RoadDriveView | null>(null);
const error = ref('');
let asked = 0;
async function read(route: string): Promise<void> {
  const ticket = ++asked;
  try {
    const answer = await api<RoadDriveView>('/api/panel/road-types/' + encodeURIComponent(route));
    if (ticket !== asked) return;
    payload.value = answer;
    error.value = '';
    document.title = 'Speed limits · ' + (answer.title ?? 'Roads');
  } catch (problem) {
    if (ticket === asked) error.value = messageOf(problem);
  }
}
const frames = computed(() => payload.value?.frames ?? []);
const runs = computed(() => roadRuns(frames.value));
const left = computed(() => frames.value.filter(toCheck).length);
const facts = computed(() => {
  const drive = payload.value;
  if (!drive) return '';
  return [drive.when, drive.minutes !== null ? drive.minutes + ' min' : null, plural(drive.frames.length, 'picture'),
    drive.reviewed ? thousands(drive.reviewed) + ' reviewed' : null].filter(Boolean).join(' · ');
});

/** Back to the list: back in the history when the list opened this drive, so it is as it was. */
function back(): void {
  if ((history.state as {fromList?: boolean} | null)?.fromList) history.back();
  else goTo('roads');
}

// -- the frame wanted, and the one shown (its picture decoded first) -----------------------------------------------------
const playing = ref(false);
const speed = ref<Speed>(2);
const cursor = ref(0);
const shown = ref(0);
const failed = ref(false);
const frame = computed(() => frames.value[shown.value] ?? null);
/** Mark start: the first frame of the stretch to correct, up to the one shown. */
const mark = ref<number | null>(null);

let cache = new Map<string, Promise<boolean>>();
function load(url: string): Promise<boolean> {
  let loaded = cache.get(url);
  if (loaded === undefined) {
    const image = new Image();
    image.decoding = 'async';
    image.src = url;
    loaded = image.decode().then(() => true, () => false);
    cache.set(url, loaded);
    if (cache.size > 48) cache.delete(cache.keys().next().value as string);
  }
  return loaded;
}
function showFrame(index: number): void {
  const url = frames.value[index]?.image ?? null;
  if (url === null) {
    shown.value = index;
    failed.value = false;
  } else {
    void load(url).then((ok) => {
      if (cursor.value !== index) return;
      shown.value = index;
      failed.value = !ok;
    });
  }
  // The next pictures (playing, the right arrow) and the two before (the left arrow).
  for (const step of [1, 2, 3, 4, 5, 6, -1, -2]) {
    const near = frames.value[index + step]?.image;
    if (near) void load(near);
  }
}
watch(cursor, showFrame);
watch(() => props.route, (route) => {
  playing.value = false;
  payload.value = null;
  mark.value = null;
  cache = new Map();
  cursor.value = 0;
  void read(route).then(() => showFrame(0));
}, {immediate: true});

function seek(index: number): void {
  cursor.value = Math.min(Math.max(0, frames.value.length - 1), Math.max(0, index));
}
let timer: number | undefined;
function schedule(): void {
  window.clearTimeout(timer);
  if (!playing.value || shown.value !== cursor.value) return;
  timer = window.setTimeout(() => {
    if (cursor.value >= frames.value.length - 1) playing.value = false;
    else seek(cursor.value + 1);
  }, 1000 / speed.value);
}
watch([playing, shown, speed], schedule);
function toggle(): void {
  if (!playing.value && cursor.value >= frames.value.length - 1) seek(0);
  playing.value = !playing.value;
}
function step(by: number): void {
  playing.value = false;
  seek(cursor.value + by);
}
const speedOptions = SPEEDS.map((value) => ({key: String(value), label: (value === 0.5 ? '½' : String(value)) + '×'}));
const speedKey = computed({
  get: (): string => String(speed.value),
  set: (value: string) => { speed.value = (SPEEDS.find((item) => String(item) === value) ?? 2) as Speed; },
});

const root = useTemplateRef<HTMLElement>('root');
function onKey(event: KeyboardEvent): void {
  if (event.altKey || event.ctrlKey || event.metaKey || !root.value || !(event.target instanceof Node)) return;
  const tag = event.target instanceof HTMLElement ? event.target.tagName : '';
  if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return;
  if (event.key === ' ' && tag !== 'BUTTON') {
    event.preventDefault();
    toggle();
  } else if (event.key === 'ArrowRight' || event.key === 'ArrowLeft') {
    event.preventDefault();
    step(event.key === 'ArrowRight' ? 1 : -1);
  }
}
onMounted(() => document.addEventListener('keydown', onKey));
onBeforeUnmount(() => {
  document.removeEventListener('keydown', onKey);
  window.clearTimeout(timer);
});

/** Next to check: the next stretch no person set where the map says nothing or is often wrong. */
function nextCheck(): void {
  const index = nextToCheck(frames.value, cursor.value);
  if (index === null) {
    toast('Nothing else to check on this drive.');
    return;
  }
  step(index - cursor.value);
}

// -- the strip: a tap or a drag goes to that point of the drive ----------------------------------------------------------
const strip = useTemplateRef<SVGSVGElement>('strip');
let dragging = false;
function stripAt(event: PointerEvent): void {
  const box = strip.value?.getBoundingClientRect();
  if (!box || box.width <= 0 || frames.value.length === 0) return;
  playing.value = false;
  seek(Math.floor(((event.clientX - box.left) / box.width) * frames.value.length));
}
function stripDown(event: PointerEvent): void {
  dragging = true;
  (event.currentTarget as Element).setPointerCapture(event.pointerId);
  stripAt(event);
}
function stripMove(event: PointerEvent): void {
  if (dragging) stripAt(event);
}

// -- the frame shown, and its correction -----------------------------------------------------------------------------------
const kind = computed(() => frame.value?.road_type ?? 'unknown');
/** Why the frame has its type, in words. */
const reason = computed((): string => {
  const item = frame.value;
  if (!item) return '';
  if (item.reviewed) return item.road_type === null ? 'A person could not tell.' : 'Set by a person.';
  return 'From the map: ' + item.basis + '.';
});
/** The frames a correction covers: from the mark to here, else the stretch shown (its run of one kind). */
const span = computed((): [number, number] | null => {
  if (!frames.value.length) return null;
  if (mark.value !== null) return [Math.min(mark.value, shown.value), Math.max(mark.value, shown.value)];
  const run = runs.value.find((item) => item.first <= shown.value && shown.value < item.first + item.count);
  return run ? [run.first, run.first + run.count - 1] : null;
});
const spanText = computed((): string => {
  const [first, last] = span.value ?? [0, -1];
  const start = payload.value?.start ?? null;
  const from = frames.value[first];
  const to = frames.value[last];
  if (!from || !to) return '';
  return plural(last - first + 1, 'picture') + ' · ' + frameTime(from.key, start) + (last > first ? ' – ' + frameTime(to.key, start) : '');
});
const saving = ref(false);
async function setType(type: RoadType | null): Promise<void> {
  const [first, last] = span.value ?? [0, -1];
  const from = frames.value[first];
  const to = frames.value[last];
  if (!from || !to || saving.value) return;
  saving.value = true;
  try {
    payload.value = await post<RoadDriveView>('/api/panel/road-types/' + encodeURIComponent(props.route),
      {from: from.key, to: to.key, road_type: type});
    mark.value = null;
    toast((type === null ? 'Can\'t tell' : ROAD_LABEL[type]) + ' · ' + plural(last - first + 1, 'picture') + ' saved');
  } catch (problem) {
    toast(messageOf(problem), true);
  } finally {
    saving.value = false;
  }
}
async function undo(): Promise<void> {
  if (saving.value) return;
  saving.value = true;
  try {
    payload.value = await post<RoadDriveView>('/api/panel/road-types/' + encodeURIComponent(props.route) + '/undo');
    toast('Last change undone');
  } catch (problem) {
    toast(messageOf(problem), true);
  } finally {
    saving.value = false;
  }
}
</script>

<template>
  <section ref="root" class="card rt-view" aria-labelledby="rt-view-title">
    <div class="rt-view-head">
      <button type="button" class="link-button rt-back" @click="back"><span aria-hidden="true">‹</span> Roads</button>
      <div class="rt-view-name">
        <h2 id="rt-view-title">{{ payload ? payload.title ?? 'Places unknown' : 'Drive' }}</h2>
        <span>{{ facts }}</span>
      </div>
    </div>

    <p v-if="error" class="banner">The drive cannot be read: {{ error }}</p>
    <p v-else-if="!payload" class="empty">Loading the drive…</p>
    <p v-else-if="frames.length === 0" class="empty">No road types for this drive yet: Find the road types reads them.</p>
    <div v-else class="rt-view-main">
      <div class="rt-stage" tabindex="0" role="img" :aria-label="'Picture ' + (cursor + 1) + ' of ' + frames.length + ': ' + ROAD_LABEL[kind]"
           @click="toggle">
        <img v-if="frame?.image && !failed" :src="frame.image" alt="" draggable="false">
        <p v-else class="wp-noimage">{{ frame?.image ? 'The picture cannot be loaded.' : 'No picture of this frame.' }}</p>
        <span v-if="!playing" class="wp-paused" aria-hidden="true"><svg class="icon" viewBox="0 0 24 24"><path :d="MARKS.play" fill="currentColor"/></svg></span>
      </div>

      <div class="rt-time">
        <svg ref="strip" class="rt-track" :viewBox="'0 0 ' + frames.length + ' 20'" preserveAspectRatio="none" role="slider"
             :aria-valuemin="1" :aria-valuemax="frames.length" :aria-valuenow="cursor + 1" aria-label="Where in the drive"
             @pointerdown="stripDown" @pointermove="stripMove" @pointerup="dragging = false" @pointercancel="dragging = false">
          <rect v-for="run in runs" :key="run.first" :x="run.first" y="3" :width="run.count" height="11" :class="'rt-' + run.kind"/>
          <template v-for="run in runs" :key="'r' + run.first">
            <rect v-if="run.reviewed" :x="run.first" y="16" :width="run.count" height="3" class="rt-done"/>
          </template>
          <rect v-if="span && mark !== null" :x="span[0]" y="0" :width="span[1] - span[0] + 1" height="2" class="rt-marked"/>
          <rect :x="cursor" y="0" :width="Math.max(1, frames.length / 250)" height="20" class="rt-head"/>
        </svg>
        <div class="rt-controls">
          <button type="button" class="btn small primary rt-play" :aria-pressed="playing" @click="toggle"><svg class="icon"
            viewBox="0 0 24 24" aria-hidden="true"><path :d="playing ? MARKS.pause : MARKS.play" fill="currentColor"/></svg>{{ playing ? 'Pause' : 'Play' }}</button>
          <button type="button" class="btn small rt-step" aria-label="Previous picture" :disabled="cursor === 0" @click="step(-1)"><svg
            class="icon" viewBox="0 0 24 24" aria-hidden="true"><path :d="MARKS.back" fill="none" stroke="currentColor" stroke-width="2.2"
            stroke-linecap="round" stroke-linejoin="round"/></svg></button>
          <button type="button" class="btn small rt-step" aria-label="Next picture" :disabled="cursor >= frames.length - 1" @click="step(1)"><svg
            class="icon" viewBox="0 0 24 24" aria-hidden="true"><path :d="MARKS.forward" fill="none" stroke="currentColor" stroke-width="2.2"
            stroke-linecap="round" stroke-linejoin="round"/></svg></button>
          <span class="rt-pos"><b>{{ frame ? frameTime(frame.key, payload.start) : '' }}</b><span class="rt-count"> ·
            {{ thousands(cursor + 1) }} / {{ thousands(frames.length) }}</span></span>
          <Segmented v-model="speedKey" :options="speedOptions" label="Speed" class="rt-speed"/>
        </div>
      </div>

      <aside class="rt-side" aria-label="The road here">
        <div class="rt-now">
          <span class="rt-chip"><i class="rt-key" :class="'rt-' + kind" aria-hidden="true"></i>{{ ROAD_LABEL[kind] }}</span>
          <span v-if="frame?.reviewed" class="rt-flag done">Reviewed</span>
          <span v-else-if="frame && !frame.sure" class="rt-flag unsure">Not sure</span>
        </div>
        <p class="rt-basis">{{ reason }}</p>

        <div class="rt-fix">
          <p class="rt-fix-head"><strong>{{ mark === null ? 'This stretch is' : 'From the mark to here is' }}</strong>
            <span>{{ spanText }}</span></p>
          <div class="seg rt-pick" role="radiogroup" aria-label="Type of road">
            <button v-for="[key, label] in CHOICES" :key="key ?? 'none'" type="button" role="radio"
                    :aria-checked="Boolean(frame?.reviewed) && mark === null && frame?.road_type === key"
                    :class="{'rt-map': !frame?.reviewed && mark === null && frame?.road_type === key}"
                    :disabled="saving || readOnly" @click="setType(key)"><i class="rt-key" :class="'rt-' + (key ?? 'unknown')"
                    aria-hidden="true"></i>{{ label }}</button>
          </div>
          <div class="rt-fix-row">
            <button type="button" class="btn small" :disabled="readOnly" @click="mark = mark === null ? shown : null">{{ mark === null ? 'Mark start' : 'Clear the mark' }}</button>
            <button v-if="payload.corrections > 0" type="button" class="btn small" :disabled="saving || readOnly" @click="undo">Undo last change</button>
          </div>
          <p class="rt-hint">{{ readOnly ? 'Read-only: corrections are turned off on this server.'
            : mark === null ? 'Tap the right type: it is saved at once. Mark start to choose where a stretch begins.'
              : 'Now go to the end of the stretch, then tap its type.' }}</p>
        </div>

        <button type="button" class="btn rt-next" :disabled="left === 0" @click="nextCheck">Next to check<span class="rt-left">{{ thousands(left) }}</span></button>
      </aside>
    </div>
  </section>
</template>
