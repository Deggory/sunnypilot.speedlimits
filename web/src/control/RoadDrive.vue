<script setup lang="ts">
// Data › Roads, a drive opened ("/roads?drive=<route>"): its pictures one at a time with the type of road there, the
// map's or a person's (GET /api/panel/road-types/<route>), and the strip of the whole drive under it, coloured as the
// list, a dark line under the stretches a person set, an orange one under those to check, a blue one over the stretch a
// tap sets; a tap or a drag on it goes there. Play, the arrows and space step as in Results › Drive (DriveView.vue). A
// person corrects the map: the type of the stretch shown (or from a marked start to here), saved at once (POST, the drive
// comes back), after a question when it is longer than a minute; Next to check goes to the next stretch the map is
// unsure about. On a phone the loop (Next to check, look, tap a type) fits one screen; the rest is under it.
import {computed, onBeforeUnmount, onMounted, ref, shallowRef, useTemplateRef, watch} from 'vue';
import {api, messageOf, post} from '../common/api';
import {plural, thousands} from '../common/format';
import type {RoadDriveView, RoadType} from '../common/panel-types';
import {askConfirm} from '../composables/confirm';
import {toast} from '../composables/toast';
import {checkStretches, frameTime, nextToCheck, reasonOf, ROAD_LABEL, ROAD_TYPES, roadRuns, stretchAt} from './roads';
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
  ['motorway', 'Motorway'], ['parking', 'Parking'], [null, 'Can\'t tell']];
/** Longer than this (pictures, one a second), a correction asks first: one tap must not overwrite minutes of a drive. */
const ASK_ABOVE = 60;

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
/** The stretches to check: a line under each on the strip, Next to check goes from one to the next. */
const checks = computed(() => checkStretches(frames.value));
/** The narrowest mark of the strip, in frames: the position, a stretch of one picture (the stretch a tap sets three
 * times that, to show beside the position). */
const thin = computed(() => Math.max(1, frames.value.length / 250));
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

const stage = useTemplateRef<HTMLElement>('stage');
/** Next to check: the next stretch no person set where the map says nothing or is often wrong. */
function nextCheck(): void {
  const index = nextToCheck(checks.value, cursor.value);
  if (index === null) {
    toast('Nothing else to check on this drive.');
    return;
  }
  step(index - cursor.value);
  // The picture out of sight (a phone scrolled down to the buttons): back to its top, under the top bar.
  const box = stage.value?.getBoundingClientRect();
  const bar = document.querySelector('.topbar')?.getBoundingClientRect().bottom ?? 0;
  if (box && (box.top < bar || box.bottom > window.innerHeight)) window.scrollTo({top: window.scrollY + box.top - bar - 8});
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
/** Why the frame has its type, in a driver's words ('' when the map's reason has none). */
const reason = computed(() => frame.value ? reasonOf(frame.value) : '');
/** The frames a correction covers: from the mark to here, else the stretch shown (stretchAt: of one kind, as sure, not
 * set by a person when this frame is not). */
const span = computed((): [number, number] | null => {
  if (!frames.value.length) return null;
  if (mark.value !== null) return [Math.min(mark.value, shown.value), Math.max(mark.value, shown.value)];
  return stretchAt(frames.value, shown.value);
});
/** The times of the stretch a correction covers: "16:13:34 – 16:21:38". */
const spanTimes = computed((): string => {
  const [first, last] = span.value ?? [0, -1];
  const start = payload.value?.start ?? null;
  const from = frames.value[first];
  const to = frames.value[last];
  if (!from || !to) return '';
  return frameTime(from.key, start) + (last > first ? ' – ' + frameTime(to.key, start) : '');
});
const spanText = computed((): string => {
  const [first, last] = span.value ?? [0, -1];
  return spanTimes.value ? plural(last - first + 1, 'picture') + ' · ' + spanTimes.value : '';
});
const saving = ref(false);
async function setType(type: RoadType | null): Promise<void> {
  const [first, last] = span.value ?? [0, -1];
  const from = frames.value[first];
  const to = frames.value[last];
  if (!from || !to || saving.value) return;
  const count = last - first + 1;
  const label = type === null ? 'Can\'t tell' : ROAD_LABEL[type];
  if (count > ASK_ABOVE && !await askConfirm({
    title: 'Set ' + plural(count, 'picture') + ' (' + Math.round(count / 60) + ' min) to ' + label + '?',
    text: 'The stretch marked in blue on the strip, ' + spanTimes.value + '. Undo last change takes it back.',
    action: 'Set to ' + label,
  })) return;
  if (saving.value) return;
  saving.value = true;
  try {
    payload.value = await post<RoadDriveView>('/api/panel/road-types/' + encodeURIComponent(props.route),
      {from: from.key, to: to.key, road_type: type});
    mark.value = null;
    toast(label + ' · ' + plural(count, 'picture') + ' saved');
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
      <div ref="stage" class="rt-stage" tabindex="0" role="img"
           :aria-label="'Picture ' + (cursor + 1) + ' of ' + frames.length + ': ' + ROAD_LABEL[kind]" @click="toggle">
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
          <rect v-for="item in checks" :key="'c' + item.first" :x="item.first" y="16" :width="Math.max(item.count, thin)" height="3"
                class="rt-tocheck"/>
          <rect v-if="span" :x="span[0]" y="0" :width="Math.max(span[1] - span[0] + 1, 3 * thin)" height="2.2" class="rt-marked"/>
          <rect :x="cursor" y="0" :width="thin" height="20" class="rt-head"/>
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
        </div>
      </div>

      <aside class="rt-side" aria-label="The road here">
        <div class="rt-now">
          <span class="rt-chip"><i class="rt-key" :class="'rt-' + kind" aria-hidden="true"></i>{{ ROAD_LABEL[kind] }}</span>
          <span v-if="frame?.reviewed" class="rt-flag done">Reviewed</span>
          <span v-else-if="frame && frame.road_type !== null && !frame.sure" class="rt-flag unsure">Not sure</span>
        </div>
        <p v-if="reason" class="rt-basis">{{ reason }}</p>

        <div class="rt-fix">
          <p class="rt-fix-head"><strong>{{ mark === null ? 'This stretch' : 'From the mark to here' }}</strong>
            <span>{{ spanText }}</span></p>
          <div class="seg rt-pick" role="radiogroup" aria-label="Type of road">
            <button v-for="[key, label] in CHOICES" :key="key ?? 'none'" type="button" role="radio"
                    :aria-checked="Boolean(frame?.reviewed) && mark === null && frame?.road_type === key"
                    :class="{'rt-map': key !== null && !frame?.reviewed && mark === null && frame?.road_type === key}"
                    :disabled="saving || readOnly" @click="setType(key)"><i class="rt-key" :class="'rt-' + (key ?? 'unknown')"
                    aria-hidden="true"></i>{{ label }}</button>
          </div>
        </div>

        <button type="button" class="btn rt-next" :disabled="checks.length === 0" @click="nextCheck">{{ checks.length ? 'Next to check'
          : 'Nothing left to check' }}<span v-if="checks.length" class="rt-left">{{ plural(checks.length, 'stretch', 'stretches') }}</span></button>

        <div class="rt-more">
          <Segmented v-model="speedKey" :options="speedOptions" label="Speed" class="rt-speed"/>
          <div class="rt-fix-row">
            <button type="button" class="btn small" :disabled="readOnly" @click="mark = mark === null ? shown : null">{{ mark === null ? 'Mark start' : 'Clear the mark' }}</button>
            <button v-if="payload.corrections > 0" type="button" class="btn small" :disabled="saving || readOnly" @click="undo">Undo last change</button>
          </div>
          <p class="rt-hint">{{ readOnly ? 'Read-only: corrections are turned off on this server.'
            : mark === null ? 'Tap the right type: the stretch marked in blue on the strip is saved at once, a long one after a question. Mark start to choose where a stretch begins.'
              : 'Now go to the end of the stretch, then tap its type.' }}</p>
          <ul class="rt-legend rt-marks" aria-label="The strip">
            <li v-for="[key, label] in ROAD_TYPES" :key="key"><span class="rt-key" :class="'rt-' + key" aria-hidden="true"></span>{{ label }}</li>
            <li><span class="rt-line rt-tocheck" aria-hidden="true"></span>To check</li>
            <li><span class="rt-line rt-done" aria-hidden="true"></span>Set by you</li>
            <li><span class="rt-line rt-marked" aria-hidden="true"></span>What a tap sets</li>
          </ul>
        </div>
      </aside>
    </div>
  </section>
</template>
