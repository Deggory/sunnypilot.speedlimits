<script setup lang="ts">
// Results › Drive: a drive replayed as in the car (drive.ts). The frames of one drive play like a video, one a second
// as they were taken; on the frame, the speed signs the networks read (green when right, orange when they read another
// number, red on nothing) and those they missed (dashed); in the corner, the limit they would show now next to the limit
// of the signs. Under it, the whole drive as a strip: green where the limit shown is right, red where it is wrong; a tap
// goes there. The chain of a classifier version (with the cutter it was measured with), on the test drives (never
// trained on) or the validation drives. Space plays, the arrows step.
import {computed, onBeforeUnmount, onMounted, ref, useTemplateRef, watch} from 'vue';
import {isoWhen, plural, thousands} from '../common/format';
import type {ClassifierModel, WatchPartition, WatchRoute} from '../common/panel-types';
import {type DriveFrame, limitText, replayDrive, stateRuns} from './drive';
import {sizeText, useModels} from './models';
import PickerSheet, {type PickerOption} from '../components/PickerSheet.vue';
import Segmented from './Segmented.vue';
import {SPEEDS, type Speed, useWatchFrames} from './watch';

const MARKS = {
  play: 'M8 5.6v12.8a.8.8 0 0 0 1.2.7l10.3-6.4a.8.8 0 0 0 0-1.4L9.2 4.9A.8.8 0 0 0 8 5.6z',
  pause: 'M7 5h3.2v14H7zM13.8 5H17v14h-3.2z',
} as const;

// -- what is replayed: a classifier version's chain, a partition, a drive -------------------------------------------------
const {payload: models} = useModels();
/** The classifier versions measured in a chain, newest first. */
const sources = computed(() => ((models.value?.roles.find((role) => role.role === 'classifier')?.versions ?? []) as ClassifierModel[])
  .filter((version) => version.chain));
const pickedSource = ref<string | null>(null);
const source = computed((): ClassifierModel | null => sources.value.find((version) => version.id === pickedSource.value)
  ?? sources.value.find((version) => version.in_use) ?? sources.value[0] ?? null);
const sourceId = computed({
  get: (): string => source.value?.id ?? '',
  set: (value: string) => { pickedSource.value = value; },
});
/** Each version as the picker shows it: when it was trained, with which cutter it was measured. */
const sourceOptions = computed((): PickerOption[] => sources.value.map((version) => ({key: version.id,
  title: 'Classifier ' + isoWhen(version.trained_at), short: isoWhen(version.trained_at), badge: version.in_use ? 'In use' : undefined,
  sub: version.chain?.cutter_size ? 'with the cutter ' + sizeText(version.chain.cutter_size) : undefined})));

const PARTITIONS = [{key: 'test' as WatchPartition, label: 'Test drives'}, {key: 'validation' as WatchPartition, label: 'Validation drives'}];
const partition = ref<WatchPartition>('test');
const wantedRoute = ref<string | null>(null);
const {payload, loading, error} = useWatchFrames(computed(() => source.value?.run ?? ''), computed(() => source.value ? 'chain' : null),
  partition, wantedRoute, computed(() => null));
/** The drives of the report: those with speed signs first, by name. */
const routes = computed((): WatchRoute[] => payload.value?.routes ?? []);
const route = computed({
  get: (): string => payload.value?.route ?? '',
  set: (value: string) => { wantedRoute.value = value; },
});
/** Each drive as the picker shows it: where, then when and how long. */
const routeOptions = computed((): PickerOption[] => routes.value.map((item) => ({key: item.route, title: item.title || item.route,
  sub: [item.when, plural(item.frames, 'frame'), plural(item.signs, 'sign')].filter(Boolean).join(' · ')})));

// -- the drive replayed -------------------------------------------------------------------------------------------------
const frames = computed(() => payload.value?.frames ?? []);
const score = computed(() => payload.value?.main_score ?? payload.value?.default_score ?? 0.1);
const replay = computed(() => replayDrive(frames.value, score.value));
const runs = computed(() => stateRuns(replay.value.frames));
const summary = computed(() => replay.value.summary);
const rightShare = computed(() => summary.value.judged ? Math.round(summary.value.right * 100 / summary.value.judged) : null);

// -- the frame wanted, and the one shown (its image decoded first, so the marks never sit on another frame) --------------
const playing = ref(false);
const speed = ref<Speed>(2);
const cursor = ref(0);
const shown = ref(0);
const failed = ref(false);
const frame = computed(() => frames.value[shown.value] ?? null);
const now = computed((): DriveFrame | null => replay.value.frames[shown.value] ?? null);

let cache = new Map<string, Promise<boolean>>();
function load(url: string): Promise<boolean> {
  let ready = cache.get(url);
  if (ready === undefined) {
    const image = new Image();
    image.decoding = 'async';
    image.src = url;
    ready = image.decode().then(() => true, () => false);
    cache.set(url, ready);
    if (cache.size > 48) cache.delete(cache.keys().next().value as string);
  }
  return ready;
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
  for (let ahead = 1; ahead <= 6; ahead++) {
    const next = frames.value[index + ahead]?.image;
    if (next) void load(next);
  }
}
watch(cursor, showFrame);
watch(payload, (value, before) => {
  playing.value = false;
  if (!before || before.route !== value?.route) cache = new Map();
  cursor.value = 0;
  showFrame(0);
});

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
const speedOptions = SPEEDS.map((value) => ({key: String(value), label: (value === 0.5 ? '½' : String(value)) + '×'}));
const speedKey = computed({
  get: (): string => String(speed.value),
  set: (value: string) => { speed.value = (SPEEDS.find((item) => String(item) === value) ?? 1) as Speed; },
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
    playing.value = false;
    seek(cursor.value + (event.key === 'ArrowRight' ? 1 : -1));
  }
}
onMounted(() => document.addEventListener('keydown', onKey));
onBeforeUnmount(() => {
  document.removeEventListener('keydown', onKey);
  window.clearTimeout(timer);
});

// -- drawing ----------------------------------------------------------------------------------------------------------
const size = computed(() => ({width: frame.value?.width ?? frames.value.find((item) => item.width)?.width ?? 1344,
                              height: frame.value?.height ?? frames.value.find((item) => item.height)?.height ?? 760}));
/** Line widths and text in pixels of the frame, for a frame shown about 900 px wide. */
const unit = computed(() => size.value.width / 900);
function rect(box: readonly number[], grow: number): {x: number; y: number; width: number; height: number} {
  const [x1 = 0, y1 = 0, x2 = 0, y2 = 0] = box;
  return {x: x1 - grow, y: y1 - grow, width: Math.max(1, x2 - x1 + 2 * grow), height: Math.max(1, y2 - y1 + 2 * grow)};
}

/** The strip: a tap or a drag goes to that point of the drive. */
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
/** The frames where a speed sign was read (the networks) or reviewed (the truth), as ticks over the strip. */
const ticks = computed(() => {
  let model = '';
  let truth = '';
  replay.value.frames.forEach((item, index) => {
    if (item.modelSign) model += `M${index + 0.5} 0V3`;
    if (item.truthSign) truth += `M${index + 0.5} 17V20`;
  });
  return {model, truth};
});
const stateWords = {right: 'Right', wrong: 'Wrong', none: 'No limit yet'} as const;
</script>

<template>
  <section ref="root" class="card drive" id="drive" aria-labelledby="drive-title">
    <h2 id="drive-title" class="sr-only">Drive</h2>
    <div class="drive-bar">
      <PickerSheet v-model="route" :options="routeOptions" label="Drive" id="drive-route"/>
      <PickerSheet v-model="sourceId" :options="sourceOptions" label="Classifier" id="drive-source"/>
      <Segmented v-model="partition" :options="PARTITIONS" label="Which drives" class="drive-part"/>
    </div>
    <p class="drive-why">{{ partition === 'test'
      ? 'Test drives: roads the networks never saw while training — how they would do on a new road.'
      : 'Validation drives: used while training to choose when to stop, so a little easier than the test.' }}</p>

    <p v-if="models !== null && sources.length === 0" class="empty">No classifier measured with its cutter yet: train one on Training.</p>
    <p v-else-if="error" class="banner">The drive cannot be read: {{ error }}</p>
    <p v-else-if="!payload" class="empty">{{ loading || models === null ? 'Loading the drive…' : '' }}</p>
    <p v-else-if="frames.length === 0" class="empty">This drive has no frames.</p>
    <div v-else class="drive-main">
      <div class="drive-stage" :style="{aspectRatio: size.width + ' / ' + size.height}" tabindex="0" role="img"
           :aria-label="'Frame ' + (cursor + 1) + ' of ' + frames.length" @click="toggle">
        <img v-if="frame?.image && !failed" :src="frame.image" alt="" draggable="false">
        <p v-else class="wp-noimage">{{ frame?.image ? 'The image of this frame cannot be loaded.' : 'No picture of this frame.' }}</p>
        <svg class="wp-overlay" :viewBox="'0 0 ' + size.width + ' ' + size.height" preserveAspectRatio="none" aria-hidden="true">
          <g v-for="(mark, index) in now?.marks ?? []" :key="index" class="drive-mark" :class="mark.tone">
            <rect v-bind="rect(mark.box, unit * 3)" :stroke-width="unit * 2.5" :stroke-dasharray="mark.tone === 'missed' ? unit * 7 + ' ' + unit * 5 : undefined"
                  :rx="unit * 3"/>
            <text :x="mark.box[0] - unit * 3" :y="mark.box[1] - unit * 8" :font-size="unit * 15" :stroke-width="unit * 4">{{ mark.text }}</text>
          </g>
        </svg>
        <!-- As the car would show it: the limit now, and under it what the signs say. -->
        <div class="drive-hud" :class="now?.state">
          <span class="drive-sign" :class="{end: now?.model === 'end', empty: now?.model === null}">{{ limitText(now?.model ?? null) }}</span>
          <span class="drive-truth">signs: <b>{{ limitText(now?.truth ?? null) }}</b></span>
        </div>
        <span v-if="!playing" class="wp-paused" aria-hidden="true"><svg class="icon" viewBox="0 0 24 24"><path :d="MARKS.play" fill="currentColor"/></svg></span>
      </div>

      <aside class="drive-side" aria-label="The drive at a glance">
        <div class="drive-big"><strong>{{ rightShare === null ? '–' : rightShare + '%' }}</strong>
          <span>of the drive with the right limit</span></div>
        <ul class="drive-facts" :aria-label="'A speed sign on ' + plural(summary.signs, 'frame')">
          <li class="drive-facts-head">A speed sign on {{ plural(summary.signs, 'frame') }}:</li>
          <li><b>{{ summary.read }}</b> read right</li>
          <li class="wrong"><b>{{ summary.misread }}</b> read as another limit</li>
          <li class="missed"><b>{{ summary.missed }}</b> missed</li>
          <li class="false"><b>{{ summary.invented }}</b> {{ summary.invented === 1 ? 'limit' : 'limits' }} read where there is no sign</li>
        </ul>
        <p class="drive-now" :class="now?.state">Now: <b>{{ stateWords[now?.state ?? 'none'] }}</b></p>
        <ul class="drive-legend" aria-label="Legend">
          <li><i class="right"></i>read right</li><li><i class="wrong"></i>another number</li>
          <li><i class="false"></i>no sign there</li><li><i class="missed"></i>missed</li>
        </ul>
      </aside>

      <div class="drive-time">
        <button type="button" class="btn small primary drive-play" id="drive-play" :aria-pressed="playing" @click="toggle"><svg
          class="icon" viewBox="0 0 24 24" aria-hidden="true"><path :d="playing ? MARKS.pause : MARKS.play" fill="currentColor"/></svg>{{ playing ? 'Pause' : 'Play' }}</button>
        <svg ref="strip" class="drive-strip" :viewBox="'0 0 ' + frames.length + ' 20'" preserveAspectRatio="none" role="slider"
             :aria-valuemin="1" :aria-valuemax="frames.length" :aria-valuenow="cursor + 1" aria-label="Where in the drive"
             @pointerdown="stripDown" @pointermove="stripMove" @pointerup="dragging = false" @pointercancel="dragging = false">
          <rect v-for="[first, count, state] in runs" :key="first" :x="first" y="5" :width="count" height="10" :class="state"/>
          <path :d="ticks.model" class="tick-model"/><path :d="ticks.truth" class="tick-truth"/>
          <rect :x="cursor" y="0" :width="Math.max(1, frames.length / 300)" height="20" class="head"/>
        </svg>
        <Segmented v-model="speedKey" :options="speedOptions" label="Speed" class="drive-speed"/>
        <span class="drive-pos">{{ thousands(cursor + 1) }} / {{ thousands(frames.length) }}</span>
      </div>
    </div>
  </section>
</template>
