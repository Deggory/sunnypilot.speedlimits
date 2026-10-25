<script setup lang="ts">
// Watch: how the cutter cuts, on the frames themselves. The frames of one route of an evaluation (test or validation)
// play in the order of the drive like a video (1× = a frame a second, as they were taken), with the reviewed signs
// outlined in the colour of their family (dashed when the cutter missed them) and the cutter's boxes from a score: green
// on a sign, grey on nothing; in a chain, what the classifier called each box (the false boxes it throws away as “not a
// sign” faded, those it lets through red). Space plays and pauses, the arrows step a frame; the scrubber marks the
// frames with a sign missed. The frames come from GET /api/panel/runs/<run>/frames, their images from the review pages.
import {computed, onBeforeUnmount, onMounted, ref, useTemplateRef, watch} from 'vue';
import {plural, thousands} from '../common/format';
import type {WatchFrame, WatchModel, WatchPartition, WatchRoute} from '../common/panel-types';
import {PARTITION_LABEL} from './evaluate';
import {perFrame} from './models';
import Segmented from './Segmented.vue';
import {type DrawnBox, type DrawnTruth, type FrameFilter, type FrameView, type Speed, SPEEDS, familyColor, frameName,
        frameView, keptFrames, nearestStep, scoreSteps, shortLabel, useWatchFrames} from './watch';

const props = defineProps<{run: string; models: WatchModel[]; version: string | null}>();
/** The player's marks, solid shapes on the 24 × 24 grid of the site's icons (components/Icon.vue), in currentColor. */
const MARKS = {
  play: 'M8 5.6v12.8a.8.8 0 0 0 1.2.7l10.3-6.4a.8.8 0 0 0 0-1.4L9.2 4.9A.8.8 0 0 0 8 5.6z',
  pause: 'M7 5h3.2v14H7zM13.8 5H17v14h-3.2z',
  back: 'M5.5 5.5h2.2v13H5.5zM18.5 6.3v11.4a.8.8 0 0 1-1.3.6l-7.3-5.7a.8.8 0 0 1 0-1.2l7.3-5.7a.8.8 0 0 1 1.3.6z',
  next: 'M16.3 5.5h2.2v13h-2.2zM5.5 6.3v11.4a.8.8 0 0 0 1.3.6l7.3-5.7a.8.8 0 0 0 0-1.2L6.8 5.7a.8.8 0 0 0-1.3.6z',
} as const;

// -- which report, partition and route --------------------------------------------------------------------------------
const pickedModel = ref<string | null>(null);
/** The model shown: the one chosen, else the chain, else the last (largest) cutter of the sizes, else the first. */
const current = computed((): WatchModel | null => props.models.find((item) => item.model === pickedModel.value)
  ?? props.models.find((item) => item.model === 'chain')
  ?? [...props.models].reverse().find((item) => item.model.startsWith('detector-sizes/'))
  ?? props.models[0] ?? null);
const modelKey = computed({
  get: (): string => current.value?.model ?? '',
  set: (value: string) => { pickedModel.value = value; },
});
const modelOptions = computed(() => props.models.map((item) => ({key: item.model, label: item.label})));
const pickedPartition = ref<WatchPartition>('test');
const partition = computed({
  get: (): WatchPartition => current.value?.partitions.includes(pickedPartition.value) ? pickedPartition.value
    : current.value?.partitions[0] ?? 'test',
  set: (value: WatchPartition) => { pickedPartition.value = value; },
});
const partitionOptions = computed(() => (current.value?.partitions ?? []).map((key) => ({key, label: PARTITION_LABEL[key] ?? key})));
/** The route chosen; kept when another report is chosen (the server shows its own first route when it lacks it). */
const wantedRoute = ref<string | null>(null);
const {payload, loading, error} = useWatchFrames(computed(() => props.run), computed(() => current.value?.model ?? null),
  partition, wantedRoute, computed(() => props.version));
const frames = computed((): WatchFrame[] => payload.value?.frames ?? []);
const routes = computed((): WatchRoute[] => payload.value?.routes ?? []);
const route = computed({
  get: (): string => payload.value?.route ?? '',
  set: (value: string) => { wantedRoute.value = value; },
});
const chain = computed(() => payload.value?.kind === 'chain');
const level = computed(() => payload.value?.level ?? 'none');

// -- the score of the cutter --------------------------------------------------------------------------------------------
const steps = computed(() => payload.value ? scoreSteps(payload.value) : []);
const pickedScore = ref<number | null>(null);
const stepIndex = computed({
  get: (): number => nearestStep(steps.value, pickedScore.value ?? payload.value?.default_score ?? null),
  set: (value: number) => { pickedScore.value = steps.value[value] ?? null; },
});
const score = computed((): number => steps.value[stepIndex.value] ?? payload.value?.main_score ?? 0);

// -- what every frame shows at that score, and the frames the filter keeps -----------------------------------------------
const views = computed((): FrameView[] => frames.value.map((frame) =>
  frameView(frame, level.value, score.value, payload.value?.steps ?? null, chain.value)));
const filter = ref<FrameFilter>('all');
const filterOptions = computed(() => [
  {key: 'all' as FrameFilter, label: 'Every frame'}, {key: 'signs' as FrameFilter, label: 'With signs'},
  ...(level.value === 'none' ? [] : [{key: 'misses' as FrameFilter, label: 'With misses'}])]);
const visible = computed(() => keptFrames(frames.value, views.value, filter.value));

// -- the frame wanted, and the frame shown (its image decoded, so the boxes never sit on another frame) ---------------------
const playing = ref(false);
const speed = ref<Speed>(1);
const cursor = ref(0);
const shown = ref(0);
const failed = ref(false);
const position = computed(() => visible.value.indexOf(cursor.value));
const frame = computed((): WatchFrame | null => frames.value[shown.value] ?? null);
const view = computed((): FrameView | null => views.value[shown.value] ?? null);

interface Loaded {image: HTMLImageElement; ready: Promise<boolean>}
let cache = new Map<string, Loaded>();
function load(url: string): Promise<boolean> {
  let item = cache.get(url);
  if (item === undefined) {
    const image = new Image();
    image.decoding = 'async';
    image.src = url;
    item = {image, ready: image.decode().then(() => true, () => false)};
    cache.set(url, item);
    // The last few dozen frames stay decoded; the oldest go first.
    if (cache.size > 48) cache.delete(cache.keys().next().value as string);
  }
  return item.ready;
}

function preload(from: number): void {
  for (let ahead = 1; ahead <= 5; ahead++) {
    const index = visible.value[from + ahead];
    const url = index === undefined ? null : frames.value[index]?.image;
    if (url) void load(url);
  }
}

/** Shows a frame once its image is decoded (at once without one), and decodes the next few. */
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
  preload(Math.max(0, visible.value.indexOf(index)));
}
watch(cursor, showFrame);

/** A new payload: the same frame when the route stays (a report written again, or another report of the same route:
 * two cutters compared on one frame), else the start of the route. It keeps playing only when the report is the same. */
watch(payload, (now, before) => {
  const same = before && now && before.route === now.route;
  if (!same || before.model !== now.model || before.partition !== now.partition) playing.value = false;
  const key = same ? before.frames[shown.value]?.key ?? null : null;
  if (!now || before?.route !== now.route) cache = new Map();
  const again = key === null ? -1 : (now?.frames ?? []).findIndex((item) => item.key === key);
  cursor.value = again >= 0 ? again : visible.value[0] ?? 0;
  // The same index in the new frames does not move the cursor: show it all the same.
  showFrame(cursor.value);
});

/** A filter (or a score) that hides the frame shown: the next frame it keeps, else the last. */
watch(visible, (list) => {
  if (list.length === 0 || list.includes(cursor.value)) return;
  cursor.value = list.find((index) => index > cursor.value) ?? list[list.length - 1] ?? 0;
});

function seek(target: number): void {
  const list = visible.value;
  if (list.length === 0) return;
  const clamped = Math.min(list.length - 1, Math.max(0, target));
  cursor.value = list[clamped] ?? cursor.value;
}

function stepBy(delta: number): void {
  const list = visible.value;
  if (list.length === 0) return;
  let here = list.indexOf(cursor.value);
  if (here < 0) here = Math.max(0, list.findIndex((index) => index > cursor.value) - (delta > 0 ? 1 : 0));
  seek(here + delta);
}

// -- playing ------------------------------------------------------------------------------------------------------
const speedOptions = SPEEDS.map((value) => ({key: String(value), label: (value === 0.5 ? '½' : String(value)) + '×'}));
const speedKey = computed({
  get: (): string => String(speed.value),
  set: (value: string) => { speed.value = (SPEEDS.find((item) => String(item) === value) ?? 1) as Speed; },
});
let timer: number | undefined;
function schedule(): void {
  window.clearTimeout(timer);
  timer = undefined;
  // The next frame a frame-time after this one is on screen: a slow image never makes a frame skip.
  if (!playing.value || shown.value !== cursor.value) return;
  timer = window.setTimeout(() => {
    if (position.value >= visible.value.length - 1) {
      playing.value = false;
      return;
    }
    stepBy(1);
  }, 1000 / speed.value);
}
watch([playing, shown, speed], schedule);

function toggle(): void {
  if (!playing.value && position.value >= visible.value.length - 1) seek(0);
  playing.value = !playing.value;
}

// -- keys: space plays and pauses, the arrows step a frame, while the panel has the focus ----------------------------
const root = useTemplateRef<HTMLElement>('root');
function onKey(event: KeyboardEvent): void {
  if (event.altKey || event.ctrlKey || event.metaKey || !root.value || !(event.target instanceof Node)
      || !root.value.contains(event.target)) return;
  const target = event.target instanceof HTMLElement ? event.target : null;
  const tag = target?.tagName ?? '';
  const type = target instanceof HTMLInputElement ? target.type : '';
  if (event.key === ' ') {
    // A button or a box ticks itself with the space bar.
    if (tag === 'BUTTON' || tag === 'SELECT' || type === 'checkbox') return;
    event.preventDefault();
    toggle();
  } else if (event.key === 'ArrowRight' || event.key === 'ArrowLeft') {
    // The scrubber and the selects move with the arrows by themselves.
    if (tag === 'SELECT' || type === 'range') return;
    event.preventDefault();
    playing.value = false;
    stepBy(event.key === 'ArrowRight' ? 1 : -1);
  }
}
onMounted(() => document.addEventListener('keydown', onKey));
onBeforeUnmount(() => {
  document.removeEventListener('keydown', onKey);
  window.clearTimeout(timer);
});

// -- drawing: sizes in pixels of the frame, k on-screen pixels per frame pixel ---------------------------------------------
const stage = useTemplateRef<HTMLElement>('stage');
const stageWidth = ref(800);
let observer: ResizeObserver | undefined;
onMounted(() => {
  observer = new ResizeObserver(([entry]) => { if (entry) stageWidth.value = entry.contentRect.width || stageWidth.value; });
  if (stage.value) observer.observe(stage.value);
});
onBeforeUnmount(() => observer?.disconnect());
watch(stage, (element) => { observer?.disconnect(); if (element) observer?.observe(element); });
const size = computed(() => ({width: frame.value?.width ?? frames.value.find((item) => item.width)?.width ?? 1344,
                              height: frame.value?.height ?? frames.value.find((item) => item.height)?.height ?? 760}));
const k = computed(() => Math.max(0.05, stageWidth.value / size.value.width));
const px = (value: number): number => value / k.value;

interface Mark {x: number; y: number; width: number; height: number}
function rect(box: readonly number[], grow = 0): Mark {
  const [x1 = 0, y1 = 0, x2 = 0, y2 = 0] = box;
  return {x: x1 - grow, y: y1 - grow, width: Math.max(1, x2 - x1 + 2 * grow), height: Math.max(1, y2 - y1 + 2 * grow)};
}
const BOX_ORDER = {junk: 0, rejected: 1, found: 3, survives: 2} as const;
const boxes = computed((): DrawnBox[] => [...(view.value?.boxes ?? [])].sort((a, b) => BOX_ORDER[a.outcome] - BOX_ORDER[b.outcome]));
const OUTCOME_WORDS = {found: 'on a sign', junk: 'on nothing', rejected: 'on nothing, thrown away by the classifier',
                       survives: 'on nothing, let through by the classifier'} as const;
function boxTitle(item: DrawnBox): string {
  return 'Cutter, score ' + item.score.toFixed(3) + ', ' + OUTCOME_WORDS[item.outcome] + (item.label ? ': ' + shortLabel(item.label) : '');
}
function truthTitle(item: DrawnTruth): string {
  const state = item.found === null ? '' : item.found ? ', found' : ', missed';
  return 'Reviewed ' + (item.family ?? 'sign').replace(/_/g, ' ') + state + (item.verdict ? ', read as ' + shortLabel(item.verdict) : '');
}

interface Tag {x: number; y: number; text: string; tone: string}
/** The words on the frame: what the classifier called a box (a chain), above it, or under it at the top edge. */
const tags = computed((): Tag[] => {
  const list: Tag[] = [];
  const font = px(12);
  const place = (box: readonly number[], text: string, tone: string) => {
    const [x1 = 0, y1 = 0, , y2 = 0] = box;
    const above = y1 - px(4);
    list.push({x: x1, y: above - font < 0 ? y2 + font + px(2) : above, text, tone});
  };
  for (const item of view.value?.boxes ?? []) {
    if (item.outcome === 'found' && item.label) place(item.box, shortLabel(item.label), item.label === 'not_a_sign' ? 'bad' : item.right ? 'ok' : 'warn');
    if (item.outcome === 'survives' && item.label) place(item.box, shortLabel(item.label), 'bad');
  }
  // An older chain report: the classifier's answer on the reviewed box it found.
  if (level.value === 'verdicts') {
    for (const item of view.value?.truth ?? []) {
      if (item.verdict) place(item.box, shortLabel(item.verdict), item.verdict === 'not_a_sign' ? 'bad' : item.right ? 'ok' : 'warn');
    }
  }
  return list;
});

// -- the scrubber's marks and the route at this score -------------------------------------------------------------------
const ticks = computed(() => {
  const list = visible.value;
  const span = Math.max(1, list.length - 1);
  let signs = '';
  let misses = '';
  list.forEach((index, at) => {
    const item = views.value[index];
    const x = (list.length === 1 ? 0.5 : at / span) * 1000;
    if (item?.missed) misses += `M${x.toFixed(1)} 0V10`;
    else if (item?.signs) signs += `M${x.toFixed(1)} 3V10`;
  });
  return {signs, misses};
});
const totals = computed(() => {
  const list = views.value;
  const sum = (pick: (item: FrameView) => number | null): number | null =>
    list.some((item) => pick(item) === null) ? null : list.reduce((total, item) => total + (pick(item) ?? 0), 0);
  return {frames: list.length, signs: sum((item) => item.signs), found: sum((item) => item.found),
          missedFrames: list.filter((item) => item.missed).length, junk: sum((item) => item.junk),
          surviving: sum((item) => item.surviving), hidden: list.some((item) => item.hidden > 0)};
});
const routeLabel = (item: WatchRoute): string => item.route + ' · ' + plural(item.frames, 'frame') + ' · ' + plural(item.signs, 'sign')
  + (item.missed === null || !item.signs ? '' : ', ' + thousands(item.missed) + ' missed');
const scoreText = computed(() => score.value.toFixed(2));
const levelNote = computed((): string => {
  const value = payload.value;
  if (!value) return '';
  // An older report kept less of the cutter's boxes; the site cannot measure it again, so the note only says what it shows.
  if (value.level === 'none') return 'This report kept none of the cutter’s boxes: only the reviewed signs are drawn.';
  if (value.level === 'verdicts') {
    return 'This report kept only score ' + (value.main_score ?? '–') + ': the signs the cutter found or missed, what the '
      + 'classifier called them, and the false boxes it let through.';
  }
  if (value.level === 'above') {
    return 'This report kept the cutter’s boxes from score ' + (value.min_score ?? '–') + ' up: lower scores cannot be shown.';
  }
  return '';
});
const human = computed(() => frame.value?.basis === 'human_review');
</script>

<template>
  <section v-if="models.length" ref="root" class="card eval-card watch" id="watch" aria-labelledby="watch-title">
    <div class="ds-head">
      <h2 id="watch-title">Watch<span class="eval-what">the cutter’s boxes on the frames, one route at a time</span></h2>
      <span v-if="payload" class="ds-when">{{ payload.label }} · {{ PARTITION_LABEL[payload.partition] ?? payload.partition }} ·
        {{ plural(payload.image_count, 'frame') }}</span>
    </div>
    <div class="seg-row">
      <Segmented v-if="modelOptions.length > 1" v-model="modelKey" :options="modelOptions" label="Report"/>
      <Segmented v-if="partitionOptions.length > 1" v-model="partition" :options="partitionOptions" label="Partition"/>
    </div>

    <p v-if="error" class="banner">The frames cannot be read: {{ error }}</p>
    <p v-else-if="!payload" class="empty">{{ loading ? 'Loading the frames…' : '' }}</p>
    <template v-else>
      <div class="wp-pick">
        <label class="wp-field">Route
          <select v-model="route" id="watch-route" :disabled="routes.length < 2">
            <option v-for="item in routes" :key="item.route" :value="item.route">{{ routeLabel(item) }}</option>
          </select>
        </label>
        <label class="wp-field wp-score">Cutter score ≥ <output id="watch-score">{{ scoreText }}</output>
          <input v-model.number="stepIndex" type="range" min="0" :max="Math.max(0, steps.length - 1)" step="1"
                 :disabled="steps.length < 2" aria-label="Lowest score of the cutter’s boxes drawn"
                 :aria-valuetext="'score ' + scoreText">
        </label>
        <Segmented v-model="filter" :options="filterOptions" label="Frames played"/>
      </div>
      <p v-if="payload.route_not_here" class="note">This report has no frame of {{ payload.route_not_here }}: showing {{ payload.route }}.</p>
      <p v-if="levelNote" class="compare-warn" id="watch-level">{{ levelNote }}</p>

      <p v-if="frames.length === 0" class="empty">This report has no frames.</p>
      <template v-else>
        <div ref="stage" class="wp-stage" :style="{aspectRatio: size.width + ' / ' + size.height}" tabindex="0"
             role="img" :aria-label="frame ? 'Frame ' + frameName(frame) + ' of route ' + payload.route : 'Frame'"
             @click="toggle">
          <img v-if="frame?.image && !failed" :src="frame.image" alt="" draggable="false">
          <p v-else class="wp-noimage">{{ frame?.image ? 'The image of this frame cannot be loaded.' : 'No preview of this frame on the review page.' }}</p>
          <svg class="wp-overlay" :viewBox="'0 0 ' + size.width + ' ' + size.height" preserveAspectRatio="none" aria-hidden="true">
            <g v-for="(item, index) in boxes" :key="'b' + index">
              <rect v-bind="rect(item.box)" class="wp-box" :class="item.outcome" :stroke-width="px(item.outcome === 'junk' || item.outcome === 'rejected' ? 1 : 1.6)">
                <title>{{ boxTitle(item) }}</title>
              </rect>
            </g>
            <g v-for="(item, index) in view?.truth ?? []" :key="'t' + index">
              <rect v-bind="rect(item.box, px(3))" class="wp-truth" :class="{missed: item.found === false}" :stroke="familyColor(item.family)"
                    :stroke-width="px(2.4)" :stroke-dasharray="item.found === false ? px(6) + ' ' + px(4) : undefined">
                <title>{{ truthTitle(item) }}</title>
              </rect>
            </g>
            <text v-for="(tag, index) in tags" :key="'l' + index" :x="tag.x" :y="tag.y" class="wp-tag" :class="tag.tone"
                  :font-size="px(12)" :stroke-width="px(3)">{{ tag.text }}</text>
          </svg>
          <span class="wp-where">{{ frame ? frameName(frame) : '' }}<span v-if="human" class="wp-human">human-reviewed</span></span>
          <span v-if="!playing" class="wp-paused" aria-hidden="true"><svg class="icon" viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path :d="MARKS.play" fill="currentColor"/></svg></span>
        </div>

        <ul v-if="view" class="wp-counts" id="watch-counts" aria-label="This frame">
          <li><strong>{{ view.signs }}</strong> {{ view.signs === 1 ? 'sign' : 'signs' }}</li>
          <li class="found"><strong>{{ view.found ?? '–' }}</strong> found</li>
          <li class="missed"><strong>{{ view.missed ?? '–' }}</strong> missed</li>
          <li class="junk"><strong>{{ view.junk ?? '–' }}</strong> junk {{ view.junk === 1 ? 'box' : 'boxes' }}<small v-if="view.hidden"> ({{ view.hidden }} not drawn)</small></li>
          <li v-if="chain" class="survives"><strong>{{ view.surviving ?? '–' }}</strong> junk let through</li>
        </ul>

        <div class="wp-controls">
          <button type="button" class="btn small" aria-label="Previous frame" title="Previous frame (←)" @click="playing = false; stepBy(-1)"
                  :disabled="position <= 0"><svg class="icon" viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path :d="MARKS.back" fill="currentColor"/></svg></button>
          <button type="button" class="btn small primary wp-play" id="watch-play" :aria-pressed="playing" @click="toggle"
                  :disabled="visible.length === 0"><svg class="icon" viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path :d="playing ? MARKS.pause : MARKS.play" fill="currentColor"/></svg>{{ playing ? 'Pause' : 'Play' }}</button>
          <button type="button" class="btn small" aria-label="Next frame" title="Next frame (→)" @click="playing = false; stepBy(1)"
                  :disabled="position < 0 || position >= visible.length - 1"><svg class="icon" viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path :d="MARKS.next" fill="currentColor"/></svg></button>
          <Segmented v-model="speedKey" :options="speedOptions" label="Speed"/>
          <span class="wp-position" id="watch-position">{{ visible.length ? thousands(Math.max(0, position) + 1) + ' of ' + thousands(visible.length) : 'no frame' }}<template
            v-if="filter !== 'all'"> kept of {{ thousands(frames.length) }}</template></span>
        </div>
        <div class="wp-scrub">
          <svg class="wp-ticks" viewBox="0 0 1000 10" preserveAspectRatio="none" aria-hidden="true">
            <path :d="ticks.signs" class="sign"/><path :d="ticks.misses" class="miss"/>
          </svg>
          <input type="range" min="0" :max="Math.max(0, visible.length - 1)" step="1" :value="Math.max(0, position)"
                 :disabled="visible.length < 2" aria-label="Frame of the route" id="watch-scrub"
                 @input="playing = false; seek(Number(($event.target as HTMLInputElement).value))">
        </div>

        <ul class="wp-legend" aria-label="Legend">
          <li><i class="sw truth"></i>reviewed sign, colour of its family</li>
          <li><i class="sw truth missed"></i>missed by the cutter</li>
          <li><i class="sw found"></i>cutter box on a sign</li>
          <li v-if="!chain"><i class="sw junk"></i>cutter box on nothing</li>
          <template v-else>
            <li><i class="sw rejected"></i>on nothing, thrown away as “not a sign”</li>
            <li><i class="sw survives"></i>on nothing, let through</li>
          </template>
          <li class="muted">ticks: frames with signs, red where one is missed · space plays, ← → step</li>
        </ul>
        <p class="note" id="watch-route-total">This route at score {{ scoreText }}: <template v-if="totals.found !== null">{{ thousands(totals.found) }} of
          {{ plural(totals.signs ?? 0, 'sign') }} found<template v-if="totals.missedFrames">, misses in {{ plural(totals.missedFrames, 'frame') }}</template></template><template
          v-else>{{ plural(totals.signs ?? 0, 'sign') }}</template><template v-if="totals.junk !== null"> · {{ perFrame(totals.junk / Math.max(1, totals.frames)) }} junk
          boxes per frame</template><template v-if="chain && totals.surviving !== null">, {{ perFrame(totals.surviving / Math.max(1, totals.frames)) }} let through</template>.<template
          v-if="totals.hidden && payload.max_boxes"> A frame keeps its best {{ payload.max_boxes }} boxes (and every box on a sign): the counts are of every box.</template><template
          v-if="payload.with_image < frames.length"> {{ plural(frames.length - payload.with_image, 'frame') }} without a preview.</template></p>
      </template>
    </template>
  </section>
</template>
