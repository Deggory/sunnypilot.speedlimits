<script setup lang="ts">
// A line chart of a training, hand-made SVG: x = epoch over all the planned epochs (the part still to come
// stays empty, so the progress reads at a glance), y from 0. Thin lines, hairline grid, the best epoch
// marked, the epoch in progress shaded while the training runs, the last values labelled at the line ends
// when they do not collide. Hover, tap or the arrow keys show the values of an epoch in a tooltip; the
// legend names the lines (and the table under the charts has every value).
import {computed, onBeforeUnmount, onMounted, ref, useTemplateRef} from 'vue';
import {epochTicks, valueTicks} from './scale';

export interface ChartSeries {
  key: string;
  label: string;
  /** A CSS colour, e.g. var(--c-train). */
  color: string;
  /** values[i] is epoch i + 1; null where the log has no value. */
  values: (number | null)[];
}

const props = defineProps<{
  title: string;
  series: ChartSeries[];
  epochs: number;
  /** Axis and tooltip text of a value. */
  format: (value: number) => string;
  /** A fixed top of the y axis (100 for percentages); otherwise a round number above the largest value. */
  top?: number | undefined;
  /** The best epoch of a series, marked on its line. */
  best?: {series: string; epoch: number; label: string} | null | undefined;
  running?: boolean | undefined;
}>();

const HEIGHT = 210;
const M = {top: 16, right: 64, bottom: 26, left: 46};
const wrap = useTemplateRef<HTMLDivElement>('wrap');
const width = ref(560);
let observer: ResizeObserver | undefined;
onMounted(() => {
  if (!wrap.value) return;
  width.value = wrap.value.clientWidth || width.value;
  observer = new ResizeObserver(([entry]) => {
    if (entry) width.value = Math.max(240, Math.round(entry.contentRect.width));
  });
  observer.observe(wrap.value);
});
onBeforeUnmount(() => observer?.disconnect());

const plotW = computed(() => Math.max(60, width.value - M.left - M.right));
const plotH = HEIGHT - M.top - M.bottom;
const epochs = computed(() => Math.max(1, props.epochs));
/** The last epoch with a value in any series. */
const last = computed(() => Math.max(0, ...props.series.map((line) => {
  for (let index = line.values.length - 1; index >= 0; index--) if (line.values[index] !== null && line.values[index] !== undefined) return index + 1;
  return 0;
})));
const yAxis = computed(() => {
  if (props.top !== undefined) {
    const top = props.top;
    return {top, ticks: [0, top / 4, top / 2, top * 3 / 4, top]};
  }
  const values = props.series.flatMap((line) => line.values).filter((value): value is number => value !== null);
  return valueTicks(values.length ? Math.max(...values) : 1);
});
const x = (epoch: number): number => M.left + (epochs.value === 1 ? plotW.value / 2 : (epoch - 1) * plotW.value / (epochs.value - 1));
const y = (value: number): number => M.top + plotH - Math.max(0, Math.min(1, value / yAxis.value.top)) * plotH;
const xTicks = computed(() => epochTicks(epochs.value, plotW.value));

function pathOf(values: (number | null)[]): string {
  let path = '';
  let pen = false;
  values.forEach((value, index) => {
    if (value === null) {
      pen = false;
      return;
    }
    path += (pen ? 'L' : 'M') + x(index + 1).toFixed(1) + ' ' + y(value).toFixed(1);
    pen = true;
  });
  return path;
}
const lines = computed(() => props.series.map((line) => {
  let end = -1;
  for (let index = line.values.length - 1; index >= 0; index--) {
    if (line.values[index] !== null && line.values[index] !== undefined) {
      end = index;
      break;
    }
  }
  const endValue = end >= 0 ? line.values[end] ?? null : null;
  return {...line, path: pathOf(line.values), end: endValue === null ? null : {epoch: end + 1, value: endValue}};
}));
/** The values at the line ends, unless two would collide (then the legend and the tooltip carry them). */
const endLabels = computed(() => {
  const ends = lines.value.filter((line) => line.end !== null)
    .map((line) => ({key: line.key, text: props.format(line.end?.value ?? 0), cx: x(line.end?.epoch ?? 1), cy: y(line.end?.value ?? 0)}));
  for (let i = 0; i < ends.length; i++) {
    for (let j = i + 1; j < ends.length; j++) {
      const a = ends[i];
      const b = ends[j];
      if (a && b && Math.abs(a.cy - b.cy) < 14 && Math.abs(a.cx - b.cx) < 60) return [];
    }
  }
  return ends;
});
const bestPoint = computed(() => {
  const best = props.best;
  if (!best) return null;
  const value = props.series.find((line) => line.key === best.series)?.values[best.epoch - 1];
  if (value === null || value === undefined) return null;
  const line = props.series.find((item) => item.key === best.series);
  return {cx: x(best.epoch), cy: y(value), color: line?.color ?? 'currentColor', label: best.label};
});
/** The epoch being trained now: from the last finished epoch to the next. */
const inProgress = computed(() => props.running && last.value < epochs.value
  ? {x: x(Math.max(1, last.value)), w: x(last.value + 1) - x(Math.max(1, last.value)), epoch: last.value + 1} : null);

// -- the epoch under the pointer, the finger or the keyboard --------------------------------------------
const hovered = ref<number | null>(null);
function epochAt(clientX: number): number | null {
  const box = wrap.value?.getBoundingClientRect();
  if (!box || last.value === 0) return null;
  const px = clientX - box.left;
  const epoch = epochs.value === 1 ? 1 : Math.round((px - M.left) * (epochs.value - 1) / plotW.value) + 1;
  return Math.max(1, Math.min(last.value, epoch));
}
function onMove(event: PointerEvent): void {
  hovered.value = epochAt(event.clientX);
}
function onLeave(event: PointerEvent): void {
  if (event.pointerType === 'mouse') hovered.value = null;
}
function onKey(event: KeyboardEvent): void {
  if (last.value === 0) return;
  const now = hovered.value ?? last.value;
  const next = event.key === 'ArrowLeft' ? now - 1 : event.key === 'ArrowRight' ? now + 1
    : event.key === 'Home' ? 1 : event.key === 'End' ? last.value : null;
  if (next === null) {
    if (event.key === 'Escape') hovered.value = null;
    return;
  }
  event.preventDefault();
  hovered.value = Math.max(1, Math.min(last.value, next));
}
function onDocumentDown(event: PointerEvent): void {
  if (!wrap.value?.contains(event.target as Node)) hovered.value = null;
}
onMounted(() => document.addEventListener('pointerdown', onDocumentDown));
onBeforeUnmount(() => document.removeEventListener('pointerdown', onDocumentDown));

const tip = computed(() => {
  const epoch = hovered.value;
  if (epoch === null) return null;
  const rows = props.series.map((line) => ({key: line.key, label: line.label, color: line.color, value: line.values[epoch - 1] ?? null}));
  return {epoch, rows, left: x(epoch), flip: x(epoch) > width.value * 0.55};
});
const tipText = computed(() => tip.value === null ? '' : 'Epoca ' + tip.value.epoch + ': ' + tip.value.rows
  .map((row) => row.label + ' ' + (row.value === null ? 'n/d' : props.format(row.value))).join(', '));
</script>

<template>
  <figure class="chart">
    <figcaption class="chart-head">
      <span class="chart-title">{{ title }}</span>
      <span class="chart-legend"><span v-for="line in series" :key="line.key" class="chart-key"><span class="chart-swatch"
        :style="{background: line.color}" aria-hidden="true"></span>{{ line.label }}</span></span>
    </figcaption>
    <div ref="wrap" class="chart-plot" tabindex="0" role="group" :aria-label="title + ': frecce per scorrere le epoche'"
         @pointermove="onMove" @pointerdown="onMove" @pointerleave="onLeave" @keydown="onKey"
         @focus="hovered = hovered ?? (last || null)" @blur="hovered = null">
      <svg :width="width" :height="HEIGHT" :viewBox="'0 0 ' + width + ' ' + HEIGHT" aria-hidden="true">
        <g class="chart-grid">
          <line v-for="tick in yAxis.ticks" :key="'g' + tick" :x1="M.left" :x2="M.left + plotW" :y1="y(tick)" :y2="y(tick)"/>
        </g>
        <g class="chart-axis">
          <text v-for="tick in yAxis.ticks" :key="'y' + tick" :x="M.left - 8" :y="y(tick) + 4" text-anchor="end">{{ format(tick) }}</text>
          <text v-for="tick in xTicks" :key="'x' + tick" :x="x(tick)" :y="HEIGHT - 8" text-anchor="middle">{{ tick }}</text>
        </g>
        <g v-if="inProgress" class="chart-now">
          <rect :x="inProgress.x" :y="M.top" :width="Math.max(2, inProgress.w)" :height="plotH" rx="3"/>
          <text :x="inProgress.x + inProgress.w / 2" :y="M.top - 4" text-anchor="middle">epoca {{ inProgress.epoch }}</text>
        </g>
        <line v-if="tip" class="chart-cross" :x1="x(tip.epoch)" :x2="x(tip.epoch)" :y1="M.top" :y2="M.top + plotH"/>
        <path v-for="line in lines" :key="'l' + line.key" class="chart-line" :d="line.path" :style="{stroke: line.color}"/>
        <g v-if="bestPoint" class="chart-best">
          <circle :cx="bestPoint.cx" :cy="bestPoint.cy" r="6.5" :style="{stroke: bestPoint.color}"/>
          <text :x="bestPoint.cx" :y="bestPoint.cy - 11" text-anchor="middle">{{ bestPoint.label }}</text>
        </g>
        <template v-for="line in lines" :key="'e' + line.key">
          <circle v-if="line.end" class="chart-dot" :class="{live: running}" :cx="x(line.end.epoch)" :cy="y(line.end.value)" r="4"
                  :style="{fill: line.color}"/>
        </template>
        <text v-for="label in endLabels" :key="'t' + label.key" class="chart-end" :x="label.cx + 8" :y="label.cy + 4">{{ label.text }}</text>
        <template v-if="tip">
          <template v-for="row in tip.rows" :key="'h' + row.key">
            <circle v-if="row.value !== null" class="chart-dot" :cx="x(tip.epoch)" :cy="y(row.value)" r="4.5" :style="{fill: row.color}"/>
          </template>
        </template>
      </svg>
      <div v-if="last === 0" class="chart-empty">Nessuna epoca ancora</div>
      <div v-if="tip" class="chart-tip" :class="{flip: tip.flip}" :style="{left: tip.left + 'px'}" aria-hidden="true">
        <div class="chart-tip-head">Epoca {{ tip.epoch }}</div>
        <div v-for="row in tip.rows" :key="row.key" class="chart-tip-row"><span class="chart-tip-key" :style="{background: row.color}"></span><strong>{{
          row.value === null ? 'n/d' : format(row.value) }}</strong><span>{{ row.label }}</span></div>
      </div>
      <span class="sr-only" aria-live="polite">{{ tipText }}</span>
    </div>
  </figure>
</template>
