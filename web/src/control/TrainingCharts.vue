<script setup lang="ts">
// How a training is going, drawn: a few figures (the epoch now, the best validation result and when, time
// per epoch and about how long the network still needs), the detector's loss, the reader's accuracy and
// loss, train against validation. Data: progress.curves of the job, one point per finished epoch.
import {computed} from 'vue';
import type {EpochPoint, JobProgress, TrainingCurve} from '../common/panel-types';
import LineChart, {type ChartSeries} from './charts/LineChart.vue';
import {elapsedText, roughText} from './jobs';

const props = defineProps<{progress: JobProgress; running: boolean}>();
type Net = 'detector' | 'reader';
const NET_NAME: Record<Net, string> = {detector: 'Detector', reader: 'Reader'};
const TRAIN = 'var(--c-train)';
const VAL = 'var(--c-val)';

const running = computed(() => props.running);
const curves = computed(() => props.progress.curves ?? {});
const detector = computed(() => curves.value.detector ?? null);
const reader = computed(() => curves.value.reader ?? null);
const hasPoints = (curve: TrainingCurve | null): curve is TrainingCurve => curve !== null && curve.points.length > 0;
const hasAny = computed(() => hasPoints(detector.value) || hasPoints(reader.value));

/** The values of one metric by epoch (index = epoch - 1), null where missing. */
function column(curve: TrainingCurve, pick: (point: EpochPoint) => number | undefined, scale = 1): (number | null)[] {
  const last = Math.max(0, ...curve.points.map((point) => point.epoch));
  const values: (number | null)[] = Array.from({length: last}, () => null);
  for (const point of curve.points) {
    const value = pick(point);
    if (value !== undefined && Number.isFinite(value) && point.epoch >= 1) values[point.epoch - 1] = value * scale;
  }
  return values;
}
function best(curve: TrainingCurve, pick: (point: EpochPoint) => number | undefined, highest: boolean): EpochPoint & {value: number} | null {
  let found: (EpochPoint & {value: number}) | null = null;
  for (const point of curve.points) {
    const value = pick(point);
    if (value === undefined || !Number.isFinite(value)) continue;
    if (found === null || (highest ? value > found.value : value < found.value)) found = {...point, value};
  }
  return found;
}
const loss = (value: number): string => value >= 10 ? value.toFixed(1) : value >= 1 ? value.toFixed(2) : value.toFixed(3);
const lossAxis = (value: number): string => value === 0 ? '0' : Number(value.toPrecision(3)).toString();
const percent = (value: number): string => (Math.round(value * 10) / 10).toString() + '%';
const percentAxis = (value: number): string => Math.round(value) + '%';

function lossSeries(curve: TrainingCurve): ChartSeries[] {
  return [
    {key: 'train', label: 'train', color: TRAIN, values: column(curve, (point) => point.train_loss)},
    {key: 'val', label: 'validation', color: VAL, values: column(curve, (point) => point.val_loss)},
  ];
}
const detectorBest = computed(() => hasPoints(detector.value) ? best(detector.value, (point) => point.val_loss, false) : null);
const readerBestLoss = computed(() => hasPoints(reader.value) ? best(reader.value, (point) => point.val_loss, false) : null);
const readerBestAccuracy = computed(() => hasPoints(reader.value) ? best(reader.value, (point) => point.val_accuracy, true) : null);

/** The network being trained now (from the steps), or the last one with points. */
const current = computed((): Net | null => {
  const step = props.progress.steps.find((item) => item.state === 'current')?.key;
  if (running.value && (step === 'detector' || step === 'reader')) return step;
  if (hasPoints(reader.value)) return 'reader';
  if (hasPoints(detector.value)) return 'detector';
  return null;
});
const currentCurve = computed(() => current.value === null ? null : curves.value[current.value] ?? null);
const lastEpoch = computed(() => Math.max(0, ...(currentCurve.value?.points ?? []).map((point) => point.epoch)));
/** Mean of the last three epochs (the first is often slower). */
const perEpoch = computed(() => {
  const seconds = (currentCurve.value?.points ?? []).map((point) => point.seconds).filter((value): value is number => value !== undefined && value > 0);
  const recent = seconds.slice(-3);
  return recent.length ? recent.reduce((sum, value) => sum + value, 0) / recent.length : null;
});
const left = computed(() => {
  const curve = currentCurve.value;
  if (!running.value || curve === null || perEpoch.value === null) return '';
  const epochsLeft = curve.epochs - lastEpoch.value;
  return epochsLeft > 0 ? 'the ' + current.value + ' ends in about ' + roughText(epochsLeft * perEpoch.value * 1000) : '';
});
/** Still in the steps before the first network. */
const beforeEpochs = computed(() => running.value && ['import', 'split', 'gpu_env', 'checks']
  .includes(props.progress.steps.find((item) => item.state === 'current')?.key ?? ''));
const isRunning = (net: Net): boolean => running.value && current.value === net;
/** Every value, for the table under the charts. */
const tables = computed(() => (['detector', 'reader'] as Net[]).flatMap((net) => {
  const curve = curves.value[net];
  return curve && curve.points.length ? [{net, rows: [...curve.points].sort((a, b) => a.epoch - b.epoch)}] : [];
}));
</script>

<template>
  <section class="card training" aria-labelledby="training-title">
    <h2 id="training-title">How the training is going</h2>
    <div v-if="!hasAny" class="training-wait">
      <svg viewBox="0 0 48 32" aria-hidden="true"><path d="M3 27h42M3 27V4" fill="none" stroke="currentColor" stroke-width="1.5"
        stroke-linecap="round"/><path d="M7 9c6 2 9 9 15 12s12 2 21 3" fill="none" stroke="currentColor" stroke-width="2"
        stroke-linecap="round" stroke-dasharray="2 4"/></svg>
      <p><strong>The curves appear at the end of the first epoch.</strong><span v-if="beforeEpochs">First the training imports the reviews,
        checks the dataset and prepares the GPU: a few minutes.</span></p>
    </div>
    <template v-else>
      <div class="tiles">
        <div class="tile">
          <span class="tile-label">{{ running ? 'Epoch' : 'Epochs done' }}</span>
          <span class="tile-value">{{ lastEpoch }}<small> of {{ currentCurve?.epochs ?? '–' }}</small></span>
          <span class="tile-note">{{ current === null ? '' : current + (running ? ', running' : '') }}</span>
        </div>
        <div v-if="detectorBest" class="tile">
          <span class="tile-label">Best validation loss</span>
          <span class="tile-value">{{ loss(detectorBest.value) }}</span>
          <span class="tile-note">detector, epoch {{ detectorBest.epoch }}</span>
        </div>
        <div v-if="readerBestAccuracy" class="tile">
          <span class="tile-label">Best validation accuracy</span>
          <span class="tile-value">{{ percent(readerBestAccuracy.value * 100) }}</span>
          <span class="tile-note">reader, epoch {{ readerBestAccuracy.epoch }}</span>
        </div>
        <div v-if="perEpoch !== null" class="tile">
          <span class="tile-label">Time per epoch</span>
          <span class="tile-value">{{ elapsedText(perEpoch * 1000) }}</span>
          <span class="tile-note">{{ left || (current + ', mean of the last epochs') }}</span>
        </div>
      </div>

      <div class="net">
        <h3>{{ NET_NAME.detector }}<span>finds the signs in the frame · loss, lower is better</span></h3>
        <LineChart v-if="hasPoints(detector)" title="Detector loss" :series="lossSeries(detector)" :epochs="detector.epochs"
                   :format="lossAxis" :best="detectorBest ? {series: 'val', epoch: detectorBest.epoch, label: 'best'} : null"
                   :running="isRunning('detector')"/>
        <p v-else class="net-wait">The detector’s curves appear at the end of its first epoch.</p>
      </div>

      <div class="net">
        <h3>{{ NET_NAME.reader }}<span>reads the limit on the sign · accuracy, higher is better, and loss</span></h3>
        <div v-if="hasPoints(reader)" class="net-charts">
          <LineChart title="Reader accuracy" :top="100" :format="percentAxis" :epochs="reader.epochs"
                     :series="[{key: 'train', label: 'train', color: TRAIN, values: column(reader, (point) => point.train_accuracy, 100)},
                               {key: 'val', label: 'validation', color: VAL, values: column(reader, (point) => point.val_accuracy, 100)}]"
                     :best="readerBestAccuracy ? {series: 'val', epoch: readerBestAccuracy.epoch, label: 'best'} : null"
                     :running="isRunning('reader')"/>
          <LineChart title="Reader loss" :series="lossSeries(reader)" :epochs="reader.epochs" :format="lossAxis"
                     :best="readerBestLoss ? {series: 'val', epoch: readerBestLoss.epoch, label: 'best'} : null"
                     :running="isRunning('reader')"/>
        </div>
        <p v-else class="net-wait">{{ running ? 'The reader starts after the detector: its curves appear at the end of its first epoch.'
          : 'The reader did not reach its first epoch.' }}</p>
      </div>

      <details class="chart-table">
        <summary>Values per epoch</summary>
        <div v-for="table in tables" :key="table.net" class="table-wrap">
          <table>
            <caption>{{ NET_NAME[table.net] }}</caption>
            <thead><tr><th class="num">Epoch</th><th class="num">Train loss</th><th class="num">Val loss</th>
              <template v-if="table.net === 'reader'"><th class="num">Train acc.</th><th class="num">Val acc.</th></template><th class="num">Time</th></tr></thead>
            <tbody>
              <tr v-for="point in table.rows" :key="point.epoch">
                <td class="num">{{ point.epoch }}</td>
                <td class="num">{{ point.train_loss === undefined ? '–' : loss(point.train_loss) }}</td>
                <td class="num">{{ point.val_loss === undefined ? '–' : loss(point.val_loss) }}</td>
                <template v-if="table.net === 'reader'">
                  <td class="num">{{ point.train_accuracy === undefined ? '–' : percent(point.train_accuracy * 100) }}</td>
                  <td class="num">{{ point.val_accuracy === undefined ? '–' : percent(point.val_accuracy * 100) }}</td>
                </template>
                <td class="num">{{ point.seconds === undefined ? '–' : elapsedText(point.seconds * 1000) }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </details>
    </template>
  </section>
</template>
