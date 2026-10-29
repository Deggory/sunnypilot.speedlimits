<script setup lang="ts">
// How a training is going, drawn: a few figures (the epoch now, the best validation result and when, time
// per epoch and about how long the network still needs), the detector's loss, the reader's accuracy and
// loss, train against validation; for the classifier with “not a sign”, the sign crops read right and the “not a
// sign” crops rejected on validation, their mean that chooses its best epoch, and its loss; before it, in two stages,
// its pretraining on the Mapillary signs alone (accuracy on our validation signs, and loss). Data: progress.curves of
// the job, one point per finished epoch.
import {computed} from 'vue';
import type {CurveNet, EpochPoint, JobProgress, TrainingCurve} from '../common/panel-types';
import LineChart, {type ChartSeries} from './charts/LineChart.vue';
import {elapsedText, roughText} from './jobs';

// title: the heading of the card (the Evaluate tab shows the curves of a past run); nets: the networks to show, else
// those of the job's steps, else those with curves, else the detector and the reader.
const props = defineProps<{progress: JobProgress; running: boolean; title?: string; nets?: CurveNet[]}>();
type Net = CurveNet;
const ALL_NETS: Net[] = ['detector', 'reader', 'pretrain', 'classifier'];
const NET_NAME: Record<Net, string> = {detector: 'Detector', reader: 'Reader', pretrain: 'Pretraining on Mapillary',
  classifier: 'Classifier'};
/** A network in a sentence ("the classifier ends in ..."). */
const NET_WORD: Record<Net, string> = {detector: 'detector', reader: 'reader', pretrain: 'pretraining', classifier: 'classifier'};
const TRAIN = 'var(--c-train)';
const VAL = 'var(--c-val)';
const REJECT = 'var(--split-3)';
const BALANCED = 'var(--split-4)';

const running = computed(() => props.running);
const curves = computed(() => props.progress.curves ?? {});
const detector = computed(() => curves.value.detector ?? null);
const reader = computed(() => curves.value.reader ?? null);
const classifier = computed(() => curves.value.classifier ?? null);
const pretrain = computed(() => curves.value.pretrain ?? null);
const hasPoints = (curve: TrainingCurve | null | undefined): curve is TrainingCurve => curve !== null && curve !== undefined
  && curve.points.length > 0;
/** The networks shown, in the order they train. */
const nets = computed((): Net[] => {
  if (props.nets?.length) return props.nets;
  const stepKeys = props.progress.steps.map((step) => step.key);
  // The pretraining is inside the classifier's step: shown once it has a curve.
  const fromSteps = ALL_NETS.filter((net) => stepKeys.includes(net)
    || (net === 'pretrain' && stepKeys.includes('classifier') && hasPoints(curves.value.pretrain)));
  if (fromSteps.length) return fromSteps;
  const withCurves = ALL_NETS.filter((net) => hasPoints(curves.value[net]));
  return withCurves.length ? withCurves : ['detector', 'reader'];
});
const shows = (net: Net): boolean => nets.value.includes(net);
const hasAny = computed(() => nets.value.some((net) => hasPoints(curves.value[net])));
/** The x axis of a curve: its planned epochs, or the epoch it stopped at when it stopped early. */
const span = (curve: TrainingCurve): number => curve.stopped_early ?? curve.epochs;

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
const cell = (value: number | undefined, text: (value: number) => string): string => value === undefined ? '–' : text(value);

function lossSeries(curve: TrainingCurve): ChartSeries[] {
  return [
    {key: 'train', label: 'train', color: TRAIN, values: column(curve, (point) => point.train_loss)},
    {key: 'val', label: 'validation', color: VAL, values: column(curve, (point) => point.val_loss)},
  ];
}
const detectorBest = computed(() => shows('detector') && hasPoints(detector.value) ? best(detector.value, (point) => point.val_loss, false) : null);
/** The cutter's best epoch: its validation recall at score 0.1, when the log has it. */
const detectorRecall = computed(() => hasPoints(detector.value) && detector.value.points.some((point) => point.val_recall !== undefined));
const detectorBestRecall = computed(() => shows('detector') && detectorRecall.value && hasPoints(detector.value)
  ? best(detector.value, (point) => point.val_recall, true) : null);
const readerBestLoss = computed(() => hasPoints(reader.value) ? best(reader.value, (point) => point.val_loss, false) : null);
const readerBestAccuracy = computed(() => shows('reader') && hasPoints(reader.value)
  ? best(reader.value, (point) => point.val_accuracy, true) : null);

/** The classifier with “not a sign” writes the sign crops and the rejection apart; an older log only the accuracy. */
const withRejection = computed(() => hasPoints(classifier.value)
  && classifier.value.points.some((point) => point.val_sign_accuracy !== undefined || point.val_rejection !== undefined));
const classifierBestLoss = computed(() => hasPoints(classifier.value) ? best(classifier.value, (point) => point.val_loss, false) : null);
/** Its best epoch: by the balanced accuracy (the mean of signs and rejection), else by the accuracy. */
const classifierBest = computed(() => {
  if (!shows('classifier') || !hasPoints(classifier.value)) return null;
  return withRejection.value ? best(classifier.value, (point) => point.val_balanced_accuracy, true)
    : best(classifier.value, (point) => point.val_accuracy, true);
});
const classifierSeries = computed((): ChartSeries[] => {
  const curve = classifier.value;
  if (!hasPoints(curve)) return [];
  if (!withRejection.value) {
    return [{key: 'train', label: 'train', color: TRAIN, values: column(curve, (point) => point.train_accuracy, 100)},
      {key: 'val', label: 'validation', color: VAL, values: column(curve, (point) => point.val_accuracy, 100)}];
  }
  return [
    {key: 'signs', label: 'signs read right', color: VAL, values: column(curve, (point) => point.val_sign_accuracy, 100)},
    {key: 'reject', label: '“not a sign” rejected', color: REJECT, values: column(curve, (point) => point.val_rejection, 100)},
    {key: 'val', label: 'balanced', color: BALANCED, values: column(curve, (point) => point.val_balanced_accuracy, 100)},
  ];
});

/** The pretraining on Mapillary alone: its accuracy on our validation signs chooses its best epoch. */
const pretrainBest = computed(() => shows('pretrain') && hasPoints(pretrain.value)
  ? best(pretrain.value, (point) => point.val_accuracy, true) : null);
const pretrainBestLoss = computed(() => hasPoints(pretrain.value) ? best(pretrain.value, (point) => point.val_loss, false) : null);

/** The network being trained now (from the steps; the pretraining until the classifier's first epoch), or the last
 * one with points. */
const current = computed((): Net | null => {
  const step = props.progress.steps.find((item) => item.state === 'current')?.key;
  if (running.value && step === 'classifier' && shows('pretrain') && !hasPoints(classifier.value)) return 'pretrain';
  if (running.value && nets.value.some((net) => net === step)) return step as Net;
  return [...nets.value].reverse().find((net) => hasPoints(curves.value[net])) ?? null;
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
  if (!running.value || curve === null || perEpoch.value === null || curve.stopped_early !== undefined) return '';
  const epochsLeft = curve.epochs - lastEpoch.value;
  return epochsLeft > 0 ? 'the ' + NET_WORD[current.value as Net] + ' ends in about ' + roughText(epochsLeft * perEpoch.value * 1000) : '';
});
/** Still in the steps before the first network. */
const beforeEpochs = computed(() => running.value && ['import', 'split', 'gpu_env', 'checks', 'mine']
  .includes(props.progress.steps.find((item) => item.state === 'current')?.key ?? ''));
const beforeText = computed(() => shows('reader')
  ? 'First the training imports the reviews, checks the dataset and prepares the GPU: a few minutes.'
  : shows('detector') ? 'First it checks the frames of the training and of the evaluation, and the GPU: a few minutes.'
    : 'First it checks the dataset, the split and the cutter in use, then runs the cutter on every frame to mine its false boxes.');
const isRunning = (net: Net): boolean => running.value && current.value === net;
/** Every value, for the table under the charts. */
const tables = computed(() => nets.value.flatMap((net) => {
  const curve = curves.value[net];
  return curve && curve.points.length ? [{net, rows: [...curve.points].sort((a, b) => a.epoch - b.epoch)}] : [];
}));
</script>

<template>
  <section class="card training" aria-labelledby="training-title">
    <h2 id="training-title">{{ title ?? 'How the training is going' }}</h2>
    <div v-if="!hasAny" class="training-wait">
      <svg viewBox="0 0 48 32" aria-hidden="true"><path d="M3 27h42M3 27V4" fill="none" stroke="currentColor" stroke-width="1.5"
        stroke-linecap="round"/><path d="M7 9c6 2 9 9 15 12s12 2 21 3" fill="none" stroke="currentColor" stroke-width="2"
        stroke-linecap="round" stroke-dasharray="2 4"/></svg>
      <p><strong>The curves appear at the end of the first epoch.</strong><span v-if="beforeEpochs">{{ beforeText }}</span></p>
    </div>
    <template v-else>
      <div class="tiles">
        <div class="tile">
          <span class="tile-label">{{ running ? 'Epoch' : 'Epochs done' }}</span>
          <span class="tile-value">{{ lastEpoch }}<small> of {{ currentCurve?.epochs ?? '–' }}</small></span>
          <span class="tile-note">{{ current === null ? '' : NET_WORD[current] + (currentCurve?.stopped_early !== undefined
            ? ', stopped early after epoch ' + currentCurve.stopped_early : running ? ', running' : '') }}</span>
        </div>
        <div v-if="detectorBestRecall" class="tile" id="detector-best-recall">
          <span class="tile-label">Best validation recall</span>
          <span class="tile-value">{{ percent(detectorBestRecall.value * 100) }}</span>
          <span class="tile-note">detector at score 0.1, epoch {{ detectorBestRecall.epoch }}</span>
        </div>
        <div v-else-if="detectorBest" class="tile">
          <span class="tile-label">Best validation loss</span>
          <span class="tile-value">{{ loss(detectorBest.value) }}</span>
          <span class="tile-note">detector, epoch {{ detectorBest.epoch }}</span>
        </div>
        <div v-if="readerBestAccuracy" class="tile">
          <span class="tile-label">Best validation accuracy</span>
          <span class="tile-value">{{ percent(readerBestAccuracy.value * 100) }}</span>
          <span class="tile-note">reader, epoch {{ readerBestAccuracy.epoch }}</span>
        </div>
        <div v-if="classifierBest" class="tile" id="classifier-best">
          <span class="tile-label">{{ withRejection ? 'Best validation, balanced' : 'Best validation accuracy' }}</span>
          <span class="tile-value">{{ percent(classifierBest.value * 100) }}</span>
          <span class="tile-note">classifier, epoch {{ classifierBest.epoch }}<template v-if="withRejection && classifierBest.val_sign_accuracy !== undefined
            && classifierBest.val_rejection !== undefined">: signs {{ percent(classifierBest.val_sign_accuracy * 100) }}, “not a sign” {{
            percent(classifierBest.val_rejection * 100) }}</template></span>
        </div>
        <div v-if="pretrainBest && !classifierBest" class="tile" id="pretrain-best">
          <span class="tile-label">Best on our signs</span>
          <span class="tile-value">{{ percent(pretrainBest.value * 100) }}</span>
          <span class="tile-note">pretraining on Mapillary, epoch {{ pretrainBest.epoch }}</span>
        </div>
        <div v-if="perEpoch !== null" class="tile">
          <span class="tile-label">Time per epoch</span>
          <span class="tile-value">{{ elapsedText(perEpoch * 1000) }}</span>
          <span class="tile-note">{{ left || ((current === null ? '' : NET_WORD[current]) + ', mean of the last epochs') }}</span>
        </div>
      </div>

      <div v-if="shows('detector')" class="net">
        <h3>{{ NET_NAME.detector }}<span>finds the signs in the frame · {{ detectorRecall ? 'recall on validation, higher is better, and loss'
          : 'loss, lower is better' }}</span></h3>
        <div v-if="hasPoints(detector) && detectorRecall" class="net-charts">
          <LineChart title="Detector recall on validation" :top="100" :format="percentAxis" :epochs="span(detector)"
                     :series="[{key: 'val', label: 'validation, score 0.1', color: VAL, values: column(detector, (point) => point.val_recall, 100)}]"
                     :best="detectorBestRecall ? {series: 'val', epoch: detectorBestRecall.epoch, label: 'best'} : null"
                     :running="isRunning('detector')"/>
          <LineChart title="Detector loss" :series="lossSeries(detector)" :epochs="span(detector)" :format="lossAxis"
                     :best="detectorBest ? {series: 'val', epoch: detectorBest.epoch, label: 'lowest'} : null"
                     :running="isRunning('detector')"/>
        </div>
        <LineChart v-else-if="hasPoints(detector)" title="Detector loss" :series="lossSeries(detector)" :epochs="span(detector)"
                   :format="lossAxis" :best="detectorBest ? {series: 'val', epoch: detectorBest.epoch, label: 'best'} : null"
                   :running="isRunning('detector')"/>
        <p v-else class="net-wait">The detector’s curves appear at the end of its first epoch.</p>
      </div>

      <div v-if="shows('reader')" class="net">
        <h3>{{ NET_NAME.reader }}<span>reads the limit on the sign · accuracy, higher is better, and loss</span></h3>
        <div v-if="hasPoints(reader)" class="net-charts">
          <LineChart title="Reader accuracy" :top="100" :format="percentAxis" :epochs="span(reader)"
                     :series="[{key: 'train', label: 'train', color: TRAIN, values: column(reader, (point) => point.train_accuracy, 100)},
                               {key: 'val', label: 'validation', color: VAL, values: column(reader, (point) => point.val_accuracy, 100)}]"
                     :best="readerBestAccuracy ? {series: 'val', epoch: readerBestAccuracy.epoch, label: 'best'} : null"
                     :running="isRunning('reader')"/>
          <LineChart title="Reader loss" :series="lossSeries(reader)" :epochs="span(reader)" :format="lossAxis"
                     :best="readerBestLoss ? {series: 'val', epoch: readerBestLoss.epoch, label: 'best'} : null"
                     :running="isRunning('reader')"/>
        </div>
        <p v-else class="net-wait">{{ running ? 'The reader starts after the detector: its curves appear at the end of its first epoch.'
          : 'The reader did not reach its first epoch.' }}</p>
      </div>

      <div v-if="shows('pretrain')" class="net" id="pretrain-curves">
        <h3>{{ NET_NAME.pretrain }}<span>the classifier learns on the Mapillary signs alone, measured on our validation signs ·
          accuracy, higher is better, and loss</span></h3>
        <div v-if="hasPoints(pretrain)" class="net-charts">
          <LineChart title="Pretraining accuracy" :top="100" :format="percentAxis" :epochs="span(pretrain)"
                     :series="[{key: 'train', label: 'train, Mapillary', color: TRAIN, values: column(pretrain, (point) => point.train_accuracy, 100)},
                               {key: 'val', label: 'validation, our signs', color: VAL, values: column(pretrain, (point) => point.val_accuracy, 100)}]"
                     :best="pretrainBest ? {series: 'val', epoch: pretrainBest.epoch, label: 'best'} : null"
                     :running="isRunning('pretrain')"/>
          <LineChart title="Pretraining loss" :series="lossSeries(pretrain)" :epochs="span(pretrain)" :format="lossAxis"
                     :best="pretrainBestLoss ? {series: 'val', epoch: pretrainBestLoss.epoch, label: 'lowest'} : null"
                     :running="isRunning('pretrain')"/>
        </div>
        <p class="note">The classifier below starts from its best epoch, then learns on your signs and “not a sign”.</p>
      </div>

      <div v-if="shows('classifier')" class="net" id="classifier-curves">
        <h3>{{ NET_NAME.classifier }}<span>says what each box is, “not a sign” included · on validation, higher is better, and loss</span></h3>
        <div v-if="hasPoints(classifier)" class="net-charts">
          <LineChart :title="withRejection ? 'Classifier on validation' : 'Classifier accuracy'" :top="100" :format="percentAxis"
                     :epochs="span(classifier)" :series="classifierSeries"
                     :best="classifierBest ? {series: 'val', epoch: classifierBest.epoch, label: 'best'} : null"
                     :running="isRunning('classifier')"/>
          <LineChart title="Classifier loss" :series="lossSeries(classifier)" :epochs="span(classifier)" :format="lossAxis"
                     :best="classifierBestLoss ? {series: 'val', epoch: classifierBestLoss.epoch, label: 'lowest'} : null"
                     :running="isRunning('classifier')"/>
        </div>
        <p v-else class="net-wait">{{ running ? (shows('pretrain') ? 'The classifier starts from the pretraining’s best epoch: its curves appear at the end of its first epoch.'
          : 'The classifier starts after the mining of the cutter’s false boxes: its curves appear at the end of its first epoch.')
          : 'The classifier did not reach its first epoch.' }}</p>
        <p v-if="withRejection" class="note">Balanced: the mean of the sign crops read right and the “not a sign” crops rejected; it
          chooses the epoch whose weights are kept.</p>
      </div>

      <details class="chart-table">
        <summary>Values per epoch</summary>
        <div v-for="table in tables" :key="table.net" class="table-wrap">
          <table>
            <caption>{{ NET_NAME[table.net] }}</caption>
            <thead><tr><th class="num">Epoch</th><th class="num">Train loss</th><th class="num">Val loss</th>
              <template v-if="table.net !== 'detector'"><th class="num">Train acc.</th><th class="num">Val acc.</th></template>
              <th v-if="table.net === 'detector' && detectorRecall" class="num">Val recall</th>
              <template v-if="table.net === 'classifier' && withRejection"><th class="num">Signs</th><th class="num">Not a sign</th>
                <th class="num">Balanced</th></template><th class="num">Time</th></tr></thead>
            <tbody>
              <tr v-for="point in table.rows" :key="point.epoch">
                <td class="num">{{ point.epoch }}</td>
                <td class="num">{{ cell(point.train_loss, loss) }}</td>
                <td class="num">{{ cell(point.val_loss, loss) }}</td>
                <template v-if="table.net !== 'detector'">
                  <td class="num">{{ cell(point.train_accuracy, (value) => percent(value * 100)) }}</td>
                  <td class="num">{{ cell(point.val_accuracy, (value) => percent(value * 100)) }}</td>
                </template>
                <td v-if="table.net === 'detector' && detectorRecall" class="num">{{ cell(point.val_recall, (value) => percent(value * 100)) }}</td>
                <template v-if="table.net === 'classifier' && withRejection">
                  <td class="num">{{ cell(point.val_sign_accuracy, (value) => percent(value * 100)) }}</td>
                  <td class="num">{{ cell(point.val_rejection, (value) => percent(value * 100)) }}</td>
                  <td class="num">{{ cell(point.val_balanced_accuracy, (value) => percent(value * 100)) }}</td>
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
