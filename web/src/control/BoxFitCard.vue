<script setup lang="ts">
// Fit box on Models: the review page's key C, which fits a rough box to its sign on this server's CPU
// (nnslr_tools.box_fit): two rankers (colour trees and edge networks), then the box refiner, a small network, when it
// beat them. Train them again on the boxes people drew (scripts/train_box_fit.sh, the panel action "box_fit_train"), the
// refiner too or not, a dry run first if wanted (it only counts the boxes); its job while it runs; then the last summary
// of each part from GET /api/panel/box-fit: the mean IoU with the person's box on drives left out, of the rough box, the
// rankers and the refiner, and whether refiner.onnx was written. Trained on a V100 (#65), the candidates cut on the CPU
// first; its dry run takes no card.
// The site loads the new models at its next restart. Hidden on a server without the action, and the summaries left out
// when the endpoint is not there yet.
import {computed, onMounted, ref, shallowRef, watch} from 'vue';
import {api} from '../common/api';
import {isoWhen, plural, sentence} from '../common/format';
import type {ActionName, ActionOptions, ActionState, BoxFitMeasure, BoxFitPayload, Job, PanelStatus} from '../common/panel-types';
import Icon from '../components/Icon.vue';
import JobProgress from './JobProgress.vue';
import {JOB_STATE, jobOf, startedWhen, terminalActivities} from './jobs';
import TerminalActivity from './TerminalActivity.vue';

const props = defineProps<{status: PanelStatus | null}>();
const emit = defineEmits<{run: [name: ActionName, state: ActionState, start?: undefined, options?: ActionOptions];
  stop: [job: Job]; details: [job: Job]}>();

const state = computed((): ActionState | null => props.status?.actions?.box_fit_train ?? null);
/** A server from before the action: no card. */
const shown = computed(() => state.value !== null);
const job = computed(() => jobOf(props.status, ['box_fit_train']));
const bigJob = computed(() => job.value && (job.value.state === 'running' || job.value.state === 'failed') ? job.value : null);
const activities = computed(() => terminalActivities(props.status, ['box_fit']));
const idle = computed(() => !bigJob.value && !activities.value.length);

const payload = shallowRef<BoxFitPayload | null>(null);
async function load(): Promise<void> {
  try {
    payload.value = await api<BoxFitPayload>('/api/panel/box-fit');
  } catch {
    // Not on this server yet (it comes with a restart), or not readable now: the card shows no summary.
    payload.value = null;
  }
}
onMounted(() => void load());
const working = computed(() => job.value?.state === 'running' || activities.value.length > 0);
watch(working, (now, before) => {
  if (before && !now) void load();
});

/** The rankers only, without the hour of the refiner. */
const withoutRefiner = ref(false);
function start(dry: boolean): void {
  if (!state.value) return;
  const options: ActionOptions = {...(dry ? {dry: 'on'} : {}), ...(withoutRefiner.value ? {refiner: 'off'} : {})};
  emit('run', 'box_fit_train', state.value, undefined, Object.keys(options).length ? options : undefined);
}

const rankers = computed(() => payload.value?.rankers ?? null);
const refiner = computed(() => payload.value?.refiner ?? null);
/** A mean IoU as written in the summaries, 0 to 1. */
const iou = (value: number | null | undefined): string => typeof value === 'number' ? value.toFixed(3) : '–';
const rankersFacts = computed(() => {
  const item = rankers.value;
  if (!item) return '';
  return [isoWhen(item.at), item.boxes !== null ? plural(item.boxes, 'box', 'boxes') : '',
    item.routes !== null ? plural(item.routes, 'drive') : '', item.folds ? item.folds + ' groups of drives' : '']
    .filter(Boolean).join(' · ');
});
const refinerFacts = computed(() => {
  const item = refiner.value;
  if (!item) return '';
  return [isoWhen(item.at), item.boxes !== null ? plural(item.boxes, 'box', 'boxes') + ' to learn' : '',
    item.held_out_boxes !== null ? item.held_out_boxes + ' left out' : '',
    typeof item.hand === 'number' ? item.hand + ' placed by hand' : '',
    item.best_epoch !== null && item.epochs !== null ? 'best epoch ' + item.best_epoch + ' of ' + item.epochs : '',
    item.minutes !== null ? item.minutes + ' min' : ''].filter(Boolean).join(' · ');
});
const measureText = (measure: BoxFitMeasure): string => 'rough ' + iou(measure.rough) + ' · rankers ' + iou(measure.rankers)
  + ' · refiner ' + iou(measure.refiner);
/** What key C does, in the title's tooltip. */
const what = 'Key C of the review page fits a rough box to its sign on this server: two rankers (colour trees and edge '
  + 'networks), then the box refiner where it beat them. They learn from the boxes people drew; the review page uses new '
  + 'models after the site restarts. Trained on a V100.';
/** What the last training did to the rankers, in words: replaced (always), their mean IoU with the person's box on
 * drives left out, and the training before's. */
const rankersLine = computed(() => {
  const item = rankers.value;
  if (!item) return '';
  const before = item.before;
  return 'Rankers replaced ' + isoWhen(item.at) + ' · IoU ' + iou(item.fit) + ' on drives left out'
    + (before ? ' (the training before: ' + iou(before.fit) + (before.boxes !== null ? ', ' + plural(before.boxes, 'box', 'boxes') : '') + ')' : '');
});
/** What the last training of the refiner did: replaced, or kept the one in use, and why, the new one against what it
 * had to beat on the same boxes. */
const refinerLine = computed(() => {
  const item = refiner.value;
  const all = item?.all;
  if (!item || !all) return '';
  const when = isoWhen(item.at);
  if (item.written) {
    return 'Refiner replaced ' + when + ' · IoU ' + iou(all.refiner)
      + (typeof item.in_use === 'number' ? ', the one before ' + iou(item.in_use) : '') + ', the rankers ' + iou(all.rankers);
  }
  const why = item.kept_because === 'in_use' ? 'the one in use ' + iou(item.in_use)
    : item.kept_because === 'small' ? 'the rankers on the small signs ' + iou(item.small?.rankers)
      + ' (it ' + iou(item.small?.refiner) + ')'
    : 'the rankers ' + iou(all.rankers);
  return 'Refiner kept, the new one not better ' + when + ' · IoU ' + iou(all.refiner) + ' against ' + why;
});
/** The details of the last training, in the tooltip. */
const trainedTitle = computed(() => {
  const lines = [rankers.value ? 'Rankers: ' + rankersFacts.value : '', refiner.value ? 'Refiner: ' + refinerFacts.value : ''];
  const item = refiner.value;
  if (item?.small) lines.push('Small signs: ' + measureText(item.small));
  // A box Fit box placed and the reviewer accepted favours the refiner in use: these were placed otherwise.
  if (item?.fair) lines.push('Not placed by Fit box: ' + measureText(item.fair));
  return lines.filter(Boolean).join('\n');
});
</script>

<template>
  <section v-if="shown" class="card import-card labeler-card" id="box-fit" aria-labelledby="bf-title">
    <h3 id="bf-title" class="rl-title" :title="what">Fit box</h3>
    <JobProgress v-if="bigJob" :job="bigJob" @details="emit('details', $event)" @stop="emit('stop', $event)"/>
    <TerminalActivity v-for="activity in activities" :key="activity.pid + activity.label" :activity="activity" :preparation="null"/>
    <div v-if="idle" class="import-row">
      <span class="import-icon" :class="{fresh: !rankers}" aria-hidden="true"><Icon name="train"/></span>
      <div class="import-what" :title="trainedTitle">
        <strong>{{ rankers ? (rankers.boxes !== null ? plural(rankers.boxes, 'box', 'boxes') : 'Trained') : 'Not trained yet' }}</strong>
        <span v-if="rankersLine" id="box-fit-rankers">{{ rankersLine }}</span>
        <span v-if="refinerLine" id="box-fit-refiner">{{ refinerLine }}</span>
      </div>
      <div class="rl-buttons">
        <button type="button" class="btn" id="box-fit-dry" :disabled="!state?.allowed" title="Only counts the boxes"
                @click="start(true)">Dry run</button>
        <button type="button" class="btn primary" id="box-fit-train" :disabled="!state?.allowed"
                @click="start(false)">Train</button>
      </div>
    </div>
    <p v-if="idle && state && !state.allowed" class="import-why">{{ sentence(state.reason) }}</p>
    <label v-if="idle" class="rl-marks"><input type="checkbox" id="box-fit-no-refiner" v-model="withoutRefiner"> Rankers only</label>
    <p v-if="job && !bigJob" class="import-last">Last run {{ startedWhen(job) }} · {{ job.state === 'interrupted' ? 'stopped' : JOB_STATE[job.state][0] }} ·
      <button type="button" class="link-button" @click="emit('details', job)">Details</button></p>
  </section>
</template>
