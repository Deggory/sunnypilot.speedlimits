<script setup lang="ts">
// Evaluate, by network: choose what to judge (the whole chain, the cutter, the classifier, the decider); every
// version of it in one table, judged by the one number that matters for that network, next to the version in use;
// the version chosen played frame by frame on the test (WatchSection), and all the numbers of its run folded below.
// The proposers of Prepare (the Vietnamese YOLO, our networks, Qwen3-VL) are judged the same way, each measured on the
// reviewed test frames with its Measure button (helpers/measure_proposer.sh); so are the AI reviewers of Settings, each
// asked about the same frames a person reviewed (helpers/measure_ai_reviewer.sh).
import {computed, onBeforeUnmount, onMounted, ref, shallowRef, watch} from 'vue';
import type {AiProvider, AiReviewerReport, AiReviewersPayload} from '../common/ai-types';
import {api, messageOf} from '../common/api';
import {isoWhen, thousands} from '../common/format';
import type {ActionName, ActionOptions, ActionState, ClassifierModel, CutterModel, DeciderModel, ModelVersion, PanelStatus,
  Proposer, ProposersPayload, WatchModel} from '../common/panel-types';
import ActionList from './ActionList.vue';
import {pct, points, useRunDetail, useRuns} from './evaluate';
import {perFrame, sizeText, useModels} from './models';
import RunDetailView from './RunDetail.vue';
import Segmented from './Segmented.vue';
import WatchSection from './WatchSection.vue';

const props = defineProps<{status: PanelStatus | null}>();
const emit = defineEmits<{run: [name: ActionName, state: ActionState, start?: undefined, options?: ActionOptions]}>();

type View = 'proposers' | 'chain' | 'cutter' | 'classifier' | 'decider' | 'ai';
const VIEWS: {key: View; label: string}[] = [
  {key: 'proposers', label: 'Proposers'}, {key: 'chain', label: 'Chain'}, {key: 'cutter', label: 'Cutter'},
  {key: 'classifier', label: 'Classifier'}, {key: 'decider', label: 'Decider'}, {key: 'ai', label: 'AI reviewers'},
];
/** What each network must do, and the number that judges it. */
const GOAL: Record<View, string> = {
  proposers: 'Who proposes the signs in Prepare, each run on the same reviewed test frames. What counts: of the speed '
    + 'signs, how many it proposes with the right number, and how many false proposals per frame you would have to delete.',
  chain: 'The cutter and the classifier together, on the test roads no training saw. What counts: of the 80 speed '
    + 'signs, how many are found and read right, and how many false boxes are left per frame.',
  cutter: 'Finds every box that could be a sign. What counts: how many of the 80 test speed signs it finds. Many '
    + 'boxes per frame are expected: the classifier drops them.',
  classifier: 'Says what each box is, or “not a sign”. What counts: the test signs read right, and the false boxes '
    + 'it throws away.',
  decider: 'Says whom a sign applies to: our road or another. What counts: the “Another road” signs it finds. '
    + 'Only on validation: the test has no such sign yet.',
  ai: 'The models of Settings that review the signs, each asked about the same frames a person reviewed, as it would '
    + 'review them. What counts: of the speed signs, how many it keeps or draws with the right number; then the frames '
    + 'without signs it leaves empty, and the false boxes per frame you would have to delete.',
};

interface Row {
  id: string;
  name: string;
  when: string;
  /** The judging share (0-1) and its words. */
  main: number | null;
  mainText: string;
  second: string;
  inUse: boolean;
  run: string;
  /** The report of the run the player shows for this row. */
  watch: string | null;
  /** Its own comparison (the chain: with the old reader on the same frames). */
  vs?: {text: string; kind: string} | undefined;
  /** What the mark of the one in use says ("in use"; an AI reviewer: "reviews"). */
  badge?: string | undefined;
}

const storedView = localStorage.getItem('nnslr-evaluate-view');
const view = ref<View>(VIEWS.find((item) => item.key === storedView)?.key ?? 'chain');
watch(view, (value) => localStorage.setItem('nnslr-evaluate-view', value));
/** The proposers as measured (helpers/measure_proposer.sh), asked again every 15 s: a measure takes minutes. */
const proposers = shallowRef<ProposersPayload | null>(null);
const proposersError = ref('');
/** The AI reviewers and their measures (helpers/measure_ai_reviewer.sh), every 15 s while their view is shown. */
const aiReviewers = shallowRef<AiReviewersPayload | null>(null);
const aiError = ref('');
let timer: number | undefined;
async function askProposers(): Promise<void> {
  try {
    proposers.value = await api<ProposersPayload>('/api/panel/proposers');
    proposersError.value = '';
  } catch (problem) {
    proposersError.value = messageOf(problem);
  }
}
async function askAiReviewers(): Promise<void> {
  try {
    aiReviewers.value = await api<AiReviewersPayload>('/api/panel/ai-reviewers');
    aiError.value = '';
  } catch (problem) {
    aiError.value = messageOf(problem);
  }
}
function poll(): void {
  void askProposers();
  if (view.value === 'ai') void askAiReviewers();
}
onMounted(() => { poll(); timer = window.setInterval(poll, 15000); });
onBeforeUnmount(() => window.clearInterval(timer));
watch(view, (value) => { if (value === 'ai') void askAiReviewers(); });
const picked = ref<string | null>(null);
const {payload: models, error: modelsError} = useModels();
const {payload: runsPayload} = useRuns();

function versionsOf(role: string): ModelVersion[] {
  return models.value?.roles.find((item) => item.role === role)?.versions ?? [];
}
function share(found: number | null | undefined, truth: number | null | undefined): number | null {
  return found === null || found === undefined || !truth ? null : found / truth;
}
function ofText(found: number | null | undefined, truth: number | null | undefined, what: string): string {
  return found === null || found === undefined || !truth ? 'not measured' : pct(found / truth) + ' · ' + found + ' of ' + truth + ' ' + what;
}

/** A row of an AI reviewer: a provider of Settings with its latest measure for its model, or an older measure. */
function aiRow(name: string, model: string, providerId: string, report: AiReviewerReport | null, reviews: boolean): Row {
  const speed = report?.speed_signs;
  return {id: 'ai:' + providerId + '/' + model, name, when: model + (report ? ' · ' + isoWhen(report.measured_at) : ''),
    main: speed ? speed.read_right_share ?? share(speed.read_right, speed.truth) : null,
    mainText: speed ? ofText(speed.read_right, speed.truth, 'speed signs with the right number') : 'not measured yet: press Measure',
    second: report ? aiSecond(report) : '', inUse: reviews, badge: 'reviews', run: '', watch: null};
}
function aiSecond(report: AiReviewerReport): string {
  const empty = report.no_sign;
  const parts = [empty.truth ? 'left ' + empty.agreed + ' of ' + empty.truth + ' frames without signs empty' : 'no frame without signs',
    perFrame(report.false_per_frame) + ' false boxes per frame'];
  if (report.seconds_per_frame !== null) parts.push(report.seconds_per_frame.toFixed(1) + ' s per frame');
  if (report.frames_failed) parts.push(report.frames_failed + ' failed');
  return parts.join(' · ');
}

const rows = computed((): Row[] => {
  const list: Row[] = [];
  if (view.value === 'proposers') {
    for (const item of proposers.value?.proposers ?? []) {
      const report = proposers.value?.reports.find((entry) => entry.proposer === item.key && entry.partition === 'test');
      const speed = report?.speed_signs;
      list.push({id: 'proposer:' + item.key, name: item.label, when: report ? isoWhen(report.measured_at) + (report.limited ? ' · a few frames only' : '') : '',
        main: speed ? share(speed.read_right, speed.truth) : null,
        mainText: speed ? ofText(speed.read_right, speed.truth, 'speed signs proposed with the right number')
          : item.unavailable ? 'not installed' : 'not measured yet: press Measure',
        second: report ? 'found ' + speed!.found + ' of ' + speed!.truth + ' · ' + perFrame(report.false_per_frame)
          + ' false proposals per frame · ' + report.seconds_per_frame.toFixed(1) + ' s per frame' : '',
        inUse: false, run: '', watch: null});
    }
  } else if (view.value === 'ai') {
    const payload = aiReviewers.value;
    const shownReports = new Set<AiReviewerReport>();
    for (const provider of payload?.providers ?? []) {
      const report = payload?.reports.find((entry) => entry.provider_id === provider.id && entry.model === provider.model) ?? null;
      if (report) shownReports.add(report);
      list.push(aiRow(provider.name, provider.model, provider.id, report, provider.is_reviewer));
    }
    // Measures of a model a provider no longer uses, or of a provider removed: still there to compare with.
    for (const report of payload?.reports ?? []) {
      if (!shownReports.has(report)) list.push(aiRow(report.provider_name, report.model, report.provider_id, report, false));
    }
  } else if (view.value === 'chain') {
    for (const version of versionsOf('classifier') as ClassifierModel[]) {
      const chain = version.chain;
      if (!chain) continue;
      const main = share(chain.correct, chain.truth);
      const old = version.chain_old_reader;
      const oldMain = old ? share(old.correct, old.truth) : null;
      list.push({id: version.id, name: 'Classifier', when: isoWhen(version.trained_at), main,
        mainText: ofText(chain.correct, chain.truth, 'found and read right'),
        second: perFrame(chain.surviving_false_per_frame) + ' false boxes per frame left, of '
          + perFrame(chain.false_per_frame) + ' from the cutter',
        inUse: version.in_use, run: version.run, watch: 'chain',
        vs: main === null || oldMain === null ? undefined : {text: points(main - oldMain) + ' vs the old reader (it keeps every false box)',
          kind: Math.abs(main - oldMain) < 0.005 ? 'same' : main > oldMain ? 'better' : 'worse'}});
    }
  } else if (view.value === 'cutter') {
    for (const version of versionsOf('cutter') as CutterModel[]) {
      const test = version.test;
      const speed = test?.by_family?.speed;
      list.push({id: version.id, name: 'Cutter ' + sizeText(version.input_size) + (version.families === 'all' ? ' · all signs' : ' · speed signs'),
        when: isoWhen(version.trained_at), main: speed ? share(speed.found, speed.truth) : share(test?.found, test?.truth_boxes),
        mainText: speed ? ofText(speed.found, speed.truth, 'speed signs found') : ofText(test?.found, test?.truth_boxes, 'signs found'),
        second: test ? perFrame(test.boxes_per_frame) + ' boxes per frame' : '', inUse: version.in_use, run: version.run,
        watch: version.part});
    }
  } else if (view.value === 'classifier') {
    for (const version of versionsOf('classifier') as ClassifierModel[]) {
      const test = version.test;
      const old = !version.class_scheme;
      list.push({id: version.id, name: old ? 'Old reader (no “not a sign”)' : 'Classifier', when: isoWhen(version.trained_at),
        main: test?.accuracy ?? null,
        mainText: test?.accuracy == null ? 'not measured' : pct(test.accuracy) + ' of ' + thousands(test.samples ?? 0) + ' signs read right',
        second: test?.rejection_rate == null ? (old ? 'cannot say “not a sign”' : '') : pct(test.rejection_rate) + ' of the false boxes thrown away',
        inUse: version.in_use, run: version.run, watch: old ? 'chain-old-reader' : 'chain'});
    }
  } else {
    for (const version of versionsOf('decider') as DeciderModel[]) {
      const numbers = version.validation;
      list.push({id: version.id, name: version.label, when: isoWhen(version.trained_at),
        main: share(numbers?.other_road_found, numbers?.other_road_support),
        mainText: ofText(numbers?.other_road_found, numbers?.other_road_support, '“Another road” signs found'),
        second: numbers?.reading_accuracy == null ? '' : pct(numbers.reading_accuracy) + ' read right',
        inUse: version.in_use, run: version.run, watch: null});
    }
  }
  return list;
});
/** The version the others are compared with: the one in use, else the best. */
const reference = computed(() => rows.value.find((row) => row.inUse)
  ?? [...rows.value].filter((row) => row.main !== null).sort((a, b) => (b.main ?? 0) - (a.main ?? 0))[0] ?? null);
const best = computed(() => Math.max(...rows.value.map((row) => row.main ?? -1)));
function versus(row: Row): {text: string; kind: string} {
  if (view.value === 'chain') return row.vs ?? {text: '', kind: ''};
  const ref0 = reference.value;
  if (row.main === null || !ref0 || ref0.main === null || ref0.id === row.id) return {text: '', kind: ''};
  const difference = row.main - ref0.main;
  return {text: points(difference) + ' vs ' + (ref0.inUse ? 'the one in use' : 'the best'),
    kind: Math.abs(difference) < 0.005 ? 'same' : difference > 0 ? 'better' : 'worse'};
}

const shown = computed(() => rows.value.find((row) => row.id === picked.value) ?? reference.value ?? rows.value[0] ?? null);
/** The proposer of the row chosen, its latest report and its Measure button. */
const shownProposer = computed((): Proposer | null => shown.value?.id.startsWith('proposer:') ? shown.value.id.slice(9) as Proposer : null);
const proposerReport = computed(() => proposers.value?.reports.find((entry) => entry.proposer === shownProposer.value && entry.partition === 'test') ?? null);
const proposerMissing = computed(() => proposers.value?.proposers.find((item) => item.key === shownProposer.value)?.unavailable ?? null);
const MEASURE: [ActionName, string][] = [['measure_proposer', 'Proposes the signs of the reviewed test frames as Prepare would, '
  + 'and compares them with the boxes you kept. Nothing changes in the reviews; minutes to an hour.']];
watch(view, () => { picked.value = null; });

/** The AI reviewer of the row chosen: its measure, and its provider when it still uses that model (for Measure). */
const shownAi = computed((): {report: AiReviewerReport | null; provider: AiProvider | null} | null => {
  const id = shown.value?.id;
  if (!id?.startsWith('ai:')) return null;
  const payload = aiReviewers.value;
  const same = (providerId: string, model: string): boolean => 'ai:' + providerId + '/' + model === id;
  return {report: payload?.reports.find((entry) => same(entry.provider_id, entry.model)) ?? null,
    provider: payload?.providers.find((entry) => same(entry.id, entry.model)) ?? null};
});
const AI_MEASURE: [ActionName, string][] = [['measure_ai_reviewer', 'Asks the AI about up to 60 frames a person reviewed, '
  + 'the same ones for every model, and compares its answers with theirs. Nothing is saved; minutes to an hour.']];
/** "25 of 31", or "–" without any. */
function countText(found: number, truth: number): string {
  return truth ? found + ' of ' + truth : '–';
}

/** Each view's data has come (the versions of the networks, the proposers, the AI reviewers), or why not. */
const loaded = computed(() => (view.value === 'proposers' ? proposers.value !== null
  : view.value === 'ai' ? aiReviewers.value !== null : models.value !== null));
const loadError = computed(() => (view.value === 'proposers' ? proposersError.value
  : view.value === 'ai' ? aiError.value : modelsError.value));
const emptyText = computed(() => (view.value === 'proposers' ? 'No proposer on this server.'
  : view.value === 'ai' ? 'No AI provider yet: add one in Settings.' : 'Nothing measured yet: train one on the Train tab.'));

const runSummary = computed(() => runsPayload.value?.runs.find((run) => run.name === shown.value?.run) ?? null);
const {detail, loading, error} = useRunDetail(computed(() => runSummary.value?.name ?? null),
  computed(() => runSummary.value?.version ?? null));
const watchModels = computed((): WatchModel[] => {
  const key = shown.value?.watch;
  const list = detail.value?.name === shown.value?.run ? detail.value?.watch ?? [] : [];
  return key ? list.filter((item) => item.model === key) : [];
});
</script>

<template>
  <h1 class="sr-only">Evaluate</h1>

  <section class="card ev-card" aria-labelledby="eval-title">
    <div class="ev-head">
      <h2 id="eval-title">What to judge</h2>
      <Segmented v-model="view" :options="VIEWS" label="Network to judge"/>
    </div>
    <p class="ev-goal">{{ GOAL[view] }}</p>
    <p v-if="!loaded" class="empty">{{ loadError || 'Loading…' }}</p>
    <p v-else-if="rows.length === 0" class="empty">{{ emptyText }} <a v-if="view === 'ai'" href="/account#ai-reviewer">Open Settings</a></p>
    <ul v-else class="ev-list" role="listbox" aria-label="Versions">
      <li v-for="row in rows" :key="row.id" role="option" :aria-selected="shown?.id === row.id"
          :class="['ev-row', {picked: shown?.id === row.id}]" tabindex="0"
          @click="picked = row.id" @keydown.enter="picked = row.id">
        <div class="ev-name"><strong>{{ row.name }}</strong><span v-if="row.inUse" class="ev-in-use">{{ row.badge ?? 'in use' }}</span>
          <span class="ev-when">{{ row.when }}</span></div>
        <div class="ev-main" :class="{top: row.main !== null && row.main === best && rows.length > 1}">{{ row.mainText }}</div>
        <div class="ev-second">{{ row.second }}<span v-if="versus(row).text" :class="['ev-vs', versus(row).kind]">{{ versus(row).text }}</span></div>
      </li>
    </ul>
  </section>

  <section v-if="shown && shownProposer" class="card ev-card" aria-labelledby="proposer-title">
    <h2 id="proposer-title">{{ shown.name }} <span class="ev-when">{{ shown.when }}</span></h2>
    <p class="ev-goal">{{ shown.mainText }}<template v-if="shown.second"> · {{ shown.second }}</template></p>
    <template v-if="proposerReport">
      <p class="ev-goal">On {{ thousands(proposerReport.frames) }} test frames. Every sign, speed signs and the others: found
        {{ proposerReport.every_sign.found }} of {{ proposerReport.every_sign.truth }}. Speed signs on the frames a person
        reviewed: {{ ofText(proposerReport.speed_signs_person.read_right, proposerReport.speed_signs_person.truth, 'with the right number') }}.</p>
      <ul class="ev-widths">
        <li v-for="item in proposerReport.by_width" :key="item.key"><span class="ev-when">{{ item.label }} wide</span>
          <strong>{{ item.truth ? item.found + ' of ' + item.truth : '–' }}</strong><span class="ev-when">found</span></li>
      </ul>
    </template>
    <p v-if="proposerMissing" class="ev-goal ev-missing">{{ proposerMissing }}</p>
    <ActionList v-else :items="MEASURE" :actions="props.status?.actions ?? null"
                @run="(name, state) => shownProposer && emit('run', name, state, undefined, {proposer: shownProposer})"/>
  </section>

  <section v-if="shown && shownAi" class="card ev-card" aria-labelledby="ai-title">
    <h2 id="ai-title">{{ shown.name }} <span class="ev-when">{{ shown.when }}</span></h2>
    <p class="ev-goal">{{ shown.mainText }}<template v-if="shown.second"> · {{ shown.second }}</template></p>
    <template v-if="shownAi.report">
      <ul class="ev-widths">
        <li><span class="ev-when">Speed signs found</span><strong>{{ countText(shownAi.report.speed_signs.found, shownAi.report.speed_signs.truth) }}</strong>
          <span class="ev-when">{{ pct(shownAi.report.speed_signs.found_share) }}</span></li>
        <li><span class="ev-when">Right kind</span><strong>{{ countText(shownAi.report.speed_signs.kind_right, shownAi.report.speed_signs.truth) }}</strong>
          <span class="ev-when">{{ pct(shownAi.report.speed_signs.kind_right_share) }}</span></li>
        <li><span class="ev-when">Right road</span><strong>{{ countText(shownAi.report.speed_signs.road_right, shownAi.report.speed_signs.truth) }}</strong>
          <span class="ev-when">{{ pct(shownAi.report.speed_signs.road_right_share) }}</span></li>
        <li><span class="ev-when">Other signs found</span><strong>{{ countText(shownAi.report.other_signs.found, shownAi.report.other_signs.truth) }}</strong>
          <span class="ev-when">{{ countText(shownAi.report.other_signs.kind_right, shownAi.report.other_signs.truth) }} right kind</span></li>
        <li><span class="ev-when">Every sign found</span><strong>{{ countText(shownAi.report.every_sign.found, shownAi.report.every_sign.truth) }}</strong>
          <span class="ev-when">{{ pct(shownAi.report.every_sign.found_share) }}</span></li>
        <li><span class="ev-when">No sign, left empty</span><strong>{{ countText(shownAi.report.no_sign.agreed, shownAi.report.no_sign.truth) }}</strong>
          <span class="ev-when">{{ shownAi.report.no_sign.said_no_sign_wrongly }} said “no sign” wrongly</span></li>
        <li><span class="ev-when">AI boxes per frame</span><strong>{{ perFrame(shownAi.report.ai_boxes_per_frame) }}</strong>
          <span class="ev-when">{{ perFrame(shownAi.report.false_per_frame) }} false</span></li>
        <li><span class="ev-when">Seconds per frame</span><strong>{{ shownAi.report.seconds_per_frame === null ? '–' : shownAi.report.seconds_per_frame.toFixed(1) }}</strong>
          <span class="ev-when">{{ shownAi.report.frames_failed }} failed</span></li>
      </ul>
      <p class="ev-goal">On {{ thousands(shownAi.report.frames_measured) }} of {{ thousands(shownAi.report.frames) }} frames a person reviewed
        (the same {{ shownAi.report.frames }} for every model: sample <code>{{ shownAi.report.sample_sha256.slice(0, 10) }}</code>)<template
          v-if="shownAi.report.frames_incomplete">; on {{ shownAi.report.frames_incomplete }} it did not judge every box</template>.</p>
      <details v-if="shownAi.report.failures.length" class="ev-failures">
        <summary>{{ shownAi.report.frames_failed }} frames without an answer</summary>
        <ul><li v-for="(failure, i) in shownAi.report.failures" :key="i"><code>{{ failure.frame_key }}</code> {{ failure.reason }}: {{ failure.message }}</li></ul>
      </details>
    </template>
    <p v-if="!shownAi.provider" class="ev-goal ev-missing">This model is no longer in Settings: only its last measure is left.</p>
    <ActionList v-else :items="AI_MEASURE" :actions="props.status?.actions ?? null"
                @run="(name, state) => shownAi?.provider && emit('run', name, state, undefined, {provider: shownAi.provider.id})"/>
  </section>

  <template v-if="shown && !shownProposer && !shownAi">
    <section class="card ev-card" aria-labelledby="watch-title">
      <h2 id="watch-title">{{ shown.name }} <span class="ev-when">{{ shown.when }}</span></h2>
      <p class="ev-goal">{{ shown.mainText }}<template v-if="shown.second"> · {{ shown.second }}</template></p>
      <WatchSection v-if="watchModels.length" :key="shown.id" :run="shown.run" :models="watchModels" :version="runSummary?.version ?? null"/>
      <p v-else-if="loading" class="empty">Loading the frames…</p>
      <p v-else class="empty">{{ shown.watch ? 'No frames to play for this version: its run kept none.' : 'Nothing to play: the decider is judged on its table.' }}</p>
    </section>
    <details v-if="runSummary" class="card ev-more">
      <summary>All the numbers of run {{ shown.run }}</summary>
      <RunDetailView :key="runSummary.name" :summary="runSummary" :detail="detail" :loading="loading" :error="error" no-watch/>
    </details>
  </template>
</template>
