<script setup lang="ts">
// Evaluate, by network: choose what to judge (the whole chain, the cutter, the classifier, the decider); every
// version of it in one table, judged by the one number that matters for that network, next to the version in use;
// the version chosen played frame by frame on the test (WatchSection), and all the numbers of its run folded below.
import {computed, ref, watch} from 'vue';
import {isoWhen, thousands} from '../common/format';
import type {ClassifierModel, CutterModel, DeciderModel, ModelVersion, WatchModel} from '../common/panel-types';
import {pct, points, useRunDetail, useRuns} from './evaluate';
import {perFrame, sizeText, useModels} from './models';
import RunDetailView from './RunDetail.vue';
import Segmented from './Segmented.vue';
import WatchSection from './WatchSection.vue';

type View = 'chain' | 'cutter' | 'classifier' | 'decider';
const VIEWS: {key: View; label: string}[] = [
  {key: 'chain', label: 'Chain'}, {key: 'cutter', label: 'Cutter'},
  {key: 'classifier', label: 'Classifier'}, {key: 'decider', label: 'Decider'},
];
/** What each network must do, and the number that judges it. */
const GOAL: Record<View, string> = {
  chain: 'The cutter and the classifier together, on the test roads no training saw. What counts: of the 80 speed '
    + 'signs, how many are found and read right, and how many false boxes are left per frame.',
  cutter: 'Finds every box that could be a sign. What counts: how many of the 80 test speed signs it finds. Many '
    + 'boxes per frame are expected: the classifier drops them.',
  classifier: 'Says what each box is, or “not a sign”. What counts: the test signs read right, and the false boxes '
    + 'it throws away.',
  decider: 'Says whom a sign applies to: our road or another. What counts: the “Another road” signs it finds. '
    + 'Only on validation: the test has no such sign yet.',
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
}

const view = ref<View>('chain');
const picked = ref<string | null>(null);
const {payload: models} = useModels();
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

const rows = computed((): Row[] => {
  const list: Row[] = [];
  if (view.value === 'chain') {
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
watch(view, () => { picked.value = null; });

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
    <p v-if="models === null" class="empty">Loading…</p>
    <p v-else-if="rows.length === 0" class="empty">Nothing measured yet: train one on the Train tab.</p>
    <ul v-else class="ev-list" role="listbox" aria-label="Versions">
      <li v-for="row in rows" :key="row.id" role="option" :aria-selected="shown?.id === row.id"
          :class="['ev-row', {picked: shown?.id === row.id}]" tabindex="0"
          @click="picked = row.id" @keydown.enter="picked = row.id">
        <div class="ev-name"><strong>{{ row.name }}</strong><span v-if="row.inUse" class="ev-in-use">in use</span>
          <span class="ev-when">{{ row.when }}</span></div>
        <div class="ev-main" :class="{top: row.main !== null && row.main === best && rows.length > 1}">{{ row.mainText }}</div>
        <div class="ev-second">{{ row.second }}<span v-if="versus(row).text" :class="['ev-vs', versus(row).kind]">{{ versus(row).text }}</span></div>
      </li>
    </ul>
  </section>

  <template v-if="shown">
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
