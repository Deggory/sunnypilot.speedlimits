<script setup lang="ts">
// The dataset and the split a run learnt from and was measured on (its training plans and the split file they
// name): their hashes, the policy of the split, the routes reviewed by hand whose places are the test, the objects of
// each partition as one stacked bar (the colours of the dataset on the Train tab), and partition by partition the
// detector's frames and boxes, the reader's crops (or the classifier's, with its “not a sign” crops) and the objects a
// person confirmed.
import {computed} from 'vue';
import {thousands} from '../common/format';
import type {DatasetUsed} from '../common/panel-types';
import {PARTITION_LABEL, partitionOrder, shortSha} from './evaluate';

const props = defineProps<{dataset: DatasetUsed}>();
const SLOT: Record<string, number> = {train: 1, validation: 2, test: 3, buffer: 4};

const split = computed(() => props.dataset.split);
const bar = computed(() => {
  const counts = split.value?.counts ?? {};
  const keys = partitionOrder(Object.keys(counts));
  const total = keys.reduce((sum, key) => sum + (counts[key] ?? 0), 0);
  return {total, parts: keys.map((key) => ({key, label: PARTITION_LABEL[key] ?? key, count: counts[key] ?? 0, slot: SLOT[key] ?? 0,
    share: total ? Math.round((counts[key] ?? 0) * 100 / total) + '%' : ''}))};
});
/** The classifier with “not a sign”: its “not a sign” crops of a partition, whatever their source. */
const classifier = computed(() => props.dataset.classifier ?? null);
function notASign(key: string): string {
  const counts = classifier.value?.negatives[key];
  if (counts === undefined) return '–';
  return thousands(typeof counts === 'number' ? counts : Object.values(counts).reduce((sum, count) => sum + count, 0));
}
/** Every partition named anywhere, with what each network had of it. */
const rows = computed(() => {
  const value = props.dataset;
  const keys = partitionOrder([...new Set([...Object.keys(value.split?.counts ?? {}), ...Object.keys(value.detector?.frames ?? {}),
    ...Object.keys(value.reader?.crops ?? {}), ...Object.keys(value.classifier?.crops ?? {})])]);
  const cell = (map: Record<string, number> | undefined, key: string): string => map === undefined ? '–' : thousands(map[key] ?? 0);
  return keys.map((key) => ({key, label: PARTITION_LABEL[key] ?? key, objects: cell(value.split?.counts, key),
    human: cell(value.split?.human_reviewed_counts, key), frames: cell(value.detector?.frames, key), boxes: cell(value.detector?.boxes, key),
    negatives: cell(value.detector?.negative_frames, key), crops: cell(value.reader?.crops, key),
    classifierCrops: cell(value.classifier?.crops, key), notASign: notASign(key),
    routes: value.split?.routes[key]?.length ?? null}));
});
</script>

<template>
  <section class="card eval-card" id="run-dataset" aria-labelledby="run-dataset-title">
    <div class="ds-head">
      <h2 id="run-dataset-title">Dataset and split</h2>
      <span class="ds-when">{{ dataset.current_split ? 'the split in use now' : 'an older split, not the one in use now' }}</span>
    </div>
    <dl class="facts">
      <div><dt>Dataset</dt><dd><code :title="dataset.dataset_sha256 ?? ''">{{ shortSha(dataset.dataset_sha256) }}</code>
        <span v-if="dataset.dataset_kind" class="muted"> {{ dataset.dataset_kind }}</span></dd></div>
      <div><dt>Split</dt><dd><code :title="dataset.split_file ?? dataset.split_sha256 ?? ''">{{ shortSha(dataset.split_sha256) }}</code></dd></div>
      <div v-if="split"><dt>Policy</dt><dd>{{ split.policy ?? '–' }}<span v-if="split.grouping.length" class="muted"> by {{ split.grouping.join(', ') }}</span></dd></div>
      <div v-if="split"><dt>Test routes</dt><dd>
        <template v-if="split.test_routes.length"><code v-for="route in split.test_routes" :key="route" class="route-code">{{ route }}</code></template>
        <span v-else class="muted">none named: the test is made of whole groups</span></dd></div>
    </dl>

    <template v-if="split && bar.total">
      <div class="split-bar" role="img" :aria-label="'Split: ' + bar.parts.map((part) => part.label + ' ' + thousands(part.count)).join(', ')">
        <span v-for="part in bar.parts.filter((item) => item.count > 0)" :key="part.key" :class="'split-' + part.slot"
              :style="{flexGrow: part.count}" :title="part.label + ': ' + thousands(part.count) + ' objects (' + part.share + ')'"></span>
      </div>
      <ul class="split-legend">
        <li v-for="part in bar.parts" :key="part.key"><span class="split-key" :class="'split-' + part.slot" aria-hidden="true"></span>{{
          part.label }}<strong>{{ thousands(part.count) }}</strong><span class="split-share">{{ part.share }}</span></li>
      </ul>
    </template>
    <p v-else-if="!split" class="empty">The split file is not on this server any more: only what the training plans recorded.</p>

    <div class="table-wrap"><table class="eval-table">
      <thead><tr><th>Partition</th><th class="num" title="Objects of the dataset">Objects</th><th class="num" title="Objects a person confirmed">Human</th>
        <th class="num" title="Routes with frames in this partition">Routes</th><th class="num" title="Frames the detector saw">Det. frames</th>
        <th class="num" title="Reviewed signs the detector learnt or was measured on">Det. boxes</th>
        <th class="num" title="Frames reviewed as having no sign">No-sign frames</th><th class="num" title="Boxes the reader read">Reader crops</th>
        <template v-if="classifier"><th class="num" title="Crops the classifier learnt from or was measured on">Classifier crops</th>
          <th class="num" title="Its “not a sign” crops: boxes reviewed as not a sign, rejected proposals, false boxes of the cutter">Not a sign</th></template></tr></thead>
      <tbody>
        <tr v-for="row in rows" :key="row.key" :class="{test: row.key === 'test'}">
          <th scope="row">{{ row.label }}</th><td class="num">{{ row.objects }}</td><td class="num">{{ row.human }}</td>
          <td class="num">{{ row.routes ?? '–' }}</td><td class="num">{{ row.frames }}</td><td class="num">{{ row.boxes }}</td>
          <td class="num">{{ row.negatives }}</td><td class="num">{{ row.crops }}</td>
          <template v-if="classifier"><td class="num">{{ row.classifierCrops }}</td><td class="num">{{ row.notASign }}</td></template>
        </tr>
      </tbody>
    </table></div>
    <p class="note">The buffer is the places next to the test: neither learnt from nor measured, so a sign seen from a few metres
      away cannot leak into the test.</p>
    <p v-if="classifier?.folded_labels?.length" class="note">Learnt as a broader label, for lack of crops of their own: {{ classifier.folded_labels
      .map((row) => row.into_name ? row.name + ' → ' + row.into_name : row.name + (row.count !== undefined ? ' (' + row.count + ')' : '')).join(', ') }}.</p>
    <p v-for="(warning, index) in split?.warnings ?? []" :key="index" class="note ds-note">{{ warning }}</p>
  </section>
</template>
