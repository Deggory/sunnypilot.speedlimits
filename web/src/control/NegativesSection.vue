<script setup lang="ts">
// What the classifier learnt “not a sign” from: the false boxes of the cutter in use mined on every frame
// (negatives.jsonl, its header: the cutter, the settings, per split the frames, the candidate boxes and those kept, by
// how the frame was reviewed), and its “not a sign” crops per split by where they come from (the training plan).
import {computed} from 'vue';
import {isoWhen, thousands} from '../common/format';
import type {NegativesMined} from '../common/panel-types';
import {PARTITION_LABEL, partitionOrder} from './evaluate';

const props = defineProps<{negatives: NegativesMined | null; bySource: Record<string, Record<string, number> | number> | null}>();
// Who reviewed the frame of a kept box (nnslr_tools.training.REVIEW_TIERS): a person, an AI after the rule that has it box
// the other signs too (settings.rule_change_at), an AI before it (its frames may hold another sign without a box).
const TIERS: [key: string, label: string][] = [['human', 'by a person'], ['ai_after_rule', 'by AI after the rule'],
  ['ai_before_rule', 'by AI before it'], ['other', 'other']];
// Where a “not a sign” crop comes from (nnslr_tools.training.NEGATIVE_SOURCES).
const SOURCES: [key: string, label: string][] = [['rejected', 'Removed proposals'], ['mined', 'Cutter’s false boxes'],
  ['reviewed', 'Reviewed as not a sign']];
const cell = (value: number | null | undefined): string => value === null || value === undefined ? '–' : thousands(value);
const tiers = (counts: Record<string, number>): string => {
  const known = TIERS.filter(([key]) => counts[key]).map(([key, label]) => thousands(counts[key] ?? 0) + ' ' + label);
  const others = Object.keys(counts).filter((key) => !TIERS.some(([tier]) => tier === key) && counts[key])
    .map((key) => thousands(counts[key] ?? 0) + ' ' + key.replace(/_/g, ' '));
  return [...known, ...others].join(' · ') || '–';
};
const settings = computed(() => {
  const value = props.negatives?.settings;
  if (!value) return '';
  return [value.score_threshold !== null ? 'cutter score ≥ ' + value.score_threshold : '',
    value.max_iou !== null ? 'overlap with a reviewed box ≤ ' + value.max_iou : '',
    value.per_frame !== null ? 'at most ' + value.per_frame + ' per frame' : '',
    value.total_factor !== null ? 'at most ' + value.total_factor + '× the sign crops' : '',
    value.seed !== null ? 'seed ' + value.seed : ''].filter(Boolean).join(', ');
});
/** The “not a sign” crops of the training per split and source: {split: {source: count}} (a bare count when the plan
 * says only how many). */
const sourceRows = computed(() => {
  const value = props.bySource ?? {};
  return partitionOrder(Object.keys(value)).map((split) => {
    const counts = value[split];
    const bySource = typeof counts === 'number' ? {} : counts ?? {};
    const total = typeof counts === 'number' ? counts : Object.values(bySource).reduce((sum, count) => sum + count, 0);
    return {split, label: PARTITION_LABEL[split] ?? split, cells: SOURCES.map(([key]) => typeof counts === 'number' ? null : bySource[key] ?? 0), total};
  });
});
</script>

<template>
  <section v-if="negatives || sourceRows.length" class="card eval-card" id="classifier-negatives" aria-labelledby="negatives-title">
    <div class="ds-head">
      <h2 id="negatives-title">“Not a sign” it learnt from<span class="eval-what">the cutter’s false boxes, mined on every frame</span></h2>
      <span v-if="negatives?.created_at" class="ds-when">mined {{ isoWhen(negatives.created_at) }}</span>
    </div>
    <template v-if="negatives">
      <dl class="facts">
        <div><dt>Cutter</dt><dd><code :title="negatives.cutter.checkpoint ?? ''">{{ negatives.cutter.id ?? negatives.cutter.checkpoint ?? '–' }}</code><span
          v-if="negatives.cutter.input_size" class="muted"> at {{ negatives.cutter.input_size.replace('x', ' × ') }}</span></dd></div>
        <div v-if="settings"><dt>Mining</dt><dd>{{ settings }}</dd></div>
        <div v-if="negatives.settings.rule_change_at"><dt>AI boxes other signs since</dt><dd>{{ isoWhen(negatives.settings.rule_change_at)
          || negatives.settings.rule_change_at }}</dd></div>
      </dl>
      <p v-if="negatives.cutter.same_split === false" class="compare-warn">The cutter learnt on another split: its false boxes on the validation and
        test frames are not those of frames it never saw.</p>
      <div class="table-wrap"><table class="eval-table" id="negatives-table">
        <thead><tr><th>Partition</th><th class="num" title="Frames the cutter ran on">Frames</th>
          <th class="num" title="Frames with at least one false box">With false boxes</th><th class="num" title="False boxes found">Candidates</th>
          <th class="num" title="Left after the cap per frame">After the frame cap</th><th class="num" title="Sign crops of the partition">Sign crops</th>
          <th class="num" title="The most kept, from the sign crops">Cap</th><th class="num">Kept</th><th>Kept, by who reviewed the frame</th></tr></thead>
        <tbody>
          <tr v-for="row in negatives.splits" :key="row.split" :class="{test: row.split === 'test'}">
            <th scope="row">{{ PARTITION_LABEL[row.split] ?? row.split }}</th><td class="num">{{ cell(row.frames) }}</td>
            <td class="num">{{ cell(row.frames_with_candidates) }}</td><td class="num">{{ cell(row.candidates) }}</td>
            <td class="num">{{ cell(row.after_frame_cap) }}</td><td class="num">{{ cell(row.positives) }}</td><td class="num">{{ cell(row.cap) }}</td>
            <td class="num">{{ cell(row.selected) }}</td><td>{{ tiers(row.by_tier) }}</td>
          </tr>
          <tr v-if="negatives.splits.length === 0"><td colspan="9" class="muted">No count in the header.</td></tr>
        </tbody>
      </table></div>
    </template>
    <div v-if="sourceRows.length" class="ds-block">
      <h3 class="ds-title">“Not a sign” crops of the training<span class="ds-when">by where they come from</span></h3>
      <div class="table-wrap"><table class="eval-table" id="negatives-sources">
        <thead><tr><th>Partition</th><th v-for="[key, label] in SOURCES" :key="key" class="num">{{ label }}</th><th class="num">Total</th></tr></thead>
        <tbody>
          <tr v-for="row in sourceRows" :key="row.split" :class="{test: row.split === 'test'}">
            <th scope="row">{{ row.label }}</th><td v-for="(count, index) in row.cells" :key="index" class="num">{{ cell(count) }}</td>
            <td class="num">{{ thousands(row.total) }}</td>
          </tr>
        </tbody>
      </table></div>
    </div>
  </section>
</template>
