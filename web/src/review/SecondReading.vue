<script setup lang="ts">
// "Second reading": what another model saw in the frame, one row per sign, with "Use" to turn it into a box.
// Only on the frames the second opinion has looked at.
import {computed} from 'vue';
import {releaseFocus} from './dom';
import {useHint} from './edits';
import {hintRows, statsLine} from './hints';
import {hintsOf} from './store';
import type {FrameEdit, FrameRow} from './types';

const props = defineProps<{row: FrameRow; work: FrameEdit}>();
const hints = computed(() => hintsOf(props.row.frame_key));
const rows = computed(() => hintRows(hints.value, props.work));
const stats = statsLine();
</script>

<template>
  <section id="hintSection" :hidden="hints.length === 0">
    <div class="section-head"><h2>Second reading</h2></div>
    <p class="hint-intro">Another model looked at the frame. Use it only if you see it too.</p>
    <div class="hint-list" id="hintList">
      <div v-for="item in rows" :key="item.index" class="hint-row">
        <div class="hint-text">
          <div class="hint-head">
            <span class="hint-what">{{ item.text }}</span>
            <span :class="'conf ' + item.confidence">{{ item.confidenceText }}</span>
            <span v-if="item.sub" class="hint-sub">{{ item.sub }}</span>
          </div>
          <p v-if="item.note" class="hint-note">{{ item.note }}</p>
        </div>
        <button v-if="item.used" type="button" class="btn small tint" :data-hint-index="item.index" disabled>Already used</button>
        <button v-else type="button" class="btn small tint" :data-hint-index="item.index" :aria-label="item.useLabel"
                @click="useHint(item.index); releaseFocus($event)">Use</button>
      </div>
    </div>
    <p class="hint-stats" id="hintStats">{{ hints.length === 0 ? '' : stats }}</p>
  </section>
</template>
