<script setup lang="ts">
// "Signs": the boxes of the frame, what each one is, where it comes from, what the reviewer changed, the
// boxes that are the same sign twice. A click opens the editor of the box, ✕ removes it.
import {computed} from 'vue';
import {boxItems} from './boxes';
import {releaseFocus} from './dom';
import {removeBox, selectBox} from './edits';
import {state} from './store';
import type {FrameEdit, FrameRow} from './types';

const props = defineProps<{row: FrameRow; work: FrameEdit}>();
const items = computed(() => boxItems(props.row, props.work));
</script>

<template>
  <div class="box-list" id="boxList">
    <div v-for="item in items" :key="item.index" :class="item.index === state.selected ? 'box-row selected' : 'box-row'"
         :data-index="item.index" :data-detection-id="item.detectionId"
         :data-duplicate-of="item.duplicateOf === null ? undefined : item.duplicateOf">
      <button type="button" class="box-main" :aria-pressed="item.index === state.selected" @click="selectBox(item.index); releaseFocus($event)"><span class="n">{{ item.index + 1 }}</span><span class="what">{{ item.what }}</span><span class="src">{{ item.source.text }}<span v-if="item.source.changed !== null" class="changed">{{ item.source.changed }}</span><span v-if="item.duplicateOf !== null" class="dup">{{ ' · same sign as ' + (item.duplicateOf + 1) }}</span></span></button>
      <button type="button" class="box-del" :title="'Delete box ' + (item.index + 1)"
              :aria-label="'Delete box ' + (item.index + 1)" @click="removeBox(item.index); releaseFocus($event)">✕</button>
    </div>
  </div>
</template>
