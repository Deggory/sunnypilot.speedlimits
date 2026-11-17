<script setup lang="ts">
// The traffic light of the selected box: Red, Amber, Green, Off. A chip makes the box an "Other sign" of that light.
// On Speed and sign review they stand on the drive's title row (SpeedReviewDrive.vue), always there while a frame is
// edited and off until a box is selected; on the page of Sign review at the top of the box's editor (BoxEditor.vue).
import {SIGN_KIND_GROUPS} from './catalog';
import {selectedBox} from './current';
import {releaseFocus} from './dom';
import {setKind} from './edits';
import {state} from './store';

const LIGHTS = SIGN_KIND_GROUPS.filter((group) => group.label === 'Traffic light');
</script>

<template>
  <div class="kinds lights" id="lightKinds">
    <div v-for="group in LIGHTS" :key="group.label" class="kind-group" role="group" :aria-label="group.label">
      <span class="field-label">{{ group.label }}<template v-if="selectedBox"> · box {{ state.selected + 1 }}</template></span>
      <button v-for="kind in group.kinds" :key="kind.id" type="button" class="chip kind" :disabled="selectedBox === null"
              :aria-pressed="selectedBox?.family === 'other_sign' && selectedBox?.sign_type === kind.id"
              :data-kind="kind.id" :title="selectedBox === null ? 'Select a box first' : kind.label"
              @click="setKind(kind.id); releaseFocus($event)">{{ kind.chip }}</button>
    </div>
  </div>
</template>
