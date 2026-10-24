<script setup lang="ts">
// The table of the routes, or how to get the first ones.
import type {RouteRow} from '../common/panel-types';
import RouteTableRow from './RouteTableRow.vue';

/** null before the first answer: the empty table, not the "no route" text. */
defineProps<{routes: RouteRow[] | null}>();
</script>

<template>
  <section class="card table-card" aria-labelledby="routes-title">
    <h2 id="routes-title">Routes</h2>
    <div class="table-wrap" id="routes-wrap" :hidden="routes !== null && routes.length === 0"><table>
      <colgroup><col class="c-route"><col class="c-seg"><col class="c-frames"><col class="c-reviewed"><col class="c-objects"><col class="c-state"></colgroup>
      <thead><tr><th>ID</th><th class="num" title="Segments extracted of those arrived">Segments</th><th class="num">Frames</th>
        <th>Confirmed</th><th class="num" title="Objects of this route already in the dataset">In dataset</th><th>State</th></tr></thead>
      <tbody id="routes"><RouteTableRow v-for="row in routes ?? []" :key="row.route" :row="row"/></tbody>
    </table></div>
    <p class="empty" id="routes-empty" :hidden="routes === null || routes.length > 0">No routes yet: download the drives with <strong>Download from the comma</strong>, above.</p>
  </section>
</template>
