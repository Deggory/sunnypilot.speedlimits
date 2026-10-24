<script setup lang="ts">
// The table of the routes, or how to get the first ones.
import type {RouteRow} from '../common/panel-types';
import RouteTableRow from './RouteTableRow.vue';

/** null before the first answer: the empty table, not the "no route" text. */
defineProps<{routes: RouteRow[] | null}>();
</script>

<template>
  <section class="card table-card" aria-labelledby="routes-title">
    <h2 id="routes-title">Route</h2>
    <div class="table-wrap" id="routes-wrap" :hidden="routes !== null && routes.length === 0"><table>
      <colgroup><col class="c-route"><col class="c-seg"><col class="c-frames"><col class="c-reviewed"><col class="c-objects"><col class="c-state"></colgroup>
      <thead><tr><th>ID</th><th class="num" title="Segmenti estratti su quelli arrivati">Segmenti</th><th class="num">Fotogrammi</th>
        <th>Confermati</th><th class="num" title="Oggetti di questa route già nel dataset">Nel dataset</th><th>Stato</th></tr></thead>
      <tbody id="routes"><RouteTableRow v-for="row in routes ?? []" :key="row.route" :row="row"/></tbody>
    </table></div>
    <p class="empty" id="routes-empty" :hidden="routes === null || routes.length > 0">Nessuna route: scarica i giri con <strong>Scarica dal comma</strong>, qui sopra.</p>
  </section>
</template>
