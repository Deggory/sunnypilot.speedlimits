<script setup lang="ts">
// The routes a running preparation is working on: the drive, its phase and how far it is (for Propose again every
// route of the batch, waiting, proposing or ready; the time left is the card's, once, above them). A drive with a
// report opens its review page: read only while Propose again works on it (the new proposals made so far).
import {thousands} from '../common/format';
import type {RouteProgress} from '../common/panel-types';
import Bar from '../components/Bar.vue';

defineProps<{routes: RouteProgress[]}>();
// The server says the phase in English; these are the codes of an older server.
const PHASE: Record<string, string> = {frame: 'extracting frames', cartelli: 'proposing signs', pronta: 'ready',
  'da collegare': 'to link'};
// The phases of a drive that has a report to open.
const OPENS = new Set(['ready', 'pronta', 'waiting', 'proposing again']);
</script>

<template>
  <ul v-if="routes.length" class="prep-routes">
    <li v-for="route in routes" :key="route.route">
      <a v-if="OPENS.has(route.phase)" class="prep-route" :href="'/route/' + encodeURIComponent(route.route) + '/'"
         :title="route.phase === 'ready' ? 'Review this drive' : 'See it, read only while it is proposed again'">{{ route.title
           || route.route.replace(/^0+(?=[0-9a-f]+--)/, '') }}</a>
      <span v-else class="prep-route" :title="route.route">{{ route.title || route.route.replace(/^0+(?=[0-9a-f]+--)/, '') }}</span>
      <span class="prep-phase">{{ PHASE[route.phase] ?? route.phase }}<template v-if="route.total && route.phase !== 'waiting'"> · {{ thousands(route.done) }}
        of {{ thousands(route.total) }} {{ route.unit }}</template></span>
      <Bar :percent="route.percent" :kind="route.percent >= 100 ? 'ok' : 'run'"/>
    </li>
  </ul>
</template>
