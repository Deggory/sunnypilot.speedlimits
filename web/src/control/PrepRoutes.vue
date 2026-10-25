<script setup lang="ts">
// The routes a running preparation is working on: the drive, its phase and how far it is (for Propose again every
// route of the batch, waiting, proposing with the time left, or ready).
import {thousands} from '../common/format';
import {durationText} from './jobs';
import type {RouteProgress} from '../common/panel-types';
import Bar from '../components/Bar.vue';

defineProps<{routes: RouteProgress[]}>();
// The server says the phase in English; these are the codes of an older server.
const PHASE: Record<string, string> = {frame: 'extracting frames', cartelli: 'proposing signs', pronta: 'ready',
  'da collegare': 'to link'};
</script>

<template>
  <ul v-if="routes.length" class="prep-routes">
    <li v-for="route in routes" :key="route.route">
      <span class="prep-route" :title="route.route">{{ route.title || route.route.replace(/^0+(?=[0-9a-f]+--)/, '') }}</span>
      <span class="prep-phase">{{ PHASE[route.phase] ?? route.phase }}<template v-if="route.total && route.phase !== 'waiting'"> · {{ thousands(route.done) }}
        of {{ thousands(route.total) }} {{ route.unit }}</template><template v-if="route.seconds_left"> · {{ durationText(route.seconds_left) }}
        left</template></span>
      <Bar :percent="route.percent" :kind="route.percent >= 100 ? 'ok' : 'run'"/>
    </li>
  </ul>
</template>
