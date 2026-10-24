<script setup lang="ts">
// The routes a running preparation is working on: phase and how far each is.
import {thousands} from '../common/format';
import type {RouteProgress} from '../common/panel-types';
import Bar from '../components/Bar.vue';

defineProps<{routes: RouteProgress[]}>();
const PHASE: Record<string, string> = {frame: 'estrae i fotogrammi', cartelli: 'propone i cartelli', pronta: 'pronta',
  'da collegare': 'da collegare'};
</script>

<template>
  <ul v-if="routes.length" class="prep-routes">
    <li v-for="route in routes" :key="route.route">
      <span class="prep-route mono" :title="route.route">{{ route.route.replace(/^0+(?=[0-9a-f]+--)/, '') }}</span>
      <span class="prep-phase">{{ PHASE[route.phase] ?? route.phase }}<template v-if="route.total"> · {{ thousands(route.done) }}
        di {{ thousands(route.total) }} {{ route.unit }}</template></span>
      <Bar :percent="route.percent" :kind="route.percent >= 100 ? 'ok' : 'run'"/>
    </li>
  </ul>
</template>
