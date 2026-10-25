<script setup lang="ts">
// The drive under review, the menu to open another one, its count and who reviews (ReviewToolbar.vue), at the top
// of the inspector (above the country of the frame): its name ("Basel → Mumpf · Wed 21 Oct 2026 · 13:55", pages.route_links), the route's code only in the title.
import {computed} from 'vue';
import ReviewToolbar from './ReviewToolbar.vue';
import {report} from './store';

/** The routes of the server, this one chosen; offline only this report. */
const options = computed(() => {
  const list = report.routes.map((route) => ({
    value: String(route.url || ''), text: String(route.name || route.route_id || ''), selected: route.route_id === report.routeId,
  }));
  if (!list.some((option) => option.selected)) {
    list.unshift({value: '', text: report.routeId ? report.routeId : 'Local report', selected: true});
  }
  return list;
});

function changeRoute(event: Event): void {
  const value = (event.target as HTMLSelectElement).value;
  if (value.startsWith('/route/')) window.location.assign(value);
}
</script>

<template>
  <div class="route-pick" :title="report.routeId">
    <label for="route" class="route-pick-label">Drive</label>
    <select id="route" :disabled="report.routes.length <= 1" @change="changeRoute">
      <option v-for="option in options" :key="option.value + option.text" :value="option.value"
              :selected="option.selected">{{ option.text }}</option>
    </select>
    <ReviewToolbar/>
  </div>
</template>
