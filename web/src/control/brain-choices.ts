// What the next training of the brain learns, kept once for Train › Brain and for its step of Train › Pipeline (the same
// choice in both): its roadworks head, its head of the kinds of road of Speed review (a person or a vehicle crossing our
// path among them since 17 Nov 2026, #112), its head of the road type (the map's,
// changed in Speed review: then only the seconds with one), the picture's features as a grid or as a mean, seed 0 alone
// (the default since 11 Nov 2026) or its three seeds (to compare in Models), three times the time, the map's limit as an
// input (since 14 Nov 2026, #85: off unless turned on), its applicability head (since 15 Nov 2026, #89: whose road a
// speed sign is of, learnt from the boxes of Sign review; the brain is then given every speed sign read), how the number in
// force reaches the speed (since 16 Nov 2026, #104: as a bell around it, or on its class alone as every brain before, to
// compare the two) and its drives: every
// dashcam drive, or the same as an earlier brain run (routes_from, its routes.json). The driver's
// inputs (the gas, the brake and the car's speed) are gone since 10 Nov 2026: the brain trained with them followed his speed.
// Its road users head (12 to 17 Nov 2026) is gone too: the crossings are kinds of road marked in Speed review.
import {computed, reactive, ref} from 'vue';
import type {ActionOptions} from '../common/panel-types';

export type OnOff = 'on' | 'off';
export type BrainPool = 'mean' | 'grid';
export const POOL_OPTIONS: {key: BrainPool; label: string}[] = [{key: 'mean', label: 'Mean'}, {key: 'grid', label: 'Grid'}];
/** How the number in force reaches the speed (brain_net.NUMBER_PATHS, #104): a bell around it, or its class alone. */
export type BrainNumberPath = 'bell' | 'class';
export const NUMBER_PATH_OPTIONS: {key: BrainNumberPath; label: string}[] = [{key: 'bell', label: 'Bell'}, {key: 'class', label: 'Class'}];
export type BrainSeeds = 'all' | 'one';
export const SEEDS_OPTIONS: {key: BrainSeeds; label: string}[] = [{key: 'one', label: 'One'}, {key: 'all', label: 'Three'}];

export function useBrainChoices() {
  const works = ref<OnOff>('on');
  const road = ref<OnOff>('on');
  const types = ref<OnOff>('on');
  const pool = ref<BrainPool>('grid');
  /** The number path (#104, 16 Nov 2026): bell, the usual one; class, the path of every brain before, the comparison. */
  const numberPath = ref<BrainNumberPath>('bell');
  const seeds = ref<BrainSeeds>('one');
  /** The map's limit the comma had as one more input of the brain (#85, 14 Nov 2026): off unless turned on, the one
   * choice here that is an input and not a head. */
  const map = ref<OnOff>('off');
  /** Whose road a speed sign is of, as a head of the brain learnt from the reviewed boxes (#89, 15 Nov 2026): on, and the
   * brain is given the boxes of every speed sign read; off, only the signs the applicability network takes. */
  const applicability = ref<OnOff>('on');
  /** The earlier brain run whose drives it takes (BrainLabels.same_drives), '' for every dashcam drive. */
  const routesFrom = ref('');
  /** The options of its training (ActionOptions.works, road, types, pool, number_path; seeds only for seed 0 alone, the
   * three when not given). */
  const option = computed((): ActionOptions => ({works: works.value, road: road.value, types: types.value,
    pool: pool.value, number_path: numberPath.value, applicability: applicability.value,
    ...(map.value === 'on' ? {map: 'on' as const} : {}),
    ...(seeds.value === 'one' ? {seeds: 'one' as const} : {}),
    ...(routesFrom.value ? {routes_from: routesFrom.value} : {})}));
  /** What it learns in words, after the training's own: "", or ", without the roadworks". */
  const words = computed(() => (works.value === 'off' ? ', without the roadworks' : '')
    + (road.value === 'off' ? ', without the kinds of road' : '')
    + (types.value === 'off' ? ', without the road types' : '')
    + (applicability.value === 'off' ? ', without the applicability head' : '')
    + (map.value === 'on' ? ', with the map’s limit as an input' : '')
    + (routesFrom.value ? ', on the drives of ' + routesFrom.value : '')
    + (pool.value === 'mean' ? ', with the picture as a mean' : '')
    + (numberPath.value === 'class' ? ', with the number on its class alone' : ''));
  /** What Mean and Grid are, on the picture's choice. */
  const note = 'Grid keeps where things are in the picture, Mean only what is in it.';
  /** What Bell and Class are, on the number's choice. */
  const numberNote = 'How the number of the sign in force reaches the speed: Bell, as a bell around it, the shape of the answer; '
    + 'Class, on its one class, as every brain before 16 Nov 2026 (to compare the two).';
  /** What the seeds are, on their choice. */
  const seedsNote = 'Seed 0 alone, or three networks that differ only by chance, to compare in Models, in three times the time.';
  /** What Drives is, on its choice. */
  const drivesNote = 'Every dashcam drive, or the drives an earlier brain run took: two runs on the same drives compare fairly in Models.';
  return reactive({works, road, types, pool, numberPath, seeds, map, applicability, routesFrom, option, words, note,
    numberNote, seedsNote, drivesNote});
}
export type BrainChoice = ReturnType<typeof useBrainChoices>;
