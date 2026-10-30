// The choices of the classifier's next training, kept once for Train › Classifier and Train › Pipeline (the same
// choices in both): its signs (ours, with Mapillary in train, or Mapillary first then ours), the places and the share
// of the Mapillary signs, and where it starts (ImageNet or a finished classifier of Models).
import {computed, reactive, ref, type Ref} from 'vue';
import {isoWhen, thousands} from '../common/format';
import type {ActionOptions, ClassifierModel, ModelRole, PanelStatus} from '../common/panel-types';
import type {PickerOption} from '../components/PickerSheet.vue';
import {jobOf, terminalActivities} from './jobs';
import {regionCrops, shareOf} from './mapillary';

export type MapillaryChoice = 'off' | 'on' | 'pretrain';
export const MAPILLARY_OPTIONS: {key: MapillaryChoice; label: string}[] = [{key: 'off', label: 'Our signs'},
  {key: 'on', label: 'With Mapillary'}, {key: 'pretrain', label: 'Mapillary, then ours'}];
type Split = 'same' | 'another' | 'unknown';
const SPLIT_WORDS: Record<Split, string> = {same: 'learnt on today’s split', another: 'learnt on another split',
  unknown: 'its split not recorded'};

export function useClassifierChoices(status: Ref<PanelStatus | null>,
  versionsOf: (role: ModelRole) => readonly {role: string}[] | undefined) {
  /** Whether the Mapillary crops are prepared (status.classifier_options). */
  const mapillaryState = computed(() => status.value?.classifier_options?.mapillary ?? null);
  const pickedMapillary = ref<MapillaryChoice | null>(null);
  const mapillary = computed({
    get: (): MapillaryChoice => mapillaryState.value?.available ? pickedMapillary.value ?? 'off' : 'off',
    set: (value: MapillaryChoice) => { pickedMapillary.value = value; },
  });
  const withMapillary = computed(() => mapillary.value !== 'off');
  /** Which Mapillary signs: the places their photos were taken (none chosen: all the world) and a share of them, in
   * percent (status.classifier_options.mapillary_regions; with Mapillary in train the server then keeps at most 300 of
   * each kind, its pretraining takes every one). */
  const mapillaryRegions = computed(() => status.value?.classifier_options?.mapillary_regions ?? null);
  const region = ref<string[]>([]);
  const shareText = ref('100');
  const share = computed(() => shareOf(shareText.value));
  const locateJob = computed(() => jobOf(status.value, ['mapillary_locations']));
  const locating = computed(() => locateJob.value?.state === 'running'
    || terminalActivities(status.value, ['mapillary']).length > 0);
  /** About how many Mapillary signs join: those of the places chosen, times the share. */
  const mapillarySigns = computed(() => Math.round(regionCrops(mapillaryRegions.value, region.value) * (share.value ?? 0) / 100));
  const mapillaryOptions = computed((): ActionOptions => withMapillary.value
    ? {mapillary: mapillary.value, region: region.value.length ? region.value.join(',') : 'all', share: String(share.value ?? shareText.value)}
    : {mapillary: 'off'});
  /** Where the classifier starts: a finished classifier of Models with “not a sign”, whose weights it goes on learning
   * from (the one in use first, and the choice until another is made), or ImageNet (its test stays fair; the choice
   * while none is in use). Not with “Mapillary, then ours”, which starts from its own Mapillary stage. A start that
   * learnt another split may have seen some of today's test signs. */
  const startVersions = computed((): ClassifierModel[] => ((versionsOf('classifier') ?? []) as ClassifierModel[])
    .filter((version) => version.role === 'classifier' && version.part === 'classifier'
      && version.class_scheme !== undefined && version.finished)
    .sort((a, b) => Number(b.in_use) - Number(a.in_use)));
  const currentSplit = computed(() => status.value?.classifier_options?.split_sha256 ?? null);
  const startSplit = (version: ClassifierModel): Split => !version.split_sha256 || !currentSplit.value
    ? 'unknown' : version.split_sha256 === currentSplit.value ? 'same' : 'another';
  const startOptions = computed((): PickerOption[] => [
    {key: 'imagenet', title: 'ImageNet', sub: 'a fresh start: always a fair test'},
    ...startVersions.value.map((version) => ({key: version.id, title: version.label + ' ' + isoWhen(version.trained_at),
      short: version.label + ' ' + isoWhen(version.trained_at), badge: version.in_use ? 'In use' : undefined,
      sub: SPLIT_WORDS[startSplit(version)]})),
  ]);
  /** The start chosen here (null: none yet, the one in use). */
  const pickedStart = ref<string | null>(null);
  const startKey = computed({
    get: (): string => pickedStart.value === 'imagenet' || startVersions.value.some((version) => version.id === pickedStart.value)
      ? pickedStart.value! : startVersions.value.find((version) => version.in_use)?.id ?? 'imagenet',
    set: (value: string) => { pickedStart.value = value; },
  });
  const startVersion = computed(() => mapillary.value === 'pretrain' ? null
    : startVersions.value.find((version) => version.id === startKey.value) ?? null);
  /** The options of “Train the classifier” (ActionOptions). */
  const options = computed((): ActionOptions => mapillary.value === 'pretrain' ? mapillaryOptions.value
    : {...mapillaryOptions.value, start: startVersion.value?.id ?? 'imagenet'});
  /** What it learns from, in words: after "A new classifier, taught on the signs and on the cutter’s mistakes". */
  const words = computed(() => (mapillary.value === 'on' ? ', with about ' + thousands(mapillarySigns.value) + ' signs of Mapillary' : '')
    + (mapillary.value === 'pretrain' ? ', first on about ' + thousands(mapillarySigns.value) + ' signs of Mapillary alone, '
      + 'then on ours from there' : '')
    + (startVersion.value ? ', starting from ' + startVersion.value.label + ' (' + isoWhen(startVersion.value.trained_at) + ')' : ''));
  /** Why it cannot start with these choices ('' when it can): a share that is not a percentage. */
  const invalid = computed(() => withMapillary.value && share.value === null
    ? 'the share of Mapillary is a percentage, more than 0 and at most 100' : '');
  return reactive({mapillaryState, mapillary, withMapillary, mapillaryRegions, region, shareText, share, locateJob, locating,
    mapillarySigns, startOptions, startKey, startVersion, startSplit, options, words, invalid});
}
export type ClassifierChoices = ReturnType<typeof useClassifierChoices>;
