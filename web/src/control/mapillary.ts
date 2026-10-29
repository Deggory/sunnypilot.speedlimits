// The Mapillary signs of “Train the classifier”: which part of the Mapillary Traffic Sign Dataset joins train, by
// where its photos were taken (nnslr_tools.mapillary: the region, then the share, then at most 300 of each kind).
import type {MapillaryRegions} from '../common/panel-types';

/** The continents as the server names them (nnslr_tools.gps.CONTINENTS). */
export const CONTINENT_NAMES: Record<string, string> = {
  europe: 'Europe', asia: 'Asia', africa: 'Africa', north_america: 'North America', south_america: 'South America',
  oceania: 'Oceania', antarctica: 'Antarctica',
};

/** A region in words: "All the world", "Europe", "Europe, CH". */
export function regionWords(region: string | undefined): string {
  if (!region || region === 'all') return 'All the world';
  return region.split(',').map((part) => CONTINENT_NAMES[part] ?? part).join(', ');
}

/** The signs of the places chosen (a country inside a chosen continent counted once); every sign when none is. */
export function regionCrops(regions: MapillaryRegions | null | undefined, chosen: readonly string[]): number {
  if (!regions) return 0;
  if (!chosen.length) return regions.crops;
  const continents = new Set(chosen.filter((key) => key in CONTINENT_NAMES));
  let total = 0;
  for (const item of regions.continents) if (continents.has(item.key)) total += item.crops;
  for (const item of regions.countries) {
    if (chosen.includes(item.code) && !(item.continent && continents.has(item.continent))) total += item.crops;
  }
  return total;
}

/** The share typed, in percent, or null when it is not one (more than 0, at most 100). */
export function shareOf(text: string): number | null {
  const value = Number(String(text).replace(',', '.').trim());
  return Number.isFinite(value) && value > 0 && value <= 100 ? Math.round(value * 100) / 100 : null;
}
