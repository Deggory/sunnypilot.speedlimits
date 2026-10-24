#!/usr/bin/env python3
"""Make src/nnslr_tools/data/countries.json.gz, the country borders the review uses to propose the country of a
frame from its GPS position (nnslr_tools.gps.country_of).

Source: Natural Earth 1:10m admin-0 countries (public domain, https://www.naturalearthdata.com), the GeoJSON of
github.com/nvkelso/natural-earth-vector. Only the countries in the AREA box (Europe and around it) are kept,
with coordinates rounded to 4 decimals (about 10 m), so the file stays small; elsewhere country_of answers
None. Run again with a larger AREA to cover more countries.

  python scripts/make_country_borders.py                 download the source and write the file
  python scripts/make_country_borders.py <file.geojson>  use a source already downloaded
"""
from __future__ import annotations

import gzip
import json
import sys
import urllib.request
from pathlib import Path

SOURCE = ('https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/'
          'ne_10m_admin_0_countries.geojson')
# lon_min, lat_min, lon_max, lat_max: Europe, from Portugal and Iceland to the Urals, Turkey and North Africa.
AREA = (-32.0, 27.0, 50.0, 72.0)
OUTPUT = Path(__file__).resolve().parents[1] / 'src/nnslr_tools/data/countries.json.gz'


def rounded(ring: list) -> list:
    points = []
    for lon, lat, *_ in ring:
        point = [round(lon, 4), round(lat, 4)]
        if not points or points[-1] != point:
            points.append(point)
    return points


def main() -> int:
    if len(sys.argv) > 1:
        data = json.loads(Path(sys.argv[1]).read_text())
    else:
        with urllib.request.urlopen(SOURCE, timeout=120) as response:
            data = json.loads(response.read())
    countries = []
    for feature in data['features']:
        properties = feature['properties']
        # ISO_A2_EH also codes the countries ISO_A2 leaves as -99 (France, Norway).
        code = properties.get('ISO_A2_EH') or properties.get('ISO_A2')
        if not code or code == '-99':
            continue
        geometry = feature['geometry']
        polygons = geometry['coordinates'] if geometry['type'] == 'MultiPolygon' else [geometry['coordinates']]
        kept = []
        for polygon in polygons:
            outer = polygon[0]
            lons = [point[0] for point in outer]
            lats = [point[1] for point in outer]
            if max(lons) < AREA[0] or min(lons) > AREA[2] or max(lats) < AREA[1] or min(lats) > AREA[3]:
                continue
            kept.append([rounded(ring) for ring in polygon])
        if not kept:
            continue
        lons = [point[0] for polygon in kept for point in polygon[0]]
        lats = [point[1] for polygon in kept for point in polygon[0]]
        countries.append({'code': code.upper(), 'name': properties.get('NAME_EN') or properties.get('NAME'),
                          'bbox': [min(lons), min(lats), max(lons), max(lats)], 'polygons': kept})
    countries.sort(key=lambda country: country['code'])
    payload = {'source': SOURCE, 'license': 'Natural Earth, public domain', 'area': list(AREA),
               'countries': countries}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_bytes(gzip.compress(json.dumps(payload, separators=(',', ':')).encode(), mtime=0))
    print(f'{len(countries)} countries -> {OUTPUT} ({OUTPUT.stat().st_size // 1024} KiB)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
