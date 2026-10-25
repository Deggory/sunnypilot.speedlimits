#!/usr/bin/env python3
"""The names of the towns and villages, to call each drive by where it went (nnslr_tools.drives).

  uv run --no-project --with osmium python scripts/osm_places.py [--data-root DIR] [--pbf FILE]

Keeps the OSM nodes tagged ``place`` = city, town, village or suburb that have a name. Writes
``derived/osm/places.jsonl``: one line per place, ``{"name", "place", "lat", "lon", "population"}``.

The map: ``raw/osm/switzerland-latest.osm.pbf`` (Geofabrik extract of OpenStreetMap, © OpenStreetMap contributors,
ODbL), read here, offline: no position of the drives leaves the server. osmium (pyosmium) comes from uv's cache for
this run only; the environments of the site and of the training stay as they are. Run it again after a newer map.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

PLACES = {'city', 'town', 'village', 'suburb'}


def main() -> int:
    import osmium

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--data-root', default=os.environ.get('NNSLR_DATA_ROOT') or str(Path(__file__).resolve().parents[1] / 'data'))
    parser.add_argument('--pbf', help='default: <data-root>/raw/osm/switzerland-latest.osm.pbf')
    args = parser.parse_args()
    root = Path(args.data_root)
    pbf = Path(args.pbf) if args.pbf else root / 'raw' / 'osm' / 'switzerland-latest.osm.pbf'
    found = []

    class Places(osmium.SimpleHandler):
        def node(self, node):
            kind = node.tags.get('place')
            name = node.tags.get('name')
            if kind in PLACES and name and node.location.valid():
                population = node.tags.get('population', '')
                found.append({'name': name, 'place': kind, 'lat': round(node.location.lat, 5),
                              'lon': round(node.location.lon, 5),
                              'population': int(population) if population.isdigit() else None})

    Places().apply_file(str(pbf))
    target = root / 'derived' / 'osm' / 'places.jsonl'
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', dir=target.parent, delete=False, suffix='.tmp') as handle:
        for item in sorted(found, key=lambda entry: (entry['name'], entry['lat'])):
            handle.write(json.dumps(item, ensure_ascii=False) + '\n')
    os.replace(handle.name, target)
    print(f'{len(found)} places written to {target}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
