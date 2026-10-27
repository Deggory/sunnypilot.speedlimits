#!/usr/bin/env python3
"""The OpenStreetMap roads near the drives, for the "another road" candidates (nnslr_tools.map_context).

  uv run --no-project --with osmium python scripts/osm_roads.py [--data-root DIR] [--pbf FILE] [--margin-m 300]

Reads every ``derived/gps/<route>/positions.jsonl`` of the data root and keeps the OSM ways with a road
``highway`` tag that have a node within about --margin-m of a position. Writes ``derived/osm/roads.jsonl``:
one line per way, ``{"id", "highway", "oneway", "name", "ref", "lanes", "maxspeed", "maxspeed_type", "motorroad",
"nodes": [[lat, lon], ...]}``. Used by nnslr map-candidates and nnslr road-types.

The map: ``raw/osm/switzerland-latest.osm.pbf`` (Geofabrik extract of OpenStreetMap, © OpenStreetMap
contributors, ODbL). It is read here, offline: no position of the drives leaves the server. osmium (pyosmium)
comes from uv's cache for this run only; the environments of the site and of the training stay as they are.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from pathlib import Path

# The roads a car drives on (and their ramps); paths, tracks and cycleways are left out.
ROAD_TYPES = {
    'motorway', 'trunk', 'primary', 'secondary', 'tertiary', 'unclassified', 'residential', 'living_street',
    'service', 'road', 'motorway_link', 'trunk_link', 'primary_link', 'secondary_link', 'tertiary_link',
}
CELL_DEGREES = 0.003  # about 330 m of latitude


def cells_near(root: Path, margin_m: float) -> set[tuple[int, int]]:
    """The grid cells within about margin_m of every GPS position of the data root."""
    reach = max(1, math.ceil(margin_m / 330))
    cells = set()
    for path in sorted((root / 'derived' / 'gps').glob('*/positions.jsonl')):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get('latitude') is None:
                continue
            a = int(item['latitude'] // CELL_DEGREES)
            b = int(item['longitude'] // CELL_DEGREES)
            for da in range(-reach, reach + 1):
                for db in range(-reach, reach + 1):
                    cells.add((a + da, b + db))
    return cells


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--data-root', default=os.environ.get('NNSLR_DATA_ROOT') or str(Path(__file__).resolve().parents[1] / 'data'))
    parser.add_argument('--pbf', help='default: <data root>/raw/osm/switzerland-latest.osm.pbf')
    parser.add_argument('--margin-m', type=float, default=300.0)
    args = parser.parse_args()
    root = Path(args.data_root).resolve()
    pbf = Path(args.pbf) if args.pbf else root / 'raw' / 'osm' / 'switzerland-latest.osm.pbf'
    if not pbf.is_file():
        print(f'osm_roads: missing {pbf}', file=sys.stderr)
        return 2
    import osmium

    cells = cells_near(root, args.margin_m)
    if not cells:
        print('osm_roads: no GPS positions yet (nnslr gps --all)', file=sys.stderr)
        return 2

    class Roads(osmium.SimpleHandler):
        def __init__(self):
            super().__init__()
            self.ways = []

        def way(self, way):
            kind = way.tags.get('highway')
            if kind not in ROAD_TYPES:
                return
            nodes = []
            near = False
            for node in way.nodes:
                if not node.location.valid():
                    continue
                lat, lon = node.location.lat, node.location.lon
                nodes.append([round(lat, 7), round(lon, 7)])
                if not near and (int(lat // CELL_DEGREES), int(lon // CELL_DEGREES)) in cells:
                    near = True
            if near and len(nodes) >= 2:
                tags = way.tags
                # maxspeed_type: the kind of limit when the map says it (CH:urban, CH:rural, CH:zone30...), from
                # maxspeed:type, else source:maxspeed (nnslr_tools.road_types).
                self.ways.append({'id': way.id, 'highway': kind, 'oneway': tags.get('oneway'),
                                  'name': tags.get('name'), 'ref': tags.get('ref'), 'lanes': tags.get('lanes'),
                                  'maxspeed': tags.get('maxspeed'),
                                  'maxspeed_type': tags.get('maxspeed:type') or tags.get('source:maxspeed'),
                                  'motorroad': tags.get('motorroad'), 'nodes': nodes})

    handler = Roads()
    # Node locations kept in a file index: a whole country does not fit comfortably in memory otherwise.
    with tempfile.TemporaryDirectory(dir=root / 'tmp' if (root / 'tmp').is_dir() else None) as scratch:
        index = f'sparse_file_array,{scratch}/nodes.idx'
        handler.apply_file(str(pbf), locations=True, idx=index)
    target = root / 'derived' / 'osm' / 'roads.jsonl'
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix('.jsonl.tmp')
    temporary.write_text(''.join(json.dumps(way, separators=(',', ':')) + '\n' for way in handler.ways))
    temporary.replace(target)
    kinds = {}
    for way in handler.ways:
        kinds[way['highway']] = kinds.get(way['highway'], 0) + 1
    print(json.dumps({'ways': len(handler.ways), 'cells': len(cells), 'by_type': dict(sorted(kinds.items())),
                      'output': str(target)}, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
