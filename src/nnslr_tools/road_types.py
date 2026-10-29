"""The type of road of every frame, from the map: the labels of the road-type head, prepared without a person.

``nnslr road-types --all`` reads the GPS position, speed and heading of every frame
(``derived/gps/<route>/positions.jsonl``, nnslr gps) and the OpenStreetMap roads and car parks near the drives
(``derived/osm/roads.jsonl`` and ``parking.jsonl``, scripts/osm_roads.py, read offline from the local Geofabrik
extracts). For each frame it finds the road we drive on (the closest one along our heading, map_context.our_road) and
names its type:

- parking, a car park: our road is a parking aisle (``highway=service``, ``service=parking_aisle``: "parking aisle",
  sure), or the position is inside a car park outline (``amenity=parking``: "car park on the map") and our road is a
  service road, or no road is found, or the car is slower than PARKING_SPEED (the GPS finds the street next to the car
  park). Faster than PARKING_MAX_SPEED an aisle or a service road found is the GPS slipping, not a car park. Sure only when the outline and a service road agree. An underground car park lies under the streets: slow
  on a street over it is not parking. Its limit to sunnypilot is always 10 km/h (a sign read there counts only when lower), and seeing it drops
  the limit remembered from the road before;
- motorway: ``highway=motorway``;
- expressway: ``highway=trunk`` (the Swiss Autostrasse), unless the map puts it in a town;
- ramp: a ramp of a motorway or an expressway (``highway=motorway_link``, ``trunk_link`` out of town), the road
  where the limit of a ramp holds until the carriageway (the road-type network learns the merge from it);
- town or rural, for every other road: the kind of limit when the map says it (``CH:urban``, ``CH:zone30`` → town,
  ``CH:rural`` → rural), else its limit (50 or less → town, 60 and more → rural), else its kind (a residential
  street, a living street → town). A road that says none of these stays unknown.

Then along the drive: a gap of unknown frames between two stretches of the same type takes that type (a tunnel, a
road of the map without a limit), and a stretch shorter than MIN_RUN frames between two stretches of the same type
takes theirs (the GPS jumping to a side road, a bridge or a car park beside the road for a second).

Writes ``derived/road_types/<route>.jsonl``: one line per frame, ``{"frame_key", "segment_index", "output_index",
"road_type" (town, rural, expressway, motorway, ramp, parking or null), "map_type" (before the drive was looked at), "basis"
(why: "maxspeed 50", "highway motorway", "parking aisle", "car park on the map", "filled", "smoothed", "no road", "no
gps"...), "sure" (false where the map is often wrong: WEAK_BASES, a car park of one signal, filled, smoothed),
"way_id", "highway", "maxspeed"}`` (the road found, also in a car park).

Measured on 27 Oct 2026, 240 frames of 11 drives looked at without the map: where the eye was sure, the map agreed on
91% of town frames, 75% of rural ones and 95% of motorway ones; most rural misses were roads at 60 in a town.

The map labels are a proposal: OpenStreetMap can be wrong or old, and a person corrects them in the road-type review
(Data › Roads, a drive opened). A correction is kept apart, in ``annotations/road_types/<route>.jsonl``: one line per
correction, appended, ``{"from", "to" (frame keys, the stretch), "road_type" (null: the person cannot tell), "reviewer",
"at"}``; the last line covering a frame wins. ``nnslr road-types`` never touches it, so a correction survives a new map
run; load_road_types gives the map labels with the corrections over them.
OpenStreetMap data © OpenStreetMap contributors (ODbL).
"""
from __future__ import annotations

import re
import threading
from datetime import datetime
from pathlib import Path

from nnslr_tools.annotations import atomic_write, jsonl_bytes, load_jsonl, strict_json_loads
from nnslr_tools.manifest import parse_route_id
from nnslr_tools.map_context import CELL_DEGREES, OUR_ROAD_M, RoadMap, our_road

ROAD_TYPES_RELATIVE = Path('derived') / 'road_types'
CORRECTIONS_RELATIVE = Path('annotations') / 'road_types'
PARKING_RELATIVE = Path('derived') / 'osm' / 'parking.jsonl'
ROAD_TYPES = ('town', 'rural', 'expressway', 'motorway', 'ramp', 'parking')
# Slower than this (m/s) inside a car park outline: parking, even with a street of the map within OUR_ROAD_M.
PARKING_SPEED = 3.0
# Faster than this (36 km/h) an aisle or a service road found is the GPS slipping onto a car park beside the road (a
# motorway ramp at Basel St. Jakob, 27 Oct 2026), never a car park: nobody drives through one that fast.
PARKING_MAX_SPEED = 10.0
# The longest gap of unknown frames (one a second) filled from the same type on both sides: a long tunnel.
MAX_GAP = 60
# A stretch shorter than this between two stretches of one type is a slip of the GPS, not a change of road.
MIN_RUN = 4
TOWN_KINDS = ('urban', 'zone')
# The reasons the map is often wrong about, measured on 27 Oct 2026 against 240 frames looked at without the map:
# a road at 60 was in a town one time in three, a ramp is between two kinds of road, a filled gap is mostly a tunnel.
# Their frames are not "sure": the review shows them first, and a training can leave them out until reviewed.
WEAK_BASES = ('maxspeed 60', 'maxspeed 70', 'highway motorway_link', 'highway trunk_link')
TOWN_HIGHWAYS = ('residential', 'living_street')
# The picture of a frame in its pre-annotation report (preview_path), served at /route/<route>/images/<8 digits>.jpg.
PREVIEW = re.compile(r'images/[0-9]{8}\.jpg')
# The drives of Data › Roads (summary) by data root: (every road-types file with its time and the time of its
# corrections, the drives read from them). About 20,000 lines for 11 drives take a fifth of a second: read again only
# when a file changed.
_DRIVES: dict[str, tuple[tuple, list[dict]]] = {}
# One correction written at a time (the request threads of the site): each rewrites its file with one line more.
_CORRECTING = threading.Lock()


class CorrectionError(ValueError):
    """A correction that cannot be kept: a frame that is not of the drive, a type that is not one."""


def _limit(value) -> int | None:
    """The limit of a maxspeed tag in km/h ("50", "50 mph" is not Swiss: None), or None."""
    if not isinstance(value, str):
        return None
    text = value.strip()
    return int(text) if text.isdigit() else None


def _kind(way: dict) -> str | None:
    """'urban', 'rural', 'zone' or 'motorway' from maxspeed_type, or a maxspeed written as "CH:rural"."""
    for value in (way.get('maxspeed_type'), way.get('maxspeed')):
        if isinstance(value, str) and ':' in value:
            kind = value.split(':', 1)[1].strip().lower()
            if kind.startswith('zone'):
                return 'zone'
            if kind in ('urban', 'rural', 'motorway', 'trunk'):
                return kind
    return None


def way_type(way: dict) -> tuple[str | None, str]:
    """The road type of an OSM way and why: ('town', 'maxspeed 50')."""
    highway = way.get('highway') or ''
    limit = _limit(way.get('maxspeed'))
    kind = _kind(way)
    if highway == 'motorway':
        return 'motorway', 'highway motorway'
    if highway == 'motorway_link':
        return 'ramp', 'highway motorway_link'
    if highway in ('trunk', 'trunk_link'):
        if kind in TOWN_KINDS or (limit is not None and limit <= 50):
            return 'town', 'highway ' + highway + ' in town'
        return ('ramp' if highway == 'trunk_link' else 'expressway'), 'highway ' + highway
    if kind in TOWN_KINDS:
        return 'town', 'limit ' + str(way.get('maxspeed_type') or way.get('maxspeed'))
    if kind == 'rural':
        return 'rural', 'limit ' + str(way.get('maxspeed_type') or way.get('maxspeed'))
    if limit is not None:
        return ('town' if limit <= 50 else 'rural'), 'maxspeed ' + str(limit)
    if highway in TOWN_HIGHWAYS:
        return 'town', 'highway ' + highway
    return None, 'highway ' + (highway or 'road') + ' without a limit'


def _inside(rings: list, lat: float, lon: float) -> bool:
    """Whether a position is inside an outline: inside an odd number of its rings (an inner ring is a hole)."""
    inside = False
    for ring in rings:
        for (a_lat, a_lon), (b_lat, b_lon) in zip(ring, ring[1:] + ring[:1]):
            if (a_lat > lat) != (b_lat > lat) and lon < a_lon + (lat - a_lat) * (b_lon - a_lon) / (b_lat - a_lat):
                inside = not inside
    return inside


class ParkingMap:
    """The car parks near the drives (derived/osm/parking.jsonl, scripts/osm_roads.py), with a grid index of their
    outlines. Empty before osm_roads.py wrote the file: no frame is parking then but by its aisle."""

    def __init__(self, parks: list[dict]):
        self.parks = parks
        self.boxes: list[tuple[float, float, float, float]] = []
        self.cells: dict[tuple[int, int], list[int]] = {}
        for p, park in enumerate(parks):
            lats = [point[0] for ring in park['rings'] for point in ring]
            lons = [point[1] for ring in park['rings'] for point in ring]
            self.boxes.append((min(lats), min(lons), max(lats), max(lons)))
            for a in range(int(min(lats) // CELL_DEGREES), int(max(lats) // CELL_DEGREES) + 1):
                for b in range(int(min(lons) // CELL_DEGREES), int(max(lons) // CELL_DEGREES) + 1):
                    self.cells.setdefault((a, b), []).append(p)

    @classmethod
    def load(cls, root: Path) -> ParkingMap:
        path = Path(root) / PARKING_RELATIVE
        if not path.is_file():
            return cls([])
        return cls([park for park in load_jsonl(path) if isinstance(park, dict) and park.get('rings')
                    and all(isinstance(ring, list) and len(ring) >= 3 for ring in park['rings'])])

    def containing(self, lat: float, lon: float) -> dict | None:
        """The car park whose outline holds the position, None outside every one."""
        for p in self.cells.get((int(lat // CELL_DEGREES), int(lon // CELL_DEGREES)), ()):
            south, west, north, east = self.boxes[p]
            if south <= lat <= north and west <= lon <= east and _inside(self.parks[p]['rings'], lat, lon):
                return self.parks[p]
        return None


def parking_basis(way: dict | None, park: dict | None, speed) -> tuple[str, bool] | None:
    """Why a frame is in a car park and whether that is sure, None when it is not: way is our road (None: no road
    found), park the car park whose outline holds the position (None: outside every one), speed in m/s or None."""
    service = way is not None and way.get('highway') == 'service'
    fast = isinstance(speed, (int, float)) and speed >= PARKING_MAX_SPEED
    if service and way.get('service') == 'parking_aisle' and not fast:
        return 'parking aisle', True
    if park is None:
        return None
    slow = isinstance(speed, (int, float)) and speed < PARKING_SPEED and park.get('parking') != 'underground'
    if (service and not fast) or way is None or slow:
        # Both signals agree only when our road is a service road of the car park.
        return 'car park on the map', service
    return None


def _runs(types: list[str | None]) -> list[list]:
    """[[type, start, end], ...]: the stretches of equal type, end excluded."""
    runs: list[list] = []
    for index, value in enumerate(types):
        if runs and runs[-1][0] == value:
            runs[-1][2] = index + 1
        else:
            runs.append([value, index, index + 1])
    return runs


def smooth(types: list[str | None]) -> list[tuple[str | None, str | None]]:
    """The types along a drive (frames in order), filled and smoothed: [(type, 'filled' | 'smoothed' | None)]."""
    result: list[tuple[str | None, str | None]] = [(value, None) for value in types]
    # Gaps of unknown frames with the same type on both sides.
    runs = _runs([value for value, _ in result])
    for at, (value, start, end) in enumerate(runs):
        if value is None and 0 < at < len(runs) - 1 and runs[at - 1][0] == runs[at + 1][0] and end - start <= MAX_GAP:
            for index in range(start, end):
                result[index] = (runs[at - 1][0], 'filled')
    # Short stretches between two of one type, repeated until none is left (two slips side by side).
    changed = True
    while changed:
        changed = False
        runs = _runs([value for value, _ in result])
        for at, (value, start, end) in enumerate(runs):
            if 0 < at < len(runs) - 1 and runs[at - 1][0] == runs[at + 1][0] and runs[at - 1][0] is not None \
                    and value != runs[at - 1][0] and end - start < MIN_RUN:
                for index in range(start, end):
                    result[index] = (runs[at - 1][0], 'smoothed')
                changed = True
                break
    return result


def frame_types(road_map: RoadMap, positions: list[dict], parking: ParkingMap | None = None) -> list[dict]:
    """One line per position (in the order of the drive) with its road type."""
    parking = ParkingMap([]) if parking is None else parking
    lines = []
    # Per line: whether the map is often wrong about its reason (not "sure" even when not filled or smoothed).
    weak = []
    for position in positions:
        line = {'frame_key': position['frame_key'], 'segment_index': position.get('segment_index'),
                'output_index': position.get('output_index'), 'road_type': None, 'map_type': None,
                'basis': 'no gps', 'way_id': None, 'highway': None, 'maxspeed': None}
        doubtful = False
        lat, lon, heading = position.get('latitude'), position.get('longitude'), position.get('bearing_deg')
        if lat is not None and lon is not None and heading is not None:
            near = road_map.nearest(lat, lon, OUR_ROAD_M)
            ours = our_road(near, heading)
            way = None if ours is None else road_map.ways[ours]
            if way is None:
                line['basis'] = 'no road'
            else:
                kind, basis = way_type(way)
                line.update(map_type=kind, basis=basis, way_id=way.get('id'), highway=way.get('highway'),
                            maxspeed=way.get('maxspeed'))
                doubtful = basis in WEAK_BASES or basis.endswith(' in town')
            parked = parking_basis(way, parking.containing(lat, lon), position.get('speed_mps'))
            if parked is not None:
                line.update(map_type='parking', basis=parked[0])
                doubtful = not parked[1]
        lines.append(line)
        weak.append(doubtful)
    for line, doubtful, (kind, how) in zip(lines, weak, smooth([line['map_type'] for line in lines])):
        line['road_type'] = kind
        line['sure'] = kind is not None and how is None and not doubtful
        if how is not None:
            line['basis'] = how + ' (' + line['basis'] + ')'
    return lines


def road_types_path(root: Path, route: str) -> Path:
    parse_route_id(route)
    return Path(root) / ROAD_TYPES_RELATIVE / f'{route}.jsonl'


def build_road_types(root: Path, route: str, positions: list[dict], road_map: RoadMap,
                     parking: ParkingMap | None = None) -> dict:
    """Writes the road types of a route; returns how many frames of each type. parking: the car parks, read from the
    data root when not given."""
    ordered = sorted(positions, key=lambda item: (item.get('segment_index') or 0, item.get('output_index') or 0))
    lines = frame_types(road_map, ordered, ParkingMap.load(root) if parking is None else parking)
    target = road_types_path(root, route)
    atomic_write(target, jsonl_bytes(lines))
    counts: dict[str, int] = {}
    for line in lines:
        key = line['road_type'] or 'unknown'
        counts[key] = counts.get(key, 0) + 1
    return {'route_id': route, 'frames': len(lines), 'types': dict(sorted(counts.items())), 'output': str(target)}


def map_road_types(root: Path, route: str) -> list[dict]:
    """The lines of the map of a route in the order of the drive, [] when nnslr road-types has not run for it."""
    path = road_types_path(root, route)
    if not path.is_file():
        return []
    return [line for line in load_jsonl(path) if isinstance(line, dict) and 'frame_key' in line]


def corrections_path(root: Path, route: str) -> Path:
    parse_route_id(route)
    return Path(root) / CORRECTIONS_RELATIVE / f'{route}.jsonl'


def load_corrections(root: Path, route: str) -> list[dict]:
    """A person's corrections of a route in the order they were made, [] before the first. A line that cannot be read
    (broken JSON, from or to not a frame key, road_type missing or not one) is skipped: the others still count, and
    the drive still opens."""
    path = corrections_path(root, route)
    if not path.is_file():
        return []
    found = []
    for text in path.read_text(encoding='utf-8', errors='replace').splitlines():
        try:
            line = strict_json_loads(text)
        except (ValueError, TypeError):
            continue
        if isinstance(line, dict) and isinstance(line.get('from'), str) and isinstance(line.get('to'), str) \
                and 'road_type' in line and (line['road_type'] is None or line['road_type'] in ROAD_TYPES):
            found.append(line)
    return found


def corrected(lines: list[dict], corrections: list[dict]) -> list[dict]:
    """The map lines with a person's corrections over them (the last covering a frame wins): road_type from the person,
    basis "reviewed", sure, reviewed true and the reviewer. A correction whose frames left the drive is skipped."""
    where = {line['frame_key']: index for index, line in enumerate(lines)}
    chosen: list[dict | None] = [None] * len(lines)
    for correction in corrections:
        start, end = where.get(correction['from']), where.get(correction['to'])
        if start is None or end is None:
            continue
        for index in range(min(start, end), max(start, end) + 1):
            chosen[index] = correction
    return [line if correction is None else
            {**line, 'road_type': correction.get('road_type'), 'basis': 'reviewed', 'sure': True, 'reviewed': True,
             'reviewer': correction.get('reviewer')}
            for line, correction in zip(lines, chosen)]


def load_road_types(root: Path, route: str) -> list[dict]:
    """The lines of a route in the order of the drive with a person's corrections over the map (corrected), [] when
    nnslr road-types has not run for it. Every reader takes these: Data › Roads, its API, a training."""
    lines = map_road_types(root, route)
    corrections = load_corrections(root, route) if lines else []
    return corrected(lines, corrections) if corrections else lines


def correct(root: Path, route: str, start, end, road_type, reviewer: str) -> dict:
    """Keeps a person's road type for the frames from start to end (frame keys of the drive's map lines, in either
    order; null: they cannot tell), appended to the corrections of the route. Returns the line kept."""
    keys = {line['frame_key'] for line in map_road_types(root, route)}
    if not isinstance(start, str) or not isinstance(end, str) or start not in keys or end not in keys:
        raise CorrectionError('from and to are frames of this drive.')
    if road_type is not None and road_type not in ROAD_TYPES:
        raise CorrectionError('road_type is one of ' + ', '.join(ROAD_TYPES) + ', or null when it cannot be told.')
    line = {'from': start, 'to': end, 'road_type': road_type, 'reviewer': reviewer,
            'at': datetime.now().astimezone().isoformat(timespec='seconds')}
    path = corrections_path(root, route)
    with _CORRECTING:
        before = path.read_bytes() if path.is_file() else b''
        if before and not before.endswith(b'\n'):
            before += b'\n'
        atomic_write(path, before + jsonl_bytes([line]))
    return line


def undo_correction(root: Path, route: str) -> bool:
    """Drops the last correction of a route (Undo of the page); False when there is none."""
    path = corrections_path(root, route)
    with _CORRECTING:
        lines = [line for line in path.read_bytes().splitlines(keepends=True) if line.strip()] if path.is_file() else []
        if not lines:
            return False
        atomic_write(path, b''.join(lines[:-1]))
    return True


def stretches(lines: list[dict]) -> list[dict]:
    """The drive as stretches of one road type: [{'type', 'from', 'to', 'frames'}], from/to frame keys."""
    found = []
    for value, start, end in _runs([line.get('road_type') for line in lines]):
        found.append({'type': value, 'from': lines[start]['frame_key'], 'to': lines[end - 1]['frame_key'],
                      'frames': end - start})
    return found


def type_counts(lines: list[dict]) -> dict[str, int]:
    """The frames of each type: {town, rural, expressway, motorway, ramp, parking, unknown: frames}."""
    types = {key: 0 for key in (*ROAD_TYPES, 'unknown')}
    for line in lines:
        key = line.get('road_type') if line.get('road_type') in ROAD_TYPES else 'unknown'
        types[key] += 1
    return types


def _drive(root: Path, route: str) -> dict:
    """One drive of summary(): its name, the frames of each type, those a person set, and its stretches."""
    from nnslr_tools.drives import drive_of
    lines = load_road_types(root, route)
    drive = drive_of(root, route) or {}
    return {'route': route, 'title': drive.get('title'), 'when': drive.get('when'), 'start': drive.get('start'),
            'minutes': drive.get('minutes'), 'frames': len(lines), 'types': type_counts(lines),
            'reviewed': sum(1 for line in lines if line.get('reviewed')), 'stretches': stretches(lines)}


def drive_view(root: Path, route: str, by_key: dict) -> dict:
    """A drive of Data › Roads opened (GET /api/panel/road-types/<route>): {'route', 'title', 'when', 'start', 'minutes',
    'frames': [{'key', 'image', 'road_type', 'basis', 'sure', 'reviewed'}] in the order of the drive, 'stretches',
    'types', 'reviewed', 'corrections': how many, 'junctions': the intersections of the drive, the map's with a person's
    corrections over them (junctions.drive_marks: {'found', 'corrections', 'marks'})}; frames [] before nnslr road-types
    ran for it. by_key: the rows of the drive's pre-annotation report by frame key, whose preview_path names the picture
    of the frame on the site (/route/<route>/images/<8 digits>.jpg)."""
    from nnslr_tools.drives import drive_of
    from nnslr_tools.junctions import drive_marks
    lines = map_road_types(root, route)
    corrections = load_corrections(root, route) if lines else []
    lines = corrected(lines, corrections)
    frames = []
    for line in lines:
        preview = (by_key.get(line['frame_key']) or {}).get('preview_path')
        frames.append({'key': line['frame_key'],
                       'image': f'/route/{route}/{preview}' if isinstance(preview, str) and PREVIEW.fullmatch(preview)
                       else None,
                       'road_type': line.get('road_type'), 'basis': line.get('basis'), 'sure': bool(line.get('sure')),
                       'reviewed': bool(line.get('reviewed'))})
    drive = drive_of(root, route) or {}
    return {'route': route, 'title': drive.get('title'), 'when': drive.get('when'), 'start': drive.get('start'),
            'minutes': drive.get('minutes'), 'frames': frames, 'stretches': stretches(lines), 'types': type_counts(lines),
            'reviewed': sum(1 for frame in frames if frame['reviewed']), 'corrections': len(corrections),
            'junctions': drive_marks(root, route, [frame['key'] for frame in frames])}


def _stamp(path: Path) -> tuple | None:
    """When a corrections file changed, None without one: each correction rewrites it (a new inode, a line more or
    less), maybe within one tick of the clock."""
    try:
        stat = path.stat()
    except OSError:
        return None
    return stat.st_mtime_ns, stat.st_size, stat.st_ino


def summary(root: Path) -> dict:
    """The road types of every drive, for Data › Roads (GET /api/panel/road-types): {'routes': [{'route', 'title',
    'when', 'minutes', 'frames', 'types': {town, rural, expressway, motorway, ramp, parking, unknown: frames}, 'reviewed':
    frames a person set, 'stretches'}] newest drive first, 'made_at': when the newest file of the map was written (None
    before the first), 'drives_without': the prepared drives (with a pre-annotation report) that have no road types
    yet}."""
    from nnslr_tools.review import list_reports
    root = Path(root)
    files = []
    for path in sorted((root / ROAD_TYPES_RELATIVE).glob('*.jsonl')):
        try:
            parse_route_id(path.stem)
            files.append((path.stem, path.stat().st_mtime_ns, _stamp(root / CORRECTIONS_RELATIVE / path.name)))
        except (ValueError, OSError):
            continue
    key = tuple(files)
    cached = _DRIVES.get(str(root))
    if cached is not None and cached[0] == key:
        drives = cached[1]
    else:
        drives = sorted((_drive(root, route) for route, _, _ in files),
                        key=lambda item: (item['start'] or '', item['route']), reverse=True)
        _DRIVES[str(root)] = (key, drives)
    newest = max((stamp for _, stamp, _ in files), default=None)
    done = {route for route, _, _ in files}
    return {'routes': [{name: value for name, value in drive.items() if name != 'start'} for drive in drives],
            'made_at': datetime.fromtimestamp(newest / 1e9).astimezone().isoformat(timespec='seconds')
            if newest is not None else None,
            'drives_without': sum(1 for route, _ in list_reports(root) if route not in done)}
