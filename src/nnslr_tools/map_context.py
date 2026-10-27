"""Other roads next to ours, from OpenStreetMap: the frames where a speed sign may be meant for another road.

``nnslr map-candidates --all`` reads ``derived/osm/roads.jsonl`` (scripts/osm_roads.py: the OSM roads near the
drives), the GPS position and heading of every frame (``nnslr gps``) and the boxes of every frame (the confirmed
review, else the model proposals). For each frame with a speed sign it finds the road we drive on (the closest
one in our direction) and the other roads near it:

- a ramp (``*_link``) within RAMP_M;
- a parallel road or the other carriageway (same direction or opposite) within PARALLEL_M;
- a side road joining or crossing ahead within SIDE_M.

When one of them is on the side where the sign stands (left or right half of the frame; a sign over the road
counts for both), the frame is a candidate: a person checks whether the sign is for us ("Ours") or for that road
("Another"). It writes ``derived/osm/candidates/<route>.jsonl``, one line per candidate frame with its reasons.

The map only proposes frames to look at; it never labels anything. OpenStreetMap data © OpenStreetMap
contributors (ODbL).

Measured on 24 Oct 2026 against the confirmed speed signs (126 "Another", 1385 "Ours"): it separates them
poorly. A side road within 20 m on the sign's side is the best signal (13% of "Another" signs against 4% of
"Ours"); a ramp says little (17% against 14%); the opposite carriageway flags half of the "Ours" signs on
motorways. So it is not a review queue: it is kept as a possible input of the decider, and the frames to
check come from the decider's own uncertainty instead.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from nnslr_tools.annotations import atomic_write, jsonl_bytes, load_jsonl
from nnslr_tools.manifest import parse_route_id

ROADS_RELATIVE = Path('derived') / 'osm' / 'roads.jsonl'
CANDIDATES_RELATIVE = Path('derived') / 'osm' / 'candidates'
CELL_DEGREES = 0.003
SPEED_FAMILIES = ('maximum_speed', 'zone', 'variable_display', 'cancellation')
# How close the car must be to a road for it to be ours, and the angle allowed with our heading.
OUR_ROAD_M = 25.0
OUR_ROAD_ANGLE = 35.0
RAMP_M = 45.0
PARALLEL_M = 45.0
PARALLEL_ANGLE = 25.0
SIDE_M = 70.0
# A sign whose centre is within this share of the frame's middle stands over the road: both sides.
MIDDLE = 0.1


def _xy(lat: float, lon: float, lat0: float, lon0: float) -> tuple[float, float]:
    """Metres east and north of (lat0, lon0)."""
    return (lon - lon0) * 111320.0 * math.cos(math.radians(lat0)), (lat - lat0) * 110540.0


def _bearing(dx: float, dy: float) -> float:
    return math.degrees(math.atan2(dx, dy)) % 360.0


def _angle(a: float, b: float) -> float:
    """The angle between two bearings, 0-180."""
    d = abs(a - b) % 360.0
    return 360.0 - d if d > 180.0 else d


class RoadMap:
    """The OSM roads near the drives, with a grid index of their segments."""

    def __init__(self, ways: list[dict]):
        self.ways = ways
        self.cells: dict[tuple[int, int], list[tuple[int, int]]] = {}
        self.node_ways: dict[tuple[float, float], set[int]] = {}
        for w, way in enumerate(ways):
            nodes = way['nodes']
            for s in range(len(nodes) - 1):
                (a_lat, a_lon), (b_lat, b_lon) = nodes[s], nodes[s + 1]
                for cell in {self._cell(a_lat, a_lon), self._cell(b_lat, b_lon)}:
                    self.cells.setdefault(cell, []).append((w, s))
            for end in (nodes[0], nodes[-1]):
                self.node_ways.setdefault((end[0], end[1]), set()).add(w)

    @staticmethod
    def _cell(lat: float, lon: float) -> tuple[int, int]:
        return int(lat // CELL_DEGREES), int(lon // CELL_DEGREES)

    @classmethod
    def load(cls, root: Path) -> RoadMap | None:
        path = Path(root) / ROADS_RELATIVE
        if not path.is_file():
            return None
        return cls([way for way in load_jsonl(path) if isinstance(way, dict) and len(way.get('nodes') or []) >= 2])

    def nearest(self, lat: float, lon: float, reach_m: float) -> dict[int, dict]:
        """For every way with a segment within reach_m: distance, bearing of the segment and offset (x, y) of
        the closest point, in metres from the position."""
        a, b = self._cell(lat, lon)
        found: dict[int, dict] = {}
        for da in (-1, 0, 1):
            for db in (-1, 0, 1):
                for w, s in self.cells.get((a + da, b + db), ()):
                    (a_lat, a_lon), (b_lat, b_lon) = self.ways[w]['nodes'][s], self.ways[w]['nodes'][s + 1]
                    ax, ay = _xy(a_lat, a_lon, lat, lon)
                    bx, by = _xy(b_lat, b_lon, lat, lon)
                    dx, dy = bx - ax, by - ay
                    length2 = dx * dx + dy * dy
                    t = 0.0 if length2 == 0 else max(0.0, min(1.0, -(ax * dx + ay * dy) / length2))
                    px, py = ax + t * dx, ay + t * dy
                    distance = math.hypot(px, py)
                    if distance > reach_m:
                        continue
                    if w not in found or distance < found[w]['distance']:
                        found[w] = {'distance': distance, 'bearing': _bearing(dx, dy), 'offset': (px, py)}
        return found

    def touching(self, w: int) -> set[int]:
        """Ways sharing an end node with way w (its continuations and the roads joining it)."""
        nodes = self.ways[w]['nodes']
        touching = set()
        for end in (nodes[0], nodes[-1]):
            touching |= self.node_ways.get((end[0], end[1]), set())
        touching.discard(w)
        return touching


def _road_name(way: dict) -> str:
    kind = way.get('highway') or 'road'
    words = {'motorway': 'motorway', 'trunk': 'expressway', 'motorway_link': 'motorway ramp',
             'trunk_link': 'expressway ramp', 'primary_link': 'ramp', 'secondary_link': 'ramp',
             'tertiary_link': 'ramp', 'service': 'service road', 'residential': 'residential street'}
    word = words.get(kind, 'road')
    label = way.get('ref') or way.get('name')
    return word + (' ' + label if label else '')


def our_road(near: dict[int, dict], heading: float) -> int | None:
    """The way we drive on among ``near`` (RoadMap.nearest): the closest within OUR_ROAD_M along our heading."""
    ours = None
    for w, item in near.items():
        aligned = min(_angle(item['bearing'], heading), _angle(item['bearing'] + 180.0, heading))
        if item['distance'] <= OUR_ROAD_M and aligned <= OUR_ROAD_ANGLE:
            if ours is None or item['distance'] < near[ours]['distance']:
                ours = w
    return ours


def other_roads(road_map: RoadMap, lat: float, lon: float, heading: float) -> list[dict]:
    """The roads near us that a sign could be meant for: [{'kind', 'side', 'distance', 'text'}]. Empty when
    no road matches our position and heading."""
    near = road_map.nearest(lat, lon, max(RAMP_M, PARALLEL_M, SIDE_M))
    ours = our_road(near, heading)
    if ours is None:
        return []
    # Our road goes on in other OSM ways (split at bridges, junctions): those in our direction are still ours.
    continuations = {w for w in road_map.touching(ours) if w in near
                     and min(_angle(near[w]['bearing'], heading), _angle(near[w]['bearing'] + 180.0, heading))
                     <= OUR_ROAD_ANGLE}
    hx, hy = math.sin(math.radians(heading)), math.cos(math.radians(heading))
    found = []
    for w, item in near.items():
        if w == ours or w in continuations:
            continue
        way = road_map.ways[w]
        ox, oy = item['offset']
        side = 'left' if hx * oy - hy * ox > 0 else 'right'
        ahead = ox * hx + oy * hy
        relative = min(_angle(item['bearing'], heading), _angle(item['bearing'] + 180.0, heading))
        kind = None
        if way['highway'].endswith('_link') and item['distance'] <= RAMP_M:
            kind = 'ramp'
        elif relative <= PARALLEL_ANGLE and 6.0 <= item['distance'] <= PARALLEL_M:
            opposite = _angle(item['bearing'], heading) > 90.0 and way.get('oneway') in ('yes', '1')
            kind = 'other carriageway' if opposite else 'parallel road'
        elif relative > 30.0 and ahead > -10.0 and item['distance'] <= SIDE_M:
            kind = 'side road'
        if kind is None:
            continue
        found.append({'kind': kind, 'side': side, 'distance': round(item['distance']),
                      'text': f"{_road_name(way)} {round(item['distance'])} m to the {side}"})
    found.sort(key=lambda item: item['distance'])
    return found


def _sign_sides(box: list[float], width: float) -> set[str]:
    middle = (box[0] + box[2]) / 2 / width
    if abs(middle - 0.5) <= MIDDLE:
        return {'left', 'right'}
    return {'left'} if middle < 0.5 else {'right'}


def _speed_boxes(detections) -> list[dict]:
    return [d for d in (detections or []) if isinstance(d, dict) and d.get('family') in SPEED_FAMILIES
            and isinstance(d.get('bbox_xyxy'), list) and len(d['bbox_xyxy']) == 4]


def route_candidates(road_map: RoadMap, rows: list[dict], positions: dict, reviews: dict) -> list[dict]:
    """Candidate frames of one route: rows of its report, positions (gps.load_positions), confirmed reviews
    by frame key (their detections win over the proposals)."""
    found = []
    for row in rows:
        key = row.get('frame_key')
        position = positions.get(key)
        if not position or position.get('latitude') is None or position.get('bearing_deg') is None:
            continue
        review = reviews.get(key)
        boxes = _speed_boxes(review['payload'].get('detections') if review else row.get('detections'))
        if not boxes:
            continue
        roads = other_roads(road_map, position['latitude'], position['longitude'], position['bearing_deg'])
        if not roads:
            continue
        width = row.get('width') or 1344
        signs = []
        for box in boxes:
            sides = _sign_sides(box['bbox_xyxy'], width)
            matching = [road for road in roads if road['side'] in sides]
            if matching:
                signs.append({'detection_id': box.get('detection_id'), 'family': box.get('family'),
                              'value_kph': box.get('value_kph'), 'applicability': box.get('applicability'),
                              'roads': matching})
        if signs:
            first = signs[0]['roads'][0]
            found.append({'frame_key': key, 'confirmed': review is not None, 'signs': signs,
                          'text': 'Map: ' + first['text'] + ': is the sign for it?'})
    return found


def candidates_path(root: Path, route: str) -> Path:
    parse_route_id(route)
    return Path(root) / CANDIDATES_RELATIVE / f'{route}.jsonl'


def build_candidates(root: Path, route: str, rows: list[dict], positions: dict, reviews: dict,
                     road_map: RoadMap) -> dict:
    found = route_candidates(road_map, rows, positions, reviews)
    atomic_write(candidates_path(root, route), jsonl_bytes(found))
    labels: dict[str, int] = {}
    for item in found:
        for sign in item['signs']:
            label = sign.get('applicability') or 'unresolved'
            labels[label] = labels.get(label, 0) + 1
    return {'route_id': route, 'frames': len(found), 'signs_by_label': labels,
            'output': str(candidates_path(root, route))}


def load_candidates(root: Path, route: str) -> dict:
    """{frame_key: candidate} of a route, {} when nnslr map-candidates has not run for it."""
    path = candidates_path(root, route)
    if not path.is_file():
        return {}
    return {item['frame_key']: item for item in load_jsonl(path) if isinstance(item, dict) and 'frame_key' in item}
