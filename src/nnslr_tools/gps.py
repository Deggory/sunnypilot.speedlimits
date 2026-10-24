"""GPS position and country of the frames of a route, from the comma logs.

``nnslr gps --route <id>`` (or ``--all``) reads ``raw/routes/<route>/<segment>/rlog.zst`` (or the qlog) and
writes ``derived/gps/<route>/positions.jsonl``: one line per frame of the route's current pre-annotation
report, ``{"frame_key", "segment_index", "output_index", "latitude", "longitude", "speed_mps", "bearing_deg",
"country", "country_from"}``, latitude to bearing null when the log has no fix within GPS_MAX_GAP_S of
the frame; the country then comes from the nearest frame with a fix (country_from "nearest_fix", else "gps").

The frame k of a segment is the first video frame at k seconds (1 frame a second, nnslr_tools.media), so
its time is the time of the segment's first encoded road frame plus k seconds; the nearest fix
(``gpsLocationExternal``, else ``gpsLocation``) gives its position. The logs are read by the Python of the
comma fetch (openpilot's LogReader, or pycapnp with NNSLR_OPENPILOT_ROOT: helpers/comma_fetch_log.py).

The country comes from the borders in ``data/countries.json.gz`` (Natural Earth, public domain;
scripts/make_country_borders.py), Europe only: elsewhere it is None. It is a proposal: the reviewer
confirms or corrects it with each frame (``country`` of the review).
"""
from __future__ import annotations

import bisect
import gzip
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

from nnslr_tools.annotations import atomic_write, jsonl_bytes, load_jsonl
from nnslr_tools.manifest import parse_route_id

GPS_RELATIVE_DIR = Path('derived') / 'gps'
GPS_MAX_GAP_S = 2.0
BORDERS = Path(__file__).resolve().parent / 'data' / 'countries.json.gz'
FETCH_PYTHON_ENV = 'COMMA_FETCH_PYTHON'
HELPERS = Path(__file__).resolve().parents[2] / 'helpers'

# Run by the log Python: per segment log, the first encoded road frame and the GPS fixes, as one JSON line.
_CHILD = r"""
import json, sys
sys.path.insert(0, sys.argv[1])
import comma_fetch_log
from pathlib import Path
ENCODE = ('roadEncodeIdx', 'narrowRoadEncodeIdx')
GPS = ('gpsLocationExternal', 'gpsLocation')
for name in sys.argv[2:]:
    first = None
    fixes = {service: [] for service in GPS}
    try:
        for message in comma_fetch_log._messages(Path(name)):
            which = message.which()
            if which in ENCODE:
                payload = getattr(message, which)
                if int(payload.segmentId) == 0 and first is None:
                    first = int(message.logMonoTime)
            elif which in GPS:
                g = getattr(message, which)
                if getattr(g, 'hasFix', True) and (g.latitude or g.longitude):
                    fixes[which].append([int(message.logMonoTime), float(g.latitude), float(g.longitude),
                                         float(g.speed), float(g.bearingDeg)])
        error = None
    except Exception as exc:
        error = type(exc).__name__ + ': ' + str(exc)[:200]
    chosen = fixes['gpsLocationExternal'] or fixes['gpsLocation']
    print(json.dumps({'log': name, 'first_frame_mono': first, 'fixes': chosen, 'error': error}), flush=True)
"""


class GpsError(RuntimeError):
    pass


def log_python() -> str:
    """The Python that reads the comma logs: COMMA_FETCH_PYTHON, else the .venv of openpilot.distill next to
    this checkout, else this one (with the [log-reader] extra and NNSLR_OPENPILOT_ROOT)."""
    if os.environ.get(FETCH_PYTHON_ENV):
        return os.environ[FETCH_PYTHON_ENV]
    distill = Path(__file__).resolve().parents[3] / 'openpilot.distill/.venv/bin/python'
    if os.access(distill, os.X_OK):
        return str(distill)
    return sys.executable


def segment_log(root: Path, route: str, segment: int) -> Path | None:
    directory = Path(root) / 'raw' / 'routes' / route / str(segment)
    for name in ('rlog.zst', 'rlog.bz2', 'rlog', 'qlog.zst', 'qlog.bz2', 'qlog'):
        if (directory / name).is_file():
            return directory / name
    return None


def read_logs(paths: list[Path], *, python: str | None = None) -> dict[str, dict]:
    """{log path: {'first_frame_mono', 'fixes': [[mono, lat, lon, speed, bearing], ...], 'error'}}."""
    if not paths:
        return {}
    proc = subprocess.run([python or log_python(), '-c', _CHILD, str(HELPERS)] + [str(p) for p in paths],
                          capture_output=True, text=True, check=False, timeout=600)
    if proc.returncode != 0:
        raise GpsError('log reader failed: ' + proc.stderr.strip()[-300:])
    result = {}
    for line in proc.stdout.splitlines():
        if line.strip():
            item = json.loads(line)
            result[item['log']] = item
    return result


def frame_position(segment_log_data: dict | None, output_index: int) -> dict | None:
    """The fix nearest to frame ``output_index`` of the segment, within GPS_MAX_GAP_S, or None."""
    if not segment_log_data or segment_log_data.get('first_frame_mono') is None or not segment_log_data['fixes']:
        return None
    fixes = segment_log_data['fixes']
    when = segment_log_data['first_frame_mono'] + int(output_index * 1e9)
    times = [fix[0] for fix in fixes]
    at = bisect.bisect_left(times, when)
    best = None
    for index in (at - 1, at):
        if 0 <= index < len(fixes) and (best is None or abs(fixes[index][0] - when) < abs(best[0] - when)):
            best = fixes[index]
    if best is None or abs(best[0] - when) > GPS_MAX_GAP_S * 1e9:
        return None
    return {'latitude': round(best[1], 6), 'longitude': round(best[2], 6), 'speed_mps': round(best[3], 2),
            'bearing_deg': round(best[4], 1)}


@lru_cache(maxsize=1)
def _borders() -> list[dict]:
    return json.loads(gzip.decompress(BORDERS.read_bytes()))['countries']


def _inside(lon: float, lat: float, ring: list) -> bool:
    inside = False
    j = len(ring) - 1
    for i in range(len(ring)):
        xi, yi = ring[i]
        xj, yj = ring[j]
        if (yi > lat) != (yj > lat) and lon < (xj - xi) * (lat - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def country_of(latitude: float | None, longitude: float | None) -> str | None:
    """ISO 3166-1 alpha-2 code of the country at a position ("CH", "IT", "PL"...), None outside the borders
    shipped (Europe) or without a position."""
    if latitude is None or longitude is None:
        return None
    for country in _borders():
        west, south, east, north = country['bbox']
        if not (west <= longitude <= east and south <= latitude <= north):
            continue
        for polygon in country['polygons']:
            if _inside(longitude, latitude, polygon[0]) and not any(_inside(longitude, latitude, hole)
                                                                      for hole in polygon[1:]):
                return country['code']
    return None


def country_names() -> dict[str, str]:
    """{code: English name} of the countries the borders know."""
    return {country['code']: country['name'] for country in _borders()}


def positions_path(root: Path, route: str) -> Path:
    parse_route_id(route)
    return Path(root) / GPS_RELATIVE_DIR / route / 'positions.jsonl'


def build_positions(root: Path, route: str, rows: list[dict], *, python: str | None = None,
                    workers: int = 8) -> dict:
    """Write positions.jsonl for the report ``rows`` of a route; returns counts."""
    segments = sorted({row['segment_index'] for row in rows})
    logs = {segment: segment_log(root, route, segment) for segment in segments}
    found = [path for path in logs.values() if path is not None]
    groups = [found[i::workers] for i in range(workers) if found[i::workers]]
    data = {}
    with ThreadPoolExecutor(max_workers=max(1, len(groups))) as pool:
        for part in pool.map(lambda group: read_logs(group, python=python), groups):
            data.update(part)
    lines = []
    located = 0
    countries: dict[str, int] = {}
    errors = []
    for row in sorted(rows, key=lambda item: (item['segment_index'], item['output_index'])):
        path = logs.get(row['segment_index'])
        segment_data = data.get(str(path)) if path is not None else None
        if segment_data and segment_data.get('error'):
            errors.append(f"segment {row['segment_index']}: {segment_data['error']}")
        position = frame_position(segment_data, row['output_index'])
        country = country_of(position['latitude'], position['longitude']) if position else None
        line = {'frame_key': row['frame_key'], 'segment_index': row['segment_index'],
                'output_index': row['output_index'], 'latitude': None, 'longitude': None, 'speed_mps': None,
                'bearing_deg': None, 'country': country, 'country_from': 'gps' if country else None}
        if position:
            line.update(position)
            located = located + 1
        if country:
            countries[country] = countries.get(country, 0) + 1
        lines.append(line)
    # A frame without a fix (the GPS still searching, a tunnel) takes the country of the nearest frame that has
    # one: a route rarely crosses a border, and the reviewer sees where the proposal comes from.
    located_at = [index for index, line in enumerate(lines) if line['country']]
    for index, line in enumerate(lines):
        if line['country'] is None and located_at:
            nearest = min(located_at, key=lambda other: abs(other - index))
            line['country'] = lines[nearest]['country']
            line['country_from'] = 'nearest_fix'
    target = positions_path(root, route)
    atomic_write(target, jsonl_bytes(lines))
    return {'route_id': route, 'frames': len(lines), 'located': located, 'countries': countries,
            'segments_without_log': [segment for segment, path in logs.items() if path is None],
            'errors': sorted(set(errors)), 'output': str(target)}


def load_positions(root: Path, route: str) -> dict:
    """{frame_key: position line} of a route, {} when ``nnslr gps`` has not run for it."""
    path = positions_path(root, route)
    if not path.is_file():
        return {}
    return {line['frame_key']: line for line in load_jsonl(path) if isinstance(line, dict) and 'frame_key' in line}
