"""The motorway signs the comma confirms on the drives of the road types: an input of the road-type network (how long
ago a motorway or expressway sign was confirmed), derived offline the way the car derives it.

On the car, speedvisiond reads the road camera every PERIOD (0.2 s) with SignNet and the classifier
(vision.read_signs), keeps the signs read sure enough (vision.MIN_CONFIDENCE) and gives their classes to
vision.SpeedLimitMemory. That confirms a sign seen in CONFIRM_FRAMES frames within CONFIRM_SECONDS and notes when a
motorway or expressway sign was confirmed (_road_sign_at). The road-type network gets the seconds since then
(sign_features). In Switzerland the motorway sign (4.01) stands where the entry ramp starts, so it is seen as the ramp
begins, and the network can learn that the carriageway comes after that ramp.

Here the car's own code does the same on the recorded drives: the car's vision.py and its two networks from the
sunnypilot clone (read only: openpilot/sunnypilot/speed_limits and its models), on each segment's fcamera.hevc
decoded to NV12 by ffmpeg (the road camera's pixel format), one decoded frame in VIDEO_STEP (20 fps: 5 a second, as
speedvisiond). The reviewed sign labels are not used: most drives were reviewed before the other signs had boxes, and
the car never has them; what the car's networks miss or misread, the road-type network meets the same way.

Per drive, data/derived/road_type_signs/<route>.json: the fingerprint of the code and networks used, the frames read
per segment, every frame with a sign read sure enough (time in seconds from the start of the drive, class, confidence)
and the road signs SpeedLimitMemory confirms (time, road). Built again when the vision code, a network or the video
changed. numpy and onnxruntime (CPU) only.

Two sources of the sign times (SIGN_SOURCES):
- 'car': the signs the car's code confirms, as above. On the first 12 drives (October 2026) the car's classifier
  confirmed no motorway sign at all: its motorway_start probability stays below 0.03 on the boxes of the reviewed
  4.01 signs, so today the car gives the network nothing.
- 'map': where the sign stands, from the map's road types: the first second of every ramp that leads from another road
  onto the motorway or expressway (road_type_metrics.ramp_entries, not from the motorway itself: an interchange has no
  4.01). The reviewed 4.01 signs were seen from 7 s before to 6 s after that second; training moves the time by up to
  SIGN_JITTER seconds (road_type_net). This is the input the network gets once the car reads the sign.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

SIGNS_RELATIVE = Path('derived') / 'road_type_signs'
RAW_RELATIVE = Path('raw') / 'routes'
PACKAGE_RELATIVE = Path('openpilot') / 'sunnypilot' / 'speed_limits'
SUNNYPILOT_ENV, SUNNYPILOT_DEFAULT = 'NNSLR_SUNNYPILOT_CLONE', '/root/sunnypilot'
NETWORK_FILES = {'finder': 'signnet.onnx', 'classifier': 'classifier.onnx'}
NOT_A_SIGN = 'not_a_sign'  # read sure enough, but neither a speed nor a road sign: left out of the cache
CACHE_VERSION = 1
VIDEO_FPS = 20
VIDEO_STEP = 4  # 20 fps / 4: the 5 frames a second of speedvisiond (PERIOD 0.2 s)
SEGMENT_SECONDS = 60
# The roads whose confirmed sign the network gets (vision.SpeedLimitMemory sets _road_sign_at for these).
SIGN_ROADS = ('motorway', 'expressway')
# The input: [min(age, SIGN_HORIZON) / SIGN_SCALE, 1 when a sign was confirmed within SIGN_HORIZON seconds]. Never, or
# longer ago: NO_SIGN.
SIGN_HORIZON = 120.0
SIGN_SCALE = 60.0
NO_SIGN = (SIGN_HORIZON / SIGN_SCALE, 0.0)
SIGN_SOURCES = ('map', 'car')
DEFAULT_SOURCE = 'map'


def sign_features(age: float | None) -> tuple[float, float]:
    """The network's sign input for a sign confirmed age seconds ago (None: never)."""
    if age is None or not np.isfinite(age) or age < 0 or age > SIGN_HORIZON:
        return NO_SIGN
    return (float(age) / SIGN_SCALE, 1.0)


def package_path(sunnypilot: Path | None = None) -> Path:
    """The car's speed_limits package in the sunnypilot clone (NNSLR_SUNNYPILOT_CLONE, else /root/sunnypilot)."""
    base = Path(sunnypilot) if sunnypilot else Path(os.environ.get(SUNNYPILOT_ENV) or SUNNYPILOT_DEFAULT)
    return base / PACKAGE_RELATIVE


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(package: Path) -> dict:
    """What the readings depend on: the car's vision.py and its two networks (sha256), the rate."""
    package = Path(package)
    return {'version': CACHE_VERSION, 'video_step': VIDEO_STEP,
            'vision': _sha256(package / 'vision.py'),
            **{name: _sha256(package / 'models' / file) for name, file in NETWORK_FILES.items()}}


def load_vision(package: Path):
    from nnslr_tools.comma_trial import load_vision as load
    return load(Path(package) / 'vision.py')


class _Network:
    """One of the car's networks under onnxruntime on one CPU thread: numpy in, numpy out."""

    def __init__(self, path: Path):
        import onnxruntime
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        self.session = onnxruntime.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])
        self.name = self.session.get_inputs()[0].name

    def __call__(self, x):
        return self.session.run(None, {self.name: np.asarray(x, dtype=np.float32)})[0]


_WORKER: dict = {}


def _worker_start(package: str) -> None:
    package = Path(package)
    vision = load_vision(package)
    contract = json.loads((package / 'models' / (NETWORK_FILES['classifier'] + '.contract.json')).read_text('utf-8'))
    _WORKER.update(vision=vision, finder=_Network(package / 'models' / NETWORK_FILES['finder']),
                   classifier=_Network(package / 'models' / NETWORK_FILES['classifier']),
                   classes=contract['output']['classes'])


def _video_size(video: Path) -> tuple[int, int]:
    output = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                             'stream=width,height', '-of', 'csv=p=0', str(video)],
                            capture_output=True, text=True, check=True).stdout.strip()
    width, height = (int(value) for value in output.split(',')[:2])
    return width, height


def read_segment(task: tuple[str, int]) -> dict:
    """The car's read_signs on one segment's road video, one frame in VIDEO_STEP: {'segment', 'frames' (read), 'signs':
    [[seconds into the segment, [[class, confidence], ...]], ...] (the frames with a sign read sure enough)}."""
    video, segment = task
    vision, finder, classifier, classes = (_WORKER[name] for name in ('vision', 'finder', 'classifier', 'classes'))
    width, height = _video_size(Path(video))
    size = width * height * 3 // 2
    command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-threads', '1', '-i', str(video), '-vf',
               f'select=not(mod(n\\,{VIDEO_STEP}))', '-vsync', '0', '-pix_fmt', 'nv12', '-f', 'rawvideo', '-']
    signnet_input = vision.SignNetInput(width, height)
    found, count = [], 0
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL) as process:
        while True:
            buffer = process.stdout.read(size)
            if len(buffer) < size:
                break
            data = np.frombuffer(buffer, dtype=np.uint8)
            y, uv = vision.nv12_planes(data, width, height, width, width * height)
            signs = vision.read_signs(y, uv, signnet_input, finder, classifier, classes)
            sure = [[sign['class'], round(sign['confidence'], 4)] for sign in signs
                    if sign['confidence'] >= vision.MIN_CONFIDENCE and sign['class'] != NOT_A_SIGN]
            if sure:
                found.append([round(count * VIDEO_STEP / VIDEO_FPS, 3), sure])
            count += 1
    return {'segment': segment, 'frames': count, 'signs': found}


def confirmations(vision, readings: list[tuple[float, list]]) -> list[list]:
    """The motorway and expressway signs SpeedLimitMemory confirms from the readings [(seconds, [[class, confidence],
    ...])] in time order: [[seconds, road], ...], the moments it sets _road_sign_at, as speedvisiond's memory does. A
    frame without a sign changes nothing there, so only the frames with one are given."""
    memory, found = vision.SpeedLimitMemory(), []
    for now, signs in readings:
        classes = [name for name, _ in signs]
        speeds = [vision.class_speed(name) for name in classes]
        before = memory._road_sign_at
        memory.update(float(now), [speed for speed in speeds if speed is not None], classes)
        if memory._road_sign_at != before:
            found.append([float(now), memory.road])
    return found


def number_confirmations(vision, readings: list[tuple[float, list]]) -> list[list]:
    """The speed signs SpeedLimitMemory confirms from the readings, as speedvisiond's memory does: [[seconds, km/h],
    ...], the moments a number (not the end of a limit) became the limit held. The junction head's measure counts the
    intersections a held number would end at (Poland: a sign holds to the next intersection)."""
    memory, found, last = vision.SpeedLimitMemory(), [], None
    for now, signs in readings:
        classes = [name for name, _ in signs]
        speeds = [speed for speed in (vision.class_speed(name) for name in classes) if speed is not None]
        memory.update(float(now), speeds, classes)
        if memory.sign_read and memory.speed_limit > 0 and float(memory.speed_limit) in [float(v) for v in speeds]:
            # One sign read for some frames is one confirmation: a new one after a pause or with another number.
            if last is None or last[1] != memory.speed_limit or float(now) - last[0] > 2.0:
                found.append([float(now), float(memory.speed_limit)])
            last = (float(now), float(memory.speed_limit))
    return found


def cache_path(root: Path, route: str) -> Path:
    return Path(root) / SIGNS_RELATIVE / f'{route}.json'


def _videos(root: Path, route: str) -> dict[int, Path]:
    folder = Path(root) / RAW_RELATIVE / route
    videos = {}
    for path in folder.glob('*/fcamera.hevc') if folder.is_dir() else []:
        if path.parent.name.isdigit():
            videos[int(path.parent.name)] = path
    return dict(sorted(videos.items()))


def _video_stamp(videos: dict[int, Path]) -> list:
    return [[segment, path.stat().st_size, int(path.stat().st_mtime)] for segment, path in videos.items()]


def load_signs(root: Path, route: str) -> dict | None:
    path = cache_path(root, route)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding='utf-8'))


def build_signs(root: Path, routes: list[str], *, package: Path | None = None, workers: int | None = None,
                log=print) -> dict[str, dict]:
    """The readings of every drive (cached; read again when the code, a network or a video changed): {route: record}.
    The segments of all drives are read in parallel, one process per core."""
    package = package_path() if package is None else Path(package)
    stamp = fingerprint(package)
    records, tasks, stamps = {}, [], {}
    for route in routes:
        videos = _videos(root, route)
        stamps[route] = _video_stamp(videos)
        record = load_signs(root, route)
        if record and record.get('fingerprint') == stamp and record.get('videos') == stamps[route]:
            records[route] = record
            continue
        tasks += [(route, segment, str(path)) for segment, path in videos.items()]
    if tasks:
        workers = workers or max(1, min(len(tasks), (os.cpu_count() or 4) // 4))
        log(f'road-type signs: reading {len(tasks)} segments of {len({t[0] for t in tasks})} drives with the car\'s '
            f'SignNet and classifier, 5 frames a second, {workers} processes')
        results: dict[str, list] = {}
        with ProcessPoolExecutor(max_workers=workers, initializer=_worker_start, initargs=(str(package),)) as pool:
            done = 0
            for (route, _, _), result in zip(tasks, pool.map(read_segment, [(v, s) for _, s, v in tasks])):
                results.setdefault(route, []).append(result)
                done += 1
                if done % 25 == 0 or done == len(tasks):
                    log(f'road-type signs: {done}/{len(tasks)} segments')
        vision = load_vision(package)
        for route, segments in results.items():
            segments.sort(key=lambda item: item['segment'])
            readings = [(item['segment'] * SEGMENT_SECONDS + seconds, signs)
                        for item in segments for seconds, signs in item['signs']]
            record = {'route': route, 'fingerprint': stamp, 'videos': stamps[route],
                      'frames': {str(item['segment']): item['frames'] for item in segments},
                      'readings': [[round(t, 3), signs] for t, signs in readings],
                      'confirmed': confirmations(vision, readings)}
            path = cache_path(root, route)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(record, separators=(',', ':')), encoding='utf-8')
            temporary.replace(path)
            records[route] = record
    return records


def sign_times(record: dict | None) -> list[float]:
    """When a motorway or expressway sign was confirmed in a drive (seconds from its start, in order)."""
    if not record:
        return []
    return sorted(float(t) for t, road in record.get('confirmed', []) if road in SIGN_ROADS)


def frame_seconds(frame: dict) -> float:
    """The time of a 1 Hz frame of the dataset in the drive (its segment's minute and its index)."""
    return float(frame['segment'] * SEGMENT_SECONDS + frame['index'])


def sign_ages(seconds, times) -> np.ndarray:
    """Seconds since the last sign at or before each moment (inf before the first, or without any)."""
    seconds = np.asarray(seconds, dtype=np.float64)
    times = np.sort(np.asarray(times, dtype=np.float64))
    if not len(times):
        return np.full(len(seconds), np.inf)
    last = np.searchsorted(times, seconds, side='right') - 1
    return np.where(last >= 0, seconds - times[np.maximum(last, 0)], np.inf)


def features_at(seconds, times) -> np.ndarray:
    """The sign input (len(seconds), 2) float32 at each moment, sign_features of sign_ages."""
    ages = sign_ages(seconds, times)
    out = np.tile(np.asarray(NO_SIGN, dtype=np.float32), (len(ages), 1))
    seen = (ages >= 0) & (ages <= SIGN_HORIZON)
    out[seen, 0] = ages[seen] / SIGN_SCALE
    out[seen, 1] = 1.0
    return out


def frame_sign_features(frames: list[dict], times: list[float]) -> np.ndarray:
    """The sign input of each frame of a drive (frames, 2): the age of the last sign confirmed at or before it."""
    return features_at([frame_seconds(frame) for frame in frames], times)


def map_sign_times(frames: list[dict], classes: tuple[str, ...]) -> list[float]:
    """Where the map puts the motorway sign in a drive (seconds from its start): the first second of every ramp from
    another road onto the motorway or expressway (frames: the drive's 1 Hz frames with their 'label')."""
    from nnslr_tools.road_type_metrics import ramp_entries
    labels = np.array([frame['label'] for frame in frames], dtype=np.int64)
    return [frame_seconds(frames[entry['ramp']]) for entry in ramp_entries(labels, classes)
            if entry['before'] != 'motorway']
