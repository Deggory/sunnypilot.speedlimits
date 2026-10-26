"""A drive as a video the browser can play, for the Drive view of Results (web/src/control/DriveView.vue).

The comma records the road camera of every minute as raw HEVC (data/raw/routes/<route>/<segment>/fcamera.hevc,
1344 x 760, 20 frames a second), which browsers do not play. Here the minutes of a drive, in order, become one H.264
MP4 960 px wide (ffmpeg, low priority and a few threads: it never slows a training), with an index of where each minute
starts in it: derived/video/<route>.mp4 and <route>.json. A frame of the reports is image N of a minute, N seconds into
it (frames are sampled one a second), so the page finds it in the video at start(minute) + N.

Made once, the first time the Drive view shows the drive (Panel.drive_video), or with `nnslr drive-video ROUTE`; a
made video is kept until its minutes change. Nothing else reads it: it is only how the site shows a drive.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable

FPS = 20
WIDTH = 960
ROUTE = re.compile(r'[0-9a-f]{8}--[0-9a-f]{10}')
# One video made at a time, by this process.
_lock = threading.Lock()
_making: dict[str, dict[str, Any]] = {}


def paths(root: Path, route: str) -> tuple[Path, Path]:
    """The video of a drive and its index."""
    folder = Path(root) / 'derived' / 'video'
    return folder / (route + '.mp4'), folder / (route + '.json')


def minutes(root: Path, route: str) -> list[tuple[int, Path]]:
    """The minutes of a drive with a road video, in order: (segment, fcamera.hevc)."""
    base = Path(root) / 'raw' / 'routes' / route
    found = []
    for folder in base.glob('*'):
        video = folder / 'fcamera.hevc'
        if folder.name.isdigit() and video.is_file():
            found.append((int(folder.name), video))
    return sorted(found)


def _frames(video: Path) -> int:
    """How many frames a raw HEVC minute holds (its packets: nothing is decoded)."""
    answer = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_packets', '-show_entries',
                             'stream=nb_read_packets', '-of', 'csv=p=0', str(video)],
                            capture_output=True, text=True, check=True, timeout=120)
    return int(answer.stdout.strip().split(',')[0] or 0)


def _signature(parts: list[tuple[int, Path]]) -> list[list]:
    """What the video is made of: each minute with its size and time, to know when it must be made again."""
    signature = []
    for segment, video in parts:
        info = video.stat()
        signature.append([segment, info.st_size, int(info.st_mtime)])
    return signature


def index(root: Path, route: str) -> dict | None:
    """The index of a made video, still matching its minutes; None when there is none or it is out of date."""
    video, index_path = paths(root, route)
    try:
        value = json.loads(index_path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None
    if not video.is_file() or value.get('source') != _signature(minutes(root, route)):
        return None
    return value


def make(root: Path, route: str, progress: Callable[[float], None] | None = None) -> dict:
    """Makes the video of a drive and its index, and returns the index. progress(0-1) while ffmpeg works."""
    if not ROUTE.fullmatch(route):
        raise ValueError('invalid route id: ' + route)
    parts = minutes(root, route)
    if not parts:
        raise ValueError('no road video of this drive on the server')
    video, index_path = paths(root, route)
    video.parent.mkdir(parents=True, exist_ok=True)
    segments, start = [], 0.0
    for segment, path in parts:
        count = _frames(path)
        segments.append({'segment': segment, 'start': round(start, 3), 'frames': count})
        start += count / FPS
    total = sum(item['frames'] for item in segments)
    # Its own partial file: the site and `nnslr drive-video` may make the same drive at once; the last one wins whole.
    partial = video.with_name(f'{route}.{os.getpid()}.{threading.get_ident()}.part.mp4')
    command = ['nice', '-n', '15', 'ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostats', '-progress', 'pipe:2',
               '-y', '-f', 'hevc', '-r', str(FPS), '-i', 'pipe:0', '-an', '-vf', f'scale={WIDTH}:-2',
               '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '27', '-g', str(2 * FPS), '-pix_fmt', 'yuv420p',
               '-threads', '8', '-movflags', '+faststart', str(partial)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE, text=False)

    def feed() -> None:
        # The minutes one after the other: each starts with its own parameter sets, so they join as one stream.
        try:
            for _, path in parts:
                with path.open('rb') as source:
                    while block := source.read(1 << 20):
                        process.stdin.write(block)  # type: ignore[union-attr]
        except (BrokenPipeError, OSError):
            pass
        finally:
            try:
                process.stdin.close()  # type: ignore[union-attr]
            except OSError:
                pass

    feeder = threading.Thread(target=feed, daemon=True)
    feeder.start()
    errors = []
    for line in process.stderr:  # type: ignore[union-attr]
        text = line.decode('utf-8', 'replace').strip()
        if text.startswith('frame=') and progress is not None and total:
            progress(min(1.0, int(text[6:] or 0) / total))
        elif text and '=' not in text:
            errors.append(text)
    feeder.join()
    if process.wait() != 0:
        partial.unlink(missing_ok=True)
        raise RuntimeError('ffmpeg failed: ' + ' '.join(errors[-3:]))
    os.replace(partial, video)
    value = {'route': route, 'fps': FPS, 'width': WIDTH, 'segments': segments, 'duration': round(start, 3),
             'source': _signature(parts), 'made_at': time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    index_path.write_text(json.dumps(value, indent=1) + '\n', encoding='utf-8')
    return value


def state(root: Path, route: str, *, start: bool = False) -> dict:
    """Where the video of a drive is: ready (with its index), being made (how far), failed (why), or missing; start:
    begin making it in the background when it is missing (one at a time: another waits its turn)."""
    if not ROUTE.fullmatch(route):
        return {'state': 'missing', 'reason': 'invalid route id'}
    made = index(root, route)
    if made is not None:
        return {'state': 'ready', 'url': '/route/' + route + '/video.mp4', 'index': made}
    with _lock:
        work = _making.get(route)
        if work is not None and work['state'] in ('waiting', 'making'):
            return {'state': work['state'], 'progress': work['progress']}
        if not minutes(root, route):
            return {'state': 'missing', 'reason': 'no road video of this drive on the server'}
        if not start:
            return {'state': 'failed', 'reason': work['reason']} if work else {'state': 'missing'}
        work = {'state': 'waiting', 'progress': 0.0, 'reason': ''}
        _making[route] = work
    threading.Thread(target=_run, args=(Path(root), route, work), daemon=True).start()
    return {'state': 'waiting', 'progress': 0.0}


_turn = threading.Lock()


def _run(root: Path, route: str, work: dict) -> None:
    with _turn:
        work['state'] = 'making'
        try:
            make(root, route, lambda share: work.__setitem__('progress', round(share, 3)))
            work['state'] = 'ready'
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
            work.update(state='failed', reason=str(exc))
