"""A drive as a video the browser can play, for the Drive view of Results (web/src/control/DriveView.vue).

The comma records the road camera of every minute as raw HEVC (data/raw/routes/<route>/<segment>/fcamera.hevc,
1344 x 760, 20 frames a second), which browsers do not play. Here the minutes of a drive, in order, become one H.264
MP4 960 px wide (a drive recorded at full definition, 2688 x 1520, 1344 wide: a quarter of the bytes, so the player
seeks fast), a key frame every second, with an index of where each minute starts in it: derived/video/<route>.mp4 and <route>.json. A frame of
the reports is image N of a minute, N seconds into it (frames are sampled one a second), so the page finds it in the
video at start(minute) + N.

On the GPU first: one ffmpeg over the first minutes decodes, scales and encodes there, while the CPU decodes the last
minutes (CPU_SHARE of the frames, CPU_WORKERS ffmpeg side by side) and the GPU encodes them; the pieces are then joined
without encoding again. The GPU's video decoder is one chip, about 220 frames a second of a full definition drive, and the
CPU's 20 cores decode about as many, so together a drive takes under half the time (a 50 minute drive at full
definition in 109 s instead of about 280). A busy CPU (a training) leaves the whole drive to the GPU. The cards are tried in turn, the one with the most free
memory first (a training may hold one); when none can make it, or the server has none, the video is not made and the
player says why: it never goes on the CPU (AGENTS.md, #108). The site's watcher tries a failed drive again after
RETRY_FAILED seconds, and a page that opens the drive tries at once.

Made once, by the site as soon as a drive's minutes have arrived (watch: a recent drive whose minutes did not change
for a scan), else the first time the Drive view shows the drive (Panel.drive_video), or with `nnslr drive-video ROUTE`;
a made video is kept until its minutes change. Nothing else reads it: it is only how the site shows a drive.
"""
from __future__ import annotations

import functools
import json
import os
import re
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

from nnslr_tools.media import FULL_SIZE, video_dimensions

FPS = 20
WIDTH = 960
# A drive recorded at full definition (nnslr_tools.media.FULL_SIZE) is played at half its width, the comma's usual frame:
# a quarter of the bytes, so the player seeks fast. Its frames at full definition stay in Sign review.
FULL_WIDTH = FULL_SIZE[0]
HD_WIDTH = FULL_SIZE[0] // 2
# A key frame every second: a seek decodes at most a second of video.
KEY_FRAMES = FPS
# The share of the frames the CPU decodes beside the GPU, by CPU_WORKERS ffmpeg of CPU_THREADS threads; none when the
# CPU is busy (its load above half its cores) or the drive has too few minutes to split.
CPU_SHARE = 0.45
CPU_WORKERS = 8
CPU_THREADS = 2
# The site makes the video of a drive whose minutes arrived in the last WATCH_DAYS, checking every WATCH_PERIOD seconds.
WATCH_DAYS = 14
WATCH_PERIOD = 300
# A drive whose video failed (every card full) is tried again by the watcher after this many seconds.
RETRY_FAILED = 1800
# Minutes whose frames are counted at once.
WORKERS = 8
GPU_ENCODER = 'h264_nvenc'
# The GPU's constant quality: about 2 MB a minute, the size libx264's crf 27 gave before the GPU made them.
GPU_QUALITY = 36
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
    # Made at full definition before 7 Nov 2026: too heavy to seek in, made again.
    if value.get('width') == FULL_WIDTH:
        return None
    return value


def _cards() -> list[int]:
    """The server's NVIDIA cards (nvidia-smi's numbers, in bus order), the one with the most free memory first; none
    without nvidia-smi."""
    try:
        answer = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'],
                                capture_output=True, text=True, timeout=30, check=True)
    except (OSError, subprocess.SubprocessError):
        return []
    cards = []
    for line in answer.stdout.splitlines():
        number, _, free = (part.strip() for part in line.partition(','))
        if number.isdigit():
            cards.append((-int(free) if free.isdigit() else 0, int(number)))
    return [number for _free, number in sorted(cards)]


def _ffmpeg(arguments: list[str], done: Callable[[int], None]) -> int:
    """Runs ffmpeg at low priority, done(frames so far) while it works; returns the frames it wrote. Its cards are
    numbered in bus order, as nvidia-smi numbers them (_cards)."""
    command = ['nice', '-n', '15', 'ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostats', '-progress', 'pipe:2',
               '-y', *arguments]
    process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stderr=subprocess.PIPE, text=False,
                               env={**os.environ, 'CUDA_DEVICE_ORDER': 'PCI_BUS_ID'})
    errors, frames = [], 0
    for line in process.stderr:  # type: ignore[union-attr]
        text = line.decode('utf-8', 'replace').strip()
        if text.startswith('frame='):
            frames = int(text[6:] or 0)
            done(frames)
        elif text and not re.fullmatch(r'\w+=\S*', text):
            errors.append(text)
    if process.wait() != 0:
        raise RuntimeError(' '.join(errors[-3:]) or 'ffmpeg failed')
    return frames


def _gpu_output(target: Path, card: int) -> list[str]:
    return ['-an', '-c:v', GPU_ENCODER, '-gpu', str(card), '-preset', 'p5', '-tune', 'hq', '-rc', 'vbr', '-cq',
            str(GPU_QUALITY), '-b:v', '0', '-spatial-aq', '1', '-g', str(KEY_FRAMES), '-bf', '0', '-movflags', '+faststart',
            str(target)]


def _gpu_piece(parts: list[tuple[int, Path]], target: Path, width: int, frames: int, done: Callable[[int], None],
               card: int, *, cpu_decodes: bool = False) -> None:
    """Minutes in a row to one MP4 encoded on the card, decoded there or by the CPU. The concat protocol reads the
    minutes as one stream, so a frame lost at a join would move every minute after it: any other count is a failure."""
    source = ['-f', 'hevc', '-r', str(FPS), '-i', 'concat:' + '|'.join(str(path) for _segment, path in parts)]
    if cpu_decodes:
        arguments = ['-threads', str(CPU_THREADS), *source, '-vf', f'scale={width}:-2:flags=fast_bilinear',
                     '-pix_fmt', 'yuv420p']
    else:
        arguments = ['-hwaccel', 'cuda', '-hwaccel_device', str(card), '-hwaccel_output_format', 'cuda', *source,
                     '-vf', f'scale_cuda={width}:-2']
    written = _ffmpeg([*arguments, *_gpu_output(target, card)], done)
    if written != frames:
        raise RuntimeError(f'{written} frames written of {frames}')


def _cpu_busy() -> bool:
    try:
        return os.getloadavg()[0] > (os.cpu_count() or 1) / 2
    except OSError:
        return True


def _split(parts: list[tuple[int, Path]], counts: list[int]) -> list[list[int]]:
    """The minutes (their numbers in parts) of each piece: the GPU's first, then the CPU's, each in a row and of about the
    same frames. One piece when the CPU is busy or the drive is short."""
    total = sum(counts)
    if CPU_SHARE <= 0 or len(parts) < 4 or _cpu_busy():
        return [list(range(len(parts)))]
    gpu_frames, number = 0, 0
    while number < len(parts) and gpu_frames < (1 - CPU_SHARE) * total:
        gpu_frames += counts[number]
        number += 1
    rest = list(range(number, len(parts)))
    left = sum(counts[i] for i in rest)
    pieces, piece, frames = [list(range(number))], [], 0
    for i in rest:
        piece.append(i)
        frames += counts[i]
        if frames >= left / CPU_WORKERS and len(pieces) < CPU_WORKERS:
            pieces.append(piece)
            piece, frames = [], 0
    if piece:
        pieces.append(piece)
    return pieces


def _encode_on_gpu(parts: list[tuple[int, Path]], counts: list[int], folder: Path, target: Path, width: int,
                   done: Callable[[int], None], card: int) -> None:
    """The whole drive to one MP4 encoded on the card, its index at the front: the first minutes decoded on the card,
    the last ones by the CPU at the same time (_split), the pieces joined without encoding again."""
    pieces = _split(parts, counts)
    if len(pieces) == 1:
        _gpu_piece(parts, target, width, sum(counts), done, card)
        return
    folder.mkdir(exist_ok=True)
    files = [folder / f'{number:04d}.mp4' for number in range(len(pieces))]
    written = [0] * len(pieces)
    counting = threading.Lock()

    def advance(number: int, frames: int) -> None:
        with counting:
            written[number] = frames
            done(sum(written))

    with ThreadPoolExecutor(max_workers=len(pieces)) as pool:
        jobs = [pool.submit(_gpu_piece, [parts[i] for i in piece], file, width, sum(counts[i] for i in piece),
                            functools.partial(advance, number), card, cpu_decodes=number > 0)
                for number, (piece, file) in enumerate(zip(pieces, files))]
        for job in jobs:
            job.result()
    _join(files, folder, target)


def _join(pieces: list[Path], folder: Path, target: Path) -> None:
    """Pieces made with the same encoder joined as they are (the concat demuxer reads the names next to its list), the
    index at the front."""
    listing = folder / 'list.txt'
    listing.write_text(''.join(f"file '{piece.name}'\n" for piece in pieces), encoding='utf-8')
    joined = subprocess.run(['nice', '-n', '15', 'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'concat',
                             '-safe', '0', '-i', str(listing), '-c', 'copy', '-movflags', '+faststart', str(target)],
                            capture_output=True, text=True, timeout=600, check=False)
    if joined.returncode != 0:
        raise RuntimeError('ffmpeg could not join the minutes: ' + joined.stderr.strip()[-300:])


def make(root: Path, route: str, progress: Callable[[float], None] | None = None) -> dict:
    """Makes the video of a drive and its index, and returns the index. progress(0-1) while ffmpeg works."""
    if not ROUTE.fullmatch(route):
        raise ValueError('invalid route id: ' + route)
    parts = minutes(root, route)
    if not parts:
        raise ValueError('no road video of this drive on the server')
    video, index_path = paths(root, route)
    video.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        counts = list(pool.map(lambda part: _frames(part[1]), parts))
    segments, start = [], 0.0
    for (segment, _path), count in zip(parts, counts):
        segments.append({'segment': segment, 'start': round(start, 3), 'frames': count})
        start += count / FPS
    total = sum(counts)
    width = HD_WIDTH if video_dimensions(parts[0][1]) == FULL_SIZE else WIDTH
    # Its own partial file and pieces: the site and `nnslr drive-video` may make the same drive at once; the last one
    # wins whole.
    stem = f'{route}.{os.getpid()}.{threading.get_ident()}'
    partial = video.with_name(stem + '.part.mp4')
    folder = video.with_name(stem + '.parts')

    def advance(frames: int) -> None:
        if progress is not None and total:
            progress(min(1.0, frames / total))

    failures = []
    try:
        # Each card in turn; none left, the video is not made: never on the CPU (#108).
        for card in _cards():
            try:
                _encode_on_gpu(parts, counts, folder, partial, width, advance, card)
                break
            except (OSError, RuntimeError) as exc:
                print(f'Drive video {route}: card {card} could not make it ({exc})', flush=True)
                failures.append(f'card {card}: {exc}')
                shutil.rmtree(folder, ignore_errors=True)
        else:
            raise RuntimeError('no free GPU card for the drive video (' + ('; '.join(failures) or 'no NVIDIA card on this '
                               'server') + '): it is made when a card is free, never on the CPU')
        os.replace(partial, video)
    finally:
        partial.unlink(missing_ok=True)
        shutil.rmtree(folder, ignore_errors=True)
    value = {'route': route, 'fps': FPS, 'width': width, 'encoder': GPU_ENCODER, 'card': card, 'segments': segments,
             'duration': round(start, 3), 'source': _signature(parts), 'made_at': time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    index_path.write_text(json.dumps(value, indent=1) + '\n', encoding='utf-8')
    return value


def state(root: Path, route: str, *, start: bool = False) -> dict:
    """Where the video of a drive is: ready (with its index), being made (how far), failed (why), or missing; start:
    begin making it in the background when it is missing (one at a time: another waits its turn)."""
    if not ROUTE.fullmatch(route):
        return {'state': 'missing', 'reason': 'invalid route id'}
    made = index(root, route)
    if made is not None:
        # The file's time in the address: the browser keeps a video (seeking back needs no new download) and a
        # video made again is another address.
        version = paths(root, route)[0].stat().st_mtime_ns
        return {'state': 'ready', 'url': f'/route/{route}/video.mp4?v={version}', 'index': made}
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
            work.update(state='failed', reason=str(exc), failed_at=time.time())


def _may_start(route: str, now: float | None = None) -> bool:
    """Whether the watcher may make a drive's video: never tried by this process, or failed RETRY_FAILED seconds ago
    or more (every card was full, #108)."""
    with _lock:
        work = _making.get(route)
    if work is None:
        return True
    now = time.time() if now is None else now
    return work['state'] == 'failed' and now - work.get('failed_at', now) >= RETRY_FAILED


def _recent(root: Path, route: str) -> list | None:
    """The signature of a drive's minutes when one of them arrived in the last WATCH_DAYS, else None."""
    parts = minutes(root, route)
    if not parts:
        return None
    signature = _signature(parts)
    newest = max(max(part[1].stat().st_ctime, part[1].stat().st_mtime) for part in parts)
    return signature if time.time() - newest < WATCH_DAYS * 86400 else None


def watch(root: Path, stop: threading.Event | None = None) -> None:
    """Makes, one at a time and newest first, the video of every recent drive without one, once its minutes did not
    change between two scans (a drive still arriving waits): the video is ready before the page asks for it."""
    root = Path(root)
    seen: dict[str, list] = {}
    stop = stop or threading.Event()
    while not stop.wait(WATCH_PERIOD):
        try:
            with _lock:
                busy = any(work['state'] in ('waiting', 'making') for work in _making.values())
            if busy:
                continue
            folder = root / 'raw' / 'routes'
            routes = sorted((path.name for path in folder.glob('*') if ROUTE.fullmatch(path.name)), reverse=True)
            for route in routes:
                signature = _recent(root, route)
                stable = signature is not None and seen.get(route) == signature
                if signature is not None:
                    seen[route] = signature
                if stable and index(root, route) is None and _may_start(route):
                    print(f'Drive video {route}: made in the background', flush=True)
                    state(root, route, start=True)
                    break
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            print(f'Drive video watch: {exc}', flush=True)


def start_watch(root: Path) -> threading.Event:
    stop = threading.Event()
    threading.Thread(target=watch, args=(Path(root), stop), daemon=True, name='drive-video-watch').start()
    return stop
