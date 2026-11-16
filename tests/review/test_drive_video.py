"""The video of a drive for the Drive view of Results (nnslr_tools.drive_video): the comma's minutes of road video joined
in order into one MP4 with an index of where each minute starts, made once and served with byte ranges."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from nnslr_tools import drive_video
from review_helpers import ROUTE, RunningServer, make_report

needs_ffmpeg = pytest.mark.skipif(shutil.which('ffmpeg') is None or shutil.which('ffprobe') is None,
                                  reason='ffmpeg is not installed')
# The video is made on a card only (#108): with the cards full these tests fail and say so, as the site does.
needs_gpu = pytest.mark.skipif(shutil.which('nvidia-smi') is None, reason='no NVIDIA card: the drive video is GPU only')


def _minute(root: Path, segment: int, seconds: int = 1) -> Path:
    """A raw HEVC "minute" as the comma writes it (here a second or two of a test picture at 20 frames a second)."""
    folder = root / 'raw' / 'routes' / ROUTE / str(segment)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / 'fcamera.hevc'
    subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'lavfi', '-i',
                    f'testsrc=size=320x180:rate=20:duration={seconds}', '-pix_fmt', 'yuv420p', '-c:v', 'libx265',
                    '-x265-params', 'log-level=error', '-f', 'hevc', str(path)], check=True, timeout=120)
    return path


def test_minutes_in_order_and_nothing_to_make_without_them(tmp_path):
    for segment in (10, 2, 9):
        folder = tmp_path / 'raw' / 'routes' / ROUTE / str(segment)
        folder.mkdir(parents=True)
        (folder / 'fcamera.hevc').write_bytes(b'x')
    (tmp_path / 'raw' / 'routes' / ROUTE / 'notes').mkdir()
    assert [segment for segment, _ in drive_video.minutes(tmp_path, ROUTE)] == [2, 9, 10]
    assert drive_video.state(tmp_path, '00000002--abcdef1234')['state'] == 'missing'
    assert drive_video.state(tmp_path, '../etc')['state'] == 'missing'


@needs_ffmpeg
@needs_gpu
def test_a_drive_becomes_one_video_with_where_each_minute_starts(tmp_path):
    _minute(tmp_path, 3, 1)
    _minute(tmp_path, 5, 2)
    shares = []
    made = drive_video.make(tmp_path, ROUTE, shares.append)
    assert [(item['segment'], item['start'], item['frames']) for item in made['segments']] == [(3, 0.0, 20), (5, 1.0, 40)]
    assert made['duration'] == 3.0 and shares and shares[-1] <= 1
    video, index = drive_video.paths(tmp_path, ROUTE)
    probe = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=codec_name,width',
                            '-of', 'json', str(video)], capture_output=True, text=True, check=True)
    assert json.loads(probe.stdout)['streams'][0] == {'codec_name': 'h264', 'width': drive_video.WIDTH}
    assert not list(video.parent.glob('*.part.mp4')) and not list(video.parent.glob('*.parts'))
    # Every frame is there, in order, encoded on a card.
    assert made['encoder'] == drive_video.GPU_ENCODER and made['card'] in drive_video._cards()
    frames = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_packets', '-show_entries',
                             'stream=nb_read_packets', '-of', 'csv=p=0', str(video)], capture_output=True, text=True, check=True)
    assert int(frames.stdout.strip()) == 60
    assert drive_video.state(tmp_path, ROUTE)['state'] == 'ready'
    # A minute changed (copied again from the comma): the video must be made again.
    _minute(tmp_path, 5, 1)
    assert drive_video.index(tmp_path, ROUTE) is None
    assert json.loads(index.read_text())['route'] == ROUTE


def _fake_minutes(tmp_path, monkeypatch, count=2):
    for segment in range(count):
        folder = tmp_path / 'raw' / 'routes' / ROUTE / str(segment)
        folder.mkdir(parents=True)
        (folder / 'fcamera.hevc').write_bytes(b'x')
    monkeypatch.setattr(drive_video, '_frames', lambda _video: 20)
    monkeypatch.setattr(drive_video, 'video_dimensions', lambda _video: (1344, 760))


def test_with_every_card_full_the_video_is_not_made_and_never_on_the_cpu(tmp_path, monkeypatch):
    _fake_minutes(tmp_path, monkeypatch)
    tried = []

    def full(_parts, _counts, folder, target, _width, _done, card):
        tried.append(card)
        folder.mkdir(exist_ok=True)
        target.write_bytes(b'half')
        raise RuntimeError('CUDA_ERROR_OUT_OF_MEMORY')
    monkeypatch.setattr(drive_video, '_cards', lambda: [1, 0])
    monkeypatch.setattr(drive_video, '_encode_on_gpu', full)
    ran = []
    monkeypatch.setattr(drive_video.subprocess, 'Popen', lambda *args, **_kw: ran.append(args) or 1 / 0)
    with pytest.raises(RuntimeError, match='no free GPU card .*card 1: CUDA_ERROR_OUT_OF_MEMORY; card 0: .*never on the CPU'):
        drive_video.make(tmp_path, ROUTE)
    assert tried == [1, 0] and ran == []
    video, index = drive_video.paths(tmp_path, ROUTE)
    assert not video.exists() and not index.exists()
    assert not list(video.parent.glob('*.part.mp4')) and not list(video.parent.glob('*.parts'))
    # A server without a card says so.
    monkeypatch.setattr(drive_video, '_cards', lambda: [])
    with pytest.raises(RuntimeError, match='no NVIDIA card on this server'):
        drive_video.make(tmp_path, ROUTE)


def test_a_full_card_leaves_the_drive_to_the_next_one(tmp_path, monkeypatch):
    _fake_minutes(tmp_path, monkeypatch)

    def encode(_parts, _counts, _folder, target, _width, _done, card):
        if card == 0:
            raise RuntimeError('CUDA_ERROR_OUT_OF_MEMORY')
        target.write_bytes(b'mp4')
    monkeypatch.setattr(drive_video, '_cards', lambda: [0, 1])
    monkeypatch.setattr(drive_video, '_encode_on_gpu', encode)
    made = drive_video.make(tmp_path, ROUTE)
    assert made['encoder'] == drive_video.GPU_ENCODER and made['card'] == 1 and made['duration'] == 2.0
    assert drive_video.paths(tmp_path, ROUTE)[0].read_bytes() == b'mp4'


def test_the_cards_come_most_free_memory_first(monkeypatch):
    class Answer:
        stdout = '0, 1200\n1, 30000\n'
    monkeypatch.setattr(drive_video.subprocess, 'run', lambda *_args, **_kw: Answer())
    assert drive_video._cards() == [1, 0]

    def missing(*_args, **_kw):
        raise FileNotFoundError('nvidia-smi')
    monkeypatch.setattr(drive_video.subprocess, 'run', missing)
    assert drive_video._cards() == []


def test_the_watch_tries_a_failed_drive_again_after_half_an_hour(monkeypatch):
    then = 1000.0
    monkeypatch.setitem(drive_video._making, ROUTE, {'state': 'failed', 'progress': 0.0, 'reason': 'full',
                                                     'failed_at': then})
    assert not drive_video._may_start(ROUTE, now=then + drive_video.RETRY_FAILED - 1)
    assert drive_video._may_start(ROUTE, now=then + drive_video.RETRY_FAILED)
    monkeypatch.setitem(drive_video._making, ROUTE, {'state': 'making', 'progress': 0.5, 'reason': ''})
    assert not drive_video._may_start(ROUTE, now=then + 10 ** 6)
    monkeypatch.delitem(drive_video._making, ROUTE)
    assert drive_video._may_start(ROUTE)


def test_the_gpu_must_write_every_frame_of_the_index(tmp_path, monkeypatch):
    # A frame lost where two minutes join would move all the minutes after it.
    monkeypatch.setattr(drive_video, '_ffmpeg', lambda _arguments, _done: 59)
    with pytest.raises(RuntimeError, match='59 frames written of 60'):
        drive_video._encode_on_gpu([(0, tmp_path / 'fcamera.hevc')], [60], tmp_path / 'parts', tmp_path / 'out.mp4',
                                   drive_video.WIDTH, lambda _frames: None, 0)


def test_the_cpu_decodes_the_last_minutes_beside_the_gpu(monkeypatch):
    monkeypatch.setattr(drive_video, '_cpu_busy', lambda: False)
    parts = [(segment, Path(f'{segment}/fcamera.hevc')) for segment in range(50)]
    pieces = drive_video._split(parts, [1200] * 50)
    # The GPU's minutes first, then the CPU's, every minute once and in order.
    assert [i for piece in pieces for i in piece] == list(range(50))
    assert len(pieces) == drive_video.CPU_WORKERS + 1
    assert len(pieces[0]) == 28
    # A busy CPU, or a short drive, leaves it all to the GPU.
    monkeypatch.setattr(drive_video, '_cpu_busy', lambda: True)
    assert drive_video._split(parts, [1200] * 50) == [list(range(50))]
    monkeypatch.setattr(drive_video, '_cpu_busy', lambda: False)
    assert drive_video._split(parts[:3], [1200] * 3) == [list(range(3))]


@needs_ffmpeg
@needs_gpu
def test_a_drive_split_between_gpu_and_cpu_is_whole_and_in_order(tmp_path, monkeypatch):
    monkeypatch.setattr(drive_video, '_cpu_busy', lambda: False)
    for segment in range(10):
        _minute(tmp_path, segment, 1)
    made = drive_video.make(tmp_path, ROUTE)
    video = drive_video.paths(tmp_path, ROUTE)[0]
    frames = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_packets', '-show_entries',
                             'stream=nb_read_packets', '-of', 'csv=p=0', str(video)], capture_output=True, text=True, check=True)
    assert int(frames.stdout.strip()) == 200 and made['duration'] == 10.0
    assert not list(video.parent.glob('*.parts'))


def test_an_old_full_definition_video_is_made_again(tmp_path):
    _folder = tmp_path / 'raw' / 'routes' / ROUTE / '0'
    _folder.mkdir(parents=True)
    (_folder / 'fcamera.hevc').write_bytes(b'x')
    video, index = drive_video.paths(tmp_path, ROUTE)
    video.parent.mkdir(parents=True)
    video.write_bytes(b'v')
    value = {'width': drive_video.FULL_WIDTH, 'source': drive_video._signature(drive_video.minutes(tmp_path, ROUTE))}
    index.write_text(json.dumps(value))
    assert drive_video.index(tmp_path, ROUTE) is None
    index.write_text(json.dumps(dict(value, width=drive_video.HD_WIDTH)))
    assert drive_video.index(tmp_path, ROUTE) is not None


def test_the_watch_makes_a_recent_drive_once_its_minutes_stopped_changing(tmp_path, monkeypatch):
    folder = tmp_path / 'raw' / 'routes' / ROUTE / '0'
    folder.mkdir(parents=True)
    (folder / 'fcamera.hevc').write_bytes(b'x')
    started = []
    monkeypatch.setattr(drive_video, 'state', lambda _root, route, start=False: started.append(route))

    class Ticks:
        def __init__(self, count):
            self.count = count

        def wait(self, _seconds):
            self.count -= 1
            return self.count < 0

    drive_video.watch(tmp_path, Ticks(1))
    assert started == []  # seen once: it may still be arriving
    drive_video.watch(tmp_path, Ticks(2))
    assert started == [ROUTE]


@needs_ffmpeg
@needs_gpu
def test_the_site_serves_the_video_with_byte_ranges(tmp_path):
    make_report(tmp_path)
    _minute(tmp_path, 0, 1)
    drive_video.make(tmp_path, ROUTE)
    size = drive_video.paths(tmp_path, ROUTE)[0].stat().st_size
    server = RunningServer(tmp_path)
    try:
        status, _, data = server.request('GET', '/api/panel/drive-video/' + ROUTE)
        answer = json.loads(data)
        assert status == 200 and answer['state'] == 'ready' and answer['url'].startswith(f'/route/{ROUTE}/video.mp4?v=')
        status, response, data = server.request('GET', answer['url'], headers={'Range': 'bytes=10-109'})
        assert status == 206 and len(data) == 100 and response.getheader('Content-Range') == f'bytes 10-109/{size}'
        assert response.getheader('Content-Type') == 'video/mp4' and response.getheader('Accept-Ranges') == 'bytes'
        assert response.getheader('Cache-Control') == 'private, max-age=86400'
        status, _, data = server.request('GET', f'/route/{ROUTE}/video.mp4')
        assert status == 200 and len(data) == size
        status, _, _ = server.request('GET', f'/route/{ROUTE}/video.mp4', headers={'Range': f'bytes={size}-'})
        assert status == 416
        # The control pages may play it (media-src).
        status, response, _ = server.request('GET', '/evaluate')
        assert "media-src 'self'" in response.getheader('Content-Security-Policy')
    finally:
        server.close()
