"""Every frame of the road video in Sign review (nnslr_tools.video_frames): GOP previews, a frame between the seconds
saved when it is confirmed (with boxes or as no sign), as a real picture of the report that Propose again and the
import keep."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time

import pytest

from nnslr_tools import cli, video_frames
from nnslr_tools.media import extract_frames, extracted_frame_rows
from nnslr_tools.preannotate import discover_frames
from nnslr_tools.review import write_review
from nnslr_tools.review_db import ReviewStore, db_path
from review_helpers import ROUTE, RUN_ID, RunningServer, json_of, proposal

NEW_RUN = '20261022T100000Z-0123abce'
BOX = {'bbox_xyxy': [4, 4, 20, 20], 'family': 'maximum_speed', 'value_kph': 50, 'value_state': 'reviewed',
       'score': None, 'model_label': None, 'class_id': None, 'review_state': 'pending', 'applicability': 'unresolved',
       'supplementary_panel': 'unknown', 'detection_id': 'manual-1'}


def report_of(root, rows, run_id):
    """A pre-annotation report of these frame rows: a proposal on the first one."""
    lines = []
    directory = root / 'derived/preannotations' / ROUTE / run_id
    (directory / 'images').mkdir(parents=True)
    for ordinal, row in enumerate(rows):
        (directory / 'images' / f'{ordinal:08d}.jpg').write_bytes(b'jpeg preview')
        lines.append({**row, 'schema_version': 1, 'kind': 'preannotated_frame', 'run_id': run_id, 'width': 64,
                      'height': 48, 'preview_path': f'images/{ordinal:08d}.jpg',
                      'detections': [proposal(row['output_index'])] if ordinal == 0 else [],
                      'review_state': 'pending', 'ground_truth': False, 'model': {'repository': 'synthetic'}})
    (directory / 'preannotations.jsonl').write_text(''.join(json.dumps(line) + '\n' for line in lines))
    (directory / 'run.json').write_text(json.dumps({'run_id': run_id, 'route_id': ROUTE}))
    write_review(directory, lines, run_id)
    (root / 'derived/preannotations' / ROUTE / 'latest.json').write_text(json.dumps({'run_id': run_id}))
    return lines


@pytest.fixture
def drive(tmp_path):
    """A 3-second road video at 20 frames a second, a keyframe every 20, its pictures at 1 a second and a report."""
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        pytest.skip('ffmpeg/ffprobe required for local media integration')
    video = tmp_path / 'raw/routes' / ROUTE / '0' / 'fcamera.hevc'
    video.parent.mkdir(parents=True)
    made = subprocess.run([
        'ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'testsrc=s=64x48:r=20:d=3', '-pix_fmt', 'yuv420p',
        '-c:v', 'libx265', '-x265-params', 'pools=1:frame-threads=1:log-level=error:bframes=0:keyint=20:min-keyint=20',
        str(video)], capture_output=True, text=True)
    assert made.returncode == 0, made.stderr
    folder = tmp_path / 'derived/frames' / ROUTE / '0'
    pictures = extracted_frame_rows(extract_frames(video, folder, fps=1), route_id=ROUTE, segment_index=0)
    (folder / 'frames.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in pictures))
    rows = report_of(tmp_path, discover_frames(tmp_path, ROUTE), RUN_ID)
    server = RunningServer(tmp_path)
    yield tmp_path, video, rows, server
    server.close()


def body(detections, run_id=RUN_ID, confirmed=True):
    return {'run_id': run_id, 'reviewer': 'Cristian', 'confirmed': confirmed, 'detections': detections,
            'site_group': None, 'encounter_id': None}


def test_a_gop_has_the_pixels_of_a_decode_of_the_whole_video(drive, tmp_path):
    root, video, rows, server = drive
    assert video_frames.gop_of(video, 25) == (20, 39) and video_frames.gop_of(video, 59) == (40, 59)
    reference = tmp_path / 'reference.jpg'
    subprocess.run(['ffmpeg', '-v', 'error', '-i', str(video), '-vf', 'select=eq(n\\,25)', '-vsync', '0',
                    '-q:v', video_frames.JPEG_QUALITY, str(reference)], check=True)
    assert video_frames.preview(root, ROUTE, 0, 25).read_bytes() == reference.read_bytes()
    # The frame is given once written; the rest of its GOP follows.
    assert video_frames.settle()
    assert len(list((root / video_frames.CACHE_DIR / ROUTE / '0').glob('*.jpg'))) == 20
    # The page: the frames of the GOP, the picture at 1 a second under its own key.
    status, _response, data = server.request('GET', f'/api/route/{ROUTE}/video/0/25')
    assert status == 200, data
    gop = json_of(data)
    assert gop['frame_count'] == 60 and gop['gop'] == [20, 39] and 'seconds' not in gop
    keys = {frame['decoded']: frame['frame_key'] for frame in gop['frames']}
    assert keys[20] == '0/1' and keys[25] == '0/1025' and len(keys) == 20
    status, response, data = server.request('GET', f'/route/{ROUTE}/video/0/25.jpg')
    assert status == 200 and response.getheader('Content-Type') == 'image/jpeg' and data == reference.read_bytes()
    assert server.request('GET', f'/api/route/{ROUTE}/video/0/60')[0] == 404
    # The cache stays within its budget, the least recently shown first.
    assert video_frames.settle()
    video_frames.prune(root, budget=0)
    assert not list((root / video_frames.CACHE_DIR).glob('*/*/*.jpg'))



def cached(root, wide=False):
    """The frames in the cache of segment 0, by decoded index."""
    folder = root / video_frames.CACHE_DIR / ROUTE / ('0-wide' if wide else '0')
    return sorted(int(path.stem) for path in folder.glob('*.jpg'))


def test_the_gop_json_answers_once_its_first_frame_is_decoded_and_a_frame_shown_reads_the_next_gop_ahead(drive, monkeypatch):
    """The arrows of Sign review ask for the GOP first: it answers once ffmpeg has written the GOP's first frame, the
    rest still being decoded. A frame given to a page has the GOP after it decoded ahead (and the one before, in the
    first third of its GOP), so that a step into it does not wait for ffmpeg."""
    root, video, rows, server = drive
    started = []
    real = video_frames._decode
    monkeypatch.setattr(video_frames, '_decode', lambda *args, **kwargs: started.append((args[3], kwargs)) or real(*args, **kwargs))
    status, _response, data = server.request('GET', f'/api/route/{ROUTE}/video/0/25')
    assert status == 200 and json_of(data)['gop'] == [20, 39]
    assert started == [(20, {})] and 20 in cached(root)
    assert video_frames.settle()
    assert cached(root) == list(range(20, 40))
    # Frame 45 (GOP 40..59, the last of the video), in the first third of its GOP: the GOP before is in the cache, no
    # GOP after it.
    started.clear()
    assert server.request('GET', f'/route/{ROUTE}/video/0/45.jpg')[0] == 200
    assert video_frames.settle()
    assert cached(root) == list(range(20, 60)) and [first for first, _kwargs in started] == [40]
    # Frame 5 (GOP 0..19): the next GOP is there already; nothing before the first.
    started.clear()
    assert server.request('GET', f'/route/{ROUTE}/video/0/5.jpg')[0] == 200
    assert video_frames.settle()
    assert cached(root) == list(range(60)) and [first for first, _kwargs in started] == [0]
    # Frame 25, past the first third of GOP 20..39, after a prune: the GOP after it is decoded ahead (niced), not the
    # one before.
    video_frames.prune(root, budget=0)
    assert video_frames.preview(root, ROUTE, 0, 25).is_file()
    started.clear()
    video_frames.read_ahead(root, ROUTE, 0, 32)
    assert video_frames.settle()
    assert cached(root) == list(range(20, 60)) and started == [(40, {'ahead': True})]
    # Stepping back from 32 to 31, past the first third of GOP 20..39: the GOP before is decoded ahead too, first.
    video_frames.prune(root, budget=0)
    assert video_frames.preview(root, ROUTE, 0, 31).is_file() and video_frames.settle()
    started.clear()
    video_frames.read_ahead(root, ROUTE, 0, 31)
    assert video_frames.settle()
    assert cached(root) == list(range(60)) and started == [(0, {'ahead': True}), (40, {'ahead': True})]


def test_requests_for_a_gop_being_decoded_share_its_decode_and_a_failed_decode_is_tried_again(drive, monkeypatch):
    root, video, _rows, _server = drive
    spawned = []
    real = video_frames.subprocess.Popen

    def popen(command, **kwargs):
        if 'ffmpeg' in command:
            spawned.append(command)
        return real(command, **kwargs)
    monkeypatch.setattr(video_frames.subprocess, 'Popen', popen)
    import threading
    shown = {}
    threads = [threading.Thread(target=lambda frame=frame: shown.setdefault(frame, video_frames.preview(root, ROUTE, 0, frame)))
               for frame in (21, 30, 39)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(spawned) == 1 and sorted(shown) == [21, 30, 39] and all(path.is_file() for path in shown.values())
    assert video_frames.settle() and cached(root) == list(range(20, 40))
    # A GOP ffmpeg cannot decode: the frame is refused, nothing half written stays, and the next request decodes again.
    monkeypatch.setattr(video_frames, 'gop_bytes', lambda *args: b'not a video')
    with pytest.raises(video_frames.VideoFrameError, match='GOP 40..59 not decoded'):
        video_frames.preview(root, ROUTE, 0, 45)
    assert not video_frames._decodes and cached(root) == list(range(20, 40))
    assert not list((root / video_frames.CACHE_DIR / ROUTE / '0').glob('*.tmp'))
    monkeypatch.undo()
    assert video_frames.preview(root, ROUTE, 0, 45).is_file()
    assert video_frames.settle() and cached(root) == list(range(20, 60))


@pytest.mark.parametrize('broken', ['not decoded by ffmpeg', 'not read'])
def test_a_gop_that_cannot_be_decoded_is_a_404_so_the_arrows_go_from_picture_to_picture(drive, monkeypatch, broken):
    """Sign review's arrows fall back on the pictures at 1 a second when the GOP of the frame they step to is refused
    (video.ts moveOne): ffmpeg writing nothing of it, or its bytes not read, is that refusal, not a GOP of frames whose
    pictures all fail."""
    root, _video, _rows, server = drive

    def unreadable(*_args):
        raise video_frames.VideoFrameError('the road video cannot be read')
    monkeypatch.setattr(video_frames, 'gop_bytes', (lambda *_args: b'not a video') if broken == 'not decoded by ffmpeg'
                        else unreadable)
    status, _response, data = server.request('GET', f'/api/route/{ROUTE}/video/0/45')
    assert status == 404 and json_of(data)['error'] == 'no_video_frame'
    assert video_frames.settle() and cached(root) == [] and not video_frames._decodes
    # The GOPs it can decode still answer.
    monkeypatch.undo()
    status, _response, data = server.request('GET', f'/api/route/{ROUTE}/video/0/45')
    assert status == 200 and json_of(data)['gop'] == [40, 59]


def test_the_cache_is_pruned_in_the_background_below_its_budget(drive, monkeypatch):
    root, video, _rows, _server = drive
    video_frames.preview(root, ROUTE, 0, 5)
    assert video_frames.settle()
    size = sum((root / video_frames.CACHE_DIR / ROUTE / '0' / f'{frame}.jpg').stat().st_size for frame in range(20))
    # Room for about 30 frames: the decode of a second GOP passes the budget, the oldest go down to PRUNE_TO of it.
    monkeypatch.setattr(video_frames, 'CACHE_BYTES', size * 3 // 2)
    video_frames.prune(root)
    video_frames.preview(root, ROUTE, 0, 25)
    assert video_frames.settle()
    left = cached(root)
    assert left[-20:] == list(range(20, 40)) and 20 < len(left) < 40
    total = sum((root / video_frames.CACHE_DIR / ROUTE / '0' / f'{frame}.jpg').stat().st_size for frame in left)
    assert total <= video_frames.CACHE_BYTES * video_frames.PRUNE_TO
    assert video_frames._cache_total[str(root)] == total

def wide_video(root, frames_s, source='smptebars'):
    """A wide road video of segment 0 (ecamera.hevc), another picture from the road video's."""
    video = root / 'raw/routes' / ROUTE / '0' / video_frames.WIDE_VIDEO
    made = subprocess.run([
        'ffmpeg', '-y', '-v', 'error', '-f', 'lavfi', '-i', f'{source}=s=64x48:r=20:d={frames_s}', '-pix_fmt',
        'yuv420p', '-c:v', 'libx265', '-x265-params',
        'pools=1:frame-threads=1:log-level=error:bframes=0:keyint=20:min-keyint=20', '-f', 'hevc', str(video)],
        capture_output=True, text=True)
    assert made.returncode == 0, made.stderr
    return video


def test_the_wide_camera_gives_the_same_frame_only_with_the_frames_of_the_road_video(drive, tmp_path):
    root, video, _rows, _server = drive
    with pytest.raises(video_frames.VideoFrameError, match='no wide road video'):
        video_frames.preview(root, ROUTE, 0, 25, wide=True)
    wide = wide_video(root, 3)
    reference = tmp_path / 'wide-reference.jpg'
    subprocess.run(['ffmpeg', '-v', 'error', '-f', 'hevc', '-i', str(wide), '-vf', 'select=eq(n\\,25)', '-vsync',
                    '0', '-q:v', video_frames.JPEG_QUALITY, str(reference)], check=True)
    shown = video_frames.preview(root, ROUTE, 0, 25, wide=True)
    assert shown.read_bytes() == reference.read_bytes() and shown.parent.name == '0-wide'
    assert shown.read_bytes() != video_frames.preview(root, ROUTE, 0, 25).read_bytes()
    # One frame short of the road video: its frame n may not be the road's, so it is not shown.
    wide_video(root, 2.95)
    with pytest.raises(video_frames.VideoFrameError, match='not the frames of the road video'):
        video_frames.wide_video_of(root, ROUTE, 0)
    assert video_frames.settle()
    video_frames.prune(root, budget=0)
    assert not list((root / video_frames.CACHE_DIR).glob('*/*/*.jpg'))


def test_a_frame_between_the_seconds_is_saved_with_a_box_and_kept_by_propose_again(drive, capsys):
    root, video, rows, server = drive
    manifest = root / 'derived/frames' / ROUTE / '0' / video_frames.VIDEO_MANIFEST
    # Not confirmed: nothing taken out of the video.
    assert server.post_frame(ROUTE, '0/1025', body([BOX], confirmed=False))[0] == 404 and not manifest.exists()
    # The picture at 1 a second under a key of the frames between: never a second key for the same image.
    assert server.post_frame(ROUTE, '0/1020', body([BOX]))[0] == 404 and not manifest.exists()
    # A save refused after the picture was taken out leaves nothing behind.
    status, _response, data = server.post_frame(ROUTE, '0/1025', body([{**BOX, 'family': 'no-such-family'}]))
    assert status == 400 and not manifest.exists() and not list(manifest.parent.glob('video_*.png'))
    status, _response, data = server.post_frame(ROUTE, '0/1025', body([BOX]))
    assert status == 200, data
    picture = manifest.parent / 'video_00001025.png'
    assert picture.is_file() and [row['decoded_frame_index'] for row in map(json.loads, manifest.open())] == [25]
    added = server.app.report(ROUTE)[2]['0/1025']
    assert added['image_path'] == str(picture.relative_to(root)) and added['preview_path'] == 'video/0/25.jpg'
    assert [row['frame_key'] for row in server.app.report(ROUTE)[1]] == ['0/0', '0/1', '0/1025', '0/2']
    # The page made again (a reload) has it among its frames: the data of the Signs side of Speed review (the drive has
    # its video, so its old page leads there).
    status, _response, data = server.request('GET', f'/api/route/{ROUTE}/page')
    assert status == 200 and '0/1025' in [row['frame_key'] for row in json_of(data)['frames']]
    store = ReviewStore(db_path(root))
    saved = store.review(ROUTE, RUN_ID, '0/1025')
    assert saved['confirmed'] and saved['payload']['image_path'] == added['image_path']
    # Its picture is the same as the decode of the whole video.
    same = extract_frames(video, root / 'check', decoded=[25])
    assert (root / 'check' / 'frame_00000000.png').read_bytes() == picture.read_bytes() and same
    # Saved again: the same picture, no second row.
    status, _response, data = server.post_frame(ROUTE, '0/1025', {**body([BOX]), 'base_revision': 1})
    assert status == 200, data
    assert len(manifest.read_text().splitlines()) == 1
    # Propose again: the frame is in the new report and its confirmation is carried onto it.
    found = discover_frames(root, ROUTE)
    assert [row['frame_key'] for row in found] == ['0/0', '0/1', '0/1025', '0/2']
    report_of(root, found, NEW_RUN)
    keys = [row['frame_key'] for row in found]
    assert store.carry_confirmed(route_id=ROUTE, from_run=RUN_ID, to_run=NEW_RUN, frame_keys=keys, client='t') == 1
    assert server.app.report(ROUTE)[2]['0/1025']['run_id'] == NEW_RUN
    assert server.app.report(ROUTE)[2]['0/1025']['preview_path'] == 'images/00000002.jpg'
    # The import: the frame between the seconds with its box, on its own picture.
    try:
        code = cli.main(['import-annotations', '--from-db', '--data-root', str(root)])
    except SystemExit as exc:
        code = exc.code
    out = capsys.readouterr()
    assert code == 0, out.out + out.err
    objects = [json.loads(line) for line in (root / 'annotations/objects.jsonl').read_text().splitlines()]
    imported = [row for row in objects if row['output_index'] == 1025]
    assert imported and all(row['image_path'] == added['image_path'] for row in imported)


def test_a_frame_between_the_seconds_with_no_sign_is_saved_too(drive):
    root, video, rows, server = drive
    status, _response, data = server.post_frame(ROUTE, '0/1045', body([]))
    assert status == 200, data
    manifest = root / 'derived/frames' / ROUTE / '0' / video_frames.VIDEO_MANIFEST
    assert [row['decoded_frame_index'] for row in map(json.loads, manifest.open())] == [45]
    assert [row['frame_key'] for row in server.app.report(ROUTE)[1]] == ['0/0', '0/1', '0/2', '0/1045']
    saved = ReviewStore(db_path(root)).review(ROUTE, RUN_ID, '0/1045')
    assert saved['confirmed'] and saved['payload']['detections'] == []


def test_a_frame_prepared_ahead_is_saved_without_decoding_again_with_the_same_picture(drive, monkeypatch, tmp_path):
    root, video, rows, server = drive
    video_frames.prepare(root, ROUTE, 0, 45)
    folder = root / 'derived/frames' / ROUTE / '0'
    prepared = folder / f'{video_frames.PREPARED_PREFIX}45'
    assert (prepared / 'row.json').is_file()
    # The save only moves the pictures prepare took out.
    monkeypatch.setattr(video_frames, 'extract_frames', lambda *args, **kwargs: pytest.fail('decoded again'))
    status, _response, data = server.post_frame(ROUTE, '0/1045', body([]))
    assert status == 200, data
    assert not prepared.exists()
    row = json.loads((folder / video_frames.VIDEO_MANIFEST).read_text())
    assert row['decoded_frame_index'] == 45 and row['image_path'] == str(folder / 'video_00001045.png')
    monkeypatch.undo()
    again = extract_frames(video, tmp_path / 'again', decoded=[45])[0]
    assert again.image_sha256 == row['image_sha256']
    # A frame already saved is not prepared again.
    video_frames.prepare(root, ROUTE, 0, 45)
    assert not prepared.exists()


def test_a_save_takes_out_its_whole_gop_and_the_saves_of_its_other_frames_only_move_pictures(drive, monkeypatch,
                                                                                               tmp_path):
    """Confirm walks the video frame by frame: one ffmpeg a GOP, not one a frame, whose saves fell minutes behind the
    reviewer one at a time (#99, 15 Nov 2026)."""
    root, video, rows, server = drive
    folder = root / 'derived/frames' / ROUTE / '0'
    calls = []
    real = video_frames.extract_frames
    monkeypatch.setattr(video_frames, 'extract_frames', lambda *args, **kwargs: calls.append(kwargs['decoded']) or
                        real(*args, **kwargs))
    assert server.post_frame(ROUTE, '0/1025', body([BOX]))[0] == 200
    # Its GOP is 20..39: 20 is the picture at 1 a second (left out), 25 is saved, the others wait for their save.
    assert calls == [list(range(21, 40))]
    assert sorted(int(path.name.rsplit('-', 1)[1]) for path in folder.glob(video_frames.PREPARED_PREFIX + '*')) == [
        index for index in range(21, 40) if index != 25]
    for key in ('0/1026', '0/1027', '0/1039'):
        assert server.post_frame(ROUTE, key, body([BOX]))[0] == 200
    assert len(calls) == 1
    manifest = [json.loads(line) for line in (folder / video_frames.VIDEO_MANIFEST).read_text().splitlines()]
    assert [row['decoded_frame_index'] for row in manifest] == [25, 26, 27, 39]
    # The same pictures as a decode of the whole video, the frames prepared included.
    for row in manifest:
        again = real(video, tmp_path / f'again-{row["decoded_frame_index"]}', decoded=[row['decoded_frame_index']])[0]
        assert again.image_sha256 == row['image_sha256'] and (folder / f'video_{row["output_index"]:08d}.png').is_file()
    # The next GOP: one ffmpeg more, without the frames already a picture.
    assert server.post_frame(ROUTE, '0/1045', body([BOX]))[0] == 200
    assert calls[1] == list(range(41, 60))


def test_the_gops_of_a_segment_are_taken_out_at_once_and_one_gop_once(drive, monkeypatch):
    """ffmpeg runs under its GOP's lock, not the segment's: two GOPs at once (each waits for the other to start), and
    two frames of one GOP share one run."""
    root, _video, _rows, _server = drive
    real = video_frames.extract_frames
    both = threading.Barrier(2, timeout=20)
    calls = []

    def together(*args, **kwargs):
        calls.append(kwargs['decoded'][0])
        both.wait()
        return real(*args, **kwargs)
    monkeypatch.setattr(video_frames, 'extract_frames', together)
    made, errors = {}, []

    def save(decoded):
        try:
            made[decoded] = video_frames.materialize(root, ROUTE, 0, decoded)
        except Exception as exc:  # noqa: BLE001 - the test says which
            errors.append(exc)
    threads = [threading.Thread(target=save, args=(decoded,)) for decoded in (25, 45, 26, 46)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors and sorted(calls) == [21, 41] and all(made[decoded][1] for decoded in (25, 45, 26, 46))
    manifest = root / 'derived/frames' / ROUTE / '0' / video_frames.VIDEO_MANIFEST
    assert sorted(json.loads(line)['decoded_frame_index'] for line in manifest.read_text().splitlines()) == [
        25, 26, 45, 46]


def test_a_save_sent_again_after_it_was_stored_is_answered_ok_without_a_new_revision(drive):
    """The page sends a save again when its answer is late (CALL_TIMEOUT_MS): with the revision it had, the stored one
    is newer. The same labels again are that save, stored already: OK, nothing written. Other labels stay a
    conflict (#99)."""
    root, _video, _rows, server = drive
    store = ReviewStore(db_path(root))
    for key in ('0/1', '0/1025'):
        status, _response, data = server.post_frame(ROUTE, key, {**body([BOX]), 'base_revision': 0})
        assert status == 200, data
        status, _response, data = server.post_frame(ROUTE, key, {**body([BOX]), 'base_revision': 0})
        assert status == 200, data
        assert json_of(data)['review']['revision'] == 1 and len(store.history(ROUTE, RUN_ID, key)) == 1
        status, _response, data = server.post_frame(ROUTE, key, {**body([dict(BOX, value_kph=60)]), 'base_revision': 0})
        assert status == 409 and json_of(data)['error'] == 'revision_conflict'
        other = {**body([BOX]), 'reviewer': 'Someone else', 'base_revision': 0}
        assert server.post_frame(ROUTE, key, other)[0] == 409
        assert store.review(ROUTE, RUN_ID, key)['revision'] == 1


def test_two_saves_of_one_frame_at_once_go_one_after_the_other(drive, monkeypatch):
    """A copy sent again while the first still runs, or another tab: the same labels are both OK; other labels are a
    conflict that never takes away the picture of the save stored (#99 review)."""
    root, _video, _rows, server = drive
    real = video_frames.extract_frames

    def slow(*args, **kwargs):
        time.sleep(0.3)
        return real(*args, **kwargs)
    monkeypatch.setattr(video_frames, 'extract_frames', slow)
    folder = root / 'derived/frames' / ROUTE / '0'
    store = ReviewStore(db_path(root))
    for key, bodies in (('0/1025', [BOX, BOX]), ('0/1045', [BOX, dict(BOX, value_kph=60)])):
        answers = []

        def save(box):
            status, _response, data = server.post_frame(ROUTE, key, {**body([box]), 'base_revision': 0})
            answers.append((status, json_of(data)))
        threads = [threading.Thread(target=save, args=(box,)) for box in bodies]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        statuses = sorted(status for status, _answer in answers)
        assert statuses == ([200, 200] if bodies[0] == bodies[1] else [200, 409]), answers
        saved = store.review(ROUTE, RUN_ID, key)
        assert saved['confirmed'] and saved['revision'] == 1
        picture = folder / f'video_{int(key.split("/")[1]):08d}.png'
        assert picture.is_file() and key in server.app.report(ROUTE)[2]
    # A save refused while the frame is confirmed already (by the REST API, MCP) keeps its picture.
    server.app.forget_video_frame(ROUTE, '0/1045', RUN_ID)
    assert (folder / 'video_00001045.png').is_file()


def test_the_saves_of_a_drive_sent_together_go_one_at_a_time_and_its_report_file_is_read_once(drive, monkeypatch):
    """Confirm goes on without waiting for its save: the saves of a drive sent together run one after the other, and
    each adds its frame to the report without the report file being read again (#100)."""
    from nnslr_tools import review as review_module
    root, _video, _rows, server = drive
    server.app.report(ROUTE)
    reads = []
    real_rows = review_module.report_rows
    monkeypatch.setattr(review_module, 'report_rows', lambda directory: reads.append(directory) or real_rows(directory))
    running, most = [0], [0]
    real_hints = server.app.hints

    def hints(*args, **kwargs):
        running[0] += 1
        most[0] = max(most[0], running[0])
        try:
            time.sleep(0.05)
            return real_hints(*args, **kwargs)
        finally:
            running[0] -= 1
    monkeypatch.setattr(server.app, 'hints', hints)
    keys = [f'0/{1000 + decoded}' for decoded in range(21, 27)]
    answers = {}

    def save(key):
        answers[key] = server.post_frame(ROUTE, key, {**body([]), 'base_revision': 0})[0]
    threads = [threading.Thread(target=save, args=(key,)) for key in keys]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert answers == {key: 200 for key in keys}
    assert most[0] == 1 and reads == []
    by_key = server.app.report(ROUTE)[2]
    assert all(key in by_key for key in keys) and reads == []
    store = ReviewStore(db_path(root))
    assert all(store.review(ROUTE, RUN_ID, key)['confirmed'] for key in keys)


def test_a_save_never_waits_for_the_ffmpeg_of_another_frame_of_its_drive(drive):
    """A frame between the seconds never prepared waits for ffmpeg (every decode slot taken by the video playing): the
    save of a frame of the report sent meanwhile answers at once, and the first is saved when ffmpeg is free (#100)."""
    root, _video, _rows, server = drive
    server.app.report(ROUTE)
    answers = {}

    def save(key):
        answers[key] = server.post_frame(ROUTE, key, {**body([]), 'base_revision': 0})[0]
    for _ in range(video_frames.MAX_DECODES):
        video_frames._decode_slots.acquire()
    between = threading.Thread(target=save, args=('0/1025',))
    try:
        between.start()
        time.sleep(0.3)
        row = threading.Thread(target=save, args=('0/1',))
        row.start()
        row.join(5)
        assert answers.get('0/1') == 200 and '0/1025' not in answers
    finally:
        for _ in range(video_frames.MAX_DECODES):
            video_frames._decode_slots.release()
    between.join(30)
    assert answers == {'0/1': 200, '0/1025': 200}
    store = ReviewStore(db_path(root))
    assert store.review(ROUTE, RUN_ID, '0/1025')['confirmed'] and store.review(ROUTE, RUN_ID, '0/1')['confirmed']


def test_only_so_many_frames_are_prepared_at_once_and_each_once(tmp_path, monkeypatch):
    started, go = [], threading.Event()

    def held(root, route, segment, decoded):
        started.append(decoded)
        go.wait(10)
    monkeypatch.setattr(video_frames, 'prepare', held)
    monkeypatch.setattr(video_frames, 'PREPARES_AT_ONCE', 3)
    monkeypatch.setattr(video_frames, 'PREPARES_AT_ONCE_ALL', 5)
    other = '00000002--0123456789'
    try:
        for decoded in (5, 5, 6, 7, 8, 9):
            video_frames.prepare_soon(tmp_path, ROUTE, 0, decoded)
        # Another drive is not kept out by this one's; all the drives together have their own count.
        for decoded in (1, 2, 3):
            video_frames.prepare_soon(tmp_path, other, 0, decoded)
        deadline = time.monotonic() + 5
        while len(started) < 5 and time.monotonic() < deadline:
            time.sleep(0.01)
        # The second 5 was already waiting; 8 and 9 came with three of their drive at work, and the 3 of the other
        # drive with five in all: their saves take them out themselves.
        assert sorted(started) == [1, 2, 5, 6, 7]
    finally:
        go.set()
    deadline = time.monotonic() + 5
    while video_frames._preparing and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not video_frames._preparing
    # Once they ended, the next is prepared again, a frame that failed too.
    monkeypatch.setattr(video_frames, 'prepare', lambda *args: (_ for _ in ()).throw(video_frames.VideoFrameError('no video')))
    video_frames.prepare_soon(tmp_path, ROUTE, 0, 8)
    deadline = time.monotonic() + 5
    while video_frames._preparing and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not video_frames._preparing


def test_a_drive_keeps_only_so_many_frames_prepared_the_oldest_go_first(drive, monkeypatch):
    root, _video, _rows, _server = drive
    folder = root / 'derived/frames' / ROUTE / '0'
    monkeypatch.setattr(video_frames, '_last_sweep', [time.monotonic()])
    video_frames.prepare(root, ROUTE, 0, 25)
    prepared = sorted(folder.glob(video_frames.PREPARED_PREFIX + '*'), key=lambda path: int(path.name.rsplit('-', 1)[1]))
    assert len(prepared) == 19
    now = time.time()
    for age, path in enumerate(prepared):
        os.utime(path, (now - 100 + age, now - 100 + age))
    # A frame prepared on another drive is not counted with these and stays.
    other = root / 'derived/frames/00000002--0123456789/3' / (video_frames.PREPARED_PREFIX + '7')
    other.mkdir(parents=True)
    os.utime(other, (now - 5000, now - 5000))
    monkeypatch.setattr(video_frames, 'PREPARED_KEPT', 5)
    video_frames._keep_prepared(root, ROUTE)
    assert sorted(folder.glob(video_frames.PREPARED_PREFIX + '*')) == sorted(prepared[-5:])
    assert other.is_dir() and not list(folder.glob('.video-drop-*'))
    # The next GOP prepared trims by itself, and so does a save that takes its GOP out; a frame that went is still
    # saved: its save takes it out again.
    video_frames.prepare(root, ROUTE, 0, 45)
    assert len(list(folder.glob(video_frames.PREPARED_PREFIX + '*'))) == 5
    for path in folder.glob(video_frames.PREPARED_PREFIX + '*'):
        video_frames._drop(path)
    _row, made = video_frames.materialize(root, ROUTE, 0, 5)
    assert made and len(list(folder.glob(video_frames.PREPARED_PREFIX + '*'))) == 5
    # A folder made again since it was listed is new: it stays. All the drives together have their own count.
    newest = max(folder.glob(video_frames.PREPARED_PREFIX + '*'), key=lambda path: path.stat().st_mtime)
    video_frames._drop_oldest([(newest.stat().st_mtime - 50, newest)], 0)
    assert newest.is_dir()
    monkeypatch.setattr(video_frames, 'PREPARED_KEPT_ALL', 2)
    os.utime(other, (now + 60, now + 60))
    video_frames.forget_old_prepared(root)
    assert other.is_dir() and len(list(folder.glob(video_frames.PREPARED_PREFIX + '*'))) == 1


def test_a_refused_save_takes_nothing_out_of_the_video_and_leaves_nothing_prepared(drive, monkeypatch):
    root, _video, _rows, server = drive
    folder = root / 'derived/frames' / ROUTE / '0'
    # No run_id: refused in any case, before ffmpeg.
    status, _response, data = server.post_frame(ROUTE, '0/1025', {'confirmed': True})
    assert status == 409 and json_of(data)['error'] == 'stale_report', data
    assert not list(folder.glob(video_frames.PREPARED_PREFIX + '*')) and not (folder / video_frames.VIDEO_MANIFEST).exists()
    # A frame past the end of a minute, and a minute without its video, say why, and take no lock.
    for key in ('0/2600', '7/1025'):
        status, _response, data = server.post_frame(ROUTE, key, body([]))
        assert status == 404 and json_of(data)['error'] == 'no_video_frame', data
    assert not [name for name in server.app._route_locks if name[0] in ('frame:0/2600', 'frame:7/1025')]


def test_prepared_frames_left_go_after_an_hour_from_every_drive_and_never_halfway_through_a_save(drive, monkeypatch):
    root, _video, _rows, server = drive
    folder = root / 'derived/frames' / ROUTE / '0'
    # No sweep in the background but those the test asks for.
    monkeypatch.setattr(video_frames, '_last_sweep', [time.monotonic()])
    video_frames.prepare(root, ROUTE, 0, 25)
    old = time.time() - video_frames.PREPARED_KEPT_S - 60
    prepared = sorted(folder.glob(video_frames.PREPARED_PREFIX + '*'))
    assert len(prepared) == 19
    for path in prepared[:-1]:
        os.utime(path, (old, old))
    # A folder a stopped server left, in a segment of another drive never opened again.
    other = root / 'derived/frames/00000002--0123456789/3'
    (other / '.video-abc').mkdir(parents=True)
    os.utime(other / '.video-abc', (old, old))
    # A save takes an old prepared frame whole while the frames left are removed at the same moment.
    real_move = video_frames._move_picture

    def move_while_forgetting(made, directory):
        video_frames.forget_old_prepared(root)
        real_move(made, directory)
    with monkeypatch.context() as patched:
        patched.setattr(video_frames, '_move_picture', move_while_forgetting)
        made, fresh = video_frames.materialize(root, ROUTE, 0, 26)
    assert fresh and made['decoded_frame_index'] == 26 and (folder / 'video_00001026.png').is_file()
    # The old ones went, the recent one stayed; no folder of the save is left.
    assert [path.name for path in folder.glob('.*')] == [prepared[-1].name] and not list(other.iterdir())
    # The sweep runs in the background at most once every SWEEP_EVERY_S.
    monkeypatch.setattr(video_frames, '_last_sweep', [])
    os.utime(prepared[-1], (old, old))
    video_frames.forget_old_prepared_soon(root)
    for thread in threading.enumerate():
        if thread.name == 'video-prepared-sweep':
            thread.join()
    assert not list(folder.glob('.*'))
    (folder / '.video-again').mkdir()
    os.utime(folder / '.video-again', (old, old))
    video_frames.forget_old_prepared_soon(root)
    assert (folder / '.video-again').is_dir()


def test_a_later_read_of_the_same_frame_keeps_one_row(drive):
    root, video, rows, server = drive
    status, _response, data = server.post_frame(ROUTE, '0/1025', body([BOX]))
    assert status == 200, data
    folder = root / 'derived/frames' / ROUTE / '0'
    read = json.loads((folder / video_frames.VIDEO_MANIFEST).read_text())
    shutil.copy(read['image_path'], folder / 'read_00001025.png')
    read.update(image_path=str(folder / 'read_00001025.png'), comma_read_kph=50.0)
    (folder / 'sign_reads.jsonl').write_text(json.dumps(read) + '\n')
    found = [row for row in discover_frames(root, ROUTE) if row['frame_key'] == '0/1025']
    assert len(found) == 1 and found[0]['comma_read_kph'] == 50.0


def test_apply_previous_and_fit_box_work_on_a_frame_of_the_video_not_saved(drive):
    """A frame between the seconds not saved yet: Apply previous carries the boxes of the frame of the report just
    before it (it has no proposals), each followed onto its sign in its picture (carry_track), and Fit box fits on its
    picture. Neither saves anything."""
    root, video, rows, server = drive
    manifest = root / 'derived/frames' / ROUTE / '0' / video_frames.VIDEO_MANIFEST

    def apply(key):
        status, _response, data = server.request('POST', f'/api/route/{ROUTE}/frame/{key}/apply-previous',
                                                 body={'run_id': RUN_ID, 'detections': []})
        return status, json_of(data)

    status, answer = apply('0/1025')
    assert status == 409 and answer['error'] == 'previous_not_confirmed' and '0/1' in answer['message']
    assert server.post_frame(ROUTE, '0/1', body([BOX]))[0] == 200
    status, answer = apply('0/1025')
    assert status == 200, answer
    assert answer['source_frame_key'] == '0/1' and answer['save'] == {'run_id': RUN_ID, 'base_revision': 0}
    carried = answer['detections']
    assert len(carried) == 1 and carried[0]['value_kph'] == 50 and answer['followed'] == 1
    assert all(abs(a - b) < 1 for a, b in zip(carried[0]['bbox_xyxy'], BOX['bbox_xyxy']))
    assert 'check_position' not in carried[0] and not answer['to_check']
    # In the next second the frame just before is its own picture, 0/2, not confirmed: the source is the last frame
    # confirmed before it, 0/1 (14 Nov 2026; before, the frame just before had to be confirmed).
    status, answer = apply('0/1041')
    assert status == 200 and answer['source_frame_key'] == '0/1', answer
    # Not such a frame: the picture at 1 a second under another key, past the end of the video.
    assert apply('0/1020')[0] == 404 and apply('0/1060')[0] == 404
    fit = f'/api/route/{ROUTE}/frame/0/1025/fit'
    status, _response, data = server.request('POST', fit, body={'run_id': RUN_ID, 'boxes': [BOX['bbox_xyxy']]})
    assert status == 200, data
    assert len(json_of(data)['boxes']) == 1
    status, _response, data = server.request('POST', fit, body={'run_id': NEW_RUN, 'boxes': [BOX['bbox_xyxy']]})
    assert status == 409 and json_of(data)['error'] == 'stale_report'
    assert not manifest.exists() and not list(manifest.parent.glob('video_*.png'))
    # The page knows the segments with their video, and how long it is once a frame of it is saved.
    status, _response, data = server.request('GET', f'/api/route/{ROUTE}/page')
    assert status == 200 and json_of(data)['videos'] == {'0': None}
    # A box still to check is refused, as on any frame; checked ("Here", or Fit box on its sign) it is saved.
    assert server.post_frame(ROUTE, '0/1025', body([dict(carried[0], check_position=True)]))[0] == 400
    assert not manifest.exists()
    checked = [{name: value for name, value in box.items() if name != 'check_position'} for box in carried]
    assert server.post_frame(ROUTE, '0/1025', body(checked))[0] == 200
    status, answer = apply('0/1026')
    assert status == 200 and answer['source_frame_key'] == '0/1025'
    status, _response, data = server.request('GET', f'/api/route/{ROUTE}/page')
    assert status == 200 and json_of(data)['videos'] == {'0': 60}


def test_apply_previous_takes_the_boxes_of_a_frame_before_still_on_its_way(drive):
    """Confirm goes on before its save comes back: A on the next frame sends the boxes of the frame before with the
    request, and the server carries them from there, a frame of the video it has not saved yet included. Only a frame
    between the server's frame before and this one is taken."""
    root, video, rows, server = drive
    assert server.post_frame(ROUTE, '0/1', body([BOX]))[0] == 200

    def apply(key, previous):
        status, _response, data = server.request('POST', f'/api/route/{ROUTE}/frame/{key}/apply-previous',
                                                 body={'run_id': RUN_ID, 'detections': [], 'previous': previous})
        return status, json_of(data)

    moved = dict(BOX, value_kph=60)
    status, answer = apply('0/1026', {'frame_key': '0/1025', 'detections': [moved]})
    assert status == 200, answer
    assert answer['source_frame_key'] == '0/1025' and answer['detections'][0]['value_kph'] == 60
    # Not before this frame.
    assert apply('0/1026', {'frame_key': '0/1030', 'detections': [moved]})[0] == 409
    # Older than the confirmed frame the server has (0/1): that one, nearer, is the source (14 Nov 2026).
    status, answer = apply('0/1026', {'frame_key': '0/0', 'detections': [moved]})
    assert status == 200 and answer['source_frame_key'] == '0/1' and answer['detections'][0]['value_kph'] == 50
    assert apply('0/1026', {'frame_key': '0/1025', 'detections': 'boxes'})[0] == 400
    manifest = root / 'derived/frames' / ROUTE / '0' / video_frames.VIDEO_MANIFEST
    assert not manifest.exists()


def test_a_frame_of_any_second_of_the_video_is_saved(drive):
    """The whole drive is in Sign review: a frame of the video between the seconds is saved in every second of its
    segment, a second whose picture at 1 a second is missing too."""
    root, _video, _rows, server = drive
    pictures = root / 'derived/frames' / ROUTE / '0' / 'frames.jsonl'
    kept = [line for line in pictures.read_text().splitlines() if json.loads(line)['output_index'] == 1]
    pictures.write_text(kept[0] + '\n')
    for second in (0, 1, 2):
        assert server.post_frame(ROUTE, f'0/{1000 + second * 20 + 5}', body([BOX]))[0] == 200


def test_no_more_than_max_decodes_ffmpeg_at_once(drive, monkeypatch):
    # Each decode at full size takes hundreds of MB: a page asking for many GOPs at once waits for a free slot (#60).
    root, _video, _rows, _server = drive
    import threading
    monkeypatch.setattr(video_frames, '_decode_slots', threading.BoundedSemaphore(1))
    real = video_frames.subprocess.Popen
    running, most, guard = [0], [0], threading.Lock()

    class Counted(real):
        def __init__(self, command, **kwargs):
            super().__init__(command, **kwargs)
            self.counted = 'ffmpeg' in command
            if self.counted:
                with guard:
                    running[0] += 1
                    most[0] = max(most[0], running[0])

        def communicate(self, *args, **kwargs):
            try:
                return super().communicate(*args, **kwargs)
            finally:
                if self.counted:
                    with guard:
                        running[0] -= 1
    monkeypatch.setattr(video_frames.subprocess, 'Popen', Counted)
    threads = [threading.Thread(target=video_frames.preview, args=(root, ROUTE, 0, frame)) for frame in (5, 25, 45)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert video_frames.settle() and most[0] == 1 and cached(root) == list(range(0, 60))


def test_a_frame_given_has_the_next_gops_decoded_ahead_and_the_minutes_index_kept_on_disk(drive, monkeypatch):
    """Fast steps outran one GOP decoded ahead: a frame given has the next AHEAD_GOPS GOPs decoded, and stepping back the
    ones before. The index of a minute (ffprobe over the whole file) is kept on disk, so a server started again does not
    read it again."""
    root, video, _rows, _server = drive
    started = []
    real = video_frames._decode
    monkeypatch.setattr(video_frames, '_decode', lambda *args, **kwargs: started.append(args[3]) or real(*args, **kwargs))
    video_frames.prune(root, budget=0)
    assert video_frames.preview(root, ROUTE, 0, 12).is_file() and video_frames.settle()
    started.clear()
    video_frames.read_ahead(root, ROUTE, 0, 12)
    assert video_frames.settle()
    assert sorted(started) == [20, 40] and cached(root) == list(range(60))
    # Stepping back from 45 to 44: the GOPs before 40 are in the cache already, none is decoded again.
    video_frames.read_ahead(root, ROUTE, 0, 45)
    started.clear()
    video_frames.read_ahead(root, ROUTE, 0, 44)
    assert video_frames.settle() and started == []
    kept = video_frames._index_file(str(video))
    assert kept is not None and kept.is_file()
    probed = []
    monkeypatch.setattr(video_frames, '_probe_index', lambda path: probed.append(path))
    video_frames._index.cache_clear()
    assert video_frames.frame_count(video) == 60 and probed == []
