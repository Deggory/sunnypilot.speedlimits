import json
import os
import time

from nnslr_tools.proposal_runs import prune_old_runs, prune_route

ROUTE = '00000001--0123456789'
OLD, MID, LATEST, LATER = ('20261101T100000Z-aaaaaaaa', '20261105T100000Z-bbbbbbbb',
                           '20261110T100000Z-cccccccc', '20261112T100000Z-dddddddd')


def _run(base, run_id, images=2):
    folder = base / run_id
    (folder / 'images').mkdir(parents=True)
    for i in range(images):
        (folder / 'images' / f'{i:08d}.jpg').write_bytes(b'x' * 1000)
    for name in ('run.json', 'preannotations.jsonl', 'request.json', 'index.html'):
        (folder / name).write_text('{}')
    return folder


def _route(tmp_path):
    base = tmp_path / 'derived/preannotations' / ROUTE
    for run_id in (OLD, MID, LATEST):
        _run(base, run_id)
    (base / 'latest.json').write_text(json.dumps({'run_id': LATEST}))
    return base


def _age(path, seconds):
    then = time.time() - seconds
    for folder, dirs, files in os.walk(path):
        for name in dirs + files:
            os.utime(os.path.join(folder, name), (then, then))
    os.utime(path, (then, then))


def test_the_pictures_of_earlier_runs_go_and_their_records_and_the_latest_run_stay(tmp_path):
    base = _route(tmp_path)
    result = prune_old_runs(tmp_path)
    assert result['runs_with_images_removed'] == 2 and result['bytes'] == 4000
    for run_id in (OLD, MID):
        assert not (base / run_id / 'images').exists()
        assert sorted(p.name for p in (base / run_id).iterdir()) == [
            'index.html', 'preannotations.jsonl', 'request.json', 'run.json']
    assert len(list((base / LATEST / 'images').iterdir())) == 2
    assert prune_old_runs(tmp_path)['bytes'] == 0


def test_a_dry_run_counts_and_removes_nothing(tmp_path):
    base = _route(tmp_path)
    result = prune_old_runs(tmp_path, ROUTE, dry_run=True)
    assert result['dry_run'] and result['runs_with_images_removed'] == 2
    assert (base / OLD / 'images').is_dir() and (base / MID / 'images').is_dir()


def test_a_stopped_run_goes_and_a_run_still_writing_or_newer_than_the_latest_stays(tmp_path):
    base = _route(tmp_path)
    stopped = _run(base, '.pending-' + MID)
    quiet_but_new = _run(base, '.pending-' + LATER)
    writing = _run(base, '.pending-20261109T100000Z-eeeeeeee')
    _age(stopped, 7200)
    _age(quiet_but_new, 7200)
    result = prune_route(base)
    assert result['pending_removed'] == ['.pending-' + MID]
    assert not stopped.exists() and quiet_but_new.is_dir() and writing.is_dir()


def test_a_drive_without_a_valid_latest_run_is_left_as_it_is(tmp_path):
    base = _route(tmp_path)
    (base / 'latest.json').write_text(json.dumps({'run_id': '20261120T100000Z-ffffffff'}))
    assert prune_route(base)['images_removed'] == []
    (base / 'latest.json').unlink()
    assert prune_old_runs(tmp_path)['routes'] == []
    assert (base / OLD / 'images').is_dir()
