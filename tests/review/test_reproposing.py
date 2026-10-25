"""A drive Propose again works on is read only (nnslr_tools.reproposing): every save on it is refused, its review
page shows the new proposals made so far (the drive at work) or its report (a drive waiting its turn), and says so."""
from __future__ import annotations

import json

import pytest

from nnslr_tools import reproposing
from nnslr_tools.review_db import ReviewRejected
from review_helpers import OTHER_ROUTE, ROUTE, RUN_ID, RunningServer, json_of, make_report
from test_route_list_and_page import payload_of

STARTED = '2026-10-25T13:12:36Z'
NEW_RUN = '20261025T131241Z-3b62cf9b'


@pytest.fixture
def batch(tmp_path, monkeypatch):
    """Propose again (Qwen3-VL) on ROUTE (one frame proposed, a second being written) and OTHER_ROUTE (waiting)."""
    rows = make_report(tmp_path, count=3)
    make_report(tmp_path, route=OTHER_ROUTE, count=2)
    (tmp_path / 'logs/prepare').mkdir(parents=True)
    (tmp_path / 'logs/prepare/batch.json').write_text(json.dumps({'started': STARTED, 'routes': [ROUTE, OTHER_ROUTE]}))
    pending = tmp_path / 'derived/preannotations' / ROUTE / ('.pending-' + NEW_RUN)
    (pending / 'images').mkdir(parents=True)
    (pending / 'images/00000000.jpg').write_bytes(b'new preview')
    fresh = dict(rows[0], run_id=NEW_RUN, detections=[])
    (pending / 'preannotations.jsonl').write_text(json.dumps(fresh) + '\n{"frame_key": "0/1", "rout')
    (pending / 'request.json').write_text(json.dumps({'model': {'proposer': 'qwen'}}))
    monkeypatch.setattr(reproposing, '_again_running', lambda: True)
    reproposing.reset_cache()
    yield tmp_path, rows, monkeypatch
    reproposing.reset_cache()


def test_the_drives_of_a_running_batch_are_read_only_until_each_is_done(batch):
    root, _rows, monkeypatch = batch
    found = reproposing.busy(root)
    assert found[ROUTE]['phase'] == 'proposing again' and found[ROUTE]['done'] == 1
    assert found[ROUTE]['proposer'] == 'Qwen3-VL'
    assert found[OTHER_ROUTE]['phase'] == 'waiting' and found[OTHER_ROUTE]['ahead'] == 1
    # The drive done (its new report in) can be reviewed; the other still waits.
    base = root / 'derived/preannotations' / ROUTE
    (base / NEW_RUN).mkdir()
    (base / 'latest.json').write_text(json.dumps({'run_id': NEW_RUN}))
    reproposing.reset_cache()
    assert set(reproposing.busy(root)) == {OTHER_ROUTE}
    # Propose again not running (ended, or stopped): nothing is read only, whatever batch.json says.
    monkeypatch.setattr(reproposing, '_again_running', lambda: False)
    reproposing.reset_cache()
    assert reproposing.busy(root) == {}


def test_a_batch_of_new_drives_is_not_propose_again(tmp_path, monkeypatch):
    make_report(tmp_path, count=1)
    (tmp_path / 'logs/prepare').mkdir(parents=True)
    (tmp_path / 'logs/prepare/batch.json').write_text(json.dumps({'started': STARTED, 'routes': [ROUTE, OTHER_ROUTE]}))
    monkeypatch.setattr(reproposing, '_again_running', lambda: True)
    reproposing.reset_cache()
    assert reproposing.busy(tmp_path) == {}


def test_every_save_on_a_busy_drive_is_refused(batch):
    root, rows, _monkeypatch = batch
    server = RunningServer(root)
    try:
        for route, key in ((ROUTE, '0/0'), (OTHER_ROUTE, '0/0')):
            body = {'run_id': RUN_ID, 'base_revision': 0, 'reviewer': 'Claude', 'reviewer_kind': 'ai',
                    'confirmed': True, 'detections': []}
            with pytest.raises(ReviewRejected) as refused:
                server.app.save_review(route, key, body, client='test', channel='rest')
            assert refused.value.status == 409 and refused.value.reason == 'drive_busy'
        assert 'read only' in str(refused.value.message)
    finally:
        server.close()


def test_the_review_page_of_the_drive_at_work_shows_its_new_proposals_read_only(batch):
    root, _rows, monkeypatch = batch
    server = RunningServer(root)
    try:
        payload = payload_of(server.request('GET', f'/route/{ROUTE}/')[2].decode())
        # Only the frame written whole; the run id of the new proposals, nothing to save on.
        assert [row['frame_key'] for row in payload['frames']] == ['0/0'] and payload['frames'][0]['detections'] == []
        assert payload['run_id'] == '.pending-' + NEW_RUN and payload['proposer'] == 'qwen'
        assert payload['busy']['phase'] == 'proposing again' and payload['busy']['done'] == 1
        assert payload['busy']['total'] == 3
        assert server.request('GET', f'/route/{ROUTE}/images/00000000.jpg')[2] == b'new preview'
        reviews = json_of(server.request('GET', f'/api/route/{ROUTE}/reviews')[2])
        assert reviews['run_id'] == '.pending-' + NEW_RUN and reviews['reviews'] == []
        # A drive waiting its turn shows its report, read only.
        waiting = payload_of(server.request('GET', f'/route/{OTHER_ROUTE}/')[2].decode())
        assert waiting['run_id'] == RUN_ID and len(waiting['frames']) == 2
        assert waiting['busy']['phase'] == 'waiting' and waiting['busy']['ahead'] == 1
        assert json_of(server.request('GET', f'/api/route/{OTHER_ROUTE}/busy')[2])['busy']['phase'] == 'waiting'
        # Once Propose again ends, the page asks again and the drive is free.
        monkeypatch.setattr(reproposing, '_again_running', lambda: False)
        reproposing.reset_cache()
        assert json_of(server.request('GET', f'/api/route/{ROUTE}/busy')[2])['busy'] is None
        assert payload_of(server.request('GET', f'/route/{ROUTE}/')[2].decode())['busy'] is None
    finally:
        server.close()
