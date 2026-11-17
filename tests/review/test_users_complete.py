"""Road users are no longer boxed in Sign review (17 Nov 2026, #113): the "Road users complete" tick and the boxes of a
person, a bicycle, a car... went with UserNet. A save (page, /api/v1, MCP) with a road user's box is refused, and so is
a tick; a page loaded before still sends the tick unticked, which is taken and dropped. Reviews saved from 12 to
17 Nov 2026 with them still read, export and import as before."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from nnslr_tools.annotations import reviewed_frame_annotations, validate_dataset
from nnslr_tools.review_db import (ReviewRejected, ReviewStore, build_reviewed_frame, db_path, import_from_db,
                                   save_frame_review)
from nnslr_tools.review_mcp import ReviewTools
from nnslr_tools.review_service import ReviewService, review_schema
from nnslr_tools.sign_types import BOXED_SIGN_TYPES, ROAD_USERS
from review_helpers import ROUTE, RUN_ID, TOKEN, RunningServer, json_of, make_report, page_body

WEB = Path(__file__).resolve().parents[2] / 'web/src/review'
AUTH = {'Authorization': 'Bearer ' + TOKEN}
ROAD_USER_REFUSED = 'Box 1: road users are no longer boxed (#113): delete this box.'
TICK_REFUSED = 'Road users complete is gone (#113): reload the page.'


@pytest.fixture
def server(tmp_path, monkeypatch):
    monkeypatch.setenv('NNSLR_REVIEW_TOKEN', TOKEN)
    rows = make_report(tmp_path, count=3)
    running = RunningServer(tmp_path)
    yield running, {row['frame_key']: row for row in rows}
    running.close()


def _payload(running, key):
    return {review['frame_key']: review for review in running.app.store.reviews(ROUTE, RUN_ID)}[key]['payload']


def _road_user(kind='road_user_car'):
    return {'bbox_xyxy': [2, 2, 10, 20], 'family': 'other_sign', 'sign_type': kind, 'value_kph': None,
            'applicability': 'unresolved', 'detection_id': 'manual-1'}


def _rest_body(**extra):
    return {'run_id': RUN_ID, 'base_revision': 0, 'reviewer': 'Claude', 'reviewer_kind': 'ai', 'confirmed': True,
            'detections': [], **extra}


def test_a_road_user_box_is_refused_on_the_page_rest_and_mcp(server, tmp_path):
    running, rows = server
    for kind in ROAD_USERS:
        body = page_body(rows['0/1'])
        body['detections'] = [_road_user(kind)]
        status, response, data = running.post_frame(ROUTE, '0/1', body)
        assert status == 400, kind
        assert (json_of(data)['error'], json_of(data)['message']) == ('invalid_sign_type', ROAD_USER_REFUSED)
    status, response, data = running.request('POST', f'/api/v1/routes/{ROUTE}/frames/0/1/review',
                                             body=_rest_body(detections=[_road_user()]), headers=AUTH)
    assert status == 400 and (json_of(data)['error'], json_of(data)['message']) == ('invalid_sign_type',
                                                                                    ROAD_USER_REFUSED)
    tools = ReviewTools(ReviewService(tmp_path, ReviewStore(db_path(tmp_path))), client='test')
    with pytest.raises(ReviewRejected) as caught:
        tools.save_frame_review(ROUTE, '0/1', run_id=RUN_ID, base_revision=0, reviewer='Claude Code',
                                detections=[_road_user('road_user_person')])
    assert (caught.value.status, caught.value.reason, caught.value.message) == (400, 'invalid_sign_type',
                                                                                ROAD_USER_REFUSED)
    # Nothing was saved, and a box of any other kind still is.
    assert running.app.store.reviews(ROUTE, RUN_ID) == []
    body = page_body(rows['0/1'])
    body['detections'] = [{**_road_user(), 'sign_type': 'give_way'}]
    status, response, data = running.post_frame(ROUTE, '0/1', body)
    assert status == 200, data


def test_a_box_that_keeps_a_proposal_of_a_road_user_is_refused_too(tmp_path):
    """A box that only names its proposal takes its kind: a classifier trained on the road users saved before may
    still propose one."""
    rows = make_report(tmp_path)
    store = ReviewStore(db_path(tmp_path))
    row = copy.deepcopy(rows[0])
    row['detections'][0].update(family='other_sign', value_kph=None, sign_type='road_user_car')
    body = {'run_id': RUN_ID, 'reviewer': 'Cristian', 'confirmed': True,
            'detections': [{'detection_id': '0/0/0', 'applicability': 'own_road'}]}
    with pytest.raises(ReviewRejected) as caught:
        save_frame_review(store, tmp_path, route_id=ROUTE, run_id=RUN_ID, report_row=row, body=body, client='test')
    assert (caught.value.reason, caught.value.message) == ('invalid_sign_type', ROAD_USER_REFUSED)
    # The reviewer gives it another kind: saved.
    body['detections'][0].update(family='other_sign', sign_type='other')
    saved = save_frame_review(store, tmp_path, route_id=ROUTE, run_id=RUN_ID, report_row=row, body=body,
                              client='test')
    assert saved['payload']['detections'][0]['sign_type'] == 'other'


def test_the_tick_unticked_is_dropped_and_ticked_is_refused(server, tmp_path):
    running, rows = server
    # A page loaded before #113 sends it false; a page of today sends nothing. Neither is stored nor read back.
    for key, sent in (('0/0', False), ('0/1', None)):
        body = page_body(rows[key])
        if sent is not None:
            body['users_complete'] = sent
        status, response, data = running.post_frame(ROUTE, key, body)
        assert status == 200, data
        assert 'users_complete' not in json_of(data)['review'] and 'users_complete' not in _payload(running, key)
    for sent in (True, 'yes', 1):
        body = page_body(rows['0/2'])
        body['users_complete'] = sent
        status, response, data = running.post_frame(ROUTE, '0/2', body)
        assert status == 400 and (json_of(data)['error'], json_of(data)['message']) == ('invalid_schema',
                                                                                        TICK_REFUSED), sent
    status, response, data = running.request('POST', f'/api/v1/routes/{ROUTE}/frames/0/2/review',
                                             body=_rest_body(users_complete=True), headers=AUTH)
    assert status == 400 and (json_of(data)['error'], json_of(data)['message']) == ('invalid_schema', TICK_REFUSED)
    status, response, data = running.request('POST', f'/api/v1/routes/{ROUTE}/frames/0/2/review',
                                             body=_rest_body(users_complete=False), headers=AUTH)
    assert status == 200 and 'users_complete' not in json_of(data)['review']
    # MCP has no such parameter any more.
    tools = ReviewTools(ReviewService(tmp_path, ReviewStore(db_path(tmp_path))), client='test')
    with pytest.raises(TypeError):
        tools.save_frame_review(ROUTE, '0/0', run_id=RUN_ID, base_revision=1, reviewer='Claude Code',
                                detections=[], users_complete=True)


def test_an_old_review_with_a_road_user_and_the_tick_still_reads_exports_and_imports(tmp_path):
    rows = make_report(tmp_path)
    store = ReviewStore(db_path(tmp_path))
    service = ReviewService(tmp_path, store)
    # 0/0 as a page of 12-17 Nov 2026 saved it: a car boxed beside the speed sign, the frame ticked.
    body = page_body(rows[0])
    body['detections'].append(_road_user())
    payload = build_reviewed_frame(rows[0], body, '2026-11-13T09:00:00.000Z')
    payload['users_complete'] = True
    store.save_review(route_id=ROUTE, run_id=RUN_ID, frame_key='0/0', segment_index=0, output_index=0,
                      payload=payload, reviewer='Cristian', client='test', base_revision=0)

    review = service.get_frame(ROUTE, '0/0')['review']
    assert 'users_complete' not in review
    assert [(d['family'], d.get('sign_type')) for d in review['detections']] == [
        ('maximum_speed', None), ('other_sign', 'road_user_car')]

    exported = reviewed_frame_annotations(payload, '0' * 64, dataset_kind='gold', name='test')
    assert sorted(row.get('sign_type') or '' for row in exported) == ['', 'road_user_car']
    assert all(row['users_complete'] is True for row in exported)
    assert validate_dataset(exported, tmp_path, dataset_kind='gold')['valid'] is True

    output = tmp_path / 'annotations/objects.jsonl'
    report = import_from_db(tmp_path, output, store=store)
    assert report['valid'] and report['imported_frame_count'] == 1
    imported = [json.loads(line) for line in output.read_text().splitlines()]
    assert sorted(row.get('sign_type') or '' for row in imported) == ['', 'road_user_car']
    assert all(row['users_complete'] is True for row in imported)


def test_an_old_review_is_read_by_the_page_without_the_tick(server, tmp_path):
    running, rows = server
    body = page_body(rows['0/1'])
    body['detections'] = [_road_user()]
    payload = build_reviewed_frame(rows['0/1'], body, '2026-11-13T09:00:00.000Z')
    payload['users_complete'] = True
    running.app.store.save_review(route_id=ROUTE, run_id=RUN_ID, frame_key='0/1', segment_index=0, output_index=1,
                                  payload=payload, reviewer='Cristian', client='test', base_revision=0)
    status, response, data = running.request('GET', f'/api/route/{ROUTE}/reviews')
    assert status == 200
    (review,) = json_of(data)['reviews']
    assert 'users_complete' not in review and review['detections'][0]['sign_type'] == 'road_user_car'
    # Saved again as it reads, the old box is refused: the reviewer deletes it.
    body = page_body(rows['0/1'], base_revision=1)
    body['detections'] = review['detections']
    status, response, data = running.post_frame(ROUTE, '0/1', body)
    assert status == 400 and json_of(data)['message'] == ROAD_USER_REFUSED
    body['detections'] = []
    status, response, data = running.post_frame(ROUTE, '0/1', body)
    assert status == 200, data


def test_apply_previous_leaves_an_old_road_user_there_and_says_so(tmp_path):
    rows = make_report(tmp_path)
    store = ReviewStore(db_path(tmp_path))
    service = ReviewService(tmp_path, store)
    body = page_body(rows[0])
    body['detections'].append(_road_user())
    payload = build_reviewed_frame(rows[0], body, '2026-11-13T09:00:00.000Z')
    store.save_review(route_id=ROUTE, run_id=RUN_ID, frame_key='0/0', segment_index=0, output_index=0,
                      payload=payload, reviewer='Cristian', client='test', base_revision=0)
    carried = service.apply_previous(ROUTE, '0/1', {'run_id': RUN_ID})
    assert carried['source_frame_key'] == '0/0' and carried['road_users'] == 1
    assert [d['family'] for d in carried['detections']] == ['maximum_speed']
    assert carried['message'].endswith('; 1 road user left there: road users are no longer boxed (#113).')
    assert 'users_complete' not in carried


def test_the_schema_offers_no_road_user_and_no_tick():
    schema = review_schema()
    assert 'users_complete' not in schema['save_body']
    assert 'road_user_' not in str(schema)
    assert all(kind in str(schema) for kind in BOXED_SIGN_TYPES)
    assert not set(BOXED_SIGN_TYPES) & set(ROAD_USERS)


def test_the_page_has_no_road_user_keys_and_no_tick():
    keyboard = (WEB / 'use-keyboard.ts').read_text()
    assert 'road_user' not in keyboard and 'toggleUsersComplete' not in keyboard and "case 'j':" not in keyboard
    assert 'road_user' not in (WEB / 'BoxEditor.vue').read_text()
    help_keys = (WEB / 'ConfirmArea.vue').read_text()
    assert not any(f'<kbd>{key}</kbd>' in help_keys for key in 'WYKJ')
    assert 'usersComplete' not in (WEB / 'Inspector.vue').read_text()
    for name in ('server.ts', 'confirm.ts', 'frames.ts', 'export.ts', 'types.ts', 'edits.ts', 'store.ts'):
        assert 'users_complete' not in (WEB / name).read_text() and 'drawKind' not in (WEB / name).read_text(), name
    # An old road user's box still shows its name, and takes no plate of distance.
    catalog = (WEB / 'catalog.ts').read_text()
    assert "label: 'Road user'" not in catalog and 'const ROAD_USER_NAMES' in catalog
    assert '!isRoadUser(d)' in catalog
