"""The country of a frame (from its GPS position, nnslr_tools.gps) and the kind of an "Other sign"
(nnslr_tools.sign_types): proposed, saved, refused when wrong, the same list on the page and on the server."""
from __future__ import annotations

import gzip
import json
import re
from pathlib import Path

import pytest

from nnslr_tools.gps import (DETAILED_BORDERS, GpsError, country_names, country_of, frame_country, frame_position,
                             positions_path, recountry_positions)
from nnslr_tools.sign_types import SIGN_TYPES
from review_helpers import ROUTE, RUN_ID, TOKEN, RunningServer, json_of, make_report, page_body

CATALOG = Path(__file__).resolve().parents[2] / 'web/src/review/catalog.ts'


@pytest.mark.parametrize('latitude, longitude, code', [
    (47.37, 8.54, 'CH'), (46.20, 6.14, 'CH'), (45.46, 9.19, 'IT'), (52.23, 21.01, 'PL'), (48.14, 11.58, 'DE'),
    (48.86, 2.35, 'FR'), (47.48, 8.05, 'CH'), (40.0, -40.0, None), (None, None, None)])
def test_the_country_of_a_position(latitude, longitude, code):
    assert country_of(latitude, longitude) == code



def test_a_frame_outside_europe_takes_the_country_of_the_world_borders(tmp_path):
    # A drive in Illinois: no country in the European borders, the United States in the world's, with its name.
    assert country_of(41.57, -87.48) is None and frame_country(41.57, -87.48) == 'US'
    assert country_names()['US'] == 'United States of America' and country_names()['CH'] == 'Switzerland'
    # At sea, and in Europe where the European borders say nothing (the finer file of the data root), still none.
    assert frame_country(40.0, -40.0) is None and frame_country(None, None) is None
    assert frame_country(47.37, 8.54) == 'CH'
    target = tmp_path / DETAILED_BORDERS
    target.parent.mkdir(parents=True)
    target.write_bytes(gzip.compress(json.dumps({'countries': [_square('DE', 7.0, 47.55, 9.0, 48.0)]}).encode()))
    assert frame_country(48.86, 2.35, tmp_path) is None and frame_country(41.57, -87.48, tmp_path) == 'US'

    # A route read before: its frames get the country now, and a review saved then without one takes it at import.
    from nnslr_tools.review_db import _with_position
    lines = [{'frame_key': '0/0', 'segment_index': 0, 'output_index': 0, 'latitude': 41.57, 'longitude': -87.48,
              'country': None, 'country_from': None},
             {'frame_key': '0/1', 'segment_index': 0, 'output_index': 1, 'latitude': None, 'longitude': None,
              'country': None, 'country_from': None}]
    path = positions_path(tmp_path, ROUTE)
    path.parent.mkdir(parents=True)
    path.write_text(''.join(json.dumps(line) + '\n' for line in lines))
    assert recountry_positions(tmp_path, ROUTE)['changed'] == {'None->US': 2}
    saved = {'route_id': ROUTE, 'segment_index': 0, 'output_index': 1, 'gps': None, 'country': None}
    assert _with_position(saved, tmp_path, {})['country'] == 'US'
    assert _with_position({**saved, 'country': 'CA'}, tmp_path, {})['country'] == 'CA'


def _square(code, west, south, east, north):
    ring = [[west, south], [east, south], [east, north], [west, north], [west, south]]
    return {'code': code, 'name': code, 'bbox': [west, south, east, north], 'polygons': [[ring]]}


def test_finer_borders_of_the_data_root_take_the_place_of_the_shipped_ones(tmp_path):
    # Bad Säckingen: Germany, though the shipped Natural Earth borders put it in Switzerland.
    assert country_of(47.5536, 7.9469) == 'CH'
    target = tmp_path / DETAILED_BORDERS
    target.parent.mkdir(parents=True)
    target.write_bytes(gzip.compress(json.dumps({'countries': [
        _square('DE', 7.0, 47.55, 9.0, 48.0), _square('CH', 7.0, 46.0, 9.0, 47.55)]}).encode()))
    assert country_of(47.5536, 7.9469, tmp_path) == 'DE' and country_of(47.5464, 7.9214, tmp_path) == 'CH'
    # Elsewhere the finer file knows nothing: no country (it covers Europe as the shipped one does).
    assert country_of(48.86, 2.35, tmp_path) is None
    # Without the file, the shipped borders.
    assert country_of(47.5536, 7.9469, tmp_path / 'other') == 'CH'

    # The routes already read get their country again, without their logs; a frame without a fix takes the
    # country of the nearest one with a fix.
    lines = [{'frame_key': '0/0', 'segment_index': 0, 'output_index': 0, 'latitude': 47.5536, 'longitude': 7.9469,
              'country': 'CH', 'country_from': 'gps'},
             {'frame_key': '0/1', 'segment_index': 0, 'output_index': 1, 'latitude': None, 'longitude': None,
              'country': 'CH', 'country_from': 'nearest_fix'},
             {'frame_key': '0/2', 'segment_index': 0, 'output_index': 2, 'latitude': 47.5464, 'longitude': 7.9214,
              'country': 'CH', 'country_from': 'gps'}]
    path = positions_path(tmp_path, ROUTE)
    path.parent.mkdir(parents=True)
    path.write_text(''.join(json.dumps(line) + '\n' for line in lines))
    result = recountry_positions(tmp_path, ROUTE)
    assert result['changed'] == {'CH->DE': 2} and result['countries'] == {'DE': 1, 'CH': 1}
    again = [json.loads(line) for line in path.read_text().splitlines()]
    assert [(line['country'], line['country_from']) for line in again] == [
        ('DE', 'gps'), ('DE', 'nearest_fix'), ('CH', 'gps')]
    assert again[0]['latitude'] == 47.5536
    with pytest.raises(GpsError):
        recountry_positions(tmp_path, '00000002--abcdef1234')


def test_the_position_of_a_frame_is_the_nearest_fix():
    log = {'first_frame_mono': 10_000_000_000,
           'fixes': [[10_000_000_000 + k * 100_000_000, 47.0 + k / 1000, 8.0, 10.0, 90.0] for k in range(50)]}
    assert frame_position(log, 2)['latitude'] == 47.02
    # Beyond the last fix by more than GPS_MAX_GAP_S: no position.
    assert frame_position(log, 9) is None
    assert frame_position({'first_frame_mono': None, 'fixes': []}, 0) is None


def test_the_page_offers_the_kinds_of_the_server():
    text = CATALOG.read_text()
    block = text[text.index('export const SIGN_KIND_GROUPS'):text.index('export function signKindOf')]
    assert sorted(re.findall(r"\{id: '([a-z_]+)', label:", block)) == sorted(SIGN_TYPES)


@pytest.fixture
def server(tmp_path, monkeypatch):
    monkeypatch.setenv('NNSLR_REVIEW_TOKEN', TOKEN)
    rows = make_report(tmp_path, count=3)
    lines = [{'frame_key': '0/0', 'segment_index': 0, 'output_index': 0, 'latitude': 47.48, 'longitude': 8.05,
              'speed_mps': 13.6, 'bearing_deg': 14.4, 'country': 'CH', 'country_from': 'gps'},
             {'frame_key': '0/1', 'segment_index': 0, 'output_index': 1, 'latitude': None, 'longitude': None,
              'speed_mps': None, 'bearing_deg': None, 'country': 'CH', 'country_from': 'nearest_fix'}]
    target = positions_path(tmp_path, ROUTE)
    target.parent.mkdir(parents=True)
    target.write_text(''.join(json.dumps(line) + '\n' for line in lines))
    running = RunningServer(tmp_path)
    yield running, {row['frame_key']: row for row in rows}
    running.close()


def test_frames_carry_their_position_and_proposed_country(server):
    running, rows = server
    status, response, data = running.request('GET', f'/api/v1/routes/{ROUTE}/frames/0/0',
                                             headers={'Authorization': 'Bearer ' + TOKEN})
    frame = json_of(data)
    assert status == 200 and frame['gps']['latitude'] == 47.48 and frame['country_proposed'] == 'CH'
    status, response, data = running.request('GET', f'/route/{ROUTE}/')
    assert status == 200 and b'"country_proposed": "CH"' in data and b'"PL": "Poland"' in data


def test_the_country_is_saved_proposed_or_picked(server):
    running, rows = server
    status, response, data = running.post_frame(ROUTE, '0/0', page_body(rows['0/0']))
    assert status == 200 and json_of(data)['review']['country'] == 'CH'
    body = page_body(rows['0/1'])
    body['country'] = 'IT'
    status, response, data = running.post_frame(ROUTE, '0/1', body)
    assert status == 200 and json_of(data)['review']['country'] == 'IT'
    body = page_body(rows['0/2'])
    body['country'] = 'Switzerland'
    status, response, data = running.post_frame(ROUTE, '0/2', body)
    assert status == 400 and json_of(data)['error'] == 'invalid_country'
    # No position read for 0/2: no country unless the reviewer gives one.
    body['country'] = None
    status, response, data = running.post_frame(ROUTE, '0/2', body)
    assert status == 200 and json_of(data)['review']['country'] is None


def test_a_review_saved_without_a_position_takes_the_one_its_frame_has_now(tmp_path):
    """A frame between the seconds saved before the positions of its route had it: gps None, with or without a
    country. The import gives it the position its frame has now; one saved with the review stays (#104)."""
    from nnslr_tools.review_db import _with_position
    fix = {'latitude': 47.48, 'longitude': 8.05, 'speed_mps': 13.6, 'bearing_deg': 14.4}
    lines = [{'frame_key': '0/0', 'segment_index': 0, 'output_index': 0, **fix, 'position_from': 'gps',
              'country': 'CH', 'country_from': 'gps'},
             {'frame_key': '0/1', 'segment_index': 0, 'output_index': 1, 'latitude': None, 'longitude': None,
              'speed_mps': None, 'bearing_deg': None, 'country': 'CH', 'country_from': 'nearest_fix'}]
    path = positions_path(tmp_path, ROUTE)
    path.parent.mkdir(parents=True)
    path.write_text(''.join(json.dumps(line) + '\n' for line in lines))
    saved = {'route_id': ROUTE, 'segment_index': 0, 'output_index': 0, 'gps': None, 'country': 'CH'}
    assert _with_position(saved, tmp_path, {}) == {**saved, 'gps': fix}
    # The line of the review itself is not touched, and the reviewer's country stays.
    assert saved['gps'] is None
    assert _with_position({**saved, 'country': 'IT'}, tmp_path, {}) == {**saved, 'gps': fix, 'country': 'IT'}
    assert _with_position({**saved, 'country': None}, tmp_path, {}) == {**saved, 'gps': fix}
    # A position saved with the review is never replaced by the one of today.
    own = {'latitude': 46.2, 'longitude': 6.14, 'speed_mps': 5.0, 'bearing_deg': 90.0}
    assert _with_position({**saved, 'gps': own}, tmp_path, {})['gps'] == own
    assert _with_position({**saved, 'gps': own, 'country': None}, tmp_path, {}) == {**saved, 'gps': own}
    # A frame without a fix, or without a line of its own: still no position.
    assert _with_position({**saved, 'output_index': 1}, tmp_path, {}) == {**saved, 'output_index': 1}
    assert _with_position({**saved, 'output_index': 2}, tmp_path, {}) == {**saved, 'output_index': 2}


def test_a_frame_saved_without_a_position_gets_it_at_the_next_import(server, tmp_path):
    """Its rows then have a place for the split and the car's speed; a frame saved with its position keeps it,
    whatever the positions of the route say later (#104)."""
    from nnslr_tools.review_db import import_from_db
    from nnslr_tools.splits import place_of
    running, rows = server
    # 0/0 has a position, 0/1 a line without a fix (and the country of the nearest one), 0/2 no line at all.
    for key in ('0/0', '0/1', '0/2'):
        status, response, data = running.post_frame(ROUTE, key, page_body(rows[key]))
        assert status == 200, data
    output = tmp_path / 'annotations/objects.jsonl'

    def imported():
        report = import_from_db(tmp_path, output, store=running.app.store)
        found = {}
        for line in output.read_text().splitlines():
            row = json.loads(line)
            found.setdefault(row['output_index'], []).append(row)
        return report, found

    report, found = imported()
    assert report['imported_frame_count'] == 3 and sorted(found) == [0, 1, 2]
    assert found[0][0]['gps']['latitude'] == 47.48
    assert all(row['gps'] is None and place_of(row) is None for index in (1, 2) for row in found[index])
    # The route's positions are read again (nnslr gps, as after a Propose again that took in the frames between the
    # seconds): every frame has a fix now, and 0/0 another one.
    lines = [{'frame_key': f'0/{index}', 'segment_index': 0, 'output_index': index, 'latitude': 47.5 + index / 1000,
              'longitude': 8.07, 'speed_mps': 12.0 + index, 'bearing_deg': 20.0, 'position_from': 'gps',
              'country': 'CH', 'country_from': 'gps'} for index in range(3)]
    positions_path(tmp_path, ROUTE).write_text(''.join(json.dumps(line) + '\n' for line in lines))
    report, found = imported()
    assert (report['replaced_frame_count'], report['already_imported_frame_count']) == (2, 1)
    assert found[0][0]['gps'] == {'latitude': 47.48, 'longitude': 8.05, 'speed_mps': 13.6, 'bearing_deg': 14.4}
    for index in (1, 2):
        for row in found[index]:
            assert row['gps'] == {'latitude': 47.5 + index / 1000, 'longitude': 8.07, 'speed_mps': 12.0 + index,
                                  'bearing_deg': 20.0}
            assert row['country'] == 'CH' and place_of(row) is not None
    # Nothing left to give: the import after that changes no frame.
    report, found = imported()
    assert (report['replaced_frame_count'], report['already_imported_frame_count']) == (0, 3)


def test_the_kind_of_an_other_sign_is_kept_and_checked(server):
    running, rows = server
    body = page_body(rows['0/0'])
    body['detections'][0].update(family='other_sign', value_kph=None, sign_type='town_entry')
    status, response, data = running.post_frame(ROUTE, '0/0', body)
    assert status == 200 and json_of(data)['review']['detections'][0]['sign_type'] == 'town_entry'
    body = page_body(rows['0/2'])
    body['detections'][0]['sign_type'] = 'town_entry'
    status, response, data = running.post_frame(ROUTE, '0/2', body)
    assert status == 400 and json_of(data)['error'] == 'invalid_sign_type'
    body['detections'][0].update(family='other_sign', value_kph=None, sign_type='lamp_post')
    status, response, data = running.post_frame(ROUTE, '0/2', body)
    assert status == 400 and 'sign_type must be one of' in json_of(data)['message']
    # The merging-traffic triangle has its kind (#103).
    body['detections'][0].update(sign_type='merging_traffic')
    status, response, data = running.post_frame(ROUTE, '0/2', body)
    assert status == 200 and json_of(data)['review']['detections'][0]['sign_type'] == 'merging_traffic'


def test_a_not_a_sign_box_on_a_vehicle_is_kept_only_as_not_a_sign(server):
    """A truck's top-speed sticker: not a sign, said "on a vehicle" (V on the page)."""
    running, rows = server
    body = page_body(rows['0/0'])
    body['detections'][0].update(family='not_a_sign', value_kph=None, sign_type='on_vehicle')
    status, response, data = running.post_frame(ROUTE, '0/0', body)
    assert status == 200 and json_of(data)['review']['detections'][0]['sign_type'] == 'on_vehicle'
    body = page_body(rows['0/2'])
    body['detections'][0].update(family='other_sign', value_kph=None, sign_type='on_vehicle')
    status, response, data = running.post_frame(ROUTE, '0/2', body)
    assert status == 400 and json_of(data)['error'] == 'invalid_sign_type'
    assert "ON_VEHICLE = 'on_vehicle'" in CATALOG.read_text()


def test_the_hours_read_on_a_plate_reach_the_dataset(server, tmp_path):
    from nnslr_tools.annotations import reviewed_frame_annotations, validate_dataset
    running, rows = server
    body = page_body(rows['0/0'])
    body['detections'][0].update(family='maximum_speed', value_kph=30, supplementary_panel='time', plate_hours='22-6')
    status, response, data = running.post_frame(ROUTE, '0/0', body)
    assert status == 200 and json_of(data)['review']['detections'][0]['plate_hours'] == '22-6'
    payload = {review['frame_key']: review for review in running.app.store.reviews(ROUTE, RUN_ID)}['0/0']['payload']
    annotations = reviewed_frame_annotations(payload, '0' * 64, dataset_kind='gold', name='test')
    assert [row['plate_hours'] for row in annotations] == ['22-6']
    assert validate_dataset(annotations, tmp_path, dataset_kind='gold')['valid'] is True
    # A box that says none has no such field: the rows of today stay as they are.
    payload['detections'][0].pop('plate_hours')
    assert 'plate_hours' not in reviewed_frame_annotations(payload, '0' * 64, dataset_kind='gold', name='test')[0]
    for panel, hours in (('time', '22-06'), ('none', '22-6')):
        body = page_body(rows['0/2'])
        body['detections'][0].update(family='maximum_speed', value_kph=30, supplementary_panel=panel, plate_hours=hours)
        status, response, data = running.post_frame(ROUTE, '0/2', body)
        assert status == 400 and json_of(data)['error'] == 'invalid_plate_hours'


def test_a_plate_of_distance_and_its_metres_reach_the_dataset(server, tmp_path):
    """A give way with "15 m" under it (#98): Panel below Distance, on any sign, with the metres when they are read."""
    from nnslr_tools.annotations import reviewed_frame_annotations, validate_dataset
    running, rows = server
    body = page_body(rows['0/0'])
    body['detections'][0].update(family='other_sign', value_kph=None, sign_type='give_way',
                                 supplementary_panel='distance', plate_distance_m=15)
    status, response, data = running.post_frame(ROUTE, '0/0', body)
    saved = json_of(data)['review']['detections'][0]
    assert status == 200 and (saved['supplementary_panel'], saved['plate_distance_m']) == ('distance', 15)
    payload = {review['frame_key']: review for review in running.app.store.reviews(ROUTE, RUN_ID)}['0/0']['payload']
    annotations = reviewed_frame_annotations(payload, '0' * 64, dataset_kind='gold', name='test')
    assert [(row['supplementary_panel'], row['plate_distance_m']) for row in annotations] == [('distance', 15)]
    assert validate_dataset(annotations, tmp_path, dataset_kind='gold')['valid'] is True
    # A road user with a plate of distance is no dataset row.
    report = validate_dataset([{**annotations[0], 'sign_type': 'road_user_person'}], tmp_path, dataset_kind='gold')
    assert report['valid'] is False and 'supplementary_panel' in json.dumps(report['errors'])
    # Not read: the panel stays, the rows have no metres; a box with no plate of distance has none either.
    payload['detections'][0]['plate_distance_m'] = None
    row = reviewed_frame_annotations(payload, '0' * 64, dataset_kind='gold', name='test')[0]
    assert row['supplementary_panel'] == 'distance' and 'plate_distance_m' not in row
    # A limit with "200 m" too.
    body = page_body(rows['0/2'])
    body['detections'][0].update(family='maximum_speed', value_kph=50, supplementary_panel='distance', plate_distance_m=200)
    status, response, data = running.post_frame(ROUTE, '0/2', body)
    assert status == 200 and json_of(data)['review']['detections'][0]['plate_distance_m'] == 200
    for panel, metres in (('distance', 0), ('distance', 5001), ('distance', 1.2), ('distance', '15'), ('none', 15),
                          ('time', 15)):
        body = page_body(rows['0/2'])
        body['detections'][0].update(family='maximum_speed', value_kph=50, supplementary_panel=panel,
                                     plate_distance_m=metres)
        status, response, data = running.post_frame(ROUTE, '0/2', body)
        assert status == 400 and json_of(data)['error'] == 'invalid_plate_distance', (panel, metres)
    for refused in ({'family': 'not_a_sign', 'value_kph': None},
                    {'family': 'other_sign', 'value_kph': None, 'sign_type': 'road_user_person'}):
        body = page_body(rows['0/2'])
        body['detections'][0].update(refused, supplementary_panel='distance')
        status, response, data = running.post_frame(ROUTE, '0/2', body)
        assert status == 400 and json_of(data)['error'] == 'invalid_supplementary_panel', refused
        assert 'only for signs' in json_of(data)['message']


def test_a_proposal_of_hours_moved_to_another_plate_saves_without_its_hours(tmp_path, monkeypatch):
    """The model reads a limit with "200 m" under it as one with hours (no class says a distance yet, #98). The page
    drops plate_hours when the reviewer chooses Distance (or None, Wet): the save must not take the proposal's hours
    back from it, which made it a 400 invalid_schema the page could not get out of."""
    from nnslr_tools.annotations import reviewed_frame_annotations, validate_dataset
    from nnslr_tools.review import write_review
    monkeypatch.setenv('NNSLR_REVIEW_TOKEN', TOKEN)
    rows = make_report(tmp_path, count=3)
    rows[0]['detections'][0].update(model_label='maximum_speed_hours:30:22-6', supplementary_panel='time',
                                    plate_hours='22-6')
    directory = tmp_path / 'derived/preannotations' / ROUTE / RUN_ID
    (directory / 'preannotations.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    write_review(directory, rows, RUN_ID)
    running = RunningServer(tmp_path)
    try:
        revision = 0
        for panel, metres in (('distance', 200), ('distance', None), ('none', None), ('wet', None)):
            body = page_body(rows[0], base_revision=revision)
            body['detections'][0].update(supplementary_panel=panel, plate_distance_m=metres)
            del body['detections'][0]['plate_hours']
            status, response, data = running.post_frame(ROUTE, '0/0', body)
            assert status == 200, (panel, json_of(data))
            saved = json_of(data)['review']['detections'][0]
            revision = json_of(data)['review']['revision']
            assert saved['supplementary_panel'] == panel and 'plate_hours' not in saved
            assert saved.get('plate_distance_m') == metres
            payload = {review['frame_key']: review for review in running.app.store.reviews(ROUTE, RUN_ID)}['0/0']['payload']
            annotations = reviewed_frame_annotations(payload, '0' * 64, dataset_kind='gold', name='test')
            assert validate_dataset(annotations, tmp_path, dataset_kind='gold')['valid'] is True, panel
        # A box that only names its proposal keeps all of it, the hours too.
        body = page_body(rows[0], base_revision=revision)
        body['detections'] = [{'detection_id': rows[0]['detections'][0]['detection_id']}]
        status, response, data = running.post_frame(ROUTE, '0/0', body)
        saved = json_of(data)['review']['detections'][0]
        assert status == 200 and (saved['supplementary_panel'], saved['plate_hours']) == ('time', '22-6')
    finally:
        running.close()


def test_an_other_sign_painted_on_the_road_keeps_its_kind(server, tmp_path):
    """The children's triangle with "SCHULE" painted in front of a school: Children, painted on the road."""
    from nnslr_tools.annotations import reviewed_frame_annotations, validate_dataset
    running, rows = server
    body = page_body(rows['0/0'])
    body['detections'][0].update(family='other_sign', value_kph=None, sign_type='children', painted=True)
    status, response, data = running.post_frame(ROUTE, '0/0', body)
    saved = json_of(data)['review']['detections'][0]
    assert status == 200 and (saved['sign_type'], saved['painted']) == ('children', True)
    payload = {review['frame_key']: review for review in running.app.store.reviews(ROUTE, RUN_ID)}['0/0']['payload']
    annotations = reviewed_frame_annotations(payload, '0' * 64, dataset_kind='gold', name='test')
    assert [(row['sign_type'], row['painted']) for row in annotations] == [('children', True)]
    assert validate_dataset(annotations, tmp_path, dataset_kind='gold')['valid'] is True
    body = page_body(rows['0/2'])
    body['detections'][0]['painted'] = True
    status, response, data = running.post_frame(ROUTE, '0/2', body)
    assert status == 400 and json_of(data)['error'] == 'invalid_painted'
    body['detections'][0].update(family='other_sign', value_kph=None, painted='yes')
    status, response, data = running.post_frame(ROUTE, '0/2', body)
    assert status == 400 and json_of(data)['error'] == 'invalid_painted'
