"""The road type of every frame from the map (nnslr_tools.road_types): the type of a way, the drive smoothed, the file."""
import json

from nnslr_tools.cli import main
from nnslr_tools.map_context import RoadMap
from nnslr_tools.road_types import build_road_types, load_road_types, smooth, stretches, way_type

ROUTE = '0000002a--0123456789'


def test_way_type_from_kind_limit_and_highway():
    assert way_type({'highway': 'motorway', 'maxspeed': '100'})[0] == 'motorway'
    assert way_type({'highway': 'motorway_link'})[0] == 'ramp'
    assert way_type({'highway': 'trunk_link'})[0] == 'ramp'
    assert way_type({'highway': 'trunk_link', 'maxspeed': '50'})[0] == 'town'
    assert way_type({'highway': 'trunk', 'maxspeed': '100'})[0] == 'expressway'
    # A trunk road at 50 runs through a town.
    assert way_type({'highway': 'trunk', 'maxspeed': '50'})[0] == 'town'
    # The kind of limit wins over the number: a rural road signed 50 stays rural.
    assert way_type({'highway': 'secondary', 'maxspeed': '50', 'maxspeed_type': 'CH:rural'})[0] == 'rural'
    assert way_type({'highway': 'tertiary', 'maxspeed_type': 'CH:urban'})[0] == 'town'
    # A 30 zone is a type of its own: sure from its kind, a guess from a bare 30; another zone is town.
    assert way_type({'highway': 'residential', 'maxspeed_type': 'CH:zone30'}) == ('zone30', 'limit CH:zone30')
    assert way_type({'highway': 'residential', 'maxspeed': '30'}) == ('zone30', 'maxspeed 30')
    assert way_type({'highway': 'residential', 'maxspeed': '20', 'maxspeed_type': 'CH:zone20'})[0] == 'town'
    assert way_type({'highway': 'residential', 'maxspeed': '30', 'maxspeed_type': 'CH:urban'})[0] == 'town'
    assert way_type({'highway': 'primary', 'maxspeed': 'CH:rural'})[0] == 'rural'
    assert way_type({'highway': 'primary', 'maxspeed': '50'}) == ('town', 'maxspeed 50')
    assert way_type({'highway': 'primary', 'maxspeed': '60'})[0] == 'rural'
    assert way_type({'highway': 'secondary', 'maxspeed': '80'})[0] == 'rural'
    assert way_type({'highway': 'residential'})[0] == 'town'
    # An ordinary road the map gives no limit: unknown, never guessed.
    assert way_type({'highway': 'unclassified'})[0] is None
    assert way_type({'highway': 'primary', 'maxspeed': 'signals'})[0] is None


def test_smooth_fills_gaps_and_drops_slips():
    town, rural = 'town', 'rural'
    # A tunnel (unknown) between two rural stretches; a GPS slip of 2 frames to a town street.
    types = [rural] * 5 + [None] * 3 + [rural] * 5 + [town] * 2 + [rural] * 5
    result = smooth(types)
    assert [value for value, _ in result] == [rural] * 20
    assert [how for _, how in result][5:8] == ['filled'] * 3
    assert [how for _, how in result][13:15] == ['smoothed'] * 2
    # A real change of road stays: unknown between two different types is not filled.
    types = [rural] * 5 + [None] * 2 + [town] * 5
    assert [value for value, _ in smooth(types)] == types
    # Unknown at the start and the end has only one side: it stays unknown.
    assert smooth([None, town, town])[0] == (None, None)


def _map(tmp_path):
    # Two roads crossing: a motorway going north at lon 8.0, a town street going east at lat 47.001.
    ways = [{'id': 1, 'highway': 'motorway', 'maxspeed': '120', 'nodes': [[47.0, 8.0], [47.002, 8.0]]},
            {'id': 2, 'highway': 'secondary', 'maxspeed': '50', 'nodes': [[47.001, 7.999], [47.001, 8.001]]}]
    path = tmp_path / 'derived' / 'osm' / 'roads.jsonl'
    path.parent.mkdir(parents=True)
    path.write_text(''.join(json.dumps(way) + '\n' for way in ways))
    return RoadMap.load(tmp_path)


def test_build_follows_heading_at_a_crossing(tmp_path):
    road_map = _map(tmp_path)
    positions = [
        # Driving north on the motorway, right over the town street: the heading says which road is ours.
        {'frame_key': '0/1', 'segment_index': 0, 'output_index': 1, 'latitude': 47.001, 'longitude': 8.0, 'bearing_deg': 2.0},
        {'frame_key': '0/0', 'segment_index': 0, 'output_index': 0, 'latitude': 47.0005, 'longitude': 8.0, 'bearing_deg': 0.0},
        # Driving east on the street.
        {'frame_key': '0/2', 'segment_index': 0, 'output_index': 2, 'latitude': 47.001, 'longitude': 8.0007, 'bearing_deg': 90.0},
        # No fix, then far from every road.
        {'frame_key': '0/3', 'segment_index': 0, 'output_index': 3, 'latitude': None, 'longitude': None, 'bearing_deg': None},
        {'frame_key': '0/4', 'segment_index': 0, 'output_index': 4, 'latitude': 47.5, 'longitude': 8.5, 'bearing_deg': 0.0},
    ]
    result = build_road_types(tmp_path, ROUTE, positions, road_map)
    assert result['types'] == {'motorway': 2, 'town': 1, 'unknown': 2}
    lines = load_road_types(tmp_path, ROUTE)
    # Written in the order of the drive.
    assert [line['frame_key'] for line in lines] == ['0/0', '0/1', '0/2', '0/3', '0/4']
    assert [line['road_type'] for line in lines] == ['motorway', 'motorway', 'town', None, None]
    assert lines[0]['way_id'] == 1 and lines[2]['way_id'] == 2 and lines[2]['basis'] == 'maxspeed 50'
    assert (lines[3]['basis'], lines[4]['basis']) == ('no gps', 'no road')
    assert [line['sure'] for line in lines] == [True, True, True, False, False]
    assert stretches(lines) == [{'type': 'motorway', 'from': '0/0', 'to': '0/1', 'frames': 2},
                                {'type': 'town', 'from': '0/2', 'to': '0/2', 'frames': 1},
                                {'type': None, 'from': '0/3', 'to': '0/4', 'frames': 2}]


def test_cli_road_types_all(tmp_path, capsys):
    _map(tmp_path)
    gps = tmp_path / 'derived' / 'gps' / ROUTE / 'positions.jsonl'
    gps.parent.mkdir(parents=True)
    gps.write_text(json.dumps({'frame_key': '0/0', 'segment_index': 0, 'output_index': 0, 'latitude': 47.0005,
                               'longitude': 8.0, 'bearing_deg': 180.0}) + '\n')
    assert main(['road-types', '--all', '--data-root', str(tmp_path)]) == 0
    assert json.loads(capsys.readouterr().out)['types'] == {'motorway': 1}
    # Without the map of the roads it says how to make it.
    (tmp_path / 'derived' / 'osm' / 'roads.jsonl').unlink()
    assert main(['road-types', '--all', '--data-root', str(tmp_path)]) == 2
    assert 'osm_roads.py' in capsys.readouterr().err


def test_summary_names_each_drive_with_its_types_and_the_drives_left(tmp_path):
    from review_helpers import make_report

    from nnslr_tools.road_types import summary
    assert summary(tmp_path) == {'routes': [], 'made_at': None, 'drives_without': 0}
    # A prepared drive without road types yet.
    make_report(tmp_path, ROUTE)
    assert summary(tmp_path)['drives_without'] == 1
    road_map = _map(tmp_path)
    positions = [{'frame_key': f'0/{index}', 'segment_index': 0, 'output_index': index, 'latitude': 47.0005,
                  'longitude': 8.0, 'bearing_deg': 0.0} for index in range(3)]
    positions.append({'frame_key': '0/3', 'segment_index': 0, 'output_index': 3, 'latitude': None, 'longitude': None,
                      'bearing_deg': None})
    build_road_types(tmp_path, ROUTE, positions, road_map)
    result = summary(tmp_path)
    assert result['drives_without'] == 0 and result['made_at'] is not None
    drive = result['routes'][0]
    assert (drive['route'], drive['frames']) == (ROUTE, 4)
    assert drive['types'] == {'town': 0, 'rural': 0, 'expressway': 0, 'motorway': 3, 'ramp': 0, 'parking': 0, 'zone30': 0,
                             'unknown': 1}
    assert drive['stretches'] == [{'type': 'motorway', 'from': '0/0', 'to': '0/2', 'frames': 3},
                                  {'type': None, 'from': '0/3', 'to': '0/3', 'frames': 1}]
    # Read again when the file changes.
    build_road_types(tmp_path, ROUTE, positions[:2], road_map)
    assert summary(tmp_path)['routes'][0]['frames'] == 2


def test_weak_reasons_are_not_sure(tmp_path):
    # A road at 60 (in a town one time in three) and a ramp: labelled, but not sure.
    ways = [{'id': 1, 'highway': 'primary', 'maxspeed': '60', 'nodes': [[47.0, 8.0], [47.002, 8.0]]},
            {'id': 2, 'highway': 'motorway_link', 'nodes': [[47.1, 8.0], [47.102, 8.0]]}]
    path = tmp_path / 'derived' / 'osm' / 'roads.jsonl'
    path.parent.mkdir(parents=True)
    path.write_text(''.join(json.dumps(way) + '\n' for way in ways))
    positions = [{'frame_key': '0/0', 'segment_index': 0, 'output_index': 0, 'latitude': 47.001, 'longitude': 8.0, 'bearing_deg': 0.0},
                 {'frame_key': '0/1', 'segment_index': 0, 'output_index': 1, 'latitude': 47.101, 'longitude': 8.0, 'bearing_deg': 0.0}]
    build_road_types(tmp_path, ROUTE, positions, RoadMap.load(tmp_path))
    lines = load_road_types(tmp_path, ROUTE)
    assert [(line['road_type'], line['sure']) for line in lines] == [('rural', False), ('ramp', False)]


# -- a person's corrections (Data › Roads, a drive opened) -----------------------------------------------------------
def _drive_of(tmp_path, count=6):
    """A drive of `count` frames on the motorway of _map, the last two without GPS."""
    positions = [{'frame_key': f'0/{index}', 'segment_index': 0, 'output_index': index, 'latitude': 47.0005,
                  'longitude': 8.0, 'bearing_deg': 0.0} for index in range(count - 2)]
    positions += [{'frame_key': f'0/{index}', 'segment_index': 0, 'output_index': index, 'latitude': None,
                   'longitude': None, 'bearing_deg': None} for index in range(count - 2, count)]
    road_map = _map(tmp_path)
    build_road_types(tmp_path, ROUTE, positions, road_map)
    return positions, road_map


def test_corrections_over_the_map_the_last_wins_and_a_new_map_keeps_them(tmp_path):
    from nnslr_tools.road_types import corrections_path, correct, map_road_types, summary, undo_correction
    positions, road_map = _drive_of(tmp_path)
    assert [line['road_type'] for line in load_road_types(tmp_path, ROUTE)] == ['motorway'] * 4 + [None] * 2
    # From the end to the start works too; the last correction covering a frame wins.
    correct(tmp_path, ROUTE, '0/5', '0/3', 'town', 'admin')
    correct(tmp_path, ROUTE, '0/4', '0/4', None, 'admin')
    lines = load_road_types(tmp_path, ROUTE)
    assert [line['road_type'] for line in lines] == ['motorway'] * 3 + ['town', None, 'town']
    assert [line.get('reviewed', False) for line in lines] == [False] * 3 + [True] * 3
    assert (lines[3]['basis'], lines[3]['sure'], lines[3]['reviewer']) == ('reviewed', True, 'admin')
    # The map stays as it was, the corrections apart, one line each.
    assert [line['road_type'] for line in map_road_types(tmp_path, ROUTE)] == ['motorway'] * 4 + [None] * 2
    kept = [json.loads(line) for line in corrections_path(tmp_path, ROUTE).read_text().splitlines()]
    assert [(line['from'], line['to'], line['road_type']) for line in kept] == [('0/5', '0/3', 'town'), ('0/4', '0/4', None)]
    assert all(line['reviewer'] == 'admin' and line['at'] for line in kept)
    drive = summary(tmp_path)['routes'][0]
    assert drive['reviewed'] == 3 and drive['types'] == {'town': 2, 'rural': 0, 'expressway': 0, 'motorway': 3,
                                                         'ramp': 0, 'parking': 0, 'zone30': 0, 'unknown': 1}

    # Find again: the map is written anew, the corrections are not touched and still win.
    build_road_types(tmp_path, ROUTE, positions, road_map)
    assert [line['road_type'] for line in load_road_types(tmp_path, ROUTE)] == ['motorway'] * 3 + ['town', None, 'town']
    assert len(corrections_path(tmp_path, ROUTE).read_text().splitlines()) == 2

    # Undo drops the last one; the summary follows.
    assert undo_correction(tmp_path, ROUTE)
    assert [line['road_type'] for line in load_road_types(tmp_path, ROUTE)] == ['motorway'] * 3 + ['town'] * 3
    assert summary(tmp_path)['routes'][0]['types']['unknown'] == 0
    assert undo_correction(tmp_path, ROUTE) and not undo_correction(tmp_path, ROUTE)
    assert load_road_types(tmp_path, ROUTE) == map_road_types(tmp_path, ROUTE)
    # A correction whose frames left the drive (a shorter map) is skipped, not applied elsewhere.
    correct(tmp_path, ROUTE, '0/0', '0/5', 'rural', 'admin')
    build_road_types(tmp_path, ROUTE, positions[:4], road_map)
    assert [line['road_type'] for line in load_road_types(tmp_path, ROUTE)] == ['motorway'] * 4


def test_a_correction_names_frames_of_the_drive_and_a_type(tmp_path):
    import pytest

    from nnslr_tools.road_types import CorrectionError, corrections_path, correct
    _drive_of(tmp_path)
    for start, end, kind in (('0/9', '0/1', 'town'), ('0/0', '1/0', 'town'), (0, '0/1', 'town'), ('0/0', None, 'town'),
                             ('../0', '0/1', 'town'), ('0/0', '0/1', 'highway'), ('0/0', '0/1', 'unknown'),
                             ('0/0', '0/1', 3)):
        with pytest.raises(CorrectionError):
            correct(tmp_path, ROUTE, start, end, kind, 'admin')
    with pytest.raises(ValueError):
        correct(tmp_path, '../etc', '0/0', '0/1', 'town', 'admin')
    # A drive without road types has no frame to correct.
    with pytest.raises(CorrectionError):
        correct(tmp_path, '0000002b--0123456789', '0/0', '0/0', 'town', 'admin')
    assert not corrections_path(tmp_path, ROUTE).exists()


def test_the_drive_over_http(tmp_path):
    from review_helpers import TOKEN, RunningServer, json_of, make_report, sign_in

    from nnslr_tools.road_types import corrections_path
    make_report(tmp_path, ROUTE, count=6)
    _drive_of(tmp_path)
    server = RunningServer(tmp_path, token=TOKEN, require_token=True)
    address = f'/api/panel/road-types/{ROUTE}'
    body = {'from': '0/1', 'to': '0/2', 'road_type': 'rural'}
    try:
        # Signed out: 401, nothing written.
        assert server.request('GET', address)[0] == 401
        assert server.request('POST', address, body=body)[0] == 401
        assert not corrections_path(tmp_path, ROUTE).exists()

        cookie = sign_in(server)
        status, response, data = server.request('GET', address, headers=cookie)
        drive = json_of(data)
        assert status == 200 and drive['route'] == ROUTE and drive['corrections'] == 0
        # The picture of each frame: the preview of its row in the report, by frame key.
        assert [(frame['key'], frame['image']) for frame in drive['frames'][:2]] == [
            ('0/0', f'/route/{ROUTE}/images/00000000.jpg'), ('0/1', f'/route/{ROUTE}/images/00000001.jpg')]
        assert server.request('GET', drive['frames'][1]['image'], headers=cookie)[0] == 200
        assert [frame['road_type'] for frame in drive['frames']] == ['motorway'] * 4 + [None] * 2
        assert drive['frames'][4]['basis'] == 'no gps' and not drive['frames'][4]['sure']
        # Not a prepared drive, or no route at all: 404.
        for other in ('0000002b--0123456789', 'not-a-route', '..'):
            status, response, data = server.request('GET', f'/api/panel/road-types/{other}', headers=cookie)
            assert status == 404 and json_of(data)['error'] in ('unknown_route', 'not_found')
            assert server.request('POST', f'/api/panel/road-types/{other}', body=body, headers=cookie)[0] == 404

        # Same origin and JSON only, as the actions of the panel; then the body is checked.
        status, response, data = server.request('POST', address, body=body, headers={**cookie, 'Origin': 'http://evil.example'})
        assert status == 403 and json_of(data)['error'] == 'cross_origin'
        status, response, data = server.request('POST', address, body=b'from=0/1',
                                                headers={**cookie, 'Content-Type': 'application/x-www-form-urlencoded'})
        assert status == 415
        for bad in ([], {'from': '0/1', 'to': '0/2'}, {**body, 'road_type': 'highway'}, {**body, 'from': '7/7'},
                    {**body, 'reviewer': 'someone'}):
            status, response, data = server.request('POST', address, body=bad, headers=cookie)
            assert status == 400, bad
        assert server.request('POST', address, body=b'{"from":', headers=cookie)[0] == 400
        assert server.request('GET', address + '/undo', headers=cookie)[0] == 405
        assert not corrections_path(tmp_path, ROUTE).exists()

        # Saved: the drive again, with the stretch reviewed; recorded with the name of the signed-in person.
        status, response, data = server.request('POST', address, body=body,
                                                headers={**cookie, 'Origin': f'http://127.0.0.1:{server.port}'})
        drive = json_of(data)
        assert status == 200 and drive['ok'] and drive['corrections'] == 1 and drive['reviewed'] == 2
        assert [(frame['road_type'], frame['reviewed']) for frame in drive['frames'][:3]] == [
            ('motorway', False), ('rural', True), ('rural', True)]
        assert json.loads(corrections_path(tmp_path, ROUTE).read_text())['reviewer'] == 'admin'
        # Can't tell, by an agent with the token.
        status, response, data = server.request('POST', address, body={'from': '0/5', 'to': '0/4', 'road_type': None},
                                                headers={'X-Review-Token': TOKEN})
        assert status == 200 and [frame['reviewed'] for frame in json_of(data)['frames']][4:] == [True, True]
        list_row = json_of(server.request('GET', '/api/panel/road-types', headers=cookie)[2])['routes'][0]
        assert list_row['reviewed'] == 4

        # Undo, twice, then nothing left to undo.
        for left in (1, 0):
            status, response, data = server.request('POST', address + '/undo', body={}, headers=cookie)
            assert status == 200 and json_of(data)['corrections'] == left
        assert server.request('POST', address + '/undo', body={}, headers=cookie)[0] == 409
    finally:
        server.close()


def test_a_read_only_panel_refuses_a_correction(tmp_path):
    from review_helpers import RunningServer, json_of, make_report

    from nnslr_tools.road_types import corrections_path
    make_report(tmp_path, ROUTE, count=6)
    _drive_of(tmp_path)
    server = RunningServer(tmp_path)
    server.app.panel.actions_enabled = False
    try:
        address = f'/api/panel/road-types/{ROUTE}'
        assert server.request('GET', address)[0] == 200
        status, response, data = server.request('POST', address, body={'from': '0/0', 'to': '0/0', 'road_type': 'town'})
        assert status == 403 and json_of(data)['error'] == 'actions_disabled'
        assert not corrections_path(tmp_path, ROUTE).exists()
    finally:
        server.close()


# -- car parks: the fifth type ---------------------------------------------------------------------------------------
def _parking_map(tmp_path, ways, parks):
    """The roads and car parks of the map in the data root, as scripts/osm_roads.py writes them."""
    folder = tmp_path / 'derived' / 'osm'
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'roads.jsonl').write_text(''.join(json.dumps(way) + '\n' for way in ways))
    (folder / 'parking.jsonl').write_text(''.join(json.dumps(park) + '\n' for park in parks))
    return RoadMap.load(tmp_path)


def _at(lat, lon, heading, speed):
    return {'frame_key': '0/0', 'segment_index': 0, 'output_index': 0, 'latitude': lat, 'longitude': lon,
            'bearing_deg': heading, 'speed_mps': speed}


def _type(road_map, parking, lat, lon, heading=90.0, speed=4.0):
    from nnslr_tools.road_types import frame_types
    line = frame_types(road_map, [_at(lat, lon, heading, speed)], parking)[0]
    return line['road_type'], line['basis'], line['sure']


# A car park from lat 47.010 to 47.012, lon 8.010 to 8.013, with a hole (a building) in its south-west corner; a
# primary road at 50 crosses it from west to east at lat 47.011, a service road runs north in its east part.
CAR_PARK = {'id': 7, 'type': 'way', 'name': 'P', 'parking': 'surface',
            'rings': [[[47.010, 8.010], [47.010, 8.013], [47.012, 8.013], [47.012, 8.010], [47.010, 8.010]],
                      [[47.0102, 8.0102], [47.0106, 8.0102], [47.0106, 8.0108], [47.0102, 8.0108], [47.0102, 8.0102]]]}
CROSSING = [{'id': 1, 'highway': 'primary', 'maxspeed': '50', 'nodes': [[47.011, 8.009], [47.011, 8.014]]},
            {'id': 2, 'highway': 'service', 'nodes': [[47.0102, 8.0125], [47.0118, 8.0125]]}]


def test_parking_from_an_aisle(tmp_path):
    road_map = _parking_map(tmp_path, [{'id': 3, 'highway': 'service', 'service': 'parking_aisle',
                                        'nodes': [[47.02, 8.0], [47.02, 8.002]]}], [])
    positions = [{'frame_key': f'0/{index}', 'segment_index': 0, 'output_index': index, 'latitude': 47.02,
                  'longitude': 8.0002 + index * 0.0003, 'bearing_deg': 90.0, 'speed_mps': 4.0} for index in range(5)]
    # No outline around it: the tag of the aisle is enough, and sure.
    assert build_road_types(tmp_path, ROUTE, positions, road_map)['types'] == {'parking': 5}
    lines = load_road_types(tmp_path, ROUTE)
    assert {(line['road_type'], line['basis'], line['sure'], line['way_id']) for line in lines} == {
        ('parking', 'parking aisle', True, 3)}
    # At 50 km/h on an aisle: the GPS slipping onto a car park beside a ramp (Basel St. Jakob), not a car park.
    from nnslr_tools.road_types import ParkingMap
    assert _type(road_map, ParkingMap([]), 47.02, 8.001, speed=13.0) == (None, 'highway service without a limit', False)
    # A car park beside the road for less than MIN_RUN frames is a slip, as any other.
    assert [value for value, _ in smooth(['town'] * 5 + ['parking'] * 2 + ['town'] * 5)] == ['town'] * 12


def test_parking_from_an_outline_with_no_road(tmp_path):
    from nnslr_tools.road_types import ParkingMap
    road_map = _parking_map(tmp_path, CROSSING, [CAR_PARK])
    parking = ParkingMap.load(tmp_path)
    # Inside the outline, no road within reach: a car park, but one signal only.
    assert _type(road_map, parking, 47.0118, 8.0105) == ('parking', 'car park on the map', False)
    # On the service road inside it: the outline and the road agree.
    assert _type(road_map, parking, 47.0104, 8.0125, heading=0.0) == ('parking', 'car park on the map', True)
    # In the hole of the outline, or outside it: no road, no type.
    assert _type(road_map, parking, 47.0104, 8.0105) == (None, 'no road', False)
    assert _type(road_map, parking, 47.0135, 8.0105) == (None, 'no road', False)
    # Without parking.jsonl (a map read before the car parks were kept) the outline is not known.
    (tmp_path / 'derived' / 'osm' / 'parking.jsonl').unlink()
    assert _type(road_map, ParkingMap.load(tmp_path), 47.0118, 8.0105) == (None, 'no road', False)


def test_not_parking_on_a_road_through_a_car_park_at_speed(tmp_path):
    from nnslr_tools.road_types import ParkingMap
    road_map = _parking_map(tmp_path, CROSSING, [CAR_PARK])
    parking = ParkingMap.load(tmp_path)
    # On the primary road across the outline at 50 km/h: the road, not the car park.
    assert _type(road_map, parking, 47.011, 8.0110, speed=14.0) == ('town', 'maxspeed 50', True)
    # Without a speed the road stays too.
    assert _type(road_map, parking, 47.011, 8.0110, speed=None) == ('town', 'maxspeed 50', True)
    # The service road inside the outline at 45 km/h: the GPS slipping, not the car park.
    assert _type(road_map, parking, 47.0104, 8.0125, heading=0.0, speed=12.5) == (
        None, 'highway service without a limit', False)
    # Slower than a walk there: the GPS finds the road beside the car park, the car is in it (not sure).
    assert _type(road_map, parking, 47.011, 8.0110, speed=1.0) == ('parking', 'car park on the map', False)
    # An underground car park lies under the street: slow over it is not parking.
    underground = ParkingMap([{**CAR_PARK, 'parking': 'underground'}])
    assert _type(road_map, underground, 47.011, 8.0110, speed=1.0) == ('town', 'maxspeed 50', True)


def test_a_correction_to_parking(tmp_path):
    from nnslr_tools.road_types import correct, summary
    _drive_of(tmp_path)
    correct(tmp_path, ROUTE, '0/0', '0/1', 'parking', 'admin')
    lines = load_road_types(tmp_path, ROUTE)
    assert [line['road_type'] for line in lines] == ['parking'] * 2 + ['motorway'] * 2 + [None] * 2
    assert (lines[0]['basis'], lines[0]['sure'], lines[0]['reviewed']) == ('reviewed', True, True)
    assert summary(tmp_path)['routes'][0]['types'] == {'town': 0, 'rural': 0, 'expressway': 0, 'motorway': 2,
                                                       'ramp': 0, 'parking': 2, 'zone30': 0, 'unknown': 2}


def test_a_broken_corrections_line_is_skipped(tmp_path):
    from nnslr_tools.road_types import corrections_path, correct, drive_view, load_corrections, summary
    _drive_of(tmp_path)
    path = corrections_path(tmp_path, ROUTE)
    path.parent.mkdir(parents=True)
    good = {'from': '0/0', 'to': '0/0', 'road_type': 'town', 'reviewer': 'admin', 'at': '2026-10-27T12:00:00+01:00'}
    path.write_text('\n'.join([
        json.dumps(good),
        '{"from": "0/1", "to":',                                        # cut short
        '[1, 2]',                                                       # not an object
        json.dumps({**good, 'from': '0/1', 'to': '0/1', 'road_type': 'highway'}),   # not a type
        json.dumps({**good, 'from': 1, 'to': '0/1'}),                   # not a frame key
        json.dumps({'from': '0/1', 'to': '0/1', 'reviewer': 'admin'}),  # no road_type
        '{"from": "0/1", "to": "0/1", "road_type": NaN}',
        '\xff\xfe not text',
        json.dumps({**good, 'from': '0/2', 'to': '0/2', 'road_type': None}),
    ]) + '\n', encoding='latin-1')
    assert [(line['from'], line['road_type']) for line in load_corrections(tmp_path, ROUTE)] == [('0/0', 'town'),
                                                                                                ('0/2', None)]
    assert [line['road_type'] for line in load_road_types(tmp_path, ROUTE)] == ['town', 'motorway', None, 'motorway',
                                                                                None, None]
    # The list of Data › Roads and the drive opened still answer.
    assert summary(tmp_path)['routes'][0]['reviewed'] == 2
    assert drive_view(tmp_path, ROUTE, {})['corrections'] == 2
    # A new correction is appended after them and counts.
    correct(tmp_path, ROUTE, '0/1', '0/1', 'rural', 'admin')
    assert [line['road_type'] for line in load_road_types(tmp_path, ROUTE)][:3] == ['town', 'rural', None]
