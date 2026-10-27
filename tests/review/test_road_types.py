"""The road type of every frame from the map (nnslr_tools.road_types): the type of a way, the drive smoothed, the file."""
import json

from nnslr_tools.cli import main
from nnslr_tools.map_context import RoadMap
from nnslr_tools.road_types import build_road_types, load_road_types, smooth, stretches, way_type

ROUTE = '0000002a--0123456789'


def test_way_type_from_kind_limit_and_highway():
    assert way_type({'highway': 'motorway', 'maxspeed': '100'})[0] == 'motorway'
    assert way_type({'highway': 'motorway_link'})[0] == 'motorway'
    assert way_type({'highway': 'trunk', 'maxspeed': '100'})[0] == 'expressway'
    # A trunk road at 50 runs through a town.
    assert way_type({'highway': 'trunk', 'maxspeed': '50'})[0] == 'town'
    # The kind of limit wins over the number: a rural road signed 50 stays rural.
    assert way_type({'highway': 'secondary', 'maxspeed': '50', 'maxspeed_type': 'CH:rural'})[0] == 'rural'
    assert way_type({'highway': 'tertiary', 'maxspeed_type': 'CH:urban'})[0] == 'town'
    assert way_type({'highway': 'residential', 'maxspeed_type': 'CH:zone30'})[0] == 'town'
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
    assert drive['types'] == {'town': 0, 'rural': 0, 'expressway': 0, 'motorway': 3, 'unknown': 1}
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
    assert [(line['road_type'], line['sure']) for line in lines] == [('rural', False), ('motorway', False)]
