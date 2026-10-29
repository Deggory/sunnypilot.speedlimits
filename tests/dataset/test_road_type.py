"""The road-type network's data and measures (nnslr_tools.road_type_data, road_type_metrics) on a tiny synthetic data
root, and its ONNX step contract (nnslr_tools.road_type_net, with torch only). numpy and the modules that need it
are imported inside the tests: collecting the tests must leave numpy out (tests/cli/test_import_isolation.py)."""
import json
from pathlib import Path

import pytest

from nnslr_tools.road_types import ROAD_TYPES

TEST_ROUTE = '00000001--aaaaaaaaaa'
OTHER_ROUTES = ('00000002--bbbbbbbbbb', '00000003--cccccccccc', '00000004--dddddddddd')
TOWN, RURAL, MOTORWAY, RAMP, PARKING = (ROAD_TYPES.index(n) for n in ('town', 'rural', 'motorway', 'ramp', 'parking'))


def _write_route(root: Path, route: str, types: list[tuple], start: tuple[float, float], step: tuple[float, float],
                 *, segments: int = 2, missing: set | None = None, picture: bool = True):
    """A drive of segments x 60 frames: types is a list of (road_type, sure, basis, highway) per frame, positions on a
    line from start by step per frame."""
    from PIL import Image

    from nnslr_tools import road_type_data as data
    lines, positions = [], []
    for i, (road_type, sure, basis, highway) in enumerate(types[:segments * 60]):
        segment, index = divmod(i, 60)
        key = f'{segment}/{index}'
        lines.append({'frame_key': key, 'segment_index': segment, 'output_index': index, 'road_type': road_type,
                      'sure': sure, 'basis': basis, 'highway': highway, 'map_type': road_type, 'way_id': 1,
                      'maxspeed': None})
        positions.append({'frame_key': key, 'segment_index': segment, 'output_index': index,
                          'latitude': start[0] + step[0] * i, 'longitude': start[1] + step[1] * i,
                          'speed_mps': 20.0, 'bearing_deg': 0.0, 'country': 'CH'})
        if picture and (segment, index) not in (missing or set()):
            path = data.frame_path(root, route, segment, index)
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new('RGB', (64, 36), (i % 255, 40, 90)).save(path)
    folder = root / 'derived' / 'road_types'
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f'{route}.jsonl').write_text(''.join(json.dumps(line) + '\n' for line in lines))
    gps = root / 'derived' / 'gps' / route
    gps.mkdir(parents=True, exist_ok=True)
    (gps / 'positions.jsonl').write_text(''.join(json.dumps(p) + '\n' for p in positions))


def _types(*parts):
    found = []
    for road_type, count, *rest in parts:
        sure = rest[0] if rest else True
        basis = rest[1] if len(rest) > 1 else 'maxspeed 50'
        highway = rest[2] if len(rest) > 2 else None
        found += [(road_type, sure, basis, highway)] * count
    return found


@pytest.fixture
def root(tmp_path):
    root = tmp_path / 'data'
    # The test drive: town, a ramp, the motorway; in an area of its own.
    _write_route(root, TEST_ROUTE, _types(('town', 40), ('ramp', 20, False, 'highway motorway_link', 'motorway_link'),
                                          ('motorway', 60, True, 'highway motorway')), (47.0, 8.0), (0.0003, 0.0))
    # Far drives for train and validation, each through its own areas.
    _write_route(root, OTHER_ROUTES[0], _types(('rural', 60, False, 'maxspeed 60'), ('town', 60)), (47.5, 8.5),
                 (0.0003, 0.0))
    _write_route(root, OTHER_ROUTES[1], _types(('motorway', 50), ('ramp', 20, False, 'highway motorway_link',
                                                                  'motorway_link'), ('parking', 50)),
                 (48.0, 9.0), (0.0003, 0.0))
    _write_route(root, OTHER_ROUTES[2], _types((None, 10, False, 'no gps'), ('town', 110)), (48.5, 9.5),
                 (0.0003, 0.0), missing={(1, 30)})
    split = {'test_routes': [TEST_ROUTE]}
    (root / 'splits').mkdir(parents=True)
    (root / 'splits' / 'abc.json').write_text(json.dumps(split))
    (root / 'splits' / 'latest.json').write_text(json.dumps({'path': 'splits/abc.json', 'sha256': 'abc'}))
    return root


def test_labels_and_weights_follow_the_road_types():
    from nnslr_tools import road_type_data as data
    assert data.CLASSES == ('town', 'rural', 'expressway', 'motorway', 'ramp', 'parking')
    assert data.frame_label({'road_type': 'town', 'sure': True}) == (TOWN, 1.0)
    # A ramp from its OSM link is unsure only for its limit: it keeps the full weight.
    assert data.frame_label({'road_type': 'ramp', 'sure': False, 'basis': 'highway motorway_link'}) == (RAMP, 1.0)
    assert data.frame_label({'road_type': 'ramp', 'sure': False, 'basis': 'smoothed (x)'}) == (RAMP, data.UNSURE_WEIGHT)
    assert data.frame_label({'road_type': 'rural', 'sure': False, 'basis': 'maxspeed 60'}) == (RURAL, data.UNSURE_WEIGHT)
    assert data.frame_label({'road_type': None, 'sure': False}) == (-1, 0.0)
    assert data.speed_features(15.0) == (0.5, 1.0)
    assert data.speed_features(100.0) == (2.0, 1.0)
    assert data.speed_features(None) == (0.0, 0.0)


def test_frames_without_gps_take_the_nearest_area():
    from nnslr_tools import road_type_data as data
    frames = [{'latitude': None, 'longitude': None}, {'latitude': 47.0, 'longitude': 8.0},
              {'latitude': None, 'longitude': None}, {'latitude': None, 'longitude': None},
              {'latitude': 47.1, 'longitude': 8.1}]
    cells = data.frame_cells(frames)
    assert cells[0] == cells[1] == cells[2] and cells[3] == cells[4] and cells[1] != cells[4]


def test_split_by_place_keeps_the_test_drive_and_a_buffer(root):
    from nnslr_tools import road_type_data as data
    dataset = data.load_dataset(root, min_length=10)
    split = dataset['split']
    assert split['policy'] == data.SPLIT_POLICY and split['test_routes'] == [TEST_ROUTE]
    assert set(split['partitions'][TEST_ROUTE]) == {'test'}
    # No area is both test (or validation) and train, and every train area is away from the test areas.
    by_cell = {}
    for route, frames in dataset['frames'].items():
        for cell, partition in zip(data.frame_cells(frames), split['partitions'][route]):
            by_cell.setdefault(cell, set()).add(partition)
    assert all(len(partitions) == 1 for partitions in by_cell.values())
    test_cells = {cell for cell, p in by_cell.items() if p == {'test'}}
    train_cells = {cell for cell, p in by_cell.items() if p == {'train'}}
    assert not data._around(test_cells) & train_cells
    assert split['counts']['validation'] > 0 and split['counts']['train'] > 0
    assert dataset['split_sha256'] == data.split_sha256(split)


def test_sequences_are_runs_in_drive_order_broken_at_gaps(root):
    import numpy as np

    from nnslr_tools import road_type_data as data
    frames = data.route_frames(root, OTHER_ROUTES[2])
    runs = data.sequences(frames, ['train'] * len(frames), min_length=10)
    # The missing picture (segment 1, frame 30) breaks the drive in two; positions follow the drive.
    assert [len(run['positions']) for run in runs] == [90, 29]
    assert all(np.all(np.diff(run['positions']) == 1) for run in runs)
    assert runs[0]['positions'][0] == 0 and frames[60]['key'] == '1/0'
    # A buffer frame breaks a run too, and short runs are dropped.
    partitions = ['train'] * 50 + ['buffer'] + ['train'] * (len(frames) - 51)
    assert [len(run['positions']) for run in data.sequences(frames, partitions, min_length=30)] == [50, 39]


def test_summary_counts_unsure_frames_per_class(root):
    from nnslr_tools import road_type_data as data
    summary = data.load_dataset(root, min_length=10)['summary']
    assert summary['all_frames']['labelled']['ramp'] == 40
    assert summary['all_frames']['unsure']['ramp'] == 40
    assert summary['all_frames']['unsure']['rural'] == 60
    assert summary['all_frames']['no_label'] == 10


def test_frame_cache_is_the_pil_bilinear_resize_and_is_reused(root):
    import numpy as np

    from nnslr_tools import road_type_data as data
    frames = {OTHER_ROUTES[2]: data.route_frames(root, OTHER_ROUTES[2])}
    paths = data.build_frame_cache(root, frames, workers=1, log=lambda _: None)
    array = np.load(paths[OTHER_ROUTES[2]])
    assert array.shape == (120, data.FRAME_HEIGHT, data.FRAME_WIDTH, 3) and array.dtype == np.uint8
    assert np.array_equal(array[5], data.resized_frame(frames[OTHER_ROUTES[2]][5]['image']))
    assert not array[90].any()  # the missing picture
    stamp = paths[OTHER_ROUTES[2]].stat().st_mtime_ns
    data.build_frame_cache(root, frames, workers=1, log=lambda _: None)
    assert paths[OTHER_ROUTES[2]].stat().st_mtime_ns == stamp


def test_per_class_f1_over_masked_frames():
    import numpy as np

    from nnslr_tools.road_type_metrics import per_class_f1
    classes = ('a', 'b', 'c')
    result = per_class_f1(np.array([0, 0, 1, 1, 2]), np.array([0, 1, 1, 1, 0]), np.array([1, 1, 1, 1, 0], bool), classes)
    assert result['classes']['a'] == {'precision': 1.0, 'recall': 0.5, 'f1': 0.6667, 'support': 2, 'predicted': 1}
    assert result['classes']['c']['f1'] is None
    assert result['macro_f1'] == round((2 / 3 + 0.8) / 2, 4) and result['frames'] == 4


def test_transitions_their_delay_and_false_switches():
    import numpy as np

    from nnslr_tools.road_type_metrics import change_points, debounced, detection_delay, majority, switches
    classes = ROAD_TYPES
    labels = np.array([RAMP] * 20 + [-1] * 2 + [MOTORWAY] * 40)
    changes = change_points(labels, classes)
    assert changes == [{'time': 22, 'old': RAMP, 'new': MOTORWAY, 'kind': 'ramp_to_motorway'}]
    predicted = np.array([RAMP] * 25 + [MOTORWAY] * 37)
    assert detection_delay(predicted, changes[0]) == 3
    assert detection_delay(np.array([RAMP] * 62), changes[0]) is None
    # Early by 2 seconds.
    assert detection_delay(np.array([RAMP] * 20 + [MOTORWAY] * 42), changes[0]) == -2
    # A short run of fewer than 10 seconds is no transition.
    assert change_points(np.array([TOWN] * 20 + [RURAL] * 5 + [TOWN] * 20), classes) == []
    flicker = np.array([TOWN] * 30 + [RURAL] + [TOWN] * 29)
    assert switches(flicker, np.array([TOWN] * 60)) == (2, 40)
    assert switches(debounced(flicker), np.array([TOWN] * 60))[0] == 0
    assert list(debounced(np.array([0, 1, 1, 1, 0, 0, 0]))) == [0, 0, 0, 1, 1, 1, 0]
    assert list(majority(np.array([0, 1, 0, 1, 1]), window=3)) == [0, 1, 0, 1, 1]


def test_after_merge_accuracy_counts_the_first_minute_on_the_motorway():
    import numpy as np

    from nnslr_tools.road_type_metrics import after_merge, sequence_metrics
    classes = ROAD_TYPES
    labels = np.array([RAMP] * 10 + [MOTORWAY] * 80)
    # The frame network says rural for 20 seconds after the ramp; the memory keeps motorway.
    frame = np.array([RAMP] * 10 + [RURAL] * 20 + [MOTORWAY] * 60)
    memory = np.array([RAMP] * 10 + [MOTORWAY] * 80)
    without = after_merge([{'labels': labels, 'predicted': frame}], classes)
    with_memory = after_merge([{'labels': labels, 'predicted': memory}], classes)
    assert without['merges'] == 1 and without['frames'] == 60
    assert without['accuracy'] == round(40 / 60, 4) and without['said']['rural'] == 20
    assert with_memory['accuracy'] == 1.0
    metrics = sequence_metrics([{'labels': labels, 'weights': np.ones(90), 'predicted': memory}], classes)
    assert metrics['transitions']['ramp_to_motorway']['found'] == 1
    assert metrics['after_merge']['merges'] == 1


def test_onnx_step_contract(root, tmp_path):
    torch = pytest.importorskip('torch')
    pytest.importorskip('onnxruntime')
    pytest.importorskip('torchvision')
    import numpy as np

    from nnslr_tools import road_type_data as data
    from nnslr_tools import road_type_net as net
    model = net.build_road_type_net('gru', pretrained=False).eval()
    # The step's GRU written out by hand gives the numbers of torch's GRU over a sequence.
    x = torch.randn(1, 5, net.FEATURES + 2)
    with torch.no_grad():
        expected, last = model.gru(x)
        hidden = torch.zeros(1, net.HIDDEN)
        for t in range(5):
            hidden = net.gru_step(model.gru, x[:, t], hidden)
            assert torch.allclose(hidden, expected[:, t], atol=1e-5)
    checkpoint = tmp_path / 'road-type-best.pt'
    torch.save({'task': 'road_type', 'architecture': 'mobilenet_v3_small+gru', 'memory': 'gru',
                'classes': list(data.CLASSES), 'state_dict': model.state_dict(), 'epoch': 1,
                'split_sha256': 's', 'dataset_sha256': 'd'}, checkpoint)
    result = net.export_road_type_onnx(checkpoint, tmp_path / 'road-type.onnx', data_root=root)
    contract = json.loads(Path(result['contract']).read_text())
    assert contract['task'] == 'road_type' and contract['verification']['passed'] is True
    assert contract['verification']['rollout']['checked'] is True
    assert [i['name'] for i in contract['inputs']] == ['image', 'speed', 'hidden']
    assert contract['inputs'][0]['shape'] == [1, 3, data.FRAME_HEIGHT, data.FRAME_WIDTH]
    assert [o['name'] for o in contract['outputs']] == ['probabilities', 'hidden_out']
    assert contract['output']['classes'] == list(data.CLASSES)
    import onnx
    graph = onnx.load(result['onnx']).graph
    assert not [node.op_type for node in graph.node if node.op_type in ('GRU', 'LSTM', 'RNN')]
    import onnxruntime
    session = onnxruntime.InferenceSession(result['onnx'], providers=['CPUExecutionProvider'])
    hidden = np.zeros((1, net.HIDDEN), np.float32)
    image = np.random.default_rng(0).random((1, 3, data.FRAME_HEIGHT, data.FRAME_WIDTH), dtype=np.float32)
    probabilities, hidden_out = session.run(None, {'image': image, 'speed': np.array([[0.5, 1.0]], np.float32),
                                                   'hidden': hidden})
    assert probabilities.shape == (1, len(data.CLASSES)) and abs(float(probabilities.sum()) - 1) < 1e-5
    assert hidden_out.shape == (1, net.HIDDEN) and np.abs(hidden_out).max() <= 1


# -- the motorway sign the car confirmed (nnslr_tools.road_type_signs) ------------------------------------------------
def test_sign_features_age_out_after_two_minutes():
    from nnslr_tools import road_type_signs as signs
    assert signs.sign_features(None) == signs.NO_SIGN == (2.0, 0.0)
    assert signs.sign_features(-1) == signs.NO_SIGN and signs.sign_features(float('inf')) == signs.NO_SIGN
    assert signs.sign_features(0) == (0.0, 1.0) and signs.sign_features(30) == (0.5, 1.0)
    assert signs.sign_features(120) == (2.0, 1.0) and signs.sign_features(120.5) == signs.NO_SIGN
    # A dataset frame is its segment's minute and its index; a sign counts from the frame it was confirmed at.
    frames = [{'segment': segment, 'index': index} for segment in range(4) for index in range(60)]
    found = signs.frame_sign_features(frames, [61.0, 10.0])
    assert found.dtype.name == 'float32' and found.shape == (240, 2)
    assert found[9].tolist() == [2.0, 0.0] and found[10].tolist() == [0.0, 1.0]
    assert found[60].tolist() == pytest.approx([50 / 60, 1.0]) and found[61].tolist() == [0.0, 1.0]
    assert found[181].tolist() == [2.0, 1.0] and found[182].tolist() == [2.0, 0.0]


def test_the_map_puts_the_motorway_sign_at_the_start_of_every_entry_ramp():
    from nnslr_tools import road_type_signs as signs
    road = [TOWN] * 30 + [RAMP] * 10 + [MOTORWAY] * 40 + [RAMP] * 10 + [MOTORWAY] * 30  # an entry, then an interchange
    frames = [{'segment': i // 60, 'index': i % 60, 'label': label} for i, label in enumerate(road)]
    assert signs.map_sign_times(frames, ROAD_TYPES) == [30.0]
    ages = signs.sign_ages([29, 30, 31.5, 200], [30.0])
    assert ages[0] == float('inf') and list(ages[1:]) == [0.0, 1.5, 170.0]
    assert signs.features_at([29, 30, 150, 151], [30.0]).tolist() == [[2.0, 0.0], [0.0, 1.0], [2.0, 1.0], [2.0, 0.0]]
    assert signs.features_at([1, 2], []).tolist() == [[2.0, 0.0], [2.0, 0.0]]


def test_confirmations_are_those_of_the_cars_memory():
    from nnslr_tools import road_type_signs as signs
    package = signs.package_path()
    if not (package / 'vision.py').is_file():
        pytest.skip('no sunnypilot clone with the car code')
    vision = signs.load_vision(package)
    start = [['other_sign:motorway_start', 0.9]]
    readings = [(10.0, start), (10.2, start), (10.4, start),  # a sign read three times: confirmed at its 2nd and 3rd
                (30.0, start),  # once only: not confirmed
                (50.0, [['other_sign:town_entry', 0.9]]), (50.2, [['other_sign:town_entry', 0.9]]),  # not a motorway
                (70.0, start), (72.0, start)]  # twice but 2 s apart: not confirmed
    assert signs.confirmations(vision, readings) == [[10.2, 'motorway'], [10.4, 'motorway']]
    assert signs.sign_times({'confirmed': [[10.2, 'motorway'], [50.2, 'town']]}) == [10.2]


# -- the comma's rule on the merges (road_type_metrics) ---------------------------------------------------------------
def _entry_drive():
    """60 s of town, 20 s of ramp, 100 s of motorway."""
    import numpy as np
    return np.array([TOWN] * 60 + [RAMP] * 20 + [MOTORWAY] * 100)


def test_ramp_entries_find_the_ramp_and_what_came_before():
    import numpy as np

    from nnslr_tools import road_type_metrics as metrics
    labels = _entry_drive()
    assert metrics.ramp_entries(labels, ROAD_TYPES) == [{'ramp': 60, 'merge': 80, 'before': 'town'}]
    # A ramp of two seconds, or one that leaves the motorway, is no entry.
    assert metrics.ramp_entries(np.array([TOWN] * 5 + [RAMP] * 2 + [MOTORWAY] * 5), ROAD_TYPES) == []
    assert metrics.ramp_entries(np.array([MOTORWAY] * 5 + [RAMP] * 5 + [TOWN] * 5), ROAD_TYPES) == []


def test_comma_release_on_time_on_the_ramp_late_and_not_held():
    import numpy as np

    from nnslr_tools import road_type_metrics as metrics
    entry = {'ramp': 60, 'merge': 80, 'before': 'town'}
    no_sign = np.full(180, np.inf)
    # The road type says ramp at the number (second 62) and motorway from the merge: released 8 s later.
    found = metrics.comma_release(_entry_drive(), no_sign, entry, ROAD_TYPES)
    assert found == {'number_at': -18, 'held_by': 'ramp', 'release': 8, 'outcome': 'in_time'}
    # Motorway from second 65, still on the ramp: a false general limit on the ramp.
    early = _entry_drive()
    early[65:80] = MOTORWAY
    assert metrics.comma_release(early, no_sign, entry, ROAD_TYPES)['release'] == -7
    assert metrics.comma_release(early, no_sign, entry, ROAD_TYPES)['outcome'] == 'on_ramp'
    # The road type says town at the number and no sign: the number is not a ramp's, nothing to release.
    town = _entry_drive()
    town[60:70] = TOWN
    assert metrics.comma_release(town, no_sign, entry, ROAD_TYPES)['outcome'] == 'not_held'
    # A motorway sign 20 s before the number holds it; town after the merge breaks the 8 s, so it is late.
    sign = np.full(180, np.inf)
    sign[42:] = np.arange(180 - 42)
    late = town.copy()
    late[80:] = np.where(np.arange(100) % 6 == 5, TOWN, MOTORWAY)
    result = metrics.comma_release(late, sign, entry, ROAD_TYPES)
    assert result['held_by'] == 'sign' and result['release'] is None and result['outcome'] == 'late'
    both = metrics.comma_release(_entry_drive(), sign, entry, ROAD_TYPES)
    assert both['held_by'] == 'ramp+sign' and both['outcome'] == 'in_time'
    summary = metrics.release_summary([found, result, both,
                                       metrics.comma_release(town, no_sign, entry, ROAD_TYPES)])
    assert summary['merges'] == 4 and summary['held'] == 3 and summary['held_by_sign'] == 2
    assert (summary['in_time'], summary['late'], summary['not_held'], summary['on_ramp']) == (2, 1, 1, 0)
    assert summary['median_release'] == 8.0 and summary['releases'] == [8, None, 8, None]


def test_confusion_counts_the_masked_frames_by_truth_and_answer():
    import numpy as np

    from nnslr_tools import road_type_metrics as metrics
    truth = np.array([0, 0, 1, 1, 2, -1])
    answer = np.array([0, 1, 1, 1, 0, 2])
    mask = np.array([1, 1, 1, 0, 1, 1])
    assert metrics.confusion(truth, answer, mask, 3) == [[1, 1, 0], [0, 1, 0], [1, 0, 0]]


def test_onnx_step_contract_with_the_sign_input(root, tmp_path):
    torch = pytest.importorskip('torch')
    pytest.importorskip('onnxruntime')
    pytest.importorskip('torchvision')
    import numpy as np

    from nnslr_tools import road_type_data as data
    from nnslr_tools import road_type_net as net
    model = net.build_road_type_net('gru', pretrained=False, signs=True).eval()
    assert model.gru.input_size == net.FEATURES + 2 + net.SIGN_INPUTS
    checkpoint = tmp_path / 'road-type-best.pt'
    torch.save({'task': 'road_type', 'architecture': 'mobilenet_v3_small+gru', 'memory': 'gru', 'signs': True,
                'classes': list(data.CLASSES), 'state_dict': model.state_dict(), 'epoch': 1,
                'split_sha256': 's', 'dataset_sha256': 'd'}, checkpoint)
    result = net.export_road_type_onnx(checkpoint, tmp_path / 'road-type.onnx', data_root=root)
    contract = json.loads(Path(result['contract']).read_text())
    assert contract['verification']['passed'] is True and contract['step']['signs'] is True
    assert [i['name'] for i in contract['inputs']] == ['image', 'speed', 'sign', 'hidden']
    assert any('_road_sign_at' in line for line in contract['car_input']['steps'])
    import onnxruntime
    session = onnxruntime.InferenceSession(result['onnx'], providers=['CPUExecutionProvider'])
    image = np.random.default_rng(0).random((1, 3, data.FRAME_HEIGHT, data.FRAME_WIDTH), dtype=np.float32)
    feeds = {'image': image, 'speed': np.array([[0.5, 1.0]], np.float32),
             'hidden': np.zeros((1, net.HIDDEN), np.float32)}
    seen = session.run(None, {**feeds, 'sign': np.array([[0.1, 1.0]], np.float32)})[0]
    unseen = session.run(None, {**feeds, 'sign': np.array([[2.0, 0.0]], np.float32)})[0]
    assert seen.shape == (1, len(data.CLASSES)) and not np.allclose(seen, unseen)


# -- the junction head (nnslr_tools.junctions labels, road_type_metrics junction measure) -----------------------------
def test_junction_events_and_their_matching():
    import numpy as np

    from nnslr_tools import road_type_metrics as metrics
    p_passed = np.array([0, .8, .9, 0, .6, 0, .8, 0, 0, 0])
    p_in = np.array([0, .9, .9, .1, 0, 0, .9, .9, 0, 0])
    # A rising edge of "passed"; a second one within 3 s is the same intersection said twice.
    assert metrics.junction_events(p_passed, p_in, .5) == [1, 4]
    assert metrics.junction_events(p_passed, p_in, .7) == [1, 6]
    # The exit decode: the first second below the threshold after one at or above it.
    assert metrics.junction_events(p_passed, p_in, .5, 'exit') == [3, 8]
    with pytest.raises(ValueError):
        metrics.junction_events(p_passed, p_in, .5, 'enter')
    # One to one, the closest pairs first.
    assert metrics.match_events([10.0, 20.0], [11.0, 25.0], 2.0) == [(0, 0)]
    assert sorted(metrics.match_events([10.0, 20.0], [11.0, 25.0], 5.0)) == [(0, 0), (1, 1)]
    assert metrics.match_events([10.0, 10.5], [10.4], 2.0) == [(1, 0)]


def test_junction_metrics_count_false_passed_per_hour_and_per_10_km():
    import numpy as np

    from nnslr_tools import road_type_metrics as metrics
    # 360 s at 10 m/s (0.1 h, 3.6 km), town then rural; passed at 50 and 250.
    count = 360
    labels = np.array([TOWN] * 180 + [RURAL] * 180)
    passed = np.zeros(count, bool)
    passed[[50, 250]] = True
    p_passed = np.zeros(count)
    p_passed[51] = .9                 # found 1 s late
    p_passed[120] = p_passed[300] = .6  # false: one in town, one outside
    p_passed[330] = .9                # on a frame not judged (after a passage whose exit was not seen)
    ignore = np.zeros(count, bool)
    ignore[325:335] = True
    sequence = {'seconds': np.arange(count, dtype=float), 'p_in': np.zeros(count), 'p_passed': p_passed,
                'in': np.zeros(count, bool), 'passed': passed, 'known': np.ones(count, bool), 'ignore': ignore,
                'labels': labels, 'predicted': labels, 'speed': np.full(count, 10.0)}
    found = metrics.junction_metrics([sequence], ROAD_TYPES)
    assert (found['passed'], found['hours'], found['km']) == (2, 0.1, 3.6)
    at_half = found['decodes']['passed']['0.5']
    assert (at_half['events'], at_half['recall_2s'], at_half['recall_4s'], at_half['precision_2s']) == (3, .5, .5, .3333)
    assert at_half['false'] == 2 and at_half['false_per_hour'] == 20.0 and at_half['false_per_10km'] == 5.56
    assert at_half['town'] == {'false': 1, 'per_hour': 20.0, 'per_10km': 5.56}
    assert at_half['outside'] == {'false': 1, 'per_hour': 20.0, 'per_10km': 5.56}
    strict = found['decodes']['passed']['0.7']
    assert (strict['events'], strict['false'], strict['f1_2s']) == (1, 0, .6667)
    # No exit events at all: F1 0, not unknown.
    assert found['decodes']['exit']['0.5']['f1_2s'] == 0.0
    # The car's point, chosen on validation: the best F1 at 2 s.
    chosen = metrics.operating_point(found)
    assert (chosen['decode'], chosen['threshold'], chosen['false_per_hour']) == ('passed', .7, 0.0)


def test_route_frames_carry_the_junction_labels_with_a_persons_corrections(root):
    from nnslr_tools import road_type_data as data
    from nnslr_tools.junctions import VERSION, correct, junctions_path
    route = OTHER_ROUTES[0]
    # Without nnslr junctions: no junction label is known.
    assert not any(frame['junction_known'] for frame in data.route_frames(root, route))
    frames = [{'frame_key': f'{s}/{i}', 'segment_index': s, 'output_index': i, 'known': True, 'in_junction': False,
               'passed': False, 'junction': None, 'distance_m': float(20 * (60 * s + i))}
              for s in range(2) for i in range(60)]
    for i in (10, 11):
        frames[i].update(in_junction=True, junction='J1')
    frames[12].update(passed=True, junction='J1')
    frames[70].update(in_junction=True, junction='J2')
    frames[90]['known'] = False
    passages = [{'id': 'J1', 'kind': 'crossing', 't_in': 9.6, 't_out': 11.4, 'passed_key': '0/12', 'closest_m': 1.0,
                 'frames_inside': 2},
                # The GPS stops in this one: no "passed" labelled, so none judged up to 4 s after it.
                {'id': 'J2', 'kind': 'crossing', 't_in': 69.6, 't_out': 70.4, 'passed_key': None, 'closest_m': 1.0,
                 'frames_inside': 1}]
    path = junctions_path(root, route)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'route_id': route, 'version': VERSION, 'rules': {}, 'passages': passages,
                                'rejected': [], 'frames': frames}))
    found = data.route_frames(root, route)
    assert [i for i, f in enumerate(found) if f['junction_in']] == [10, 11, 70]
    assert [i for i, f in enumerate(found) if f['junction_passed']] == [12]
    assert [i for i, f in enumerate(found) if f['junction_ignore']] == list(range(70, 75))
    assert found[12]['junction'] == 'J1' and found[12]['distance_m'] == 240.0
    assert found[89]['junction_known'] and not found[90]['junction_known']
    # "Not a junction" on the review page: its frames are no junction for the training either.
    correct(root, route, '0/11', 'none', 'admin')
    found = data.route_frames(root, route)
    assert [i for i, f in enumerate(found) if f['junction_in'] or f['junction_passed']] == [70]


def test_onnx_step_contract_with_the_junction_head(root, tmp_path):
    torch = pytest.importorskip('torch')
    pytest.importorskip('onnxruntime')
    pytest.importorskip('torchvision')
    import numpy as np

    from nnslr_tools import road_type_data as data
    from nnslr_tools import road_type_net as net
    model = net.build_road_type_net('gru', pretrained=False, signs=True, junctions=True).eval()
    checkpoint = tmp_path / 'road-type-best.pt'
    torch.save({'task': 'road_type', 'architecture': 'mobilenet_v3_small+gru', 'memory': 'gru', 'signs': True,
                'junctions': True, 'classes': list(data.CLASSES), 'state_dict': model.state_dict(), 'epoch': 1,
                'split_sha256': 's', 'dataset_sha256': 'd'}, checkpoint)
    (tmp_path / 'result.json').write_text(json.dumps({'junction_operating': {'decode': 'exit', 'threshold': 0.3}}))
    result = net.export_road_type_onnx(checkpoint, tmp_path / 'road-type.onnx', data_root=root)
    contract = json.loads(Path(result['contract']).read_text())
    assert contract['verification']['passed'] is True and contract['step']['junctions'] is True
    # The road type's outputs keep their names and order; the junction comes after them.
    assert [o['name'] for o in contract['outputs']] == ['probabilities', 'hidden_out', 'junction']
    junction = contract['outputs'][2]
    assert junction['names'] == ['in_junction', 'passed'] and junction['operating'] == {'decode': 'exit',
                                                                                       'threshold': 0.3}
    assert 'in_junction < 0.3' in junction['decode']
    import onnxruntime
    session = onnxruntime.InferenceSession(result['onnx'], providers=['CPUExecutionProvider'])
    image = np.random.default_rng(0).random((1, 3, data.FRAME_HEIGHT, data.FRAME_WIDTH), dtype=np.float32)
    probabilities, hidden_out, crossing = session.run(None, {
        'image': image, 'speed': np.array([[0.5, 1.0]], np.float32), 'sign': np.array([[2.0, 0.0]], np.float32),
        'hidden': np.zeros((1, net.HIDDEN), np.float32)})
    assert probabilities.shape == (1, len(data.CLASSES)) and abs(float(probabilities.sum()) - 1) < 1e-5
    assert crossing.shape == (1, 2) and ((crossing >= 0) & (crossing <= 1)).all()
