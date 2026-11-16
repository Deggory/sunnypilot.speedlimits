"""The step of the brain as the car runs it, on an untrained network. Skipped in .venv: run it as test_brain_net.py says."""
import json

import pytest

pytest.importorskip('torch')
pytest.importorskip('onnx')
pytest.importorskip('onnxruntime')

from nnslr_tools import brain_export  # noqa: E402
from nnslr_tools.brain_targets import inputs_sha256  # noqa: E402

GROUPS = ('signs', 'plate', 'hour', 'lead', 'country')


def checkpoint(path, monkeypatch, ahead=False, **changes):
    """load_brain giving an untrained network, and what a checkpoint says of itself."""
    from nnslr_tools.brain_net import build_brain_net
    model = build_brain_net(GROUPS, works=False, ahead=ahead, pretrained=False).eval()
    saved = {'groups': GROUPS, 'hidden': model.gru.hidden_size, 'inputs_sha256': inputs_sha256(), 'epoch': 0, 'split': 'x',
             'reader': None, **changes}
    path.write_bytes(b'weights')
    monkeypatch.setattr(brain_export, 'load_brain', lambda _: (model, saved))
    return path


def test_refuses_a_network_trained_with_another_file_of_the_inputs(tmp_path, monkeypatch):
    path = checkpoint(tmp_path / 'brain-last.pt', monkeypatch, inputs_sha256='0' * 64)
    with pytest.raises(ValueError, match='train it again'):
        brain_export.export_brain_onnx(path, tmp_path / 'brain.onnx')


def test_refuses_a_network_with_an_input_the_car_does_not_give(tmp_path, monkeypatch):
    path = checkpoint(tmp_path / 'brain-last.pt', monkeypatch, groups=(*GROUPS, 'wheel'))
    with pytest.raises(ValueError, match='does not give'):
        brain_export.export_brain_onnx(path, tmp_path / 'brain.onnx')


def test_the_contract_says_what_the_car_reads(tmp_path, monkeypatch):
    path = checkpoint(tmp_path / 'brain-last.pt', monkeypatch)
    done = brain_export.export_brain_onnx(path, tmp_path / 'brain.onnx')
    contract = json.loads((tmp_path / 'brain.onnx.contract.json').read_text())
    assert done['verification']['passed']
    assert contract['inputs_sha256'] == inputs_sha256()
    assert contract['step']['groups'] == list(GROUPS) and contract['step']['rate_hz'] == 1.0
    assert contract['output']['kph'] == list(range(1, 201)) and contract['output']['shape'] == [1, 200]


def test_the_limit_ahead_has_none_after_the_last_speed(tmp_path, monkeypatch):
    path = checkpoint(tmp_path / 'brain-last.pt', monkeypatch, ahead=True)
    brain_export.export_brain_onnx(path, tmp_path / 'brain.onnx')
    contract = json.loads((tmp_path / 'brain.onnx.contract.json').read_text())
    shapes = {output['name']: output['shape'] for output in contract['outputs']}
    assert shapes['speed'] == [1, 200] and shapes['next'] == [1, 201] and shapes['metres'] == [1, 1]


def saved_brain(path, kph=None, ahead=True, road_kinds=(), users=False):
    """A checkpoint as brain_train writes it, works_head (and road_head with road_kinds) and its table of the speeds among
    the weights."""
    import torch

    from nnslr_tools.brain_net import build_brain_net
    torch.manual_seed(0)
    model = build_brain_net(GROUPS, pool='grid', works=True, ahead=ahead, pretrained=False, kph=kph,
                            road_kinds=len(road_kinds), users=users)
    torch.save({'weights': model.state_dict(), 'epoch': 0, 'groups': GROUPS, 'pool': 'grid', 'works': True,
                'road_kinds': list(road_kinds), 'features': 128, 'hidden': 128, 'ahead': ahead, 'rate_hz': 4.0,
                'picture': 'brain', 'split': 'x', 'reader': None, 'inputs_sha256': inputs_sha256()}, path)
    return model


def test_a_brain_that_learnt_the_kinds_of_road_goes_to_the_car_as_one_that_did_not(tmp_path):
    # road_head is only in training: the car's network, its weights and its contract are those of a brain without it.
    import torch

    from nnslr_tools.brain_net import car_weights
    with_road, without = tmp_path / 'road' / 'brain-last.pt', tmp_path / 'plain' / 'brain-last.pt'
    with_road.parent.mkdir()
    without.parent.mkdir()
    trained = saved_brain(with_road, road_kinds=('narrow', 'country', 'gravel', 'potholes'))
    assert trained.road_head is not None and 'road_head.weight' in torch.load(with_road, weights_only=False)['weights']
    saved_brain(without)
    model, _saved = brain_export.load_brain(with_road)
    plain, _ = brain_export.load_brain(without)
    assert model.road_head is None and model.works_head is None
    weights, kept = car_weights(model), car_weights(trained)
    assert set(weights) == set(car_weights(plain)) == set(kept)
    assert all(torch.equal(weights[name], kept[name]) for name in weights)
    contracts = []
    for path in (with_road, without):
        brain_export.export_brain_onnx(path, path.parent / 'brain.onnx')
        contract = json.loads((path.parent / 'brain.onnx.contract.json').read_text())
        contracts.append((contract['inputs'], contract['outputs'], contract['output']))
    assert contracts[0] == contracts[1]


def test_a_brain_that_learnt_the_road_users_goes_to_the_car_as_one_that_did_not(tmp_path):
    # users_head is only in training (brain_net.TRAINING_ONLY): the car's network, weights and contract are the same.
    import torch

    from nnslr_tools.brain_net import car_weights
    with_users, without = tmp_path / 'users' / 'brain-last.pt', tmp_path / 'plain' / 'brain-last.pt'
    with_users.parent.mkdir()
    without.parent.mkdir()
    trained = saved_brain(with_users, users=True)
    assert 'users_head.weight' in torch.load(with_users, weights_only=False)['weights']
    saved_brain(without)
    model, _saved = brain_export.load_brain(with_users)
    assert model.users_head is None
    weights, kept = car_weights(model), car_weights(trained)
    assert set(weights) == set(kept) and not [name for name in kept if name.startswith('users_head')]
    assert all(torch.equal(weights[name], kept[name]) for name in weights)
    contracts = []
    for path in (with_users, without):
        brain_export.export_brain_onnx(path, path.parent / 'brain.onnx')
        contract = json.loads((path.parent / 'brain.onnx.contract.json').read_text())
        contracts.append((contract['inputs'], contract['outputs'], contract['output']))
    assert contracts[0] == contracts[1]


def test_a_brain_of_5_km_h_classes_loads_with_its_own_table_and_says_it_to_the_car(tmp_path):
    import torch
    path = tmp_path / 'brain-last.pt'
    trained = saved_brain(path, kph=list(range(5, 205, 5)))
    model, saved = brain_export.load_brain(path)
    assert model.class_kph.tolist() == [float(kph) for kph in range(5, 205, 5)]
    assert (model.speed_head.out_features, model.next_head.out_features) == (40, 41)
    assert torch.equal(model.speed_head.weight, trained.speed_head.weight) and model.works_head is None
    brain_export.export_brain_onnx(path, tmp_path / 'brain.onnx')
    contract = json.loads((tmp_path / 'brain.onnx.contract.json').read_text())
    assert contract['output']['kph'] == list(range(5, 205, 5)) and contract['output']['shape'] == [1, 40]
    assert {output['name']: output['shape'] for output in contract['outputs']}['next'] == [1, 41]


def test_a_brain_of_today_loads_with_a_class_a_km_h(tmp_path):
    path = tmp_path / 'brain-last.pt'
    saved_brain(path)
    model, _saved = brain_export.load_brain(path)
    assert brain_export.table_of(model) == list(range(1, 201)) and model.next_head.out_features == 201


def test_along_a_drive_the_step_of_a_brain_of_before_says_its_own_km_h(tmp_path):
    import numpy

    from nnslr_tools.brain_targets import INPUT_SIZES
    from nnslr_tools.road_type_net import FrameStore
    path = tmp_path / 'brain-last.pt'
    saved_brain(path, kph=list(range(5, 205, 5)))
    model, _saved = brain_export.load_brain(path)
    step = brain_export.step_module(model)
    frames = 5
    pictures = numpy.random.default_rng(0).integers(0, 255, (frames, 144, 256, 3), dtype=numpy.uint8)
    width = sum(INPUT_SIZES[name] for name in GROUPS)
    drive = {'route': 'r', 'frames': [{}] * frames, 'partitions': ['test'] * frames, 'cuts': {3},
             'extra': numpy.random.default_rng(1).random((frames, width), dtype=numpy.float32)}

    def run(image, extra, memory):
        import torch
        with torch.no_grad():
            return [value.numpy() for value in step(*(torch.from_numpy(value) for value in (image, extra, memory)))]

    import torch
    threads, heard = torch.get_num_threads(), []
    found = brain_export.drive_rollout(model, run, drive, FrameStore({'r': pictures}, 'cpu'), log=heard.append)
    assert found['other_speed'] == 0 and found['other_ahead'] == 0 and found['frames'] == frames
    assert found['said'] and all(kph % 5 == 0 for kph in found['said'])
    # The log hears the drive and its progress; PyTorch has its threads back.
    assert heard[0] == 'brain check along r: 5 frames, one at a time as on the car'
    assert heard[1:] == [f'brain check: {n} of 5 frames' for n in range(1, 5)]
    assert torch.get_num_threads() == threads
    # Wherever PyTorch ran the drive, the network is back on the CPU for the step's other checks.
    assert next(model.parameters()).device.type == 'cpu'

    # A frame where ONNX gives another class a hair more than PyTorch's speed is a tie, not another speed.
    def nudged(by):
        def run_nudged(image, extra, memory):
            outputs = run(image, extra, memory)
            scores = outputs[0].reshape(-1)
            scores[(int(scores.argmax()) + 1) % len(scores)] = scores.max() + by
            return outputs
        return run_nudged

    found = brain_export.drive_rollout(model, nudged(brain_export.TIE / 10), drive, FrameStore({'r': pictures}, 'cpu'))
    assert found['other_speed'] == 0 and found['tied_speed'] == frames
    found = brain_export.drive_rollout(model, nudged(1.0), drive, FrameStore({'r': pictures}, 'cpu'))
    assert found['other_speed'] == frames and found['tied_speed'] == 0


def test_the_checks_onnxruntime_runs_on_one_thread_not_spinning(tmp_path):
    import onnxruntime
    path = tmp_path / 'brain-last.pt'
    saved_brain(path)
    brain_export.export_brain_onnx(path, tmp_path / 'brain.onnx')
    session = brain_export.check_session(onnxruntime, tmp_path / 'brain.onnx')
    options = session.get_session_options()
    assert options.intra_op_num_threads == 1 and options.inter_op_num_threads == 1
    assert options.get_session_config_entry('session.intra_op.allow_spinning') == '0'


def test_a_network_of_an_earlier_file_with_the_same_groups_exports_with_its_own_hash(tmp_path, monkeypatch):
    from nnslr_tools.brain_targets import BRAIN_INPUTS
    earlier = BRAIN_INPUTS.SAME_GROUPS[0]
    path = checkpoint(tmp_path / 'brain-last.pt', monkeypatch, inputs_sha256=earlier)
    brain_export.export_brain_onnx(path, tmp_path / 'brain.onnx')
    assert json.loads((tmp_path / 'brain.onnx.contract.json').read_text())['inputs_sha256'] == earlier


@pytest.mark.parametrize('kinds', [None, 50])
def test_a_brain_exports_with_the_kinds_of_other_sign_it_was_trained_with(tmp_path, monkeypatch, kinds):
    """#103: a brain of before 16 Nov 2026 does not say its kinds: it exports 98 numbers of road_signs and a contract
    without the word (the one on the car stays as it was). One trained since says them, and the contract tells the car."""
    from nnslr_tools.brain_net import build_brain_net
    from nnslr_tools.brain_targets import brain_sizes
    groups = ('signs', 'road_signs', 'lead')
    said = {} if kinds is None else {'road_sign_kinds': kinds}
    sizes = brain_sizes(said)
    model = build_brain_net(groups, works=False, pretrained=False, sizes=sizes).eval()
    saved = {'groups': groups, 'hidden': model.gru.hidden_size, 'inputs_sha256': inputs_sha256(), 'epoch': 0, 'split': 'x',
             'reader': None, **said}
    path = tmp_path / 'brain-last.pt'
    path.write_bytes(b'weights')
    monkeypatch.setattr(brain_export, 'load_brain', lambda _: (model, saved))
    brain_export.export_brain_onnx(path, tmp_path / 'brain.onnx')
    contract = json.loads((tmp_path / 'brain.onnx.contract.json').read_text())
    width = sum(sizes[name] for name in groups)
    assert sizes['road_signs'] == 2 * (49 if kinds is None else kinds)
    assert next(item for item in contract['inputs'] if item['name'] == 'extra')['shape'] == [1, width]
    assert contract['step'].get('road_sign_kinds') == kinds and ('road_sign_kinds' in contract['step']) == (kinds is not None)


@pytest.mark.parametrize('saved, hours', [({}, False), ({'hours': True}, True)])
def test_the_contract_says_whether_the_car_gives_the_hours(tmp_path, monkeypatch, saved, hours):
    path = checkpoint(tmp_path / 'brain-last.pt', monkeypatch, **saved)
    brain_export.export_brain_onnx(path, tmp_path / 'brain.onnx')
    assert json.loads((tmp_path / 'brain.onnx.contract.json').read_text())['step']['hours'] is hours


def test_a_brain_that_learnt_whose_road_a_sign_is_of_goes_to_the_car_without_that_head(tmp_path):
    """#89, stage 1: applicability_head is only in training (brain_net.TRAINING_ONLY); the car's file has the speed alone,
    and its contract asks the car for the boxes of every sign read."""
    import torch

    from nnslr_tools.brain_net import build_brain_net, car_weights
    groups = (*GROUPS, 'sign_boxes')
    torch.manual_seed(0)
    trained = build_brain_net(groups, pool='grid', works=True, pretrained=False, applicability=True)
    path = tmp_path / 'brain-last.pt'
    torch.save({'weights': trained.state_dict(), 'epoch': 0, 'groups': groups, 'pool': 'grid', 'works': True, 'features': 128,
                'hidden': 128, 'rate_hz': 4.0, 'picture': 'brain', 'split': 'x', 'reader': None, 'applicability': True,
                'other_road_boxes': True, 'inputs_sha256': inputs_sha256()}, path)
    assert 'applicability_head.weight' in torch.load(path, weights_only=False)['weights']
    model, _saved = brain_export.load_brain(path)
    assert model.applicability_head is None
    weights, kept = car_weights(model), car_weights(trained)
    assert set(weights) == set(kept) and not [name for name in kept if name.startswith('applicability_head')]
    assert all(torch.equal(weights[name], kept[name]) for name in weights)
    brain_export.export_brain_onnx(path, tmp_path / 'brain.onnx')
    contract = json.loads((tmp_path / 'brain.onnx.contract.json').read_text())
    assert contract['step']['other_road_boxes'] is True and contract['step']['outputs'] == ['speed', 'hidden_out']


@pytest.mark.parametrize('saved, every', [({}, False), ({'other_road_boxes': False}, False), ({'other_road_boxes': True}, True)])
def test_the_contract_says_whether_the_car_gives_the_boxes_of_every_sign(tmp_path, monkeypatch, saved, every):
    """#89: a brain that learnt whose road a sign is of was given the boxes of the signs of another road too; the car
    gives them only to a brain whose contract says so (brain.BrainReader), a brain of before keeps the signs taken."""
    path = checkpoint(tmp_path / 'brain-last.pt', monkeypatch, **saved)
    brain_export.export_brain_onnx(path, tmp_path / 'brain.onnx')
    assert json.loads((tmp_path / 'brain.onnx.contract.json').read_text())['step']['other_road_boxes'] is every


@pytest.mark.parametrize('saved, plates', [({}, False), ({'plates': False}, False), ({'plates': True}, True)])
def test_the_contract_says_whether_the_car_gives_the_plates(tmp_path, monkeypatch, saved, plates):
    path = checkpoint(tmp_path / 'brain-last.pt', monkeypatch, **saved)
    brain_export.export_brain_onnx(path, tmp_path / 'brain.onnx')
    contract = json.loads((tmp_path / 'brain.onnx.contract.json').read_text())
    assert contract['step']['plates'] is plates


def test_a_brain_that_learnt_the_road_type_gives_it_to_the_screen_with_the_same_speed(tmp_path):
    # type_head goes in the ONNX as road_type, before hidden_out, with its names in the contract; the speed's weights stay.
    import torch

    from nnslr_tools.brain_net import build_brain_net, car_weights
    path = tmp_path / 'brain-last.pt'
    torch.manual_seed(0)
    trained = build_brain_net(GROUPS, pool='grid', works=True, ahead=True, pretrained=False, road_types=3)
    torch.save({'weights': trained.state_dict(), 'epoch': 0, 'groups': GROUPS, 'pool': 'grid', 'works': True,
                'road_kinds': [], 'road_types': ['town', 'rural', 'motorway'], 'features': 128, 'hidden': 128,
                'ahead': True, 'rate_hz': 4.0, 'picture': 'brain', 'split': 'x', 'reader': None,
                'inputs_sha256': inputs_sha256()}, path)
    model, _saved = brain_export.load_brain(path)
    assert torch.equal(model.type_head.weight, trained.type_head.weight)
    assert set(car_weights(model)) == set(car_weights(trained))
    done = brain_export.export_brain_onnx(path, tmp_path / 'brain.onnx')
    contract = json.loads((tmp_path / 'brain.onnx.contract.json').read_text())
    assert done['verification']['passed']
    assert contract['output']['road_types'] == ['town', 'rural', 'motorway']
    assert [output['name'] for output in contract['outputs']] == ['speed', 'next', 'metres', 'road_type', 'hidden_out']
    assert contract['step']['outputs'] == ['speed', 'next', 'metres', 'road_type', 'hidden_out']


def test_a_brain_with_the_maps_path_steps_on_the_car_as_in_its_training(tmp_path):
    """trust_map goes to the car in the file: the step the car runs says the speeds forward says, with the map and without."""
    import numpy as np
    import torch

    from nnslr_tools.brain_net import build_brain_net
    from nnslr_tools.brain_targets import BRAIN_INPUTS, INPUT_SIZES
    groups = (*GROUPS, 'map_limit')
    torch.manual_seed(0)
    trained = build_brain_net(groups, pool='grid', works=True, pretrained=False, map_path=True)
    with torch.no_grad():
        trained.trust_map.weight.normal_()
        trained.trust_map.bias.fill_(2.0)
    path = tmp_path / 'brain-last.pt'
    torch.save({'weights': trained.state_dict(), 'epoch': 0, 'groups': groups, 'pool': 'grid', 'works': True,
                'road_kinds': [], 'features': 128, 'hidden': 128, 'rate_hz': 4.0, 'picture': 'brain', 'split': 'x',
                'reader': None, 'inputs_sha256': inputs_sha256()}, path)
    model, _saved = brain_export.load_brain(path)
    assert model.trust_map is not None and torch.equal(model.trust_map.bias, trained.trust_map.bias)
    step = brain_export.step_module(model)
    width = sum(INPUT_SIZES[name] for name in groups)
    image, hidden = torch.rand(1, 3, 144, 256), torch.zeros(1, 128)
    for kph in (80, None):
        extra = torch.zeros(1, width)
        extra[0, -2:] = torch.from_numpy(BRAIN_INPUTS.map_limit_row(kph))
        with torch.no_grad():
            stepped = step(image, extra, hidden)[0]
            whole = model(image[None], extra[None])[0][0]
        assert np.allclose(stepped.numpy(), whole.numpy(), atol=1e-5)
    # The file for the car is made, checked against the network by the export itself, and its contract names the group.
    brain_export.export_brain_onnx(path, tmp_path / 'brain.onnx')
    contract = json.loads((tmp_path / 'brain.onnx.contract.json').read_text())
    assert 'map_limit' in json.dumps(contract)
    # A checkpoint with the map as an input alone (the first one, 14 Nov 2026) still loads: no path, no weights for it.
    plain = build_brain_net(groups, pool='grid', works=True, pretrained=False)
    torch.save({'weights': plain.state_dict(), 'epoch': 0, 'groups': groups, 'pool': 'grid', 'works': True,
                'road_kinds': [], 'features': 128, 'hidden': 128, 'rate_hz': 4.0, 'picture': 'brain', 'split': 'x',
                'reader': None, 'inputs_sha256': inputs_sha256()}, path)
    assert brain_export.load_brain(path)[0].trust_map is None


@pytest.mark.parametrize('kinds, kph', [(None, None), (50, list(range(5, 205, 5)))])
def test_two_of_the_four_comparisons_with_pytorch_carry_a_number_in_force(tmp_path, monkeypatch, kinds, kph):
    """#104: the number counts only where signs says no end was read since, an exact 0 that a random float never is, so
    its direct path to the speed (BrainNet.number_in_force) was off in every comparison of the export. Two of the four
    now carry a number of the brain's own table, wherever its signs are among the inputs (#103: its own sizes); the
    example the exporter is given stays the random one, so the file of a brain keeps the bytes recorded for the car."""
    import torch

    from nnslr_tools.brain_net import build_brain_net
    from nnslr_tools.brain_targets import BRAIN_INPUTS, brain_sizes
    groups = ('lead', 'road_signs', 'signs', 'country')
    said = {} if kinds is None else {'road_sign_kinds': kinds}
    sizes = brain_sizes(said)
    model = build_brain_net(groups, works=False, pretrained=False, sizes=sizes, kph=kph).eval()
    saved = {'groups': groups, 'hidden': model.gru.hidden_size, 'inputs_sha256': inputs_sha256(), 'epoch': 0, 'split': 'x',
             'reader': None, **said}
    path = tmp_path / 'brain-last.pt'
    path.write_bytes(b'weights')
    monkeypatch.setattr(brain_export, 'load_brain', lambda _: (model, saved))
    export, session_of, traced, compared = brain_export._export, brain_export.check_session, [], []

    def heard_export(step, example, output, names):
        traced.append([value.clone() for value in example])
        export(step, example, output, names)

    class Heard:
        def __init__(self, session):
            self.session = session

        def run(self, names, feed):
            compared.append(torch.from_numpy(feed['extra'].copy()))
            return self.session.run(names, feed)

        def get_providers(self):
            return self.session.get_providers()

    monkeypatch.setattr(brain_export, '_export', heard_export)
    monkeypatch.setattr(brain_export, 'check_session', lambda ort, file: Heard(session_of(ort, file)))
    assert brain_export.export_brain_onnx(path, tmp_path / 'brain.onnx')['verification']['passed']
    marked = [model.number_in_force(extra)[0] for extra in compared]
    assert [int(row.sum()) for row in marked] == [0, 1, 0, 1]
    table = brain_export.table_of(model)
    signs_at = sizes['lead'] + sizes['road_signs']
    for extra, row in zip(compared[1::2], marked[1::2]):
        number = table[int(row.argmax())]
        assert extra[0, signs_at].item() == pytest.approx(number / BRAIN_INPUTS.SIGN_KPH_SCALE)
        assert (extra[0, signs_at + 1].item(), extra[0, signs_at + 6].item()) == (1.0, 0.0)
    # What the exporter traced: the random example of always, with no number in force.
    width = sum(sizes[name] for name in groups)
    example = brain_export._random_inputs(torch, width, model.gru.hidden_size, torch.Generator().manual_seed(1))
    assert len(traced) == 1 and all(torch.equal(given, made) for given, made in zip(traced[0], example))
    assert int(model.number_in_force(traced[0][1]).sum()) == 0


def test_a_file_that_lost_the_numbers_path_does_not_pass_the_comparison_with_pytorch(tmp_path, monkeypatch):
    """#104: what the number in force of the comparisons is for. A file that says the speed of the memory alone, the
    number adding nothing, passed as long as every comparison was random: only Publish's check along a drive saw it, and
    the set exported for Models › Drive has none."""
    import torch
    path = checkpoint(tmp_path / 'brain-last.pt', monkeypatch)
    model, _saved = brain_export.load_brain(path)
    with torch.no_grad():
        model.trust.weight.zero_()
        model.trust.bias.fill_(2.0)                         # a number in force adds 2 to its class, whatever the memory
    assert brain_export.export_brain_onnx(path, tmp_path / 'brain.onnx')['verification']['passed']
    export = brain_export._export

    def without_the_number(step, example, output, names):
        with torch.no_grad():
            step.model.trust.bias.zero_()
        try:
            export(step, example, output, names)
        finally:
            with torch.no_grad():
                step.model.trust.bias.fill_(2.0)

    monkeypatch.setattr(brain_export, '_export', without_the_number)
    with pytest.raises(ValueError, match='brain ONNX differs from PyTorch'):
        brain_export.export_brain_onnx(path, tmp_path / 'brain.onnx')
