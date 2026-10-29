"""The road type in the panel: "Train the road type" (scripts/run_road_type.sh) from the Train tab, its progress read
from its log (nnslr_tools.job_progress), and its two networks, with memory and without, in Models
(nnslr_tools.model_roles). Every run is a synthetic folder whose numbers come from the real metric functions of
nnslr_tools.road_type_metrics: nothing is trained. numpy and road_type_metrics are imported inside the tests: collecting
the tests must leave numpy out (tests/cli/test_import_isolation.py).
"""
from __future__ import annotations

import json
import re

from nnslr_tools.job_progress import progress
from nnslr_tools.model_roles import ModelIndex, road_type_digest
from nnslr_tools.panel import Panel, _run_of
from nnslr_tools.road_types import ROAD_TYPES
from nnslr_tools.runs import RunsIndex
from test_panel import checkout, finished, full_log, root  # noqa: F401  (fixtures and helpers)

TOWN, RURAL, EXPRESSWAY, MOTORWAY, RAMP, PARKING = range(6)

LOG = """== Road type with memory ==
run:          2026-10-29_1200-road-type
== 1/6 Check the road types, the split and the pipeline (CPU-only) ==
dry run road_type_gru epoch 1/1 train_loss 1.7000 val_macro_f1 0.1000 val_accuracy 0.2000 5s
== 2/6 Explicit CUDA smoke test ==
== 3/6 Resize the frames to 256x144 ==
frame cache: 30 drives ready
== 4/6 Train with memory (GRU) ==
road_type_gru epoch 15/60 train_loss 0.8000 val_macro_f1 0.5500 val_accuracy 0.7000 12s
"""


def ready_payload(gpu_processes=0):
    return {'actions_enabled': True, 'routes': [{'state': 'pronta', 'confirmed': 4}] * 3, 'reviewed_routes': 3,
            'min_routes': 3, 'system': {'gpu': {'processes': gpu_processes}}}


def road_type_script(checkout):
    script = checkout / 'scripts' / 'run_road_type.sh'
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text('#!/usr/bin/env bash\necho "== Road type with memory =="\necho "run:          $1"\n'
                      'echo "== 4/6 Train with memory (GRU) =="\n'
                      'echo "road_type_gru epoch 2/60 train_loss 1.0 val_macro_f1 0.4 val_accuracy 0.6 6s"\n')
    script.chmod(0o755)
    return script


# -- its progress -------------------------------------------------------------------------------------------------
def test_the_progress_counts_the_epochs_of_each_network_and_not_those_of_the_dry_run():
    result = progress('train_road_type', 'running', LOG)
    assert [(step['key'], step['state']) for step in result['steps']] == [
        ('checks', 'done'), ('frames', 'done'), ('road_type_gru', 'current'), ('road_type_frame', 'todo'), ('export', 'todo')]
    # 5 + 5 done, then 15 of the 60 epochs of the 45 of the network with memory.
    assert result['percent'] == 21.2
    assert result['detail'] == 'Train with memory · epoch 15 of 60'
    assert list(result['curves']) == ['road_type_gru']
    assert [point['epoch'] for point in result['curves']['road_type_gru']['points']] == [15]

    later = LOG + ('road_type_gru stops early after epoch 52: no better validation macro F1 for 15 epochs\n'
                   '== 5/6 Train without memory ==\n'
                   'road_type_frame epoch 30/60 train_loss 0.9000 val_macro_f1 0.5000 val_accuracy 0.7000 10s\n')
    result = progress('train_road_type', 'running', later)
    assert [step['state'] for step in result['steps']] == ['done', 'done', 'done', 'current', 'todo']
    assert result['percent'] == 75.0 and result['detail'] == 'Train without memory · epoch 30 of 60'


# -- "Train the road type" ----------------------------------------------------------------------------------------
def test_the_road_type_is_trained_from_the_train_tab_once_the_road_types_are_found(root, checkout, monkeypatch):
    panel = Panel(root, repo=checkout)
    assert panel._action_states(ready_payload(), [])['train_road_type'] == {
        'label': 'Train the road type', 'allowed': False, 'reason': 'scripts/run_road_type.sh is missing in this installation'}
    road_type_script(checkout)
    assert panel._action_states(ready_payload(), [])['train_road_type']['reason'] == \
        'no road types yet: Data › Roads, Find the road types'
    (root / 'derived' / 'road_types').mkdir(parents=True)
    (root / 'derived' / 'road_types' / 'r1.jsonl').write_text('{}\n')
    assert panel._action_states(ready_payload(gpu_processes=1), [])['train_road_type']['reason'] == \
        'the GPU is busy with another process'
    # Its labels are the map's road types: no confirmed sign, no reviewed drive needed.
    payload = {**ready_payload(), 'routes': [], 'reviewed_routes': 0}
    assert panel._action_states(payload, [])['train_road_type']['allowed'] is True

    monkeypatch.setattr(panel, 'activities', lambda: [])
    job = panel.start('train_road_type', None)
    done = finished(panel, job['id'])
    first = full_log(panel, job['id']).splitlines()[0].split()
    assert first[:2] == ['==', 'scripts/run_road_type.sh'] and re.fullmatch(r'\d{4}-\d{2}-\d{2}_\d{4}-road-type', first[2])
    assert (done['state'], done['run'], done['label']) == ('ok', first[2], 'Train the road type')
    assert _run_of(panel.logs / (job['id'] + '.log'), 'train_road_type') == first[2]


def test_a_road_type_training_started_from_a_terminal_has_the_progress_of_its_run_log(root, checkout):
    panel = Panel(root, repo=checkout)
    log = root / 'runs' / '2026-10-29_1200-road-type' / 'road-type.log'
    log.parent.mkdir(parents=True)
    log.write_text(LOG)
    payload = {'jobs': [], 'activities': [{'group': 'train', 'label': 'road type training', 'pid': 1}]}
    panel._terminal_training(payload)
    assert payload['activities'][0]['progress']['detail'] == 'Train with memory · epoch 15 of 60'


# -- its two networks in Models -----------------------------------------------------------------------------------
def report(predicted_motorway_after_ramp: bool) -> dict:
    """A partition of one drive: town, a ramp, then the motorway; the network says the motorway after the ramp, or
    rural for its first 20 seconds."""
    import numpy as np

    from nnslr_tools.road_type_metrics import after_merge, sequence_metrics
    labels = np.array([TOWN] * 60 + [RAMP] * 20 + [MOTORWAY] * 100)
    predicted = labels.copy()
    predicted[60:80] = MOTORWAY
    if not predicted_motorway_after_ramp:
        predicted[80:100] = RURAL
    sequences = [{'labels': labels, 'weights': np.ones(len(labels)), 'predicted': predicted}]
    return {**sequence_metrics(sequences, ROAD_TYPES), 'after_merge': after_merge(sequences, ROAD_TYPES)}


def result(memory: str, *, smoothed: bool = False) -> dict:
    gru = memory == 'gru'
    value = {'task': 'road_type', 'memory': memory, 'name': 'road_type_' + ('gru' if gru else 'frame'),
             'architecture': 'mobilenet_v3_small+' + ('gru' if gru else 'mlp'), 'classes': list(ROAD_TYPES),
             'best_epoch': 7, 'config': {'epochs': 60}, 'history': [{'epoch': n} for n in range(1, 23)],
             'parameters': 1101478 if gru else 1018406, 'flops_per_step': 83975936,
             'step_seconds': {'device': 'cuda', 'milliseconds': 3.9}, 'dry_run': False,
             'validation': report(gru), 'test': report(gru)}
    if smoothed:
        value['test_smoothed'] = report(True)
    return value


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value if isinstance(value, str) else json.dumps(value))


def test_both_networks_are_versions_of_the_road_type_with_their_numbers(tmp_path):
    root = tmp_path / 'data'
    run = root / 'runs' / '2026-10-29_1200-road-type' / 'road-type'
    write(run / 'gru' / 'road-type-best.pt', 'weights')
    write(run / 'gru' / 'result.json', result('gru'))
    write(run / 'frame' / 'road-type-best.pt', 'weights')
    write(run / 'frame' / 'result.json', result('none', smoothed=True))
    # The dry run of the CPU check is no version.
    write(run / 'dry-run' / 'road-type-best.pt', 'weights')
    runs = RunsIndex(root)
    runs.running_runs = lambda: {}
    models = ModelIndex(runs)

    found = {entry['part']: entry for entry in models.artifacts(('road_type',))['road_type']}
    assert sorted(found) == ['road-type/frame', 'road-type/gru']
    gru, frame = found['road-type/gru'], found['road-type/frame']
    assert (gru['id'], gru['label'], gru['memory'], gru['finished'], gru['onnx']) == (
        'road_type:2026-10-29_1200-road-type/road-type/gru', 'With memory', 'gru', True, False)
    assert (gru['best_epoch'], gru['epochs'], gru['epochs_run'], gru['parameters'], gru['step_ms']) == (7, 60, 22, 1101478, 3.9)
    assert gru['classes'] == list(ROAD_TYPES)
    # With memory the motorway after the ramp is right; without it the first 20 of its 60 seconds are rural.
    assert gru['test']['after_merge'] == {'merges': 1, 'seconds': 60, 'accuracy': 1.0, 'accuracy_debounced': 1.0}
    assert frame['test']['after_merge']['merges'] == 1 and round(frame['test']['after_merge']['accuracy'], 3) == 0.667
    assert gru['test']['transitions']['town_entry'] == {'count': 0, 'found': 0, 'median_delay': None}
    assert gru['test']['transitions']['ramp_to_motorway']['count'] == 1
    assert gru['test']['classes']['expressway'] == {'f1': None, 'support': 0}
    assert frame['test_smoothed'] is not None and gru['test_smoothed'] is None

    write(run / 'export' / 'road-type.onnx', 'onnx')
    write(run / 'export' / 'road-type.onnx.contract.json', {'format': 'road-type-step'})
    found = {entry['part']: entry for entry in models.artifacts(('road_type',))['road_type']}
    assert (found['road-type/gru']['onnx'], found['road-type/frame']['onnx']) == (True, False)

    chosen = models.set_active('road_type', gru['id'], who='admin', via='sign-in')
    assert chosen['id'] == gru['id']
    roles = {role['role']: role for role in models.index()['roles']}
    assert roles['road_type']['label'] == 'Road type' and roles['road_type']['active']['id'] == gru['id']


def test_a_result_of_another_task_is_no_road_type():
    try:
        road_type_digest({'task': 'detector'})
    except ValueError:
        return
    raise AssertionError('a detector result was read as a road type')
