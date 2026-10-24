"""The Models tab's data: every trained cutter, classifier and decider of <data root>/runs by role
(nnslr_tools.model_roles), the model in use per role (<data root>/models/active.json), GET /api/panel/models and
POST /api/panel/models/active.

Every run is a synthetic folder in a temporary data root, the reports built by the real metric functions of
nnslr_tools.evaluation (helpers of test_panel_runs): nothing is trained or evaluated.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import nnslr_tools.model_roles as model_roles
from nnslr_tools.evaluation import (detector_metrics, detector_size_metrics, human_reviewed_metrics, recall_by_width,
                                    score_frame, threshold_metrics)
from nnslr_tools.model_roles import ModelChoiceError, ModelIndex
from nnslr_tools.panel import Panel, PanelError
from nnslr_tools.runs import RunsIndex
from review_helpers import TOKEN, RunningServer, json_of, make_report, sign_in
from test_panel_runs import AI, HUMAN, experiment_result, make_run, write

THRESHOLDS = (0.05, 0.1, 0.25, 0.3, 0.5)
# Signs 10, 15, 30 and 50 px wide. At score 0.1 the 10 and 30 px ones are found and a 60 px box is false; at 0.25
# only the 30 px one is found.
SIZE_FRAMES = [
    (HUMAN, [[100, 100, 110, 110]], [{'box': [100, 100, 110, 110], 'score': 0.15}]),
    (AI, [[0, 0, 15, 15], [200, 200, 230, 230]],
     [{'box': [200, 200, 230, 230], 'score': 0.8}, {'box': [500, 500, 560, 560], 'score': 0.12}]),
    (AI, [[300, 300, 350, 350]], []),
]


def sizes_report(partition: str = 'test', ms: float = 12.5) -> dict:
    """An evaluation of nnslr_tools.evaluation.evaluate_detector as scripts/run_detector_sizes.sh writes it."""
    scored = {value: [{**score_frame(truth, predictions, score_threshold=value, iou_threshold=0.5), 'label_basis': basis}
                      for basis, truth, predictions in SIZE_FRAMES] for value in THRESHOLDS}
    main = scored[0.25]
    return {'schema_version': 1, 'task': 'detector', 'partition': partition, **detector_metrics(main),
            'recall_by_width': recall_by_width(main), 'human_reviewed': human_reviewed_metrics(main, detector_size_metrics),
            'score_threshold': 0.25, 'iou_threshold': 0.5,
            'by_score_threshold': {f'{value:g}': threshold_metrics(scored[value]) for value in THRESHOLDS},
            'input_size': [640, 360], 'input_size_label': '640x360', 'anchor_min_ratio': 0.1, 'checkpoint_epoch': 6,
            'inference': {'device_name': 'Tesla V100-PCIE-32GB', 'mean_ms_per_frame': ms}, 'images': []}


def make_size(run: Path, size: str, *, finished: bool = True, evaluated: bool = True) -> Path:
    folder = run / 'detector-sizes' / size
    write(folder / 'detector-best.pt', 'weights')
    if finished:
        write(folder / 'result.json', {'task': 'detector', 'architecture': 'ssdlite320_mobilenet_v3_large',
                                       'input_size': [int(side) for side in size.split('x')], 'anchor_min_ratio': 0.1,
                                       'augmentation': {'brightness': [0.4, 1.2]}, 'best_epoch': 6,
                                       'best_validation_loss': 1.1, 'initialization': 'imagenet1k_v1',
                                       'config': {'epochs': 20}, 'history': [{'epoch': n, 'validation_loss': 2.0 - n / 10}
                                                                             for n in range(1, 12)]})
    if evaluated:
        write(folder / 'evaluation/detector-test.json', sizes_report('test'))
        write(folder / 'evaluation/detector-validation.json', sizes_report('validation', ms=13.0))
    return folder


def touch(path: Path, seconds: float) -> None:
    os.utime(path, (seconds, seconds))


@pytest.fixture
def root(tmp_path):
    return tmp_path / 'data'


@pytest.fixture
def models(root):
    runs = RunsIndex(root)
    runs.running_runs = lambda: {}
    return ModelIndex(runs)


def pipeline(root: Path, name: str = 'r1') -> Path:
    """A finished training run with the weights of its detector and reader."""
    run = make_run(root, name)
    write(run / 'detector/detector-best.pt', 'weights')
    write(run / 'reader/reader-best.pt', 'weights')
    return run


def experiment(root: Path, name: str = 'expB', variants=('crop', 'crop_geom')) -> Path:
    run = root / 'runs' / name
    for variant in variants:
        write(run / 'experiment-b' / variant / 'result.json', experiment_result(variant, 0.9, 0.6))
        write(run / 'experiment-b' / variant / f'{variant}-best.pt', 'weights')
    return run


def by_role(index: dict) -> dict:
    return {role['role']: role for role in index['roles']}


# -- the index --------------------------------------------------------------------------------------------
def test_without_runs_every_role_is_empty_and_nothing_is_in_use(models):
    index = models.index()
    assert [role['role'] for role in index['roles']] == ['cutter', 'classifier', 'decider']
    assert all(role['versions'] == [] and role['active'] is None for role in index['roles'])
    assert (index['runs_dir_exists'], index['active_problem'], index['cutter_score_threshold']) == (False, None, 0.1)


def test_every_artifact_by_role_with_its_numbers(root, models):
    run = pipeline(root)
    make_size(run, '640x360')
    experiment(root)
    roles = by_role(models.index())
    assert {entry['id'] for entry in roles['cutter']['versions']} == {'cutter:r1/detector', 'cutter:r1/detector-sizes/640x360'}
    cutters = {entry['part']: entry for entry in roles['cutter']['versions']}

    # The detector of the pipeline: its report has no 0.1, so the numbers are at the report's 0.25; the widths come from
    # its frames (detector_report of test_panel_runs: 10 and 30 px found, 15 and 50 px missed, 2 false boxes).
    main = cutters['detector']
    assert (main['label'], main['detail'], main['finished'], main['in_use']) == ('Detector 320 × 320', 'training pipeline', True, False)
    assert (main['input_size'], main['input_size_recorded'], main['anchor_min_ratio'], main['anchor_recorded']) == ('320x320', False, 0.2, False)
    assert main['best_epoch'] == 2 and main['epochs_run'] == 2 and main['trained_at']
    test = main['test']
    assert (test['score_threshold'], test['at_cutter_score'], test['found'], test['truth_boxes'], test['recall']) == (0.25, False, 2, 4, 0.5)
    assert test['boxes_per_frame'] == 1.0 and test['false_per_frame'] == 0.5 and test['ms_per_frame'] is None
    assert [(row['key'], row['found'], row['truth']) for row in test['widths']] == [
        ('lt12', 1, 1), ('12-20', 0, 1), ('20-40', 1, 1), ('ge40', 0, 1)]
    assert test['human'] == {'images': 2, 'truth_boxes': 2, 'found': 1, 'recall': 0.5} and main['validation'] is None

    # A size of scripts/run_detector_sizes.sh: at the cutter's score 0.1, on test and validation.
    size = cutters['detector-sizes/640x360']
    assert (size['label'], size['detail'], size['input_size'], size['anchor_min_ratio'], size['best_epoch']) == (
        'Detector 640 × 360', 'input-size comparison', '640x360', 0.1, 6)
    assert size['augmentation'] is True and size['epochs_run'] == 11
    test = size['test']
    assert (test['score_threshold'], test['at_cutter_score'], test['found'], test['truth_boxes'], test['recall']) == (0.1, True, 2, 4, 0.5)
    assert test['boxes_per_frame'] == 1.0 and test['false_per_frame'] == pytest.approx(1 / 3)
    assert [(row['key'], row['found'], row['truth']) for row in test['widths']] == [
        ('lt12', 1, 1), ('12-20', 0, 1), ('20-40', 1, 1), ('ge40', 0, 1)]
    assert (test['ms_per_frame'], test['device'], test['input_size'], test['checkpoint_epoch']) == (12.5, 'Tesla V100-PCIE-32GB', '640x360', 6)
    assert test['human']['recall'] == 1.0 and size['validation']['partition'] == 'validation'

    [reader] = roles['classifier']['versions']
    assert (reader['id'], reader['label'], reader['finished'], reader['best_epoch'], reader['best_validation_accuracy']) == (
        'classifier:r1/reader', 'Reader', True, 2, 0.8)
    assert reader['test'] == {'samples': 6, 'accuracy': pytest.approx(4 / 6), 'human_samples': 2, 'human_accuracy': 1.0}
    assert reader['classes'] == 3

    deciders = {entry['variant']: entry for entry in roles['decider']['versions']}
    assert set(deciders) == {'crop', 'crop_geom'} and deciders['crop']['id'] == 'decider:expB/experiment-b/crop'
    crop = deciders['crop']
    assert (crop['label'], crop['detail'], crop['inputs'], crop['best_epoch'], crop['smoke_limit']) == (
        'Crop', 'experiment B prototype', ['crop', 'box position'], 7, None)
    assert crop['validation'] == {'boxes': 40, 'reading_accuracy': 0.9, 'ownership_samples': 30, 'macro_f1': 0.6,
                                  'abstain_rate': 0.1, 'other_road_recall': 0.6, 'other_road_precision': 0.5,
                                  'other_road_support': None, 'other_road_found': None}
    assert crop['test']['reading_accuracy'] == 0.9
    json.dumps(models.index(), allow_nan=False)


def test_newest_first_by_the_time_of_the_weights(root, models):
    run = pipeline(root)
    old = make_size(run, '320x320')
    new = make_size(run, '960x544')
    touch(run / 'detector/detector-best.pt', 1_000_000)
    touch(old / 'detector-best.pt', 2_000_000)
    touch(new / 'detector-best.pt', 3_000_000)
    roles = by_role(models.index())
    assert [entry['part'] for entry in roles['cutter']['versions']] == [
        'detector-sizes/960x544', 'detector-sizes/320x320', 'detector']


def test_weights_without_a_result_are_not_finished_and_files_unreadable_are_said(root, models):
    run = pipeline(root)
    make_size(run, '640x360', finished=False, evaluated=False)
    (run / 'evaluation/reader-test.json').write_text('{"task": "reader", "samp')
    roles = by_role(models.index())
    size = next(entry for entry in roles['cutter']['versions'] if entry['part'] == 'detector-sizes/640x360')
    assert (size['finished'], size['test'], size['input_size'], size['input_size_recorded']) == (False, None, '640x360', True)
    [reader] = roles['classifier']['versions']
    assert reader['test'] is None and reader['problems'] == ['evaluation/reader-test.json is unreadable']
    # Folders that are not a size, a variant without weights, a run without weights: not models.
    write(run / 'detector-sizes/notasize/detector-best.pt', 'x')
    write(run / 'detector-sizes/640x360.log', 'log')
    write(root / 'runs/expB/experiment-b/full/result.json', experiment_result('full', 0.8, 0.5))
    make_run(root, 'no-weights')
    roles = by_role(models.index())
    assert {entry['id'] for entry in roles['cutter']['versions']} == {'cutter:r1/detector', 'cutter:r1/detector-sizes/640x360'}
    assert roles['decider']['versions'] == [] and [entry['run'] for entry in roles['classifier']['versions']] == ['r1']


def test_a_weights_file_outside_its_run_is_not_listed(root, models, tmp_path):
    run = pipeline(root)
    outside = tmp_path / 'elsewhere'
    write(outside / 'detector-best.pt', 'weights')
    (run / 'detector-sizes/640x360').mkdir(parents=True)
    (run / 'detector-sizes/640x360/detector-best.pt').symlink_to(outside / 'detector-best.pt')
    roles = by_role(models.index())
    assert [entry['id'] for entry in roles['cutter']['versions']] == ['cutter:r1/detector']


# -- the model in use -------------------------------------------------------------------------------------
def test_choosing_the_model_in_use_writes_only_active_json(root, models):
    run = pipeline(root)
    make_size(run, '640x360')
    experiment(root)
    entry = models.set_active('cutter', 'cutter:r1/detector-sizes/640x360', who='admin', via='sign-in')
    assert (entry['id'], entry['who'], entry['via']) == ('cutter:r1/detector-sizes/640x360', 'admin', 'sign-in') and entry['at']
    path = root / 'models' / 'active.json'
    saved = json.loads(path.read_text())
    assert saved == {'schema_version': 1, 'cutter': entry, 'classifier': None, 'decider': None}
    # Atomic: the temporary file is renamed, nothing is left next to it.
    assert sorted(item.name for item in path.parent.iterdir()) == ['active.json']

    models.set_active('decider', 'decider:expB/experiment-b/crop_geom', who=None, via='token')
    roles = by_role(models.index())
    assert roles['cutter']['active']['id'] == 'cutter:r1/detector-sizes/640x360' and roles['cutter']['active']['found']
    assert [entry['in_use'] for entry in roles['cutter']['versions'] if entry['part'] == 'detector-sizes/640x360'] == [True]
    assert sum(entry['in_use'] for entry in roles['cutter']['versions']) == 1
    assert roles['decider']['active']['via'] == 'token' and roles['classifier']['active'] is None
    # Another cutter replaces the first; the other roles stay.
    models.set_active('cutter', 'cutter:r1/detector', who='admin', via='sign-in')
    saved = json.loads(path.read_text())
    assert saved['cutter']['id'] == 'cutter:r1/detector' and saved['decider']['id'] == 'decider:expB/experiment-b/crop_geom'


@pytest.mark.parametrize('role, model_id, status, reason', [
    ('reader', 'classifier:r1/reader', 400, 'bad_role'),
    (None, 'classifier:r1/reader', 400, 'bad_role'),
    ('cutter', None, 400, 'bad_id'),
    ('cutter', 42, 400, 'bad_id'),
    ('cutter', '../../etc/passwd', 400, 'bad_id'),
    ('cutter', 'cutter:r1/../../etc/detector', 400, 'bad_id'),
    ('cutter', 'cutter:r1/detector/../../x', 400, 'bad_id'),
    ('cutter', 'cutter:/abs/detector', 400, 'bad_id'),
    ('cutter', 'cutter:r1/detector-sizes/../detector', 400, 'bad_id'),
    ('cutter', 'cutter:r1/detector' + 'x' * 400, 400, 'bad_id'),
    ('cutter', 'classifier:r1/reader', 400, 'wrong_role'),
    ('cutter', 'cutter:nope/detector', 404, 'unknown_model'),
    ('cutter', 'cutter:r1/detector-sizes/1280x720', 404, 'unknown_model'),
    ('cutter', 'cutter:r1/detector-sizes/640x360', 409, 'not_finished'),
])
def test_a_choice_is_checked_against_the_index(root, models, role, model_id, status, reason):
    run = pipeline(root)
    make_size(run, '640x360', finished=False)
    with pytest.raises(ModelChoiceError) as error:
        models.set_active(role, model_id, who='admin', via='sign-in')
    assert (error.value.status, error.value.reason) == (status, reason)
    assert not (root / 'models').exists()


def test_an_unreadable_active_json_is_never_overwritten(root, models):
    pipeline(root)
    path = write(root / 'models' / 'active.json', '{"cutter": ')
    index = models.index()
    assert index['active_problem'] == 'models/active.json cannot be read'
    assert all(role['active'] is None for role in index['roles'])
    with pytest.raises(ModelChoiceError) as error:
        models.set_active('cutter', 'cutter:r1/detector', who='admin', via='sign-in')
    assert error.value.status == 409 and path.read_text() == '{"cutter": '


def test_a_choice_whose_files_are_gone_and_entries_that_are_not_choices(root, models):
    pipeline(root)
    write(root / 'models' / 'active.json', {
        'cutter': {'id': 'cutter:gone/detector', 'who': 'admin', 'via': 'sign-in', 'at': '2026-10-24T16:00:00+02:00'},
        'classifier': {'id': '../../reader', 'who': 'x'}, 'decider': 'decider:expB/experiment-b/crop'})
    roles = by_role(models.index())
    assert roles['cutter']['active'] == {'id': 'cutter:gone/detector', 'who': 'admin', 'via': 'sign-in',
                                         'at': '2026-10-24T16:00:00+02:00', 'found': False}
    assert not any(entry['in_use'] for entry in roles['cutter']['versions'])
    assert roles['classifier']['active'] is None and roles['decider']['active'] is None


def test_a_failed_write_leaves_the_old_choice_and_no_temporary_file(root, models, monkeypatch):
    pipeline(root)
    models.set_active('cutter', 'cutter:r1/detector', who='admin', via='sign-in')
    path = root / 'models' / 'active.json'
    before = path.read_text()

    def broken(source, target):
        raise OSError('disk full')
    monkeypatch.setattr(model_roles.os, 'replace', broken)
    with pytest.raises(OSError):
        models.set_active('classifier', 'classifier:r1/reader', who='admin', via='sign-in')
    assert path.read_text() == before and sorted(item.name for item in path.parent.iterdir()) == ['active.json']


def test_models_folder_leading_out_of_the_data_root_is_refused(root, models, tmp_path):
    pipeline(root)
    outside = tmp_path / 'outside-models'
    outside.mkdir()
    (root / 'models').symlink_to(outside)
    assert models.index()['active_problem'] == 'models/ of the data root leads outside it'
    with pytest.raises(ModelChoiceError) as error:
        models.set_active('cutter', 'cutter:r1/detector', who='admin', via='sign-in')
    assert error.value.reason == 'bad_models_folder' and list(outside.iterdir()) == []


def test_a_read_only_panel_refuses_the_choice(root):
    pipeline(root)
    panel = Panel(root, repo=root / 'no-checkout', actions_enabled=False)
    assert [entry['id'] for entry in by_role(panel.models())['classifier']['versions']] == ['classifier:r1/reader']
    with pytest.raises(PanelError) as error:
        panel.set_active_model({'role': 'classifier', 'id': 'classifier:r1/reader'}, who='admin', via='sign-in')
    assert error.value.status == 403 and not (root / 'models').exists()
    panel = Panel(root, repo=root / 'no-checkout', actions_enabled=True)
    for body in ([], 'x', None):
        with pytest.raises(PanelError) as error:
            panel.set_active_model(body, who='admin', via='sign-in')
        assert error.value.status == 400


# -- over HTTP --------------------------------------------------------------------------------------------
def test_models_over_http_with_the_sign_in(tmp_path):
    make_report(tmp_path)
    run = pipeline(tmp_path)
    make_size(run, '640x360')
    server = RunningServer(tmp_path, token=TOKEN, require_token=True)
    server.app.panel.runs_index.running_runs = lambda: {}
    active = tmp_path / 'models' / 'active.json'
    body = {'role': 'cutter', 'id': 'cutter:r1/detector-sizes/640x360'}
    try:
        # Signed out: the page is the sign-in page, the API answers 401 and nothing is written.
        status, response, data = server.request('GET', '/models')
        assert status in (200, 401) and b'id="app"' in data and b'aria-current="page">Models' not in data
        assert server.request('GET', '/api/panel/models')[0] == 401
        assert server.request('POST', '/api/panel/models/active', body=body)[0] == 401
        assert not active.exists()

        cookie = sign_in(server)
        status, response, data = server.request('GET', '/models', headers=cookie)
        assert status == 200 and 'aria-current="page">Models</a>' in data.decode()
        status, response, data = server.request('GET', '/api/panel/models', headers=cookie)
        roles = by_role(json_of(data))
        assert status == 200 and [entry['id'] for entry in roles['classifier']['versions']] == ['classifier:r1/reader']
        assert roles['cutter']['active'] is None
        # Counts only: never the per-frame lists of the reports.
        assert len(data) < 20000

        # Same origin and JSON only, as the actions of the panel.
        status, response, data = server.request('POST', '/api/panel/models/active', body=body,
                                                headers={**cookie, 'Origin': 'http://evil.example'})
        assert status == 403 and json_of(data)['error'] == 'cross_origin'
        status, response, data = server.request('POST', '/api/panel/models/active', body=b'role=cutter',
                                                headers={**cookie, 'Content-Type': 'application/x-www-form-urlencoded'})
        assert status == 415
        status, response, data = server.request('POST', '/api/panel/models/active', body={'role': 'cutter', 'id': '/etc/passwd'},
                                                headers=cookie)
        assert status == 400 and json_of(data)['error'] == 'bad_id'
        status, response, data = server.request('POST', '/api/panel/models/active', body={'role': 'cutter', 'id': 'cutter:r2/detector'},
                                                headers=cookie)
        assert status == 404 and json_of(data)['error'] == 'unknown_model'
        assert server.request('GET', '/api/panel/models/active', headers=cookie)[0] == 405
        assert not active.exists()

        status, response, data = server.request('POST', '/api/panel/models/active', body=body,
                                                headers={**cookie, 'Origin': f'http://127.0.0.1:{server.port}'})
        answer = json_of(data)
        assert status == 200 and answer['ok'] and answer['role'] == 'cutter'
        assert (answer['active']['id'], answer['active']['who'], answer['active']['via']) == (body['id'], 'admin', 'sign-in')
        assert json.loads(active.read_text())['cutter']['id'] == body['id']
        roles = by_role(json_of(server.request('GET', '/api/panel/models', headers=cookie)[2]))
        assert roles['cutter']['active']['found'] and [entry['in_use'] for entry in roles['cutter']['versions']].count(True) == 1

        # An agent with the access token in its header: recorded as such, without a name.
        status, response, data = server.request('POST', '/api/panel/models/active',
                                                body={'role': 'classifier', 'id': 'classifier:r1/reader'},
                                                headers={'X-Review-Token': TOKEN})
        assert status == 200 and (json_of(data)['active']['who'], json_of(data)['active']['via']) == (None, 'token')
    finally:
        server.close()


def test_on_127_0_0_1_the_choice_is_recorded_as_local(tmp_path):
    make_report(tmp_path)
    pipeline(tmp_path)
    server = RunningServer(tmp_path)
    server.app.panel.runs_index.running_runs = lambda: {}
    try:
        status, response, data = server.request('POST', '/api/panel/models/active',
                                                body={'role': 'classifier', 'id': 'classifier:r1/reader'})
        assert status == 200 and (json_of(data)['active']['who'], json_of(data)['active']['via']) == (None, 'local')
    finally:
        server.close()
