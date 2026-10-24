"""The Evaluate tab's data: the training runs of <data root>/runs summarised from their files (nnslr_tools.runs),
and GET /api/panel/runs, /api/panel/runs/<run>.

Every run here is a synthetic folder in a temporary data root, with reports built by the real metric functions of
nnslr_tools.evaluation: nothing is trained or evaluated.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import nnslr_tools.runs as runs_module
from nnslr_tools.evaluation import (detector_metrics, human_reviewed_metrics, label_basis_counts, match_detections,
                                    reader_metrics)
from nnslr_tools.panel import Panel, PanelError
from nnslr_tools.runs import RunsIndex
from review_helpers import TOKEN, RunningServer, json_of, make_report, sign_in

DATASET_SHA = 'd' * 64
SPLIT_SHA = 'e' * 64
TEST_ROUTE = '0000012d--6552a45378'
OTHER_ROUTE = '00000123--8fe1b32468'
HUMAN = 'human_review'
AI = 'ai_review'


def write(path: Path, value) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value if isinstance(value, str) else json.dumps(value, indent=1), encoding='utf-8')
    return path


def detector_report():
    """Four test frames: signs 10, 15, 30 and 50 px wide; the 10 and 30 px ones found, a false box of 8 px on a
    frame without signs and one of 60 px."""
    frames = [
        (TEST_ROUTE, HUMAN, [[100, 100, 110, 110]], [{'box': [100, 100, 110, 110], 'score': 0.9}]),
        (TEST_ROUTE, AI, [[0, 0, 15, 15], [200, 200, 230, 230]],
         [{'box': [200, 200, 230, 230], 'score': 0.8}, {'box': [500, 500, 560, 560], 'score': 0.7}]),
        (OTHER_ROUTE, HUMAN, [[300, 300, 350, 350]], []),
        (OTHER_ROUTE, AI, [], [{'box': [10, 10, 18, 18], 'score': 0.5}]),
    ]
    images = []
    for index, (route, basis, truth, predictions) in enumerate(frames):
        matched = match_detections(predictions, truth, iou_threshold=0.5)
        images.append({'image_path': f'f/{index}.jpg', 'image_sha256': str(index) * 64, 'route_id': route,
                       'segment_index': 0, 'label_basis': basis, 'truth_boxes': truth, 'predictions': predictions,
                       'truth_count': len(truth), 'prediction_count': len(predictions),
                       'true_positives': matched['true_positives'], 'false_positives': matched['false_positives'],
                       'false_negatives': matched['false_negatives'], 'matches': matched['matches'],
                       'false_positive_details': matched['false_positive_details']})
    return {'schema_version': 1, 'task': 'detector', 'partition': 'test', **detector_metrics(images),
            'counts_by_label_basis': label_basis_counts(images),
            'human_reviewed': human_reviewed_metrics(images, detector_metrics),
            'score_threshold': 0.25, 'iou_threshold': 0.5, 'checkpoint': 'x', 'dataset_sha256': DATASET_SHA,
            'split_sha256': SPLIT_SHA, 'device': 'cuda', 'images': images}


def reader_report():
    """Six test crops: 80 read right twice and once as 60, 60 once right and once as 80, an end of limit right."""
    rows = [('maximum_speed:80', 'maximum_speed:80', HUMAN), ('maximum_speed:80', 'maximum_speed:80', AI),
            ('maximum_speed:80', 'maximum_speed:60', AI), ('maximum_speed:60', 'maximum_speed:60', AI),
            ('maximum_speed:60', 'maximum_speed:80', AI), ('cancellation', 'cancellation', HUMAN)]
    samples = [{'annotation_id': str(index), 'image_path': f'f/{index}.jpg', 'truth': truth, 'predicted': predicted,
                'confidence': 0.9, 'correct': truth == predicted, 'label_basis': basis}
               for index, (truth, predicted, basis) in enumerate(rows)]
    classes = ['cancellation', 'maximum_speed:60', 'maximum_speed:80']
    return {'schema_version': 1, 'task': 'reader', 'partition': 'test', 'classes': classes,
            **reader_metrics(samples, classes), 'counts_by_label_basis': label_basis_counts(samples),
            'human_reviewed': human_reviewed_metrics(samples, lambda items: reader_metrics(items, classes)),
            'samples': samples, 'checkpoint': 'x', 'dataset_sha256': DATASET_SHA, 'split_sha256': SPLIT_SHA, 'device': 'cuda'}


TRAIN_LOG = '''== NNSLR baseline pipeline ==
run:          {name}
training mode: {mode}
== 4/10 Train detector ==
{weights}detector epoch 1/2 train_loss 2.0000 val_loss 1.5000 30s
detector epoch 2/2 train_loss 1.0000 val_loss 1.2000 31s
== 5/10 Train reader ==
reader epoch 1/2 train_loss 1.0000 val_loss 0.9000 train_accuracy 0.5000 val_accuracy 0.6000 10s
reader epoch 2/2 train_loss 0.5000 val_loss 0.8000 train_accuracy 0.7000 val_accuracy 0.8000 11s
'''


def make_run(root: Path, name: str, *, upto: str = 'done', mode: str = 'fresh', weights: str = '', log_tail: str = '') -> Path:
    """A run folder as scripts/run_v100_baseline.sh leaves it when it stops at the step `upto` (detector, reader,
    evaluate, export, bundle), or at its end ('done')."""
    run = root / 'runs' / name
    split = write(root / 'splits' / (SPLIT_SHA + '.json'), {
        'policy': 'place-v3', 'seed': 0, 'grouping': ['place'], 'test_routes': [TEST_ROUTE],
        'routes': {'train': [OTHER_ROUTE], 'validation': [OTHER_ROUTE], 'test': [TEST_ROUTE, OTHER_ROUTE], 'buffer': [OTHER_ROUTE]},
        'counts': {'train': 70, 'validation': 14, 'test': 12, 'buffer': 3}, 'human_reviewed_counts': {'test': 5},
        'warnings': ['groups are incomplete'], 'assignments': {str(index): 'train' for index in range(70)}})
    write(root / 'splits' / 'latest.json', {'path': 'splits/' + split.name, 'sha256': SPLIT_SHA})
    common = {'schema_version': 1, 'dataset_kind': 'gold', 'dataset_sha256': DATASET_SHA, 'split_sha256': SPLIT_SHA,
              'split_path': str(split), 'dataset_path': str(root / 'annotations/objects.jsonl')}
    write(run / 'detector/training-plan.json', {**common, 'task': 'detector', 'classes': ['speed_sign'],
                                                 'counts_by_split': {'train': 60, 'validation': 10, 'test': 4},
                                                 'boxes_by_split': {'train': 30, 'validation': 6, 'test': 4},
                                                 'negative_frames_by_split': {'train': 40, 'validation': 5, 'test': 1},
                                                 'frames': [{'image_path': 'f/0.jpg'}] * 50})
    (run / 'evaluation').mkdir(parents=True, exist_ok=True)
    (run / 'export').mkdir(parents=True, exist_ok=True)
    write(root / 'logs' / f'train-{name}.log', TRAIN_LOG.format(name=name, mode=mode, weights=weights) + log_tail)
    steps = ('detector', 'reader', 'evaluate', 'export', 'bundle')
    reached = steps[:steps.index(upto)] if upto in steps else steps
    if 'detector' in reached:
        write(run / 'detector/result.json', '{"task": "detector", "best_validation_loss": 1.2, "initialization": "imagenet1k_v1",'
              ' "config": {"epochs": 2}, "gpu": {"name": "Tesla V100"}, "history": [{"epoch": 1, "train_loss": NaN,'
              ' "validation_loss": 1.5}, {"epoch": 2, "train_loss": 1.0, "validation_loss": 1.2}]}')
        write(run / 'reader/training-plan.json', {**common, 'task': 'reader', 'classes': ['a', 'b', 'c'],
                                                   'counts_by_split': {'train': 20, 'validation': 4, 'test': 6}, 'records': []})
    if 'reader' in reached:
        write(run / 'reader/result.json', {'task': 'reader', 'best_validation_accuracy': 0.8, 'initialization': 'imagenet1k_v1',
                                           'config': {'epochs': 2}, 'history': [
                                               {'epoch': 1, 'train_loss': 1.0, 'validation_loss': 0.9, 'train_accuracy': 0.5, 'validation_accuracy': 0.6},
                                               {'epoch': 2, 'train_loss': 0.5, 'validation_loss': 0.8, 'train_accuracy': 0.7, 'validation_accuracy': 0.8}]})
    if 'evaluate' in reached:
        write(run / 'evaluation/detector-test.json', detector_report())
        write(run / 'evaluation/reader-test.json', reader_report())
    if 'export' in reached:
        write(run / 'export/detector.onnx', 'onnx')
        write(run / 'export/reader.onnx', 'onnx')
    if 'bundle' in reached:
        write(run / 'bundle/sha256sums.json', {})
    return run


@pytest.fixture
def root(tmp_path):
    return tmp_path / 'data'


@pytest.fixture
def index(root):
    value = RunsIndex(root)
    value.running_runs = lambda: {}
    return value


def test_without_runs_the_list_is_empty(index):
    assert index.summaries() == {'runs': [], 'runs_dir_exists': False}


def test_a_finished_run_has_its_headline_numbers(root, index):
    make_run(root, '2026-10-20_0900')
    [run] = index.summaries()['runs']
    assert (run['name'], run['kind'], run['state'], run['stage']) == ('2026-10-20_0900', 'training', 'finished', 'done')
    assert run['started_at'].startswith('2026-10-20T09:00:00') and run['finished_at']
    assert (run['mode'], run['resumed'], run['start']['kind'], run['start']['label']) == ('fresh', False, 'imagenet', 'Pre-trained on ImageNet')
    assert run['reader'] == {'samples': 6, 'accuracy': pytest.approx(4 / 6), 'human_samples': 2, 'human_accuracy': 1.0}
    detector = run['detector']
    assert (detector['found'], detector['missed'], detector['false_boxes'], detector['truth_boxes']) == (2, 2, 2, 4)
    assert detector['recall'] == detector['precision'] == 0.5
    # The human-reviewed frames: the 10 px sign found, the 50 px one missed, no false box.
    assert (detector['human_recall'], detector['human_precision']) == (0.5, 1.0)
    assert run['best'] == {'detector_val_loss': 1.2, 'reader_val_accuracy': 0.8}
    assert run['split_sha256'] == SPLIT_SHA and run['evaluated'] and run['problems'] == []


def test_the_detail_has_labels_sizes_routes_split_and_curves(root, index):
    make_run(root, 'r1')
    detail = index.detail('r1')
    reader = detail['reader_eval']
    # By count, then label; the wrong answer it gives most.
    assert [(row['label'], row['name'], row['count'], row['correct']) for row in reader['labels']] == [
        ('maximum_speed:80', 'Speed limit 80', 3, 2), ('maximum_speed:60', 'Speed limit 60', 2, 1), ('cancellation', 'End of limit', 1, 1)]
    assert reader['labels'][0]['wrong'] == {'label': 'maximum_speed:60', 'name': 'Speed limit 60', 'count': 1}
    assert reader['labels'][2]['wrong'] is None
    assert [(row['label'], row['count']) for row in reader['labels_human']] == [('cancellation', 1), ('maximum_speed:80', 1)]
    sizes = {row['key']: (row['truth'], row['found'], row['false_boxes']) for row in detail['detector_eval']['sizes']}
    assert sizes == {'lt12': (1, 1, 1), '12-20': (1, 0, 0), '20-40': (1, 1, 0), 'ge40': (1, 0, 1)}
    human_sizes = {row['key']: (row['truth'], row['found']) for row in detail['detector_eval']['sizes_human']}
    assert human_sizes == {'lt12': (1, 1), '12-20': (0, 0), '20-40': (0, 0), 'ge40': (1, 0)}
    routes = {row['route']: (row['images'], row['found'], row['missed'], row['false_boxes']) for row in detail['detector_eval']['routes']}
    assert routes == {TEST_ROUTE: (2, 2, 1, 1), OTHER_ROUTE: (2, 0, 1, 1)}
    dataset = detail['dataset']
    assert dataset['split']['policy'] == 'place-v3' and dataset['split']['test_routes'] == [TEST_ROUTE]
    assert dataset['current_split'] and dataset['split_file'] == f'splits/{SPLIT_SHA}.json'
    assert dataset['detector']['boxes'] == {'train': 30, 'validation': 6, 'test': 4} and dataset['reader']['crops']['test'] == 6
    assert 'assignments' not in dataset['split']
    # The curves of the log, with the time of each epoch.
    assert detail['curves']['detector']['points'][1] == {'epoch': 2, 'train_loss': 1.0, 'val_loss': 1.2, 'seconds': 31}
    assert detail['curves']['reader']['epochs'] == 2
    assert detail['training']['detector']['gpu'] == 'Tesla V100' and detail['training']['reader']['best_epoch'] == 2
    assert all(detail['files'].values()) and detail['experiment'] is None


def test_without_a_log_the_curves_come_from_result_json(root, index):
    make_run(root, 'r1')
    (root / 'logs' / 'train-r1.log').unlink()
    detail = index.detail('r1')
    assert detail['curves']['reader']['points'][0] == {'epoch': 1, 'train_loss': 1.0, 'val_loss': 0.9, 'train_accuracy': 0.5,
                                                       'val_accuracy': 0.6}
    # A NaN of the history is left out (the answer is strict JSON).
    assert detail['curves']['detector']['points'][0] == {'epoch': 1, 'val_loss': 1.5}
    assert detail['mode'] is None and detail['start']['kind'] == 'imagenet'
    json.dumps(detail, allow_nan=False)


def test_a_running_run_and_a_stopped_one(root, index):
    make_run(root, '2026-10-24_1000', upto='detector')
    make_run(root, '2026-10-23_1000', upto='evaluate',
             log_tail='Traceback (most recent call last):\nValueError: checkpoint_split_mismatch\nERROR: training failed\n')
    index.running_runs = lambda: {'2026-10-24_1000': {'pipeline'}}
    running, stopped = index.summaries()['runs']
    assert (running['name'], running['state'], running['stage'], running['progress']) == ('2026-10-24_1000', 'running', 'detector', 'reader · epoch 2 of 2')
    assert running['start']['kind'] is None and running['start']['label'] == 'Not recorded yet'
    assert not running['evaluated'] and running['reader'] is None
    assert (stopped['state'], stopped['stage'], stopped['error']) == ('incomplete', 'evaluate', 'ERROR: training failed')
    assert stopped['progress'] is None
    # Not running any more: the same folder is a stopped run.
    index.running_runs = lambda: {}
    assert index.summaries()['runs'][0]['state'] == 'incomplete'


def test_where_the_weights_started_and_a_resume(root, index):
    make_run(root, 'from-run', mode='fresh', weights='continue detector from weights: /srv/data/runs/older_1/detector/detector-best.pt\n',
             log_tail='training mode: resume\n')
    make_run(root, 'from-stable', mode='continue',
             weights='continue detector from weights: cristianku/models@main/training/releases/v1/detector.pt\n')
    starts = {run['name']: run for run in index.summaries()['runs']}
    assert starts['from-run']['start'] == {'kind': 'run', 'label': 'Continued from run older_1', 'run': 'older_1',
                                           'source': '/srv/data/runs/older_1/detector/detector-best.pt'}
    assert starts['from-run']['mode'] == 'fresh' and starts['from-run']['resumed'] is True
    assert starts['from-stable']['start']['kind'] == 'stable' and starts['from-stable']['mode'] == 'continue'


def test_newest_first_and_unreadable_files(root, index):
    make_run(root, '2026-10-20_0900')
    run = make_run(root, '2026-10-22_0900')
    (run / 'evaluation/reader-test.json').write_text('{"task": "reader", "samp')
    first, second = index.summaries()['runs']
    assert (first['name'], second['name']) == ('2026-10-22_0900', '2026-10-20_0900')
    assert first['reader'] is None and first['detector'] is not None
    assert first['problems'] == ['evaluation/reader-test.json is unreadable']


def test_digests_are_kept_until_the_file_changes(root, index, monkeypatch):
    run = make_run(root, 'r1')
    calls = []
    original = runs_module.reader_digest
    monkeypatch.setattr(runs_module, 'reader_digest', lambda report: calls.append(1) or original(report))
    index.summaries()
    index.summaries()
    index.detail('r1')
    assert len(calls) == 1
    path = run / 'evaluation/reader-test.json'
    report = json.loads(path.read_text())
    report['accuracy'] = 0.25
    path.write_text(json.dumps(report))
    os.utime(path, ns=(path.stat().st_atime_ns, path.stat().st_mtime_ns + 10**9))
    assert index.summaries()['runs'][0]['reader']['accuracy'] == 0.25 and len(calls) == 2


def test_strict_names_and_no_way_out_of_the_data_root(root, index, tmp_path):
    make_run(root, 'r1')
    for name in ('..', '../r1', '.hidden', 'r1/..', '', 'a' * 101, 'r 1', 'r1\x00'):
        assert index.detail(name) is None, name
    # A run folder that is a symlink to somewhere else is not a run.
    outside = make_run(tmp_path / 'elsewhere', 'secret')
    (root / 'runs' / 'linked').symlink_to(outside, target_is_directory=True)
    assert [run['name'] for run in index.summaries()['runs']] == ['r1']
    assert index.detail('linked') is None
    # A report inside the run that is a symlink out of it is not read.
    report = root / 'runs' / 'r1' / 'evaluation' / 'reader-test.json'
    report.unlink()
    report.symlink_to(outside / 'evaluation' / 'reader-test.json')
    assert index.detail('r1')['reader_eval'] is None


def experiment_result(variant: str, reading: float, other_recall: float, *, human: bool = True) -> dict:
    """A result.json of nnslr_tools.context_experiment (metrics.<partition>.reading / ownership / human_reviewed)."""
    def report(scale: float) -> dict:
        return {'box_count': 40, 'reading': {'accuracy': reading * scale, 'expected_calibration_error': 0.05, 'sample_count': 40},
                'ownership': {'sample_count': 30, 'accuracy': 0.8, 'macro_f1': 0.6 * scale, 'abstain_rate': 0.1,
                              'decided_accuracy': 0.85, 'expected_calibration_error': 0.07,
                              'other_road': {'precision': 0.5, 'recall': other_recall * scale, 'f1': 0.4},
                              'confusion': {'other_road': {'other_road': 2, 'own_road': 1, 'unresolved': 0},
                                            'own_road': {'other_road': 1, 'own_road': 20, 'unresolved': 2},
                                            'unresolved': {'other_road': 0, 'own_road': 1, 'unresolved': 3}}}}
    test = {'loss': 0.7, **report(1.0), 'counts_by_label_basis': {'human_review': 10, 'ai_review': 30},
            'human_reviewed': {'label_basis': 'human_review', **report(0.9)} if human else None}
    return {'schema_version': 1, 'task': 'experiment_b', 'variant': variant, 'inputs': ['crop', 'box_geometry'],
            'best_epoch': 7, 'config': {'epochs': 20, 'seed': 0}, 'best_validation_selection_score': 0.7,
            'selection_metric': 'mean of validation reading accuracy and validation ownership macro-F1',
            'metrics': {'validation': {'loss': 0.6, **report(1.0), 'human_reviewed': None}, 'test': test},
            'split_sha256': SPLIT_SHA, 'initialization': 'random', 'history': [{'epoch': 1}], 'limit': None}


def test_experiment_b_variants_side_by_side(root, index):
    run = make_run(root, 'r1')
    write(run / 'experiment-b/crop/result.json', experiment_result('crop', 0.9, 0.2))
    write(run / 'experiment-b/full/result.json', experiment_result('full', 0.88, 0.6, human=False))
    assert index.summaries()['runs'][0]['experiment_b'] == ['crop', 'full']
    experiment = index.detail('r1')['experiment']
    assert [(item['key'], item['found'], item['best_epoch']) for item in experiment['variants']] == [
        ('crop', True, 7), ('crop_geom', False, None), ('full', True, 7)]
    assert experiment['variants'][0]['inputs'] == ['crop', 'box position'] and experiment['partitions'] == ['test', 'validation']
    rows = {row['key']: row for row in experiment['metrics']}
    assert rows['reading_accuracy']['values']['crop']['test'] == {'all': 0.9, 'human': pytest.approx(0.81)}
    assert rows['reading_accuracy']['values']['full']['test'] == {'all': 0.88, 'human': None}
    assert rows['reading_accuracy']['values']['crop_geom']['test'] == {'all': None, 'human': None}
    assert rows['other_road_recall']['better'] == 'higher' and rows['reading_calibration']['better'] == 'lower'
    assert rows['loss']['values']['crop']['validation']['all'] == 0.6
    assert list(rows)[0] == 'reading_accuracy' and 'seed' not in rows
    [confusion] = experiment['confusions']
    matrix = confusion['variants']['crop']['test']['all']
    assert matrix['labels'] == ['own_road', 'other_road', 'unresolved'] and matrix['names'] == ['Ours', 'Another', 'Unclear']
    assert matrix['matrix'][0] == [20, 1, 2] and confusion['variants']['full']['test']['human'] is None


def test_experiment_b_as_context_experiment_writes_it(root, index):
    """The reader of result.json against the module that writes it (skipped while that module is not importable)."""
    try:
        from nnslr_tools.context_experiment import partition_report
    except ImportError:
        pytest.skip('nnslr_tools.context_experiment not available')
    samples = [{'truth': 'a', 'predicted': 'a', 'correct': True, 'confidence': 0.9, 'label_basis': HUMAN,
                'ownership_truth': 'own_road', 'ownership_predicted': 'own_road', 'ownership_confidence': 0.8},
               {'truth': 'b', 'predicted': 'a', 'correct': False, 'confidence': 0.6, 'label_basis': AI,
                'ownership_truth': 'other_road', 'ownership_predicted': 'unresolved', 'ownership_confidence': 0.5}]
    run = make_run(root, 'r1')
    result = {'variant': 'crop', 'metrics': {'validation': None, 'test': {'loss': 0.5, **partition_report(samples, ['a', 'b'])}}}
    write(run / 'experiment-b/crop/result.json', json.dumps(result))
    rows = {row['key']: row for row in index.detail('r1')['experiment']['metrics']}
    assert rows['reading_accuracy']['values']['crop']['test'] == {'all': 0.5, 'human': 1.0}
    assert rows['abstain_rate']['values']['crop']['test']['all'] == 0.5
    assert rows['box_count']['values']['crop']['test'] == {'all': 2, 'human': 1}


def test_a_folder_with_experiment_b_only(root, index):
    base = root / 'runs' / 'experiment-b-001' / 'experiment-b'
    write(base / 'crop/result.json', experiment_result('crop', 0.9, 0.2))
    write(base / 'crop_geom.log', 'experiment_b_crop_geom epoch 3/20 train_loss 1.0 val_loss 1.0 4s\n')
    write(base / 'dry-run.json', {'record_count': 3})
    index.running_runs = lambda: {'experiment-b-001': {'experiment'}}
    [run] = index.summaries()['runs']
    assert (run['kind'], run['state'], run['stage']) == ('experiment', 'running', 'experiment')
    assert run['progress'] == 'experiment B · crop_geom · epoch 3 of 20' and run['split_sha256'] == SPLIT_SHA
    assert run['start']['kind'] == 'random' and not run['evaluated']
    # The logs and the dry run next to the variant folders are not variants.
    assert run['experiment_b'] == ['crop'] and run['updated_at']
    assert [item['key'] for item in index.detail('experiment-b-001')['experiment']['variants']] == ['crop', 'crop_geom', 'full']
    for variant in ('crop_geom', 'full'):
        write(base / variant / 'result.json', experiment_result(variant, 0.9, 0.3))
    index.running_runs = lambda: {}
    [run] = index.summaries()['runs']
    assert (run['state'], run['stage'], run['experiment_b']) == ('finished', 'done', ['crop', 'crop_geom', 'full'])


def test_the_panel_answers_404_for_an_unknown_run(root):
    panel = Panel(root, repo=root / 'no-checkout', actions_enabled=False)
    panel.runs_index.running_runs = lambda: {}
    make_run(root, 'r1')
    assert [run['name'] for run in panel.runs()['runs']] == ['r1']
    assert panel.run('r1')['name'] == 'r1'
    with pytest.raises(PanelError) as error:
        panel.run('nope')
    assert error.value.status == 404 and error.value.reason == 'unknown_run'


def test_runs_over_http_with_the_sign_in(tmp_path):
    make_report(tmp_path)
    make_run(tmp_path, 'r1')
    server = RunningServer(tmp_path, token=TOKEN, require_token=True)
    server.app.panel.runs_index.running_runs = lambda: {}
    try:
        # Signed out: the page is the sign-in page and the API answers 401.
        status, response, data = server.request('GET', '/evaluate')
        assert status in (200, 401) and b'id="app"' in data
        assert server.request('GET', '/api/panel/runs')[0] == 401
        assert server.request('GET', '/api/panel/runs/r1')[0] == 401
        cookie = sign_in(server)
        status, response, data = server.request('GET', '/api/panel/runs', headers=cookie)
        assert status == 200 and [run['name'] for run in json_of(data)['runs']] == ['r1']
        status, response, data = server.request('GET', '/api/panel/runs/r1', headers=cookie)
        detail = json_of(data)
        assert status == 200 and detail['detector_eval']['sizes'][0]['key'] == 'lt12'
        # Counts only: never the per-image or per-crop lists of the reports.
        assert detail['detector_eval']['images'] == 4 and detail['reader_eval']['samples'] == 6 and len(data) < 20000
        status, response, data = server.request('GET', '/api/panel/runs/nope', headers=cookie)
        assert status == 404 and json_of(data)['error'] == 'unknown_run'
        for path in ('/api/panel/runs/..%2F..%2Fsplits', '/api/panel/runs/../splits', '/api/panel/runs/r1/x', '/api/panel/runs/%2e%2e'):
            assert server.request('GET', path, headers=cookie)[0] == 404, path
        status, response, data = server.request('GET', '/evaluate?run=r1', headers=cookie)
        assert status == 200 and 'aria-current="page">Evaluate</a>' in data.decode()
    finally:
        server.close()
