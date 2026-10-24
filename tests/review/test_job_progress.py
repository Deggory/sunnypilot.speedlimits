"""The progress of a panel action read from its log: steps, percent, detail, last line."""
from __future__ import annotations

from nnslr_tools.job_progress import progress

TRAIN_LOG = """== helpers/3_train.sh
== a. importo le revisioni confermate
Frame nuovi nel dataset: 0
== b. controllo il dataset e fisso lo split
== c. ambiente GPU
Ok: .venv-gpu
== d. training r1 (fresh) -> data/runs/r1
== 1/10 Validate exact training populations (CPU-only) ==
== 3/10 Explicit CUDA smoke test ==
== 4/10 Train detector ==
detector epoch 5/20 train_loss 0.5000 val_loss 0.6000 40s
"""


def states(result):
    return {step['key']: step['state'] for step in result['steps']}


def test_training_in_the_detector_counts_its_epochs():
    result = progress('train', 'running', TRAIN_LOG)
    assert states(result) == {'import': 'done', 'split': 'done', 'gpu_env': 'done', 'checks': 'done',
                              'detector': 'current', 'reader': 'todo', 'evaluate': 'todo', 'export': 'todo',
                              'bundle': 'todo'}
    # 2 + 2 + 6 + 3 done, and 5 of 20 epochs of the detector's 40.
    assert result['percent'] == 23.0
    assert result['detail'] == 'detector · epoch 5 of 20'
    assert result['last_line'].startswith('detector epoch 5/20')
    assert result['curves'] == {'detector': {'epochs': 20, 'points': [
        {'epoch': 5, 'train_loss': 0.5, 'val_loss': 0.6, 'seconds': 40}]}}


def test_the_curves_of_both_networks_and_a_resumed_epoch():
    log = ('detector epoch 1/2 train_loss 0.9000 val_loss 1.0000 30s\n'
           'detector epoch 1/2 train_loss 0.8000 val_loss 0.9000 31s\n'
           'detector epoch 2/2 train_loss 0.7000 val_loss 0.8000 30s\n'
           'reader epoch 1/3 train_loss 1.2000 val_loss 1.1000 train_accuracy 0.6000 val_accuracy 0.5500 9s\n')
    curves = progress('train', 'running', log)['curves']
    assert [p['train_loss'] for p in curves['detector']['points']] == [0.8, 0.7]
    assert curves['reader'] == {'epochs': 3, 'points': [{'epoch': 1, 'train_loss': 1.2, 'val_loss': 1.1,
                                                          'train_accuracy': 0.6, 'val_accuracy': 0.55, 'seconds': 9}]}


def test_a_failed_step_is_marked_and_a_finished_job_is_complete():
    log = '== a. importo le revisioni confermate\n== b. controllo\n== c. ambiente GPU\nERROR: venv\n'
    failed = progress('train', 'failed', log + '\n== finito con codice 1\n')
    assert states(failed)['gpu_env'] == 'failed' and failed['last_line'] == 'ERROR: venv'
    done = progress('dataset', 'ok', '== a. importo le revisioni confermate\n== b. controllo\n')
    assert done['percent'] == 100.0 and set(states(done).values()) == {'done'}


def test_preparation_uses_its_own_count_and_fetch_counts_segments():
    prepared = progress('prepare', 'running', '== a. collego i video\n== b+c. 3 route da preparare\n',
                        {'percent': 40, 'routes_done': 1, 'routes_total': 3, 'working': []})
    assert prepared['percent'] == 40.0 and prepared['detail'] == '1 of 3 routes ready'
    fetched = progress('fetch', 'running', 'r1 sul comma: 0 1. Salto:. Scarico: 0 1\n== r1--0 -> /x\n')
    assert fetched['percent'] is None and fetched['detail'] == '1 segment downloaded · r1'


def test_the_progress_inside_the_current_step_comes_from_the_last_report():
    from nnslr_tools.job_progress import with_step_progress
    running = progress('train', 'running', TRAIN_LOG)
    inside = with_step_progress(running, 'train', {'task': 'detector', 'done': 50, 'total': 200, 'text': 'epoch 3 of 20'})
    assert inside['step_progress'] == {'done': 50, 'total': 200, 'percent': 25.0, 'text': 'epoch 3 of 20'}
    # 13 of 100 done before the detector, and a quarter of its 40.
    assert inside['percent'] == 23.0
    # A report of another step is not this one's: the epochs of the log stay.
    other = with_step_progress(running, 'train', {'task': 'validate', 'done': 1, 'total': 2})
    assert other['step_progress'] == running['step_progress'] and other['percent'] == running['percent']
    setup = progress('train', 'running', '== a. importo\n== b. controllo\n== c. ambiente GPU\n')
    assert with_step_progress(setup, 'train', None, 'Downloading torch (900 MB)')['step_progress']['text'] == \
        'Downloading torch (900 MB)'


def test_the_current_step_has_its_own_percent_from_the_log():
    # The detector: 5 of its 20 epochs.
    assert progress('train', 'running', TRAIN_LOG)['step_progress'] == {
        'done': 5, 'total': 20, 'percent': 25.0, 'text': 'epoch 5 of 20'}
    # The checks are the sub-steps 1-3 of the pipeline: at the smoke test, two of three are done.
    checks = progress('train', 'running', TRAIN_LOG.split('== 4/10')[0])
    assert states(checks)['checks'] == 'current'
    assert checks['step_progress'] == {'done': 2, 'total': 3, 'percent': 66.7,
                                       'text': 'Explicit CUDA smoke test · 3 of 3'}
    # 10 of 100 before the checks, and two thirds of their 3.
    assert checks['percent'] == 12.0
    # Nothing to count yet, and nothing for a job that is not running.
    assert progress('train', 'running', '== a. importo le revisioni confermate\n')['step_progress'] is None
    assert progress('train', 'failed', TRAIN_LOG)['step_progress'] is None


def test_the_last_line_is_never_a_piece_of_json():
    log = TRAIN_LOG.split('== 4/10')[0] + '{\n  "cuda": true,\n  "devices": [\n    "sm_90"\n  ],\n  "bytes": 3407,\n  12\n}\n'
    assert progress('train', 'running', log)['last_line'] == '== 3/10 Explicit CUDA smoke test =='
