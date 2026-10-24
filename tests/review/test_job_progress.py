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
    assert result['detail'] == 'detector · epoca 5 di 20'
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
    assert prepared['percent'] == 40.0 and prepared['detail'] == '1 di 3 route pronte'
    fetched = progress('fetch', 'running', 'r1 sul comma: 0 1. Salto:. Scarico: 0 1\n== r1--0 -> /x\n')
    assert fetched['percent'] is None and fetched['detail'] == '1 segmento scaricato · r1'
