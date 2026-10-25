"""The progress of a panel action read from its log: steps, percent, detail, last line."""
from __future__ import annotations

from nnslr_tools.job_progress import last_line, progress

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
    assert prepared['percent'] == 40.0 and prepared['detail'] == '1 of 3 drives ready'
    # The helper speaks English now; the steps and the end of its log read the same.
    english = progress('prepare', 'failed', '== a. linking the videos of /v\n== b+c. 3 drives to prepare, 2 at a time\n'
                       'r1: proposing failed (see r1.log)\n\n== finished with code 1\n')
    assert states(english)['link'] == 'done' and english['last_line'] == 'r1: proposing failed (see r1.log)'
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


# -- scripts/run_classifier.sh: the classifier with "not a sign" -----------------------------------------------------
CLASSIFIER_LOG = """== scripts/run_classifier.sh 2026-10-24_1730-classifier
== Classifier (cutter -> classifier -> decider) ==
run:          2026-10-24_1730-classifier
== 1/7 Check the dataset, the split and the cutter (CPU-only) ==
== 2/7 Explicit CUDA smoke test ==
== 3/7 Mine the cutter's false boxes ==
== 4/7 Train the classifier ==
classifier epoch 3/20 train_loss 0.5123 val_loss 0.6012 train_accuracy 0.8123 val_accuracy 0.8501 val_sign_accuracy 0.7012 val_rejection 0.9534 41s
"""


def test_the_classifier_training_has_its_own_steps_and_curves():
    from nnslr_tools.job_progress import CLASSIFIER_STEPS
    assert sum(weight for _, _, weight in CLASSIFIER_STEPS) == 100
    assert max(CLASSIFIER_STEPS, key=lambda step: step[2])[0] == 'classifier'
    result = progress('train_classifier', 'running', CLASSIFIER_LOG)
    assert states(result) == {'checks': 'done', 'mine': 'done', 'classifier': 'current', 'evaluate': 'todo', 'chain': 'todo'}
    assert [step['label'] for step in result['steps']] == ['Checks and GPU smoke test', 'Mine the cutter’s false boxes',
                                                            'Train the classifier', 'Evaluate the classifier alone',
                                                            'Evaluate the chain end to end']
    # 4 + 14 done, and 3 of 20 epochs of the classifier's 60.
    assert result['percent'] == 27.0
    assert result['detail'] == 'classifier · epoch 3 of 20'
    assert result['step_progress'] == {'done': 3, 'total': 20, 'percent': 15.0, 'text': 'epoch 3 of 20'}
    [point] = result['curves']['classifier']['points']
    assert point == {'epoch': 3, 'train_loss': 0.5123, 'val_loss': 0.6012, 'train_accuracy': 0.8123, 'val_accuracy': 0.8501,
                     'val_sign_accuracy': 0.7012, 'val_rejection': 0.9534,
                     'val_balanced_accuracy': (0.7012 + 0.9534) / 2, 'seconds': 41}
    assert result['curves']['classifier']['epochs'] == 20 and 'stopped_early' not in result['curves']['classifier']


def test_the_steps_of_the_classifier_training_from_its_phase_lines():
    head = CLASSIFIER_LOG.split('== 2/7')[0]
    checks = progress('train_classifier', 'running', head)
    # The checks are the phases 1 and 2: at the first, none of the two is done.
    assert states(checks)['checks'] == 'current' and checks['percent'] == 0.0
    smoke = progress('train_classifier', 'running', CLASSIFIER_LOG.split('== 3/7')[0])
    assert smoke['step_progress'] == {'done': 1, 'total': 2, 'percent': 50.0, 'text': 'Explicit CUDA smoke test · 2 of 2'}
    assert smoke['percent'] == 2.0
    later = CLASSIFIER_LOG + '== 5/7 Evaluate the classifier alone ==\n'
    assert states(progress('train_classifier', 'running', later))['evaluate'] == 'current'
    chain = later + '== 6/7 Evaluate the chain ==\n== 7/7 Compare with the old reader ==\n'
    result = progress('train_classifier', 'running', chain)
    assert states(result)['chain'] == 'current' and result['step_progress']['text'] == 'Compare with the old reader · 2 of 2'
    assert result['percent'] == 92.5
    failed = progress('train_classifier', 'failed', chain + 'error: no old reader\n\n== finito con codice 1\n')
    assert states(failed)['chain'] == 'failed' and failed['last_line'] == 'error: no old reader'
    assert progress('train_classifier', 'ok', chain)['percent'] == 100.0


def test_the_phase_lines_of_one_script_never_move_the_steps_of_the_other():
    # "== 4/10 Train detector" of run_v100_baseline.sh in a classifier log, "== 5/7" in a training log.
    assert states(progress('train_classifier', 'running', '== 4/10 Train detector ==\n'))['checks'] == 'current'
    train = progress('train', 'running', '== a. importo\n== b. controllo\n== c. ambiente GPU\n== 5/7 Evaluate the classifier alone ==\n')
    assert states(train)['gpu_env'] == 'current'


def test_a_classifier_that_stops_early():
    log = CLASSIFIER_LOG + 'classifier stops early after epoch 3: validation balanced accuracy not better for 2 epochs (best 0.8300)\n'
    result = progress('train_classifier', 'running', log)
    assert result['curves']['classifier']['stopped_early'] == 3
    assert result['step_progress'] == {'done': 1, 'total': 1, 'percent': 100.0, 'text': 'stopped early after epoch 3'}
    assert result['detail'] == 'classifier · stopped early after epoch 3'
    # 4 + 14 done and the whole of the classifier's 60.
    assert result['percent'] == 78.0
    # A training of the same network started again in the same log is not stopped any more.
    again = progress('train_classifier', 'running', log + 'classifier epoch 1/20 train_loss 1.0 val_loss 1.0 3s\n')
    assert 'stopped_early' not in again['curves']['classifier'] and again['detail'] == 'classifier · epoch 1 of 20'


def test_the_reports_of_the_mining_the_classifier_and_the_chain():
    from nnslr_tools.job_progress import with_step_progress
    mining = progress('train_classifier', 'running', CLASSIFIER_LOG.split('== 4/7')[0])
    inside = with_step_progress(mining, 'train_classifier', {'task': 'mine', 'done': 50, 'total': 200, 'text': '50 of 200 frames'})
    assert inside['step_progress'] == {'done': 50, 'total': 200, 'percent': 25.0, 'text': '50 of 200 frames'}
    # 4 done before the mining, and a quarter of its 14.
    assert inside['percent'] == 7.5
    training = with_step_progress(progress('train_classifier', 'running', CLASSIFIER_LOG), 'train_classifier',
                                  {'task': 'classifier', 'done': 30, 'total': 100, 'text': 'epoch 3 of 20'})
    assert training['percent'] == 36.0
    chain = progress('train_classifier', 'running', CLASSIFIER_LOG + '== 5/7 x ==\n== 6/7 Evaluate the chain ==\n')
    assert with_step_progress(chain, 'train_classifier', {'task': 'chain', 'done': 1, 'total': 4})['percent'] == 88.8
    # The report of the mining is not the training's.
    assert with_step_progress(chain, 'train_classifier', {'task': 'mine', 'done': 1, 'total': 4})['percent'] == chain['percent']


# -- scripts/run_detector_sizes.sh with one size: "Train the cutter" -------------------------------------------------
CUTTER_LOG = """== scripts/run_detector_sizes.sh 2026-10-24_1800-cutter 960x544
== Detector input sizes ==
run:          2026-10-24_1800-cutter
== 1/4 Validate the training and evaluation populations (CPU-only) ==
== 2/4 Explicit CUDA smoke test ==
== 3/4 Train the detector at 960x544 ==
detector epoch 3/20 train_loss 1.2000 val_loss 1.5000 val_recall 0.4125 87s
"""


def test_the_cutter_training_has_its_steps_from_the_titles_of_its_phases():
    from nnslr_tools.job_progress import CUTTER_STEPS
    assert sum(weight for _, _, weight in CUTTER_STEPS) == 100
    result = progress('train_cutter', 'running', CUTTER_LOG)
    assert states(result) == {'checks': 'done', 'detector': 'current', 'evaluate': 'todo'}
    assert [step['label'] for step in result['steps']] == ['Checks and GPU smoke test', 'Train the cutter',
                                                            'Evaluate on test and validation']
    # 5 done, and 3 of 20 epochs of the cutter's 80.
    assert result['percent'] == 17.0 and result['detail'] == 'detector · epoch 3 of 20'
    assert result['curves']['detector']['points'] == [{'epoch': 3, 'train_loss': 1.2, 'val_loss': 1.5, 'val_recall': 0.4125, 'seconds': 87}]
    smoke = progress('train_cutter', 'running', CUTTER_LOG.split('== 3/4')[0])
    assert smoke['step_progress'] == {'done': 1, 'total': 2, 'percent': 50.0, 'text': 'Explicit CUDA smoke test · 2 of 2'}
    evaluating = progress('train_cutter', 'running', CUTTER_LOG + '== 4/4 Evaluate the detector at 960x544 on test validation ==\n')
    assert states(evaluating)['evaluate'] == 'current' and evaluating['percent'] == 85.0
    # The totals change with the sizes: two sizes are out of 6, the titles say the same.
    two = progress('train_cutter', 'running', '== 1/6 Validate the training and evaluation populations (CPU-only) ==\n'
                                              '== 2/6 Explicit CUDA smoke test ==\n== 3/6 Train the detector at 640x360 ==\n')
    assert states(two)['detector'] == 'current'
    done = progress('train_cutter', 'ok', CUTTER_LOG + '== Comparison (the cutter\'s recall first) ==\nDetector sizes complete: x\n')
    assert done['percent'] == 100.0 and done['last_line'] == 'Detector sizes complete: x'


def test_the_cutter_that_stops_early_and_its_reports():
    from nnslr_tools.job_progress import with_step_progress
    log = CUTTER_LOG + 'detector stops early after epoch 3: validation recall not better for 5 epochs\n'
    result = progress('train_cutter', 'running', log)
    assert result['curves']['detector']['stopped_early'] == 3 and result['percent'] == 85.0
    report = with_step_progress(progress('train_cutter', 'running', CUTTER_LOG), 'train_cutter',
                                {'task': 'detector', 'done': 50, 'total': 100, 'text': 'epoch 3 of 20'})
    assert report['percent'] == 45.0 and report['step_progress']['text'] == 'epoch 3 of 20'
    # The numbered phases of the pipeline are not the cutter's, nor the cutter's titles the pipeline's.
    assert states(progress('train_cutter', 'running', '== 4/10 Train detector ==\n'))['checks'] == 'current'
    assert states(progress('train', 'running', '== a. importo\n== 3/4 Train the detector at 960x544 ==\n'))['import'] == 'current'


def test_the_last_line_is_the_error_not_the_hint_after_it():
    log = ('== 3/3 Try it on the GPU\ntorch.AcceleratorError: CUDA error: out of memory\n'
           "Search for `cudaErrorMemoryAllocation' in https://docs.nvidia.com/cuda/ for more information.\n\n== finito con codice 1\n")
    assert last_line(log) == 'torch.AcceleratorError: CUDA error: out of memory'


def test_a_proposer_measured_counts_its_frames():
    log = ('== helpers/measure_proposer.sh\nMeasuring the proposer qwen on the test frames...\n'
           'Preannotating 279/604: segment 17, frame 24, 0 proposals\n[transformers] Kwargs passed\n'
           'Preannotating 280/604: segment 17, frame 25, 1 proposals\n')
    running = progress('measure_proposer', 'running', log)
    assert running['percent'] == 46.4 and running['detail'] == '280 of 604 frames'
    assert progress('measure_proposer', 'running', log.splitlines()[0])['percent'] is None
    assert progress('measure_proposer', 'ok', log + 'Done\n== finito con codice 0\n')['percent'] == 100.0


def test_propose_again_has_one_time_left_and_no_raw_line():
    preparation = {'percent': 12.5, 'routes_done': 1, 'routes_total': 4, 'again': True, 'seconds_left': 7380,
                   'proposer': 'Qwen3-VL', 'working': [{'route': 'r2', 'title': 'Basel → Mumpf', 'phase': 'proposing again',
                                                       'done': 612, 'total': 1560, 'seconds_left': 1440}]}
    log = '== helpers/1_prepare.sh --again\n00000128--edfa65432f: proposing the signs again (qwen)...\n'
    result = progress('prepare_again', 'running', log, preparation)
    assert result['percent'] == 12.5 and result['seconds_left'] == 7380
    assert result['detail'] == 'Qwen3-VL · 1 of 4 drives ready' and result['last_line'] == ''
    # Before the pace is known the time left is unknown, not guessed from the percent.
    assert progress('prepare_again', 'running', log, {**preparation, 'seconds_left': None})['seconds_left'] is None
    # A new preparation has no time of the server; a finished one keeps its last line.
    assert 'seconds_left' not in progress('prepare', 'running', log, {**preparation, 'again': False})
    assert progress('prepare_again', 'ok', log)['last_line'].endswith('(qwen)...')


def test_a_known_failure_is_one_plain_sentence():
    from nnslr_tools.job_progress import failure_text
    oom = ('Measuring the proposer qwen on the test frames...\npreannotation worker error: CUDA out of memory. Tried to '
           'allocate 16.33 GiB.\nerror: preannotation failed; diagnostic request: /root/x/data/tmp/proposer-1/request.json\n'
           '== finito con codice 2\n')
    failed = progress('measure_proposer', 'failed', oom)
    assert failed['failure'].startswith('The GPU ran out of memory') and 'request.json' in failed['last_line']
    assert failure_text('ERROR: per dividere il dataset servono almeno 5 route (o luoghi) indipendenti\n').startswith('More reviewed')
    assert failure_text('ssh: connect to host comma port 22: Connection refused\n').startswith('The comma did not answer')
    assert failure_text('Traceback\nValueError: something new\n') == ''
    assert 'failure' not in progress('measure_proposer', 'running', oom)
