"""The Train tab's data: what the dataset holds, where a training can start, and the start option."""
from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import datetime

import pytest

from nnslr_tools.panel import Panel, PanelError
from test_panel import checkout, finished, full_log, only_this_checkout, root  # noqa: F401  (fixtures and helpers)

ENV_TRAIN = '#!/usr/bin/env bash\necho "train $* init=${NNSLR_TRAIN_INIT:-} from=${NNSLR_TRAIN_FROM_RUN:-}"\n'


def write_dataset(root):
    (root / 'annotations').mkdir(parents=True, exist_ok=True)
    rows = [
        {'route_id': 'r1', 'image_path': 'f/1.png', 'sign_family': 'maximum_speed', 'provenance': {'label_basis': 'human_review'}},
        {'route_id': 'r1', 'image_path': 'f/1.png', 'sign_family': 'not_a_sign', 'provenance': {'label_basis': 'ai_review'}},
        {'route_id': 'r2', 'image_path': 'f/2.png', 'sign_family': 'other_sign', 'provenance': {'label_basis': 'ai_review'}},
    ]
    (root / 'annotations' / 'objects.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    (root / 'splits').mkdir(exist_ok=True)
    (root / 'splits' / 'abc.json').write_text(json.dumps({'counts': {'train': 2, 'test': 1}, 'warnings': ['w']}))
    (root / 'splits' / 'latest.json').write_text(json.dumps({'path': 'splits/abc.json'}))
    (root / 'logs').mkdir(exist_ok=True)
    (root / 'logs' / 'import.json').write_text(json.dumps({
        'imported_frame_count': 1, 'already_imported_frame_count': 4, 'changed_after_import': ['a', 'b']}))


def test_the_dataset_summary_counts_frames_objects_families_and_reviewers(root, checkout):
    write_dataset(root)
    summary = Panel(root, repo=checkout).dataset_summary()
    assert (summary['frames'], summary['objects'], summary['routes'], summary['human'], summary['ai']) == (2, 3, 2, 1, 2)
    assert [(item['key'], item['count']) for item in summary['families']][0] == ('maximum_speed', 1)
    assert {item['label'] for item in summary['families']} == {'Speed limit', 'Not a sign', 'Other sign'}
    assert summary['per_route'][0] == {'route': 'r1', 'frames': 1, 'objects': 2, 'speed_signs': 1, 'human': 1, 'ai': 1}
    assert summary['split']['counts'] == {'train': 2, 'test': 1} and summary['split']['warnings'] == ['w']
    # An import made before changed frames were replaced: they were left unapplied.
    assert summary['last_import'] == {'at': summary['last_import']['at'], 'new_frames': 1, 'already': 4,
                                      'changed_after_import': 2, 'replaced_frames': 0, 'removed_frames': 0,
                                      'warnings': []}
    (root / 'logs' / 'import.json').write_text(json.dumps({
        'imported_frame_count': 1, 'already_imported_frame_count': 4, 'replaced_frame_count': 2,
        'changed_after_import': ['a', 'b'], 'removed_withdrawn_frame_count': 3}))
    imported = Panel(root, repo=checkout).dataset_summary()['last_import']
    assert (imported['changed_after_import'], imported['replaced_frames'], imported['removed_frames']) == (0, 2, 3)


def test_the_split_says_which_dataset_it_was_made_of_and_the_rejected_proposals_it_leaves_out(root, checkout):
    write_dataset(root)
    sha = hashlib.sha256((root / 'annotations' / 'objects.jsonl').read_bytes()).hexdigest()
    summary = Panel(root, repo=checkout).dataset_summary()
    # A split without them (an older one): nothing to compare, nothing left out.
    assert summary['sha256'] == sha and (summary['split']['dataset_sha256'], summary['split']['rejected']) == (None, 0)
    (root / 'splits' / 'abc.json').write_text(json.dumps({'counts': {'train': 2}, 'warnings': [], 'dataset_sha256': sha,
                                                          'excluded_rejected_ids': ['x']}))
    split = Panel(root, repo=checkout).dataset_summary()['split']
    assert (split['dataset_sha256'], split['rejected']) == (sha, 1)
    # The dataset changes after the split: its sha256 is no longer the one the split recorded.
    with (root / 'annotations' / 'objects.jsonl').open('a') as target:
        target.write(json.dumps({'route_id': 'r3', 'image_path': 'f/3.png', 'sign_family': 'zone'}) + '\n')
    summary = Panel(root, repo=checkout).dataset_summary()
    assert summary['sha256'] != summary['split']['dataset_sha256'] and summary['objects'] == 4


def test_training_starts_list_imagenet_random_stable_and_finished_runs(root, checkout):
    run = root / 'runs' / '2026-10-24_1014'
    for task in ('detector', 'reader'):
        (run / task).mkdir(parents=True)
        (run / task / (task + '-best.pt')).write_bytes(b'x')
    (root / 'runs' / 'unfinished' / 'detector').mkdir(parents=True)
    panel = Panel(root, repo=checkout)
    panel._stable = (time.monotonic(), False)
    starts = {item['key']: item for item in panel.training_starts()}
    assert list(starts) == ['imagenet', 'random', 'stable', 'run:2026-10-24_1014']
    assert starts['imagenet']['default'] and starts['imagenet']['available']
    assert not starts['stable']['available'] and starts['stable']['reason'].startswith('No stable model')


@pytest.mark.parametrize('start, expected', [(None, 'train  init= from='), ('random', 'train  init=random from='),
                                             ('stable', 'train --continue init= from=')])
def test_the_start_of_a_training_becomes_a_flag_or_the_environment(root, checkout, start, expected, monkeypatch):
    (checkout / 'helpers' / '3_train.sh').write_text(ENV_TRAIN)
    panel = Panel(root, repo=checkout)
    # A real training running on this machine would make the panel refuse to start another one.
    monkeypatch.setattr(panel, 'activities', lambda: [])
    panel._stable = (time.monotonic(), True)
    job = panel.start('train', {'start': start} if start else None)
    finished(panel, job['id'])
    assert expected in full_log(panel, job['id'])


def test_a_bad_or_unavailable_start_is_refused(root, checkout, monkeypatch):
    panel = Panel(root, repo=checkout)
    monkeypatch.setattr(panel, 'activities', lambda: [])
    panel._stable = (time.monotonic(), False)
    for options, status in (({'start': 'rm -rf'}, 400), ({'start': 'stable'}, 409), ({'start': 'run:nope'}, 400)):
        with pytest.raises(PanelError) as refused:
            panel.start('train', options)
        assert refused.value.status == status, options
    with pytest.raises(PanelError) as other:
        panel.start('prepare', {'start': 'random'})
    assert other.value.status == 400


def test_jobs_keep_their_temporary_files_in_the_data_root(root, checkout, monkeypatch):
    """Not in /tmp: the service's private /tmp is emptied at every restart of the site."""
    (checkout / 'helpers' / '3_train.sh').write_text('#!/usr/bin/env bash\necho "tmp=$TMPDIR"\n')
    panel = Panel(root, repo=checkout)
    monkeypatch.setattr(panel, 'activities', lambda: [])
    job = panel.start('train')
    finished(panel, job['id'])
    assert f'tmp={root / "tmp"}' in full_log(panel, job['id']) and (root / 'tmp').is_dir()


def test_a_stopped_training_can_be_resumed_on_its_run(root, checkout):
    from nnslr_tools.panel import PanelError, _run_of
    panel = Panel(root, repo=checkout)
    logs = root / 'logs' / 'panel'
    logs.mkdir(parents=True, exist_ok=True)
    log = logs / '20261024-145218-train.log'
    log.write_text('== helpers/3_train.sh\n== a. importo\n== d. training 2026-10-24_1452 (fresh) -> /x\n')
    assert _run_of(log) == '2026-10-24_1452'
    for bad in ({}, {'run': '../etc'}, {'run': 'missing-run'}):
        try:
            panel.start('train_resume', bad)
        except PanelError as error:
            assert error.status == 400
        else:
            raise AssertionError('accepted ' + repr(bad))
    try:
        panel.start('train', {'run': '2026-10-24_1452'})
    except PanelError as error:
        assert error.status == 400


# -- "Train the classifier" (scripts/run_classifier.sh) --------------------------------------------------------------
FAKE_CLASSIFIER = ('#!/usr/bin/env bash\necho "== Classifier (cutter -> classifier -> decider) =="\necho "run:          $1"\n'
                   'echo "args $*"\nsleep "${FAKE_CLASSIFIER_SLEEP:-0}"\n')


def classifier_script(checkout, text=FAKE_CLASSIFIER):
    (checkout / 'scripts').mkdir(exist_ok=True)
    script = checkout / 'scripts' / 'run_classifier.sh'
    script.write_text(text)
    script.chmod(0o755)
    return script


def cutter_in_use(root, panel):
    """A finished pipeline run whose detector is the cutter in use."""
    from test_panel_runs import make_run
    run = make_run(root, 'r1')
    (run / 'detector' / 'detector-best.pt').write_bytes(b'weights')
    panel.models_index.set_active('cutter', 'cutter:r1/detector', who='admin', via='sign-in')
    return run


def ready_payload(gpu_processes=0):
    """A status where every rule of the trainings is met: confirmed frames on three reviewed routes, a free GPU."""
    return {'actions_enabled': True, 'routes': [{'state': 'pronta', 'confirmed': 4}] * 3, 'reviewed_routes': 3,
            'min_routes': 3, 'system': {'gpu': {'processes': gpu_processes}}}


def test_the_classifier_training_follows_the_rules_of_a_training_and_needs_a_cutter(root, checkout):
    panel = Panel(root, repo=checkout)
    assert panel.available()
    # Without the script the other actions are there; this one says why it is not.
    assert panel._action_states(ready_payload(), [])['train_classifier'] == {
        'label': 'Train the classifier', 'allowed': False, 'reason': 'scripts/run_classifier.sh is missing in this installation'}
    assert panel._action_states(ready_payload(), [])['train']['allowed']
    classifier_script(checkout)
    assert panel._action_states(ready_payload(), [])['train_classifier']['reason'] == 'no cutter in use: choose one on the Models tab'
    run = cutter_in_use(root, panel)
    assert panel._action_states(ready_payload(), [])['train_classifier'] == {'label': 'Train the classifier', 'allowed': True, 'reason': ''}
    assert panel._action_states(ready_payload(gpu_processes=1), [])['train_classifier']['reason'] == 'the GPU is busy with another process'
    few = {**ready_payload(), 'reviewed_routes': 2}
    assert panel._action_states(few, [])['train_classifier']['reason'] == '3 reviewed drives are needed, now 2'
    none = {**ready_payload(), 'routes': [{'state': 'revisionare', 'confirmed': 0}]}
    assert panel._action_states(none, [])['train_classifier']['reason'] == 'no confirmed frame'
    busy = [{'group': 'train', 'label': 'classifier training', 'pid': 1}]
    assert panel._action_states(ready_payload(), busy)['train_classifier']['reason'] == 'A classifier training is running'
    (run / 'detector' / 'detector-best.pt').unlink()
    assert panel._action_states(ready_payload(), [])['train_classifier']['reason'] == (
        'the cutter in use is no longer on this server: choose another one on the Models tab')
    # The page's status has it too, with its reason.
    assert panel.status()['actions']['train_classifier']['allowed'] is False


def test_the_classifier_training_runs_on_a_new_run_named_by_its_time(root, checkout, monkeypatch):
    from nnslr_tools.panel import _run_of
    classifier_script(checkout)
    panel = Panel(root, repo=checkout)
    monkeypatch.setattr(panel, 'activities', lambda: [])
    cutter_in_use(root, panel)
    for options in ({'start': 'random'}, {'run': 'r1'}):
        with pytest.raises(PanelError) as refused:
            panel.start('train_classifier', options)
        assert refused.value.status == 400
    job = panel.start('train_classifier')
    assert (job['action'], job['label']) == ('train_classifier', 'Train the classifier')
    done = finished(panel, job['id'])
    log = full_log(panel, job['id'])
    name = log.splitlines()[0].split()[-1]
    assert log.startswith('== scripts/run_classifier.sh ') and f'args {name}\n' in log
    assert re.fullmatch(r'\d{4}-\d{2}-\d{2}_\d{4}-classifier', name)
    assert (done['state'], done['run']) == ('ok', name)
    assert _run_of(panel.logs / (job['id'] + '.log'), 'train_classifier') == name
    # The run line of the script, when the first line is not the panel's.
    other = panel.logs / '20261024-173000-train_classifier.log'
    other.write_text('something\nrun:          2026-10-24_1730-classifier-2\n')
    assert _run_of(other, 'train_classifier') == '2026-10-24_1730-classifier-2' and _run_of(other) is None


def test_the_name_of_a_classifier_run_never_reuses_a_folder(root, checkout, monkeypatch):
    import nnslr_tools.panel as panel_module

    class Fixed(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 10, 24, 17, 30)
    monkeypatch.setattr(panel_module, 'datetime', Fixed)
    panel = Panel(root, repo=checkout)
    assert panel.new_run_name('classifier') == '2026-10-24_1730-classifier'
    (root / 'runs' / '2026-10-24_1730-classifier').mkdir(parents=True)
    assert panel.new_run_name('classifier') == '2026-10-24_1730-classifier-2'
    (root / 'runs' / '2026-10-24_1730-classifier-2').write_text('a file is taken too')
    assert panel.new_run_name('classifier') == '2026-10-24_1730-classifier-3'


def test_the_classifier_training_is_refused_without_its_script_or_a_cutter(root, checkout, monkeypatch):
    panel = Panel(root, repo=checkout)
    monkeypatch.setattr(panel, 'activities', lambda: [])
    with pytest.raises(PanelError) as missing:
        panel.start('train_classifier')
    assert (missing.value.status, missing.value.message) == (403, 'scripts/run_classifier.sh is missing in this installation.')
    classifier_script(checkout)
    with pytest.raises(PanelError) as no_cutter:
        panel.start('train_classifier')
    assert (no_cutter.value.status, no_cutter.value.reason, no_cutter.value.message) == (
        409, 'no_cutter', 'No cutter in use: choose one on the Models tab.')
    assert panel.jobs() == []


def test_a_running_classifier_training_is_seen_and_stopped_after_a_restart(root, checkout, monkeypatch):
    monkeypatch.setenv('FAKE_CLASSIFIER_SLEEP', '30')
    classifier_script(checkout)
    panel = Panel(root, repo=checkout)
    cutter_in_use(root, panel)
    job = panel.start('train_classifier')
    # The wrapper runs the script with its run: the work of the train group, also as an activity.
    assert any(item['label'] == 'classifier training' for item in panel.activities())
    with pytest.raises(PanelError) as busy:
        panel.start('train')
    assert busy.value.status == 409
    restarted = Panel(root, repo=checkout)
    assert [item['stoppable'] for item in restarted.jobs() if item['id'] == job['id']] == [True]
    assert restarted.stop(job['id']) == {'id': job['id'], 'stopping': True}
    assert finished(restarted, job['id'])['state'] in ('interrupted', 'failed')


def test_a_classifier_training_started_from_a_terminal_has_the_progress_of_its_run_log(root, checkout):
    panel = Panel(root, repo=checkout)
    log = root / 'runs' / 'c1' / 'classifier.log'
    log.parent.mkdir(parents=True)
    log.write_text('== 1/7 Check ==\n== 2/7 Smoke ==\n== 3/7 Mine ==\n== 4/7 Train the classifier ==\n'
                   'classifier epoch 2/20 train_loss 0.9000 val_loss 1.0000 train_accuracy 0.5000 val_accuracy 0.6000 30s\n')
    (root / 'logs').mkdir(exist_ok=True)
    (root / 'logs' / 'train-r1.log').write_text('== a. importo\n== 4/10 Train detector ==\n')
    payload = {'jobs': [], 'activities': [{'group': 'train', 'label': 'classifier training', 'pid': 1},
                                          {'group': 'train', 'label': 'dataset and training', 'pid': 2}]}
    panel._terminal_training(payload)
    classifier, training = payload['activities']
    assert classifier['progress']['detail'] == 'classifier · epoch 2 of 20'
    assert [step['key'] for step in classifier['progress']['steps']] == ['checks', 'mine', 'classifier', 'evaluate', 'chain']
    assert training['progress']['steps'][4] == {'key': 'detector', 'label': 'Train the detector', 'state': 'current'}


# -- "Train the cutter" (scripts/run_detector_sizes.sh NAME SIZE) ----------------------------------------------------
FAKE_SIZES = ('#!/usr/bin/env bash\necho "== Detector input sizes =="\necho "run:          $1"\n'
              'echo "args $* families=${NNSLR_CUTTER_FAMILIES:-}"\nsleep "${FAKE_SIZES_SLEEP:-0}"\n')


def sizes_script(checkout, text=FAKE_SIZES):
    (checkout / 'scripts').mkdir(exist_ok=True)
    script = checkout / 'scripts' / 'run_detector_sizes.sh'
    script.write_text(text)
    script.chmod(0o755)
    return script


def test_the_cutter_training_follows_the_rules_of_a_training(root, checkout):
    panel = Panel(root, repo=checkout)
    states = panel._action_states(ready_payload(), [])
    assert states['train_cutter'] == {'label': 'Train the cutter', 'allowed': False,
                                      'reason': 'scripts/run_detector_sizes.sh is missing in this installation'}
    assert states['train']['label'] == 'Whole pipeline'
    sizes_script(checkout)
    # No cutter in use is needed: it trains one.
    assert panel._action_states(ready_payload(), [])['train_cutter'] == {'label': 'Train the cutter', 'allowed': True, 'reason': ''}
    assert panel._action_states(ready_payload(gpu_processes=2), [])['train_cutter']['reason'] == 'the GPU is busy with another process'
    assert panel._action_states({**ready_payload(), 'reviewed_routes': 1}, [])['train_cutter']['reason'] == '3 reviewed drives are needed, now 1'
    status = panel.status()
    assert status['cutter_options'] == {'sizes': ['640x360', '960x544', '1344x760'], 'size': '960x544',
                                        'families': ['all', 'speed'], 'family': 'all'}


@pytest.mark.parametrize('options, expected', [
    (None, '960x544 families=all'), ({'size': '640x360'}, '640x360 families=all'),
    ({'size': '1344x760', 'families': 'speed'}, '1344x760 families=speed')])
def test_the_cutter_training_runs_one_size_on_a_new_run(root, checkout, monkeypatch, options, expected):
    from nnslr_tools.panel import _run_of
    sizes_script(checkout)
    panel = Panel(root, repo=checkout)
    monkeypatch.setattr(panel, 'activities', lambda: [])
    job = panel.start('train_cutter', options)
    done = finished(panel, job['id'])
    log = full_log(panel, job['id'])
    first = log.splitlines()[0].split()
    name = first[2]
    assert first[:2] == ['==', 'scripts/run_detector_sizes.sh'] and first[3] == expected.split()[0]
    assert re.fullmatch(r'\d{4}-\d{2}-\d{2}_\d{4}-cutter', name) and f'args {name} {expected}\n' in log
    assert (done['state'], done['run'], done['label']) == ('ok', name, 'Train the cutter')
    other = panel.logs / '20261024-180000-train_cutter.log'
    other.write_text('== Detector input sizes ==\nrun:          2026-10-24_1800-cutter-2\n')
    assert _run_of(other, 'train_cutter') == '2026-10-24_1800-cutter-2'


@pytest.mark.parametrize('action, options, status', [
    ('train_cutter', {'size': '320x320'}, 400), ('train_cutter', {'size': '960x544 ; rm -rf /'}, 400),
    ('train_cutter', {'families': 'other'}, 400), ('train_cutter', {'start': 'random'}, 400),
    ('train_cutter', {'run': 'r1'}, 400), ('train', {'size': '960x544'}, 400), ('train_classifier', {'families': 'all'}, 400)])
def test_bad_options_of_the_cutter_training_are_refused(root, checkout, monkeypatch, action, options, status):
    sizes_script(checkout)
    classifier_script(checkout)
    panel = Panel(root, repo=checkout)
    monkeypatch.setattr(panel, 'activities', lambda: [])
    with pytest.raises(PanelError) as refused:
        panel.start(action, options)
    assert refused.value.status == status and panel.jobs() == []


def test_the_cutter_training_needs_its_script_and_its_run_is_new(root, checkout, monkeypatch):
    import nnslr_tools.panel as panel_module
    panel = Panel(root, repo=checkout)
    monkeypatch.setattr(panel, 'activities', lambda: [])
    with pytest.raises(PanelError) as missing:
        panel.start('train_cutter')
    assert (missing.value.status, missing.value.message) == (403, 'scripts/run_detector_sizes.sh is missing in this installation.')

    class Fixed(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 10, 24, 18, 0)
    monkeypatch.setattr(panel_module, 'datetime', Fixed)
    (root / 'runs' / '2026-10-24_1800-cutter').mkdir(parents=True)
    assert panel.new_run_name('cutter') == '2026-10-24_1800-cutter-2'


def test_a_running_cutter_training_is_seen_and_stopped_after_a_restart(root, checkout, monkeypatch):
    monkeypatch.setenv('FAKE_SIZES_SLEEP', '30')
    sizes_script(checkout)
    panel = Panel(root, repo=checkout)
    job = panel.start('train_cutter', {'size': '640x360', 'families': 'speed'})
    assert any(item['label'] == 'cutter training' for item in panel.activities())
    with pytest.raises(PanelError) as busy:
        panel.start('train')
    assert (busy.value.status, busy.value.reason) == (409, 'busy')
    # The wrapper has the run and the size after the command: a new server still knows it as this action's.
    restarted = Panel(root, repo=checkout)
    assert [item['stoppable'] for item in restarted.jobs() if item['id'] == job['id']] == [True]
    assert restarted.stop(job['id']) == {'id': job['id'], 'stopping': True}
    assert finished(restarted, job['id'])['state'] in ('interrupted', 'failed')


def test_the_cutter_options_over_http(root, checkout):
    from review_helpers import RunningServer, json_of
    sizes_script(checkout)
    server = RunningServer(root)
    server.app.panel = Panel(root, repo=checkout)
    server.app.panel.activities = lambda: []
    try:
        status, response, data = server.request('POST', '/api/panel/run/train_cutter', body={'size': '1344x760', 'families': 'x'})
        assert status == 400 and json_of(data)['error'] == 'bad_option'
        status, response, data = server.request('POST', '/api/panel/run/train_cutter', body={'size': '1344x760', 'families': 'speed'})
        assert status == 200, data
        job_id = json_of(data)['id']
        finished(server.app.panel, job_id)
        assert ' 1344x760 families=speed\n' in full_log(server.app.panel, job_id)
    finally:
        server.close()


def test_a_cutter_training_started_from_a_terminal_has_the_progress_of_its_size_log(root, checkout):
    panel = Panel(root, repo=checkout)
    log = root / 'runs' / 'k1' / 'detector-sizes' / '960x544.log'
    log.parent.mkdir(parents=True)
    log.write_text('detector epoch 4/20 train_loss 1.1000 val_loss 1.3000 val_recall 0.5000 80s\n')
    (root / 'logs').mkdir(exist_ok=True)
    (root / 'logs' / 'train-r1.log').write_text('== a. importo\n== 5/10 Train reader ==\n')
    payload = {'jobs': [], 'activities': [{'group': 'train', 'label': 'cutter training', 'pid': 1}]}
    panel._terminal_training(payload)
    progress = payload['activities'][0]['progress']
    assert [(step['key'], step['state']) for step in progress['steps']] == [('checks', 'done'), ('detector', 'current'), ('evaluate', 'todo')]
    assert progress['detail'] == 'detector · epoch 4 of 20' and progress['curves']['detector']['points'][0]['val_recall'] == 0.5
