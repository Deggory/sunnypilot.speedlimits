"""Train Fit box in the panel (scripts/train_box_fit.sh) on Models: its options (a dry run, without the refiner), its
refusals, its group of its own (a card of its own or shared, beside the other trainings, never two at once), its progress read from the log
and the last summaries the card shows (GET /api/panel/box-fit). Nothing is trained: the script is a stand-in that prints
its arguments."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nnslr_tools.job_progress import progress  # noqa: E402
from nnslr_tools.panel import ACTION_OPTIONS, ACTIONS, CONFLICTS, Panel, PanelError, _conflicting_group  # noqa: E402
from review_helpers import RunningServer, json_of  # noqa: E402
from test_panel import checkout, finished, full_log, root  # noqa: E402,F401  (fixtures and helpers)


@pytest.fixture(autouse=True)
def no_cards_of_this_server(monkeypatch):
    """Not the server's own cards (a real run takes one, #65): no nvidia-smi and no holder of the GPU lock."""
    import nnslr_tools.panel as panel_module
    monkeypatch.setattr(panel_module, '_nvidia', lambda arguments: None)
    monkeypatch.setattr(panel_module, '_gpu_lock_holders', lambda path=None: [])


def ready_payload(gpu_processes=0):
    return {'actions_enabled': True, 'routes': [], 'reviewed_routes': 0, 'min_routes': 3,
            'system': {'gpu': {'processes': gpu_processes}}}


def box_fit_script(checkout, venv=True, hold=0.0):
    """A stand-in for scripts/train_box_fit.sh that prints its arguments (and waits `hold` seconds), and the Python of
    .venv-gpu the refiner needs."""
    script = checkout / 'scripts' / 'train_box_fit.sh'
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text('#!/usr/bin/env bash\necho "arguments: $*"\n' + (f'sleep {hold}\n' if hold else ''))
    script.chmod(0o755)
    if venv:
        python = checkout / '.venv-gpu' / 'bin' / 'python'
        python.parent.mkdir(parents=True, exist_ok=True)
        python.write_text('#!/bin/sh\n')
        python.chmod(0o755)
    return script


def test_the_action_is_in_the_panel_with_its_options():
    assert ACTIONS['box_fit_train'] == {'label': 'Train Fit box', 'group': 'box_fit',
                                        'command': ['scripts/train_box_fit.sh']}
    assert ACTION_OPTIONS['box_fit_train'] == ('dry', 'refiner')


def test_it_runs_beside_a_gpu_training_but_never_twice_nor_under_an_update():
    assert _conflicting_group('box_fit', {'train'}) is None
    assert _conflicting_group('train', {'box_fit'}) is None
    assert _conflicting_group('box_fit', {'box_fit'}) == 'box_fit'
    assert _conflicting_group('box_fit', {'update'}) == 'update'
    assert 'box_fit' in CONFLICTS['update']


def test_the_review_server_passes_the_refiner_on():
    # review.py's own list of the keys it hands to Panel.start (a key left out of it is dropped without a word: the
    # refiner turned off and dropped would start the hour of the refiner).
    text = (Path(__file__).resolve().parents[2] / 'src/nnslr_tools/review.py').read_text(encoding='utf-8')
    assert "for key in ('refiner',)" in text and "for key in ('dry', 'labeler', 'marks')" in text


def test_it_says_why_it_cannot_start(root, checkout):
    panel = Panel(root, repo=checkout)
    reason = lambda **kw: panel._action_states(ready_payload(**kw), [])['box_fit_train']  # noqa: E731
    assert reason()['reason'] == 'scripts/train_box_fit.sh is missing in this installation'
    box_fit_script(checkout)
    # No confirmed frame nor reviewed drives asked (the script counts the boxes), and a busy GPU leaves it allowed: the
    # card is taken when it starts, so a dry run on the CPU stays possible.
    assert reason()['allowed'] is True
    assert reason(gpu_processes=2)['allowed'] is True
    # Another training of Fit box from a terminal: it waits for it.
    states = panel._action_states(ready_payload(), [{'group': 'box_fit', 'label': 'Fit box training', 'pid': 1}])
    assert states['box_fit_train'] == {'label': 'Train Fit box', 'allowed': False,
                                       'reason': 'A training of Fit box is running'}
    # A training on the GPU does not stop it.
    states = panel._action_states(ready_payload(), [{'group': 'train', 'label': 'cutter training', 'pid': 1}])
    assert states['box_fit_train']['allowed'] is True


@pytest.fixture
def ready(root, checkout, monkeypatch):
    panel = Panel(root, repo=checkout)
    box_fit_script(checkout)
    monkeypatch.setattr(panel, 'activities', lambda: [])
    monkeypatch.setattr(panel, 'system', lambda: {'gpu': {'processes': 0}})
    return panel


def start(panel, action, options=None):
    """panel.start, again a moment later when a job started in the same second (a job's id is its second)."""
    for _ in range(40):
        try:
            return panel.start(action, options)
        except PanelError as busy:
            if busy.reason != 'busy':
                raise
            time.sleep(0.05)
    raise AssertionError('the panel stayed busy')


def arguments(panel, job):
    finished(panel, job['id'])
    return next(line for line in full_log(panel, job['id']).splitlines() if line.startswith('arguments:'))


def test_a_dry_run_starts_with_the_gpu_busy_a_real_run_waits(ready, monkeypatch):
    # The dry run counts the boxes on the CPU; a real run trains on a card (#65).
    monkeypatch.setattr(ready, 'system', lambda: {'gpu': {'processes': 2}})
    assert arguments(ready, start(ready, 'box_fit_train', {'dry': 'on'})) == 'arguments: --dry-run'
    with pytest.raises(PanelError) as refused:
        ready.start('box_fit_train', {'refiner': 'off'})
    assert (refused.value.status, refused.value.reason) == (409, 'busy')


def test_the_options_reach_the_script(ready):
    assert arguments(ready, start(ready, 'box_fit_train')) == 'arguments: '
    assert arguments(ready, start(ready, 'box_fit_train', {'refiner': 'on'})) == 'arguments: '
    assert arguments(ready, start(ready, 'box_fit_train', {'refiner': 'off'})) == 'arguments: --no-refiner'
    assert arguments(ready, start(ready, 'box_fit_train', {'dry': 'on'})) == 'arguments: --dry-run'
    assert arguments(ready, start(ready, 'box_fit_train', {'dry': 'on', 'refiner': 'off'})) == (
        'arguments: --dry-run --no-refiner')


def test_the_steps_come_from_the_panels_own_log(ready):
    # The first line the panel writes (the command) tells a dry run and the rankers alone apart.
    def steps(options):
        job = start(ready, 'box_fit_train', options)
        finished(ready, job['id'])
        return [step['key'] for step in progress('box_fit_train', 'ok', full_log(ready, job['id']))['steps']]
    assert steps({'dry': 'on'}) == ['boxes']
    assert steps({'dry': 'on', 'refiner': 'off'}) == ['boxes']
    assert steps({'refiner': 'off'}) == ['boxes', 'rankers', 'refiner']
    assert steps(None) == ['boxes', 'rankers', 'refiner']


@pytest.mark.parametrize('options', [
    {'dry': 'yes'}, {'dry': True}, {'dry': 'off'}, {'refiner': False}, {'refiner': 'no'}, {'refiner': ['off']},
    {'labeler': 'x'}, {'marks': 'off'}, {'start': 'random'},
])
def test_a_bad_or_unknown_option_is_refused_and_nothing_starts(ready, options):
    with pytest.raises(PanelError) as refused:
        ready.start('box_fit_train', options)
    assert (refused.value.status, refused.value.reason) == (400, 'bad_option')
    assert ready.jobs() == []


def test_no_other_action_takes_the_refiner(ready):
    for action in ('road_labeler_train', 'train_brain', 'fetch'):
        with pytest.raises(PanelError) as refused:
            ready.start(action, {'refiner': 'off'})
        assert refused.value.reason == 'bad_option'


def test_a_real_run_needs_venv_gpu_but_a_dry_run_does_not(root, checkout, monkeypatch):
    # The trees, the MLPs and the refiner train on the GPU with the PyTorch of .venv-gpu (#65); the dry run only counts.
    panel = Panel(root, repo=checkout)
    box_fit_script(checkout, venv=False)
    monkeypatch.setattr(panel, 'activities', lambda: [])
    for options in (None, {'refiner': 'on'}, {'refiner': 'off'}):
        with pytest.raises(PanelError) as refused:
            panel.start('box_fit_train', options)
        assert (refused.value.status, refused.value.reason) == (409, 'not_available')
    assert panel.jobs() == []
    assert arguments(panel, start(panel, 'box_fit_train', {'dry': 'on'})) == 'arguments: --dry-run'


def test_a_second_one_waits_for_the_first(root, checkout, monkeypatch):
    panel = Panel(root, repo=checkout)
    box_fit_script(checkout, hold=3)
    monkeypatch.setattr(panel, 'activities', lambda: [])
    job = panel.start('box_fit_train', {'dry': 'on'})
    time.sleep(1.1)
    with pytest.raises(PanelError) as refused:
        panel.start('box_fit_train', {'dry': 'on'})
    assert (refused.value.status, refused.value.reason) == (409, 'busy')
    assert 'A training of Fit box is running' in refused.value.message
    finished(panel, job['id'])


# -- the versions the card shows (#94) --------------------------------------------------------------------------------
RUN = '2026-11-16_0900-box-fit'
RECORD = {'schema_version': 1, 'run': RUN, 'at': '2026-11-16T09:40:00+00:00', 'boxes': 4100, 'routes': 31, 'folds': 4,
          'rankers': {'mean_iou': 0.8123, 'rough_mean_iou': 0.6012}, 'refiner': 'trained', 'refiner_from': None,
          'refiner_boxes': 3000, 'held_out_boxes': 900, 'in_use_then': 'box_fit:code',
          'left_out': {'all': {'cases': 2700, 'rough': 0.6, 'rankers': 0.81, 'version': 0.84, 'in_use': 0.83,
                               'rankers>=0.8': 0.7},
                       'size:small': {'cases': 800, 'rough': 0.5, 'rankers': 0.7, 'version': 0.74, 'in_use': 0.75}},
          'measured_clean': {'version': True, 'in_use': False}, 'better_than_in_use': True}


def a_version(root, run=RUN, *, record=RECORD, refiner=True):
    """A version of Fit box in the data root: its run folder with its files (stand-ins) and its record."""
    folder = root / 'runs' / run
    folder.mkdir(parents=True)
    for name in ('colour.npz', 'edges.npz', *(('refiner.onnx',) if refiner else ())):
        (folder / name).write_bytes(b'x')
    if record is not None:
        (folder / 'box-fit.json').write_text(json.dumps(record))
    return folder


def test_the_card_lists_every_version_newest_first_and_the_one_in_use(root, checkout):
    panel = Panel(root, repo=checkout)
    view = panel.box_fit()
    assert [version['id'] for version in view['versions']] == ['box_fit:code'] and view['versions'][0]['in_use']
    assert view['in_use'] == {'id': 'box_fit:code', 'found': True, 'problem': None, 'who': None, 'via': None, 'at': None}
    a_version(root)
    a_version(root, '2026-11-15_1026-box-fit', record=None, refiner=False)
    # A training still running, a folder without the rankers, a link: not versions.
    (root / 'runs' / f'.{RUN}.partial').mkdir()
    (root / 'runs' / '2026-11-16_1100-box-fit').mkdir()
    (root / 'runs' / '2026-11-16_1200-box-fit').symlink_to(root / 'runs' / RUN)
    view = panel.box_fit()
    assert [version['id'] for version in view['versions']] == [
        'box_fit:' + RUN, 'box_fit:2026-11-15_1026-box-fit', 'box_fit:code']
    assert view['versions'][0] == {
        'id': 'box_fit:' + RUN, 'run': RUN, 'at': '2026-11-16T09:40:00+00:00',
        'files': ['colour.npz', 'edges.npz', 'refiner.onnx'], 'in_use': False, 'record': True, 'boxes': 4100,
        'routes': 31, 'rankers_fit': 0.8123, 'rankers_rough': 0.6012, 'refiner': 'trained', 'refiner_from': None,
        'refiner_boxes': 3000, 'in_use_then': 'box_fit:code', 'better_than_in_use': True,
        'all': {'cases': 2700, 'rough': 0.6, 'rankers': 0.81, 'version': 0.84, 'in_use': 0.83},
        'small': {'cases': 800, 'rough': 0.5, 'rankers': 0.7, 'version': 0.74, 'in_use': 0.75}, 'fair': None,
        'measured_clean': {'version': True, 'in_use': False}}
    older = view['versions'][1]
    assert (older['record'], older['refiner'], older['all'], older['files']) == (False, 'none', None,
                                                                                 ['colour.npz', 'edges.npz'])
    assert older['measured_clean'] == {'version': None, 'in_use': None}
    assert [version['in_use'] for version in view['versions']] == [False, False, True]


def test_a_version_is_chosen_by_its_id_and_nothing_else(root, checkout):
    panel = Panel(root, repo=checkout)
    a_version(root)
    entry = panel.set_box_fit({'id': 'box_fit:' + RUN}, who='cristianku', via='sign-in')
    assert (entry['id'], entry['who'], entry['via']) == ('box_fit:' + RUN, 'cristianku', 'sign-in')
    saved = json.loads((root / 'models' / 'box-fit.json').read_text())
    assert saved['id'] == 'box_fit:' + RUN and saved['schema_version'] == 1
    view = panel.box_fit()
    assert view['in_use']['id'] == 'box_fit:' + RUN and view['in_use']['who'] == 'cristianku'
    assert [version['in_use'] for version in view['versions']] == [True, False]
    for body, status, reason in (([], 400, 'invalid_json'), ({'id': '../../etc'}, 400, 'bad_id'), ({'id': 5}, 400, 'bad_id'),
                                 ({'id': 'box_fit:2026-11-17_0000-box-fit'}, 404, 'unknown_version'),
                                 ({'id': 'cutter:' + RUN}, 400, 'bad_id')):
        with pytest.raises(PanelError) as refused:
            panel.set_box_fit(body, who=None, via='local')
        assert (refused.value.status, refused.value.reason) == (status, reason), body
    with pytest.raises(PanelError) as refused:
        panel.set_box_fit({'id': 'box_fit:code'}, who='guest:x', via='reviewer')
    assert refused.value.status == 403
    with pytest.raises(PanelError) as refused:
        Panel(root, repo=checkout, actions_enabled=False).set_box_fit({'id': 'box_fit:code'}, who=None, via='local')
    assert refused.value.reason == 'actions_disabled'
    assert json.loads((root / 'models' / 'box-fit.json').read_text())['id'] == 'box_fit:' + RUN


def test_a_chosen_version_that_is_gone_falls_back_to_the_code_and_says_so(root, checkout):
    import shutil
    panel = Panel(root, repo=checkout)
    a_version(root)
    panel.set_box_fit({'id': 'box_fit:' + RUN}, who=None, via='local')
    shutil.rmtree(root / 'runs' / RUN)
    used = panel.box_fit()['in_use']
    assert used['id'] == 'box_fit:code' and used['found'] is False and 'no longer on this server' in used['problem']
    # A choice that cannot be read is never overwritten.
    (root / 'models' / 'box-fit.json').write_text('{"id": ')
    with pytest.raises(PanelError) as refused:
        panel.set_box_fit({'id': 'box_fit:code'}, who=None, via='local')
    assert (refused.value.status, refused.value.reason) == (409, 'choice_unreadable')
    assert (root / 'models' / 'box-fit.json').read_text() == '{"id": '
    assert 'cannot be read' in panel.box_fit()['in_use']['problem']


def test_the_review_server_serves_the_versions_and_takes_a_choice(root, checkout):
    box_fit_script(checkout)
    a_version(root)
    server = RunningServer(root)
    server.app.panel = Panel(root, repo=checkout)
    server.app.panel.activities = list
    try:
        for body in ({'dry': True}, {'refiner': False}, {'refiner': 0}):
            status, _, data = server.request('POST', '/api/panel/run/box_fit_train', body=body)
            assert status == 400 and json_of(data)['error'] == 'bad_option', body
        assert server.app.panel.jobs() == []
        status, _, data = server.request('POST', '/api/panel/run/box_fit_train', body={'dry': 'on', 'refiner': 'off'})
        assert status == 200, data
        assert arguments(server.app.panel, json_of(data)) == 'arguments: --dry-run --no-refiner'
        status, _, data = server.request('GET', '/api/panel/box-fit')
        assert status == 200 and json_of(data)['versions'][0]['boxes'] == 4100
        assert server.request('GET', '/api/panel/box-fit/use')[0] == 405
        use = '/api/panel/box-fit/use'
        status, _, data = server.request('POST', use, body={'id': 'box_fit:' + RUN},
                                         headers={'Origin': 'http://evil.example'})
        assert status == 403 and json_of(data)['error'] == 'cross_origin'
        status, _, data = server.request('POST', use, body=b'{}', headers={'Content-Type': 'text/plain'})
        assert status == 415
        status, _, data = server.request('POST', use, body={'id': 'box_fit:nope'})
        assert status == 400 and json_of(data)['error'] == 'bad_id'
        assert not (root / 'models' / 'box-fit.json').exists()
        status, _, data = server.request('POST', use, body={'id': 'box_fit:' + RUN})
        assert status == 200 and json_of(data)['in_use']['id'] == 'box_fit:' + RUN
        status, _, data = server.request('GET', '/api/panel/box-fit')
        assert json_of(data)['in_use']['id'] == 'box_fit:' + RUN
    finally:
        server.close()


# -- its progress -------------------------------------------------------------------------------------------------
HEAD = """== scripts/train_box_fit.sh
1/3 The boxes drawn by people (dry run)
{"boxes": 4100, "min_boxes": 200, "dry_run": true}
2/3 Measure on routes left out, then train on every box (12 processes)
12300 cases extracted in 300 s
fold 0: 3000 cases, mean IoU 0.81 (rough 0.60), trained in 40 s
"""
REFINER = """{"boxes": 4100}
Summary in /data/evaluations/box_fit/20261109T100000Z.json.
3/3 The box refiner: train on the CPU, measure against the rankers on routes left out (12 threads)
3000 boxes to train on, 2700 rough boxes left out, cut in 20 s
epoch 1/40 loss 0.4000 60 s, mean IoU left out 0.7100
epoch 2/40 loss 0.3000 61 s, mean IoU left out 0.7600
"""


def test_it_goes_step_by_step_then_epoch_by_epoch():
    result = progress('box_fit_train', 'running', HEAD)
    assert [(step['key'], step['state']) for step in result['steps']] == [
        ('boxes', 'done'), ('rankers', 'current'), ('refiner', 'todo')]
    assert result['percent'] == pytest.approx(2.0)
    result = progress('box_fit_train', 'running', HEAD + REFINER)
    assert [step['state'] for step in result['steps']] == ['done', 'done', 'current']
    # 2 + 20 + 78 x 2/40.
    assert result['percent'] == pytest.approx(25.9)
    assert result['detail'] == 'Box refiner · epoch 2 of 40'
    assert result['step_progress']['text'] == 'epoch 2 of 40'
    assert result['last_line'] == 'epoch 2/40 loss 0.3000 61 s, mean IoU left out 0.7600'
    assert progress('box_fit_train', 'ok', HEAD + REFINER)['percent'] == 100.0


def test_without_the_refiner_or_dry_it_has_fewer_steps():
    # #94: without the refiner, step 3 only measures the new version against the one in use.
    log = HEAD.replace('== scripts/train_box_fit.sh', '== scripts/train_box_fit.sh --no-refiner')
    result = progress('box_fit_train', 'running', log)
    assert [(step['key'], step['state']) for step in result['steps']] == [
        ('boxes', 'done'), ('rankers', 'current'), ('refiner', 'todo')]
    assert result['steps'][2]['label'] == 'Measure against the version in use'
    assert result['percent'] == pytest.approx(5.0)
    # The script of before #94 numbered the steps of --no-refiner out of 2: those two.
    old = log.replace('/3 ', '/2 ')
    assert [step['key'] for step in progress('box_fit_train', 'running', old)['steps']] == ['boxes', 'rankers']
    assert len(progress('box_fit_train', 'running', HEAD.replace('/3 ', '/2 '))['steps']) == 2
    dry = '== scripts/train_box_fit.sh --dry-run\n1/3 The boxes drawn by people (dry run)\n'
    assert [step['key'] for step in progress('box_fit_train', 'running', dry)['steps']] == ['boxes']
    assert progress('box_fit_train', 'ok', dry)['steps'] == [
        {'key': 'boxes', 'label': 'Count the boxes drawn by people', 'state': 'done'}]


@pytest.mark.parametrize('line,text', [
    ('Too few boxes drawn by people to train the box fit: nothing changed.',
     'Too few boxes drawn by people to train Fit box yet: nothing changed.'),
    ('The PyTorch environment (.venv-gpu with torch, onnx and opencv) is missing: run '
     'scripts/setup_v100_training_env.sh. The refiner is unchanged.',
     'The PyTorch of .venv-gpu is missing (scripts/setup_v100_training_env.sh): no new version, nothing changed.'),
    ('ValueError: only 12 boxes to train on: too few for the refiner',
     'Too few confirmed boxes to train the box refiner: no new version, nothing changed.'),
])
def test_a_known_failure_says_so(line, text):
    assert progress('box_fit_train', 'failed', HEAD + line + '\n')['failure'] == text
