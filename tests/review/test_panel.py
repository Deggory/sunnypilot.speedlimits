"""Control panel: route state, the fixed actions run in the background with their log, HTTP protections.

The helpers are fakes in a temporary checkout (echo, sleep, exit codes): nothing real is prepared or
trained.
"""
from __future__ import annotations

import shutil
import time
from pathlib import Path

import pytest

from nnslr_tools.pages import asset
from nnslr_tools.panel import Panel, PanelError
from review_helpers import ROUTE, TOKEN, RunningServer, json_of, make_report, sign_in

REPO = Path(__file__).resolve().parents[2]
FAKE_PREPARE = '#!/usr/bin/env bash\necho "prepare $*"\necho "data root $NNSLR_DATA_ROOT"\nsleep "${FAKE_SLEEP:-0}"\n'
FAKE_TRAIN = '#!/usr/bin/env bash\necho "train $*"\nexit 3\n'
FAKE_UPDATE = '#!/usr/bin/env bash\necho "update $*"\nsleep "${FAKE_UPDATE_SLEEP:-0}"\n'
FAKE_FETCH = '#!/usr/bin/env bash\necho "fetch $*"\necho "python $COMMA_FETCH_PYTHON"\nsleep "${FAKE_FETCH_SLEEP:-0}"\n'


@pytest.fixture(autouse=True)
def only_this_checkout(monkeypatch):
    """A download or a training really running on this machine must not make these tests busy: here every
    group counts only from the test checkout."""
    import nnslr_tools.panel as panel_module
    monkeypatch.setattr(panel_module, 'OWN_GROUPS', ('fetch', 'prepare', 'train', 'update', 'other'))


@pytest.fixture
def checkout(tmp_path):
    repo = tmp_path / 'repo'
    (repo / 'helpers').mkdir(parents=True)
    shutil.copy(REPO / 'helpers/route_status.py', repo / 'helpers/route_status.py')
    for name, text in (('1_prepare.sh', FAKE_PREPARE), ('3_train.sh', FAKE_TRAIN), ('update.sh', FAKE_UPDATE),
                       ('fetch_comma_segments.sh', FAKE_FETCH)):
        script = repo / 'helpers' / name
        script.write_text(text)
        script.chmod(0o755)
    return repo


@pytest.fixture
def root(tmp_path):
    data = tmp_path / 'data'
    make_report(data)
    return data


def finished(panel, job_id, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = next(j for j in panel.jobs() if j['id'] == job_id)
        if job['state'] != 'running':
            return job
        time.sleep(0.05)
    raise AssertionError(f'{job_id} still running')


def full_log(panel, job_id):
    return panel.log(job_id, 0)['text']


def test_status_explains_why_each_action_is_not_available(root, checkout, monkeypatch, tmp_path):
    monkeypatch.setenv('COMMA_VIDEOS_DIR', str(tmp_path / 'no-videos'))
    status = Panel(root, repo=checkout).status()
    assert status['available'] and status['actions_enabled']
    assert [row['route'] for row in status['routes']] == [ROUTE]
    assert status['routes'][0]['state'] == 'revisionare'
    assert status['next_action'] == 'review'
    assert status['actions']['prepare'] == {'label': 'Prepare new routes', 'allowed': False,
                                            'reason': 'no new route to prepare'}
    assert status['actions']['dataset']['reason'] == 'no confirmed frame'
    assert status['actions']['train']['allowed'] is False
    # The machine for the Stato page: GPU, CPU per core, RAM, disk, as distill-web's Status.
    assert set(status['system']) == {'gpu', 'disk', 'cpus', 'load', 'loads', 'cpu', 'memory'}


def test_action_runs_in_background_and_keeps_its_exit_code(root, checkout):
    panel = Panel(root, repo=checkout)
    job = panel.start('prepare')
    assert job['action'] == 'prepare' and job['label'] == 'Prepare new routes'
    done = finished(panel, job['id'])
    assert (done['state'], done['code']) == ('ok', 0)
    text = full_log(panel, job['id'])
    assert 'prepare --jobs=2' in text and f'data root {root}' in text
    assert text.rstrip().endswith('== finito con codice 0')
    failed = finished(panel, panel.start('dataset')['id'])
    assert (failed['state'], failed['code']) == ('failed', 3)
    assert 'train --dataset-only' in full_log(panel, failed['id'])
    # A new server (restart) still knows how the actions ended, from their logs.
    states = {j['action']: j['state'] for j in Panel(root, repo=checkout).jobs()}
    assert states == {'prepare': 'ok', 'dataset': 'failed'}


def test_one_action_per_kind_and_stop(root, checkout, monkeypatch):
    monkeypatch.setenv('FAKE_SLEEP', '30')
    panel = Panel(root, repo=checkout)
    job = panel.start('prepare')
    with pytest.raises(PanelError) as busy:
        panel.start('prepare')
    assert busy.value.status == 409
    assert any(item['label'] == 'preparation' for item in panel.activities())
    assert panel.status()['actions']['prepare']['reason'] == 'A preparation is running'
    assert panel.stop(job['id']) == {'id': job['id'], 'stopping': True}
    assert finished(panel, job['id'])['state'] in ('interrupted', 'failed')
    with pytest.raises(PanelError) as unknown:
        panel.stop('20260101-000000-prepare')
    assert unknown.value.status == 404


def test_stop_works_after_a_restart_of_the_server(root, checkout, monkeypatch):
    monkeypatch.setenv('FAKE_SLEEP', '30')
    job = Panel(root, repo=checkout).start('prepare')
    # A new Panel is what the server has after a restart: it never saw the process start.
    restarted = Panel(root, repo=checkout)
    assert [j['stoppable'] for j in restarted.jobs() if j['id'] == job['id']] == [True]
    assert restarted.stop(job['id']) == {'id': job['id'], 'stopping': True}
    assert finished(restarted, job['id'])['state'] in ('interrupted', 'failed')
    with pytest.raises(PanelError) as done:
        restarted.stop(job['id'])
    assert done.value.status == 409


def test_update_runs_without_printing_the_token_link(root, checkout):
    panel = Panel(root, repo=checkout)
    assert panel.status()['actions']['update'] == {'label': 'Update the code', 'allowed': True, 'reason': ''}
    done = finished(panel, panel.start('update')['id'])
    assert (done['state'], done['code']) == ('ok', 0)
    assert 'update --no-link' in full_log(panel, done['id'])


def test_update_never_runs_together_with_a_preparation_or_a_training(root, checkout, monkeypatch):
    monkeypatch.setenv('FAKE_SLEEP', '30')
    panel = Panel(root, repo=checkout)
    job = panel.start('prepare')
    with pytest.raises(PanelError) as busy:
        panel.start('update')
    assert busy.value.status == 409
    assert panel.status()['actions']['update']['reason'] == 'A preparation is running'
    panel.stop(job['id'])
    finished(panel, job['id'])

    monkeypatch.setenv('FAKE_UPDATE_SLEEP', '30')
    update = panel.start('update')
    for action in ('prepare', 'dataset'):
        with pytest.raises(PanelError) as busy:
            panel.start(action)
        assert busy.value.status == 409
    assert panel.status()['actions']['prepare']['reason'] == 'A code update is running'
    panel.stop(update['id'])
    finished(panel, update['id'])


def test_update_needs_its_helper(root, checkout):
    (checkout / 'helpers/update.sh').unlink()
    panel = Panel(root, repo=checkout)
    assert panel.status()['actions']['update']['reason'] == 'helpers/update.sh is missing in this installation'
    with pytest.raises(PanelError) as missing:
        panel.start('update')
    assert missing.value.status == 403


@pytest.fixture
def no_log_reader(monkeypatch):
    monkeypatch.delenv('COMMA_FETCH_PYTHON', raising=False)
    monkeypatch.delenv('NNSLR_OPENPILOT_ROOT', raising=False)


def test_fetch_uses_the_log_reader_of_openpilot_distill(root, checkout, no_log_reader):
    distill = checkout.parent / 'openpilot.distill/.venv/bin/python'
    distill.parent.mkdir(parents=True)
    distill.write_text('#!/bin/sh\n')
    distill.chmod(0o755)
    panel = Panel(root, repo=checkout)
    # No confirmed frame needed: the download comes before everything else.
    assert panel.status()['actions']['fetch'] == {'label': 'Download from the comma', 'allowed': True, 'reason': ''}
    done = finished(panel, panel.start('fetch')['id'])
    assert (done['state'], done['code']) == ('ok', 0)
    text = full_log(panel, done['id'])
    assert 'fetch \n' in text and f'python {distill}' in text


def test_fetch_needs_a_log_reader(root, checkout, no_log_reader):
    panel = Panel(root, repo=checkout)
    assert panel.status()['actions']['fetch']['reason'] == 'the reader of the comma logs is missing (see docs/GUIDA.md)'
    with pytest.raises(PanelError) as missing:
        panel.start('fetch')
    assert missing.value.status == 403


def test_fetch_never_runs_together_with_a_preparation_or_an_update(root, checkout, monkeypatch):
    monkeypatch.setenv('COMMA_FETCH_PYTHON', '/usr/bin/python3')
    monkeypatch.setenv('FAKE_FETCH_SLEEP', '30')
    panel = Panel(root, repo=checkout)
    fetch = panel.start('fetch')
    assert any(item['label'] == 'download from the comma' for item in panel.activities())
    for action in ('fetch', 'prepare', 'update'):
        with pytest.raises(PanelError) as busy:
            panel.start(action)
        assert busy.value.status == 409
    assert panel.status()['actions']['prepare']['reason'] == 'A download from the comma is running'
    # A training does not touch the videos.
    assert panel.status()['actions']['dataset']['reason'] != 'A download from the comma is running'
    panel.stop(fetch['id'])
    finished(panel, fetch['id'])

    monkeypatch.setenv('FAKE_SLEEP', '30')
    prepare = panel.start('prepare')
    assert panel.status()['actions']['fetch']['reason'] == 'A preparation is running'
    panel.stop(prepare['id'])
    finished(panel, prepare['id'])


def make_route(root, route, segments, extracted, frames_per_segment=4):
    for index in range(segments):
        (root / 'raw/routes' / route / str(index)).mkdir(parents=True)
        (root / 'raw/routes' / route / str(index) / 'fcamera.hevc').write_bytes(b'video')
    for index in range(extracted):
        manifest = root / 'derived/frames' / route / str(index) / 'frames.jsonl'
        manifest.parent.mkdir(parents=True)
        manifest.write_text('{}\n' * frames_per_segment)


def test_preparation_progress_from_what_is_on_disk(root, checkout, monkeypatch, tmp_path):
    monkeypatch.setenv('COMMA_VIDEOS_DIR', str(tmp_path / 'no-videos'))
    extracting = '0000011f--77d17f9ee3'
    annotating = '00000122--2a20602737'
    make_route(root, extracting, segments=4, extracted=1)
    make_route(root, annotating, segments=2, extracted=2)
    pending = root / 'derived/preannotations' / annotating / '.pending-20261021T200000Z-abcd1234'
    pending.mkdir(parents=True)
    (pending / 'preannotations.jsonl').write_text('{}\n' * 4)
    panel = Panel(root, repo=checkout)
    rows = panel.status()['routes']
    assert panel.status()['preparation'] is None  # nothing running
    running = [{'group': 'prepare', 'label': 'preparation', 'pid': 1}]

    # A run of an older 1_prepare.sh: no batch file, no overall percent.
    assert panel.preparation(rows, running) == {'percent': None, 'routes_done': 0, 'routes_total': 0, 'working': []}

    (root / 'logs/prepare').mkdir(parents=True)
    (root / 'logs/prepare/batch.json').write_text(
        '{"started": "2026-10-21T20:00:00Z", "routes": ["%s", "%s", "%s"]}' % (ROUTE, extracting, annotating))
    progress = panel.preparation(rows, running)
    # ROUTE has its report: 100. extracting: 1/4 segments = 12. annotating: 4/8 frames = 75.
    assert (progress['percent'], progress['routes_done'], progress['routes_total']) == ((100 + 12 + 75) // 3, 1, 3)
    assert panel._route_progress(next(r for r in rows if r['route'] == extracting)) == {
        'route': extracting, 'phase': 'extracting frames', 'done': 1, 'total': 4, 'unit': 'segments', 'percent': 12}
    assert panel._route_progress(next(r for r in rows if r['route'] == annotating)) == {
        'route': annotating, 'phase': 'proposing signs', 'done': 4, 'total': 8, 'unit': 'frames', 'percent': 75}


def test_read_only_panel_and_unknown_actions(root, checkout):
    panel = Panel(root, repo=checkout, actions_enabled=False)
    with pytest.raises(PanelError) as disabled:
        panel.start('prepare')
    assert disabled.value.status == 403
    assert panel.status()['actions']['train']['reason'] == 'the panel is read-only'
    with pytest.raises(PanelError) as unknown:
        Panel(root, repo=checkout).start('rm')
    assert unknown.value.status == 404
    with pytest.raises(PanelError):
        panel.log('../../etc/passwd', 0)


def test_without_helpers_the_panel_is_read_only(root, tmp_path):
    status = Panel(root, repo=tmp_path / 'wheel-install').status()
    assert status['available'] is False and status['actions_enabled'] is False
    assert status['routes'] == []


@pytest.fixture
def served(root, checkout):
    server = RunningServer(root)
    server.app.panel = Panel(root, repo=checkout)
    yield server
    server.close()


def test_panel_page_status_and_run_over_http(served):
    status, response, data = served.request('GET', '/')
    assert status == 200
    csp = response.getheader('Content-Security-Policy')
    assert "script-src 'nonce-" in csp and "connect-src 'self'" in csp
    nonce = csp.split("'nonce-")[1].split("'")[0]
    assert f'<script nonce="{nonce}">' in data.decode()
    # The tabs are one page (Overview is the home page); the old addresses lead to them.
    for path in ('/prepare', '/train', '/evaluate'):
        assert served.request('GET', path)[0] == 200, path
    for path, target in (('/pannello', '/'), ('/stato', '/'), ('/panoramica', '/'), ('/prepara', '/prepare'),
                         ('/allena', '/train')):
        status, response, data = served.request('GET', path)
        assert status == 303 and response.getheader('Location') == target, path
    payload = json_of(served.request('GET', '/api/panel/status')[2])
    assert payload['routes'][0]['route'] == ROUTE and 'prepare' in payload['actions']

    status, response, data = served.request('POST', '/api/panel/run/prepare', body={})
    assert status == 200, data
    job_id = json_of(data)['id']
    finished(served.app.panel, job_id)
    log = json_of(served.request('GET', f'/api/panel/job/{job_id}/log?offset=0')[2])
    assert 'prepare --jobs=2' in log['text'] and log['offset'] == log['size']
    rest = json_of(served.request('GET', f'/api/panel/job/{job_id}/log?offset={log["offset"]}')[2])
    assert rest['text'] == ''


def test_panel_actions_refuse_other_origins_forms_and_unknown_actions(served):
    status, response, data = served.request('POST', '/api/panel/run/prepare', body={},
                                            headers={'Origin': 'http://evil.example'})
    assert status == 403 and json_of(data)['error'] == 'cross_origin'
    status, response, data = served.request('POST', '/api/panel/run/prepare', body=b'x=1',
                                            headers={'Content-Type': 'application/x-www-form-urlencoded'})
    assert status == 415
    status, response, data = served.request('POST', '/api/panel/run/rm', body={})
    assert status == 404 and json_of(data)['error'] == 'unknown_action'
    assert served.app.panel.jobs() == []


def test_panel_needs_the_sign_in_like_the_rest_of_the_site(root, checkout):
    server = RunningServer(root, token=TOKEN, require_token=True)
    server.app.panel = Panel(root, repo=checkout)
    try:
        # Signed out, the pages show the sign-in page and the API answers 401.
        for path in ['/', '/prepare', '/train', '/evaluate']:
            status, response, data = server.request('GET', path)
            assert status in (200, 401) and asset('js/login.js').encode() in data, path
        assert server.request('GET', '/api/panel/status')[0] == 401
        assert server.request('POST', '/api/panel/run/prepare', body={})[0] == 401
        assert server.request('GET', '/api/panel/status', headers=sign_in(server))[0] == 200
    finally:
        server.close()


def test_a_training_started_from_a_terminal_has_the_progress_of_its_own_log(root, checkout):
    panel = Panel(root, repo=checkout)
    (root / 'logs').mkdir(exist_ok=True)
    (root / 'logs' / 'train-r1.log').write_text('== a. importo le revisioni confermate\n== b. controllo\n'
                                                '== c. ambiente GPU\n== d. training r1\n== 4/10 Train detector ==\n'
                                                'detector epoch 2/20 train_loss 0.9000 val_loss 1.0000 30s\n')
    payload = {'jobs': [], 'activities': [{'group': 'train', 'label': 'dataset e training', 'pid': 1}]}
    panel._terminal_training(payload)
    progress = payload['activities'][0]['progress']
    assert progress['detail'] == 'detector · epoch 2 of 20'
    assert progress['curves']['detector']['points'] == [{'epoch': 2, 'train_loss': 0.9, 'val_loss': 1.0, 'seconds': 30}]
