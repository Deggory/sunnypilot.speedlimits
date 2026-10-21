"""Control panel: route state, the fixed actions run in the background with their log, HTTP protections.

The helpers are fakes in a temporary checkout (echo, sleep, exit codes): nothing real is prepared or
trained.
"""
from __future__ import annotations

import shutil
import time
from pathlib import Path

import pytest

from nnslr_tools.panel import Panel, PanelError
from review_helpers import ROUTE, TOKEN, RunningServer, json_of, make_report

REPO = Path(__file__).resolve().parents[2]
FAKE_PREPARE = '#!/usr/bin/env bash\necho "prepare $*"\necho "data root $NNSLR_DATA_ROOT"\nsleep "${FAKE_SLEEP:-0}"\n'
FAKE_TRAIN = '#!/usr/bin/env bash\necho "train $*"\nexit 3\n'
FAKE_UPDATE = '#!/usr/bin/env bash\necho "update $*"\nsleep "${FAKE_UPDATE_SLEEP:-0}"\n'


@pytest.fixture
def checkout(tmp_path):
    repo = tmp_path / 'repo'
    (repo / 'helpers').mkdir(parents=True)
    shutil.copy(REPO / 'helpers/route_status.py', repo / 'helpers/route_status.py')
    for name, text in (('1_prepare.sh', FAKE_PREPARE), ('3_train.sh', FAKE_TRAIN), ('update.sh', FAKE_UPDATE)):
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
    assert status['actions']['prepare'] == {'label': 'Prepara le route nuove', 'allowed': False,
                                            'reason': 'nessuna route nuova da preparare'}
    assert status['actions']['dataset']['reason'] == 'nessun fotogramma confermato'
    assert status['actions']['train']['allowed'] is False
    assert set(status['system']) == {'gpu', 'disk', 'cpus', 'load'}


def test_action_runs_in_background_and_keeps_its_exit_code(root, checkout):
    panel = Panel(root, repo=checkout)
    job = panel.start('prepare')
    assert job['action'] == 'prepare' and job['label'] == 'Prepara le route nuove'
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
    assert any(item['label'] == 'preparazione' for item in panel.activities())
    assert panel.status()['actions']['prepare']['reason'] == "c'è già una preparazione in corso"
    assert panel.stop(job['id']) == {'id': job['id'], 'stopping': True}
    assert finished(panel, job['id'])['state'] in ('interrupted', 'failed')
    with pytest.raises(PanelError) as unknown:
        panel.stop('20260101-000000-prepare')
    assert unknown.value.status == 404


def test_update_runs_without_printing_the_token_link(root, checkout):
    panel = Panel(root, repo=checkout)
    assert panel.status()['actions']['update'] == {'label': 'Aggiorna il codice', 'allowed': True, 'reason': ''}
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
    assert panel.status()['actions']['update']['reason'] == "c'è già una preparazione in corso"
    panel.stop(job['id'])
    finished(panel, job['id'])

    monkeypatch.setenv('FAKE_UPDATE_SLEEP', '30')
    update = panel.start('update')
    for action in ('prepare', 'dataset'):
        with pytest.raises(PanelError) as busy:
            panel.start(action)
        assert busy.value.status == 409
    assert panel.status()['actions']['prepare']['reason'] == "c'è già un aggiornamento del codice in corso"
    panel.stop(update['id'])
    finished(panel, update['id'])


def test_update_needs_its_helper(root, checkout):
    (checkout / 'helpers/update.sh').unlink()
    panel = Panel(root, repo=checkout)
    assert panel.status()['actions']['update']['reason'] == 'manca helpers/update.sh in questa installazione'
    with pytest.raises(PanelError) as missing:
        panel.start('update')
    assert missing.value.status == 403


def test_read_only_panel_and_unknown_actions(root, checkout):
    panel = Panel(root, repo=checkout, actions_enabled=False)
    with pytest.raises(PanelError) as disabled:
        panel.start('prepare')
    assert disabled.value.status == 403
    assert panel.status()['actions']['train']['reason'] == 'pannello in sola lettura'
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
    status, response, data = served.request('GET', '/pannello')
    assert status == 200
    csp = response.getheader('Content-Security-Policy')
    assert "script-src 'nonce-" in csp and "connect-src 'self'" in csp
    nonce = csp.split("'nonce-")[1].split("'")[0]
    assert f'<script nonce="{nonce}">' in data.decode()
    assert '/pannello' in served.request('GET', '/')[2].decode()
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


def test_panel_needs_the_token_like_the_rest_of_the_site(root, checkout):
    server = RunningServer(root, token=TOKEN, require_token=True)
    server.app.panel = Panel(root, repo=checkout)
    try:
        for path in ['/pannello', '/api/panel/status']:
            assert server.request('GET', path)[0] == 403, path
        assert server.request('POST', '/api/panel/run/prepare', body={})[0] == 403
        cookie = {'Cookie': f'{server.app.cookie_name}={TOKEN}'}
        assert server.request('GET', '/api/panel/status', headers=cookie)[0] == 200
    finally:
        server.close()
