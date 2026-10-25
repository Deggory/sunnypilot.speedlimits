"""Settings > Hugging Face (nnslr_tools.hf_account): the token publishing uses, checked on Hugging Face, kept on the
server only, never shown again, given to the publishing job in its environment only."""
from __future__ import annotations

import stat

import pytest

from nnslr_tools import hf_account
from nnslr_tools.panel import Panel
from review_helpers import TOKEN, RunningServer, json_of, make_report, sign_in

SECRET = 'hf_TestTokenThatIsNotRealAtAll0123456789'


@pytest.fixture(autouse=True)
def no_server_login(monkeypatch, tmp_path):
    monkeypatch.setattr(hf_account, 'SERVER_LOGIN', tmp_path / 'no-login' / 'token')


@pytest.fixture
def lan(tmp_path):
    make_report(tmp_path)
    server = RunningServer(tmp_path, token=TOKEN, require_token=True)
    yield server, tmp_path
    server.close()


def test_only_a_token_that_can_write_the_users_repositories_publishes():
    assert hf_account.can_write({'name': 'u', 'auth': {'accessToken': {'role': 'write'}}})
    assert not hf_account.can_write({'name': 'u', 'auth': {'accessToken': {'role': 'read'}}})
    grained = {'name': 'u', 'auth': {'accessToken': {'role': 'fineGrained', 'fineGrained': {'scoped': [
        {'entity': {'type': 'user', 'name': 'u'}, 'permissions': ['repo.content.read', 'repo.write']}]}}}}
    assert hf_account.can_write(grained)
    grained['auth']['accessToken']['fineGrained']['scoped'][0]['permissions'] = ['repo.content.read']
    assert not hf_account.can_write(grained)


def test_a_session_saves_the_token_and_never_sees_it_again(lan, monkeypatch):
    server, root = lan
    checked = []
    monkeypatch.setattr(hf_account, 'check_token', lambda token: checked.append(token) or
                        {'user': 'cristianku', 'role': 'write', 'can_write': True})
    session = sign_in(server)
    state = json_of(server.request('GET', '/api/settings/huggingface', headers=session)[2])
    assert state['connected'] is False and state['can_edit'] is True
    status, response, data = server.request('POST', '/api/settings/huggingface', headers=session, body={'token': SECRET})
    assert status == 200 and SECRET.encode() not in data
    assert json_of(data)['user'] == 'cristianku' and json_of(data)['source'] == 'portal' and checked == [SECRET]
    kept = hf_account.path(root)
    assert stat.S_IMODE(kept.stat().st_mode) == 0o600 and hf_account.token(root) == SECRET
    assert SECRET.encode() not in server.request('GET', '/api/settings/huggingface', headers=session)[2]
    status, response, data = server.request('POST', '/api/settings/huggingface/remove', headers=session, body={})
    assert status == 200 and json_of(data)['connected'] is False and not kept.exists()


def test_a_wrong_or_read_only_token_is_refused_without_repeating_it(lan, monkeypatch):
    server, root = lan
    session = sign_in(server)
    status, response, data = server.request('POST', '/api/settings/huggingface', headers=session, body={'token': 'abc'})
    assert status == 400 and json_of(data)['error'] == 'invalid_token'
    monkeypatch.setattr(hf_account, 'check_token', lambda token: {'user': 'u', 'role': 'read', 'can_write': False})
    status, response, data = server.request('POST', '/api/settings/huggingface', headers=session, body={'token': SECRET})
    assert status == 400 and json_of(data)['error'] == 'read_only_token' and SECRET.encode() not in data
    assert not hf_account.path(root).exists()


def test_the_access_token_reads_but_never_saves_one(lan, monkeypatch):
    server, root = lan
    monkeypatch.setattr(hf_account, 'check_token', lambda token: {'user': 'u', 'role': 'write', 'can_write': True})
    token = {'X-Review-Token': TOKEN}
    assert json_of(server.request('GET', '/api/settings/huggingface', headers=token)[2])['can_edit'] is False
    status, response, data = server.request('POST', '/api/settings/huggingface', headers=token, body={'token': SECRET})
    assert status == 403 and json_of(data)['error'] == 'session_required' and not hf_account.path(root).exists()


def test_the_publishing_job_gets_the_token_in_its_environment_only(tmp_path, monkeypatch):
    import json
    import re
    import time
    root, checkout = tmp_path / 'data', tmp_path / 'repo'
    (root / 'publish' / 'networks' / 'pkg-1').mkdir(parents=True)
    (root / 'publish' / 'networks' / 'pkg-1' / 'networks.json').write_text(json.dumps({'created': 'x', 'networks': {}}))
    script = checkout / 'scripts' / 'publish_networks.sh'
    script.parent.mkdir(parents=True)
    script.write_text('#!/usr/bin/env bash\necho "args: $*"\necho "token: ${HF_TOKEN:+given}"\n')
    script.chmod(0o755)
    hf_account.path(root).parent.mkdir(parents=True)
    hf_account.path(root).write_text(json.dumps({'token': SECRET, 'user': 'u', 'can_write': True}))
    panel = Panel(root, repo=checkout, actions_enabled=True)
    monkeypatch.setattr(panel, 'available', lambda: True)
    monkeypatch.setattr(panel, 'activities', lambda: [])
    job = panel.start('networks_publish', {'package': 'pkg-1'})
    log = panel.logs / (job['id'] + '.log')
    for _ in range(100):
        if 'finished with code' in log.read_text():
            break
        time.sleep(.05)
    text = log.read_text()
    assert 'token: given' in text and SECRET not in text and re.search(r'args: --upload --name=pkg-1$', text, re.M)
