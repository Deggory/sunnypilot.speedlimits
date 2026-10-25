"""The LLM providers file and the HTTP client of the AI reviewer: keys stay on the server, no redirect is
followed, every failure is an LlmError with the key masked. Only the fake server of fake_llm is called."""
from __future__ import annotations

import json
import os
import socket
import stat

import pytest

from fake_llm import FakeLlm
from nnslr_tools import llm_providers
from nnslr_tools.llm_providers import ID_PATTERN, LlmError
from nnslr_tools.review_db import ReviewRejected

KEY = 'sk-test-secret-key-0123456789'


def add(root, name='strata', base_url='https://strata.example/v1', model='qwen3-vl-32b-instruct', **extra):
    return llm_providers.save_provider(root, {'name': name, 'base_url': base_url, 'model': model, **extra})


def mode(path):
    return stat.S_IMODE(os.stat(path).st_mode)


def test_saved_file_is_private_and_answers_never_hold_the_key(tmp_path):
    saved = add(tmp_path, key=KEY)
    path = llm_providers.providers_path(tmp_path)
    assert path == tmp_path / 'web' / 'private' / 'llm-providers.json'
    assert mode(path) == 0o600
    assert mode(path.parent) == 0o700
    assert json.loads(path.read_text())['providers'][0]['key'] == KEY
    assert saved['has_key'] is True and 'key' not in saved
    settings = llm_providers.public_settings(tmp_path)
    assert KEY not in json.dumps(settings) and KEY not in json.dumps(saved)
    assert settings['providers'][0]['reviewer_name'] == 'ai/strata/qwen3-vl-32b-instruct'
    assert llm_providers.provider(tmp_path, 'strata')['key'] == KEY


def test_a_loose_file_is_made_private_again_on_read(tmp_path):
    add(tmp_path)
    path = llm_providers.providers_path(tmp_path)
    os.chmod(path, 0o644)
    llm_providers.public_settings(tmp_path)
    assert mode(path) == 0o600


def test_key_absent_keeps_empty_removes_text_sets(tmp_path):
    add(tmp_path, key=KEY)
    kept = add(tmp_path, id='strata', name='Strata farm')
    assert kept['has_key'] is True and kept['name'] == 'Strata farm'
    assert llm_providers.provider(tmp_path, 'strata')['key'] == KEY
    changed = add(tmp_path, id='strata', key='sk-other-key')
    assert changed['has_key'] is True
    assert llm_providers.provider(tmp_path, 'strata')['key'] == 'sk-other-key'
    removed = add(tmp_path, id='strata', key='')
    assert removed['has_key'] is False
    assert 'key' not in llm_providers.provider(tmp_path, 'strata')
    with pytest.raises(ReviewRejected) as error:
        add(tmp_path, id='strata', key='has space')
    assert error.value.reason == 'invalid_provider' and 'key' in error.value.message
    assert add(tmp_path, name='no key')['has_key'] is False


@pytest.mark.parametrize('name, expected', [
    ('strata', 'strata'),
    ('My llama.cpp Server!', 'my-llama-cpp-server'),
    ('123 box', 'p-123-box'),
    ('!!!', 'provider'),
    ('Ünïcode name', 'n-code-name'),
    ('x' * 60, 'x' * 32),
])
def test_ids_are_slugs_of_the_name(tmp_path, name, expected):
    saved = add(tmp_path, name=name[:40])
    assert saved['id'] == expected[:32]
    assert ID_PATTERN.fullmatch(saved['id'])


def test_ids_never_collide_and_never_change(tmp_path):
    first = add(tmp_path, name='strata')
    second = add(tmp_path, name='Strata')
    third = add(tmp_path, name='STRATA')
    assert [first['id'], second['id'], third['id']] == ['strata', 'strata-2', 'strata-3']
    renamed = add(tmp_path, id='strata-2', name='something else')
    assert renamed['id'] == 'strata-2'
    long = [add(tmp_path, name='y' * 40)['id'] for _ in range(3)]
    assert long == ['y' * 32, 'y' * 30 + '-2', 'y' * 30 + '-3']


@pytest.mark.parametrize('url, expected', [
    ('https://strata.example/v1/', 'https://strata.example/v1'),
    ('http://10.0.0.5:8080/v1', 'http://10.0.0.5:8080/v1'),
    ('https://HOST.example/openai/v1', 'https://host.example/openai/v1'),
])
def test_base_url_is_normalized(tmp_path, url, expected):
    assert add(tmp_path, base_url=url)['base_url'] == expected


@pytest.mark.parametrize('url', [
    'ftp://host/v1', 'https://user:pw@host/v1', 'https://host/v1?x=1', 'https://host/v1#a', 'https://host/api',
    'https:///v1', 'https://host/' + 'a' * 200 + '/v1', 'https://host/v 1', 'https://host:99999/v1', '',
])
def test_bad_base_urls_are_refused(tmp_path, url):
    with pytest.raises(ReviewRejected) as error:
        add(tmp_path, base_url=url)
    assert error.value.status == 400 and error.value.reason == 'invalid_provider'
    assert 'base_url' in error.value.message


@pytest.mark.parametrize('body', [
    {'name': '', 'base_url': 'https://h/v1', 'model': 'm'},
    {'name': 'n' * 41, 'base_url': 'https://h/v1', 'model': 'm'},
    {'name': 'tab\there', 'base_url': 'https://h/v1', 'model': 'm'},
    {'name': 'n', 'base_url': 'https://h/v1', 'model': 'two words'},
    {'name': 'n', 'base_url': 'https://h/v1', 'model': ''},
    {'name': 'n', 'base_url': 'https://h/v1', 'model': 'm', 'key': 'k' * 513},
])
def test_bad_fields_are_refused(tmp_path, body):
    with pytest.raises(ReviewRejected) as error:
        llm_providers.save_provider(tmp_path, body)
    assert error.value.reason == 'invalid_provider'


def test_the_env_variable_names_the_file(tmp_path, monkeypatch):
    target = tmp_path / 'elsewhere' / 'providers.json'
    monkeypatch.setenv(llm_providers.PROVIDERS_ENV, str(target))
    add(tmp_path / 'root', key=KEY)
    assert target.is_file() and mode(target) == 0o600
    assert not (tmp_path / 'root' / 'web').exists()


def refused_reviewer(root, provider_id) -> str:
    with pytest.raises(ReviewRejected) as error:
        llm_providers.set_reviewer(root, provider_id)
    assert (error.value.status, error.value.reason) == (409, 'ai_not_tested')
    return error.value.message


def test_reviewer_choice_tests_and_removal(tmp_path):
    add(tmp_path)
    assert llm_providers.ai_state(tmp_path) == {'available': False, 'reason': 'Choose an AI reviewer in Settings.',
                                               'provider': None}
    # Only a provider that passed its last Test reviews: never tested, then failed, it is refused.
    assert refused_reviewer(tmp_path, 'strata') == ('Test strata in Settings first: only a provider that passed its '
                                                    'Test can review.')
    llm_providers.record_test(tmp_path, 'strata', {'ok': False, 'message': 'It cannot see the frames.'})
    assert refused_reviewer(tmp_path, 'strata') == ('strata did not pass its last Test. Fix it, then test it again '
                                                    'in Settings.')
    assert llm_providers.public_settings(tmp_path)['reviewer'] is None
    llm_providers.record_test(tmp_path, 'strata', {'ok': True, 'message': 'Sees images · 1.0 s', 'extra': 1})
    test = llm_providers.public_settings(tmp_path)['providers'][0]['last_test']
    assert test['ok'] is True and 'extra' not in test
    llm_providers.set_reviewer(tmp_path, 'strata')
    state = llm_providers.ai_state(tmp_path)
    assert state['available'] is True and state['reason'] == ''
    assert state['provider'] == {'id': 'strata', 'name': 'strata', 'model': 'qwen3-vl-32b-instruct',
                                 'reviewer_name': 'ai/strata/qwen3-vl-32b-instruct'}
    assert llm_providers.public_settings(tmp_path)['providers'][0]['is_reviewer'] is True
    add(tmp_path, id='strata', name='strata farm')
    assert llm_providers.public_settings(tmp_path)['providers'][0]['last_test'] is not None
    # A new model clears the Test: the reviewer stays chosen but cannot review until it passes again.
    add(tmp_path, id='strata', model='another-model')
    assert llm_providers.public_settings(tmp_path)['providers'][0]['last_test'] is None
    state = llm_providers.ai_state(tmp_path)
    assert state['available'] is False and state['provider']['id'] == 'strata'
    assert state['reason'] == 'Test strata in Settings first: only a provider that passed its Test can review.'
    with pytest.raises(ReviewRejected) as error:
        llm_providers.check_can_review(llm_providers.reviewer_provider(tmp_path))
    assert error.value.reason == 'ai_not_tested'
    # No reviewer is always possible.
    llm_providers.set_reviewer(tmp_path, None)
    assert llm_providers.public_settings(tmp_path)['reviewer'] is None
    with pytest.raises(ReviewRejected) as error:
        llm_providers.set_reviewer(tmp_path, 'nobody')
    assert error.value.status == 404 and error.value.reason == 'unknown_provider'
    llm_providers.delete_provider(tmp_path, 'strata')
    assert llm_providers.public_settings(tmp_path) == {'providers': [], 'reviewer': None}
    assert llm_providers.reviewer_provider(tmp_path) is None


def test_at_most_twenty_providers(tmp_path):
    for number in range(llm_providers.MAX_PROVIDERS):
        add(tmp_path, name=f'p{number}')
    with pytest.raises(ReviewRejected) as error:
        add(tmp_path, name='one more')
    assert error.value.reason == 'too_many_providers'


def test_reviewer_name_and_model_slug():
    record = {'id': 'strata', 'model': 'Qwen/Qwen3-VL-32B-Instruct  GGUF:Q4_K_M'}
    assert llm_providers.reviewer_name(record) == 'strata/Qwen-Qwen3-VL-32B-Instruct-GGUF:Q4_K_M'
    long = {'id': 'p' * 32, 'model': 'm' * 200}
    assert len(llm_providers.reviewer_name(long)) == 97
    assert len('ai/' + llm_providers.reviewer_name(long)) == 100
    assert llm_providers.model_slug('m' * 200) == 'm' * 80
    assert llm_providers.model_slug('///') == 'model'


# ---------- HTTP ----------

def record(base_url, key=KEY, model='fake-vl'):
    return {'id': 'fake', 'name': 'fake', 'base_url': base_url, 'model': model, **({'key': key} if key else {})}


def test_chat_sends_the_key_and_asks_for_json():
    with FakeLlm('{"boxes": []}') as fake:
        text, seconds = llm_providers.chat(record(fake.base_url), [{'role': 'user', 'content': 'hi'}], timeout=5)
    assert text == '{"boxes": []}' and seconds >= 0
    assert fake.auth_headers == ['Bearer ' + KEY]
    body = fake.requests[0]
    assert body['temperature'] == 0 and body['response_format'] == {'type': 'json_object'}
    assert body['model'] == 'fake-vl' and body['chat_template_kwargs'] == {'enable_thinking': False}


def test_chat_without_a_key_sends_no_authorization():
    with FakeLlm('{}') as fake:
        llm_providers.chat(record(fake.base_url, key=None), [], timeout=5, json_mode=False)
    assert fake.auth_headers == [None]
    assert 'response_format' not in fake.requests[0]


def test_a_server_refusing_json_mode_is_asked_again_without_it():
    with FakeLlm('{"ok": 1}', refuse_json_mode=True) as fake:
        text, _ = llm_providers.chat(record(fake.base_url), [], timeout=5)
    assert text == '{"ok": 1}'
    # Without chat_template_kwargs first (OpenAI refuses it), then thinking off again and without response_format.
    assert [('response_format' in body, 'chat_template_kwargs' in body) for body in fake.requests] == [
        (True, True), (True, False), (False, True)]


def test_a_server_that_does_not_know_chat_template_kwargs_is_asked_again_without_it():
    with FakeLlm('{"ok": 1}', refuse_template_kwargs=True) as fake:
        text, _ = llm_providers.chat(record(fake.base_url), [], timeout=5)
    assert text == '{"ok": 1}'
    assert [('response_format' in body, 'chat_template_kwargs' in body) for body in fake.requests] == [
        (True, True), (True, False)]
    # Refusing both: the last request has neither.
    with FakeLlm('{"ok": 2}', refuse_template_kwargs=True, refuse_json_mode=True) as fake:
        assert llm_providers.chat(record(fake.base_url), [], timeout=5)[0] == '{"ok": 2}'
    assert len(fake.requests) == 4 and set(fake.requests[-1]) == {'model', 'messages', 'temperature', 'max_tokens'}


def test_redirects_are_refused_and_the_key_never_follows():
    with FakeLlm() as target, FakeLlm(redirect=None) as source:
        source.redirect = target.base_url + '/chat/completions'
        with pytest.raises(LlmError) as error:
            llm_providers.chat(record(source.base_url), [], timeout=5)
        assert error.value.status == 502 and error.value.reason == 'ai_redirect'
        with pytest.raises(LlmError) as error:
            llm_providers.list_models(None, {'base_url': source.base_url, 'key': KEY})
        assert error.value.reason == 'ai_redirect'
    assert target.paths == []


def test_http_errors_mask_the_key():
    with FakeLlm(require_key='the-right-key', error_message='Incorrect API key provided: ' + KEY) as fake:
        with pytest.raises(LlmError) as error:
            llm_providers.chat(record(fake.base_url), [], timeout=5)
    assert error.value.status == 502 and error.value.reason == 'ai_http_error'
    assert error.value.message.startswith('HTTP 401: Incorrect API key provided: ***')
    assert KEY not in str(error.value) and KEY not in error.value.message
    with FakeLlm(status=500, error_message='broken ' + KEY[:12] + '\x07 here') as fake:
        with pytest.raises(LlmError) as error:
            llm_providers.chat(record(fake.base_url), [], timeout=5)
    assert error.value.message == 'HTTP 500: broken *** here'


def test_timeouts_and_unreachable_servers():
    with FakeLlm(delay=1.5) as fake:
        with pytest.raises(LlmError) as error:
            llm_providers.chat(record(fake.base_url), [], timeout=0.3)
    assert error.value.status == 504 and error.value.reason == 'ai_timeout'
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        port = probe.getsockname()[1]
    with pytest.raises(LlmError) as error:
        llm_providers.chat(record(f'http://127.0.0.1:{port}/v1'), [], timeout=2)
    assert error.value.status == 502 and error.value.reason == 'ai_unreachable'
    assert len(error.value.message) <= 200


def test_an_empty_answer_is_a_bad_answer():
    with FakeLlm('   ') as fake:
        with pytest.raises(LlmError) as error:
            llm_providers.chat(record(fake.base_url), [], timeout=5)
    assert error.value.reason == 'ai_bad_answer'


def test_models_say_whether_they_see_images(tmp_path):
    with FakeLlm(models=('vl-a', 'vl-b')) as fake:
        assert llm_providers.list_models(tmp_path, {'base_url': fake.base_url + '/'}) == [
            {'id': 'vl-a', 'sees_images': True}, {'id': 'vl-b', 'sees_images': True}]
    with FakeLlm(models=('text-only',), text_only=True) as fake:
        assert llm_providers.list_models(tmp_path, {'base_url': fake.base_url}) == [
            {'id': 'text-only', 'sees_images': False}]


def test_a_stored_key_goes_only_to_its_stored_address(tmp_path):
    with FakeLlm() as stored, FakeLlm() as typed:
        add(tmp_path, name='fake', base_url=stored.base_url, model='fake-vl', key=KEY)
        llm_providers.list_models(tmp_path, {'id': 'fake'})
        llm_providers.list_models(tmp_path, {'id': 'fake', 'base_url': stored.base_url})
        llm_providers.list_models(tmp_path, {'id': 'fake', 'base_url': typed.base_url})
        llm_providers.list_models(tmp_path, {'id': 'fake', 'base_url': typed.base_url, 'key': 'typed-key'})
    assert stored.auth_headers == ['Bearer ' + KEY, 'Bearer ' + KEY]
    assert typed.auth_headers == [None, 'Bearer typed-key']
    with pytest.raises(ReviewRejected) as error:
        llm_providers.list_models(tmp_path, {})
    assert error.value.reason == 'invalid_provider'
