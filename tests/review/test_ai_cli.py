"""nnslr llm-provider, nnslr ai-review and nnslr measure-ai-reviewer: the key is read from a file and never
printed; progress goes to stderr and the result as JSON to stdout. Only fake_llm's server is called."""
from __future__ import annotations

import json
import os

import pytest

from fake_llm import FakeLlm, answer_json
from nnslr_tools import cli, llm_providers
from nnslr_tools.review_db import ReviewStore, db_path, save_frame_review
from review_helpers import ROUTE, RUN_ID, make_report, page_body

KEY = 'sk-cli-secret-key-987654321'


def invoke(capsys, *args):
    capsys.readouterr()
    try:
        code = cli.main(list(args))
    except SystemExit as exc:
        code = exc.code
    return code, capsys.readouterr()


@pytest.fixture
def root(tmp_path):
    make_report(tmp_path, keys=[(0, 0), (0, 1), (0, 2)])
    return tmp_path


def key_file(tmp_path, mode=0o600):
    path = tmp_path / 'key'
    path.write_text(KEY + '\n')
    os.chmod(path, mode)
    return path


def test_add_list_use_models_remove(root, capsys):
    data = str(root)
    with FakeLlm(models=('vl-first', 'vl-second'), require_key=KEY) as fake:
        code, out = invoke(capsys, 'llm-provider', 'add', '--name', 'Strata', '--base-url', fake.base_url + '/',
                           '--key-file', str(key_file(root, 0o644)), '--reviewer', '--data-root', data)
        assert code == 0
        assert 'warning: other users can read the key file' in out.err
        assert 'model: vl-first (the first the server lists)' in out.err
        assert out.out.strip() == f'added  strata  Strata  {fake.base_url}  vl-first  key: set  [reviewer]'
        assert llm_providers.provider(root, 'strata')['key'] == KEY

        # The same name updates it; without --key-file the key stays, and the stored key lists the models.
        code, out = invoke(capsys, 'llm-provider', 'add', '--name', 'strata', '--base-url', fake.base_url,
                           '--data-root', data)
        assert code == 0 and out.out.startswith('updated  strata  strata  ')
        assert llm_providers.provider(root, 'strata')['key'] == KEY

        code, out = invoke(capsys, 'llm-provider', 'models', 'strata', '--data-root', data)
        assert out.out.splitlines() == ['vl-first  sees images: yes', 'vl-second  sees images: yes']

        code, out = invoke(capsys, 'llm-provider', 'add', '--name', 'local', '--base-url', fake.base_url,
                           '--model', 'vl-second', '--data-root', data)
        assert out.out.strip().endswith('vl-second  key: none')

        code, out = invoke(capsys, 'llm-provider', 'list', '--data-root', data)
        assert out.out.splitlines() == [f'strata  strata  {fake.base_url}  vl-first  key: set  [reviewer]',
                                        f'local  local  {fake.base_url}  vl-second  key: none']
        code, out = invoke(capsys, 'llm-provider', 'use', 'local', '--data-root', data)
        assert out.out.strip() == 'AI reviewer: local'
        code, out = invoke(capsys, 'llm-provider', 'use', 'none', '--data-root', data)
        assert out.out.strip() == 'AI reviewer: none'
        assert llm_providers.public_settings(root)['reviewer'] is None
        code, out = invoke(capsys, 'llm-provider', 'remove', 'local', '--data-root', data)
        assert code == 0 and out.out.strip() == 'removed local'
        code, out = invoke(capsys, 'llm-provider', 'remove', 'local', '--data-root', data)
        assert code == 2 and 'unknown_provider' in out.err
    # /models of the first add (the key file), of the update and of "models" (the stored key, same address).
    assert fake.auth_headers == ['Bearer ' + KEY] * 3


def test_the_key_is_never_printed(root, capsys):
    data = str(root)
    with FakeLlm(require_key='another-key', error_message='bad key ' + KEY) as fake:
        code, out = invoke(capsys, 'llm-provider', 'add', '--name', 'strata', '--base-url', fake.base_url,
                           '--key-file', str(key_file(root)), '--data-root', data)
        assert code == 2 and 'ai_http_error: HTTP 401: bad key ***' in out.err
        code, out = invoke(capsys, 'llm-provider', 'add', '--name', 'strata', '--base-url', fake.base_url,
                           '--model', 'm', '--key-file', str(key_file(root)), '--data-root', data)
        assert code == 0 and 'warning' not in out.err
        printed = out.out + out.err
        for args in (('list',), ('models', 'strata'), ('test', 'strata')):
            code, out = invoke(capsys, 'llm-provider', *args, '--data-root', data)
            printed = printed + out.out + out.err
    assert KEY not in printed
    assert 'bad key ***' in printed


def test_missing_and_empty_key_files(root, capsys):
    code, out = invoke(capsys, 'llm-provider', 'add', '--name', 'x', '--base-url', 'https://h.example/v1',
                       '--model', 'm', '--key-file', str(root / 'nothing'), '--data-root', str(root))
    assert code == 2 and 'cannot read the key file' in out.err
    (root / 'empty').write_text('\n')
    code, out = invoke(capsys, 'llm-provider', 'add', '--name', 'x', '--base-url', 'https://h.example/v1',
                       '--model', 'm', '--key-file', str(root / 'empty'), '--data-root', str(root))
    assert code == 2 and 'is empty' in out.err
    assert llm_providers.public_settings(root)['providers'] == []


def test_ai_review_needs_a_reviewer(root, capsys):
    code, out = invoke(capsys, 'ai-review', ROUTE, '--data-root', str(root))
    assert code == 2 and out.err.strip() == 'error: no AI reviewer: nnslr llm-provider use ID'
    code, out = invoke(capsys, 'measure-ai-reviewer', '--provider', 'nobody', '--data-root', str(root))
    assert code == 2 and 'unknown_provider' in out.err


def test_ai_review_and_measure_commands(root, capsys):
    data = str(root)
    store = ReviewStore(db_path(root))
    rows = {}
    for line in (root / 'derived/preannotations' / ROUTE / RUN_ID / 'preannotations.jsonl').read_text().splitlines():
        row = json.loads(line)
        rows[row['frame_key']] = row
    body = page_body(rows['0/0'], value_kph=30)
    save_frame_review(store, root, route_id=ROUTE, run_id=RUN_ID, report_row=rows['0/0'], body=body, client='test')
    answer = answer_json(boxes=[{'n': 1, 'verdict': 'keep', 'value_kph': 30}])

    def reply(chat):
        text = chat['messages'][1]['content'][0]['text']
        return answer if '\n1. [' in text else answer_json(no_sign=True)
    with FakeLlm(reply) as fake:
        invoke(capsys, 'llm-provider', 'add', '--name', 'fake', '--base-url', fake.base_url, '--model', 'fake-vl',
               '--reviewer', '--data-root', data)
        code, out = invoke(capsys, 'ai-review', ROUTE, '--data-root', data)
        assert code == 0
        summary = json.loads(out.out)
        assert (summary['frames'], summary['suggested'], summary['saved'], summary['save']) == (2, 2, 0, False)
        assert out.err.splitlines()[0].startswith('[1/2] 0/1: no sign · ')
        assert 'AI review done: 2 suggested, 0 skipped, 0 failed.' in out.err
        assert len(store.reviews(ROUTE, RUN_ID)) == 1

        code, out = invoke(capsys, 'ai-review', ROUTE, '--save', '--limit', '1', '--provider', 'fake',
                           '--data-root', data)
        assert code == 0 and json.loads(out.out)['saved'] == 1
        saved = store.review(ROUTE, RUN_ID, '0/1')
        assert saved['reviewer'] == 'ai/fake/fake-vl' and saved['channel'] == 'web'
        assert saved['reviewer_kind'] == 'ai'

        code, out = invoke(capsys, 'ai-review', ROUTE, '--limit', '0', '--data-root', data)
        assert code == 2

        code, out = invoke(capsys, 'measure-ai-reviewer', '--limit', '5', '--data-root', data)
        assert code == 0
        report = json.loads(out.out)
        assert report['task'] == 'ai_reviewer' and report['frames'] == 1
        assert report['speed_signs']['read_right'] == 1
        assert 'AI reviewer measured: 1 of 1 frames, 0 failed.' in out.err
