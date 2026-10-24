"""The pages of the site: one shared look (shared.css, the tokens of distill-web), the tabs, the Account
menu (appearance, settings, sign out), the Stato page, and the scripts built from the Vue and TypeScript of
web/src."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from nnslr_tools.pages import asset
from review_helpers import ROUTE, RunningServer, make_report

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def served(tmp_path):
    make_report(tmp_path)
    server = RunningServer(tmp_path)
    yield server
    server.close()


PAGES = [('/', 'Overview'), ('/prepare', 'Prepare'), ('/train', 'Train'), ('/evaluate', 'Evaluate'), ('/models', 'Models'),
         ('/routes', 'Review'),
         ('/account', None), (f'/route/{ROUTE}/', None), ('/nonexistent', None)]


@pytest.mark.parametrize('path, tab', PAGES)
def test_every_page_has_the_shared_look_the_account_menu_and_its_script(served, path, tab):
    status, response, data = served.request('GET', path)
    page = data.decode()
    nonce = response.getheader('Content-Security-Policy').split("'nonce-")[1].split("'")[0]
    # The theme is applied before the first paint; the page script carries the same nonce.
    assert f'<script nonce="{nonce}">try{{var t=localStorage.getItem(\'nnslr-theme\')' in page
    assert page.count(f'<script nonce="{nonce}">') == 2
    assert asset('shared.css') in page
    # The theme choice is in the Account menu, not a separate switch; the review page draws its own bar
    # (and the menu in it) with Vue.
    if not path.startswith('/route/'):
        assert 'id="account-menu"' in page and 'id="account-btn"' in page
    assert 'theme-toggle' not in page
    assert '__NNSLR_' not in page.replace('"server": false', '')
    tabs = re.search(r'<nav class="tabs".*?</nav>', page, re.S)
    if tab is not None:
        assert tabs and re.findall(r'>([^<]+)</a>', tabs.group(0)) == ['Overview', 'Prepare', 'Review', 'Train', 'Evaluate',
                                                                       'Models']
        assert f'aria-current="page">{tab}</a>' in tabs.group(0)


def test_control_page_reads_the_panel_status(served):
    page = served.request('GET', '/')[2].decode()
    script = asset('js/control.js')
    assert script in page and '/api/panel/status' in script and 'cores' in script and 'gpu-procs' in script
    # The classifier with "not a sign": its action on Train, its sections on Evaluate.
    assert 'train_classifier' in script and 'chain-eval' in script and 'classifier-eval' in script
    # The Train tab by role: the cutter (its sizes and signs), the classifier, the decider, the whole pipeline.
    for marker in ('train_cutter', 'train-roles', 'train-cutter', 'train-decider', 'train-pipeline', 'cutter-eval'):
        assert marker in script, marker


def test_built_scripts_are_the_build_of_the_typescript():
    """src/nnslr_tools/web/js is committed: it must be what helpers/build_web.sh makes of web/src (vue-tsc, Vite)."""
    if not (REPO / 'web/node_modules/vite').is_dir() or shutil.which('uvx') is None:
        pytest.skip('web/node_modules missing: run helpers/build_web.sh once')
    result = subprocess.run([str(REPO / 'helpers/build_web.sh'), '--check'], capture_output=True, text=True,
                            env=dict(os.environ), timeout=300)
    assert result.returncode == 0, result.stdout + result.stderr


def test_every_page_script_has_its_source():
    built = {path.stem for path in (REPO / 'src/nnslr_tools/web/js').glob('*.js')}
    sources = {path.stem for path in (REPO / 'web/src/pages').glob('*.ts')}
    assert built == sources
