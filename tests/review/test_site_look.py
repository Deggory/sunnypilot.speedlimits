"""The pages of the site: one shared look (shared.css, the tokens of distill-web), the tabs, the theme
switch, the Account icon, the Stato page, and the scripts built from the TypeScript of web/src."""
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


PAGES = [('/', 'Route'), ('/stato', 'Stato'), ('/pannello', 'Pannello'), ('/account', None),
         (f'/route/{ROUTE}/', None), ('/nonexistent', None)]


@pytest.mark.parametrize('path, tab', PAGES)
def test_every_page_has_the_shared_look_the_switch_and_its_script(served, path, tab):
    status, response, data = served.request('GET', path)
    page = data.decode()
    nonce = response.getheader('Content-Security-Policy').split("'nonce-")[1].split("'")[0]
    # The theme is applied before the first paint; the page script carries the same nonce.
    assert f'<script nonce="{nonce}">try{{var t=localStorage.getItem(\'nnslr-theme\')' in page
    assert page.count(f'<script nonce="{nonce}">') == 2
    assert asset('shared.css') in page
    assert 'id="theme-toggle"' in page and 'class="icon-btn account-btn" href="/account"' in page
    assert '__NNSLR_' not in page.replace('"server": false', '')
    tabs = re.search(r'<nav class="tabs".*?</nav>', page, re.S)
    if tab is not None:
        assert tabs and re.findall(r'>([^<]+)</a>', tabs.group(0)) == ['Route', 'Pannello', 'Stato']
        assert f'aria-current="page">{tab}</a>' in tabs.group(0)


def test_status_page_reads_the_panel_status(served):
    page = served.request('GET', '/stato')[2].decode()
    assert 'id="cpu-cores"' in page and 'id="gpu-procs"' in page and '"/api/panel/status"' in page


def test_built_scripts_are_the_build_of_the_typescript():
    """src/nnslr_tools/web/js is committed: it must be what helpers/build_web.sh makes of web/src."""
    if not (REPO / 'web/node_modules/esbuild').is_dir() or shutil.which('uvx') is None:
        pytest.skip('web/node_modules missing: run helpers/build_web.sh once')
    result = subprocess.run([str(REPO / 'helpers/build_web.sh'), '--check'], capture_output=True, text=True,
                            env=dict(os.environ), timeout=300)
    assert result.returncode == 0, result.stdout + result.stderr


def test_every_page_script_has_its_source():
    built = {path.stem for path in (REPO / 'src/nnslr_tools/web/js').glob('*.js')}
    sources = {path.stem for path in (REPO / 'web/src/pages').glob('*.ts')}
    assert built == sources
