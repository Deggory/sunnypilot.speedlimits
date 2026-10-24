"""The look shared with distill-web: the same tokens on every page, the tabs, the theme switch, the Stato page."""
from __future__ import annotations

import re
from importlib.resources import files

import pytest

from review_helpers import RunningServer, make_report

PAGES = ('site.css', 'review.html', 'panel.html', 'status.html')


def token_block(text: str) -> str:
    """The distill :root palette, from its first line to the light one under prefers-color-scheme."""
    start = text.index(':root{--bg:#0b0f16')
    end = text.index('--error-fg:#991b1b;--on-ok:#fff;--spark-gpu:#16a34a;--spark-cpu:#2563eb}}', start)
    return text[start:end]


def test_every_page_has_the_same_tokens():
    blocks = {name: token_block(files('nnslr_tools').joinpath(name).read_text()) for name in PAGES}
    assert len(set(blocks.values())) == 1, 'the token block differs between ' + ', '.join(PAGES)


@pytest.fixture
def served(tmp_path):
    make_report(tmp_path)
    server = RunningServer(tmp_path)
    yield server
    server.close()


@pytest.mark.parametrize('path, current', [('/', 'Route'), ('/stato', 'Stato'), ('/pannello', 'Pannello')])
def test_pages_have_tabs_theme_switch_and_account_icon(served, path, current):
    status, response, data = served.request('GET', path)
    assert status == 200
    page = data.decode()
    nonce = response.getheader('Content-Security-Policy').split("'nonce-")[1].split("'")[0]
    # The theme is read before the first paint, the switch works with the page's own nonce.
    assert f'<script nonce="{nonce}">try{{var t=localStorage.getItem(\'nnslr-theme\')' in page
    assert 'id="theme-toggle"' in page and '__NNSLR_' not in page
    tabs = re.search(r'<nav class="tabs".*?</nav>', page, re.S).group(0)
    assert re.findall(r'>([^<]+)</a>', tabs) == ['Route', 'Pannello', 'Stato']
    assert f'aria-current="page">{current}</a>' in tabs
    # Account is the icon at the top right, not a tab.
    assert 'class="icon-btn account-btn" href="/account"' in page


def test_status_page_reads_the_panel_status(served):
    page = served.request('GET', '/stato')[2].decode()
    assert "api('/api/panel/status')" in page and 'id="cpu-cores"' in page and 'id="gpu-procs"' in page
