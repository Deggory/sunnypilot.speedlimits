"""The AI reviewer in the pages built from web/src: Settings (the providers, the key never shown), the Account menu
item, Ask AI on the review page with its own overlay colour, "Review with AI" on the route list, and the AI reviewers
of Evaluate. The scripts are checked as built (src/nnslr_tools/web/js); test_site_look checks they are the build of
the sources."""
from __future__ import annotations

import re

import pytest

from nnslr_tools.pages import asset
from review_helpers import ROUTE, RunningServer, make_report

PAGE_SCRIPTS = ('account', 'control', 'review', 'site')


@pytest.mark.parametrize('page', PAGE_SCRIPTS)
def test_the_account_menu_of_every_page_has_one_settings_item(page):
    # One item, Settings (/account): the password, the signed-in browsers and the AI reviewer are on that page.
    script = asset(f'js/{page}.js')
    assert 'account-settings' in script and 'Settings' in script
    assert 'Password and signed-in browsers' not in script


def test_settings_lists_the_providers_and_never_reads_a_key_back():
    script = asset('js/account.js')
    for path in ('/api/settings/ai', '/api/settings/ai/providers', '/api/settings/ai/reviewer',
                 '/api/settings/ai/models', '/api/settings/ai/test', '/delete'):
        assert path in script, path
    assert '`ai-reviewer`' in script or '"ai-reviewer"' in script
    # Only "a key is set" comes back: the placeholder says so, "Remove key" sends "" on Save.
    assert 'Saved (leave empty to keep)' in script
    assert 'Remove key' in script and 'has_key' in script
    assert 'Use to review' in script and 'Not tested yet' in script
    # The local site hides the Administrator and password cards.
    assert '.local' in script


def test_the_review_page_asks_the_ai_and_applies_without_saving():
    script = asset('js/review.js')
    assert '/ai-suggest' in script and '/api/ai/state' in script
    assert 'AI suggestion applied: check each box, then Confirm.' in script
    for verdict in ('Keep', 'Correct', 'Add', 'Delete', 'Not judged'):
        assert verdict in script, verdict
    assert 'AI suggestion · added' in script
    # Key I, in the legend too.
    assert 'ask AI' in script
    page = asset('review.html')
    assert '#aiLayer { pointer-events: none; }' in page
    assert '.ai-box' in page and 'var(--ov-ai)' in page


def _tokens(css: str, selector: str) -> dict[str, str]:
    tokens: dict[str, str] = {}
    for block in re.findall(re.escape(selector) + r'\{([^}]*)\}', css):
        tokens.update(dict(re.findall(r'(--[a-z0-9-]+):([^;]+)', block)))
    return tokens


def test_the_ai_overlay_colour_differs_from_the_other_marks():
    css = asset('shared.css')
    dark = _tokens(css, ':root')
    light = {**dark, **_tokens(css, ':root[data-theme="light"]')}
    for palette in (dark, light):
        ai = palette['--ov-ai'].strip().lower()
        for other in ('--ov-hint', '--ov-selected', '--warn', '--sus', '--ok'):
            assert palette[other].strip().lower() != ai, other


def test_the_route_list_opens_review_with_ai_and_saving_is_asked_first():
    script = asset('js/site.js')
    assert 'button[data-ai-route]' in script
    assert '/api/panel/run/ai_review' in script and '/api/ai/state' in script
    assert 'Try without saving' in script and 'Review and save' in script
    assert 'AI saves are final: nobody approves them again' in script
    assert 'Started: follow it on the' in script


def test_a_sheet_and_its_backdrop_are_not_styled_as_the_open_button_of_a_route():
    # .open is the shown state of a sheet: the route list's "Start" pill rule must not reach it.
    css = asset('site.css')
    assert not re.search(r'(^|[},])\s*\.open\s*\{', css, re.M)
    assert '.card-row .open{' in css


def test_evaluate_has_the_ai_reviewers():
    script = asset('js/control.js')
    assert '/api/panel/ai-reviewers' in script
    assert 'AI reviewers' in script and 'measure_ai_reviewer' in script and 'ai_review' in script


@pytest.fixture
def served(tmp_path):
    make_report(tmp_path)
    server = RunningServer(tmp_path)
    yield server
    server.close()


def test_the_pages_that_hold_the_ai_parts_carry_their_scripts(served):
    status, _response, data = served.request('GET', '/account')
    assert status == 200
    assert asset('js/account.js')[:200] in data.decode()
    status, _response, data = served.request('GET', '/routes')
    assert status == 200
    assert asset('js/site.js')[:200] in data.decode()
    status, _response, data = served.request('GET', f'/route/{ROUTE}/')
    assert status == 200
    assert asset('js/review.js')[:200] in data.decode()
