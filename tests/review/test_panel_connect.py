"""Download from comma connect in the control panel (#115): the link checked and passed as <dongle>|<route>, never as
typed; refused on other actions; never beside the comma's download or a preparation; its progress from the log.

The helper is a fake in a temporary checkout: nothing is downloaded.
"""
from __future__ import annotations

import shutil
import time
from pathlib import Path

import pytest

import nnslr_tools.panel as panel_module
from nnslr_tools.job_progress import progress
from nnslr_tools.panel import Panel, PanelError
from review_helpers import make_report

REPO = Path(__file__).resolve().parents[2]
FAKE = '#!/usr/bin/env bash\necho "$0 $*"\n'
SLOW = '#!/usr/bin/env bash\nsleep 5\n'
DONGLE = '35edd31af7e86e74'
DRIVE = '00000004--3188701194'


@pytest.fixture(autouse=True)
def only_this_checkout(monkeypatch):
    monkeypatch.setattr(panel_module, 'OWN_GROUPS', ('fetch', 'connect', 'prepare', 'train', 'update', 'ai', 'other'))


@pytest.fixture
def checkout(tmp_path):
    repo = tmp_path / 'repo'
    (repo / 'helpers').mkdir(parents=True)
    shutil.copy(REPO / 'helpers/route_status.py', repo / 'helpers/route_status.py')
    for name in ('1_prepare.sh', '3_train.sh', 'fetch_comma_segments.sh', 'fetch_connect_route.sh'):
        (repo / 'helpers' / name).write_text(FAKE)
        (repo / 'helpers' / name).chmod(0o755)
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


def test_the_link_reaches_the_script_as_dongle_and_route(root, checkout):
    panel = Panel(root, repo=checkout)
    job = panel.start('fetch_connect', {'link': f'  https://connect.comma.ai/{DONGLE}/{DRIVE}/12/40 '})
    finished(panel, job['id'])
    assert f'helpers/fetch_connect_route.sh {DONGLE}|{DRIVE}\n' in (panel.logs / (job['id'] + '.log')).read_text()


@pytest.mark.parametrize('link', [None, '', 'hello', f'{DONGLE}/2024-01-01--10-00-00', f'{DONGLE}|{DRIVE}; rm -rf /'])
def test_a_wrong_link_is_refused_before_anything_starts(root, checkout, link):
    panel = Panel(root, repo=checkout)
    options = {} if link is None else {'link': link}
    if link and link.endswith('rm -rf /'):
        # The drive is found inside it, and only the drive goes on: nothing else of the text reaches the command.
        job = panel.start('fetch_connect', options)
        finished(panel, job['id'])
        assert 'rm' not in (panel.logs / (job['id'] + '.log')).read_text().splitlines()[-1]
        return
    with pytest.raises(PanelError) as refused:
        panel.start('fetch_connect', options)
    assert refused.value.reason == 'bad_option'
    assert panel.jobs() == []


def test_only_the_connect_download_takes_a_link(root, checkout):
    panel = Panel(root, repo=checkout)
    with pytest.raises(PanelError) as refused:
        panel.start('fetch', {'link': f'{DONGLE}|{DRIVE}'})
    assert refused.value.reason == 'bad_option'


def test_never_beside_the_comma_download_or_a_preparation(root, checkout):
    (checkout / 'helpers/fetch_connect_route.sh').write_text(SLOW)
    panel = Panel(root, repo=checkout)
    job = panel.start('fetch_connect', {'link': f'{DONGLE}|{DRIVE}'})
    try:
        actions = panel.status()['actions']
        assert not actions['fetch']['allowed'] and 'comma connect' in actions['fetch']['reason']
        assert not actions['prepare']['allowed']
        assert not actions['fetch_connect']['allowed']
        # The comma pill shows only the comma's own download.
        assert panel._download_state()['downloading'] is False
    finally:
        panel.stop(job['id'])


def test_its_progress_is_the_segment_it_is_at():
    log = (f'{DRIVE} of {DONGLE} (VOLKSWAGEN_GOLF_MK7): 10 segments in connect\n'
           'segment 0 (1/10): fcamera.hevc\nsegment 0 (1/10): ecamera.hevc\nsegment 3 (4/10): rlog.bz2\n')
    result = progress('fetch_connect', 'running', log)
    assert result['percent'] == 35.0
    assert result['detail'] == 'segment 4 of 10 · rlog.bz2'
    assert [step['key'] for step in result['steps']] == ['fetch']
