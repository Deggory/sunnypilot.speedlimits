"""The networks saved on GitHub (#92): what a save copies of Fit box (#94), and whether GitHub is up to date, through a
save and a restore against a local bare repository (no GitHub: its API is not called)."""
from __future__ import annotations

import json
import os
import subprocess

import pytest

from nnslr_tools import networks_github as github
from nnslr_tools.box_fit import versions as box_versions

RUN = '2026-11-16_0900-box-fit'
GIT = ['-c', 'user.name=test', '-c', 'user.email=test@example.invalid']


@pytest.fixture
def code(tmp_path, monkeypatch):
    """The code's Fit box files: stand-ins, not the checkout's."""
    folder = tmp_path / 'code-box-fit'
    folder.mkdir()
    for name in github.BOX_FIT:
        (folder / name).write_bytes(b'code ' + name.encode())
    monkeypatch.setattr(github, 'BOX_FIT_DIR', folder)
    github._fresh.clear()
    return folder


@pytest.fixture
def root(tmp_path, code):
    root = tmp_path / 'data'
    folder = root / 'runs' / RUN
    folder.mkdir(parents=True)
    (folder / box_versions.MEASURE).mkdir()
    for name in (*box_versions.FILES, 'box-fit.json', 'rankers.json', 'refiner.json', 'refiner.log',
                 *(f'{box_versions.MEASURE}/{name}' for name in box_versions.FILES)):
        (folder / name).write_bytes(name.encode())
    (root / 'models').mkdir()
    (root / 'models' / 'box-fit.json').write_text(json.dumps({'id': 'box_fit:' + RUN}))
    return root


@pytest.fixture
def remote(tmp_path, monkeypatch):
    """A bare repository with one commit on main, as GitHub creates one (auto_init)."""
    bare = tmp_path / 'remote.git'
    subprocess.run(['git', 'init', '-q', '--bare', '-b', github.BRANCH, str(bare)], check=True)
    work = tmp_path / 'init'
    subprocess.run(['git', 'clone', '-q', str(bare), str(work)], check=True, capture_output=True)
    (work / 'README.md').write_text('# networks\n')
    subprocess.run(['git', '-C', str(work), *GIT, 'add', '-A'], check=True)
    subprocess.run(['git', '-C', str(work), *GIT, 'commit', '-qm', 'init'], check=True)
    subprocess.run(['git', '-C', str(work), 'push', '-q', 'origin', 'HEAD:' + github.BRANCH], check=True)
    monkeypatch.setenv(github.REMOTE_ENV, str(bare))
    monkeypatch.setenv(github.REPO_ENV, 'example/networks')
    monkeypatch.setattr(github, 'ensure_repo', lambda repo, **_: {'repo': repo, 'url': 'https://github.com/' + repo,
                                                                   'created': False, 'checked': 'token'})
    return bare


def test_a_save_copies_every_version_of_fit_box_and_the_choice(root, code):
    planned = github.plan(root)
    assert planned['versions']['box_fit:' + RUN] == {'role': 'box_fit', 'files': [
        f'data/runs/{RUN}/{name}' for name in github.BOX_FIT_RUN_FILES]}
    assert planned['versions'][box_versions.CODE_ID]['files'] == [
        f'code/{github.BOX_FIT_PATH}{name}' for name in github.BOX_FIT]
    assert planned['files']['data/models/box-fit.json'] == root / 'models' / 'box-fit.json'
    # A training still running and its log are not saved.
    (root / 'runs' / f'.{RUN}.partial').mkdir()
    assert not any('refiner.log' in path or '.partial' in path for path in github.plan(root)['files'])


def test_before_the_first_save_nothing_is_on_github(root):
    fresh = github.freshness(root)
    assert fresh['up_to_date'] is None and fresh['commit'] is None
    assert set(fresh['versions'].values()) == {'missing'}
    assert fresh['records'] == {'models/box-fit.json': 'missing'}
    assert fresh['unsaved_count'] == 3 and 'box_fit:' + RUN in fresh['unsaved']


def test_after_a_save_github_is_up_to_date_until_a_file_changes(root, code, remote, tmp_path):
    clone = tmp_path / 'clone'
    saved = github.save(root, clone=clone, log=lambda line: None)
    assert saved['pushed'] and saved['files'] == 13
    record = json.loads((root / 'publish' / 'github' / 'saved-files.json').read_text())
    assert record['commit'] == saved['commit'] and f'data/runs/{RUN}/colour.npz' in record['files']
    fresh = github.freshness(root)
    assert fresh['up_to_date'] is True and fresh['unsaved'] == [] and fresh['commit'] == saved['commit']
    assert set(fresh['versions'].values()) == {'saved'}
    # Only touched (another time, the same bytes): still saved.
    later = os.stat(root / 'runs' / RUN / 'edges.npz').st_mtime + 60
    os.utime(root / 'runs' / RUN / 'edges.npz', (later, later))
    assert github.freshness(root, now=10 ** 8)['up_to_date'] is True
    # Kept FRESH_SECONDS: a change shows at the next read after it.
    (root / 'runs' / RUN / 'edges.npz').write_bytes(b'other bytes')
    assert github.freshness(root, now=10 ** 8 + 1)['up_to_date'] is True
    fresh = github.freshness(root, now=10 ** 9)
    assert fresh['up_to_date'] is False and fresh['versions']['box_fit:' + RUN] == 'changed'
    assert fresh['versions'][box_versions.CODE_ID] == 'saved'
    # A new version: not on GitHub.
    newer = root / 'runs' / '2026-11-17_0900-box-fit'
    newer.mkdir()
    for name in box_versions.RANKERS:
        (newer / name).write_bytes(b'new')
    fresh = github.freshness(root, now=2 * 10 ** 9)
    assert fresh['versions']['box_fit:2026-11-17_0900-box-fit'] == 'missing'
    assert fresh['unsaved'] == ['box_fit:2026-11-17_0900-box-fit', 'box_fit:' + RUN] and fresh['unsaved_count'] == 2
    # The next save takes both: the new version's two files and the changed one.
    again = github.save(root, clone=clone, log=lambda line: None)
    assert (again['pushed'], again['new'], again['updated']) == (True, 2, 1)
    assert github.freshness(root)['up_to_date'] is True
    # A save with nothing new pushes nothing and GitHub is still up to date.
    assert github.save(root, clone=clone, log=lambda line: None)['pushed'] is False
    assert github.freshness(root)['up_to_date'] is True


def test_a_restore_puts_every_version_back_and_says_github_has_them(root, code, remote, tmp_path):
    github.save(root, clone=tmp_path / 'clone', log=lambda line: None)
    fresh_root = tmp_path / 'fresh'
    done = github.restore(fresh_root, clone=tmp_path / 'clone-2', log=lambda line: None)
    assert done['copied'] == 10 and done['box_fit'] == {f'{github.BOX_FIT_PATH}{name}': 'same' for name in github.BOX_FIT}
    for name in github.BOX_FIT_RUN_FILES:
        assert (fresh_root / 'runs' / RUN / name).read_bytes() == (root / 'runs' / RUN / name).read_bytes()
        assert (fresh_root / 'runs' / RUN / name).stat().st_mtime == pytest.approx(
            (root / 'runs' / RUN / name).stat().st_mtime, abs=1e-3)
    assert box_versions.in_use(fresh_root)['id'] == 'box_fit:' + RUN
    fresh = github.freshness(fresh_root)
    assert fresh['up_to_date'] is True, fresh
    # No save from here: the panel compares with the save restored, and says which.
    status = github.status(fresh_root)
    assert status['saved'] is None and status['freshness']['commit'] and status['freshness']['saved_at']
    # A fresh clone of the code: its files have other times, the same bytes.
    for name in github.BOX_FIT:
        os.utime(code / name, (1, 1))
    assert github.freshness(fresh_root, now=10 ** 9)['versions'][box_versions.CODE_ID] == 'saved'
    (code / 'edges.npz').write_bytes(b'code edges.npX')
    assert github.freshness(fresh_root, now=2 * 10 ** 9)['versions'][box_versions.CODE_ID] == 'changed'
    assert github.restore(fresh_root, clone=tmp_path / 'clone-2', log=lambda line: None)['box_fit'][
        github.BOX_FIT_PATH + 'edges.npz'] == 'differs'


def test_the_panel_status_says_it_and_never_fails_on_it(root, monkeypatch):
    assert github.status(root)['freshness']['up_to_date'] is None

    def broken(_root, **_):
        raise ValueError('cannot read')
    monkeypatch.setattr(github, 'freshness', broken)
    assert github.status(root)['freshness'] is None


def test_a_save_or_a_choice_of_another_process_shows_at_once(root, code, remote, tmp_path):
    assert github.freshness(root, now=10 ** 8)['up_to_date'] is None
    # The save runs in a script: its clearing of the cache does not reach the server's.
    github.save(root, clone=tmp_path / 'clone', log=lambda line: None)
    github._fresh[str(root)] = (10 ** 8, github._fresh_stamp(root), {'up_to_date': None})
    (root / 'publish' / 'github' / 'saved-files.json').write_text(
        (root / 'publish' / 'github' / 'saved-files.json').read_text() + ' ')
    assert github.freshness(root, now=10 ** 8 + 1)['up_to_date'] is True
    # Another Fit box version chosen in the panel: the choice record has changed, so does the answer.
    (root / 'models' / 'box-fit.json').write_text(json.dumps({'id': box_versions.CODE_ID, 'who': 'x'}))
    assert github.freshness(root, now=10 ** 8 + 2)['records']['models/box-fit.json'] == 'changed'


def test_a_freshness_that_fails_is_said_once_and_kept(root, monkeypatch, capsys):
    calls = []

    def broken(_root):
        calls.append(1)
        raise OSError('cannot read')
    monkeypatch.setattr(github, 'plan', broken)
    for now in (10 ** 8, 10 ** 8 + 1):
        with pytest.raises(OSError):
            github.freshness(root, now=now)
    assert len(calls) == 1 and capsys.readouterr().out.count('cannot be told') == 1
    assert github.status(root)['freshness'] is None


@pytest.mark.parametrize('remote, allowed', [
    ('git@github.com:example/networks.git', True), ('https://github.com/Example/Networks', True),
    ('/srv/backup/networks.git', True), ('git@github.com:example/public-copy.git', False),
    ('git@github.com:example/networks/extra', False)])
def test_a_github_remote_must_be_the_repository_checked_to_be_private(monkeypatch, remote, allowed):
    monkeypatch.setenv(github.REMOTE_ENV, remote)
    if allowed:
        assert github.remote_of('example/networks') == remote
    else:
        with pytest.raises(ValueError, match='example/networks'):
            github.remote_of('example/networks')
