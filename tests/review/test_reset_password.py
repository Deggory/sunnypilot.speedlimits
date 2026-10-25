"""helpers/reset_password.sh, for a forgotten password: the site stops, the administrator file is moved aside and the
site starts again, so the next visit asks for the setup code. A fake systemctl first in PATH stands for the service
and the data root is temporary: the live site is never touched."""
from __future__ import annotations

import json
import os
import stat
import subprocess
from pathlib import Path

from nnslr_tools.review_auth import AdminAuth

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / 'helpers' / 'reset_password.sh'
FAKE_SYSTEMCTL = """#!/usr/bin/env bash
echo "$*" >> "$SYSTEMCTL_LOG"
if [ "$1" = "list-unit-files" ] && [ "$SERVICE_INSTALLED" = "1" ]; then
  echo "speedlimits-web.service enabled enabled"
fi
exit 0
"""


def run(tmp_path: Path, *args: str, installed: bool = True) -> tuple[subprocess.CompletedProcess, list[str]]:
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir(exist_ok=True)
    fake = bin_dir / 'systemctl'
    fake.write_text(FAKE_SYSTEMCTL)
    fake.chmod(0o755)
    log = tmp_path / 'systemctl.log'
    log.unlink(missing_ok=True)
    env = {**os.environ, 'PATH': f'{bin_dir}:{os.environ["PATH"]}', 'NNSLR_DATA_ROOT': str(tmp_path / 'data'),
           'SYSTEMCTL_LOG': str(log), 'SERVICE_INSTALLED': '1' if installed else '0'}
    result = subprocess.run(['bash', str(SCRIPT), *args], env=env, capture_output=True, text=True, timeout=60)
    return result, log.read_text().splitlines() if log.exists() else []


def administrator(tmp_path: Path) -> Path:
    folder = tmp_path / 'data' / 'web' / 'auth'
    folder.mkdir(parents=True)
    admin = folder / 'admin.json'
    admin.write_text(json.dumps({'version': 1, 'user': 'cristian', 'password': {'scrypt': 'x'}, 'changed': 1}))
    admin.chmod(0o600)
    (folder / 'sessions.json').write_text('{}')
    return admin


def test_without_an_administrator_nothing_changes(tmp_path):
    result, calls = run(tmp_path)
    assert result.returncode == 0 and 'No administrator yet' in result.stdout
    assert calls == [] and not (tmp_path / 'data').exists()


def test_the_service_stops_the_administrator_moves_aside_and_the_service_starts(tmp_path):
    admin = administrator(tmp_path)
    content = admin.read_text()
    assert AdminAuth(admin.parent).configured()
    result, calls = run(tmp_path)
    assert result.returncode == 0, result.stderr
    assert calls == ['list-unit-files speedlimits-web.service', 'stop speedlimits-web', 'start speedlimits-web',
                     'is-active --quiet speedlimits-web']
    assert not admin.exists()
    kept = list(admin.parent.glob('admin.json.*.old'))
    assert len(kept) == 1 and kept[0].read_text() == content
    assert stat.S_IMODE(kept[0].stat().st_mode) == 0o600
    # The next visit asks for the setup code: no administrator any more.
    assert not AdminAuth(admin.parent).configured()
    assert 'asks for the setup code' in result.stdout and 'Service started again.' in result.stdout
    # The token is only named, never read or printed.
    assert 'NNSLR_REVIEW_TOKEN in the .env' in result.stdout


def test_without_a_running_site_the_file_still_moves(tmp_path):
    admin = administrator(tmp_path)
    result, calls = run(tmp_path, installed=False)
    assert result.returncode == 0, result.stderr
    assert calls == ['list-unit-files speedlimits-web.service']
    assert not admin.exists() and 'No site is running' in result.stdout


def test_help_and_unknown_options(tmp_path):
    result, calls = run(tmp_path, '--help')
    assert result.returncode == 0 and 'Forgotten password of the site' in result.stdout and calls == []
    result, calls = run(tmp_path, '--force')
    assert result.returncode == 1 and 'unknown option' in result.stderr and calls == []
