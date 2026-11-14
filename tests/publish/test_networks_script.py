"""scripts/publish_networks.sh with stand-ins for nnslr and uvx: nothing is exported and nothing leaves the machine."""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / 'scripts' / 'publish_networks.sh'

NNSLR = """#!/usr/bin/env bash
echo "nnslr $*" >> "$CALLS"
if [ "$1" = networks-package ]; then
  name=""
  while [ $# -gt 0 ]; do [ "$1" = --name ] && name="$2"; shift; done
  mkdir -p "$NNSLR_DATA_ROOT/publish/networks/$name"
  echo "== 1/4 Export the cutter (c)"
  echo "== 4/4 Write the manifest and the model card, and check the package"
fi
"""
UVX = """#!/usr/bin/env bash
echo "uvx" >> "$CALLS"
cat > /dev/null
echo "Uploaded 9 files"
"""


def run(tmp_path, *arguments):
    venv, tools, data = tmp_path / 'venv' / 'bin', tmp_path / 'tools', tmp_path / 'data'
    for folder in (venv, tools, data):
        folder.mkdir(parents=True, exist_ok=True)
    for path, body in ((venv / 'nnslr', NNSLR), (tools / 'uvx', UVX)):
        path.write_text(body)
        path.chmod(0o755)
    environment = dict(os.environ, NNSLR_GPU_VENV=str(venv.parent), NNSLR_DATA_ROOT=str(data), CALLS=str(tmp_path / 'calls'),
                       PATH=f'{tools}:{os.environ["PATH"]}')
    done = subprocess.run(['bash', str(SCRIPT), *arguments], capture_output=True, text=True, env=environment, timeout=60)
    calls = (tmp_path / 'calls').read_text().splitlines() if (tmp_path / 'calls').exists() else []
    return done, calls, data


def test_one_call_prepares_a_new_package_and_publishes_it(tmp_path):
    """Models › Publish's one button (--publish): the package of the networks in use, then its upload, as four steps."""
    done, calls, data = run(tmp_path, '--upload', '--publish', '--name=speedlimits-20261115-0900')
    assert done.returncode == 0, done.stderr
    steps = re.findall(r'^== (\d)/4 ', done.stdout, re.M)
    assert steps == ['1', '2', '3', '4']  # the export's own four are plain lines, not steps of this job
    assert '== 1/4 Prepare the package speedlimits-20261115-0900 of the networks in use' in done.stdout
    assert '   1 of 4: Export the cutter (c)' in done.stdout
    assert done.stdout.rstrip().endswith('tree/main/networks/speedlimits-20261115-0900')
    assert [call.split()[1] for call in calls if call.startswith('nnslr')] == [
        'networks-package', 'verify-networks', 'networks-catalog', 'networks-published']
    assert calls.count('uvx') == 2 and (data / 'publish' / 'networks' / 'speedlimits-20261115-0900').is_dir()


def test_a_package_prepared_before_is_published_as_it_is(tmp_path):
    (tmp_path / 'data' / 'publish' / 'networks' / 'pkg-1').mkdir(parents=True)
    done, calls, _data = run(tmp_path, '--upload', '--name=pkg-1')
    assert done.returncode == 0, done.stderr
    assert '== 1/4 The package pkg-1, prepared before' in done.stdout and '== 4/4 Upload to' in done.stdout
    assert not [call for call in calls if 'networks-package' in call]
    # One that is not here is refused before anything is asked of Hugging Face.
    done, calls, _data = run(tmp_path, '--upload', '--name=gone')
    assert done.returncode != 0 and 'no package gone' in done.stderr


def test_without_a_name_the_one_call_names_the_package_by_the_time(tmp_path):
    done, _calls, data = run(tmp_path, '--publish')
    assert done.returncode == 0, done.stderr
    (made,) = [path.name for path in (data / 'publish' / 'networks').iterdir()]
    assert re.fullmatch(r'speedlimits-\d{8}-\d{4}', made)
