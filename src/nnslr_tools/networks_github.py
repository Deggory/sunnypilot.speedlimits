"""Our networks saved on GitHub (#92, 15 Nov 2026): a copy of every network we use, in a private repository of its own
(cristianku/smartspeed-networks), so that a lost data root can start again from networks that work, without Hugging
Face. Models › Publish › GitHub: **Save on GitHub** (scripts/save_networks_github.sh) and **Restore from GitHub**
(scripts/restore_networks_github.sh).

What a save copies, each version once (it stays in the repository afterwards; git keeps every change of a file):
- the networks in use (models/active.json: cutter, classifier, applicability, brain), those on the car
  (models/on-car.json) and the sign networks each saved brain read its signs with (its result.json's sign_networks),
  each with the files the Models tab reads to show it and a training's Start from reads: its PyTorch weights and its
  result.json, the classifier's test reports, every variant of an applicability run, a brain's routes.json and its
  run's networks.json;
- the folder of the networks each saved brain read its signs with (derived/road_type_signs/sets/<key>/), which cannot
  be made again for every brain;
- the road labeler Speed review uses (the run prepared last, else the newest trained);
- the splits the versions name (splits/<sha256>.json) and the brain's own (splits/speed-brain-*.json);
- the records: models/active.json, models/on-car.json, models/train-settings.json and models/box-fit.json (the
  version of Fit box in use);
- every version of Fit box (#94, nnslr_tools.box_fit.versions): each training's run folder (runs/<run>-box-fit/: its
  colour.npz, edges.npz, refiner.onnx, box-fit.json and its summaries, and its measure/ folder), put back by a restore
  as the other runs; and the code's files (src/nnslr_tools/box_fit/, the version box_fit:code), saved only: a restore leaves the checkout as
  it is (they come back with the code repository; the saved ones can be copied by hand from the clone).

Not saved: the dataset and the reviews (the database is in the PBS backup of the LXC), the ONNX exports (Publish makes
them again), the training states, plans, samples and logs.

The repository mirrors the data root under data/ and this checkout under code/, so that a restore is a copy, and its
manifest.json lists every file saved (size, sha256, time, mode) and every version with its files. A restore first
checks every file of the clone against the manifest, then puts back every file missing here with its time (the Models
tab orders the versions by the time of their weights) and its mode; it never overwrites a file of the data root: one
that differs is kept and listed. The records come back only when missing: after a lost data root the networks in use
are those of the last save. Any data root of any machine will do (a fresh clone of the code repository too): the
Models tab, Publish and the regression find a brain's sign networks by their key under the data root, and a brain's
networks.json, which names their folder by its full path, is written again when its run starts again.

The repository is private (the cutter and the classifier learnt from Mapillary's pictures, CC BY-NC-SA 4.0, and the
files name our drives): each save checks it on GitHub's API first and creates it, private, when it is missing; it never
pushes to a public one. The token is read from its file for the API only, never printed nor written; git reads and
pushes over SSH. The step lines (== N/M) are this module's: the scripts only start it.

Whether GitHub is up to date (:func:`freshness`, in the panel's status): after each push the save writes the size and
time of every file it saved (<data root>/publish/github/saved-files.json); the panel compares the files a save would
copy now with them (by size and time, else size and sha256), so each version and record is saved as it is here,
changed since, or not on GitHub. Files read only, no network, and at most every FRESH_SECONDS.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Callable

REPO_ENV, REPO = 'NNSLR_NETWORKS_REPO', 'cristianku/smartspeed-networks'
# Beside the code checkout (/root/smartspeed-networks here), outside it and the data root.
CLONE_ENV, CLONE = 'NNSLR_NETWORKS_CLONE', str(Path(__file__).resolve().parents[3] / 'smartspeed-networks')
# git@github.com:<repo>.git unless given (the tests give a local bare repository).
REMOTE_ENV = 'NNSLR_NETWORKS_REMOTE'
TOKEN_ENV, TOKEN_FILE = 'NNSLR_GITHUB_TOKEN_FILE', '/root/.secrets/github-token'
API_ENV, API = 'NNSLR_GITHUB_API', 'https://api.github.com'
BRANCH = 'main'
MANIFEST = 'manifest.json'
README = 'README.md'
DATA, CODE = 'data', 'code'
RECORDS = ('models/active.json', 'models/on-car.json', 'models/train-settings.json', 'models/box-fit.json')
BRAIN_SPLITS = ('splits/speed-brain-split.json', 'splits/speed-brain-frames.json')
BOX_FIT_DIR = Path(__file__).resolve().parent / 'box_fit'
BOX_FIT = ('colour.npz', 'edges.npz', 'refiner.onnx')
BOX_FIT_PATH = 'src/nnslr_tools/box_fit/'
# The files of a version of Fit box in its run folder (box_fit.versions.FILES and RECORD, the script's summaries).
# A Fit box version's files, with the rankers of its measure (versions.measure_folder) and the copy of its refiner.
BOX_FIT_RUN_FILES = ('colour.npz', 'edges.npz', 'refiner.onnx', 'box-fit.json', 'rankers.json', 'refiner.json',
                     'measure/colour.npz', 'measure/edges.npz', 'measure/refiner.onnx')
SETS_RELATIVE = 'derived/road_type_signs/sets'
# The road labeler's files in its run (road_labeler.CHECKPOINT, RESULT, PREPARED).
LABELER_FILES = ('road-labeler.pt', 'result.json')
LABELER_PREPARED = 'prepared.json'
RECORD_DIR = Path('publish') / 'github'
SAVED, RESTORED, SAVED_FILES = 'saved.json', 'restored.json', 'saved-files.json'
# The panel's status reads freshness() again at most this often.
FRESH_SECONDS = 30
# GitHub refuses a file over 100 MB.
MAX_FILE = 95 * 1024 * 1024
GIT_TIMEOUT = 900
API_TIMEOUT = 30
REPO_NAME = re.compile(r'[A-Za-z0-9][A-Za-z0-9-]{0,38}/[A-Za-z0-9._-]{1,100}')
SET_KEY = re.compile(r'[0-9a-f]{16}')
SHA256 = re.compile(r'[0-9a-f]{64}')
BRAIN_SEED = re.compile(r'(.+)-s[0-9]+')
# The commits of the clone: no identity is configured on this server.
IDENTITY = {'GIT_AUTHOR_NAME': 'smartspeed networks', 'GIT_AUTHOR_EMAIL': 'cristianku@users.noreply.github.com',
            'GIT_COMMITTER_NAME': 'smartspeed networks', 'GIT_COMMITTER_EMAIL': 'cristianku@users.noreply.github.com'}
LICENCE = ('Private: the cutter and the classifier learnt from Mapillary\'s pictures (CC BY-NC-SA 4.0, non-commercial '
           'use only), and the files name our drives.')


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec='seconds')


def repository() -> str:
    repo = os.environ.get(REPO_ENV) or REPO
    if not REPO_NAME.fullmatch(repo):
        raise ValueError(f'not a GitHub repository name: {repo}')
    return repo


# A remote on GitHub, by ssh or https: its owner/name.
GITHUB_REMOTE = re.compile(r'(?:git@github\.com:|ssh://git@github\.com/|https://github\.com/)([^/]+/[^/]+?)(?:\.git)?/?',
                           re.IGNORECASE)


def remote_of(repo: str) -> str:
    """Where the clone pushes: REMOTE_ENV (a local path in the tests, or another server), else GitHub's ``repo``. A
    GitHub remote must be ``repo`` itself: the repository whose privacy ensure_repo checked, never another one."""
    remote = os.environ.get(REMOTE_ENV)
    if not remote:
        return f'git@github.com:{repo}.git'
    named = GITHUB_REMOTE.fullmatch(remote.strip())
    if named is not None and named[1].lower() != repo.lower():
        raise ValueError(f'{REMOTE_ENV} is {named[1]} on GitHub, not {repo}, the repository checked to be private: '
                         'set both to the same repository')
    if named is None and 'github.com' in remote.lower():
        raise ValueError(f'{REMOTE_ENV} names GitHub in a way that cannot be checked against {repo}: use '
                         f'git@github.com:{repo}.git')
    return remote


def clone_path() -> Path:
    return Path(os.environ.get(CLONE_ENV) or CLONE)


def token_path() -> Path:
    return Path(os.environ.get(TOKEN_ENV) or TOKEN_FILE)


# -- what is saved ---------------------------------------------------------------------------------------------------
def _real_file(root: Path, relative: str) -> Path | None:
    """<root>/<relative> when it is a real file inside the root (no link on the way), else None."""
    path = root / relative
    try:
        if path.is_symlink() or not path.is_file():
            return None
        if not path.resolve().is_relative_to(root.resolve()):
            return None
    except OSError:
        return None
    return path


def _json(path: Path | None) -> Any:
    if path is None:
        return None
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


def _version_files(root: Path, model_id: str) -> dict | None:
    """The files of a version (relative to the data root): {'role', 'run', 'required', 'optional', 'needs' (the ids it
    depends on: a brain's sign networks), 'siblings' (the other variants of an applicability run), 'sets' (the folders
    of a brain's sign networks)}; None for an id that is not one."""
    from nnslr_tools.model_roles import EXPERIMENT_DIR, MODEL_ID, SIGNNET_PART
    from nnslr_tools.runs import VARIANTS

    match = MODEL_ID.fullmatch(model_id) if isinstance(model_id, str) else None
    if match is None:
        return None
    role, run, part = match.groups()
    base = f'runs/{run}/'
    found = {'role': role, 'run': run, 'required': [], 'optional': [], 'needs': [], 'siblings': [], 'sets': []}
    if role == 'cutter' and part == SIGNNET_PART:
        found['required'] = [base + SIGNNET_PART + '-best.pt', base + 'result.json']
        found['optional'] = [base + 'subset.json']
    elif role == 'cutter':
        evaluation = 'evaluation' if part == 'detector' else part + '/evaluation'
        found['required'] = [f'{base}{part}/detector-best.pt', f'{base}{part}/result.json']
        found['optional'] = [f'{base}{evaluation}/detector-test.json', f'{base}{evaluation}/detector-validation.json']
    elif role == 'classifier' and part == 'reader':
        found['required'] = [base + 'reader/reader-best.pt', base + 'reader/result.json']
        found['optional'] = [base + 'evaluation/reader-test.json']
    elif role == 'classifier':
        found['required'] = [base + 'classifier/reader-best.pt', base + 'classifier/result.json']
        found['optional'] = [base + 'evaluation/' + name for name in
                             ('classifier-test.json', 'chain-test.json', 'chain-test-old-reader.json')]
        # The split of an older classifier is only in its plan (ModelIndex._plan_split): the plan then, else not.
        result = _json(_real_file(root, base + 'classifier/result.json'))
        if not (isinstance(result, dict) and result.get('split_sha256')):
            found['optional'].append(base + 'classifier/training-plan.json')
    elif role == 'decider':
        variant = part.rsplit('/', 1)[-1]
        found['required'] = [f'{base}{part}/{variant}-best.pt', f'{base}{part}/result.json']
        folder = root / 'runs' / run / EXPERIMENT_DIR
        try:
            names = sorted({*VARIANTS, *(entry.name for entry in folder.iterdir() if entry.is_dir()
                                         and not entry.is_symlink())}) if folder.is_dir() else []
        except OSError:
            names = []
        found['siblings'] = [f'decider:{run}/{EXPERIMENT_DIR}/{name}' for name in names if name != variant]
    elif role == 'brain':
        found['required'] = [base + 'brain-last.pt', base + 'result.json']
        found['optional'] = [base + 'routes.json']
        parent = BRAIN_SEED.fullmatch(run)
        if parent:
            found['optional'] += [f'runs/{parent.group(1)}/networks.json', f'runs/{parent.group(1)}/routes.json']
        result = _json(_real_file(root, base + 'result.json'))
        recorded = result.get('sign_networks') if isinstance(result, dict) else None
        if isinstance(recorded, dict):
            from nnslr_tools import sign_networks
            ids = {key: value for key, value in recorded.items() if key in (*sign_networks.PARTS, sign_networks.OLD_PART)
                   and isinstance(value, str) and value}
            found['needs'] = list(ids.values())
            if all(ids.get(part_name) for part_name in sign_networks.PARTS):
                found['sets'] = [sign_networks.set_key(ids)]
    else:
        return None
    return found


def _files_under(root: Path, relative: str) -> list[str]:
    """Every real file under <root>/<relative> (relative to the root), links left out."""
    folder = root / relative
    if folder.is_symlink() or not folder.is_dir():
        return []
    files = []
    for top, folders, names in os.walk(folder, followlinks=False):
        folders[:] = sorted(name for name in folders if not (Path(top) / name).is_symlink())
        for name in sorted(names):
            path = Path(top) / name
            if not path.is_symlink() and path.is_file():
                files.append(path.relative_to(root).as_posix())
    return files


def road_labeler_run(root: Path) -> str | None:
    """The road labeler Speed review uses: the run prepared last, else the newest trained run; None before one."""
    from nnslr_tools import labeler_runs

    runs = labeler_runs.trained_runs(root)
    prepared = []
    for run in runs:
        found = labeler_runs._prepared(root / 'runs' / run)
        if found:
            prepared.append((datetime.fromisoformat(found['at']).timestamp(), run))
    if prepared:
        return max(prepared)[1]
    return runs[-1] if runs else None


def plan(root: Path) -> dict:
    """What a save copies: {'versions': {id: {'role', 'files'}}, 'files': {path in the repository: source}, 'in_use':
    {role: id}, 'on_car': {role: id}, 'road_labeler': run or None, 'warnings': [...]}. A version whose files are not
    all here is left out with a warning; the rest is saved."""
    from nnslr_tools.model_roles import ModelIndex
    from nnslr_tools.runs import RunsIndex

    root = Path(root)
    index = ModelIndex(RunsIndex(root))
    warnings: list[str] = []
    active, problem = index.read_active()
    if problem:
        warnings.append(problem)
    on_car, car_problem = index.read_on_car()
    if car_problem:
        warnings.append(car_problem)
    in_use = {role: entry['id'] for role, entry in active.items() if entry and entry.get('id')}
    car = {role: model_id for role, model_id in ((on_car or {}).get('ids') or {}).items() if model_id}
    versions: dict[str, dict] = {}
    files: dict[str, Path] = {}
    left: set[str] = set()
    queue = [(model_id, True) for model_id in (*in_use.values(), *car.values())]
    while queue:
        model_id, asked = queue.pop(0)
        if model_id in versions or model_id in left:
            continue
        found = _version_files(root, model_id)
        if found is None:
            left.add(model_id)
            warnings.append(f'{model_id} is not a version id: left out')
            continue
        missing = [relative for relative in found['required'] if _real_file(root, relative) is None]
        if missing:
            left.add(model_id)
            if asked:
                warnings.append(f'{model_id}: {", ".join(missing)} not on this server: left out')
            continue
        chosen = [*found['required'], *(relative for relative in found['optional'] if _real_file(root, relative))]
        for key in found['sets']:
            folder = f'{SETS_RELATIVE}/{key}'
            inside = _files_under(root, folder)
            if inside:
                chosen += inside
            else:
                warnings.append(f'{model_id}: the folder of its sign networks ({folder}) is not on this server')
        result = _json(_real_file(root, next(relative for relative in found['required'] if relative.endswith('.json'))))
        split = result.get('split_sha256') if isinstance(result, dict) else None
        if isinstance(split, str) and SHA256.fullmatch(split) and _real_file(root, f'splits/{split}.json'):
            chosen.append(f'splits/{split}.json')
        versions[model_id] = {'role': found['role'], 'files': [f'{DATA}/{relative}' for relative in chosen]}
        for relative in chosen:
            files[f'{DATA}/{relative}'] = root / relative
        queue += [(needed, True) for needed in found['needs']]
        queue += [(sibling, False) for sibling in found['siblings']]
    labeler = road_labeler_run(root)
    if labeler is not None:
        relatives = [f'runs/{labeler}/{name}' for name in (*LABELER_FILES, LABELER_PREPARED)]
        chosen = [relative for relative in relatives if _real_file(root, relative)]
        versions[f'road_labeler:{labeler}'] = {'role': 'road_labeler', 'files': [f'{DATA}/{relative}' for relative in chosen]}
        for relative in chosen:
            files[f'{DATA}/{relative}'] = root / relative
    if any(version['role'] == 'brain' for version in versions.values()):
        for relative in BRAIN_SPLITS:
            if _real_file(root, relative):
                files[f'{DATA}/{relative}'] = root / relative
    for relative in RECORDS:
        if _real_file(root, relative):
            files[f'{DATA}/{relative}'] = root / relative
    # Fit box: every version on this server (#94), each training's run folder, and the code's files.
    from nnslr_tools.box_fit import versions as box_versions
    for version in box_versions.versions(root):
        if version['run'] is None:
            continue
        chosen = [f'runs/{version["run"]}/{name}' for name in BOX_FIT_RUN_FILES
                  if _real_file(root, f'runs/{version["run"]}/{name}')]
        versions[version['id']] = {'role': 'box_fit', 'files': [f'{DATA}/{relative}' for relative in chosen]}
        for relative in chosen:
            files[f'{DATA}/{relative}'] = root / relative
    box_fit = [name for name in BOX_FIT if (BOX_FIT_DIR / name).is_file() and not (BOX_FIT_DIR / name).is_symlink()]
    if box_fit:
        versions[box_versions.CODE_ID] = {'role': 'box_fit',
                                          'files': [f'{CODE}/{BOX_FIT_PATH}{name}' for name in box_fit]}
        for name in box_fit:
            files[f'{CODE}/{BOX_FIT_PATH}{name}'] = BOX_FIT_DIR / name
    return {'versions': versions, 'files': files, 'in_use': in_use, 'on_car': car, 'road_labeler': labeler,
            'warnings': warnings}


def plan_summary(planned: dict) -> dict:
    """plan() as JSON: the versions, the files with their sizes, the warnings."""
    sizes = {path: source.stat().st_size for path, source in planned['files'].items()}
    return {'versions': {model_id: version['role'] for model_id, version in planned['versions'].items()},
            'files': len(sizes), 'bytes': sum(sizes.values()), 'largest': max(sizes.items(), key=lambda item: item[1],
                                                                                default=None),
            'in_use': planned['in_use'], 'on_car': planned['on_car'], 'road_labeler': planned['road_labeler'],
            'warnings': planned['warnings']}


# -- files ---------------------------------------------------------------------------------------------------------
def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def _copy(source: Path, target: Path, *, mode: int | None = None, mtime: float | None = None) -> str:
    """source copied to target through a temporary file beside it (renamed into place), its sha256 that of the bytes
    written; the mode and the time set when given."""
    target.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    handle, temporary = tempfile.mkstemp(prefix='.' + target.name + '.', dir=target.parent)
    try:
        with os.fdopen(handle, 'wb') as out, source.open('rb') as read:
            for block in iter(lambda: read.read(1 << 20), b''):
                digest.update(block)
                out.write(block)
            out.flush()
            os.fsync(out.fileno())
        if mode is not None:
            os.chmod(temporary, mode)
        if mtime is not None:
            os.utime(temporary, (mtime, mtime))
        os.replace(temporary, target)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return digest.hexdigest()


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    try:
        with os.fdopen(handle, 'w', encoding='utf-8') as out:
            out.write(text)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _write_json(path: Path, value: Any) -> None:
    _write_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def safe_path(path: Any) -> bool:
    """A path of the repository the manifest may name: relative, under data/ or code/, no '..', no empty part."""
    if not isinstance(path, str) or not path or '\\' in path or '\0' in path:
        return False
    parts = PurePosixPath(path).parts
    return (len(parts) >= 2 and parts[0] in (DATA, CODE) and not PurePosixPath(path).is_absolute()
            and all(part not in ('', '.', '..') for part in parts) and PurePosixPath(*parts).as_posix() == path)


def read_manifest(clone: Path) -> dict | None:
    value = _json(_real_file(Path(clone), MANIFEST))
    if not isinstance(value, dict) or value.get('schema_version') != 1:
        return None
    if not isinstance(value.get('files'), dict) or not isinstance(value.get('versions'), dict):
        return None
    return value


# -- save ----------------------------------------------------------------------------------------------------------
def copy_networks(root: Path, clone: Path, repo: str, *, log: Callable[[str], None] = print) -> dict:
    """The networks of plan() copied into the clone (only the files new or changed), the manifest merged with the
    saved ones (versions and files accumulate: nothing saved before is dropped, but a file gone from the clone), the
    README written. {'new', 'changed', 'same', 'files', 'bytes', 'versions', 'in_use', 'on_car', 'road_labeler',
    'warnings', 'message' (the commit's)}."""
    root, clone = Path(root), Path(clone)
    planned = plan(root)
    old = read_manifest(clone)
    if old is None and _real_file(clone, MANIFEST) is not None:
        raise ValueError(f'{clone}/{MANIFEST} cannot be read: fix it in the repository first')
    files: dict[str, dict] = dict((old or {}).get('files') or {})
    versions: dict[str, dict] = dict((old or {}).get('versions') or {})
    warnings = list(planned['warnings'])
    for path in list(files):
        entry = files[path]
        target = _real_file(clone, path) if safe_path(path) else None
        if target is None or not isinstance(entry, dict) or target.stat().st_size != entry.get('size'):
            warnings.append(f'{path} was saved but is not in the repository any more: dropped from the manifest')
            del files[path]
    at = _now()
    counts = {'new': 0, 'changed': 0, 'same': 0}
    for path, source in sorted(planned['files'].items()):
        info = source.stat()
        if info.st_size > MAX_FILE:
            raise ValueError(f'{source} is {info.st_size // (1024 * 1024)} MB: GitHub takes files up to 100 MB')
        mode = stat.S_IMODE(info.st_mode)
        before = files.get(path)
        target = clone / path
        if before is not None and before.get('size') == info.st_size and before.get('sha256') == _sha256(source) \
                and _real_file(clone, path) is not None:
            if before.get('mtime') != info.st_mtime or before.get('mode') != mode:
                files[path] = {**before, 'mtime': info.st_mtime, 'mode': mode}
            counts['same'] += 1
            continue
        if target.is_symlink():
            raise ValueError(f'{target} is a link: the clone must hold the files themselves')
        digest = _copy(source, target)
        files[path] = {'size': target.stat().st_size, 'sha256': digest, 'mtime': info.st_mtime, 'mode': mode,
                       'saved_at': at}
        counts['changed' if before is not None else 'new'] += 1
    for model_id, version in planned['versions'].items():
        earlier = versions.get(model_id) if isinstance(versions.get(model_id), dict) else {}
        versions[model_id] = {'role': version['role'], 'files': version['files'],
                              'saved_at': earlier.get('saved_at') or at, 'last_saved_at': at}
    try:
        commit = subprocess.run(['git', '-C', str(BOX_FIT_DIR), 'rev-parse', 'HEAD'], capture_output=True, text=True,
                                timeout=30, check=False).stdout.strip() or None
    except (OSError, subprocess.TimeoutExpired):
        commit = None
    manifest = {'schema_version': 1, 'repository': repo, 'saved_at': at, 'code_commit': commit,
                'in_use': planned['in_use'], 'on_car': planned['on_car'], 'road_labeler': planned['road_labeler'],
                'versions': dict(sorted(versions.items())), 'files': dict(sorted(files.items()))}
    # Nothing new (the same files, versions and records): the manifest and the README stay as they are, so that the
    # save makes no commit.
    if old is None or _stable(old) != _stable(manifest):
        _write_json(clone / MANIFEST, manifest)
        _write_text(clone / README, readme(manifest))
    total = sum(entry['size'] for entry in files.values())
    log(f'{len(planned["versions"])} versions now: {counts["new"]} files new, {counts["changed"]} changed, '
        f'{counts["same"]} already saved; the repository holds {len(files)} files, {total / 1e6:.1f} MB, '
        f'{len(versions)} versions')
    for warning in warnings:
        log('warning: ' + warning)
    words = lambda ids: ', '.join(f'{role} {model_id.split(":", 1)[1]}' for role, model_id in ids.items()) or 'none'  # noqa: E731
    message = (f'Networks of {at[:16].replace("T", " ")}: {counts["new"]} files new, {counts["changed"]} changed\n\n'
               f'In use: {words(planned["in_use"])}\nOn the car: {words(planned["on_car"])}\n'
               + (f'Road labeler: {planned["road_labeler"]}\n' if planned['road_labeler'] else '')
               + '\nSaved from the review site (Models › Publish › GitHub: scripts/save_networks_github.sh).\n')
    return {**counts, 'files': len(files), 'bytes': total, 'versions': len(versions),
            'in_use': planned['in_use'], 'on_car': planned['on_car'], 'road_labeler': planned['road_labeler'],
            'warnings': warnings, 'message': message}


def _stable(manifest: dict) -> dict:
    """A manifest without the times of the save, to tell whether a save changes anything."""
    versions = {model_id: {key: value for key, value in version.items() if key != 'last_saved_at'}
                for model_id, version in (manifest.get('versions') or {}).items() if isinstance(version, dict)}
    return {**{key: value for key, value in manifest.items() if key not in ('saved_at', 'code_commit')},
            'versions': versions}


def readme(manifest: dict) -> str:
    """The README.md of the repository: what it is, how to restore, what the last save had in use."""
    rows = '\n'.join(f'| `{model_id}` | {version.get("role")} | {str(version.get("saved_at") or "")[:16].replace("T", " ")} |'
                     for model_id, version in manifest['versions'].items())
    ids = lambda found: '\n'.join(f'- {role}: `{model_id}`' for role, model_id in found.items()) or '- none'  # noqa: E731
    return f"""# Smartspeed networks

A copy of every network the speed-limit work uses (cristianku/sunnypilot.smartspeed, issue #92), saved from the review
site: Models › Publish › GitHub › **Save on GitHub**. {LICENCE}

- `data/` mirrors the data root (`/root/smartspeed/data`): the runs of the networks (PyTorch weights, result.json and
  the files the Models tab reads), the folders of the brains' sign networks, the splits and the records of what is in
  use (`models/active.json`, `models/box-fit.json`) and on the car (`models/on-car.json`); every version of Fit box
  (`runs/<run>-box-fit/`).
- `code/` mirrors the code checkout: the code's Fit box (`src/nnslr_tools/box_fit/`), saved only.
- `manifest.json`: every file with its size, sha256, time and mode, and every version with its files.

Every version saved stays here; git keeps every change of a file. Not here: the dataset and the reviews (the PBS backup
of the LXC), the ONNX exports (Publish makes them again).

## Restore

On the server: Models › Publish › GitHub › **Restore from GitHub**, or `scripts/restore_networks_github.sh` in the code
checkout. On a new machine, with an SSH key that may read this repository:

```sh
git clone https://github.com/cristianku/sunnypilot.smartspeed.git smartspeed && cd smartspeed
python3 -m venv .venv && .venv/bin/pip install -e .
scripts/restore_networks_github.sh       # this repository, cloned beside the checkout, into ./data
.venv/bin/nnslr review                   # the site: Models shows these networks, in use and on the car
```

It checks every file against the manifest, puts back every file missing in the data root with its time and mode, never
overwrites one that is there, and puts back the records only when they are missing. Any data root will do. The code's
Fit box comes back with the code repository; the saved one can be copied by hand from `code/`.

## In use at the last save ({str(manifest.get('saved_at') or '')[:16].replace('T', ' ')})

{ids(manifest.get('in_use') or {})}
{'- road labeler: `' + manifest['road_labeler'] + '`' if manifest.get('road_labeler') else ''}

## On the car

{ids(manifest.get('on_car') or {})}

## Every version saved ({len(manifest['versions'])})

| Version | Role | First saved |
|---|---|---|
{rows}
"""


# -- GitHub ------------------------------------------------------------------------------------------------------------
def _token() -> str | None:
    path = token_path()
    try:
        token = path.read_text(encoding='utf-8').strip()
    except OSError:
        return None
    return token or None


def _request(method: str, path: str, token: str | None, body: dict | None = None) -> tuple[int, Any]:
    """GitHub's API: (status, JSON body). The token goes in the header only; no error names it."""
    headers = {'Accept': 'application/vnd.github+json', 'User-Agent': 'smartspeed-networks',
               'X-GitHub-Api-Version': '2022-11-28'}
    if token:
        headers['Authorization'] = 'Bearer ' + token
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        headers['Content-Type'] = 'application/json'
    request = urllib.request.Request((os.environ.get(API_ENV) or API).rstrip('/') + path, data=data, method=method,
                                     headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=API_TIMEOUT) as response:
            raw = response.read()
            return response.status, json.loads(raw) if raw else None
    except urllib.error.HTTPError as error:
        try:
            raw = error.read()
            return error.code, json.loads(raw) if raw else None
        except (OSError, ValueError):
            return error.code, None
    except (urllib.error.URLError, OSError, ValueError) as error:
        reason = getattr(error, 'reason', None) or type(error).__name__
        raise ValueError(f'GitHub cannot be reached ({reason})') from None


def _private_or_refuse(repo: str, body: Any) -> str:
    if not isinstance(body, dict) or body.get('private') is not True:
        raise ValueError(f'{repo} is public on GitHub: nothing is pushed to a public repository (make it private '
                         'in its settings first)')
    return body.get('html_url') or f'https://github.com/{repo}'


def ensure_repo(repo: str, *, create: bool = True, log: Callable[[str], None] = print) -> dict:
    """The repository on GitHub, private: checked with the token (created, private, when it is missing); without a
    token GitHub answers 404 for a private repository and 200 for a public one, which is refused. {'repo', 'url',
    'created', 'checked' ('token' or 'anonymous')}."""
    token = _token()
    status, body = _request('GET', f'/repos/{repo}', token)
    if status == 200:
        url = _private_or_refuse(repo, body)
        log(f'{repo}: private on GitHub')
        return {'repo': repo, 'url': url, 'created': False, 'checked': 'token' if token else 'anonymous'}
    if status == 401:
        raise ValueError(f'GitHub refuses the token of this server (401): renew {token_path()}')
    if status != 404:
        raise ValueError(f'GitHub answered {status} for {repo}')
    if token is None:
        log(f'{repo}: not public on GitHub (no token here to check more: {token_path()})')
        return {'repo': repo, 'url': f'https://github.com/{repo}', 'created': False, 'checked': 'anonymous'}
    if not create:
        raise ValueError(f'{repo} does not exist on GitHub: save the networks first')
    owner, name = repo.split('/', 1)
    status, user = _request('GET', '/user', token)
    login = user.get('login') if status == 200 and isinstance(user, dict) else None
    where = '/user/repos' if isinstance(login, str) and login.lower() == owner.lower() else f'/orgs/{owner}/repos'
    status, body = _request('POST', where, token, {
        'name': name, 'private': True, 'auto_init': True, 'has_issues': False, 'has_wiki': False,
        'has_projects': False,
        'description': 'Smartspeed networks: a private copy of every network in use (sunnypilot.smartspeed #92)'})
    if status != 201:
        message = body.get('message') if isinstance(body, dict) else None
        raise ValueError(f'GitHub did not create {repo}: {status}{" " + message if message else ""}')
    status, body = _request('GET', f'/repos/{repo}', token)
    if status != 200:
        raise ValueError(f'{repo} was created but GitHub answers {status} for it')
    url = _private_or_refuse(repo, body)
    log(f'{repo}: created, private')
    return {'repo': repo, 'url': url, 'created': True, 'checked': 'token'}


def _git_env() -> dict:
    env = {**os.environ, **{key: value for key, value in IDENTITY.items() if not os.environ.get(key)}}
    env.setdefault('GIT_SSH_COMMAND', 'ssh -o BatchMode=yes -o ConnectTimeout=30')
    env['GIT_TERMINAL_PROMPT'] = '0'
    return env


def _git(folder: Path | None, *args: str) -> str:
    command = ['git', *(('-C', str(folder)) if folder is not None else ()), *args]
    try:
        done = subprocess.run(command, capture_output=True, text=True, timeout=GIT_TIMEOUT, check=False, env=_git_env())
    except subprocess.TimeoutExpired:
        raise ValueError(f'git {args[0]} took more than {GIT_TIMEOUT} s') from None
    if done.returncode != 0:
        raise ValueError(f'git {" ".join(args[:2])}: {done.stderr.strip() or "failed"}')
    return done.stdout.strip()


def sync(clone: Path, remote: str, *, log: Callable[[str], None] = print) -> str:
    """The clone equal to origin's main, cloned (only its last commit) when missing; left over changes of a save that
    failed are dropped (the clone is this module's alone). Its commit."""
    clone = Path(clone)
    if not clone.exists():
        clone.parent.mkdir(parents=True, exist_ok=True)
        _git(None, 'clone', '--quiet', '--depth', '1', '--branch', BRANCH, remote, str(clone))
        log(f'cloned into {clone}')
    else:
        if not (clone / '.git').is_dir():
            raise ValueError(f'{clone} is not a git clone: move it away first')
        url = _git(clone, 'remote', 'get-url', 'origin')
        if url != remote:
            raise ValueError(f'{clone} is a clone of {url}, not of {remote}')
        _git(clone, 'fetch', '--quiet', '--depth', '1', 'origin', f'+refs/heads/{BRANCH}:refs/remotes/origin/{BRANCH}')
        _git(clone, 'reset', '--quiet', '--hard')
        _git(clone, 'clean', '-fdq')
        _git(clone, 'checkout', '--quiet', '-B', BRANCH, f'refs/remotes/origin/{BRANCH}')
    commit = _git(clone, 'rev-parse', 'HEAD')
    log(f'{clone} at {commit[:12]}')
    return commit


def push(clone: Path, start: str, message: str) -> str | None:
    """Everything in the clone committed and pushed to origin's main: the commit, None when nothing changed. A failure
    puts the clone back on start."""
    _git(clone, 'add', '-A')
    if not _git(clone, 'status', '--porcelain'):
        return None
    try:
        _git(clone, 'commit', '--quiet', '-m', message)
        commit = _git(clone, 'rev-parse', 'HEAD')
        _git(clone, 'push', '--quiet', 'origin', f'HEAD:refs/heads/{BRANCH}')
    except ValueError:
        _git(clone, 'reset', '--quiet', '--hard', start)
        raise
    return commit


def save(root: Path, *, clone: Path | None = None, log: Callable[[str], None] = print) -> dict:
    """scripts/save_networks_github.sh: the repository checked (created, private), the clone brought up to date, the
    networks copied, committed and pushed; the record <data root>/publish/github/saved.json written after the push."""
    root = Path(root)
    repo = repository()
    clone = Path(clone or clone_path())
    log(f'== 1/4 The repository {repo} on GitHub')
    found = ensure_repo(repo, log=log)
    log('== 2/4 Its copy on this server')
    start = sync(clone, remote_of(repo), log=log)
    log('== 3/4 The networks')
    copied = copy_networks(root, clone, repo, log=log)
    log('== 4/4 Push to GitHub')
    commit = push(clone, start, copied['message'])
    if commit is None:
        log('Nothing new: every network is on GitHub already')
    else:
        log(f'pushed {commit[:12]}')
    record = {'at': _now(), 'repo': repo, 'url': found['url'], 'commit': commit or start, 'pushed': commit is not None,
              'commit_url': f'{found["url"]}/commit/{commit or start}', 'updated': copied['changed'],
              **{key: copied[key] for key in ('new', 'same', 'files', 'bytes', 'versions', 'in_use', 'on_car',
                                              'road_labeler', 'warnings')}}
    _write_json(root / RECORD_DIR / SAVED, record)
    _write_saved_files(root, read_manifest(clone), commit or start, record['at'])
    return record


def _write_saved_files(root: Path, manifest: dict | None, commit: str, at: str) -> None:
    """<data root>/publish/github/saved-files.json: the size, time and sha256 of the source of every file the
    repository holds now (its manifest after the push), for freshness()."""
    files = {path: [entry.get('size'), entry.get('mtime'), entry.get('sha256')]
             for path, entry in ((manifest or {}).get('files') or {}).items() if isinstance(entry, dict)}
    _write_json(root / RECORD_DIR / SAVED_FILES, {'schema_version': 1, 'commit': commit, 'at': at, 'files': files})
    with _fresh_lock:
        _fresh.pop(str(root), None)


# -- restore -------------------------------------------------------------------------------------------------------
def verify(clone: Path) -> dict:
    """The clone's manifest with every file checked (a path of data/ or code/, a real file, its size and sha256):
    raises ValueError on the first that is not, before anything is written."""
    clone = Path(clone)
    manifest = read_manifest(clone)
    if manifest is None:
        raise ValueError(f'{clone} has no {MANIFEST} this module wrote: nothing was saved there yet')
    for path, entry in manifest['files'].items():
        if not safe_path(path):
            raise ValueError(f'{MANIFEST} names {path!r}: not a path of data/ or code/')
        if not isinstance(entry, dict) or not isinstance(entry.get('size'), int) \
                or not (isinstance(entry.get('sha256'), str) and SHA256.fullmatch(entry['sha256'])):
            raise ValueError(f'{MANIFEST}: the entry of {path} is not a file record')
        target = _real_file(clone, path)
        if target is None:
            raise ValueError(f'{path} is in {MANIFEST} but not a file of the clone')
        if target.stat().st_size != entry['size'] or _sha256(target) != entry['sha256']:
            raise ValueError(f'{path} is not the file saved (its size or sha256 differ)')
    return manifest


def put_back(root: Path, clone: Path, manifest: dict, *, log: Callable[[str], None] = print) -> dict:
    """Every file of data/ missing in the data root copied there with its mode and time; one that is there and
    differs kept (never overwritten), one that is the same left. code/ (Fit box) only compared with this checkout.
    The records (models/*.json) come last: a restore stopped halfway never leaves a record naming networks not put
    back yet. {'copied', 'same', 'kept' (paths), 'box_fit' ('same', 'differs' or 'missing' per file), 'versions'}."""
    root, clone = Path(root), Path(clone)
    copied, same, kept = 0, 0, []
    box_fit = {}
    records = {f'{DATA}/{relative}' for relative in RECORDS}
    for path, entry in sorted(manifest['files'].items(), key=lambda item: (item[0] in records, item[0])):
        top, relative = path.split('/', 1)
        if top == CODE:
            here = (BOX_FIT_DIR / relative[len(BOX_FIT_PATH):] if relative.startswith(BOX_FIT_PATH)
                    else Path(__file__).resolve().parents[2] / relative)
            box_fit[relative] = ('missing' if not here.is_file() else
                                 'same' if _sha256(here) == entry['sha256'] else 'differs')
            continue
        target = root / relative
        if target.is_symlink() or target.exists():
            if not target.is_symlink() and target.is_file() and target.stat().st_size == entry['size'] \
                    and _sha256(target) == entry['sha256']:
                same += 1
            else:
                kept.append(relative)
            continue
        mode = entry.get('mode') if isinstance(entry.get('mode'), int) else None
        mtime = entry.get('mtime') if isinstance(entry.get('mtime'), (int, float)) else None
        digest = _copy(clone / path, target, mode=mode & 0o777 if mode is not None else None, mtime=mtime)
        if digest != entry['sha256']:
            target.unlink(missing_ok=True)
            raise ValueError(f'{path} changed while it was copied')
        copied += 1
    log(f'{copied} files put back, {same} already here, {len(kept)} here and different (kept as they are)')
    for relative in kept[:20]:
        log(f'kept: {relative}')
    if len(kept) > 20:
        log(f'kept: and {len(kept) - 20} more')
    differs = [relative for relative, state in box_fit.items() if state != 'same']
    if differs:
        log('Fit box: the saved ' + ', '.join(Path(relative).name for relative in differs) + ' differ from this '
            f'checkout\'s and stay as they are (by hand: cp {clone}/{CODE}/{BOX_FIT_PATH}<file> {BOX_FIT_DIR}/)')
    return {'copied': copied, 'same': same, 'kept': kept, 'box_fit': box_fit, 'versions': len(manifest['versions'])}


def restore(root: Path, *, clone: Path | None = None, log: Callable[[str], None] = print) -> dict:
    """scripts/restore_networks_github.sh: the clone brought up to date, every file checked, the missing ones put back
    in the data root; the records <data root>/publish/github/restored.json and saved-files.json (what GitHub holds)."""
    root = Path(root)
    repo = repository()
    clone = Path(clone or clone_path())
    log(f'== 1/3 The copy of {repo} on this server')
    commit = sync(clone, remote_of(repo), log=log)
    log('== 2/3 Check the saved files')
    manifest = verify(clone)
    log(f'{len(manifest["files"])} files of {len(manifest["versions"])} versions, saved {manifest.get("saved_at")}: '
        'all as saved')
    log('== 3/3 Put back what is missing here')
    root.mkdir(parents=True, exist_ok=True)
    done = put_back(root, clone, manifest, log=log)
    record = {'at': _now(), 'repo': repo, 'commit': commit, 'saved_at': manifest.get('saved_at'),
              **{key: done[key] for key in ('copied', 'same', 'versions', 'box_fit')}, 'kept': done['kept'][:50],
              'kept_count': len(done['kept'])}
    _write_json(root / RECORD_DIR / RESTORED, record)
    # What GitHub holds is now the clone's manifest: the files put back carry its sizes and times, so the panel tells
    # them saved, and the ones kept here as changed.
    _write_saved_files(root, manifest, commit, manifest.get('saved_at') if isinstance(manifest.get('saved_at'), str)
                       else record['at'])
    return record


# -- the panel ---------------------------------------------------------------------------------------------------------
# Per data root: (when, the stamp of the records it depends on, the answer or the error it gave).
_fresh: dict[str, tuple[float, tuple, dict | Exception]] = {}
_fresh_lock = threading.Lock()
# The records whose change shows at once, whichever process wrote them: the save's and the restore's (a save runs in a
# script, not in the server), and the choices of what is in use (the networks', Fit box's).
FRESH_RECORDS = (f'{RECORD_DIR.as_posix()}/{SAVED_FILES}', f'{RECORD_DIR.as_posix()}/{SAVED}',
                 f'{RECORD_DIR.as_posix()}/{RESTORED}', 'models/active.json', 'models/box-fit.json')
UNSAVED_SHOWN = 50


_digests: dict[tuple[str, int, int], str] = {}


def _digest(path: Path, info: os.stat_result) -> str:
    """The sha256 of a file, kept while its time and size stay the same."""
    key = (str(path), info.st_mtime_ns, info.st_size)
    found = _digests.get(key)
    if found is None:
        found = _sha256(path)
        if len(_digests) > 4096:
            _digests.clear()
        _digests[key] = found
    return found


def _saved_files(root: Path) -> tuple[dict | None, str | None, str | None]:
    """{path: (size, mtime, sha256)} of the last save, its commit and time: saved-files.json, else the manifest of the
    clone (a save of before it); (None, None, None) before the first save."""
    data = _json(_real_file(root, (RECORD_DIR / SAVED_FILES).as_posix()))
    if isinstance(data, dict) and isinstance(data.get('files'), dict):
        files = {path: (*value, None)[:3] for path, value in data['files'].items()
                 if isinstance(value, list) and len(value) in (2, 3)}
        return files, data.get('commit') if isinstance(data.get('commit'), str) else None, \
            data.get('at') if isinstance(data.get('at'), str) else None
    saved = _json(_real_file(root, (RECORD_DIR / SAVED).as_posix()))
    if not isinstance(saved, dict):
        return None, None, None
    manifest = read_manifest(clone_path())
    if manifest is None:
        return None, None, None
    files = {path: (entry.get('size'), entry.get('mtime'), entry.get('sha256')) for path, entry in manifest['files'].items()
             if isinstance(entry, dict)}
    return files, saved.get('commit') if isinstance(saved.get('commit'), str) else None, \
        saved.get('at') if isinstance(saved.get('at'), str) else None


def _file_state(saved: dict, path: str, source: Path) -> str:
    entry = saved.get(path)
    if entry is None:
        return 'missing'
    try:
        info = source.stat()
    except OSError:
        return 'changed'
    size, mtime, digest = entry
    if size != info.st_size:
        return 'changed'
    if isinstance(mtime, (int, float)) and abs(mtime - info.st_mtime) < 1e-3:
        return 'saved'
    # Another time, the same size (a fresh clone of the code, a file copied again): the same bytes are saved.
    try:
        return 'saved' if isinstance(digest, str) and _digest(source, info) == digest else 'changed'
    except OSError:
        return 'changed'


def _worst(states: list[str]) -> str:
    if not states or all(state == 'missing' for state in states):
        return 'missing'
    return 'saved' if all(state == 'saved' for state in states) else 'changed'


def _fresh_stamp(root: Path) -> tuple:
    stamp = []
    for relative in FRESH_RECORDS:
        try:
            info = (root / relative).stat()
            stamp.append((info.st_mtime_ns, info.st_size))
        except OSError:
            stamp.append(None)
    return tuple(stamp)


def freshness(root: Path, *, now: float | None = None) -> dict:
    """Whether GitHub has what a save would copy now (#92): {'up_to_date' (None before the first save), 'versions':
    {id: 'saved' | 'changed' | 'missing'}, 'records': {path of the data root: ...} (the records and the brain's
    splits), 'unsaved' (the first UNSAVED_SHOWN ids and paths not saved as they are here), 'unsaved_count',
    'saved_at', 'commit' (of the save it compares with: the last one, or the one a restore put back), 'warnings' (what
    a save would leave out, plan's)}. A file is saved when its size and time are those of the last save, or its size and sha256
    (another time). Kept FRESH_SECONDS while the records of FRESH_RECORDS stay the same, an error too (said once in the
    log, raised again at each read)."""
    root = Path(root)
    key, now, stamp = str(root), time.monotonic() if now is None else now, _fresh_stamp(Path(root))
    with _fresh_lock:
        cached = _fresh.get(key)
    if cached is not None and now - cached[0] < FRESH_SECONDS and cached[1] == stamp:
        if isinstance(cached[2], Exception):
            raise cached[2]
        return cached[2]
    try:
        found = _freshness(root)
    except Exception as error:
        print(f'networks_github: whether GitHub is up to date cannot be told: {type(error).__name__}: {error}')
        with _fresh_lock:
            _fresh[key] = (now, stamp, error)
        raise
    with _fresh_lock:
        _fresh[key] = (now, stamp, found)
    return found


def _freshness(root: Path) -> dict:
    planned = plan(root)
    saved, commit, at = _saved_files(root)
    states = {path: _file_state(saved, path, source) if saved is not None else 'missing'
              for path, source in planned['files'].items()}
    versions = {model_id: _worst([states[path] for path in version['files'] if path in states])
                for model_id, version in planned['versions'].items()}
    in_versions = {path for version in planned['versions'].values() for path in version['files']}
    records = {path.split('/', 1)[1]: state for path, state in sorted(states.items()) if path not in in_versions}
    unsaved = [*(model_id for model_id, state in versions.items() if state != 'saved'),
               *(path for path, state in records.items() if state != 'saved')]
    return {'up_to_date': None if saved is None else not unsaved, 'versions': versions, 'records': records,
            'unsaved': unsaved[:UNSAVED_SHOWN], 'unsaved_count': len(unsaved), 'saved_at': at, 'commit': commit,
            'warnings': list(planned['warnings'])[:UNSAVED_SHOWN]}


def status(root: Path) -> dict:
    """For the panel's status (networks.github): the repository, the last save and restore of this data root,
    whether the token file is there, and whether GitHub is up to date (freshness; None when it cannot be told)."""
    root = Path(root)
    try:
        repo = repository()
    except ValueError:
        repo = REPO
    saved = _json(_real_file(root, (RECORD_DIR / SAVED).as_posix()))
    restored = _json(_real_file(root, (RECORD_DIR / RESTORED).as_posix()))
    try:
        fresh = freshness(root)
    except Exception:  # noqa: BLE001 - the panel's status must not fail on a data root it cannot read (freshness logs it)
        fresh = None
    return {'repo': repo, 'url': f'https://github.com/{repo}', 'clone': str(clone_path()),
            'token': token_path().is_file(), 'saved': saved if isinstance(saved, dict) else None,
            'restored': restored if isinstance(restored, dict) else None, 'freshness': fresh}
