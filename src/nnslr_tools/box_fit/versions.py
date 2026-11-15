"""The versions of Fit box (#94): every training is a version of its own, and a person chooses the one the review page
uses, as for the other networks. Nothing is replaced by a training.

A version is the three files of one training together: the colour trees (``colour.npz``), the edge networks
(``edges.npz``) and the box refiner (``refiner.onnx``, when it has one), in its run folder
``<data root>/runs/<YYYY-MM-DD_HHMM>-box-fit/``, with ``box-fit.json``, the record of its training (when, on how many
boxes, where its refiner comes from) and its measure: on the boxes of the drives left out, the rough box, its rankers
alone, the version as the review page runs it, and the version in use then as the review page runs it, on the same
boxes. The three go together because the refiner is measured on top of its own rankers.

The files of the code checkout (``src/nnslr_tools/box_fit/``) are the version ``box_fit:code``: the one in use until
someone chooses, and on a fresh clone. The choice is ``<data root>/models/box-fit.json``, ``{"id", "who", "via", "at"}``,
written in one step like models/active.json, but in a file of its own: nnslr_tools.model_roles writes active.json whole,
with its own roles only. Choosing copies nothing: the review page reads the folder in use at each fit
(:func:`folder_in_use`), so a choice holds from the next fit, without a restart. A choice whose folder is gone falls
back to the code's files, and says so.

The measure is on drives no arm was trained on: about a quarter of the routes, the same every time (:func:`held_out`).
A version trains its rankers on every box, those routes too, so it keeps a second pair trained without them in its
``measure/`` folder, with a copy of its refiner: :func:`measure_folder` gives that folder to the measure, and the
version's own folder only when it has none (``box_fit:code`` and the versions trained before 15 Nov 2026).

Standard library only: the review page imports it at every fit.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import tempfile
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
RUN_SUFFIX = '-box-fit'
RUN = re.compile(r'\d{4}-\d{2}-\d{2}_\d{4}' + RUN_SUFFIX)
CODE = 'code'
CODE_ID = 'box_fit:' + CODE
VERSION_ID = re.compile(r'box_fit:(' + CODE + r'|' + RUN.pattern + r')')
CHOICE_FILE = 'models/box-fit.json'
RECORD = 'box-fit.json'
RANKERS = ('colour.npz', 'edges.npz')
REFINER = 'refiner.onnx'
FILES = (*RANKERS, REFINER)
# The rankers trained without the routes left out of the measure, and a copy of the version's refiner.
MEASURE = 'measure'
# How the person who chose was recognised (as model_roles.VIA).
VIA = ('sign-in', 'token', 'local')
MAX_WHO = 100

_write_lock = threading.Lock()
# The folder in use per data root, kept while models/box-fit.json does not change: (its mtime_ns and size, the folder).
_in_use_cache: dict[str, tuple[tuple[int, int] | None, Path]] = {}
_cache_lock = threading.Lock()


class ChoiceError(Exception):
    """A choice that is refused: HTTP status, reason, message (as model_roles.ModelChoiceError)."""

    def __init__(self, status: int, reason: str, message: str):
        super().__init__(message)
        self.status, self.reason, self.message = status, reason, message


def run_of(version_id: str) -> str | None:
    """The run of a version id (None for the code's files and for what is not an id)."""
    match = VERSION_ID.fullmatch(version_id) if isinstance(version_id, str) else None
    return None if match is None or match.group(1) == CODE else match.group(1)


def _runs(root: Path) -> Path | None:
    """<root>/runs, None when it leads out of the data root."""
    runs = Path(root) / 'runs'
    try:
        if not runs.resolve().is_relative_to(Path(root).resolve()):
            return None
    except (OSError, RuntimeError):
        return None
    return runs


def run_folder(root: Path, run: str) -> Path | None:
    """The folder of a version's run, when it is one: a real folder of <root>/runs (no link) holding both rankers."""
    if not isinstance(run, str) or RUN.fullmatch(run) is None:
        return None
    runs = _runs(root)
    if runs is None:
        return None
    folder = runs / run
    try:
        if folder.is_symlink() or not folder.is_dir():
            return None
        if not all((folder / name).is_file() and not (folder / name).is_symlink() for name in RANKERS):
            return None
    except OSError:
        return None
    return folder


def folder_of(root: Path, version_id: str) -> Path | None:
    """The folder of a version: the code's files for box_fit:code, else its run folder; None when there is none."""
    if version_id == CODE_ID:
        return HERE
    run = run_of(version_id)
    return None if run is None else run_folder(root, run)


def _choice_path(root: Path) -> Path | None:
    models = Path(root) / 'models'
    try:
        if not models.resolve().is_relative_to(Path(root).resolve()):
            return None
    except (OSError, RuntimeError):
        return None
    return models / Path(CHOICE_FILE).name


def read_choice(root: Path) -> tuple[dict | None, str | None]:
    """The version chosen (None: nobody chose, the code's files are in use) and why the file cannot be used, if it
    cannot."""
    path = _choice_path(root)
    if path is None:
        return None, 'models/ of the data root leads outside it'
    try:
        if not path.exists() and not path.is_symlink():
            return None, None
        if path.is_symlink() or not path.is_file():
            return None, CHOICE_FILE + ' is not a file'
        data = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None, CHOICE_FILE + ' cannot be read'
    if not isinstance(data, dict) or not isinstance(data.get('id'), str) or VERSION_ID.fullmatch(data['id']) is None:
        return None, CHOICE_FILE + ' does not name a version of Fit box'
    who, via, at = data.get('who'), data.get('via'), data.get('at')
    return {'id': data['id'], 'who': who if isinstance(who, str) and len(who) <= MAX_WHO else None,
            'via': via if via in VIA else None, 'at': at if isinstance(at, str) and len(at) <= 40 else None}, None


def in_use(root: Path) -> dict:
    """The version in use: {'id', 'folder', 'chosen' (the record of the choice, or None), 'found' (False: the chosen
    version's files are gone, the code's are used), 'problem'}."""
    chosen, problem = read_choice(root)
    if chosen is None:
        return {'id': CODE_ID, 'folder': HERE, 'chosen': None, 'found': True, 'problem': problem}
    folder = folder_of(root, chosen['id'])
    if folder is None:
        return {'id': CODE_ID, 'folder': HERE, 'chosen': chosen, 'found': False,
                'problem': 'the version chosen, ' + chosen['id'] + ', is no longer on this server: the code’s files are used'}
    return {'id': chosen['id'], 'folder': folder, 'chosen': chosen, 'found': True, 'problem': None}


def folder_in_use(root: Path) -> Path:
    """The folder of the version in use, for the review page's fit: read again only when models/box-fit.json changed
    (its time and size), so a choice holds from the next fit. A chosen version whose folder is missing is not kept:
    the code's files serve until it comes back (a restore puts it back without touching the choice), then it does."""
    path = _choice_path(root)
    try:
        stat = path.stat() if path is not None else None
        stamp = None if stat is None else (stat.st_mtime_ns, stat.st_size)
    except OSError:
        stamp = None
    key = str(root)
    with _cache_lock:
        cached = _in_use_cache.get(key)
    if (cached is not None and cached[0] == stamp and cached[0] is not None
            and all((cached[1] / name).is_file() for name in RANKERS)):
        return cached[1]
    used = in_use(root)
    with _cache_lock:
        if used['found']:
            _in_use_cache[key] = (stamp, used['folder'])
        else:
            _in_use_cache.pop(key, None)
    return used['folder']


def held_out(route: str) -> bool:
    """Whether a route is left out of the measure: about a quarter of the routes, the same every time."""
    return int(hashlib.sha1(route.encode()).hexdigest(), 16) % 4 == 0


def _plain_file(path: Path) -> bool:
    return path.is_file() and not path.is_symlink()


def measure_folder(folder: Path) -> tuple[Path, bool]:
    """The folder the measure runs a version from, and whether it is clean (trained without the routes left out): its
    ``measure/`` folder when that holds both rankers and the refiner exactly when the version has one, else the
    version's own folder (not clean)."""
    folder = Path(folder)
    measure = folder / MEASURE
    if (not measure.is_symlink() and measure.is_dir() and all(_plain_file(measure / name) for name in RANKERS)
            and _plain_file(measure / REFINER) == _plain_file(folder / REFINER)):
        return measure, True
    return folder, False


def record(folder: Path) -> dict | None:
    """The box-fit.json of a run folder, or None."""
    try:
        data = json.loads((folder / RECORD).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def files_of(folder: Path) -> list[str]:
    """The files of a version that are there, in FILES order."""
    return [name for name in FILES if (folder / name).is_file() and not (folder / name).is_symlink()]


def versions(root: Path) -> list[dict]:
    """Every version on this server, newest first, the code's files last: {'id', 'run' (None for the code's), 'at'
    (the record's, else the folder's name), 'files', 'record' (box-fit.json, or None)}."""
    found = []
    runs = _runs(root)
    try:
        names = sorted((entry.name for entry in runs.iterdir() if RUN.fullmatch(entry.name)), reverse=True) \
            if runs is not None and runs.is_dir() else []
    except OSError:
        names = []
    for run in names:
        folder = run_folder(root, run)
        if folder is None:
            continue
        data = record(folder)
        at = data.get('at') if isinstance(data, dict) and isinstance(data.get('at'), str) else None
        found.append({'id': 'box_fit:' + run, 'run': run, 'at': at, 'files': files_of(folder), 'record': data})
    found.append({'id': CODE_ID, 'run': None, 'at': None, 'files': files_of(HERE), 'record': None})
    return found


def _write_atomic(path: Path, value: dict) -> None:
    """The JSON written next to path, flushed, then renamed over it: a reader sees the old file or the new one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=path.parent, prefix='.' + path.name + '.',
                                         suffix='.tmp', delete=False)
    try:
        with handle:
            handle.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(handle.name, 0o644)
        os.replace(handle.name, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(handle.name)
        raise


def choose(root: Path, version_id: Any, *, who: str | None, via: str) -> dict:
    """POST /api/panel/box-fit/use: the version the review page uses from its next fit. The id must be a version on
    this server; only models/box-fit.json is written, never a path of the request."""
    if not isinstance(version_id, str) or len(version_id) > 80 or VERSION_ID.fullmatch(version_id) is None:
        raise ChoiceError(400, 'bad_id', 'Not a version of Fit box.')
    if via not in VIA:
        raise ValueError('via')
    if folder_of(root, version_id) is None:
        raise ChoiceError(404, 'unknown_version', 'No such version of Fit box on this server.')
    path = _choice_path(root)
    if path is None:
        raise ChoiceError(409, 'bad_models_folder', 'models/ of the data root leads outside it.')
    entry = {'id': version_id, 'who': who[:MAX_WHO] if isinstance(who, str) else None, 'via': via,
             'at': datetime.now().astimezone().isoformat(timespec='seconds')}
    with _write_lock:
        _, problem = read_choice(root)
        if problem is not None:
            # Never overwrite a file that cannot be read: it may hold a choice someone wants to keep.
            raise ChoiceError(409, 'choice_unreadable', problem + ': fix or remove it first.')
        _write_atomic(path, {'schema_version': 1, **entry})
    return entry


def _number(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def write_record(folder: Path, run: str, rankers: dict, refiner: dict | None, *, refiner_from: str | None = None) -> dict:
    """box-fit.json of a new version (scripts/train_box_fit.sh, at the end of its training): when, on how many boxes,
    the rankers' measure on drives left out (the folds of ``nnslr train-box-fit``), where its refiner comes from
    ('trained', 'carried' from the version named by refiner_from, or 'none'), the version in use then, and the measure
    of refiner_train on the full frames left out: this version and the one in use on the same boxes, and which of the
    two were measured clean (:func:`measure_folder`)."""
    cross = (rankers.get('cross_validation') or {}).get('overall') or {}
    refiner = refiner or {}
    left_out = refiner.get('left_out') if isinstance(refiner.get('left_out'), dict) else {}
    clean = refiner.get('clean') if isinstance(refiner.get('clean'), dict) else {}
    kept = {group: left_out[group] for group in ('all', 'size:small', 'size:mid', 'size:big', 'fair') if group in left_out}
    data = {'schema_version': 1, 'run': run, 'at': datetime.now().astimezone().isoformat(timespec='seconds'),
            'boxes': _number(rankers.get('boxes')), 'routes': _number(rankers.get('routes')),
            'folds': _number(rankers.get('folds')),
            'rankers': {'mean_iou': _number(cross.get('mean_iou')), 'rough_mean_iou': _number(cross.get('rough_mean_iou'))},
            'refiner': refiner.get('refiner') if refiner.get('refiner') in ('trained', 'carried', 'none') else 'none',
            'refiner_from': refiner_from, 'refiner_boxes': _number(refiner.get('train_boxes')),
            'held_out_boxes': _number(refiner.get('held_out_boxes')), 'best_epoch': _number(refiner.get('best_epoch')),
            'epochs': _number(refiner.get('epochs')),
            'in_use_then': (refiner.get('in_use') or {}).get('id'), 'left_out': kept,
            'measured_clean': {arm: clean[arm] for arm in ('version', 'in_use') if isinstance(clean.get(arm), bool)},
            'better_than_in_use': refiner.get('better_than_in_use') if isinstance(refiner.get('better_than_in_use'),
                                                                                   bool) else None}
    _write_atomic(Path(folder) / RECORD, data)
    return data


def main(argv=None) -> int:
    """``python -m nnslr_tools.box_fit.versions in-use --data-root DIR``: the id and the folder of the version in use,
    tab-separated (scripts/train_box_fit.sh measures a new version against it). ``record --folder DIR --run RUN
    --rankers JSON [--refiner JSON] [--refiner-from ID]``: box-fit.json of a new version from its summaries."""
    import argparse
    import sys
    parser = argparse.ArgumentParser(prog='python -m nnslr_tools.box_fit.versions')
    parser.add_argument('what', choices=('in-use', 'record'))
    parser.add_argument('--data-root')
    parser.add_argument('--folder')
    parser.add_argument('--run')
    parser.add_argument('--rankers', help="the summary of nnslr train-box-fit")
    parser.add_argument('--refiner', help='the summary of refiner_train (its last line)')
    parser.add_argument('--refiner-from', help='the version whose refiner.onnx this one carries')
    args = parser.parse_args(argv)
    if args.what == 'record':
        if not args.folder or not args.run or RUN.fullmatch(args.run) is None or not args.rankers:
            parser.error('record needs --folder, --run (YYYY-MM-DD_HHMM-box-fit) and --rankers')

        def last_json(path: str | None) -> dict | None:
            if not path:
                return None
            lines = [line for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
            value = json.loads(lines[-1]) if lines else None
            return value if isinstance(value, dict) else None
        rankers = json.loads(Path(args.rankers).read_text(encoding='utf-8'))
        data = write_record(Path(args.folder), args.run, rankers if isinstance(rankers, dict) else {},
                            last_json(args.refiner), refiner_from=args.refiner_from)
        print(json.dumps(data))
        return 0
    if not args.data_root:
        parser.error('in-use needs --data-root')
    used = in_use(Path(args.data_root))
    if used['problem']:
        print('Fit box: ' + used['problem'] + '.', file=sys.stderr)
    print(used['id'] + '\t' + str(used['folder']))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
