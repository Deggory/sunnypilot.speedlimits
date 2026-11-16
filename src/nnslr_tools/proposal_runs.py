"""Old proposal runs of a drive: their pictures go, their records stay.

Every Prepare and every Propose again writes a new run under derived/preannotations/<route>/<run_id>/ and points
latest.json at it. Sign review, the import and the panel read only that latest run (review.latest_report), so the
pictures of the earlier runs (images/, about 200 MB a run) are read by nobody. This removes them and keeps the small
files beside them (run.json, preannotations.jsonl, request.json, index.html) as the record of what each run proposed.
It also removes the .pending-<run_id> folders a run left behind when it failed or was stopped, once a later run is the
latest and the folder has not changed for an hour (a run still writing is never touched).

The review database is never read or written here: confirmed frames keep their rows whatever happens to the pictures.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import time
from pathlib import Path

RUN_ID = re.compile(r'[0-9]{8}T[0-9]{6}Z-[0-9a-f]{8}')
PENDING_QUIET_SECONDS = 3600


def _size(path: Path) -> int:
    total = 0
    for folder, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.lstat(os.path.join(folder, name)).st_size
            except OSError:
                pass
    return total


def _newest_change(path: Path) -> float:
    newest = path.stat().st_mtime
    for folder, dirs, files in os.walk(path):
        for name in dirs + files:
            try:
                newest = max(newest, os.lstat(os.path.join(folder, name)).st_mtime)
            except OSError:
                pass
    return newest


def _latest_run(base: Path) -> str | None:
    try:
        run_id = json.loads((base / 'latest.json').read_text()).get('run_id')
    except (OSError, ValueError, AttributeError):
        return None
    if not isinstance(run_id, str) or not RUN_ID.fullmatch(run_id) or not (base / run_id).is_dir():
        return None
    return run_id


def prune_route(base: Path, *, dry_run: bool = False, now: float | None = None) -> dict:
    """Prune one drive's proposal folder; a drive without a valid latest.json is left as it is."""
    latest = _latest_run(base)
    result = {'route': base.name, 'latest': latest, 'images_removed': [], 'pending_removed': [], 'bytes': 0}
    if latest is None:
        return result
    now = time.time() if now is None else now
    for entry in sorted(base.iterdir()):
        if entry.is_symlink() or not entry.is_dir():
            continue
        if RUN_ID.fullmatch(entry.name) and entry.name < latest:
            images = entry / 'images'
            if images.is_dir() and not images.is_symlink():
                size = _size(images)
                if not dry_run:
                    shutil.rmtree(images)
                result['images_removed'].append(entry.name)
                result['bytes'] += size
        elif entry.name.startswith('.pending-'):
            run_id = entry.name[len('.pending-'):]
            if RUN_ID.fullmatch(run_id) and run_id < latest and now - _newest_change(entry) > PENDING_QUIET_SECONDS:
                size = _size(entry)
                if not dry_run:
                    shutil.rmtree(entry)
                result['pending_removed'].append(entry.name)
                result['bytes'] += size
    return result


def prune_old_runs(root: Path, route: str | None = None, *, dry_run: bool = False) -> dict:
    """Prune every drive's old proposal runs, or one drive's."""
    top = root / 'derived' / 'preannotations'
    if route is not None:
        from nnslr_tools.manifest import parse_route_id
        parse_route_id(route)
        bases = [top / route]
    else:
        bases = sorted(p for p in top.iterdir() if p.is_dir() and not p.is_symlink()) if top.is_dir() else []
    routes = [prune_route(base, dry_run=dry_run) for base in bases if base.is_dir()]
    routes = [r for r in routes if r['images_removed'] or r['pending_removed']]
    return {'dry_run': dry_run, 'routes': routes,
            'runs_with_images_removed': sum(len(r['images_removed']) for r in routes),
            'pending_removed': sum(len(r['pending_removed']) for r in routes),
            'bytes': sum(r['bytes'] for r in routes),
            'gib': round(sum(r['bytes'] for r in routes) / 2**30, 2)}
