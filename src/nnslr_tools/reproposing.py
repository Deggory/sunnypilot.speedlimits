"""The drives Propose again is working on (helpers/1_prepare.sh --again): read only until their new proposals are in.

When it starts, helpers/1_prepare.sh --again lists the drives of its batch in <data root>/logs/prepare/batch.json.
For each drive in turn it writes the new proposals aside (derived/preannotations/<drive>/.pending-<run>/), and they
replace the report when that drive is done. Only drives nobody has reviewed are proposed again, so a frame
confirmed on one of them while it waits or is at work would be confirmed on a report about to go: while the batch
runs, every drive of it not done yet is read only. Every save comes through review_db.save_frame_review (the review
page, /api/v1, MCP, the AI review), which refuses them (refuse_if_busy). The review page of the drive at work shows
the new proposals made so far (at_work), and the drives still waiting show their current report.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import threading
import time


BATCH_FILE = 'logs/prepare/batch.json'
# The helper, in the command line of the panel's wrapper or of a terminal.
HELPER = b'1_prepare.sh'
AGAIN = b'--again'
# How long an answer is reused: a save asks every time, the scan of the processes is not free.
CACHE_SECONDS = 3.0

_lock = threading.Lock()
_cache: dict[str, tuple[float, dict]] = {}


def _again_running() -> bool:
    """Does helpers/1_prepare.sh --again run on this machine (from the panel or from a terminal)?"""
    proc = Path('/proc')
    if not proc.is_dir():
        return False
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            parts = (entry / 'cmdline').read_bytes().split(b'\0')
        except OSError:
            continue
        if AGAIN in parts and any(part.endswith(HELPER) for part in parts):
            return True
    return False


def _run_id(base: Path) -> str | None:
    try:
        run_id = json.loads((base / 'latest.json').read_text(encoding='utf-8')).get('run_id')
    except (OSError, ValueError, AttributeError):
        return None
    return run_id if isinstance(run_id, str) else None


def _line_count(path: Path) -> int:
    try:
        with path.open('rb') as handle:
            return sum(1 for line in handle if line.endswith(b'\n'))
    except OSError:
        return 0


def _proposer(pending: Path) -> str | None:
    try:
        model = json.loads((pending / 'request.json').read_text(encoding='utf-8')).get('model') or {}
    except (OSError, ValueError, AttributeError):
        return None
    return model.get('proposer') if isinstance(model, dict) and isinstance(model.get('proposer'), str) else None


def _read(root: Path) -> dict:
    try:
        batch = json.loads((root / BATCH_FILE).read_text(encoding='utf-8'))
        started = datetime.strptime(batch['started'], '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc)
        routes = [route for route in batch['routes'] if isinstance(route, str)]
    except (OSError, ValueError, KeyError, TypeError):
        return {}
    if not routes or not _again_running():
        return {}
    stamp = started.strftime('%Y%m%dT%H%M%SZ')
    found, proposer, ahead = {}, None, 0
    for route in routes:
        base = root / 'derived/preannotations' / route
        run_id = _run_id(base)
        if run_id is None:
            # Not a drive with a report: a batch of new drives, not of Propose again.
            return {}
        if run_id >= stamp:
            continue
        pending = sorted(base.glob('.pending-*')) if base.is_dir() else []
        fresh = pending[-1] if pending and pending[-1].name[len('.pending-'):] >= stamp else None
        if fresh is not None:
            proposer = proposer or _proposer(fresh)
            found[route] = {'phase': 'proposing again', 'done': _line_count(fresh / 'preannotations.jsonl'),
                            'pending': fresh}
        else:
            found[route] = {'phase': 'waiting', 'ahead': ahead}
        ahead += 1
    from nnslr_tools.preannotate import PROPOSER_NAMES
    for item in found.values():
        item['proposer_id'] = proposer
        item['proposer'] = PROPOSER_NAMES.get(proposer, proposer) if proposer else None
    return found


def busy(root: Path) -> dict:
    """{drive: {'phase': 'waiting' | 'proposing again', 'proposer': its name or None, and 'done' (frames proposed so
    far) with 'pending' (the folder they are written to) at work, 'ahead' (drives before it) waiting}} for the drives
    of a running Propose again not done yet; {} when none runs."""
    key = str(root)
    now = time.monotonic()
    with _lock:
        cached = _cache.get(key)
        if cached is not None and now - cached[0] < CACHE_SECONDS:
            return cached[1]
    found = _read(Path(root))
    with _lock:
        _cache[key] = (now, found)
    return found


def of(root: Path, route: str) -> dict | None:
    """The state of one drive in busy(), or None when it can be reviewed."""
    return busy(root).get(route)


def message(item: dict) -> str:
    who = item.get('proposer') or 'Propose again'
    if item['phase'] == 'proposing again':
        return (who + ' is proposing the signs of this drive again: it is read only until its new proposals are in '
                '(Prepare shows how far it is).')
    return ('This drive is waiting for Propose again (' + who + '): its proposals will be replaced, so it is read '
            'only until that is done.')


def refuse_if_busy(root: Path, route: str) -> None:
    """A save on a drive of a running Propose again is refused (409 drive_busy)."""
    item = of(root, route)
    if item is not None:
        from nnslr_tools.review_db import ReviewRejected
        raise ReviewRejected(409, 'drive_busy', message(item))


def public(item: dict | None, total: int | None = None) -> dict | None:
    """The state of a drive for the review page (no paths): phase, proposer, done and total frames, ahead."""
    if item is None:
        return None
    result = {'phase': item['phase'], 'proposer': item.get('proposer'), 'message': message(item)}
    if item['phase'] == 'proposing again':
        result['done'] = item.get('done', 0)
        result['total'] = total
    else:
        result['ahead'] = item.get('ahead', 0)
    return result


def at_work_rows(item: dict) -> list[dict]:
    """The rows proposed so far in the pending folder of the drive at work; a line still being written is skipped."""
    rows = []
    try:
        with (item['pending'] / 'preannotations.jsonl').open('rb') as handle:
            for line in handle:
                if not line.endswith(b'\n'):
                    break
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if isinstance(row, dict):
                    rows.append(row)
    except (OSError, KeyError):
        return []
    return rows


def run_dir_name(item: dict) -> str:
    """The run the new proposals will have once in (.pending-<run> without its prefix)."""
    return item['pending'].name[len('.pending-'):]


def reset_cache() -> None:
    with _lock:
        _cache.clear()

