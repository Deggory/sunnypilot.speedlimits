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
# How long the scan of the processes is reused: a save asks every time, and it is not free. The drives themselves
# are read each time: the page of the drive at work asks every few seconds how far it is.
CACHE_SECONDS = 3.0
# How fast the drive at work goes: from the times of the preview pictures of its last PACE_FRAMES frames. Qwen3-VL
# answers several frames together (NNSLR_QWEN_BATCH, 8), written within SAME_STEP seconds: a step. The page shows how
# far the step at work is when a step takes at least STEP_SHOWN seconds (a faster one would only flicker).
PACE_FRAMES = 41
SAME_STEP = 1.0
STEP_SHOWN = 5.0

_lock = threading.Lock()
_running: list = []  # [(when asked, the answer)] of _again_running, at most one


def _running_now() -> bool:
    now = time.monotonic()
    with _lock:
        if _running and now - _running[0][0] < CACHE_SECONDS:
            return _running[0][1]
    answer = _again_running()
    with _lock:
        _running[:] = [(now, answer)]
    return answer


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


def _pace(pending: Path, done: int) -> dict:
    """{'last_at': when the newest frames were written (epoch seconds), 'frame_seconds': the seconds a frame takes,
    'step_frames' and 'step_seconds': the frames written together and how long they take}, None for what the preview
    pictures (images/<index>.jpg: the first one, and those of the last PACE_FRAMES frames for the steps) do not tell
    yet."""
    times = []
    for index in range(max(0, done - PACE_FRAMES), done):
        try:
            times.append((index, (pending / 'images' / f'{index:08d}.jpg').stat().st_mtime))
        except OSError:
            continue
    pace = {'last_at': times[-1][1] if times else None, 'frame_seconds': None, 'step_frames': None,
            'step_seconds': None}
    steps = []  # [first index, last index, time of the last]
    for index, at in times:
        if steps and at - steps[-1][2] <= SAME_STEP:
            steps[-1][1:] = [index, at]
        else:
            steps.append([index, index, at])
    # The time a frame takes since the first frames of the drive (the steps of Qwen3-VL take 5 to 25 s, as its answers
    # are short or long): from the end of the first step to the end of the last.
    try:
        first_at = (pending / 'images' / f'{0:08d}.jpg').stat().st_mtime
    except OSError:
        first_at = None
    first_step = steps[-1][1] - steps[-1][0] if len(steps) >= 2 else 0
    if first_at is not None and times and times[-1][0] - first_step > 0 and times[-1][1] > first_at:
        pace['frame_seconds'] = (times[-1][1] - first_at) / (times[-1][0] - first_step)
    gaps = sorted(later[2] - earlier[2] for earlier, later in zip(steps, steps[1:]))
    if gaps and gaps[len(gaps) // 2] >= STEP_SHOWN:
        pace['step_seconds'] = gaps[len(gaps) // 2]
        pace['step_frames'] = steps[-1][1] - steps[-1][0] + 1
    return pace


def _read(root: Path) -> dict:
    try:
        batch = json.loads((root / BATCH_FILE).read_text(encoding='utf-8'))
        started = datetime.strptime(batch['started'], '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc)
        routes = [route for route in batch['routes'] if isinstance(route, str)]
    except (OSError, ValueError, KeyError, TypeError):
        return {}
    if not routes or not _running_now():
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
            done = _line_count(fresh / 'preannotations.jsonl')
            found[route] = {'phase': 'proposing again', 'done': done, 'pending': fresh, 'pace': _pace(fresh, done)}
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
    return _read(Path(root))


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
    """The state of a drive for the review page (no paths): phase, proposer, ahead (drives before it) waiting; at work
    done and total frames, left_seconds (an estimate, None when not known) and step, the frames being proposed now:
    {'frames', 'seconds' (how long a step takes), 'since' (seconds since the last step was written)} or None."""
    if item is None:
        return None
    result = {'phase': item['phase'], 'proposer': item.get('proposer'), 'message': message(item)}
    if item['phase'] == 'proposing again':
        done = item.get('done', 0)
        pace = item.get('pace') or {}
        seconds = pace.get('frame_seconds')
        result.update(done=done, total=total, left_seconds=None, step=None)
        if seconds and total and total > done:
            result['left_seconds'] = round((total - done) * seconds)
        if pace.get('step_seconds') and pace.get('last_at') and (total is None or total > done):
            frames = pace['step_frames'] if total is None else min(pace['step_frames'], total - done)
            result['step'] = {'frames': frames, 'seconds': round(pace['step_seconds'], 1),
                              'since': round(max(0.0, time.time() - pace['last_at']), 1)}
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
        _running.clear()

