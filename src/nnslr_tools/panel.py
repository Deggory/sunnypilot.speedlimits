# [control-panel] - START
"""Control panel of the review site: the state of every route, the machine, and buttons that run
the easy helpers (helpers/fetch_comma_segments.sh, helpers/1_prepare.sh, helpers/3_train.sh,
helpers/update.sh) in the background with a live log.

Only the fixed actions of ACTIONS can run, never a command taken from a request. They are the
helpers of the checkout this package runs from (editable install); without them (wheel
install) the panel is read-only. NNSLR_PANEL_ACTIONS=0 also makes it read-only, e.g. for a
review service that other people use. The output of every action goes to
<data-root>/logs/panel/<job id>.log, and its last line is the exit code, so the history
survives a restart of the server.
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import signal
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

from nnslr_tools.job_progress import progress

REPO = Path(__file__).resolve().parents[2]
ACTIONS_ENV = 'NNSLR_PANEL_ACTIONS'
PREPARE_JOBS_ENV = 'NNSLR_PANEL_PREPARE_JOBS'
LOG_CHUNK_BYTES = 64 * 1024
# The progress of a job is read from at most the last 4 MB of its log.
PROGRESS_BYTES = 4 * 1024 * 1024
# Written by helpers/1_prepare.sh when it starts: the routes of that run, for the progress.
BATCH_FILE = 'logs/prepare/batch.json'
HISTORY = 12
JOB_ID = re.compile(r'[0-9]{8}-[0-9]{6}-[a-z_]+')
FINISHED = re.compile(r'== finito con codice (\d+)\s*$')
# Runs the helper and appends its exit code as the last line of the log.
WRAPPER = '"$@"; code=$?; echo; echo "== finito con codice $code"; exit $code'

ACTIONS = {
    # Every route still on the comma, over ssh with this machine's key (Host comma in ~/.ssh/config).
    'fetch': {'label': 'Scarica dal comma', 'group': 'fetch', 'command': ['helpers/fetch_comma_segments.sh']},
    'prepare': {'label': 'Prepara le route nuove', 'group': 'prepare', 'command': ['helpers/1_prepare.sh']},
    'dataset': {'label': 'Importa le revisioni', 'group': 'train',
                'command': ['helpers/3_train.sh', '--dataset-only']},
    'train': {'label': 'Allena da zero', 'group': 'train', 'command': ['helpers/3_train.sh']},
    'train_continue': {'label': 'Allena dai pesi stable', 'group': 'train',
                       'command': ['helpers/3_train.sh', '--continue']},
    # git pull + restart of this server (KillMode=process: the action itself survives the restart).
    # --no-link: the link with the token must not land in the log.
    'update': {'label': 'Aggiorna il codice', 'group': 'update', 'command': ['helpers/update.sh', '--no-link']},
}
# The same work started from a terminal: (group, label, fragment of the command line).
ACTIVITIES = (
    # Also the copy in openpilot.distill: both write the shared folder of the comma videos.
    ('fetch', 'download dal comma', 'fetch_comma_segments.sh'),
    ('prepare', 'preparazione', 'helpers/1_prepare.sh'),
    ('prepare', 'estrazione dei frame', 'nnslr extract-frames'),
    ('prepare', 'pre-annotazione', 'nnslr preannotate'),
    ('train', 'dataset e training', 'helpers/3_train.sh'),
    ('train', 'training su GPU', 'run_v100_baseline.sh'),
    ('update', 'aggiornamento del codice', 'helpers/update.sh'),
    # Shown only: the driving model of openpilot.distill shares the GPU (the training refuses a busy GPU).
    ('other', 'training openpilot.distill', 'supervised.train'),
)
GROUP_NAMES = {'fetch': 'un download dal comma', 'prepare': 'una preparazione',
               'train': 'un import o un training', 'update': 'un aggiornamento del codice'}
# Groups that never run together with a group, besides itself. The download and the preparation
# both write the links of data/raw/routes; the update rewrites the helpers under a running bash
# and restarts this server.
CONFLICTS = {'fetch': ('prepare', 'update'), 'prepare': ('fetch', 'update'), 'train': ('update',),
             'update': ('fetch', 'prepare', 'train')}
# The fetch reads the comma logs (trip length, reverse gear) with openpilot's LogReader; the
# .venv of this repository has it only with the [log-reader] extra and the schema checkouts.
FETCH_PYTHON_ENV = 'COMMA_FETCH_PYTHON'
DISTILL_PYTHON = 'openpilot.distill/.venv/bin/python'


class PanelError(Exception):
    def __init__(self, status: int, reason: str, message: str):
        super().__init__(message)
        self.status = status
        self.reason = reason
        self.message = message


class Panel:
    def __init__(self, root: Path, *, repo: Path = REPO, actions_enabled: bool | None = None):
        self.root = root
        self.repo = repo
        self.logs = root / 'logs' / 'panel'
        if actions_enabled is None:
            actions_enabled = os.environ.get(ACTIONS_ENV, '1') != '0'
        self.actions_enabled = actions_enabled
        # Reentrant: start() holds it while _running_groups() takes it again.
        self._lock = threading.RLock()
        self._processes = {}
        self._status_module = None
        # Progress of each job, by job id: (log size, progress), so a log is read again only when it grew.
        self._progress: dict[str, tuple[int, str, dict]] = {}

    # -- helpers of the checkout -------------------------------------------------------------
    def available(self) -> bool:
        return all((self.repo / name).is_file() for name in
                   ('helpers/route_status.py', 'helpers/1_prepare.sh', 'helpers/3_train.sh'))

    def fetch_python(self) -> str | None:
        """The Python the fetch reads the comma logs with: COMMA_FETCH_PYTHON, else the .venv of
        openpilot.distill next to this checkout. '' when the fetch finds its own (NNSLR_OPENPILOT_ROOT
        set for the [log-reader] extra), None when there is none."""
        if os.environ.get(FETCH_PYTHON_ENV):
            return os.environ[FETCH_PYTHON_ENV]
        distill = self.repo.parent / DISTILL_PYTHON
        if os.access(distill, os.X_OK):
            return str(distill)
        if os.environ.get('NNSLR_OPENPILOT_ROOT'):
            return ''
        return None

    def _route_status(self):
        if self._status_module is None:
            spec = importlib.util.spec_from_file_location('nnslr_panel_route_status',
                                                          self.repo / 'helpers/route_status.py')
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self._status_module = module
        return self._status_module

    # -- what is running -----------------------------------------------------------------------
    def activities(self) -> list[dict]:
        """Helpers and nnslr steps running on this machine, also those started from a terminal."""
        found = {}
        proc = Path('/proc')
        if not proc.is_dir():
            return []
        own = os.getpid()
        for entry in proc.iterdir():
            if not entry.name.isdigit() or int(entry.name) == own:
                continue
            try:
                command = (entry / 'cmdline').read_bytes().replace(b'\0', b' ').decode('utf-8', 'replace')
            except OSError:
                continue
            for group, label, fragment in ACTIVITIES:
                if fragment in command and label not in found:
                    found[label] = {'group': group, 'label': label, 'pid': int(entry.name)}
        return list(found.values())

    def _running_groups(self, activities: list[dict]) -> set[str]:
        groups = {item['group'] for item in activities}
        with self._lock:
            for job_id, process in self._processes.items():
                if process.poll() is None:
                    groups.add(ACTIONS[job_id.split('-', 2)[2]]['group'])
        return groups

    def jobs(self, activities: list[dict] | None = None) -> list[dict]:
        """The last actions started from the panel, newest first."""
        if activities is None:
            activities = self.activities()
        if not self.logs.is_dir():
            return []
        busy = {item['group'] for item in activities}
        names = sorted((p for p in self.logs.glob('*.log') if JOB_ID.fullmatch(p.stem)), reverse=True)
        result = []
        for path in names[:HISTORY]:
            job_id = path.stem
            action = job_id.split('-', 2)[2]
            if action not in ACTIONS:
                continue
            with self._lock:
                process = self._processes.get(job_id)
            code = _exit_code(path)
            if process is not None and process.poll() is None:
                state = 'running'
            elif code is not None:
                state = 'ok' if code == 0 else 'failed'
            elif process is None and ACTIONS[action]['group'] in busy:
                # Started before a restart of this server and still going.
                state = 'running'
            else:
                state = 'interrupted'
            started = datetime.strptime(job_id[:15], '%Y%m%d-%H%M%S')
            result.append({'id': job_id, 'action': action, 'label': ACTIONS[action]['label'],
                           'started': started.strftime('%Y-%m-%d %H:%M:%S'),
                           # The same moment with the server's offset, for the elapsed time in the page.
                           'started_at': started.astimezone().isoformat(timespec='seconds'),
                           'state': state, 'code': code,
                           'stoppable': state == 'running' and self._group(job_id) is not None,
                           'size': path.stat().st_size})
        return result

    # -- the machine ---------------------------------------------------------------------------
    def system(self) -> dict:
        """The machine, for the Stato page (as distill-web's Status): GPU, CPU per core, RAM, disk."""
        info = {'gpu': None, 'disk': None, 'cpus': os.cpu_count(), 'load': None, 'loads': None,
                'cpu': _cpu_usage(), 'memory': _memory()}
        try:
            loads = os.getloadavg()
            info['load'] = round(loads[0], 1)
            info['loads'] = [round(value, 2) for value in loads]
        except OSError:
            pass
        try:
            usage = shutil.disk_usage(self.root)
            info['disk'] = {'free_gb': round(usage.free / 1e9, 1), 'total_gb': round(usage.total / 1e9, 1)}
        except OSError:
            pass
        gpu = _nvidia(['--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw',
                       '--format=csv,noheader,nounits'])
        if gpu:
            fields = [part.strip() for part in gpu.splitlines()[0].split(',')]
            if len(fields) == 6:
                apps = _nvidia(['--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader,nounits']) or ''
                process_list = []
                for line in apps.splitlines():
                    parts = line.split(',')
                    if len(parts) < 3:
                        continue
                    pid = parts[0].strip()
                    process_list.append({'pid': int(pid) if pid.isdigit() else None,
                                         'name': ','.join(parts[1:-1]).strip(), 'memory_mb': _number(parts[-1])})
                # 'processes' stays a count: the training actions refuse a busy GPU with it.
                info['gpu'] = {'name': fields[0], 'utilization': _number(fields[1]),
                               'memory_used_mb': _number(fields[2]), 'memory_total_mb': _number(fields[3]),
                               'temperature': _number(fields[4]), 'power_w': _number(fields[5]),
                               'processes': len(process_list), 'process_list': process_list}
        return info

    # -- everything for the page ---------------------------------------------------------------
    def status(self) -> dict:
        activities = self.activities()
        system = self.system()
        payload = {'available': self.available(), 'actions_enabled': self.actions_enabled and self.available(),
                   'routes': [], 'next_step': '', 'next_action': None, 'reviewed_routes': 0, 'min_routes': 5,
                   'activities': activities, 'jobs': self.jobs(activities), 'system': system, 'actions': {},
                   'preparation': None}
        if payload['available']:
            module = self._route_status()
            rows = module.route_rows(self.root, module.videos_dir())
            payload['routes'] = rows
            payload['preparation'] = self.preparation(rows, activities)
            payload['next_step'] = module.next_step(rows)
            payload['min_routes'] = module.MIN_REVIEWED_ROUTES
            payload['reviewed_routes'] = sum(1 for row in rows if row['confirmed'])
            payload['next_action'] = _next_action(payload['next_step'])
        payload['actions'] = self._action_states(payload, activities)
        for job in payload['jobs']:
            job['progress'] = self.job_progress(job, payload['preparation'])
        self._terminal_training(payload)
        return payload

    def _terminal_training(self, payload: dict) -> None:
        """A training started from a terminal (helpers/3_train.sh) is no job of the panel: its progress comes
        from the log the helper writes itself, the newest data/logs/train-<run>.log."""
        from_panel = any(job['state'] == 'running' and ACTIONS[job['action']]['group'] == 'train'
                         for job in payload['jobs'])
        trainings = [item for item in payload['activities'] if item['group'] == 'train']
        logs = sorted((self.root / 'logs').glob('train-*.log'), key=lambda path: path.stat().st_mtime)
        if from_panel or not trainings or not logs:
            return
        size = logs[-1].stat().st_size
        with logs[-1].open('rb') as log:
            log.seek(max(0, size - PROGRESS_BYTES))
            text = log.read().decode('utf-8', 'replace')
        result = progress('train', 'running', text)
        for item in trainings:
            item['progress'] = result

    def job_progress(self, job: dict, preparation: dict | None) -> dict:
        """Steps, percent and last line of a job, from its log (nnslr_tools.job_progress)."""
        key = (job['size'], job['state'])
        cached = self._progress.get(job['id'])
        # A preparation's percent changes without its log growing: never cached while it runs.
        if cached is not None and cached[:2] == key and not (job['action'] == 'prepare' and job['state'] == 'running'):
            return cached[2]
        try:
            with (self.logs / (job['id'] + '.log')).open('rb') as log:
                log.seek(max(0, job['size'] - PROGRESS_BYTES))
                text = log.read().decode('utf-8', 'replace')
        except OSError:
            text = ''
        result = progress(job['action'], job['state'], text, preparation if job['action'] == 'prepare' else None)
        self._progress[job['id']] = (key[0], key[1], result)
        return result

    def _action_states(self, payload: dict, activities: list[dict]) -> dict:
        running = self._running_groups(activities)
        rows = payload['routes']
        gpu = payload['system']['gpu']
        to_prepare = sum(1 for row in rows if row['state'] in ('collegare', 'estrarre', 'preannotare'))
        confirmed = sum(row['confirmed'] for row in rows)
        states = {}
        for name, action in ACTIONS.items():
            reason = ''
            conflict = _conflicting_group(action['group'], running)
            if not payload['actions_enabled']:
                reason = 'pannello in sola lettura'
            elif conflict is not None:
                reason = 'c\'è già ' + GROUP_NAMES[conflict] + ' in corso'
            elif name == 'update' and not (self.repo / 'helpers/update.sh').is_file():
                reason = 'manca helpers/update.sh in questa installazione'
            elif name == 'update':
                pass
            elif name == 'fetch' and not (self.repo / 'helpers/fetch_comma_segments.sh').is_file():
                reason = 'manca helpers/fetch_comma_segments.sh in questa installazione'
            elif name == 'fetch' and self.fetch_python() is None:
                reason = 'manca il lettore dei log del comma (vedi docs/GUIDA.md)'
            elif name == 'fetch':
                pass
            elif name == 'prepare' and not to_prepare:
                reason = 'nessuna route nuova da preparare'
            elif name != 'prepare' and not confirmed:
                reason = 'nessun fotogramma confermato'
            elif name.startswith('train') and payload['reviewed_routes'] < payload['min_routes']:
                reason = (f'servono {payload["min_routes"]} route revisionate, '
                          f'ora {payload["reviewed_routes"]}')
            elif name.startswith('train') and gpu is not None and gpu['processes']:
                reason = 'la GPU è occupata da un altro processo'
            states[name] = {'label': action['label'], 'allowed': not reason, 'reason': reason}
        return states

    # -- progress of the preparation -----------------------------------------------------------
    def preparation(self, rows: list[dict], activities: list[dict]) -> dict | None:
        """How far the running preparation is, from what it already wrote; None when none runs.

        A route counts half for the extraction (segments with frames.jsonl out of the linked ones)
        and half for the pre-annotation (frames already in the report being written out of the
        extracted ones). The overall percent is over the routes that helpers/1_prepare.sh listed in
        BATCH_FILE when it started; without that file (a run of an older 1_prepare.sh) only the
        routes being worked on are shown.
        """
        running = False
        for item in activities:
            if item['group'] == 'prepare':
                running = True
        if not running:
            return None
        by_route = {}
        for row in rows:
            by_route[row['route']] = row
        result = {'percent': None, 'routes_done': 0, 'routes_total': 0, 'working': []}
        batch = self._batch_routes()
        if batch is not None:
            total = 0
            for route in batch:
                row = by_route.get(route)
                if row is None:
                    continue
                progress = self._route_progress(row)
                total = total + progress['percent']
                result['routes_total'] = result['routes_total'] + 1
                if progress['percent'] == 100:
                    result['routes_done'] = result['routes_done'] + 1
            if result['routes_total']:
                result['percent'] = total // result['routes_total']
        for route in sorted(self._working_routes()):
            row = by_route.get(route)
            if row is not None:
                result['working'].append(self._route_progress(row))
        return result

    def _route_progress(self, row: dict) -> dict:
        route = row['route']
        if row['run_id'] is not None:
            return {'route': route, 'phase': 'pronta', 'done': row['report_frames'],
                    'total': row['report_frames'], 'unit': 'frame', 'percent': 100}
        if row['linked'] == 0:
            return {'route': route, 'phase': 'da collegare', 'done': 0, 'total': 0, 'unit': 'segmenti', 'percent': 0}
        if row['extracted'] < row['linked']:
            return {'route': route, 'phase': 'frame', 'done': row['extracted'], 'total': row['linked'],
                    'unit': 'segmenti', 'percent': 50 * row['extracted'] // row['linked']}
        done = self._preannotated_frames(route)
        total = row['frames']
        percent = 50
        if total:
            percent = 50 + 50 * min(done, total) // total
        # The report counts only once it is moved in place (run_id above).
        if percent > 99:
            percent = 99
        return {'route': route, 'phase': 'cartelli', 'done': done, 'total': total, 'unit': 'frame', 'percent': percent}

    def _batch_routes(self) -> list[str] | None:
        try:
            routes = json.loads((self.root / BATCH_FILE).read_text()).get('routes')
        except (OSError, ValueError, AttributeError):
            return None
        if not isinstance(routes, list):
            return None
        result = []
        for route in routes:
            if isinstance(route, str):
                result.append(route)
        return result

    def _preannotated_frames(self, route: str) -> int:
        """Frames in the report the pre-annotation is writing (the newest .pending-<run id>)."""
        base = self.root / 'derived/preannotations' / route
        if not base.is_dir():
            return 0
        pending = sorted(base.glob('.pending-*'))
        if not pending:
            return 0
        return _line_count(pending[-1] / 'preannotations.jsonl')

    def _working_routes(self) -> set[str]:
        """Routes an nnslr extract-frames or nnslr preannotate is working on right now."""
        found = set()
        proc = Path('/proc')
        if not proc.is_dir():
            return found
        for entry in proc.iterdir():
            if not entry.name.isdigit():
                continue
            try:
                arguments = (entry / 'cmdline').read_bytes().decode('utf-8', 'replace').split('\0')
            except OSError:
                continue
            if 'extract-frames' not in arguments and 'preannotate' not in arguments:
                continue
            if '--route' not in arguments:
                continue
            position = arguments.index('--route')
            if position + 1 < len(arguments):
                found.add(arguments[position + 1])
        return found

    # -- actions -------------------------------------------------------------------------------
    def start(self, action: str) -> dict:
        if action not in ACTIONS:
            raise PanelError(404, 'unknown_action', 'Azione sconosciuta.')
        if not self.actions_enabled or not self.available():
            raise PanelError(403, 'actions_disabled', 'Il pannello è in sola lettura su questo server.')
        group = ACTIONS[action]['group']
        if action == 'update' and not (self.repo / 'helpers/update.sh').is_file():
            raise PanelError(403, 'actions_disabled', 'Manca helpers/update.sh in questa installazione.')
        fetch_python = None
        if action == 'fetch':
            if not (self.repo / 'helpers/fetch_comma_segments.sh').is_file():
                raise PanelError(403, 'actions_disabled', 'Manca helpers/fetch_comma_segments.sh in questa installazione.')
            fetch_python = self.fetch_python()
            if fetch_python is None:
                raise PanelError(403, 'actions_disabled', 'Manca il lettore dei log del comma (vedi docs/GUIDA.md).')
        with self._lock:
            conflict = _conflicting_group(group, self._running_groups(self.activities()))
            if conflict is not None:
                raise PanelError(409, 'busy', 'C\'è già ' + GROUP_NAMES[conflict] + ' in corso: aspetta che finisca.')
            self.logs.mkdir(parents=True, exist_ok=True)
            job_id = datetime.now().strftime('%Y%m%d-%H%M%S') + '-' + action
            path = self.logs / (job_id + '.log')
            if path.exists():
                raise PanelError(409, 'busy', 'Un\'azione è appena partita: riprova tra un secondo.')
            command = list(ACTIONS[action]['command'])
            if action == 'prepare':
                command.append('--jobs=' + os.environ.get(PREPARE_JOBS_ENV, '2'))
            environment = dict(os.environ, NNSLR_DATA_ROOT=str(self.root))
            if fetch_python:
                environment[FETCH_PYTHON_ENV] = fetch_python
            with path.open('wb') as log:
                log.write(('== ' + ' '.join(command) + '\n').encode())
                log.flush()
                # A new session: the action outlives this server and can be stopped as a group.
                process = subprocess.Popen(['bash', '-c', WRAPPER, 'panel', *command], cwd=self.repo,
                                           stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                           env=environment, start_new_session=True)
            self._processes[job_id] = process
            # The process group of the action (a new session: its pid), so Ferma works also after a
            # restart of this server, which forgets self._processes.
            path.with_suffix('.pid').write_text(str(process.pid) + '\n')
        # Reaps the process as soon as it ends, so it never stays a zombie.
        threading.Thread(target=process.wait, daemon=True).start()
        return {'id': job_id, 'action': action, 'label': ACTIONS[action]['label']}

    def _group(self, job_id: str) -> int | None:
        """The process group of an action still running, also one started before a restart of this
        server: the pid written at start, if that process is still the wrapper of this action."""
        if not JOB_ID.fullmatch(job_id):
            return None
        with self._lock:
            process = self._processes.get(job_id)
        if process is not None:
            return process.pid if process.poll() is None else None
        try:
            pid = int((self.logs / (job_id + '.pid')).read_text().strip())
            command = (Path('/proc') / str(pid) / 'cmdline').read_bytes().split(b'\0')
        except (OSError, ValueError):
            return None
        # The pid may have been reused by another program: it must be the wrapper of this action.
        action = job_id.split('-', 2)[2]
        wanted = [part.encode() for part in ACTIONS.get(action, {}).get('command', [])]
        if not wanted or command[:4] != [b'bash', b'-c', WRAPPER.encode(), b'panel'] or command[4:4 + len(wanted)] != wanted:
            return None
        return pid

    def stop(self, job_id: str) -> dict:
        group = self._group(job_id)
        if group is None:
            if JOB_ID.fullmatch(job_id) and (self.logs / (job_id + '.log')).is_file():
                raise PanelError(409, 'not_running', 'L\'azione è già finita.')
            raise PanelError(404, 'unknown_job', 'Azione non avviata dal pannello (fermala dal terminale).')
        try:
            os.killpg(group, signal.SIGTERM)
        except ProcessLookupError:
            pass
        return {'id': job_id, 'stopping': True}

    def log(self, job_id: str, offset: int) -> dict:
        if not JOB_ID.fullmatch(job_id):
            raise PanelError(404, 'unknown_job', 'Azione sconosciuta.')
        path = self.logs / (job_id + '.log')
        if not path.is_file():
            raise PanelError(404, 'unknown_job', 'Azione sconosciuta.')
        size = path.stat().st_size
        if offset < 0 or offset > size:
            offset = 0
        # The first call shows only the end of a long log.
        if offset == 0 and size > LOG_CHUNK_BYTES:
            offset = size - LOG_CHUNK_BYTES
        with path.open('rb') as source:
            source.seek(offset)
            data = source.read(LOG_CHUNK_BYTES)
        return {'id': job_id, 'text': data.decode('utf-8', 'replace'), 'offset': offset + len(data), 'size': size}


def _conflicting_group(group: str, running: set[str]) -> str | None:
    """The running group that keeps an action of `group` from starting, or None.

    One action per group at a time, and never together with the groups of CONFLICTS.
    """
    for other in (group, *CONFLICTS.get(group, ())):
        if other in running:
            return other
    return None


def _line_count(path: Path) -> int:
    count = 0
    try:
        with path.open('rb') as handle:
            for line in handle:
                if line.strip():
                    count = count + 1
    except OSError:
        return 0
    return count


def _exit_code(path: Path) -> int | None:
    try:
        with path.open('rb') as source:
            source.seek(max(0, path.stat().st_size - 200))
            tail = source.read().decode('utf-8', 'replace')
    except OSError:
        return None
    match = FINISHED.search(tail)
    return int(match.group(1)) if match else None


# The last /proc/stat sample: the CPU use is the busy share of the time between two status calls.
_cpu_sample = {'times': None}
_cpu_lock = threading.Lock()


def _proc_stat(text: str) -> list[tuple[int, int]]:
    """(busy, total) jiffies of the whole CPU (first) and of every core."""
    rows = []
    for line in text.splitlines():
        if not line.startswith('cpu'):
            continue
        try:
            numbers = [int(value) for value in line.split()[1:]]
        except ValueError:
            continue
        if len(numbers) < 4:
            continue
        # idle + iowait are not busy; guest time is already inside user and nice.
        idle = numbers[3] + (numbers[4] if len(numbers) > 4 else 0)
        total = sum(numbers[:8])
        rows.append((total - idle, total))
    return rows


def _cpu_usage() -> dict | None:
    """{'total': %, 'cores': [%, ...]} since the previous call; None on the first call or without /proc."""
    try:
        current = _proc_stat(Path('/proc/stat').read_text())
    except OSError:
        return None
    with _cpu_lock:
        previous = _cpu_sample['times']
        _cpu_sample['times'] = current
    if previous is None or len(previous) != len(current):
        # The first call after a start: a short second sample, so the page shows a value at once.
        time.sleep(0.25)
        return _cpu_usage()
    percents = []
    for (busy, total), (old_busy, old_total) in zip(current, previous):
        elapsed = total - old_total
        percents.append(round(max(0.0, min(100.0, (busy - old_busy) * 100 / elapsed)), 1) if elapsed > 0 else 0.0)
    return {'total': percents[0], 'cores': percents[1:]}


def _memory() -> dict | None:
    """RAM from /proc/meminfo (inside the LXC it is the container's own memory)."""
    try:
        text = Path('/proc/meminfo').read_text()
    except OSError:
        return None
    values = {}
    for line in text.splitlines():
        name, separator, rest = line.partition(':')
        parts = rest.split()
        if separator and parts and parts[0].isdigit():
            values[name.strip()] = int(parts[0]) * 1024
    if 'MemTotal' not in values or 'MemAvailable' not in values:
        return None
    total = values['MemTotal']
    available = values['MemAvailable']
    return {'total_gb': round(total / 1e9, 1), 'available_gb': round(available / 1e9, 1),
            'used_percent': round((total - available) * 100 / total, 1)}


def _nvidia(arguments: list[str]) -> str | None:
    try:
        result = subprocess.run(['nvidia-smi', *arguments], capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout if result.returncode == 0 else None


def _number(text: str):
    try:
        return int(float(text))
    except ValueError:
        return None


def _next_action(next_step: str) -> str | None:
    if '1_prepare' in next_step:
        return 'prepare'
    if '2_review' in next_step:
        return 'review'
    if '3_train' in next_step:
        return 'train'
    if 'fetch_comma_segments' in next_step:
        return 'fetch'
    return None
# [control-panel] - END
