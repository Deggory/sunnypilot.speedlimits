# [control-panel] - START
"""Control panel of the review site: the state of every route, the machine, and buttons that run
the easy helpers (helpers/1_prepare.sh, helpers/3_train.sh, helpers/update.sh) in the background
with a live log.

Only the fixed actions of ACTIONS can run, never a command taken from a request. They are the
helpers of the checkout this package runs from (editable install); without them (wheel
install) the panel is read-only. NNSLR_PANEL_ACTIONS=0 also makes it read-only, e.g. for a
review service that other people use. The output of every action goes to
<data-root>/logs/panel/<job id>.log, and its last line is the exit code, so the history
survives a restart of the server.
"""
from __future__ import annotations

import importlib.util
import os
import re
import shutil
import signal
import subprocess
import threading
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ACTIONS_ENV = 'NNSLR_PANEL_ACTIONS'
PREPARE_JOBS_ENV = 'NNSLR_PANEL_PREPARE_JOBS'
LOG_CHUNK_BYTES = 64 * 1024
HISTORY = 12
JOB_ID = re.compile(r'[0-9]{8}-[0-9]{6}-[a-z_]+')
FINISHED = re.compile(r'== finito con codice (\d+)\s*$')
# Runs the helper and appends its exit code as the last line of the log.
WRAPPER = '"$@"; code=$?; echo; echo "== finito con codice $code"; exit $code'

ACTIONS = {
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
    ('prepare', 'preparazione', 'helpers/1_prepare.sh'),
    ('prepare', 'estrazione dei frame', 'nnslr extract-frames'),
    ('prepare', 'pre-annotazione', 'nnslr preannotate'),
    ('train', 'dataset e training', 'helpers/3_train.sh'),
    ('train', 'training su GPU', 'run_v100_baseline.sh'),
    ('update', 'aggiornamento del codice', 'helpers/update.sh'),
    # Shown only: the driving model of openpilot.distill shares the GPU (the training refuses a busy GPU).
    ('other', 'training openpilot.distill', 'supervised.train'),
)
GROUP_NAMES = {'prepare': 'una preparazione', 'train': 'un import o un training',
               'update': 'un aggiornamento del codice'}


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

    # -- helpers of the checkout -------------------------------------------------------------
    def available(self) -> bool:
        return all((self.repo / name).is_file() for name in
                   ('helpers/route_status.py', 'helpers/1_prepare.sh', 'helpers/3_train.sh'))

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
                           'started': started.strftime('%Y-%m-%d %H:%M:%S'), 'state': state, 'code': code,
                           'stoppable': process is not None and state == 'running',
                           'size': path.stat().st_size})
        return result

    # -- the machine ---------------------------------------------------------------------------
    def system(self) -> dict:
        info = {'gpu': None, 'disk': None, 'cpus': os.cpu_count(), 'load': None}
        try:
            info['load'] = round(os.getloadavg()[0], 1)
        except OSError:
            pass
        try:
            usage = shutil.disk_usage(self.root)
            info['disk'] = {'free_gb': round(usage.free / 1e9, 1), 'total_gb': round(usage.total / 1e9, 1)}
        except OSError:
            pass
        gpu = _nvidia(['--query-gpu=name,utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'])
        if gpu:
            fields = [part.strip() for part in gpu.splitlines()[0].split(',')]
            if len(fields) == 4:
                apps = _nvidia(['--query-compute-apps=pid', '--format=csv,noheader']) or ''
                info['gpu'] = {'name': fields[0], 'utilization': _number(fields[1]),
                               'memory_used_mb': _number(fields[2]), 'memory_total_mb': _number(fields[3]),
                               'processes': len([line for line in apps.splitlines() if line.strip()])}
        return info

    # -- everything for the page ---------------------------------------------------------------
    def status(self) -> dict:
        activities = self.activities()
        system = self.system()
        payload = {'available': self.available(), 'actions_enabled': self.actions_enabled and self.available(),
                   'routes': [], 'next_step': '', 'next_action': None, 'reviewed_routes': 0, 'min_routes': 5,
                   'activities': activities, 'jobs': self.jobs(activities), 'system': system, 'actions': {}}
        if payload['available']:
            module = self._route_status()
            rows = module.route_rows(self.root, module.videos_dir())
            payload['routes'] = rows
            payload['next_step'] = module.next_step(rows)
            payload['min_routes'] = module.MIN_REVIEWED_ROUTES
            payload['reviewed_routes'] = sum(1 for row in rows if row['confirmed'])
            payload['next_action'] = _next_action(payload['next_step'])
        payload['actions'] = self._action_states(payload, activities)
        return payload

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

    # -- actions -------------------------------------------------------------------------------
    def start(self, action: str) -> dict:
        if action not in ACTIONS:
            raise PanelError(404, 'unknown_action', 'Azione sconosciuta.')
        if not self.actions_enabled or not self.available():
            raise PanelError(403, 'actions_disabled', 'Il pannello è in sola lettura su questo server.')
        group = ACTIONS[action]['group']
        if action == 'update' and not (self.repo / 'helpers/update.sh').is_file():
            raise PanelError(403, 'actions_disabled', 'Manca helpers/update.sh in questa installazione.')
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
            with path.open('wb') as log:
                log.write(('== ' + ' '.join(command) + '\n').encode())
                log.flush()
                # A new session: the action outlives this server and can be stopped as a group.
                process = subprocess.Popen(['bash', '-c', WRAPPER, 'panel', *command], cwd=self.repo,
                                           stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                           env=environment, start_new_session=True)
            self._processes[job_id] = process
        # Reaps the process as soon as it ends, so it never stays a zombie.
        threading.Thread(target=process.wait, daemon=True).start()
        return {'id': job_id, 'action': action, 'label': ACTIONS[action]['label']}

    def stop(self, job_id: str) -> dict:
        with self._lock:
            process = self._processes.get(job_id)
        if process is None:
            raise PanelError(404, 'unknown_job', 'Azione non avviata da questo server (fermala dal terminale).')
        if process.poll() is not None:
            raise PanelError(409, 'not_running', 'L\'azione è già finita.')
        try:
            os.killpg(process.pid, signal.SIGTERM)
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

    One action per group at a time. The update rewrites the helpers under a running bash and
    restarts this server, so it never runs together with a preparation or a training.
    """
    if group in running:
        return group
    if group == 'update':
        for other in ('prepare', 'train'):
            if other in running:
                return other
    elif 'update' in running:
        return 'update'
    return None


def _exit_code(path: Path) -> int | None:
    try:
        with path.open('rb') as source:
            source.seek(max(0, path.stat().st_size - 200))
            tail = source.read().decode('utf-8', 'replace')
    except OSError:
        return None
    match = FINISHED.search(tail)
    return int(match.group(1)) if match else None


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
