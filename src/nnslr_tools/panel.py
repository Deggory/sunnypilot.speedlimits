# [control-panel] - START
"""Control panel of the review site: the state of every route, the machine, and buttons that run
the easy helpers (helpers/fetch_comma_segments.sh, helpers/1_prepare.sh, helpers/3_train.sh,
helpers/update.sh; scripts/run_detector_sizes.sh for the cutter and scripts/run_classifier.sh for the
classifier with "not a sign"; helpers/ai_review.sh and helpers/measure_ai_reviewer.sh for the AI reviewer of
Settings) in the background with a live log.

Only the fixed actions of ACTIONS can run, never a command taken from a request. They are the
helpers of the checkout this package runs from (editable install); without them (wheel
install) the panel is read-only. NNSLR_PANEL_ACTIONS=0 also makes it read-only, e.g. for a
review service that other people use. The output of every action goes to
<data-root>/logs/panel/<job id>.log, and its last line is the exit code, so the history
survives a restart of the server.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import shutil
import signal
import stat
import subprocess
import threading
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from nnslr_tools.job_progress import last_line, progress, with_step_progress
from nnslr_tools.model_roles import ModelChoiceError, ModelIndex
from nnslr_tools.runs import RunsIndex, WatchError

REPO = Path(__file__).resolve().parents[2]
ACTIONS_ENV = 'NNSLR_PANEL_ACTIONS'
PREPARE_JOBS_ENV = 'NNSLR_PANEL_PREPARE_JOBS'
LOG_CHUNK_BYTES = 64 * 1024
# The progress of a job is read from at most the last 4 MB of its log.
PROGRESS_BYTES = 4 * 1024 * 1024
# Written by helpers/1_prepare.sh when it starts: the routes of that run, for the progress.
BATCH_FILE = 'logs/prepare/batch.json'
HISTORY = 12
# A download announced by pip in the GPU setup log.
DOWNLOADING = re.compile(r'Downloading (\S+?)(?:\.whl)? \(([0-9.]+) (kB|MB|GB)\)')
RUN_NAME = re.compile(r'[A-Za-z0-9._-]+')
STABLE_CHECK_SECONDS = 600
JOB_ID = re.compile(r'[0-9]{8}-[0-9]{6}-[a-z_]+')
# The last line of a log; "finito con codice" in the logs written before it was in English.
FINISHED = re.compile(r'== (?:finished with code|finito con codice) (\d+)\s*$')
# Runs the helper and appends its exit code as the last line of the log.
WRAPPER = '"$@"; code=$?; echo; echo "== finished with code $code"; exit $code'
# The wrapper of the actions started before the footer was in English: still running ones are still ours (_group).
OLD_WRAPPERS = ('"$@"; code=$?; echo; echo "== finito con codice $code"; exit $code',)
# A training run's name, as helpers/3_train.sh writes it: "== d. training <run> (<mode>) -> ...".
RUN_NAME = re.compile(r'[A-Za-z0-9][A-Za-z0-9._-]{0,99}')
RUN_LINE = re.compile(r'^== d\. training (\S+) \(', re.M)
# The classifier's and the cutter's trainings: the panel names their run after the command ("== scripts/run_classifier.sh
# <run>", "== scripts/run_detector_sizes.sh <run> <size>", the first line of the log), and the script says it again near
# its top ("run:          <run>").
CLASSIFIER_SCRIPT = 'scripts/run_classifier.sh'
CUTTER_SCRIPT = 'scripts/run_detector_sizes.sh'
# The decider: experiment B, its three variants (the crop alone; with position and speed; with the scene), on a run of
# its own (start() adds <time>-decider).
DECIDER_SCRIPT = 'scripts/run_experiment_b.sh'
# Our three networks in use as one package (ONNX, model card), and that package on Hugging Face.
NETWORKS_SCRIPT = 'scripts/publish_networks.sh'
NETWORKS_REPO = os.environ.get('HF_REPO', 'cristianku/sunnypilot-speedlimits-models')
NETWORK_PACKAGES_SHOWN = 5
RUN_ARGUMENT_LINE = re.compile(r'^run:\s+(\S+)\s*$', re.M)
RUN_LINES = {
    'train_classifier': (re.compile(r'^== scripts/run_classifier\.sh (\S+)\s*$', re.M), RUN_ARGUMENT_LINE),
    'train_cutter': (re.compile(r'^== scripts/run_detector_sizes\.sh (\S+)(?: \S+)*\s*$', re.M), RUN_ARGUMENT_LINE),
    'train_decider': (re.compile(r'^== scripts/run_experiment_b\.sh (\S+)\s*$', re.M), RUN_ARGUMENT_LINE),
}
# The cutter's training: one input size of the detector (scripts/run_detector_sizes.sh NAME SIZE), and the boxes it
# learns to find (NNSLR_CUTTER_FAMILIES): every sign (the speed signs and the other signs) or the speed signs only.
CUTTER_SIZES = ('640x360', '960x544', '1344x760')
CUTTER_SIZE = '1344x760'
CUTTER_FAMILIES = ('all', 'speed')
CUTTER_FAMILY = 'all'
# The options an action takes, besides a training's start and the run to resume (checked by Panel.start).
# The GPU memory Qwen3-VL-8B takes in float16 (scripts/setup_vlm.sh: 17.5 GB), with a margin for one frame.
QWEN_GPU_MB = 18 * 1024
ACTION_OPTIONS = {'train_cutter': ('size', 'families'), 'prepare': ('proposer',), 'prepare_again': ('proposer',),
                  'measure_proposer': ('proposer',), 'networks_publish': ('package',),
                  'ai_review': ('route', 'provider', 'save', 'limit'), 'measure_ai_reviewer': ('provider', 'limit')}
# [ai-reviewer] - START
# The AI actions and their options (strings; checked by Panel._ai_arguments). The provider is an id of Settings
# (nnslr_tools.llm_providers): its key never goes into a command line, an environment or a log, the job reads it.
AI_ACTIONS = ('ai_review', 'measure_ai_reviewer')
AI_OPTIONS = ('route', 'provider', 'save', 'limit')
PROVIDER_ID = re.compile(r'[a-z][a-z0-9-]{0,31}')
MAX_AI_LIMIT = 5000
# [ai-reviewer] - END
# Who proposes the signs of a route (nnslr preannotate --proposer): the choice of the Prepare tab, kept on the server
# (PROPOSER_FILE in the data root) once chosen; before that, the one measured best (Panel.proposer_choice).
PROPOSERS = ('vietnam', 'ours', 'qwen')
PROPOSER_FILE = 'web/proposer.json'
# The proposer when none is measured yet: the helpers' own default (helpers/1_prepare.sh).
DEFAULT_PROPOSER = 'vietnam'
# The actions that take a proposer, and the two that prepare routes (their progress is the preparation's).
PROPOSER_ACTIONS = ('prepare', 'prepare_again', 'measure_proposer')
PREPARING = ('prepare', 'prepare_again')
# A proposer in the log of a job started before the panel kept its options (<job id>.json): "Measuring the proposer
# qwen" (helpers/measure_proposer.sh), "proposing the signs (again) (qwen)" (helpers/1_prepare.sh).
PROPOSER_LINE = re.compile(r'(?:Measuring the proposer |(?:proposing the signs(?: again)?|propongo di nuovo i cartelli) \()'
                           r'(vietnam|ours|qwen)\b')


def _run_of(log: Path, action: str = 'train') -> str | None:
    """The run a training job wrote to, from the start of its log (the line comes after the import; for the
    classifier's training, the first line)."""
    try:
        with log.open('rb') as handle:
            head = handle.read(65536).decode('utf-8', 'replace')
    except OSError:
        return None
    for pattern in RUN_LINES.get(action, (RUN_LINE,)):
        match = pattern.search(head)
        if match and RUN_NAME.fullmatch(match.group(1)):
            return match.group(1)
    return None

ACTIONS = {
    # Every route still on the comma, over ssh with this machine's key (Host comma in ~/.ssh/config).
    'fetch': {'label': 'Download from the comma', 'group': 'fetch', 'command': ['helpers/fetch_comma_segments.sh']},
    'prepare': {'label': 'Prepare new drives', 'group': 'prepare', 'command': ['helpers/1_prepare.sh']},
    # The routes nobody has reviewed a frame of, proposed again (after choosing another proposer).
    'prepare_again': {'label': 'Propose again', 'group': 'prepare', 'command': ['helpers/1_prepare.sh', '--again']},
    'dataset': {'label': 'Import reviews', 'group': 'train',
                'command': ['helpers/3_train.sh', '--dataset-only']},
    # The whole pipeline: import, split, the detector and the reader, evaluation, export, bundle.
    'train': {'label': 'Whole pipeline', 'group': 'train', 'command': ['helpers/3_train.sh']},
    'train_continue': {'label': 'Train from the stable model', 'group': 'train',
                       'command': ['helpers/3_train.sh', '--continue']},
    # The steps a stopped training still misses, on the same run ({'run': name}: helpers/3_train.sh --resume).
    'train_resume': {'label': 'Resume the training', 'group': 'train', 'command': ['helpers/3_train.sh']},
    # The classifier with "not a sign": the false boxes of the cutter in use mined, the classifier trained on them,
    # measured alone and in the chain, on a run of its own (its name is added by start(): <time>-classifier).
    'train_classifier': {'label': 'Train the classifier', 'group': 'train', 'command': [CLASSIFIER_SCRIPT]},
    # The cutter: one detector of the size chosen, trained and measured on test and validation, on a run of its own
    # (start() adds <time>-cutter and the size; {'size', 'families'}).
    'train_cutter': {'label': 'Train the cutter', 'group': 'train', 'command': [CUTTER_SCRIPT]},
    'train_decider': {'label': 'Train the decider', 'group': 'train', 'command': [DECIDER_SCRIPT]},
    # The cutter, classifier and decider in use exported as one package (start() names it); nothing is uploaded.
    'networks_prepare': {'label': 'Prepare the package', 'group': 'publish', 'command': [NETWORKS_SCRIPT]},
    # A package prepared here ({'package': name}) uploaded to Hugging Face, public: other people download it.
    'networks_publish': {'label': 'Publish on Hugging Face', 'group': 'publish', 'command': [NETWORKS_SCRIPT, '--upload']},
    # A proposer of Prepare measured on the test frames (the Proposers view of Evaluate); in the train group: the GPU.
    'measure_proposer': {'label': 'Measure a proposer', 'group': 'train', 'command': ['helpers/measure_proposer.sh']},
    # Qwen3-VL as the third proposer of Prepare: its packages, its weights (17.5 GB, what is already there is kept after
    # its sha256 is checked) and a test on the GPU; in the train group: the GPU.
    'setup_vlm': {'label': 'Install Qwen3-VL', 'group': 'train', 'command': ['scripts/setup_vlm.sh']},
    # git pull + restart of this server (KillMode=process: the action itself survives the restart).
    # --no-link: the link with the token must not land in the log.
    'update': {'label': 'Update the site', 'group': 'update', 'command': ['helpers/update.sh', '--no-link']},
    # [ai-reviewer] - START
    # The frames of a route nobody confirmed, reviewed by an AI provider of Settings: a trial ({'save': 'no'}, the
    # suggestions go to data/ai_reviews/<route>/) or saved as AI reviews ({'save': 'yes'}: final, the next import
    # trains on them). Frames a person confirmed are never touched.
    'ai_review': {'label': 'Review a route with AI', 'group': 'ai', 'command': ['helpers/ai_review.sh']},
    # An AI provider measured on the frames people reviewed (the AI reviewers view of Evaluate); nothing is saved.
    'measure_ai_reviewer': {'label': 'Measure an AI reviewer', 'group': 'ai',
                            'command': ['helpers/measure_ai_reviewer.sh']},
    # [ai-reviewer] - END
}
# The same work started from a terminal: (group, label, fragment of the command line).
ACTIVITIES = (
    # Also the copy in openpilot.distill: both write the shared folder of the comma videos.
    ('fetch', 'download from the comma', 'fetch_comma_segments.sh'),
    ('prepare', 'preparation', 'helpers/1_prepare.sh'),
    ('prepare', 'frame extraction', 'nnslr extract-frames'),
    ('prepare', 'pre-annotation', 'nnslr preannotate'),
    ('train', 'dataset and training', 'helpers/3_train.sh'),
    ('train', 'GPU training', 'run_v100_baseline.sh'),
    ('train', 'classifier training', 'run_classifier.sh'),
    ('train', 'cutter training', 'run_detector_sizes.sh'),
    ('update', 'code update', 'helpers/update.sh'),
    ('publish', 'package of the networks', 'publish_networks.sh'),
    # [ai-reviewer] - START
    ('ai', 'AI review', 'nnslr ai-review'),
    ('ai', 'AI reviewer measure', 'nnslr measure-ai-reviewer'),
    # [ai-reviewer] - END
    # Shown only: the driving model of openpilot.distill shares the GPU (the training refuses a busy GPU).
    ('other', 'openpilot.distill training', 'supervised.train'),
)
# The groups of this checkout's own helpers (see Panel._ours).
OWN_GROUPS = ('prepare', 'train', 'update', 'ai', 'publish')
GROUP_NAMES = {'fetch': 'a download from the comma', 'prepare': 'a preparation',
               'train': 'an import or a training', 'update': 'a code update', 'ai': 'an AI review',
               'publish': 'a package of the networks'}
# What runs, in the reason of an action that must wait ("A proposer measurement is running"): the job of the site by
# its action, the work from a terminal by its label (ACTIVITIES); GROUP_NAMES when neither is known.
RUNNING_NAMES = {'fetch': 'a download from the comma', 'prepare': 'a preparation', 'prepare_again': 'a preparation',
                 'measure_proposer': 'a proposer measurement', 'setup_vlm': 'the install of Qwen3-VL',
                 'dataset': 'an import of the reviews', 'train': 'a training', 'train_continue': 'a training',
                 'train_resume': 'a training', 'train_classifier': 'a classifier training',
                 'train_cutter': 'a cutter training', 'train_decider': 'a decider training', 'update': 'a code update',
                 'networks_prepare': 'the preparation of a package', 'networks_publish': 'a publication',
                 'ai_review': 'an AI review',
                 'measure_ai_reviewer': 'an AI reviewer measurement'}
ACTIVITY_NAMES = {'download from the comma': 'a download from the comma', 'preparation': 'a preparation',
                  'frame extraction': 'a preparation', 'pre-annotation': 'a preparation',
                  'dataset and training': 'an import or a training', 'GPU training': 'a training',
                  'classifier training': 'a classifier training', 'cutter training': 'a cutter training',
                  'code update': 'a code update', 'AI review': 'an AI review',
                  'package of the networks': 'a package of the networks',
                  'AI reviewer measure': 'an AI reviewer measurement'}
# Groups that never run together with a group, besides itself. The download and the preparation
# both write the links of data/raw/routes; the update rewrites the helpers under a running bash
# and restarts this server; a preparation replaces the report an AI review is working on.
CONFLICTS = {'fetch': ('prepare', 'update'), 'prepare': ('fetch', 'update', 'ai'), 'train': ('update',),
             'update': ('fetch', 'prepare', 'train', 'ai', 'publish'), 'ai': ('prepare', 'update'),
             'publish': ('update',)}
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
        self._dataset: tuple[tuple, dict | None] | None = None
        # (time of the last check, published?): None until the first answer.
        self._stable: tuple[float, bool | None] = (-float('inf'), None)
        self._stable_checking = False
        # The training runs of <root>/runs, for the Evaluate tab (nnslr_tools.runs).
        self.runs_index = RunsIndex(root)
        # The same runs by role (cutter, classifier, decider) and the model in use, for the Models tab.
        self.models_index = ModelIndex(self.runs_index)

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
                if fragment in command and label not in found and self._ours(entry, group):
                    found[label] = {'group': group, 'label': label, 'pid': int(entry.name)}
        return list(found.values())

    def _ours(self, process: Path, group: str) -> bool:
        """The helpers of this checkout run from it (the panel starts them there, the guide says so for a
        terminal): the same helper of another checkout is not ours. A download and other programs (the
        copy in openpilot.distill, its training) count wherever they run."""
        if group not in OWN_GROUPS:
            return True
        try:
            cwd = Path(os.readlink(process / 'cwd'))
        except OSError:
            return True
        return cwd.is_relative_to(self.repo.resolve())

    def _running_groups(self, activities: list[dict]) -> set[str]:
        groups = {item['group'] for item in activities}
        with self._lock:
            for job_id, process in self._processes.items():
                if process.poll() is None:
                    groups.add(ACTIONS[job_id.split('-', 2)[2]]['group'])
        return groups

    def _busy_reason(self, group: str, activities: list[dict]) -> str:
        """Why an action waits for `group`, in words: "A proposer measurement is running" (the running job of the site
        in that group, else the work from a terminal, else the group)."""
        name = None
        with self._lock:
            for job_id, process in self._processes.items():
                action = job_id.split('-', 2)[2]
                if process.poll() is None and ACTIONS[action]['group'] == group:
                    name = RUNNING_NAMES.get(action)
                    break
        if name is None:
            name = next((ACTIVITY_NAMES[item['label']] for item in activities
                         if item['group'] == group and item['label'] in ACTIVITY_NAMES), None)
        name = name or GROUP_NAMES[group]
        return name[0].upper() + name[1:] + ' is running'

    def running(self) -> dict:
        """What runs now, for the dots of the busy tabs on the pages outside the panel (GET /api/panel/system): the
        running jobs of the site (id, action, label, state) and the work started from a terminal."""
        activities = self.activities()
        jobs = [{key: job[key] for key in ('id', 'action', 'label', 'state')}
                for job in self.jobs(activities) if job['state'] == 'running']
        return {'jobs': jobs, 'activities': activities}

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
            elif process is None and (self._group(job_id) is not None or (
                    ACTIONS[action]['group'] in busy and not (self.logs / (job_id + '.pid')).exists())):
                # Started before a restart of this server and still going: its wrapper is still alive (the pid the
                # panel wrote when it started it), or, for a job from before those pids, its group is busy. A job
                # with a pid that is gone was interrupted, whatever else of its group runs now: a measure that died
                # at 416 of 604 frames showed as running for hours while a cutter trained.
                state = 'running'
            else:
                state = 'interrupted'
            started = datetime.strptime(job_id[:15], '%Y%m%d-%H%M%S')
            info = path.stat()
            # Who proposes (a preparation) or is measured: "Measure Qwen3-VL" rather than "Measure a proposer".
            proposer = _proposer_of(path) if action in PROPOSER_ACTIONS else None
            label = ACTIONS[action]['label']
            if action == 'measure_proposer' and proposer is not None:
                from nnslr_tools.preannotate import PROPOSER_NAMES
                label = 'Measure ' + PROPOSER_NAMES.get(proposer, proposer)
            result.append({'id': job_id, 'action': action, 'label': label,
                           'run': _run_of(path, action) if ACTIONS[action]['group'] == 'train' else None,
                           'proposer': proposer,
                           'started': started.strftime('%Y-%m-%d %H:%M:%S'),
                           # The same moment with the server's offset, for the elapsed time in the page.
                           'started_at': started.astimezone().isoformat(timespec='seconds'),
                           # When it ended: the last write of its log (the wrapper's exit code, or where it stopped).
                           'ended_at': None if state == 'running'
                           else datetime.fromtimestamp(info.st_mtime).astimezone().isoformat(timespec='seconds'),
                           'state': state, 'code': code,
                           'stoppable': state == 'running' and self._group(job_id) is not None,
                           'size': info.st_size})
        return result

    # -- the machine ---------------------------------------------------------------------------
    def comma(self) -> dict:
        """The comma pill of the top bar: reachable or not (nnslr_tools.comma_link), and whether a download from it
        runs now (a job of the panel or helpers/fetch_comma_segments.sh from a terminal)."""
        from nnslr_tools import comma_link
        return {**comma_link.state(self.root), 'downloading': 'fetch' in self._running_groups(self.activities())}

    def test_comma(self) -> dict:
        """The comma checked now (its Test button)."""
        from nnslr_tools import comma_link
        return {**comma_link.check_now(self.root), 'downloading': 'fetch' in self._running_groups(self.activities())}

    def set_comma_address(self, body) -> dict:
        """The address of the comma from its panel ({'address': '192.168.1.20'}; '' for the ssh host "comma")."""
        from nnslr_tools import comma_link
        address = body.get('address') if isinstance(body, dict) else None
        if not isinstance(address, str):
            raise PanelError(400, 'bad_address', 'The address is a string.')
        try:
            comma_link.set_address(self.root, address)
        except comma_link.AddressError as exc:
            raise PanelError(400, 'bad_address', str(exc)) from None
        return self.comma()

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
                   'routes': [], 'next_step': '', 'next_action': None, 'reviewed_routes': 0, 'min_routes': 3,
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
        # Who can propose the signs on this server (the choice of the Prepare tab), and why not.
        from nnslr_tools.preannotate import ours_missing, vlm_missing
        # qwen: why it is not installed; qwen_gpu: why it cannot load on the GPU now (start() refuses it then); ours:
        # why Our networks are not whole yet.
        payload['proposers'] = {'qwen': vlm_missing(self.root), 'qwen_gpu': _qwen_gpu_reason(system['gpu']) or None,
                                'ours': ours_missing(self.root)}
        # The packages of our networks prepared here, newest first, and where they are published (Models tab).
        from nnslr_tools import hf_account
        from nnslr_tools.networks_package import packages
        payload['networks'] = {'packages': packages(self.root)[:NETWORK_PACKAGES_SHOWN], 'repo': NETWORKS_REPO,
                               'account': hf_account.public_state(self.root)}
        # [ai-reviewer] - START
        # The AI reviewer of Settings (a read of its file, no network): whether a route can be reviewed with AI.
        payload['ai_reviewer'], ai_providers = self._ai_settings()
        payload['actions'] = self._action_states(payload, activities, ai_providers=ai_providers)
        # [ai-reviewer] - END
        # The choices of "Train the cutter", for its buttons.
        payload['cutter_options'] = {'sizes': list(CUTTER_SIZES), 'size': CUTTER_SIZE, 'families': list(CUTTER_FAMILIES),
                                     'family': CUTTER_FAMILY}
        payload['dataset'] = self.dataset_summary()
        payload['training_starts'] = self.training_starts()
        payload['next_step_text'], payload['all_ready'] = _next_step_text(payload)
        for job in payload['jobs']:
            job['progress'] = self.job_progress(job, payload['preparation'])
        self._terminal_training(payload)
        return payload

    def _terminal_training(self, payload: dict) -> None:
        """A training started from a terminal (helpers/3_train.sh) is no job of the panel: its progress comes
        from the log the helper writes itself, the newest data/logs/train-<run>.log; the classifier's
        (scripts/run_classifier.sh) from the newest data/runs/<run>/classifier.log; the cutter's
        (scripts/run_detector_sizes.sh) from the newest data/runs/<run>/detector-sizes/<size>.log, which holds the
        training of a size alone (its epochs: the checks before it are over)."""
        from_panel = any(job['state'] == 'running' and ACTIONS[job['action']]['group'] == 'train'
                         for job in payload['jobs'])
        trainings = [item for item in payload['activities'] if item['group'] == 'train']
        if from_panel or not trainings:
            return
        terminal = {'classifier training': 'train_classifier', 'cutter training': 'train_cutter'}
        patterns = (('train', 'logs/train-*.log'), ('train_classifier', 'runs/*/classifier.log'),
                    ('train_cutter', 'runs/*/detector-sizes/*.log'))
        for action, pattern in patterns:
            items = [item for item in trainings if terminal.get(item['label'], 'train') == action]
            logs = sorted(self.root.glob(pattern), key=lambda path: path.stat().st_mtime) if items else []
            if not logs:
                continue
            size = logs[-1].stat().st_size
            with logs[-1].open('rb') as log:
                log.seek(max(0, size - PROGRESS_BYTES))
                text = log.read().decode('utf-8', 'replace')
            if action == 'train_cutter':
                text = '== 3/4 Train the detector at ' + logs[-1].stem + ' ==\n' + text
            result = progress(action, 'running', text)
            for item in items:
                item['progress'] = result

    def job_progress(self, job: dict, preparation: dict | None) -> dict:
        """Steps, percent and last line of a job, from its log (nnslr_tools.job_progress)."""
        key = (job['size'], job['state'])
        cached = self._progress.get(job['id'])
        # A preparation's percent changes without its log growing: never cached while it runs.
        if cached is not None and cached[:2] == key and not (job['action'] in PREPARING and job['state'] == 'running'):
            result = cached[2]
        else:
            result = self._read_progress(job, preparation)
            self._progress[job['id']] = (key[0], key[1], result)
        # The progress inside the current step changes without the log growing: never from the cache.
        if job['state'] == 'running':
            result = self._with_step_progress(result, job['action'], job['started_at'])
        return result

    def _read_progress(self, job: dict, preparation: dict | None) -> dict:
        try:
            with (self.logs / (job['id'] + '.log')).open('rb') as log:
                log.seek(max(0, job['size'] - PROGRESS_BYTES))
                text = log.read().decode('utf-8', 'replace')
        except OSError:
            text = ''
        return progress(job['action'], job['state'], text, preparation if job['action'] in PREPARING else None)

    def _with_step_progress(self, result: dict, action: str, started_at: str) -> dict:
        """The progress inside the current step: the last report of a command (logs/progress.json, written
        by nnslr_tools.progress) if it came after the job started, or what the GPU setup is doing."""
        record = _read_json(self.root / 'logs' / 'progress.json')
        try:
            started = datetime.fromisoformat(started_at).timestamp()
        except ValueError:
            started = 0.0
        if record is not None and not (isinstance(record.get('at'), (int, float)) and record['at'] >= started):
            record = None
        setup = ''
        download = None
        try:
            with (self.root / 'logs' / 'setup-gpu.log').open('rb') as log:
                log.seek(max(0, log.seek(0, 2) - 8192))
                tail = log.read().decode('utf-8', 'replace')
            setup = last_line(tail)
            download = self._download(tail)
        except OSError:
            pass
        return with_step_progress(result, action, record, setup, download)

    def _download(self, setup_log: str) -> dict | None:
        """The file pip is downloading for the GPU environment: its name and size from the log
        ("Downloading torch-....whl (906.4 MB)"), and how much is already written, from the size of the
        temporary file pip has open (seen through /proc, whatever /tmp the job has)."""
        found = DOWNLOADING.findall(setup_log)
        if not found:
            return None
        name, size, unit = found[-1]
        total = float(size) * {'kB': 1e3, 'MB': 1e6, 'GB': 1e9}[unit]
        written = 0
        for entry in Path('/proc').iterdir():
            if not entry.name.isdigit():
                continue
            try:
                command = (entry / 'cmdline').read_bytes()
                if b'pip' not in command or b'install' not in command:
                    continue
                for handle in (entry / 'fd').iterdir():
                    target = os.readlink(handle)
                    if '/tmp' in target and not target.endswith(('.py', '.pyc', '.so')):
                        info = os.stat(handle)
                        if stat.S_ISREG(info.st_mode) and info.st_size <= total * 1.05:
                            written = max(written, info.st_size)
            except OSError:
                continue
        return {'name': name.split('-')[0], 'written': written, 'total': total}

    def _action_states(self, payload: dict, activities: list[dict], *, ai_providers: int = 0) -> dict:
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
                reason = 'the panel is read-only'
            elif conflict is not None:
                reason = self._busy_reason(conflict, activities)
            elif name == 'update' and not (self.repo / 'helpers/update.sh').is_file():
                reason = 'helpers/update.sh is missing in this installation'
            elif name == 'update':
                pass
            elif name == 'fetch' and not (self.repo / 'helpers/fetch_comma_segments.sh').is_file():
                reason = 'helpers/fetch_comma_segments.sh is missing in this installation'
            elif name == 'fetch' and self.fetch_python() is None:
                reason = 'the reader of the comma logs is missing (see docs/GUIDA.md)'
            elif name == 'fetch':
                pass
            elif name == 'prepare' and not to_prepare:
                reason = 'no new drive to prepare'
            elif name == 'train_classifier' and not (self.repo / CLASSIFIER_SCRIPT).is_file():
                reason = CLASSIFIER_SCRIPT + ' is missing in this installation'
            elif name == 'train_cutter' and not (self.repo / CUTTER_SCRIPT).is_file():
                reason = CUTTER_SCRIPT + ' is missing in this installation'
            elif name == 'train_decider' and not (self.repo / DECIDER_SCRIPT).is_file():
                reason = DECIDER_SCRIPT + ' is missing in this installation'
            elif name in ('networks_prepare', 'networks_publish') and not (self.repo / NETWORKS_SCRIPT).is_file():
                reason = NETWORKS_SCRIPT + ' is missing in this installation'
            elif name == 'networks_prepare':
                # Our networks only whole (the same rule as Prepare's proposer).
                reason = payload.get('proposers', {}).get('ours') or ''
            elif name == 'networks_publish' and not payload.get('networks', {}).get('packages'):
                reason = 'no package prepared yet'
            elif name == 'networks_publish' and not (payload.get('networks', {}).get('account') or {}).get('connected'):
                reason = 'connect Hugging Face first: Settings, Hugging Face'
            # [ai-reviewer] - START
            elif name in AI_ACTIONS and not (self.repo / action['command'][0]).is_file():
                reason = action['command'][0] + ' is missing in this installation'
            elif name == 'ai_review' and not payload.get('ai_reviewer', {}).get('available'):
                # A reviewer chosen but not passing its Test is named in the state, with why it cannot review.
                reason = ('the AI reviewer has not passed its Test: test it in Settings'
                          if payload.get('ai_reviewer', {}).get('provider') else 'no AI reviewer: choose one in Settings')
            elif name == 'ai_review' and not any(row.get('run_id') for row in rows):
                reason = 'no prepared drive'
            elif name == 'ai_review':
                # It reviews the frames nobody confirmed: it needs no confirmed frame.
                pass
            elif name == 'measure_ai_reviewer' and not ai_providers:
                reason = 'no AI provider: add one in Settings'
            # [ai-reviewer] - END
            elif name != 'prepare' and not confirmed:
                reason = 'no confirmed frame'
            elif name.startswith('train') and payload['reviewed_routes'] < payload['min_routes']:
                reason = (f'{payload["min_routes"]} reviewed drives are needed, '
                          f'now {payload["reviewed_routes"]}')
            elif name.startswith('train') and gpu is not None and gpu['processes']:
                reason = 'the GPU is busy with another process'
            elif (name == 'setup_vlm' and gpu is not None and gpu.get('memory_used_mb') and gpu.get('memory_total_mb')
                  and gpu['memory_used_mb'] > gpu['memory_total_mb'] / 2):
                # Its last step loads Qwen3-VL on the GPU (about 17 GB): it would only run out of memory.
                reason = 'the GPU memory is taken by another program: free it first'
            elif name == 'train_classifier':
                # It mines the false boxes of the cutter in use (models/active.json): there must be one.
                reason = self.models_index.in_use('cutter')[1]
            states[name] = {'label': action['label'], 'allowed': not reason, 'reason': reason}
        return states

    # -- the dataset and the starting points of a training ----------------------------------------
    def dataset_summary(self) -> dict | None:
        """What the gold dataset holds (frames, objects by family, human/AI, per route), its split and the
        last import. Read again only when one of the files changed."""
        objects = self.root / 'annotations' / 'objects.jsonl'
        files = [objects, self.root / 'logs' / 'import.json', self.root / 'splits' / 'latest.json']
        key = tuple(path.stat().st_mtime_ns if path.is_file() else 0 for path in files)
        if self._dataset is not None and self._dataset[0] == key:
            return self._dataset[1]
        summary = _dataset_summary(self.root) if objects.is_file() else None
        self._dataset = (key, summary)
        return summary

    def training_starts(self) -> list[dict]:
        """Where a training can start: ImageNet weights (the default), random weights, the stable model
        on Hugging Face (when one is published), or a finished run of this server."""
        starts = [
            {'key': 'imagenet', 'label': 'Pre-trained on ImageNet', 'default': True, 'available': True, 'reason': '',
             'description': 'MobileNetV3 that already knows shapes, edges and circles from a million photos: '
                            'it learns the Swiss signs fastest.'},
            {'key': 'random', 'label': 'Random weights', 'default': False, 'available': True, 'reason': '',
             'description': 'A network that knows nothing: it learns everything from the reviewed frames. '
                            'Slower, and weaker with few frames.'},
        ]
        stable = self._stable_published()
        starts.append({'key': 'stable', 'label': 'The stable model', 'default': False, 'available': stable is True,
                       'reason': '' if stable is True else ('Checking Hugging Face…' if stable is None else
                                                             'No stable model published on Hugging Face yet'),
                       'description': 'Continues from the last model published as stable on Hugging Face.'})
        for run in self._finished_runs():
            starts.append({'key': 'run:' + run, 'label': 'Run ' + run, 'default': False, 'available': True,
                           'reason': '', 'description': 'Continues from the best weights of this earlier '
                                                        'training on this server.'})
        return starts

    def _finished_runs(self) -> list[str]:
        """Runs of this server with the best weights of both networks, newest first."""
        runs = self.root / 'runs'
        if not runs.is_dir():
            return []
        found = [entry for entry in runs.iterdir() if RUN_NAME.fullmatch(entry.name)
                 and (entry / 'detector' / 'detector-best.pt').is_file() and (entry / 'reader' / 'reader-best.pt').is_file()]
        return [entry.name for entry in sorted(found, key=lambda entry: entry.stat().st_mtime, reverse=True)]

    def _stable_published(self) -> bool | None:
        """True when training/stable.json exists on Hugging Face; None while not checked yet. Asked at most
        every 10 minutes, in the background, so a status call never waits for the network."""
        with self._lock:
            checked_at, published = self._stable
            if time.monotonic() - checked_at > STABLE_CHECK_SECONDS and not self._stable_checking:
                self._stable_checking = True
                threading.Thread(target=self._check_stable, daemon=True).start()
        return published

    def _check_stable(self) -> None:
        repo = os.environ.get('NNSLR_HF_REPO', 'cristianku/sunnypilot-speedlimits-models')
        url = 'https://huggingface.co/' + repo + '/resolve/main/training/stable.json'
        try:
            with urllib.request.urlopen(urllib.request.Request(url, method='HEAD'), timeout=10) as answer:
                published = answer.status == 200
        except (OSError, ValueError):
            # 401/404 (no repository or no stable.json) and no network alike: not available.
            published = False
        with self._lock:
            self._stable = (time.monotonic(), published)
            self._stable_checking = False

    # -- the training runs, for the Evaluate tab ------------------------------------------------------
    def runs(self) -> dict:
        """Every run under <root>/runs with its state and headline numbers, newest first."""
        return self.runs_index.summaries()

    def run(self, name: str) -> dict:
        """One run: its evaluation per label and per sign size, the dataset and split it used, its curves."""
        detail = self.runs_index.detail(name)
        if detail is None:
            raise PanelError(404, 'unknown_run', 'No such training run.')
        return detail

    def frames(self, name: str, model, partition, route=None, previews=None) -> dict:
        """The frames of one route of an evaluation of a run, with the boxes over each, for the Watch panel
        (nnslr_tools.runs.RunsIndex.frames)."""
        try:
            return self.runs_index.frames(name, model, partition, route, previews)
        except WatchError as exc:
            raise PanelError(exc.status, exc.reason, exc.message) from exc

    # -- the networks by role, for the Models tab -----------------------------------------------------
    def proposers(self) -> dict:
        """The latest measure of each proposer of Prepare (helpers/measure_proposer.sh), and why one cannot run."""
        from nnslr_tools.preannotate import PROPOSERS, ours_missing, vlm_missing
        from nnslr_tools.proposer_eval import latest_reports
        reports = latest_reports(self.root)
        unavailable = {'qwen': vlm_missing(self.root), 'ours': ours_missing(self.root)}
        best = _best_proposer(reports, {key for key, why in unavailable.items() if why})
        return {'proposers': [{'key': key, 'label': label, 'unavailable': unavailable.get(key)}
                              for key, label in PROPOSERS.items()], 'reports': reports,
                # The choice of Prepare: kept on the server once made (one that cannot run stays chosen, and Prepare
                # says why with its buttons off), else the best measured that can run.
                'chosen': self._kept_proposer() or best or DEFAULT_PROPOSER, 'best': best}

    def _kept_proposer(self) -> str | None:
        """The proposer chosen on the Prepare tab (PROPOSER_FILE), or None before a choice."""
        kept = (_read_json(self.root / PROPOSER_FILE) or {}).get('proposer')
        return kept if kept in PROPOSERS else None

    def set_proposer(self, body) -> dict:
        """Keeps the choice of Prepare's Found by ({'proposer': 'ours'}) for every browser; proposers() with it. A
        read-only panel refuses it, as it refuses the actions."""
        if not self.actions_enabled:
            raise PanelError(403, 'actions_disabled', 'The panel is read-only on this server.')
        proposer = body.get('proposer') if isinstance(body, dict) else None
        if proposer not in PROPOSERS:
            raise PanelError(400, 'bad_option', 'Unknown proposer: ' + ', '.join(PROPOSERS) + '.')
        path = self.root / PROPOSER_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name('.' + path.name + '.tmp')
        temporary.write_text(json.dumps({'proposer': proposer}) + '\n', encoding='utf-8')
        os.replace(temporary, path)
        return self.proposers()

    # [ai-reviewer] - START
    def ai_reviewers(self) -> dict:
        """The AI providers of Settings (never a key) with the latest measure of each (helpers/measure_ai_reviewer.sh),
        for the AI reviewers view of the Evaluate tab."""
        from nnslr_tools import ai_review, llm_providers
        return {**llm_providers.public_settings(self.root), 'reports': ai_review.latest_reports(self.root)}

    def _ai_settings(self) -> tuple[dict, int]:
        """The state of the AI reviewer (llm_providers.ai_state) and how many providers there are. A providers file
        that cannot be read makes the AI unavailable, never the whole panel."""
        from nnslr_tools import llm_providers
        from nnslr_tools.review_db import ReviewRejected
        try:
            return llm_providers.ai_state(self.root), len(llm_providers.public_settings(self.root)['providers'])
        except ReviewRejected as exc:
            return {'available': False, 'reason': exc.message, 'provider': None}, 0

    def _ai_arguments(self, action: str, options: dict) -> list[str]:
        """The arguments of an AI action from its options, all strings: the route (ai_review only, one with a
        prepared report), --provider=<id of Settings> (the reviewer when none is given), --save (ai_review with
        save 'yes'; 'no' is the default) and --limit=<frames, 1 to MAX_AI_LIMIT>. Never a key: the job reads the
        providers file itself."""
        from nnslr_tools import llm_providers
        arguments = []
        if action == 'ai_review':
            route = options.get('route')
            if not route:
                raise PanelError(400, 'bad_option', 'Choose the route to review.')
            from nnslr_tools.preannotate import PreannotationError
            from nnslr_tools.review import latest_report
            try:
                latest_report(self.root, route)
            except (PreannotationError, ValueError, OSError):
                raise PanelError(400, 'bad_option', 'Unknown route, or a route without a prepared report.') from None
            arguments.append(route)
        settings = llm_providers.public_settings(self.root)
        provider = options.get('provider')
        if provider is None:
            provider = settings['reviewer']
            if provider is None:
                raise PanelError(409, 'no_ai_reviewer', 'Choose an AI reviewer in Settings.')
        elif not PROVIDER_ID.fullmatch(provider) or provider not in {item['id'] for item in settings['providers']}:
            raise PanelError(400, 'bad_option', 'Unknown AI provider: add it in Settings.')
        arguments.append('--provider=' + provider)
        save = options.get('save', 'no')
        if save not in ('yes', 'no'):
            raise PanelError(400, 'bad_option', 'Save is yes or no.')
        if save == 'yes':
            # Its saves are final: only a provider that passed its last Test (also checked by the job itself).
            refusal = llm_providers.review_refusal(llm_providers.provider(self.root, provider))
            if refusal:
                raise PanelError(409, 'ai_not_tested', refusal)
            arguments.append('--save')
        limit = options.get('limit')
        if limit is not None:
            if not re.fullmatch(r'[0-9]{1,4}', limit) or not 1 <= int(limit) <= MAX_AI_LIMIT:
                raise PanelError(400, 'bad_option', f'The limit is a number of frames from 1 to {MAX_AI_LIMIT}.')
            arguments.append('--limit=' + str(int(limit)))
        return arguments
    # [ai-reviewer] - END

    def models(self) -> dict:
        """Every trained cutter, classifier and decider with its numbers, newest first, and the one in use."""
        return self.models_index.index()

    def set_active_model(self, body, *, who: str | None, via: str) -> dict:
        """The model in use for a role ({"role", "id"}), written to <root>/models/active.json. A read-only panel
        refuses it, as it refuses the actions."""
        if not self.actions_enabled:
            raise PanelError(403, 'actions_disabled', 'The panel is read-only on this server.')
        if not isinstance(body, dict):
            raise PanelError(400, 'invalid_json', 'A JSON object is required.')
        try:
            return self.models_index.set_active(body.get('role'), body.get('id'), who=who, via=via)
        except ModelChoiceError as exc:
            raise PanelError(exc.status, exc.reason, exc.message) from exc

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
        again = self._again_progress(by_route)
        if again is not None:
            return again
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

    def _again_progress(self, by_route: dict) -> dict | None:
        """Propose again (helpers/1_prepare.sh --again): every route of the batch already has a report, the new
        proposals are written aside (.pending-<run>) and replace it when the route is done. Per route: waiting, its
        frames proposed so far, or ready (a report newer than the batch); the percent over the frames of the batch,
        the time left from the pace of the route at work, and who proposes. None for a batch of new routes."""
        try:
            batch = json.loads((self.root / 'logs/prepare/batch.json').read_text(encoding='utf-8'))
            started = datetime.strptime(batch['started'], '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc)
            routes = [route for route in batch['routes'] if isinstance(route, str)]
        except (OSError, ValueError, KeyError, TypeError):
            return None
        if not routes or any(by_route.get(route, {}).get('run_id') is None for route in routes):
            return None
        from nnslr_tools.drives import drive_of
        from nnslr_tools.preannotate import PROPOSER_NAMES
        now = datetime.now(timezone.utc)
        rows, done_frames, all_frames, pace = [], 0, 0, None
        for route in routes:
            row = by_route[route]
            base = self.root / 'derived/preannotations' / route
            total = row.get('frames') or row.get('report_frames') or 0
            pending = sorted(base.glob('.pending-*'))
            fresh = pending[-1] if pending and datetime.fromtimestamp(pending[-1].stat().st_mtime, timezone.utc) >= started else None
            if fresh is not None:
                done = min(_line_count(fresh / 'preannotations.jsonl'), total)
                since = (now - datetime.fromtimestamp(fresh.stat().st_ctime, timezone.utc)).total_seconds()
                if done and since > 0:
                    pace = since / done
                phase, percent = 'proposing again', (min(99, 100 * done // total) if total else 0)
            elif str(row['run_id']) >= started.strftime('%Y%m%dT%H%M%SZ'):
                done, phase, percent = total, 'ready', 100
            else:
                done, phase, percent = 0, 'waiting', 0
            done_frames, all_frames = done_frames + done, all_frames + total
            rows.append({'route': route, 'title': (drive_of(self.root, route) or {}).get('title'), 'phase': phase,
                         'done': done, 'total': total, 'unit': 'frames', 'percent': percent})
        left = round(pace * (all_frames - done_frames)) if pace else None
        for item in rows:
            if item['phase'] == 'proposing again' and pace:
                item['seconds_left'] = round(pace * (item['total'] - item['done']))
        proposer = self._again_proposer()
        return {'percent': round(100 * done_frames / all_frames, 1) if all_frames else None,
                'routes_done': sum(1 for item in rows if item['phase'] == 'ready'), 'routes_total': len(rows),
                'working': rows, 'again': True, 'seconds_left': left,
                'proposer': PROPOSER_NAMES.get(proposer, proposer) if proposer else None}

    def _again_proposer(self) -> str | None:
        """Who proposes again: the "(qwen)" of the newest prepare_again job's log."""
        logs = sorted(self.logs.glob('*-prepare_again.log')) if self.logs.is_dir() else []
        if not logs:
            return None
        found = re.findall(r'(?:proposing the signs again|propongo di nuovo i cartelli) \((\w+)\)',
                           logs[-1].read_text(encoding='utf-8', errors='replace'))
        return found[-1] if found else None

    def _route_progress(self, row: dict) -> dict:
        route = row['route']
        if row['run_id'] is not None:
            return {'route': route, 'phase': 'ready', 'done': row['report_frames'],
                    'total': row['report_frames'], 'unit': 'frames', 'percent': 100}
        if row['linked'] == 0:
            return {'route': route, 'phase': 'to link', 'done': 0, 'total': 0, 'unit': 'segments', 'percent': 0}
        if row['extracted'] < row['linked']:
            return {'route': route, 'phase': 'extracting frames', 'done': row['extracted'], 'total': row['linked'],
                    'unit': 'segments', 'percent': 50 * row['extracted'] // row['linked']}
        done = self._preannotated_frames(route)
        total = row['frames']
        percent = 50
        if total:
            percent = 50 + 50 * min(done, total) // total
        # The report counts only once it is moved in place (run_id above).
        if percent > 99:
            percent = 99
        return {'route': route, 'phase': 'proposing signs', 'done': done, 'total': total, 'unit': 'frames', 'percent': percent}

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
    def start(self, action: str, options: dict | None = None) -> dict:
        """Starts an action. The only options are the starting point of a training ({'start': key} of
        training_starts()), turned into a flag or the environment of helpers/3_train.sh, the run to resume, and the
        cutter's size and families (one of CUTTER_SIZES, CUTTER_FAMILIES), and the route, provider, save and limit of
        an AI action (_ai_arguments): never a command. The classifier's and the cutter's trainings run on a new run
        named here (new_run_name)."""
        if action not in ACTIONS:
            raise PanelError(404, 'unknown_action', 'Unknown action.')
        start_env = {}
        start_args = []
        for key in ('size', 'families'):
            if (options or {}).get(key) is not None and key not in ACTION_OPTIONS.get(action, ()):
                raise PanelError(400, 'bad_option', 'Only "Train the cutter" takes a ' + ('size' if key == 'size' else 'choice of signs') + '.')
        # [ai-reviewer] - START
        for key in AI_OPTIONS:
            value = (options or {}).get(key)
            if value is not None and key not in ACTION_OPTIONS.get(action, ()):
                raise PanelError(400, 'bad_option', '"' + ACTIONS[action]['label'] + '" does not take a ' + key + '.')
            if value is not None and not isinstance(value, str):
                raise PanelError(400, 'bad_option', 'The ' + key + ' is a string.')
        if action in AI_ACTIONS:
            start_args += self._ai_arguments(action, options or {})
        if action == 'fetch':
            # The address set in the comma panel of the top bar; without one the script's ssh host "comma".
            from nnslr_tools.comma_link import configured
            if configured(self.root):
                start_args.append('--host=' + configured(self.root))
        # [ai-reviewer] - END
        proposer = (options or {}).get('proposer')
        if proposer is not None:
            if action not in ('prepare', 'prepare_again', 'measure_proposer'):
                raise PanelError(400, 'bad_option', 'Only preparing routes takes a proposer.')
            if proposer not in PROPOSERS:
                raise PanelError(400, 'bad_option', 'Unknown proposer: ' + ', '.join(PROPOSERS) + '.')
            if proposer == 'ours':
                from nnslr_tools.preannotate import ours_missing
                missing = ours_missing(self.root)
                if missing:
                    raise PanelError(409, 'not_available', missing + '.')
            if proposer == 'qwen':
                from nnslr_tools.preannotate import vlm_missing
                missing = vlm_missing(self.root)
                if missing:
                    raise PanelError(409, 'not_available', missing + '.')
                # Qwen3-VL needs about 17.5 GB of the GPU: refused, not run out of memory, when they are taken
                # (another Qwen job, a training, llama.cpp in another container).
                taken = _qwen_gpu_reason(self.system().get('gpu'))
                if taken:
                    raise PanelError(409, 'gpu_taken', taken + '.')
            start_env['NNSLR_PROPOSER'] = proposer
        if action == 'train_cutter':
            size = (options or {}).get('size') or CUTTER_SIZE
            families = (options or {}).get('families') or CUTTER_FAMILY
            if size not in CUTTER_SIZES:
                raise PanelError(400, 'bad_option', 'Unknown size of the cutter: ' + ', '.join(CUTTER_SIZES) + '.')
            if families not in CUTTER_FAMILIES:
                raise PanelError(400, 'bad_option', 'Unknown signs for the cutter: all or speed.')
            start_env['NNSLR_CUTTER_FAMILIES'] = families
        package = (options or {}).get('package')
        if package is not None and action != 'networks_publish':
            raise PanelError(400, 'bad_option', 'Only "Publish on Hugging Face" takes a package.')
        if action == 'networks_publish':
            from nnslr_tools.networks_package import packages
            if not isinstance(package, str) or package not in {item['name'] for item in packages(self.root)}:
                raise PanelError(400, 'bad_option', 'Unknown package: prepare it first.')
            from nnslr_tools import hf_account
            if not hf_account.public_state(self.root)['connected']:
                raise PanelError(409, 'not_available', 'Connect Hugging Face first: Settings, Hugging Face.')
            start_args.append('--name=' + package)
            # The token kept in Settings, in the job's environment only (never its command line or log); without
            # it the script uses this server's own Hugging Face login.
            kept = hf_account.token(self.root)
            if kept:
                start_env['HF_TOKEN'] = kept
        if action == 'networks_prepare':
            from nnslr_tools.networks_package import new_name
            from nnslr_tools.preannotate import ours_missing
            missing = ours_missing(self.root)
            if missing:
                raise PanelError(409, 'not_available', missing + '.')
            start_args.append('--name=' + new_name())
        resumed = (options or {}).get('run')
        if action == 'train_resume':
            if not isinstance(resumed, str) or not RUN_NAME.fullmatch(resumed) or not (self.root / 'runs' / resumed).is_dir():
                raise PanelError(400, 'bad_option', 'Unknown run to resume.')
            start_args.append('--resume=' + resumed)
        elif resumed is not None:
            raise PanelError(400, 'bad_option', 'Only "Resume the training" takes a run.')
        start = (options or {}).get('start')
        if start is not None:
            if action != 'train':
                raise PanelError(400, 'bad_option', 'Only a training has a starting point.')
            choice = next((item for item in self.training_starts() if item['key'] == start), None)
            if choice is None:
                raise PanelError(400, 'bad_option', 'Unknown starting point.')
            if not choice['available']:
                raise PanelError(409, 'not_available', choice['reason'] + '.')
            if start == 'random':
                start_env['NNSLR_TRAIN_INIT'] = 'random'
            elif start == 'stable':
                start_args.append('--continue')
            elif start.startswith('run:'):
                start_env['NNSLR_TRAIN_FROM_RUN'] = start[len('run:'):]
        if not self.actions_enabled or not self.available():
            raise PanelError(403, 'actions_disabled', 'The panel is read-only on this server.')
        group = ACTIONS[action]['group']
        if action == 'update' and not (self.repo / 'helpers/update.sh').is_file():
            raise PanelError(403, 'actions_disabled', 'helpers/update.sh is missing in this installation.')
        fetch_python = None
        if action == 'fetch':
            if not (self.repo / 'helpers/fetch_comma_segments.sh').is_file():
                raise PanelError(403, 'actions_disabled', 'helpers/fetch_comma_segments.sh is missing in this installation.')
            fetch_python = self.fetch_python()
            if fetch_python is None:
                raise PanelError(403, 'actions_disabled', 'The reader of the comma logs is missing (see docs/GUIDA.md).')
        if action == 'train_cutter' and not (self.repo / CUTTER_SCRIPT).is_file():
            raise PanelError(403, 'actions_disabled', CUTTER_SCRIPT + ' is missing in this installation.')
        if action == 'train_decider' and not (self.repo / DECIDER_SCRIPT).is_file():
            raise PanelError(403, 'actions_disabled', DECIDER_SCRIPT + ' is missing in this installation.')
        if action in ('networks_prepare', 'networks_publish') and not (self.repo / NETWORKS_SCRIPT).is_file():
            raise PanelError(403, 'actions_disabled', NETWORKS_SCRIPT + ' is missing in this installation.')
        # [ai-reviewer] - START
        if action in AI_ACTIONS and not (self.repo / ACTIONS[action]['command'][0]).is_file():
            raise PanelError(403, 'actions_disabled', ACTIONS[action]['command'][0] + ' is missing in this installation.')
        # [ai-reviewer] - END
        if action == 'train_classifier':
            if not (self.repo / CLASSIFIER_SCRIPT).is_file():
                raise PanelError(403, 'actions_disabled', CLASSIFIER_SCRIPT + ' is missing in this installation.')
            cutter, why = self.models_index.in_use('cutter')
            if cutter is None:
                raise PanelError(409, 'no_cutter', why[0].upper() + why[1:] + '.')
        with self._lock:
            activities = self.activities()
            conflict = _conflicting_group(group, self._running_groups(activities))
            if conflict is not None:
                raise PanelError(409, 'busy', self._busy_reason(conflict, activities) + ': wait for it to finish.')
            self.logs.mkdir(parents=True, exist_ok=True)
            job_id = datetime.now().strftime('%Y%m%d-%H%M%S') + '-' + action
            path = self.logs / (job_id + '.log')
            if path.exists():
                raise PanelError(409, 'busy', 'An action has just started: try again in a second.')
            command = list(ACTIONS[action]['command']) + start_args
            if action == 'prepare':
                command.append('--jobs=' + os.environ.get(PREPARE_JOBS_ENV, '2'))
            if action == 'train_classifier':
                command.append(self.new_run_name('classifier'))
            if action == 'train_decider':
                command.append(self.new_run_name('decider'))
            if action == 'train_cutter':
                # NAME SIZE: one size, so the run holds one cutter.
                command += [self.new_run_name('cutter'), size]
            # Temporary files in the data root, not in /tmp: the service has a private /tmp (PrivateTmp)
            # that systemd empties at every restart, which broke a pip download of a running training.
            temporary = self.root / 'tmp'
            temporary.mkdir(parents=True, exist_ok=True)
            environment = dict(os.environ, NNSLR_DATA_ROOT=str(self.root), TMPDIR=str(temporary), **start_env)
            if fetch_python:
                environment[FETCH_PYTHON_ENV] = fetch_python
            if proposer is not None:
                # Who proposes, for the page (the running job's choice) and Try again: the log does not always say.
                path.with_suffix('.json').write_text(json.dumps({'proposer': proposer}) + '\n', encoding='utf-8')
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

    def new_run_name(self, kind: str) -> str:
        """The run of a new classifier or cutter training: the time it starts and what it is,
        2026-10-24_1730-classifier, 2026-10-24_1730-cutter; -2, -3... when a run of that name is already there."""
        base = datetime.now().strftime('%Y-%m-%d_%H%M') + '-' + kind
        name, number = base, 1
        while (self.root / 'runs' / name).exists() or (self.root / 'runs' / name).is_symlink():
            number += 1
            name = f'{base}-{number}'
        return name

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
        wrappers = [[b'bash', b'-c', wrapper.encode(), b'panel'] for wrapper in (WRAPPER, *OLD_WRAPPERS)]
        if not wanted or command[:4] not in wrappers or command[4:4 + len(wanted)] != wanted:
            return None
        return pid

    def stop(self, job_id: str) -> dict:
        group = self._group(job_id)
        if group is None:
            if JOB_ID.fullmatch(job_id) and (self.logs / (job_id + '.log')).is_file():
                raise PanelError(409, 'not_running', 'The action has already finished.')
            raise PanelError(404, 'unknown_job', 'Not an action of the panel (stop it from its terminal).')
        try:
            os.killpg(group, signal.SIGTERM)
        except ProcessLookupError:
            pass
        return {'id': job_id, 'stopping': True}

    def log(self, job_id: str, offset: int) -> dict:
        if not JOB_ID.fullmatch(job_id):
            raise PanelError(404, 'unknown_job', 'This job is not on the server.')
        path = self.logs / (job_id + '.log')
        if not path.is_file():
            raise PanelError(404, 'unknown_job', 'This job is not on the server.')
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


def _proposer_of(log: Path) -> str | None:
    """The proposer of a job: the one kept next to its log when it started (<job id>.json, Panel.start), else the
    one its log names near its top (PROPOSER_LINE); None when neither says (the helpers' default)."""
    options = _read_json(log.with_suffix('.json'))
    if options is not None and options.get('proposer') in PROPOSERS:
        return options['proposer']
    try:
        with log.open('rb') as handle:
            head = handle.read(65536).decode('utf-8', 'replace')
    except OSError:
        return None
    match = PROPOSER_LINE.search(head)
    return match.group(1) if match else None


def _best_proposer(reports: list[dict], left_out: set[str]) -> str | None:
    """The proposer whose latest measure on the test frames reads the most speed signs right (then finds the most,
    then is the fastest), among those not left out (one that cannot run here); None when none is measured."""
    def score(report: dict) -> tuple:
        speed = report.get('speed_signs') or {}
        return (speed.get('read_right_share') or 0.0, speed.get('found_share') or 0.0,
                -(report.get('seconds_per_frame') or 0.0))
    measured = [report for report in reports if report.get('partition') == 'test' and report.get('proposer') in PROPOSERS
                and report['proposer'] not in left_out and (report.get('speed_signs') or {}).get('truth')]
    return max(measured, key=score)['proposer'] if measured else None


def _qwen_gpu_reason(gpu: dict | None) -> str:
    """Why Qwen3-VL cannot load on the GPU now ('' when it can, or when the GPU is not known): it needs QWEN_GPU_MB
    free (another Qwen job, a training, llama.cpp in another container take it)."""
    used, total = (gpu or {}).get('memory_used_mb'), (gpu or {}).get('memory_total_mb')
    if used is None or not total or total - used >= QWEN_GPU_MB:
        return ''
    return (f'Qwen3-VL needs about {QWEN_GPU_MB // 1024} GB of the GPU and only {max(0, total - used) / 1024:.1f} GB '
            'are free: another job or program is using it')


# English names of the sign families of the dataset (sign_family of objects.jsonl).
FAMILY_LABELS = {'maximum_speed': 'Speed limit', 'variable_display': 'Variable display', 'zone': 'Zone',
                 'cancellation': 'End of limit', 'other_sign': 'Other sign', 'not_a_sign': 'Not a sign',
                 'unreadable': 'Unreadable', 'road_marking': 'Road marking'}
SPEED_FAMILIES = ('maximum_speed', 'variable_display', 'zone', 'cancellation', 'road_marking')


def _dataset_summary(root: Path) -> dict:
    families: dict[str, int] = {}
    per_route: dict[str, dict] = {}
    frames: set[str] = set()
    human = ai = objects = 0
    # The sha256 of the file, as a split records it (dataset_sha256): the split is of this dataset when they match.
    digest = hashlib.sha256()
    with (root / 'annotations' / 'objects.jsonl').open('rb') as source:
        for line in source:
            digest.update(line)
            try:
                item = json.loads(line)
            except ValueError:
                continue
            objects += 1
            family = item.get('sign_family') or 'unknown'
            families[family] = families.get(family, 0) + 1
            by_ai = (item.get('provenance') or {}).get('label_basis') == 'ai_review'
            ai += by_ai
            human += not by_ai
            route = per_route.setdefault(item.get('route_id') or '?', {'frames': set(), 'objects': 0,
                                                                        'speed_signs': 0, 'human': 0, 'ai': 0})
            route['frames'].add(item.get('image_path'))
            frames.add(item.get('image_path'))
            route['objects'] += 1
            route['speed_signs'] += family in SPEED_FAMILIES
            route['ai' if by_ai else 'human'] += 1
    # The split in use: data/splits/latest.json points to it (logs/splits.json is rewritten while a
    # training makes a new one).
    pointer = _read_json(root / 'splits' / 'latest.json') or {}
    split_path = root / str(pointer.get('path', '')) if pointer.get('path') else None
    split = _read_json(split_path) if split_path is not None and split_path.resolve().is_relative_to(root.resolve()) else None
    imported = _read_json(root / 'logs' / 'import.json')
    return {
        'frames': len(frames), 'objects': objects, 'routes': len(per_route), 'human': human, 'ai': ai,
        'families': [{'key': key, 'label': FAMILY_LABELS.get(key, key.replace('_', ' ').capitalize()), 'count': count}
                     for key, count in sorted(families.items(), key=lambda pair: -pair[1])],
        'per_route': [{'route': name, 'frames': len(values['frames']), 'objects': values['objects'],
                       'speed_signs': values['speed_signs'], 'human': values['human'], 'ai': values['ai']}
                      for name, values in sorted(per_route.items())],
        'changed_at': _mtime(root / 'annotations' / 'objects.jsonl'),
        'sha256': digest.hexdigest(),
        'split': None if not split or split_path is None else {'counts': split.get('counts') or {}, 'made_at': _mtime(split_path),
                                         'warnings': split.get('warnings') or [],
                                         # The dataset it was made of, and the proposals a person rejected it leaves out.
                                         'dataset_sha256': split.get('dataset_sha256') if isinstance(split.get('dataset_sha256'), str) else None,
                                         'rejected': len(split['excluded_rejected_ids'])
                                         if isinstance(split.get('excluded_rejected_ids'), list) else 0},
        'last_import': None if not imported else {
            'at': _mtime(root / 'logs' / 'import.json'),
            'new_frames': imported.get('imported_frame_count') or 0,
            'already': imported.get('already_imported_frame_count') or 0,
            # Frames reviewed again after their import replace their old rows, and withdrawn ones leave the
            # dataset; only an import made before that (no replaced_frame_count) left changed frames unapplied.
            'changed_after_import': 0 if 'replaced_frame_count' in imported
            else len(imported.get('changed_after_import') or []),
            'replaced_frames': imported.get('replaced_frame_count') or 0,
            'removed_frames': imported.get('removed_withdrawn_frame_count') or 0,
            'warnings': imported.get('warnings') or []},
    }


def _read_json(path: Path) -> dict | None:
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _mtime(path: Path) -> str | None:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat(timespec='seconds')
    except OSError:
        return None


def _next_step_text(payload: dict) -> tuple[str, bool]:
    """The next step in words for the page (route_status.next_step names the helpers, for the terminal), and
    whether everything is ready (dataset current, a training can start)."""
    rows = payload['routes']
    states = {row['state'] for row in rows}
    if not rows:
        return 'No drives yet: download drives from the comma.', False
    if states & {'collegare', 'estrarre', 'preannotare'}:
        return 'New videos to turn into frames with proposed signs: prepare the new drives.', False
    if 'revisionare' in states:
        return 'Open the drives to review and confirm or correct the signs.', False
    if payload['reviewed_routes'] < payload['min_routes']:
        return (f'Training needs at least {payload["min_routes"]} reviewed drives (now {payload["reviewed_routes"]}): '
                'download more drives from the comma.'), False
    if 'importare' in states:
        return 'The reviews are ready: import them and train the detector and the reader on the GPU.', False
    return 'The dataset is up to date: you can train a new model whenever you like.', True


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
