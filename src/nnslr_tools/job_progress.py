"""How far an action of the panel is, from its log: the steps, the current one, a percent when it is known.

The helpers write a line per phase (``== a. ...`` in helpers/3_train.sh, ``== 4/10 Train detector`` in
scripts/run_v100_baseline.sh, ``== b+c. ...`` in helpers/1_prepare.sh) and the training one line per
epoch (``detector epoch 3/20 ...``, nnslr_tools.training). Nothing here guesses a time: a percent comes
only from steps and epochs actually written, or from the preparation's own count of finished routes.
"""
from __future__ import annotations

import re

# Steps of the import and of the training, with their weight in the whole (they add up to 100): the
# detector and the reader are most of a training.
TRAIN_STEPS = (
    ('import', 'Importa le revisioni', 2),
    ('split', 'Controlla il dataset e fissa lo split', 2),
    ('gpu_env', 'Prepara l’ambiente GPU', 6),
    ('checks', 'Controlli e prova della GPU', 3),
    ('detector', 'Allena il detector', 40),
    ('reader', 'Allena il reader', 30),
    ('evaluate', 'Valuta su route mai viste', 7),
    ('export', 'Errori difficili ed export ONNX', 6),
    ('bundle', 'Prepara e verifica il pacchetto', 4),
)
DATASET_STEPS = TRAIN_STEPS[:2]
PREPARE_STEPS = (('link', 'Collega i video', 5), ('frames', 'Estrae i fotogrammi e propone i cartelli', 95))
FETCH_STEPS = (('fetch', 'Scarica i giri dal comma', 100),)
UPDATE_STEPS = (('update', 'Aggiorna il codice e riavvia il sito', 100),)
STEPS = {'fetch': FETCH_STEPS, 'prepare': PREPARE_STEPS, 'dataset': DATASET_STEPS, 'train': TRAIN_STEPS,
         'train_continue': TRAIN_STEPS, 'update': UPDATE_STEPS}

# The sub-steps of scripts/run_v100_baseline.sh ("== 4/10 ...") and the step they belong to.
PIPELINE = {1: 'checks', 2: 'checks', 3: 'checks', 4: 'detector', 5: 'reader', 6: 'evaluate', 7: 'export',
            8: 'export', 9: 'bundle', 10: 'bundle'}
MARKERS = (
    (re.compile(r'^== a\+b\. '), 'gpu_env', ('import', 'split')),
    (re.compile(r'^== a\. importo'), 'import', ()),
    (re.compile(r'^== b\. '), 'split', ()),
    (re.compile(r'^== c\. '), 'gpu_env', ()),
    (re.compile(r'^== a\. collego'), 'link', ()),
    (re.compile(r'^== b\+c\. '), 'frames', ()),
)
SUBSTEP = re.compile(r'^== (\d+)/10 ')
EPOCH = re.compile(r'^(detector|reader) epoch (\d+)/(\d+)\b')
# The values of an epoch line: "train_loss 0.5000 val_loss 0.6000 ... 40s".
METRIC = re.compile(r'\b(train_loss|val_loss|train_accuracy|val_accuracy) (-?[0-9.]+(?:e-?[0-9]+)?)\b')
SECONDS = re.compile(r'\b([0-9]+)s$')
FETCHED = re.compile(r'^== (\S+)--(\d+) -> ')
ON_COMMA = re.compile(r'^(\S+) sul comma: ')
FINISHED = re.compile(r'^== finito con codice ')


def last_line(text: str) -> str:
    """The last line that says something: not the start and end markers of the wrapper, not empty."""
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line and not FINISHED.match(line) and not line.startswith(('== helpers/', '== scripts/')):
            return line[:300]
    return ''


def progress(action: str, state: str, text: str, preparation: dict | None = None) -> dict:
    """{'steps': [{'key', 'label', 'state': done|current|todo|failed}], 'percent': 0-100 or None,
    'detail': str, 'last_line': str, 'curves': {'detector'|'reader': {'epochs': total, 'points':
    [{'epoch', 'train_loss', 'val_loss', ('train_accuracy', 'val_accuracy',) 'seconds'}]}}} for an action
    in state running, ok, failed or interrupted."""
    steps = STEPS.get(action, ())
    keys = [key for key, _, _ in steps]
    current = keys[0] if keys else ''
    done: set[str] = set()
    epoch: tuple[str, int, int] | None = None
    fetched = 0
    route = ''
    # The curves of the training: per network, one point per epoch.
    curves: dict[str, dict] = {}
    for raw in text.splitlines():
        line = raw.strip()
        for pattern, step, skipped in MARKERS:
            if pattern.match(line) and step in keys:
                done.update(skipped)
                current = step
        match = SUBSTEP.match(line)
        if match and PIPELINE.get(int(match.group(1))) in keys:
            current = PIPELINE[int(match.group(1))]
        match = EPOCH.match(line)
        if match:
            epoch = (match.group(1), int(match.group(2)), int(match.group(3)))
            point: dict = {'epoch': epoch[1]}
            point.update((name, float(value)) for name, value in METRIC.findall(line))
            seconds = SECONDS.search(line)
            if seconds:
                point['seconds'] = int(seconds.group(1))
            curve = curves.setdefault(epoch[0], {'epochs': epoch[2], 'points': []})
            curve['epochs'] = epoch[2]
            # A resumed run writes its epochs again: the last line of an epoch wins.
            curve['points'] = [p for p in curve['points'] if p['epoch'] != epoch[1]] + [point]
        if FETCHED.match(line):
            fetched += 1
        match = ON_COMMA.match(line)
        if match:
            route = match.group(1)
    if current in keys:
        done.update(keys[:keys.index(current)])

    finished_ok = state == 'ok'
    rows = []
    for key, label, _ in steps:
        if finished_ok or key in done:
            step_state = 'done'
        elif key == current:
            step_state = 'current' if state == 'running' else 'failed' if state == 'failed' else 'todo'
        else:
            step_state = 'todo'
        rows.append({'key': key, 'label': label, 'state': step_state})

    percent: float | None = None
    detail = ''
    if finished_ok:
        percent = 100.0
    elif action in ('train', 'train_continue', 'dataset'):
        total = sum(weight for _, _, weight in steps)
        reached = sum(weight for key, _, weight in steps if key in done)
        if epoch is not None and epoch[0] == current and epoch[2]:
            weight = dict((key, weight) for key, _, weight in steps)[current]
            reached += weight * epoch[1] / epoch[2]
        percent = round(reached * 100 / total, 1)
    elif action == 'prepare' and preparation is not None and preparation.get('percent') is not None:
        percent = float(preparation['percent'])

    if epoch is not None and epoch[0] == current and state == 'running':
        detail = epoch[0] + ' · epoca ' + str(epoch[1]) + ' di ' + str(epoch[2])
    elif action == 'prepare' and preparation is not None and preparation.get('routes_total'):
        detail = str(preparation['routes_done']) + ' di ' + str(preparation['routes_total']) + ' route pronte'
    elif action == 'fetch' and (fetched or route):
        detail = (str(fetched) + (' segmento scaricato' if fetched == 1 else ' segmenti scaricati')
                  + (' · ' + route if route and state == 'running' else ''))
    return {'steps': rows, 'percent': percent, 'detail': detail, 'last_line': last_line(text), 'curves': curves}
