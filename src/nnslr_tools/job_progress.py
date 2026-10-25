"""How far an action of the panel is, from its log: the steps, the current one, a percent when it is known.

The helpers write a line per phase (``== a. ...`` in helpers/3_train.sh, ``== 4/10 Train detector`` in
scripts/run_v100_baseline.sh, ``== 3/7 Mine the cutter's false boxes ==`` in scripts/run_classifier.sh,
``== 3/4 Train the detector at 960x544 ==`` in scripts/run_detector_sizes.sh, ``== b+c. ...`` in helpers/1_prepare.sh) and the training one line per epoch (``detector epoch 3/20 ...``,
``classifier epoch 3/20 ...``, nnslr_tools.training). Nothing here guesses a time: a percent comes only from
steps and epochs actually written, or from the preparation's own count of finished routes.
"""
from __future__ import annotations

import re

# Steps of the import and of the training, with their weight in the whole (they add up to 100): the
# detector and the reader are most of a training.
TRAIN_STEPS = (
    ('import', 'Import the reviews', 2),
    ('split', 'Check the dataset and fix the split', 2),
    ('gpu_env', 'Set up the GPU environment', 6),
    ('checks', 'Checks and GPU smoke test', 3),
    ('detector', 'Train the detector', 40),
    ('reader', 'Train the reader', 30),
    ('evaluate', 'Evaluate on unseen routes', 7),
    ('export', 'Hard examples and ONNX export', 6),
    ('bundle', 'Package and verify the bundle', 4),
)
DATASET_STEPS = TRAIN_STEPS[:2]
# scripts/run_classifier.sh: the false boxes of the cutter in use mined, then the classifier with "not a sign" trained
# on them, measured alone and in the chain (cutter -> classifier). Its training is most of it.
CLASSIFIER_STEPS = (
    ('checks', 'Checks and GPU smoke test', 4),
    ('mine', 'Mine the cutter’s false boxes', 14),
    ('classifier', 'Train the classifier', 60),
    ('evaluate', 'Evaluate the classifier alone', 7),
    ('chain', 'Evaluate the chain end to end', 15),
)
# scripts/run_detector_sizes.sh with one size: a detector trained as the cutter, then measured on test and validation.
CUTTER_STEPS = (
    ('checks', 'Checks and GPU smoke test', 5),
    ('detector', 'Train the cutter', 80),
    ('evaluate', 'Evaluate on test and validation', 15),
)
PREPARE_STEPS = (('link', 'Link the videos', 5), ('frames', 'Extract the frames and propose the signs', 95))
FETCH_STEPS = (('fetch', 'Download the drives from the comma', 100),)
UPDATE_STEPS = (('update', 'Update the code and restart the site', 100),)
# [ai-reviewer] - START
# One step each: nnslr ai-review and nnslr measure-ai-reviewer report their frames (nnslr_tools.progress).
AI_REVIEW_STEPS = (('review', 'Review the frames with the AI', 100),)
AI_MEASURE_STEPS = (('measure', 'Measure the AI reviewer', 100),)
# [ai-reviewer] - END
STEPS = {'fetch': FETCH_STEPS, 'prepare': PREPARE_STEPS, 'dataset': DATASET_STEPS, 'train': TRAIN_STEPS,
         'train_continue': TRAIN_STEPS, 'train_resume': TRAIN_STEPS, 'train_classifier': CLASSIFIER_STEPS,
         'train_cutter': CUTTER_STEPS, 'update': UPDATE_STEPS, 'ai_review': AI_REVIEW_STEPS,
         'measure_ai_reviewer': AI_MEASURE_STEPS}
# The actions whose percent is the weighted sum of their steps (the AI's: its frames done, from its reports).
WEIGHTED = ('train', 'train_continue', 'train_resume', 'train_classifier', 'train_cutter', 'dataset', 'ai_review',
            'measure_ai_reviewer')

# The sub-steps of scripts/run_v100_baseline.sh ("== 4/10 ...") and the step they belong to.
PIPELINE = {1: 'checks', 2: 'checks', 3: 'checks', 4: 'detector', 5: 'reader', 6: 'evaluate', 7: 'export',
            8: 'export', 9: 'bundle', 10: 'bundle'}
# The sub-steps of scripts/run_classifier.sh ("== 3/7 ..."): the comparison with the old reader closes the chain.
CLASSIFIER_PIPELINE = {1: 'checks', 2: 'checks', 3: 'mine', 4: 'classifier', 5: 'evaluate', 6: 'chain', 7: 'chain'}
# scripts/run_detector_sizes.sh numbers its phases out of 2 + 2 x the sizes: they are known by their titles. The checks
# are always its first two.
CUTTER_PIPELINE = {1: 'checks', 2: 'checks'}
CUTTER_TITLES = ((re.compile(r'Validate the training'), 'checks'), (re.compile(r'Explicit CUDA smoke test'), 'checks'),
                 (re.compile(r'Train the detector at '), 'detector'), (re.compile(r'Evaluate the detector at '), 'evaluate'))
# Which numbered sub-steps an action's log has: (how many, their steps; None: known by their titles, TITLES). Every
# other action: run_v100_baseline.sh.
SUBSTEPS = {'train_classifier': (7, CLASSIFIER_PIPELINE), 'train_cutter': (None, CUTTER_PIPELINE)}
TITLES = {'train_cutter': CUTTER_TITLES}
MARKERS = (
    (re.compile(r'^== a\+b\. '), 'gpu_env', ('import', 'split')),
    (re.compile(r'^== a\. importo'), 'import', ()),
    (re.compile(r'^== b\. '), 'split', ()),
    (re.compile(r'^== c\. '), 'gpu_env', ()),
    (re.compile(r'^== a\. collego'), 'link', ()),
    (re.compile(r'^== b\+c\. '), 'frames', ()),
)
SUBSTEP = re.compile(r'^== (\d+)/(\d+) ')
EPOCH = re.compile(r'^(detector|reader|classifier) epoch (\d+)/(\d+)\b')
# "classifier stops early after epoch 12: ...": the training ends before its planned epochs.
EARLY = re.compile(r'^(detector|reader|classifier) stops early after epoch (\d+)\b')
# The values of an epoch line: "train_loss 0.5000 val_loss 0.6000 ... 40s" (the detector also writes
# val_recall, its validation recall at score 0.1, which chooses its best epoch; the classifier with "not a sign"
# val_sign_accuracy, the sign crops read right, and val_rejection, the "not a sign" crops it rejects: their mean,
# the balanced accuracy, chooses its best epoch).
METRIC = re.compile(r'\b(train_loss|val_loss|train_accuracy|val_accuracy|val_recall|val_sign_accuracy|val_rejection|'
                    r'val_balanced_accuracy) (-?[0-9.]+(?:e-?[0-9]+)?)\b')
SECONDS = re.compile(r'\b([0-9]+)s$')
FETCHED = re.compile(r'^== (\S+)--(\d+) -> ')
ON_COMMA = re.compile(r'^(\S+) sul comma: ')
FINISHED = re.compile(r'^== finito con codice ')
# A line of JSON printed in the log (the report of the CUDA smoke test, ...): a brace, a bracket, a quoted key
# or a bare value. Never the last line shown: a lone "}" says nothing. A frame counter ("[12/60] 3/27: 2 signs ...",
# nnslr ai-review) is no JSON.
JSON_LINE = re.compile(r'^(?!\[\d+/\d+\] )(?:[{}\[\]"]|(?:-?\d[\d.eE+-]*|true|false|null),?$)')


def last_line(text: str) -> str:
    """The last line that says something: not the start and end markers of the wrapper, not JSON, not empty."""
    for line in reversed(text.splitlines()):
        line = line.strip()
        if (line and not FINISHED.match(line) and not JSON_LINE.match(line)
                and not line.startswith(('== helpers/', '== scripts/'))):
            return line[:300]
    return ''


def progress(action: str, state: str, text: str, preparation: dict | None = None) -> dict:
    """{'steps': [{'key', 'label', 'state': done|current|todo|failed}], 'percent': 0-100 or None,
    'detail': str, 'last_line': str, 'step_progress': how far the current step is from the log alone (see
    _inside_step) or None, 'curves': {'detector'|'reader'|'classifier': {'epochs': total, ('stopped_early': epoch,)
    'points': [{'epoch', 'train_loss', 'val_loss', ('train_accuracy', 'val_accuracy',) ('val_recall',)
    ('val_sign_accuracy', 'val_rejection', 'val_balanced_accuracy',) 'seconds'}]}}} for an action in state running,
    ok, failed or interrupted."""
    steps = STEPS.get(action, ())
    keys = [key for key, _, _ in steps]
    current = keys[0] if keys else ''
    done: set[str] = set()
    epoch: tuple[str, int, int] | None = None
    # The numbered sub-steps of the action's script, and the last one written: its number and its title.
    count, pipeline = SUBSTEPS.get(action, (10, PIPELINE))
    substep: tuple[int, str] | None = None
    # The networks that stopped early, and after which epoch.
    stopped: dict[str, int] = {}
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
        if match:
            number, title = int(match.group(1)), line[match.end():].strip(' =')
            if count is None:
                step = next((step for pattern, step in TITLES.get(action, ()) if pattern.match(title)), None)
            else:
                step = pipeline.get(number) if int(match.group(2)) == count else None
            if step in keys:
                current = step
                substep = (number, title)
        match = EPOCH.match(line)
        if match:
            epoch = (match.group(1), int(match.group(2)), int(match.group(3)))
            point: dict = {'epoch': epoch[1]}
            point.update((name, float(value)) for name, value in METRIC.findall(line))
            if 'val_sign_accuracy' in point and 'val_rejection' in point and 'val_balanced_accuracy' not in point:
                # The measure that chooses the classifier's best epoch: the mean of the two.
                point['val_balanced_accuracy'] = (point['val_sign_accuracy'] + point['val_rejection']) / 2
            seconds = SECONDS.search(line)
            if seconds:
                point['seconds'] = int(seconds.group(1))
            curve = curves.setdefault(epoch[0], {'epochs': epoch[2], 'points': []})
            curve['epochs'] = epoch[2]
            # A resumed run writes its epochs again: the last line of an epoch wins.
            curve['points'] = [p for p in curve['points'] if p['epoch'] != epoch[1]] + [point]
            if stopped.pop(epoch[0], None) is not None:
                curve.pop('stopped_early', None)
        match = EARLY.match(line)
        if match:
            stopped[match.group(1)] = int(match.group(2))
            if match.group(1) in curves:
                curves[match.group(1)]['stopped_early'] = int(match.group(2))
        if FETCHED.match(line):
            fetched += 1
        match = ON_COMMA.match(line)
        if match:
            route = match.group(1)
    if current in keys:
        done.update(keys[:keys.index(current)])

    inside = _inside_step(current, epoch, substep, pipeline, stopped) if current in keys else None
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
    elif action in WEIGHTED:
        total = sum(weight for _, _, weight in steps)
        reached = sum(weight for key, _, weight in steps if key in done)
        if inside is not None:
            weight = dict((key, weight) for key, _, weight in steps)[current]
            reached += weight * inside['done'] / inside['total']
        percent = round(reached * 100 / total, 1)
    elif action == 'prepare' and preparation is not None and preparation.get('percent') is not None:
        percent = float(preparation['percent'])

    if epoch is not None and epoch[0] == current and state == 'running' and epoch[0] in stopped:
        detail = epoch[0] + ' · stopped early after epoch ' + str(stopped[epoch[0]])
    elif epoch is not None and epoch[0] == current and state == 'running':
        detail = epoch[0] + ' · epoch ' + str(epoch[1]) + ' of ' + str(epoch[2])
    elif action == 'prepare' and preparation is not None and preparation.get('routes_total'):
        detail = str(preparation['routes_done']) + ' of ' + str(preparation['routes_total']) + ' routes ready'
    elif action == 'fetch' and (fetched or route):
        detail = (str(fetched) + (' segment downloaded' if fetched == 1 else ' segments downloaded')
                  + (' · ' + route if route and state == 'running' else ''))
    return {'steps': rows, 'percent': percent, 'detail': detail, 'last_line': last_line(text),
            'step_progress': inside if state == 'running' and current in keys else None, 'curves': curves}


def _inside_step(current: str, epoch: tuple[str, int, int] | None, substep: tuple[int, str] | None,
                 pipeline: dict[int, str] = PIPELINE, stopped: dict[str, int] | None = None) -> dict | None:
    """{'done', 'total', 'percent', 'text'} of the current step from its log: the epochs written by a training
    (all of them once it stopped early), or which of its numbered sub-steps it is in (the checks of
    scripts/run_v100_baseline.sh are three, the export and the bundle two: the second of three is at 33%). None
    when the log says nothing more."""
    if epoch is not None and epoch[0] == current and stopped and current in stopped:
        return {'done': 1, 'total': 1, 'percent': 100.0, 'text': 'stopped early after epoch ' + str(stopped[current])}
    if epoch is not None and epoch[0] == current and epoch[2]:
        done, total = min(epoch[1], epoch[2]), epoch[2]
        return {'done': done, 'total': total, 'percent': round(done * 100 / total, 1),
                'text': 'epoch ' + str(epoch[1]) + ' of ' + str(total)}
    parts = sorted(number for number, step in pipeline.items() if step == current)
    if substep is None or substep[0] not in parts or len(parts) < 2:
        return None
    done = parts.index(substep[0])
    return {'done': done, 'total': len(parts), 'percent': round(done * 100 / len(parts), 1),
            'text': substep[1] + ' · ' + str(done + 1) + ' of ' + str(len(parts))}


# The tasks that report their own progress (nnslr_tools.progress) and the step they belong to.
TASK_STEPS = {'import': 'import', 'validate': 'split', 'split': 'split', 'detector': 'detector',
              'reader': 'reader', 'evaluate': 'evaluate', 'hard_examples': 'export', 'export': 'export',
              # scripts/run_classifier.sh: the mining of the cutter's false boxes, the classifier, the chain.
              'mine': 'mine', 'classifier': 'classifier', 'chain': 'chain',
              # [ai-reviewer] - START
              # nnslr ai-review and nnslr measure-ai-reviewer: a frame at a time.
              'ai_review': 'review', 'ai_measure': 'measure'}
              # [ai-reviewer] - END


def with_step_progress(result: dict, action: str, record: dict | None, setup_line: str = '',
                       download: dict | None = None) -> dict:
    """Adds 'step_progress' ({'done', 'total', 'percent', 'text'}) for the current step of a running job:
    from the last report of a command (when it belongs to that step) or, while the GPU environment is
    made, the last line of its log; otherwise what the log alone says (progress(): epochs, sub-steps). With a
    count, the job's percent counts it too."""
    current = next((step['key'] for step in result['steps'] if step['state'] == 'current'), None)
    if current is None:
        return result
    step_progress = result.get('step_progress')
    if record is not None and TASK_STEPS.get(record.get('task')) == current and record.get('total'):
        done, total = int(record['done']), int(record['total'])
        step_progress = {'done': done, 'total': total, 'percent': round(min(done, total) * 100 / total, 1),
                         'text': str(record.get('text') or '')}
        steps = STEPS.get(action, ())
        if action in WEIGHTED and steps:
            weights = {key: weight for key, _, weight in steps}
            reached = sum(weights[step['key']] for step in result['steps'] if step['state'] == 'done')
            result = {**result, 'percent': round((reached + weights[current] * min(done, total) / total) * 100
                                                 / sum(weights.values()), 1)}
    elif current == 'gpu_env' and download is not None and download['written']:
        # The file pip is downloading: the GPU environment has several, of sizes known only when they start.
        written, total = download['written'], download['total']
        step_progress = {'done': round(written / 1e6), 'total': round(total / 1e6),
                         'percent': round(min(written, total) * 100 / total, 1),
                         'text': f"Downloading {download['name']} · {written / 1e6:.0f} of {total / 1e6:.0f} MB"}
    elif current == 'gpu_env' and setup_line:
        step_progress = {'done': 0, 'total': 0, 'percent': None, 'text': setup_line[:200]}
    return {**result, 'step_progress': step_progress}
