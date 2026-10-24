# [control-panel] - START
"""The training runs of this server, for the Evaluate tab of the site: what each run under
<data root>/runs/<run>/ is (how it started, how far it got) and how its networks did on the test partition.

Everything is read from what the training writes (scripts/run_v100_baseline.sh, nnslr_tools.training,
nnslr_tools.evaluation) and from the log of helpers/3_train.sh (<data root>/logs/train-<run>.log):

- ``detector/training-plan.json``, ``reader/training-plan.json``: the dataset and split of the run (the split
  file itself is the one the plan names, under <data root>/splits/);
- ``detector/result.json``, ``reader/result.json``: a network trained to the end (history, initialization);
- ``evaluation/detector-test.json``, ``evaluation/reader-test.json``: the measures on the test partition,
  with every test image and crop;
- ``export/detector.onnx``, ``export/reader.onnx``, ``bundle/sha256sums.json``: the last steps;
- ``experiment-b/<variant>/result.json``: experiment B (nnslr_tools.context_experiment), read defensively.

Every file is reduced once to a small digest, kept while its size and mtime stay the same: the evaluation
reports and the training plans (megabytes, a row per sample) are never parsed again at every request.
Run and variant names follow a strict pattern, and a file is read only when its real path (after the
symlinks) is inside the run's own folder. Nothing here writes anything.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from nnslr_tools.job_progress import progress

# The name of a run (helpers/3_train.sh: letters, digits, . _ -), never "." or "..".
RUN_NAME = re.compile(r'[A-Za-z0-9][A-Za-z0-9._-]{0,99}')
# helpers/3_train.sh names a run by the time it starts: 2026-10-24_1452.
TIME_NAME = re.compile(r'(\d{4})-(\d{2})-(\d{2})_(\d{2})(\d{2})')
LOG_TAIL_BYTES = 4 * 1024 * 1024
HUMAN = 'human_review'

# The steps of a run in order (scripts/run_v100_baseline.sh) and the words for each, while it runs.
STAGE_LABELS = {'checks': 'checking the dataset and the GPU', 'detector': 'training the detector',
                'reader': 'training the reader', 'evaluate': 'evaluating on the test', 'export': 'exporting to ONNX',
                'bundle': 'packaging the bundle', 'done': 'finished',
                # A folder of scripts/run_experiment_b.sh alone.
                'experiment': 'training the variants of experiment B'}

# The scripts that write into a run folder, named by the run in their first argument.
SCRIPTS = ((b'run_v100_baseline.sh', 'pipeline'), (b'run_experiment_b.sh', 'experiment'))
# Folders and files of the training pipeline (a folder with none of them holds only experiment B).
PIPELINE_PARTS = ('detector', 'reader', 'evaluation', 'export', 'bundle')
EXPERIMENT_EPOCH = re.compile(r'^experiment_b_(\w+) epoch (\d+)/(\d+)\b', re.M)

# Sign sizes for the detector's recall: the width of the reviewed box in pixels of the frame.
SIZE_BUCKETS = (('lt12', '< 12 px', 0.0, 12.0), ('12-20', '12–20 px', 12.0, 20.0), ('20-40', '20–40 px', 20.0, 40.0),
                ('ge40', '≥ 40 px', 40.0, math.inf))

# Experiment B: the variants in the order of the plan (docs/NEURAL_NETWORK.md, step 2).
EXPERIMENT_DIR = 'experiment-b'
VARIANTS = ('crop', 'crop_geom', 'full')
VARIANT_LABELS = {'crop': 'Crop', 'crop_geom': 'Crop + geometry', 'full': 'Crop + geometry + scene'}
# Parts of an experiment report that are not measures (lists of samples, settings).
SKIPPED_KEYS = {'samples', 'images', 'predictions', 'records', 'history', 'config', 'frames', 'rows', 'per_sample',
                'errors', 'warnings', 'limitations', 'classes', 'labels', 'schema_version', 'seed', 'batch_size',
                'learning_rate', 'weight_decay', 'num_workers', 'input_size'}
HUMAN_KEYS = {'human_reviewed', 'human'}
MAX_METRICS = 60
MAX_MATRIX = 8


class _Unreadable:
    """A file that exists but could not be digested (a broken or half-written JSON)."""


UNREADABLE = _Unreadable()


def _num(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return int(value)


def _ratio(part: int, whole: int) -> float | None:
    return part / whole if whole else None


def _iso(seconds: float | None) -> str | None:
    if seconds is None:
        return None
    return datetime.fromtimestamp(seconds).astimezone().isoformat(timespec='seconds')


def _mtime(path: Path | None) -> float | None:
    if path is None:
        return None
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def label_name(label: str) -> str:
    """'maximum_speed:60' -> 'Speed limit 60', 'variable_display:unknown' -> 'Variable display, value unknown'."""
    from nnslr_tools.panel import FAMILY_LABELS
    family, _, value = str(label).partition(':')
    name = FAMILY_LABELS.get(family, family.replace('_', ' ').capitalize())
    if value == 'unknown':
        return name + ', value unknown'
    return name + (' ' + value if value else '')


# -- the reader's report ---------------------------------------------------------------------------------
def _tables_from_samples(samples: list) -> tuple[dict, dict]:
    per_class: dict[str, dict] = {}
    confusion: dict[str, Counter] = defaultdict(Counter)
    for sample in samples:
        if not isinstance(sample, dict) or not isinstance(sample.get('truth'), str):
            continue
        truth = sample['truth']
        predicted = str(sample.get('predicted'))
        row = per_class.setdefault(truth, {'support': 0, 'correct': 0})
        row['support'] += 1
        correct = sample.get('correct')
        if correct is None:
            correct = predicted == truth
        row['correct'] += bool(correct)
        confusion[truth][predicted] += 1
    return per_class, {truth: dict(counts) for truth, counts in confusion.items()}


def _label_table(per_class: Any, confusion: Any) -> list[dict]:
    """Per label, most frequent first: count, correct, accuracy and the wrong answer it gets most."""
    if not isinstance(per_class, dict):
        return []
    confusion = confusion if isinstance(confusion, dict) else {}
    rows = []
    for label, values in per_class.items():
        if not isinstance(values, dict):
            continue
        count = _int(values.get('support')) or 0
        correct = _int(values.get('correct')) or 0
        wrong = None
        answers = confusion.get(label)
        if isinstance(answers, dict):
            mistakes = [(n, str(answer)) for answer, n in answers.items() if answer != label and _int(n)]
            if mistakes:
                top = sorted(mistakes, key=lambda pair: (-pair[0], pair[1]))[0]
                wrong = {'label': top[1], 'name': label_name(top[1]), 'count': int(top[0])}
        rows.append({'label': str(label), 'name': label_name(label), 'count': count, 'correct': correct,
                     'accuracy': _ratio(correct, count), 'wrong': wrong})
    return sorted(rows, key=lambda row: (-row['count'], row['label']))


def reader_digest(report: Any) -> dict:
    """The reader's evaluation (nnslr_tools.evaluation.evaluate_reader) without its list of samples."""
    if not isinstance(report, dict):
        raise ValueError('not a report')
    samples = report.get('samples') if isinstance(report.get('samples'), list) else None
    if samples is not None:
        per_class, confusion = _tables_from_samples(samples)
        human_samples = [item for item in samples if isinstance(item, dict) and item.get('label_basis') == HUMAN]
        human_tables = _tables_from_samples(human_samples) if human_samples else None
    else:
        per_class, confusion = report.get('per_class'), report.get('confusion')
        human_report = report.get('human_reviewed')
        human_tables = ((human_report.get('per_class'), human_report.get('confusion'))
                        if isinstance(human_report, dict) else None)
    labels = _label_table(per_class, confusion)
    count = _int(report.get('sample_count'))
    if count is None:
        count = sum(row['count'] for row in labels)
    correct = sum(row['correct'] for row in labels)
    accuracy = _num(report.get('accuracy'))
    if accuracy is None:
        accuracy = _ratio(correct, count)
    human = None
    human_report = report.get('human_reviewed')
    if isinstance(human_report, dict):
        human_count = _int(human_report.get('sample_count')) or 0
        human_accuracy = _num(human_report.get('accuracy'))
        human = {'samples': human_count, 'accuracy': human_accuracy,
                 'correct': round(human_accuracy * human_count) if human_accuracy is not None else None}
    classes = report.get('classes')
    return {
        'partition': str(report.get('partition') or 'test'),
        'samples': count, 'correct': correct, 'accuracy': accuracy, 'human': human,
        'counts_by_label_basis': {str(key): _int(value) or 0 for key, value in (report.get('counts_by_label_basis') or {}).items()}
        if isinstance(report.get('counts_by_label_basis'), dict) else {},
        'labels': labels,
        'labels_human': _label_table(*human_tables) if human_tables else None,
        'unseen_labels': [str(item) for item in report.get('unseen_truth_labels') or [] if isinstance(item, str)],
        'classes': len(classes) if isinstance(classes, list) else None,
        'dataset_sha256': report.get('dataset_sha256') if isinstance(report.get('dataset_sha256'), str) else None,
        'split_sha256': report.get('split_sha256') if isinstance(report.get('split_sha256'), str) else None,
    }


# -- the detector's report -------------------------------------------------------------------------------
def _box(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    numbers = [_num(item) for item in value]
    if any(item is None for item in numbers):
        return None
    return numbers  # type: ignore[return-value]


def _iou(a: list[float], b: list[float]) -> float:
    width = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = width * height
    union = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1]) + max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1]) - inter
    return 0.0 if union <= 0 else inter / union


def _match(predictions: list[tuple[int, list[float], float]], truth: list[list[float]], threshold: float) -> tuple[set, set]:
    """The greedy score-ordered matching of nnslr_tools.evaluation.match_detections, keeping which prediction
    matched which reviewed box: (matched truth indexes, matched prediction indexes)."""
    unmatched = set(range(len(truth)))
    matched_truth, matched_predictions = set(), set()
    for index, box, _ in sorted(predictions, key=lambda item: item[2], reverse=True):
        best_iou, best = max(((_iou(box, truth[t]), t) for t in unmatched), default=(0.0, -1))
        if best >= 0 and best_iou >= threshold:
            unmatched.discard(best)
            matched_truth.add(best)
            matched_predictions.add(index)
    return matched_truth, matched_predictions


def _bucket(width: float) -> str:
    for key, _, low, high in SIZE_BUCKETS:
        if low <= width < high:
            return key
    return SIZE_BUCKETS[-1][0]


def _size_table(images: list[dict], threshold: float) -> list[dict] | None:
    """Recall by the width of the reviewed box, and the false boxes by their own width; None without boxes."""
    rows = {key: {'key': key, 'label': label, 'truth': 0, 'found': 0, 'false_boxes': 0} for key, label, _, _ in SIZE_BUCKETS}
    seen = False
    for image in images:
        truth_raw = image.get('truth_boxes')
        if not isinstance(truth_raw, list):
            continue
        seen = True
        truth = [(index, box) for index, box in ((index, _box(item)) for index, item in enumerate(truth_raw)) if box is not None]
        predictions = []
        for index, item in enumerate(image.get('predictions') or []):
            box = _box(item.get('box')) if isinstance(item, dict) else None
            score = _num(item.get('score')) if isinstance(item, dict) else None
            if box is not None and score is not None:
                predictions.append((index, box, score))
        local_truth, local_predictions = _match(predictions, [box for _, box in truth], threshold)
        recorded = image.get('matches')
        if isinstance(recorded, list) and all(isinstance(item, dict) for item in recorded):
            # The evaluation's own matching, by the index of the reviewed box.
            found = {item.get('truth_index') for item in recorded if isinstance(item.get('truth_index'), int)}
        else:
            found = {truth[position][0] for position in local_truth}
        for index, box in truth:
            row = rows[_bucket(box[2] - box[0])]
            row['truth'] += 1
            row['found'] += index in found
        for index, box, _ in predictions:
            if index not in local_predictions:
                rows[_bucket(box[2] - box[0])]['false_boxes'] += 1
    if not seen:
        return None
    return [{**row, 'missed': row['truth'] - row['found'], 'recall': _ratio(row['found'], row['truth'])} for row in rows.values()]


def _counts(found: int, missed: int, false_boxes: int, images: int | None) -> dict:
    precision = _ratio(found, found + false_boxes)
    recall = _ratio(found, found + missed)
    f1 = None
    if precision is not None and recall is not None:
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {'images': images, 'truth_boxes': found + missed, 'found': found, 'missed': missed, 'false_boxes': false_boxes,
            'recall': recall, 'precision': precision, 'f1': f1}


def detector_digest(report: Any) -> dict:
    """The detector's evaluation (nnslr_tools.evaluation.evaluate_detector) without its list of images."""
    if not isinstance(report, dict):
        raise ValueError('not a report')
    images = [item for item in report.get('images') or [] if isinstance(item, dict)] if isinstance(report.get('images'), list) else []
    threshold = _num(report.get('iou_threshold')) or 0.5
    found, missed, false_boxes = (_int(report.get(key)) for key in ('true_positives', 'false_negatives', 'false_positives'))
    if found is None or missed is None or false_boxes is None:
        found = sum(_int(item.get('true_positives')) or 0 for item in images)
        missed = sum(_int(item.get('false_negatives')) or 0 for item in images)
        false_boxes = sum(_int(item.get('false_positives')) or 0 for item in images)
    image_count = _int(report.get('image_count'))
    overall = _counts(found, missed, false_boxes, len(images) if image_count is None else image_count)
    human = None
    human_report = report.get('human_reviewed')
    if isinstance(human_report, dict):
        human = _counts(_int(human_report.get('true_positives')) or 0, _int(human_report.get('false_negatives')) or 0,
                        _int(human_report.get('false_positives')) or 0, _int(human_report.get('image_count')))
    routes: dict[str, dict] = {}
    for image in images:
        row = routes.setdefault(str(image.get('route_id') or '?'), {'images': 0, 'found': 0, 'missed': 0, 'false_boxes': 0})
        row['images'] += 1
        row['found'] += _int(image.get('true_positives')) or 0
        row['missed'] += _int(image.get('false_negatives')) or 0
        row['false_boxes'] += _int(image.get('false_positives')) or 0
    human_images = [image for image in images if image.get('label_basis') == HUMAN]
    return {
        'partition': str(report.get('partition') or 'test'),
        **overall,
        'false_per_image': _num(report.get('false_positives_per_image')),
        'score_threshold': _num(report.get('score_threshold')),
        'iou_threshold': threshold,
        'human': human,
        'counts_by_label_basis': {str(key): _int(value) or 0 for key, value in (report.get('counts_by_label_basis') or {}).items()}
        if isinstance(report.get('counts_by_label_basis'), dict) else {},
        'sizes': _size_table(images, threshold),
        'sizes_human': _size_table(human_images, threshold) if human_images else None,
        'routes': [{'route': name, **_counts(row['found'], row['missed'], row['false_boxes'], row['images'])}
                   for name, row in sorted(routes.items())],
        'dataset_sha256': report.get('dataset_sha256') if isinstance(report.get('dataset_sha256'), str) else None,
        'split_sha256': report.get('split_sha256') if isinstance(report.get('split_sha256'), str) else None,
    }


# -- training results, plans, the split, the log ----------------------------------------------------------
def result_digest(result: Any) -> dict:
    """result.json of a network (nnslr_tools.training): best value and epoch, epochs, initialization, history."""
    if not isinstance(result, dict):
        raise ValueError('not a result')
    history = []
    for item in result.get('history') or []:
        if not isinstance(item, dict) or _int(item.get('epoch')) is None:
            continue
        point: dict[str, float | int] = {'epoch': int(item['epoch'])}
        for source, target in (('train_loss', 'train_loss'), ('validation_loss', 'val_loss'),
                               ('train_accuracy', 'train_accuracy'), ('validation_accuracy', 'val_accuracy')):
            value = _num(item.get(source))
            if value is not None:
                point[target] = value
        history.append(point)
    config = result.get('config') if isinstance(result.get('config'), dict) else {}
    best_loss = _num(result.get('best_validation_loss'))
    best_accuracy = _num(result.get('best_validation_accuracy'))
    best_epoch = None
    if best_accuracy is not None:
        best_epoch = next((p['epoch'] for p in history if p.get('val_accuracy') == best_accuracy), None)
    elif best_loss is not None:
        best_epoch = next((p['epoch'] for p in history if p.get('val_loss') == best_loss), None)
    gpu = result.get('gpu') if isinstance(result.get('gpu'), dict) else None
    classes = result.get('classes')
    return {
        'architecture': str(result.get('architecture') or ''),
        'initialization': str(result.get('initialization') or '') or None,
        'epochs': _int(config.get('epochs')) or (history[-1]['epoch'] if history else None),
        'best_validation_loss': best_loss, 'best_validation_accuracy': best_accuracy, 'best_epoch': best_epoch,
        'classes': len(classes) if isinstance(classes, list) else None,
        'device': str(result.get('device') or '') or None,
        'gpu': str(gpu.get('name')) if gpu and gpu.get('name') else None,
        'history': history,
    }


def _counts_map(value: Any) -> dict[str, int]:
    return {str(key): _int(count) or 0 for key, count in value.items()} if isinstance(value, dict) else {}


def plan_digest(plan: Any) -> dict:
    """training-plan.json of a network: dataset and split identity and the counts, without its rows."""
    if not isinstance(plan, dict):
        raise ValueError('not a plan')
    text = lambda key: plan.get(key) if isinstance(plan.get(key), str) else None  # noqa: E731
    classes = plan.get('classes')
    return {'task': text('task'), 'dataset_kind': text('dataset_kind'), 'dataset_sha256': text('dataset_sha256'),
            'split_sha256': text('split_sha256'), 'split_path': text('split_path'),
            'counts_by_split': _counts_map(plan.get('counts_by_split')),
            'boxes_by_split': _counts_map(plan.get('boxes_by_split')),
            'negative_frames_by_split': _counts_map(plan.get('negative_frames_by_split')),
            'classes': len(classes) if isinstance(classes, list) else None}


def split_digest(split: Any) -> dict:
    """A split file (data/splits/<sha>.json) without its assignments."""
    if not isinstance(split, dict):
        raise ValueError('not a split')
    routes = split.get('routes') if isinstance(split.get('routes'), dict) else {}
    return {
        'policy': split.get('policy') if isinstance(split.get('policy'), str) else None,
        'seed': _int(split.get('seed')),
        'grouping': [str(item) for item in split.get('grouping') or [] if isinstance(item, str)],
        'test_routes': [str(item) for item in split.get('test_routes') or [] if isinstance(item, str)],
        'routes': {str(key): [str(item) for item in value if isinstance(item, str)]
                   for key, value in routes.items() if isinstance(value, list)},
        'counts': _counts_map(split.get('counts')),
        'human_reviewed_counts': _counts_map(split.get('human_reviewed_counts')),
        'warnings': [str(item) for item in split.get('warnings') or [] if isinstance(item, str)],
    }


MODE_LINE = re.compile(r'^training mode: (\w+)', re.M)
WEIGHTS_LINE = re.compile(r'^continue (?:detector|reader) from weights: (.+?)\s*$', re.M)
ERROR_LINE = re.compile(r'^(?:ERROR: |error: |[A-Za-z_][\w.]*(?:Error|Exception): )')
FROM_RUN = re.compile(r'/runs/([A-Za-z0-9][A-Za-z0-9._-]{0,99})/(?:detector|reader)/')


def log_digest(text: str) -> dict:
    """The log of helpers/3_train.sh for a run: the training mode(s), where the weights came from, the curves,
    the step it reached, the last error line."""
    modes = MODE_LINE.findall(text)
    weights = WEIGHTS_LINE.findall(text)
    error = None
    for line in reversed(text.splitlines()[-400:]):
        line = line.strip()
        if ERROR_LINE.match(line):
            error = line[:300]
            break
    running = progress('train', 'running', text)
    return {'modes': modes, 'weights': weights[0] if weights else None, 'error': error,
            'curves': running['curves'], 'detail': running['detail'], 'percent': running['percent'],
            'complete': 'Pipeline complete:' in text}


def experiment_log_digest(text: str) -> dict:
    """<run>/experiment-b/<variant>.log: the last epoch line and the last error line."""
    epochs = EXPERIMENT_EPOCH.findall(text)
    error = None
    for line in reversed(text.splitlines()[-400:]):
        line = line.strip()
        if ERROR_LINE.match(line):
            error = line[:300]
            break
    last = epochs[-1] if epochs else None
    return {'variant': last[0] if last else None, 'epoch': int(last[1]) if last else None,
            'epochs': int(last[2]) if last else None, 'error': error}


# -- experiment B -----------------------------------------------------------------------------------------
# result.json of nnslr_tools.context_experiment: metrics.<validation|test> = {loss, box_count, reading: {accuracy,
# expected_calibration_error, ...}, ownership: {accuracy, macro_f1, abstain_rate, decided_accuracy, other_road:
# {precision, recall, f1}, confusion, expected_calibration_error, ...}, human_reviewed: the same on the boxes a
# person confirmed, or None}. The rows of the comparison: (key, label, path in a partition, unit, better).
EXPERIMENT_ROWS = (
    ('reading_accuracy', 'Reading accuracy', ('reading', 'accuracy'), 'percent', 'higher'),
    ('other_road_precision', 'Another road: precision', ('ownership', 'other_road', 'precision'), 'percent', 'higher'),
    ('other_road_recall', 'Another road: recall', ('ownership', 'other_road', 'recall'), 'percent', 'higher'),
    ('other_road_f1', 'Another road: F1', ('ownership', 'other_road', 'f1'), 'percent', 'higher'),
    ('ownership_macro_f1', 'Whom it applies to: macro-F1', ('ownership', 'macro_f1'), 'percent', 'higher'),
    ('ownership_accuracy', 'Whom it applies to: accuracy', ('ownership', 'accuracy'), 'percent', 'higher'),
    ('decided_accuracy', 'Accuracy when it decides', ('ownership', 'decided_accuracy'), 'percent', 'higher'),
    ('abstain_rate', 'Abstains (unclear)', ('ownership', 'abstain_rate'), 'percent', None),
    ('reading_calibration', 'Reading calibration error', ('reading', 'expected_calibration_error'), 'percent', 'lower'),
    ('ownership_calibration', 'Ownership calibration error', ('ownership', 'expected_calibration_error'), 'percent', 'lower'),
    ('loss', 'Loss', ('loss',), 'number', 'lower'),
    ('box_count', 'Boxes', ('box_count',), 'count', None),
    ('ownership_boxes', 'Boxes with whom it applies to', ('ownership', 'sample_count'), 'count', None),
)
EXPERIMENT_PARTITIONS = ('test', 'validation')
OWNERSHIP_ORDER = ('own_road', 'other_road', 'unresolved')
OWNERSHIP_NAMES = {'own_road': 'Ours', 'other_road': 'Another', 'unresolved': 'Unclear'}
INPUT_NAMES = {'crop': 'crop', 'box_geometry': 'box position', 'car_speed': 'car speed', 'scene': 'scene'}


def _dig(value: Any, path: tuple[str, ...]) -> Any:
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _matrix(value: Any) -> dict | None:
    """A confusion matrix as {'labels', 'matrix'} (rows: truth, columns: answer), from a dict of dicts, a dict
    with labels and matrix, or a list of rows."""
    labels: list[str]
    rows: list
    if isinstance(value, dict) and isinstance(value.get('matrix'), list):
        rows = value['matrix']
        labels = [str(item) for item in value.get('labels') or range(len(rows))]
    elif isinstance(value, dict) and value and all(isinstance(item, dict) for item in value.values()):
        names = {str(key) for key in value} | {str(key) for item in value.values() for key in item}
        labels = [name for name in OWNERSHIP_ORDER if name in names] + sorted(names - set(OWNERSHIP_ORDER))
        rows = [[(value.get(truth) or {}).get(answer, 0) for answer in labels] for truth in labels]
    elif isinstance(value, list) and value and all(isinstance(row, list) for row in value):
        rows = value
        labels = [str(index) for index in range(len(rows))]
    else:
        return None
    if not labels or len(labels) > MAX_MATRIX or len(rows) != len(labels):
        return None
    cleaned = []
    for row in rows:
        if not isinstance(row, list) or len(row) != len(labels):
            return None
        cleaned.append([_num(item) or 0 for item in row])
    return {'labels': labels, 'names': [OWNERSHIP_NAMES.get(label, _words(label)) for label in labels], 'matrix': cleaned}


def _flatten(value: Any, path: tuple[str, ...], human: bool, metrics: dict, depth: int = 0) -> None:
    """Every number of an unknown report by its path; a human_reviewed object (anywhere) gives the 'human'
    value of the same path."""
    if depth > 5:
        return
    number = _num(value)
    if number is not None:
        if path:
            metrics.setdefault(path, {})['human' if human else 'all'] = number
        return
    if not isinstance(value, dict):
        return
    for key, item in value.items():
        if not isinstance(key, str) or key in SKIPPED_KEYS or 'confusion' in key.lower():
            continue
        if key in HUMAN_KEYS:
            _flatten(item, path, True, metrics, depth + 1)
        else:
            _flatten(item, path + (key,), human, metrics, depth + 1)


def _words(segment: str) -> str:
    words = segment.replace('_', ' ').replace('-', ' ').split()
    fixed = {'f1': 'F1', 'ece': 'ECE', 'iou': 'IoU', 'auc': 'AUC'}
    return ' '.join(fixed.get(word.lower(), word) for word in words)


def _metric_label(path: tuple[str, ...]) -> str:
    text = ' › '.join(_words(segment) for segment in path)
    return text[:1].upper() + text[1:]


LOWER_IS_BETTER = ('error', 'loss', 'calibration', 'brier', 'false', 'missed', 'wrong')
HIGHER_IS_BETTER = ('accuracy', 'precision', 'recall', 'f1', 'auc', 'correct')


def _metric_kind(path: tuple[str, ...], values: list[float]) -> tuple[str | None, str]:
    """(better: higher | lower | None, unit: percent | count | number) of a measure found by its path."""
    text = '.'.join(path).lower()
    better = None
    if any(word in text for word in LOWER_IS_BETTER):
        better = 'lower'
    elif any(word in text for word in HIGHER_IS_BETTER):
        better = 'higher'
    if 'count' in text or (values and all(float(value).is_integer() and value > 1 for value in values)):
        return better, 'count'
    if better is not None and values and all(0 <= value <= 1 for value in values):
        return better, 'percent'
    return better, 'number'


def experiment_digest(result: Any) -> dict:
    """One variant's result.json: the rows of EXPERIMENT_ROWS by partition (all boxes and human-reviewed), the
    ownership confusion, the variant's inputs and chosen epoch. A report of another shape: every number of it
    under the partition 'result' (generic rows)."""
    if not isinstance(result, dict):
        raise ValueError('not a result')
    partitions: dict[str, dict] = {}
    matrices: dict[str, dict] = {}
    generic: dict[str, dict] = {}
    metrics = result.get('metrics')
    if isinstance(metrics, dict):
        for name, report in metrics.items():
            if not isinstance(name, str):
                continue
            values: dict[str, dict] = {}
            human = report.get('human_reviewed') if isinstance(report, dict) and isinstance(report.get('human_reviewed'), dict) else None
            for key, _, path, _, _ in EXPERIMENT_ROWS:
                every = _num(_dig(report, path))
                if every is None and isinstance(report, dict):
                    every = _num(report.get(key))
                mine = _num(_dig(human, path))
                if mine is None:
                    # A human_reviewed object next to the measure, if a report puts it there.
                    mine = _num(_dig(report, path[:-1] + ('human_reviewed', path[-1])))
                if every is not None or mine is not None:
                    values[key] = {'all': every, 'human': mine}
            partitions[name] = values
            matrices[name] = {'all': _matrix(_dig(report, ('ownership', 'confusion'))),
                              'human': _matrix(_dig(human, ('ownership', 'confusion')))}
    else:
        flat: dict[tuple[str, ...], dict] = {}
        _flatten(result, (), False, flat)
        values = {}
        for path, cell in list(flat.items())[:MAX_METRICS]:
            key = '.'.join(path)
            better, unit = _metric_kind(path, [value for value in cell.values() if value is not None])
            generic[key] = {'label': _metric_label(path), 'unit': unit, 'better': better}
            values[key] = {'all': cell.get('all'), 'human': cell.get('human')}
        partitions['result'] = values
    inputs = result.get('inputs')
    text = lambda key: result.get(key) if isinstance(result.get(key), str) else None  # noqa: E731
    return {
        'partitions': partitions, 'matrices': matrices, 'generic': generic,
        'inputs': [INPUT_NAMES.get(str(item), _words(str(item))) for item in inputs] if isinstance(inputs, list) else None,
        'best_epoch': _int(result.get('best_epoch')),
        'epochs': _int(_dig(result, ('config', 'epochs'))),
        'selection_score': _num(result.get('best_validation_selection_score')),
        'selection_metric': text('selection_metric'),
        'initialization': text('initialization'),
        'dataset_sha256': text('dataset_sha256'), 'split_sha256': text('split_sha256'),
        'limit': _int(result.get('limit')),
    }


def experiment_table(digests: dict[str, dict], variants: list[str]) -> dict:
    """The variants side by side: one row per measure, a cell per variant and partition."""
    present = {name for digest in digests.values() for name in digest['partitions']}
    partitions = [name for name in EXPERIMENT_PARTITIONS if name in present] + sorted(present - set(EXPERIMENT_PARTITIONS))
    definitions = [(key, label, unit, better) for key, label, _, unit, better in EXPERIMENT_ROWS]
    seen = {key for key, _, _, _ in definitions}
    for digest in digests.values():
        for key, meta in digest['generic'].items():
            if key not in seen:
                seen.add(key)
                definitions.append((key, meta['label'], meta['unit'], meta['better']))
    empty = {'all': None, 'human': None}
    rows = []
    for key, label, unit, better in definitions:
        values = {variant: {partition: (digests[variant]['partitions'].get(partition, {}).get(key, empty)
                                        if variant in digests else empty) for partition in partitions}
                  for variant in variants}
        if any(cell['all'] is not None or cell['human'] is not None for row in values.values() for cell in row.values()):
            rows.append({'key': key, 'label': label, 'unit': unit, 'better': better, 'values': values})
    confusions = []
    if any(digest['matrices'] for digest in digests.values()):
        confusions.append({'key': 'ownership', 'label': 'Whom it applies to',
                           'variants': {variant: {partition: (digests[variant]['matrices'].get(partition, empty)
                                                              if variant in digests else empty)
                                                  for partition in partitions} for variant in variants}})
    return {'partitions': partitions, 'metrics': rows, 'confusions': confusions}


# -- the runs ---------------------------------------------------------------------------------------------
class RunsIndex:
    """The runs of <root>/runs, summarised from their files; digests kept by (mtime, size) of each file."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self._lock = threading.Lock()
        self._cache: dict[str, tuple[tuple[int, int], Any]] = {}

    # -- files -----------------------------------------------------------------------------------------
    @property
    def runs_dir(self) -> Path:
        return self.root / 'runs'

    def _run_dir(self, name: str) -> Path | None:
        """The folder of a run: a strict name, a real directory directly inside <root>/runs."""
        if not isinstance(name, str) or not RUN_NAME.fullmatch(name):
            return None
        path = self.runs_dir / name
        try:
            if path.is_symlink() or not path.is_dir():
                return None
            if path.resolve().parent != self.runs_dir.resolve() or not path.resolve().is_relative_to(self.root.resolve()):
                return None
        except OSError:
            return None
        return path

    @staticmethod
    def _inside(path: Path, base: Path) -> Path | None:
        """path when it is a regular file whose real path is inside base, else None."""
        try:
            real = path.resolve(strict=True)
            if not real.is_relative_to(base.resolve()) or not real.is_file():
                return None
        except (OSError, RuntimeError):
            return None
        return real

    def _load(self, path: Path | None, digest: Callable[[Any], Any], *, text: bool = False) -> Any:
        """The digest of a file (None when missing, UNREADABLE when it cannot be parsed), parsed again only
        when its mtime or size changed."""
        if path is None:
            return None
        try:
            info = path.stat()
        except OSError:
            return None
        key = (info.st_mtime_ns, info.st_size)
        with self._lock:
            hit = self._cache.get(str(path))
            if hit is not None and hit[0] == key:
                return hit[1]
            try:
                if text:
                    with path.open('rb') as source:
                        source.seek(max(0, info.st_size - LOG_TAIL_BYTES))
                        value = digest(source.read().decode('utf-8', 'replace'))
                else:
                    value = digest(_read_json(path))
            except (OSError, ValueError, TypeError, KeyError, AttributeError, RecursionError):
                value = UNREADABLE
            self._cache[str(path)] = (key, value)
            return value

    def _file(self, run: Path, relative: str) -> Path | None:
        return self._inside(run / relative, run)

    def _log(self, name: str) -> Path | None:
        return self._inside(self.root / 'logs' / ('train-' + name + '.log'), self.root / 'logs')

    def _split_file(self, plan: dict | None) -> Path | None:
        """The split a plan names, only inside <root>/splits; else the content-addressed file of its sha."""
        splits = self.root / 'splits'
        if plan is None:
            return None
        if plan.get('split_path'):
            found = self._inside(Path(plan['split_path']), splits)
            if found is not None:
                return found
        sha = plan.get('split_sha256')
        if isinstance(sha, str) and re.fullmatch(r'[0-9a-f]{64}', sha):
            return self._inside(splits / (sha + '.json'), splits)
        return None

    # -- what is running -------------------------------------------------------------------------------
    def running_runs(self) -> dict[str, set[str]]:
        """Runs with work running on this data root, and which: 'pipeline' (scripts/run_v100_baseline.sh <run>)
        and/or 'experiment' (scripts/run_experiment_b.sh <run>)."""
        found: dict[str, set[str]] = {}
        proc = Path('/proc')
        if not proc.is_dir():
            return found
        root = self.root.resolve()
        for entry in proc.iterdir():
            if not entry.name.isdigit():
                continue
            try:
                arguments = (entry / 'cmdline').read_bytes().split(b'\0')
            except OSError:
                continue
            for index, argument in enumerate(arguments[:-1]):
                kind = next((kind for script, kind in SCRIPTS if argument.endswith(script)), None)
                if kind is None:
                    continue
                name = arguments[index + 1].decode('utf-8', 'replace')
                if RUN_NAME.fullmatch(name) and self._same_root(entry, argument.decode('utf-8', 'replace'), root):
                    found.setdefault(name, set()).add(kind)
        return found

    @staticmethod
    def _same_root(process: Path, script: str, root: Path) -> bool:
        """The pipeline writes to NNSLR_DATA_ROOT, else to data/ of its checkout."""
        try:
            environment = (process / 'environ').read_bytes().split(b'\0')
        except OSError:
            return True
        for item in environment:
            if item.startswith(b'NNSLR_DATA_ROOT='):
                return Path(item[len(b'NNSLR_DATA_ROOT='):].decode('utf-8', 'replace')).resolve() == root
        try:
            script_path = Path(script)
            if not script_path.is_absolute():
                script_path = Path(os.readlink(process / 'cwd')) / script_path
            return (script_path.resolve().parents[1] / 'data').resolve() == root
        except (OSError, IndexError):
            return True

    # -- one run ---------------------------------------------------------------------------------------
    def _gather(self, name: str, run: Path, running: dict[str, set[str]]) -> dict:
        files = {
            'detector_plan': self._file(run, 'detector/training-plan.json'),
            'reader_plan': self._file(run, 'reader/training-plan.json'),
            'detector_result': self._file(run, 'detector/result.json'),
            'reader_result': self._file(run, 'reader/result.json'),
            'detector_eval': self._file(run, 'evaluation/detector-test.json'),
            'reader_eval': self._file(run, 'evaluation/reader-test.json'),
            'detector_onnx': self._file(run, 'export/detector.onnx'),
            'reader_onnx': self._file(run, 'export/reader.onnx'),
            'bundle': self._file(run, 'bundle/sha256sums.json'),
        }
        digests = {
            'detector_plan': self._load(files['detector_plan'], plan_digest),
            'reader_plan': self._load(files['reader_plan'], plan_digest),
            'detector_result': self._load(files['detector_result'], result_digest),
            'reader_result': self._load(files['reader_result'], result_digest),
            'detector_eval': self._load(files['detector_eval'], detector_digest),
            'reader_eval': self._load(files['reader_eval'], reader_digest),
        }
        log_path = self._log(name)
        log = self._load(log_path, log_digest, text=True)
        log = None if log is UNREADABLE else log
        problems = [key for key, value in digests.items() if value is UNREADABLE]
        digests = {key: (None if value is UNREADABLE else value) for key, value in digests.items()}

        kinds = running.get(name, set())
        variants = self._variants(run)
        # A folder of scripts/run_experiment_b.sh alone has experiment-b/ and nothing of the pipeline.
        pipeline = log_path is not None or any((run / part).exists() for part in PIPELINE_PARTS)
        kind = 'training' if pipeline or not (run / EXPERIMENT_DIR).is_dir() else 'experiment'
        experiment_log = None
        if kind == 'experiment' or 'experiment' in kinds:
            experiment_log = self._experiment_log(run)
        if kind == 'experiment':
            done = sum(1 for variant, path in variants if variant in VARIANTS and path is not None)
            stage = 'done' if done == len(VARIANTS) else 'experiment'
        elif not files['detector_plan'] and not files['detector_result']:
            stage = 'checks'
        elif not files['detector_result']:
            stage = 'detector'
        elif not files['reader_result']:
            stage = 'reader'
        elif not (files['detector_eval'] and files['reader_eval']):
            stage = 'evaluate'
        elif not (files['detector_onnx'] and files['reader_onnx']):
            stage = 'export'
        elif not files['bundle']:
            stage = 'bundle'
        else:
            stage = 'done'
        is_running = ('pipeline' if kind == 'training' else 'experiment') in kinds
        state = 'running' if is_running else 'finished' if stage == 'done' else 'incomplete'

        experiment_files = [path for _, path in variants]
        experiment_files += [self._inside(run / EXPERIMENT_DIR / (variant + '.log'), run) for variant in VARIANTS]
        times = [time for time in (_mtime(path) for path in [*files.values(), log_path, *experiment_files]) if time is not None]
        started = None
        match = TIME_NAME.fullmatch(name)
        if match:
            try:
                started = datetime(*(int(part) for part in match.groups())).astimezone().isoformat(timespec='seconds')
            except ValueError:
                started = None
        if started is None and kind == 'training':
            started = _iso(_mtime(files['detector_plan']) or _mtime(run))
        if started is None and kind == 'experiment':
            started = _iso(_mtime(run / EXPERIMENT_DIR / 'dry-run.json') or _mtime(run))
        return {'name': name, 'run': run, 'kind': kind, 'files': files, 'digests': digests, 'log': log, 'problems': problems,
                'stage': stage, 'state': state, 'started_at': started, 'variants': variants,
                'experiment_running': 'experiment' in kinds, 'experiment_log': experiment_log,
                'updated_at': _iso(max(times)) if times else None,
                'finished_at': (_iso(_mtime(files['bundle'])) if kind == 'training' else
                                max((_iso(_mtime(path)) or '' for _, path in variants), default=None) or None)
                if stage == 'done' else None}

    def _start(self, gathered: dict) -> dict:
        """How the networks started: ImageNet, random weights, the stable model or an earlier run."""
        log = gathered['log'] or {}
        modes = log.get('modes') or []
        mode = modes[0] if modes else None
        resumed = 'resume' in modes[1:] or mode == 'resume'
        weights = log.get('weights')
        results = [gathered['digests']['detector_result'], gathered['digests']['reader_result']]
        if gathered['kind'] == 'experiment':
            results = [self._load(path, experiment_digest) for _, path in gathered['variants'] if path is not None]
        initialization = next((digest['initialization'] for digest in results
                               if isinstance(digest, dict) and digest.get('initialization')), None)
        if weights:
            from_run = FROM_RUN.search(weights)
            if from_run:
                start = {'kind': 'run', 'label': 'Continued from run ' + from_run.group(1), 'run': from_run.group(1)}
            elif '@' in weights:
                start = {'kind': 'stable', 'label': 'Continued from the stable model', 'run': None}
            else:
                start = {'kind': 'weights', 'label': 'Continued from ' + Path(weights).name, 'run': None}
            start['source'] = weights
        elif mode == 'continue':
            start = {'kind': 'stable', 'label': 'Continued from the stable model', 'run': None, 'source': None}
        elif initialization == 'imagenet1k_v1':
            start = {'kind': 'imagenet', 'label': 'Pre-trained on ImageNet', 'run': None, 'source': None}
        elif initialization == 'random':
            start = {'kind': 'random', 'label': 'Random weights', 'run': None, 'source': None}
        else:
            start = {'kind': None, 'label': 'Not recorded yet' if gathered['state'] == 'running' else 'Not recorded',
                     'run': None, 'source': None}
        return {'mode': mode, 'resumed': resumed, 'start': start}

    def _summary(self, gathered: dict) -> dict:
        digests = gathered['digests']
        log = gathered['log'] or {}
        reader = digests['reader_eval']
        detector = digests['detector_eval']
        plan = digests['detector_plan'] or digests['reader_plan'] or {}
        split_sha = ((reader or {}).get('split_sha256') or (detector or {}).get('split_sha256') or plan.get('split_sha256'))
        dataset_sha = ((reader or {}).get('dataset_sha256') or (detector or {}).get('dataset_sha256') or plan.get('dataset_sha256'))
        experiment = gathered['variants']
        stage = gathered['stage']
        state = gathered['state']
        experiment_log = gathered['experiment_log'] or {}
        experiment_progress = None
        if gathered['experiment_running']:
            experiment_progress = ' · '.join(
                ['experiment B']
                + ([experiment_log['variant']] if experiment_log.get('variant') else [])
                + (['epoch ' + str(experiment_log['epoch']) + ' of ' + str(experiment_log['epochs'])]
                   if experiment_log.get('epoch') is not None else []))
        if gathered['kind'] == 'experiment':
            first = next((self._load(path, experiment_digest) for _, path in experiment if path is not None), None)
            first = first if isinstance(first, dict) else {}
            split_sha = split_sha or first.get('split_sha256')
            dataset_sha = dataset_sha or first.get('dataset_sha256')
            progress_text, error = experiment_progress, experiment_log.get('error')
        else:
            progress_text, error = log.get('detail') or None, log.get('error')
        summary = {
            'name': gathered['name'],
            'kind': gathered['kind'],
            'state': state,
            'stage': stage,
            'stage_label': STAGE_LABELS[stage],
            'progress': progress_text if state == 'running' else None,
            'error': error if state == 'incomplete' else None,
            'experiment_running': gathered['experiment_running'],
            'experiment_progress': experiment_progress,
            'started_at': gathered['started_at'],
            'updated_at': gathered['updated_at'],
            'finished_at': gathered['finished_at'],
            **self._start(gathered),
            'dataset_sha256': dataset_sha,
            'split_sha256': split_sha,
            'evaluated': reader is not None or detector is not None,
            'reader': None if reader is None else {
                'samples': reader['samples'], 'accuracy': reader['accuracy'],
                'human_samples': (reader['human'] or {}).get('samples'), 'human_accuracy': (reader['human'] or {}).get('accuracy')},
            'detector': None if detector is None else {
                'truth_boxes': detector['truth_boxes'], 'found': detector['found'], 'missed': detector['missed'],
                'false_boxes': detector['false_boxes'], 'recall': detector['recall'], 'precision': detector['precision'],
                'f1': detector['f1'],
                'human_recall': (detector['human'] or {}).get('recall'), 'human_precision': (detector['human'] or {}).get('precision')},
            'best': {'detector_val_loss': (digests['detector_result'] or {}).get('best_validation_loss'),
                     'reader_val_accuracy': (digests['reader_result'] or {}).get('best_validation_accuracy')},
            'experiment_b': [variant for variant, path in experiment if path is not None],
            'problems': [PROBLEM_FILES[key] + ' is unreadable' for key in gathered['problems']],
        }
        summary['version'] = hashlib.sha1(json.dumps([summary, self._stat_key(gathered)], sort_keys=True, default=str)
                                          .encode()).hexdigest()[:12]
        return summary

    @staticmethod
    def _stat_key(gathered: dict) -> list:
        key = []
        for path in gathered['files'].values():
            if path is not None:
                try:
                    info = path.stat()
                    key.append((str(path), info.st_mtime_ns, info.st_size))
                except OSError:
                    pass
        return key

    def _variants(self, run: Path) -> list[tuple[str, Path | None]]:
        """Experiment B: the known variants first, then any other folder with a result.json."""
        base = run / EXPERIMENT_DIR
        names = list(VARIANTS)
        try:
            if base.is_dir() and not base.is_symlink():
                names += sorted(entry.name for entry in base.iterdir()
                                if RUN_NAME.fullmatch(entry.name) and entry.name not in VARIANTS
                                and entry.is_dir() and not entry.is_symlink())
        except OSError:
            pass
        found = [(variant, self._inside(base / variant / 'result.json', run)) for variant in names]
        return found if any(path is not None for _, path in found) else []

    def _experiment_log(self, run: Path) -> dict | None:
        """The newest <run>/experiment-b/<variant>.log: the variant training now, its epoch, the last error."""
        base = run / EXPERIMENT_DIR
        newest = None
        for variant in VARIANTS:
            path = self._inside(base / (variant + '.log'), run)
            if path is not None and (newest is None or (_mtime(path) or 0) > (_mtime(newest) or 0)):
                newest = path
        digest = self._load(newest, experiment_log_digest, text=True)
        if not isinstance(digest, dict) or newest is None:
            return None
        # Before its first epoch line, the variant is the name of its log.
        return {**digest, 'variant': digest['variant'] or newest.stem}

    def _experiment(self, run: Path) -> dict | None:
        """Experiment B of this run: the variants side by side, None when no variant has a result yet."""
        variants = self._variants(run)
        if not variants:
            return None
        digests = {}
        listed = []
        for variant, path in variants:
            digest = self._load(path, experiment_digest)
            entry = {'key': variant, 'label': VARIANT_LABELS.get(variant, _words(variant)), 'found': path is not None,
                     'unreadable': digest is UNREADABLE, 'at': _iso(_mtime(path)), 'inputs': None, 'best_epoch': None,
                     'epochs': None, 'selection_score': None, 'initialization': None, 'split_sha256': None, 'limit': None}
            if digest is not None and digest is not UNREADABLE:
                digests[variant] = digest
                entry.update({key: digest[key] for key in ('inputs', 'best_epoch', 'epochs', 'selection_score',
                                                           'initialization', 'split_sha256', 'limit')})
            listed.append(entry)
        selection = next((digest['selection_metric'] for digest in digests.values() if digest['selection_metric']), None)
        return {'variants': listed, 'selection_metric': selection,
                **experiment_table(digests, [entry['key'] for entry in listed])}

    def _dataset(self, gathered: dict) -> dict | None:
        digests = gathered['digests']
        detector_plan, reader_plan = digests['detector_plan'], digests['reader_plan']
        plan = detector_plan or reader_plan
        if plan is None:
            return None
        split_path = self._split_file(plan)
        split = self._load(split_path, split_digest)
        latest = None
        try:
            pointer = _read_json(self.root / 'splits' / 'latest.json')
            latest = pointer.get('sha256') if isinstance(pointer, dict) else None
        except (OSError, ValueError):
            pass
        relative = None
        if split_path is not None:
            try:
                relative = str(split_path.relative_to(self.root.resolve()))
            except ValueError:
                relative = None
        return {
            'dataset_sha256': plan.get('dataset_sha256'), 'split_sha256': plan.get('split_sha256'),
            'dataset_kind': plan.get('dataset_kind'), 'split_file': relative,
            'split': None if split is None or split is UNREADABLE else split,
            'current_split': latest is not None and latest == plan.get('split_sha256'),
            'detector': None if detector_plan is None else {
                'frames': detector_plan['counts_by_split'], 'boxes': detector_plan['boxes_by_split'],
                'negative_frames': detector_plan['negative_frames_by_split']},
            'reader': None if reader_plan is None else {'crops': reader_plan['counts_by_split'], 'classes': reader_plan['classes']},
        }

    @staticmethod
    def _curves(gathered: dict) -> dict:
        """The curves of the two networks: from the log (with the time of each epoch), else from result.json."""
        curves = dict((gathered['log'] or {}).get('curves') or {})
        for net in ('detector', 'reader'):
            result = gathered['digests'][net + '_result']
            if net not in curves and result and result['history']:
                curves[net] = {'epochs': result['epochs'] or result['history'][-1]['epoch'], 'points': result['history']}
        return curves

    # -- the endpoints ---------------------------------------------------------------------------------
    def names(self) -> list[str]:
        runs = self.runs_dir
        try:
            if not runs.is_dir():
                return []
            return [entry.name for entry in runs.iterdir() if self._run_dir(entry.name) is not None]
        except OSError:
            return []

    def summaries(self) -> dict:
        """GET /api/panel/runs: every run, newest first."""
        running = self.running_runs()
        rows = []
        for name in self.names():
            run = self._run_dir(name)
            if run is not None:
                rows.append(self._summary(self._gather(name, run, running)))
        rows.sort(key=lambda row: (row['started_at'] or '', row['name']), reverse=True)
        self._prune()
        return {'runs': rows, 'runs_dir_exists': self.runs_dir.is_dir()}

    def detail(self, name: str) -> dict | None:
        """GET /api/panel/runs/<run>: everything about one run; None when there is no such run."""
        run = self._run_dir(name)
        if run is None:
            return None
        gathered = self._gather(name, run, self.running_runs())
        digests = gathered['digests']
        training = {}
        for net in ('detector', 'reader'):
            result = digests[net + '_result']
            training[net] = None if result is None else {key: value for key, value in result.items() if key != 'history'}
        return {
            **self._summary(gathered),
            'files': {key: path is not None for key, path in gathered['files'].items()},
            'training': training,
            'reader_eval': digests['reader_eval'],
            'detector_eval': digests['detector_eval'],
            'dataset': self._dataset(gathered),
            'curves': self._curves(gathered),
            'experiment': self._experiment(run),
        }

    def _prune(self) -> None:
        """Forget the digests of files that are gone (a run deleted)."""
        with self._lock:
            for key in [key for key in self._cache if not Path(key).exists()]:
                del self._cache[key]


PROBLEM_FILES = {'detector_plan': 'detector/training-plan.json', 'reader_plan': 'reader/training-plan.json',
                 'detector_result': 'detector/result.json', 'reader_result': 'reader/result.json',
                 'detector_eval': 'evaluation/detector-test.json', 'reader_eval': 'evaluation/reader-test.json'}
# [control-panel] - END
