# [model-roles] - START
"""The networks of the pipeline by role, for the Models tab of the site (docs/NEURAL_NETWORK.md): the cutter finds
every box that could be a sign (today the detector), the classifier says what each box is (today the reader), the
decider says whom a sign applies to (today only the prototypes of experiment B).

Every trained artifact under <data root>/runs is listed by role, newest first, with the numbers that judge it, read
from what the training and the evaluation wrote:

- cutter: ``<run>/detector/detector-best.pt`` (measured in ``<run>/evaluation/detector-test.json``) and
  ``<run>/detector-sizes/<size>/detector-best.pt`` (scripts/run_detector_sizes.sh, measured in
  ``<size>/evaluation/detector-{test,validation}.json``): recall overall and by sign width at the cutter's score
  (CUTTER_SCORE, else the report's own threshold), boxes and false boxes per frame, time per frame;
- classifier: ``<run>/reader/reader-best.pt`` (``<run>/evaluation/reader-test.json``): right answers on the test,
  overall and on the crops a person confirmed; ``<run>/classifier/reader-best.pt`` of scripts/run_classifier.sh, the
  classifier with "not a sign" (``<run>/evaluation/classifier-test.json``, ``<run>/evaluation/chain-test.json``):
  right answers on the sign crops, the "not a sign" crops it rejects, and the chain end to end at the cutter's score
  (the speed signs found and read right, the false boxes per frame that survive it);
- decider: ``<run>/experiment-b/<variant>/<variant>-best.pt`` with its ``result.json`` (nnslr_tools.context_experiment):
  whom the sign applies to on validation and test, and the reading.

Each artifact has a stable id ``<role>:<run>/<part>``: ``cutter:2026-10-24_1452/detector``,
``cutter:<run>/detector-sizes/640x360``, ``classifier:<run>/reader``, ``classifier:<run>/classifier``,
``decider:<run>/experiment-b/crop_geom``.

The model in use for each role is <data root>/models/active.json, ``{"cutter": {"id", "who", "via", "at"} or null,
"classifier": ..., "decider": ...}``: null until someone picks one, never a default here. Prepare's "ours" proposer
reads the three in use, only when all three are (nnslr_tools.preannotate.our_networks); scripts/run_classifier.sh the
cutter in use.
Choosing copies, exports or deploys nothing. Only ModelIndex.set_active writes, only that file, atomically; the id
must be one of the index, and nothing else of a request reaches the file system.

The files are read through the RunsIndex of the Evaluate tab (its strict names, its checks that a file is inside its
run, its digests kept while a file does not change).
"""
from __future__ import annotations

import contextlib
import json
import os
import re
import tempfile
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from nnslr_tools.checkpoints import DEFAULT_DETECTOR_ANCHOR_MIN_RATIO, DEFAULT_DETECTOR_INPUT_SIZE
from nnslr_tools.runs import (EXPERIMENT_DIR, INPUT_NAMES, RUN_NAME, SIZE_BUCKETS, UNREADABLE, VARIANT_LABELS,
                              VARIANTS, RunsIndex, _dig, _int, _iso, _mtime, _num, _ratio, _size_table, _words,
                              chain_digest, chain_headline, classifier_headline, classifier_result_digest,
                              reader_digest, result_digest)

ROLES = ('cutter', 'classifier', 'decider')
ROLE_LABELS = {'cutter': 'Cutter', 'classifier': 'Classifier', 'decider': 'Decider'}
# The cutter must miss nothing: it is judged on recall at a low score (nnslr_tools.evaluation.CUTTER_SCORE_THRESHOLDS).
CUTTER_SCORE = 0.1
# What a cutter learnt to find (result.json and its reports, "families"): every sign (the speed signs and the other
# signs) or the speed signs only (every cutter trained before it was recorded).
FAMILIES = ('all', 'speed')
# A detector folder of scripts/run_detector_sizes.sh: WIDTHxHEIGHT.
SIZE_NAME = re.compile(r'[1-9][0-9]{0,4}x[1-9][0-9]{0,4}')
_RUN = RUN_NAME.pattern
MODEL_ID = re.compile(rf'(cutter|classifier|decider):({_RUN})/(detector|detector-sizes/{SIZE_NAME.pattern}|reader|'
                      rf'classifier|{EXPERIMENT_DIR}/{_RUN})')
MAX_ID = 320
ACTIVE_FILE = 'models/active.json'
# How the person who chose was recognised: signed in on the site, the access token of an agent, 127.0.0.1 without
# sign-in.
VIA = ('sign-in', 'token', 'local')
DECIDER_PARTITIONS = ('validation', 'test')


class ModelChoiceError(Exception):
    """A choice of the model in use that is refused: HTTP status, reason, message (as PanelError)."""

    def __init__(self, status: int, reason: str, message: str):
        super().__init__(message)
        self.status = status
        self.reason = reason
        self.message = message


def _size_label(value: Any) -> str | None:
    """[640, 360] or '640x360' -> '640x360'; None when it is neither."""
    if isinstance(value, str) and SIZE_NAME.fullmatch(value):
        return value
    if isinstance(value, (list, tuple)) and len(value) == 2:
        sides = [_int(side) for side in value]
        if all(side is not None and side > 0 for side in sides):
            return f'{sides[0]}x{sides[1]}'
    return None


def _pretty_size(size: str) -> str:
    return size.replace('x', ' × ')


# -- digests of the files ---------------------------------------------------------------------------------
def _widths(table: Any) -> list[dict] | None:
    """recall_by_width of nnslr_tools.evaluation ({key: {truth, found, recall}}) or the rows of runs._size_table,
    as rows in the order of SIZE_BUCKETS."""
    if isinstance(table, list):
        table = {row.get('key'): row for row in table if isinstance(row, dict)}
    if not isinstance(table, dict):
        return None
    rows = []
    for key, label, _, _ in SIZE_BUCKETS:
        row = table.get(key) if isinstance(table.get(key), dict) else {}
        truth, found = _int(row.get('truth')) or 0, _int(row.get('found')) or 0
        recall = _num(row.get('recall'))
        rows.append({'key': key, 'label': label, 'truth': truth, 'found': found,
                     'recall': recall if recall is not None else _ratio(found, truth)})
    return rows


def _box_counts(metrics: dict) -> dict:
    """Found, reviewed, recall, precision, boxes and false boxes per frame of detector_metrics (or a report)."""
    found, missed, false_boxes = (_int(metrics.get(key)) for key in ('true_positives', 'false_negatives', 'false_positives'))
    truth = _int(metrics.get('truth_box_count'))
    if truth is None and found is not None and missed is not None:
        truth = found + missed
    images = _int(metrics.get('image_count'))
    recall = _num(metrics.get('recall_iou'))
    precision = _num(metrics.get('precision_iou'))
    boxes = _num(metrics.get('predictions_per_image'))
    false_per_frame = _num(metrics.get('false_positives_per_image'))
    if recall is None and found is not None and truth:
        recall = found / truth
    if precision is None and found is not None and false_boxes is not None:
        precision = _ratio(found, found + false_boxes)
    if boxes is None and images and found is not None and false_boxes is not None:
        # Reports written before predictions_per_image: found + false boxes per frame.
        boxes = (found + false_boxes) / images
    if false_per_frame is None and images and false_boxes is not None:
        false_per_frame = false_boxes / images
    return {'images': images, 'truth_boxes': truth, 'found': found, 'false_boxes': false_boxes, 'recall': recall,
            'precision': precision, 'boxes_per_frame': boxes, 'false_per_frame': false_per_frame}


def _families(value: Any) -> str | None:
    return value if value in FAMILIES else None


def _by_family(table: Any) -> dict | None:
    """recall_by_family of a report ({"speed": {truth, found, recall}, "other_sign": ...}): the speed signs keep the
    headline of a cutter that also finds the other signs comparable with one that finds the speed signs only."""
    if not isinstance(table, dict):
        return None
    rows = {}
    for key, row in table.items():
        if isinstance(row, dict):
            truth, found = _int(row.get('truth')) or 0, _int(row.get('found')) or 0
            recall = _num(row.get('recall'))
            rows[str(key)] = {'truth': truth, 'found': found, 'recall': recall if recall is not None else _ratio(found, truth)}
    return rows or None


def cutter_digest(report: Any) -> dict:
    """A detector evaluation (nnslr_tools.evaluation.evaluate_detector) as the cutter is judged: at CUTTER_SCORE when
    the report measured it (by_score_threshold), else at the report's own threshold; recall by the width of the
    reviewed box, and by family (the speed signs apart) for a cutter that also finds the other signs; the boxes per
    frame passed on; the time per frame; the input size and default boxes it ran with, and what it finds."""
    if not isinstance(report, dict):
        raise ValueError('not a report')
    chosen = None
    by_score = report.get('by_score_threshold')
    if isinstance(by_score, dict):
        for key, metrics in by_score.items():
            try:
                score = float(key)
            except (TypeError, ValueError):
                continue
            if abs(score - CUTTER_SCORE) < 1e-9 and isinstance(metrics, dict):
                chosen = metrics
    metrics = chosen if chosen is not None else report
    widths = _widths(metrics.get('recall_by_width'))
    if widths is None and chosen is None and isinstance(report.get('images'), list):
        # Reports written before recall_by_width: from the frames, as the Evaluate tab does.
        widths = _widths(_size_table([item for item in report['images'] if isinstance(item, dict)],
                                     _num(report.get('iou_threshold')) or 0.5))
    human = metrics.get('human_reviewed') if isinstance(metrics.get('human_reviewed'), dict) else None
    inference = report.get('inference') if isinstance(report.get('inference'), dict) else {}
    device = inference.get('device_name') or inference.get('device')
    return {
        'partition': str(report.get('partition') or 'test'),
        'score_threshold': CUTTER_SCORE if chosen is not None else _num(report.get('score_threshold')),
        'at_cutter_score': chosen is not None,
        **_box_counts(metrics),
        'widths': widths,
        'human': None if human is None else {key: value for key, value in _box_counts(human).items()
                                             if key in ('images', 'truth_boxes', 'found', 'recall')},
        'ms_per_frame': _num(inference.get('mean_ms_per_frame')),
        'device': str(device) if isinstance(device, str) and device else None,
        'input_size': _size_label(report.get('input_size_label')) or _size_label(report.get('input_size')),
        'anchor_min_ratio': _num(report.get('anchor_min_ratio')),
        'checkpoint_epoch': _int(report.get('checkpoint_epoch')),
        'families': _families(report.get('families')),
        'by_family': _by_family(metrics.get('recall_by_family')),
    }


def model_result_digest(result: Any) -> dict:
    """result.json of a detector or a reader (nnslr_tools.training): how it was trained, without its history."""
    base = result_digest(result)
    augmentation = result.get('augmentation')
    return {
        'architecture': base['architecture'] or None,
        'initialization': base['initialization'],
        # The epoch whose weights are *-best.pt: recorded by newer trainings, else found in the history.
        'best_epoch': _int(result.get('best_epoch')) or base['best_epoch'],
        'epochs': base['epochs'],
        'epochs_run': len(base['history']) or None,
        'classes': base['classes'],
        'best_validation_loss': base['best_validation_loss'],
        'best_validation_accuracy': base['best_validation_accuracy'],
        'input_size': _size_label(result.get('input_size')),
        'anchor_min_ratio': _num(result.get('anchor_min_ratio')),
        'augmentation': None if 'augmentation' not in result else bool(augmentation),
        'gpu': base['gpu'],
        'families': _families(result.get('families')),
    }


def decider_digest(result: Any) -> dict:
    """result.json of an experiment B variant: whom the sign applies to (the "Another" road found and right,
    macro-F1, abstentions) and the reading, on validation and test."""
    if not isinstance(result, dict):
        raise ValueError('not a result')
    metrics = result.get('metrics') if isinstance(result.get('metrics'), dict) else {}
    partitions: dict[str, dict | None] = {}
    for name in DECIDER_PARTITIONS:
        report = metrics.get(name)
        if not isinstance(report, dict):
            partitions[name] = None
            continue
        other = _dig(report, ('ownership', 'other_road'))
        other = other if isinstance(other, dict) else {}
        partitions[name] = {
            'boxes': _int(report.get('box_count')),
            'reading_accuracy': _num(_dig(report, ('reading', 'accuracy'))),
            'ownership_samples': _int(_dig(report, ('ownership', 'sample_count'))),
            'macro_f1': _num(_dig(report, ('ownership', 'macro_f1'))),
            'abstain_rate': _num(_dig(report, ('ownership', 'abstain_rate'))),
            'other_road_recall': _num(other.get('recall')),
            'other_road_precision': _num(other.get('precision')),
            'other_road_support': _int(other.get('support')),
            'other_road_found': _int(other.get('true_positives')),
        }
    inputs = result.get('inputs')
    text = lambda key: result.get(key) if isinstance(result.get(key), str) else None  # noqa: E731
    return {
        'partitions': partitions,
        'inputs': [INPUT_NAMES.get(str(item), _words(str(item))) for item in inputs] if isinstance(inputs, list) else None,
        'architecture': text('architecture'),
        'initialization': text('initialization'),
        'best_epoch': _int(result.get('best_epoch')),
        'epochs': _int(_dig(result, ('config', 'epochs'))),
        'epochs_run': len(result['history']) if isinstance(result.get('history'), list) and result['history'] else None,
        'limit': _int(result.get('limit')),
    }


def _active_entry(role: str, value: Any) -> dict | None:
    """One role of active.json, checked: None when null or not a choice this module could have written."""
    if not isinstance(value, dict) or not isinstance(value.get('id'), str):
        return None
    match = MODEL_ID.fullmatch(value['id'])
    if match is None or match.group(1) != role or len(value['id']) > MAX_ID:
        return None
    who = value.get('who')
    via = value.get('via')
    at = value.get('at')
    return {'id': value['id'], 'who': who if isinstance(who, str) and len(who) <= 100 else None,
            'via': via if via in VIA else None, 'at': at if isinstance(at, str) and len(at) <= 40 else None}


def _write_atomic(path: Path, value: dict) -> None:
    """The JSON written next to path, flushed to the disk, then renamed over it: a reader sees the old file or the
    new one, never half of it."""
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
    with contextlib.suppress(OSError):
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)


# -- the index --------------------------------------------------------------------------------------------
class ModelIndex:
    """Every trained artifact of <root>/runs by role, and the one in use for each role (models/active.json)."""

    def __init__(self, runs: RunsIndex):
        self.runs = runs
        self.root = runs.root
        self._write_lock = threading.Lock()

    # -- the artifacts ---------------------------------------------------------------------------------
    def _base(self, role: str, run: str, part: str, checkpoint: Path, result: Any, problems: list[str]) -> dict:
        return {'id': f'{role}:{run}/{part}', 'role': role, 'run': run, 'part': part,
                'trained_at': _iso(_mtime(checkpoint)), '_mtime': _mtime(checkpoint) or 0.0,
                'finished': isinstance(result, dict), 'in_use': False, 'problems': problems}

    def _read(self, run: Path, relative: str, digest, problems: list[str], name: str | None = None) -> Any:
        """The digest of a file of the run (None when missing); an unreadable file is a problem of the version."""
        value = self.runs._load(self.runs._file(run, relative), digest)
        if value is UNREADABLE:
            problems.append((name or relative) + ' is unreadable')
            return None
        return value

    def _cutter(self, name: str, run: Path, folder: str, evaluation: str, part: str, size: str | None) -> dict | None:
        checkpoint = self.runs._file(run, folder + '/detector-best.pt')
        if checkpoint is None:
            return None
        problems: list[str] = []
        result = self._read(run, folder + '/result.json', model_result_digest, problems)
        test = self._read(run, evaluation + '/detector-test.json', cutter_digest, problems)
        validation = self._read(run, evaluation + '/detector-validation.json', cutter_digest, problems)
        trained = result or {}
        measured = test or validation or {}
        # The input size and default boxes it ran with; before they were recorded every detector had the defaults.
        input_size = measured.get('input_size') or trained.get('input_size') or size
        anchor = measured.get('anchor_min_ratio')
        if anchor is None:
            anchor = trained.get('anchor_min_ratio')
        default_size = '{}x{}'.format(*DEFAULT_DETECTOR_INPUT_SIZE)
        families = trained.get('families') or measured.get('families')
        entry = self._base('cutter', name, part, checkpoint, result, problems)
        entry.update({
            'label': 'Detector ' + _pretty_size(input_size or default_size),
            'detail': 'input-size comparison' if size else 'training pipeline',
            'architecture': trained.get('architecture'), 'initialization': trained.get('initialization'),
            'best_epoch': trained.get('best_epoch') or measured.get('checkpoint_epoch'),
            'epochs': trained.get('epochs'), 'epochs_run': trained.get('epochs_run'),
            'input_size': input_size or default_size, 'input_size_recorded': input_size is not None,
            'anchor_min_ratio': anchor if anchor is not None else DEFAULT_DETECTOR_ANCHOR_MIN_RATIO,
            'anchor_recorded': anchor is not None,
            'augmentation': trained.get('augmentation'),
            # What it finds: every sign or the speed signs only (before it was recorded, the speed signs only).
            'families': families or 'speed', 'families_recorded': families is not None,
            'test': test, 'validation': validation,
        })
        return entry

    def _classifier(self, name: str, run: Path) -> dict | None:
        checkpoint = self.runs._file(run, 'reader/reader-best.pt')
        if checkpoint is None:
            return None
        problems: list[str] = []
        result = self._read(run, 'reader/result.json', model_result_digest, problems)
        report = self._read(run, 'evaluation/reader-test.json', reader_digest, problems)
        trained = result or {}
        entry = self._base('classifier', name, 'reader', checkpoint, result, problems)
        entry.update({
            'label': 'Reader', 'detail': 'training pipeline',
            'architecture': trained.get('architecture'), 'initialization': trained.get('initialization'),
            'best_epoch': trained.get('best_epoch'), 'epochs': trained.get('epochs'), 'epochs_run': trained.get('epochs_run'),
            'classes': trained.get('classes') or (report or {}).get('classes'),
            'best_validation_accuracy': trained.get('best_validation_accuracy'),
            'test': None if report is None else {
                'samples': report['samples'], 'accuracy': report['accuracy'],
                'human_samples': (report['human'] or {}).get('samples'),
                'human_accuracy': (report['human'] or {}).get('accuracy')},
        })
        return entry

    def _classifier_v2(self, name: str, run: Path) -> dict | None:
        """The classifier with "not a sign" of scripts/run_classifier.sh (its weights still say task "reader")."""
        checkpoint = self.runs._file(run, 'classifier/reader-best.pt')
        if checkpoint is None:
            return None
        problems: list[str] = []
        result = self._read(run, 'classifier/result.json', classifier_result_digest, problems)
        report = self._read(run, 'evaluation/classifier-test.json', reader_digest, problems)
        chain = self._read(run, 'evaluation/chain-test.json', chain_digest, problems)
        old_reader = self._read(run, 'evaluation/chain-test-old-reader.json', chain_digest, problems)
        trained = result or {}
        entry = self._base('classifier', name, 'classifier', checkpoint, result, problems)
        entry.update({
            'label': 'Classifier', 'detail': 'with “not a sign”',
            'class_scheme': trained.get('class_scheme') or (report or {}).get('class_scheme') or 'v2',
            'architecture': trained.get('architecture') or None, 'initialization': trained.get('initialization'),
            'best_epoch': trained.get('best_epoch'), 'epochs': trained.get('epochs'), 'epochs_run': trained.get('epochs_run'),
            'classes': trained.get('classes') or (report or {}).get('classes'),
            'best_validation_accuracy': trained.get('best_validation_accuracy'),
            # The mean of the two below chose its best epoch.
            'best_validation_balanced_accuracy': trained.get('best_validation_balanced_accuracy'),
            'best_validation_sign_accuracy': trained.get('best_validation_sign_accuracy'),
            'best_validation_rejection': trained.get('best_validation_rejection'),
            'augmentation': trained.get('augmentation'),
            'test': None if report is None else classifier_headline(report),
            # The chain end to end at the cutter's score, and the old reader in the same chain.
            'chain': None if chain is None else chain_headline(chain),
            'chain_old_reader': None if old_reader is None else chain_headline(old_reader),
        })
        return entry

    def _deciders(self, name: str, run: Path) -> list[dict]:
        base = run / EXPERIMENT_DIR
        names = list(VARIANTS)
        try:
            if base.is_dir() and not base.is_symlink():
                names += sorted(entry.name for entry in base.iterdir()
                                if RUN_NAME.fullmatch(entry.name) and entry.name not in VARIANTS
                                and entry.is_dir() and not entry.is_symlink())
        except OSError:
            pass
        found = []
        for variant in names:
            checkpoint = self.runs._file(run, f'{EXPERIMENT_DIR}/{variant}/{variant}-best.pt')
            if checkpoint is None:
                continue
            problems: list[str] = []
            result = self._read(run, f'{EXPERIMENT_DIR}/{variant}/result.json', decider_digest, problems)
            trained = result or {}
            entry = self._base('decider', name, f'{EXPERIMENT_DIR}/{variant}', checkpoint, result, problems)
            entry.update({
                'label': VARIANT_LABELS.get(variant, _words(variant)), 'detail': 'experiment B prototype',
                'variant': variant, 'inputs': trained.get('inputs'),
                'architecture': trained.get('architecture'), 'initialization': trained.get('initialization'),
                'best_epoch': trained.get('best_epoch'), 'epochs': trained.get('epochs'),
                'epochs_run': trained.get('epochs_run'),
                # A smoke run of experiment B (--limit): a few frames per partition, not a measure.
                'smoke_limit': trained.get('limit'),
                'validation': (trained.get('partitions') or {}).get('validation'),
                'test': (trained.get('partitions') or {}).get('test'),
            })
            found.append(entry)
        return found

    def artifacts(self, roles: tuple[str, ...] = ROLES) -> dict[str, list[dict]]:
        """Every trained artifact by role (of the roles asked), newest first (by the time of its weights)."""
        found: dict[str, list[dict]] = {role: [] for role in ROLES}
        for name in self.runs.names():
            run = self.runs._run_dir(name)
            if run is None:
                continue
            if 'classifier' in roles:
                found['classifier'] += [entry for entry in (self._classifier(name, run), self._classifier_v2(name, run))
                                        if entry is not None]
            if 'decider' in roles:
                found['decider'] += self._deciders(name, run)
            if 'cutter' not in roles:
                continue
            cutters = [self._cutter(name, run, 'detector', 'evaluation', 'detector', None)]
            sizes = run / 'detector-sizes'
            try:
                folders = sorted(entry.name for entry in sizes.iterdir()
                                 if SIZE_NAME.fullmatch(entry.name) and entry.is_dir() and not entry.is_symlink()) \
                    if sizes.is_dir() and not sizes.is_symlink() else []
            except OSError:
                folders = []
            for size in folders:
                folder = 'detector-sizes/' + size
                cutters.append(self._cutter(name, run, folder, folder + '/evaluation', folder, size))
            found['cutter'] += [entry for entry in cutters if entry is not None]
        for versions in found.values():
            versions.sort(key=lambda entry: (entry['_mtime'], entry['id']), reverse=True)
            for entry in versions:
                del entry['_mtime']
        return found

    # -- the model in use ------------------------------------------------------------------------------
    def _active_path(self) -> Path | None:
        """<root>/models/active.json, None when models/ leads out of the data root."""
        models = self.root / 'models'
        try:
            if not models.resolve().is_relative_to(self.root.resolve()):
                return None
        except (OSError, RuntimeError):
            return None
        return models / 'active.json'

    def read_active(self) -> tuple[dict[str, dict | None], str | None]:
        """The model in use per role (None: nobody picked one), and why the file cannot be used, if it cannot."""
        empty: dict[str, dict | None] = {role: None for role in ROLES}
        path = self._active_path()
        if path is None:
            return empty, 'models/ of the data root leads outside it'
        if not path.exists() and not path.is_symlink():
            return empty, None
        real = self.runs._inside(path, path.parent)
        if real is None:
            return empty, ACTIVE_FILE + ' is not a file inside models/'
        try:
            data = json.loads(real.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            return empty, ACTIVE_FILE + ' cannot be read'
        if not isinstance(data, dict):
            return empty, ACTIVE_FILE + ' is not a JSON object'
        return {role: _active_entry(role, data.get(role)) for role in ROLES}, None

    def in_use(self, role: str) -> tuple[dict | None, str]:
        """The version in use for a role, its files still on this server; else None and why (for the panel: the
        classifier's training mines the false boxes of the cutter in use)."""
        active, problem = self.read_active()
        if problem is not None:
            return None, problem
        chosen = active.get(role)
        if chosen is None:
            return None, 'no ' + role + ' in use: choose one on the Models tab'
        version = next((entry for entry in self.artifacts((role,))[role] if entry['id'] == chosen['id']), None)
        if version is None:
            return None, 'the ' + role + ' in use is no longer on this server: choose another one on the Models tab'
        return version, ''

    def index(self) -> dict:
        """GET /api/panel/models: the three roles, each with its versions newest first and the one in use."""
        artifacts = self.artifacts()
        active, problem = self.read_active()
        roles = []
        for role in ROLES:
            chosen = active[role]
            versions = artifacts[role]
            for entry in versions:
                entry['in_use'] = chosen is not None and entry['id'] == chosen['id']
            roles.append({'role': role, 'label': ROLE_LABELS[role], 'versions': versions,
                          'active': None if chosen is None else {
                              **chosen, 'found': any(entry['in_use'] for entry in versions)}})
        self.runs._prune()
        return {'roles': roles, 'active_problem': problem, 'active_file': ACTIVE_FILE,
                'runs_dir_exists': self.runs.runs_dir.is_dir(), 'cutter_score_threshold': CUTTER_SCORE}

    def set_active(self, role: Any, model_id: Any, *, who: str | None, via: str) -> dict:
        """POST /api/panel/models/active: the model in use for a role. The id must be a finished artifact of the
        index; the file keeps the other roles as they are."""
        if not isinstance(role, str) or role not in ROLES:
            raise ModelChoiceError(400, 'bad_role', 'Unknown role: cutter, classifier or decider.')
        if not isinstance(model_id, str) or len(model_id) > MAX_ID:
            raise ModelChoiceError(400, 'bad_id', 'Not a model id.')
        match = MODEL_ID.fullmatch(model_id)
        if match is None:
            raise ModelChoiceError(400, 'bad_id', 'Not a model id.')
        if match.group(1) != role:
            raise ModelChoiceError(400, 'wrong_role', 'That model is not a ' + role + '.')
        if via not in VIA:
            raise ValueError('via')
        version = next((entry for entry in self.artifacts()[role] if entry['id'] == model_id), None)
        if version is None:
            raise ModelChoiceError(404, 'unknown_model', 'No such model on this server.')
        if not version['finished']:
            raise ModelChoiceError(409, 'not_finished', 'This model has not finished its training: its weights may still change.')
        path = self._active_path()
        if path is None:
            raise ModelChoiceError(409, 'bad_models_folder', 'models/ of the data root leads outside it.')
        with self._write_lock:
            current, problem = self.read_active()
            if problem is not None:
                # Never overwrite a file that cannot be read: it may hold choices someone wants to keep.
                raise ModelChoiceError(409, 'active_unreadable', problem + ': fix or remove it first.')
            entry = {'id': version['id'], 'who': who[:100] if isinstance(who, str) else None, 'via': via,
                     'at': datetime.now().astimezone().isoformat(timespec='seconds')}
            _write_atomic(path, {'schema_version': 1, **current, role: entry})
        return entry
# [model-roles] - END
