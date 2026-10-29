"""How well the road-type network finds the kind of road and its changes, from its answers on whole sequences.

* per class: precision, recall and F1 over the frames of weight 1 (sure, or a ramp from its OSM link), and macro F1
  over the classes that have frames;
* transitions: a change of the map's road type with at least STABLE_SECONDS of the old type before it and of the new
  one after it (a gap of frames without a type up to GAP_SECONDS between them). Its delay is the time from the change
  to the first second from which the network answers the new type CONFIRM_SECONDS in a row, looked for from
  STABLE_SECONDS before the change (an early answer is a negative delay) to WINDOW_SECONDS after it; later it is
  missed. Kinds: ramp_to_motorway (the merge), motorway_to_ramp (the exit), town_entry, town_exit, other;
* false switches per hour: changes of the network's answer inside a stretch of one road type, MARGIN_SECONDS away from
  its ends, raw (every argmax change) and debounced (an answer taken only after CONFIRM_SECONDS in a row, like the
  car's memory does);
* after the merge: Cristian's case for the memory. After a ramp the motorway can look like a rural road frame by
  frame; the memory should keep "motorway" because it just saw the ramp. For every merge (a ramp of at least
  MERGE_RAMP_SECONDS, then motorway or expressway after a gap of at most GAP_SECONDS) the accuracy over the first
  AFTER_MERGE_SECONDS of the carriageway (until another type starts), raw and debounced, and what was said instead.

One frame a second: a frame's position in its sequence is its time in seconds. numpy only.
"""
from __future__ import annotations

import numpy as np

STABLE_SECONDS = 10
GAP_SECONDS = 10
WINDOW_SECONDS = 60
CONFIRM_SECONDS = 3
MARGIN_SECONDS = 10
TRANSITION_KINDS = ('ramp_to_motorway', 'motorway_to_ramp', 'town_entry', 'town_exit', 'other')
MERGE_RAMP_SECONDS = 3
AFTER_MERGE_SECONDS = 60


def per_class_f1(truth: np.ndarray, predicted: np.ndarray, mask: np.ndarray, classes: tuple[str, ...]) -> dict:
    """{'classes': {name: {precision, recall, f1, support}}, 'macro_f1', 'accuracy', 'frames'} over the frames of
    mask; a class without support has None values and stays out of the macro F1."""
    truth, predicted, mask = np.asarray(truth), np.asarray(predicted), np.asarray(mask, dtype=bool)
    truth, predicted = truth[mask], predicted[mask]
    found, f1s = {}, []
    for index, name in enumerate(classes):
        support = int((truth == index).sum())
        hit = int(((truth == index) & (predicted == index)).sum())
        said = int((predicted == index).sum())
        if support == 0:
            found[name] = {'precision': None, 'recall': None, 'f1': None, 'support': 0, 'predicted': said}
            continue
        precision = hit / said if said else 0.0
        recall = hit / support
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        found[name] = {'precision': round(precision, 4), 'recall': round(recall, 4), 'f1': round(f1, 4),
                       'support': support, 'predicted': said}
        f1s.append(f1)
    return {'classes': found, 'macro_f1': round(float(np.mean(f1s)), 4) if f1s else None,
            'accuracy': round(float((truth == predicted).mean()), 4) if len(truth) else None, 'frames': int(len(truth))}


def confusion(truth: np.ndarray, predicted: np.ndarray, mask: np.ndarray, classes: int) -> list[list[int]]:
    """Frames of mask by [true class][answer]."""
    truth, predicted, mask = np.asarray(truth), np.asarray(predicted), np.asarray(mask, dtype=bool)
    matrix = np.zeros((classes, classes), dtype=np.int64)
    keep = mask & (truth >= 0)
    np.add.at(matrix, (truth[keep].astype(int), predicted[keep].astype(int)), 1)
    return matrix.tolist()


def _runs(labels: np.ndarray) -> list[tuple[int, int, int]]:
    """(label, start, end exclusive) of every run of one value."""
    runs, start = [], 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            runs.append((int(labels[start]), start, i))
            start = i
    return runs


def transition_kind(old: int, new: int, classes: tuple[str, ...]) -> str:
    a, b = classes[old], classes[new]
    if a == 'ramp' and b in ('motorway', 'expressway'):
        return 'ramp_to_motorway'
    if a in ('motorway', 'expressway') and b == 'ramp':
        return 'motorway_to_ramp'
    if b == 'town' and a != 'parking':
        return 'town_entry'
    if a == 'town' and b != 'parking':
        return 'town_exit'
    return 'other'


def change_points(labels: np.ndarray, classes: tuple[str, ...]) -> list[dict]:
    """The map's changes of road type with STABLE_SECONDS of each side (labels: class per second, -1 unknown):
    [{'time', 'old', 'new', 'kind'}]; the time is the first second of the new type."""
    runs = [run for run in _runs(np.asarray(labels)) if run[0] >= 0]
    found = []
    for (old, start_a, end_a), (new, start_b, end_b) in zip(runs, runs[1:]):
        if old == new or end_a - start_a < STABLE_SECONDS or end_b - start_b < STABLE_SECONDS:
            continue
        if start_b - end_a > GAP_SECONDS:
            continue
        found.append({'time': start_b, 'old': old, 'new': new, 'kind': transition_kind(old, new, classes)})
    return found


def detection_delay(predicted: np.ndarray, change: dict) -> int | None:
    """Seconds from a change to the first second of CONFIRM_SECONDS answers of its new type in a row (see the module's
    text), None when missed."""
    predicted = np.asarray(predicted)
    begin = max(change['time'] - STABLE_SECONDS, 0)
    end = min(change['time'] + WINDOW_SECONDS, len(predicted))
    run = 0
    for i in range(begin, end):
        run = run + 1 if predicted[i] == change['new'] else 0
        if run >= CONFIRM_SECONDS:
            return i - CONFIRM_SECONDS + 1 - change['time']
    return None


def debounced(predicted: np.ndarray, confirm: int = CONFIRM_SECONDS) -> np.ndarray:
    """The answer the car would hold: a new class only after confirm seconds of it in a row (causal)."""
    predicted = np.asarray(predicted)
    if not len(predicted):
        return predicted.copy()
    held, candidate, run, out = int(predicted[0]), int(predicted[0]), 0, np.empty_like(predicted)
    for i, value in enumerate(predicted):
        value = int(value)
        if value == held:
            run = 0
        else:
            run = run + 1 if value == candidate else 1
            candidate = value
            if run >= confirm:
                held, run = value, 0
        out[i] = held
    return out


def majority(predicted: np.ndarray, window: int = 5, classes: int = 6) -> np.ndarray:
    """A causal majority vote over the last window answers (a frame network's cheap memory; ties to the newest)."""
    predicted = np.asarray(predicted)
    out = np.empty_like(predicted)
    for i in range(len(predicted)):
        recent = predicted[max(0, i - window + 1):i + 1]
        counts = np.bincount(recent, minlength=classes)
        best = counts.max()
        out[i] = next(int(v) for v in recent[::-1] if counts[v] == best)
    return out


def switches(predicted: np.ndarray, labels: np.ndarray) -> tuple[int, int]:
    """(answer changes, seconds) inside the stretches of one known road type, MARGIN_SECONDS from their ends."""
    predicted, count, seconds = np.asarray(predicted), 0, 0
    for label, start, end in _runs(np.asarray(labels)):
        if label < 0:
            continue
        inner_start, inner_end = start + MARGIN_SECONDS, end - MARGIN_SECONDS
        if inner_end - inner_start < 2:
            continue
        seconds += inner_end - inner_start
        part = predicted[inner_start:inner_end]
        count += int((part[1:] != part[:-1]).sum())
    return count, seconds


def merges(labels: np.ndarray, classes: tuple[str, ...]) -> list[tuple[int, int]]:
    """(first second, end exclusive) of the carriageway after every merge: a ramp of MERGE_RAMP_SECONDS or more, then
    motorway or expressway within GAP_SECONDS, up to AFTER_MERGE_SECONDS or the next other type."""
    ramp = classes.index('ramp')
    roads = {classes.index(name) for name in ('motorway', 'expressway') if name in classes}
    runs = [run for run in _runs(np.asarray(labels)) if run[0] >= 0]
    found = []
    for (old, start_a, end_a), (new, start_b, end_b) in zip(runs, runs[1:]):
        if old == ramp and new in roads and end_a - start_a >= MERGE_RAMP_SECONDS and start_b - end_a <= GAP_SECONDS:
            found.append((start_b, min(end_b, start_b + AFTER_MERGE_SECONDS)))
    return found


def after_merge(sequences: list[dict], classes: tuple[str, ...]) -> dict:
    """{'merges', 'frames', 'accuracy', 'accuracy_debounced', 'said': {class: frames}, 'per_merge': [{'frames',
    'correct'}]} over the first seconds of the carriageway after each merge (see the module's text)."""
    total = correct = steady = 0
    said = {name: 0 for name in classes}
    per_merge = []
    for sequence in sequences:
        truth, answer = np.asarray(sequence['labels']), np.asarray(sequence['predicted'])
        held = debounced(answer)
        for start, end in merges(truth, classes):
            hits = int((answer[start:end] == truth[start:end]).sum())
            total += end - start
            correct += hits
            steady += int((held[start:end] == truth[start:end]).sum())
            for value in answer[start:end]:
                said[classes[int(value)]] += 1
            per_merge.append({'frames': end - start, 'correct': hits})
    return {'merges': len(per_merge), 'frames': total,
            'accuracy': round(correct / total, 4) if total else None,
            'accuracy_debounced': round(steady / total, 4) if total else None,
            'said': said, 'per_merge': per_merge}


def sequence_metrics(sequences: list[dict], classes: tuple[str, ...]) -> dict:
    """The measure over sequences [{'labels', 'weights', 'predicted'}] (arrays per second): per_class_f1 of weight-1
    frames and of every labelled frame, the transitions by kind (count, found, missed, delays, median and mean delay)
    and the false switches per hour, raw and debounced."""
    labels = np.concatenate([np.asarray(s['labels']) for s in sequences]) if sequences else np.zeros(0, int)
    weights = np.concatenate([np.asarray(s['weights']) for s in sequences]) if sequences else np.zeros(0)
    predicted = np.concatenate([np.asarray(s['predicted']) for s in sequences]) if sequences else np.zeros(0, int)
    kinds = {kind: {'count': 0, 'found': 0, 'missed': 0, 'delays': []} for kind in TRANSITION_KINDS}
    raw = steady = seconds = 0
    for sequence in sequences:
        truth, answer = np.asarray(sequence['labels']), np.asarray(sequence['predicted'])
        for change in change_points(truth, classes):
            delay = detection_delay(answer, change)
            entry = kinds[change['kind']]
            entry['count'] += 1
            if delay is None:
                entry['missed'] += 1
            else:
                entry['found'] += 1
                entry['delays'].append(delay)
        count, length = switches(answer, truth)
        raw += count
        seconds += length
        steady += switches(debounced(answer), truth)[0]
    for entry in kinds.values():
        delays = entry['delays']
        entry['median_delay'] = float(np.median(delays)) if delays else None
        entry['mean_delay'] = round(float(np.mean(delays)), 2) if delays else None
    hours = seconds / 3600
    return {'weight_1': per_class_f1(labels, predicted, weights >= 1, classes),
            'labelled': per_class_f1(labels, predicted, labels >= 0, classes),
            'confusion': confusion(labels, predicted, weights >= 1, len(classes)),
            'transitions': kinds,
            'false_switches_per_hour': round(raw / hours, 2) if hours else None,
            'false_switches_per_hour_debounced': round(steady / hours, 2) if hours else None,
            'stable_hours': round(hours, 3),
            'after_merge': after_merge(sequences, classes)}


# -- the comma's rule: when the limit of a ramp ends on the carriageway --------------------------------------------
# vision.SpeedLimitMemory (sunnypilot speed_limits): a number read on a ramp (the road type says "ramp") or within
# RAMP_SECONDS of a motorway or expressway sign is the ramp's limit; it ends, and the motorway's general limit comes
# back, once the road type says motorway or expressway MERGE_SECONDS in a row (the first such step starts the count,
# the release comes MERGE_SECONDS later). The road type given to it is the network's answer after the CONFIRM_SECONDS
# debounce, one step a second.
RAMP_SECONDS = 45.0
MERGE_SECONDS = 8
LATE_SECONDS = 60
NUMBER_SECOND = 2  # the ramp's number is read at its third second (the middle of a shorter ramp)
RELEASE_OUTCOMES = ('on_ramp', 'in_time', 'late', 'not_held')


def ramp_entries(labels: np.ndarray, classes: tuple[str, ...]) -> list[dict]:
    """Every ramp followed by the carriageway (motorway or expressway within GAP_SECONDS), as merges() finds them, with
    the ramp's first second: [{'ramp': first second of the ramp, 'merge': first second of the carriageway, 'before':
    the road type before the ramp (None: none known)}]."""
    ramp = classes.index('ramp')
    roads = {classes.index(name) for name in ('motorway', 'expressway') if name in classes}
    runs = [run for run in _runs(np.asarray(labels)) if run[0] >= 0]
    found = []
    for i, ((old, start_a, end_a), (new, start_b, _)) in enumerate(zip(runs, runs[1:])):
        if old == ramp and new in roads and end_a - start_a >= MERGE_RAMP_SECONDS and start_b - end_a <= GAP_SECONDS:
            before = classes[runs[i - 1][0]] if i > 0 and start_a - runs[i - 1][2] <= GAP_SECONDS else None
            found.append({'ramp': start_a, 'merge': start_b, 'before': before})
    return found


def comma_release(held: np.ndarray, sign_age: np.ndarray, entry: dict, classes: tuple[str, ...]) -> dict:
    """The comma's rule on one merge (held: the debounced answer per second of the drive; sign_age: seconds since the
    last motorway or expressway sign the car confirmed, inf when none): a number read at the ramp's NUMBER_SECOND;
    held as the ramp's when the road type says ramp then or the sign is at most RAMP_SECONDS old; released after
    MERGE_SECONDS of motorway or expressway in a row. {'number_at', 'held_by' (ramp, sign, ramp+sign or None),
    'release' (seconds after the merge, negative: on the ramp; None: not by LATE_SECONDS after it), 'outcome'}."""
    held, sign_age = np.asarray(held), np.asarray(sign_age, dtype=float)
    ramp, roads = classes.index('ramp'), {classes.index(n) for n in ('motorway', 'expressway') if n in classes}
    start, merge = entry['ramp'], entry['merge']
    number_at = start + min(NUMBER_SECOND, max(0, (merge - start) // 2))
    by_ramp = int(held[number_at]) == ramp
    by_sign = bool(sign_age[number_at] <= RAMP_SECONDS)
    held_by = '+'.join(name for name, flag in (('ramp', by_ramp), ('sign', by_sign)) if flag) or None
    result = {'number_at': number_at - merge, 'held_by': held_by, 'release': None}
    if held_by is None:
        return {**result, 'outcome': 'not_held'}
    since = None
    for t in range(number_at + 1, min(len(held), merge + LATE_SECONDS + 1)):
        if int(held[t]) not in roads:
            since = None
        elif since is None:
            since = t
        elif t - since >= MERGE_SECONDS:
            outcome = 'on_ramp' if t < merge else 'in_time'
            return {**result, 'release': t - merge, 'outcome': outcome}
    return {**result, 'outcome': 'late'}


def release_summary(found: list[dict]) -> dict:
    """{'merges', 'held', 'held_by_ramp', 'held_by_sign', 'on_ramp', 'in_time', 'late', 'not_held', 'median_release'
    (of the releases on the carriageway within LATE_SECONDS), 'releases' (every release, None for none)} of
    comma_release results."""
    counts = {outcome: sum(1 for item in found if item['outcome'] == outcome) for outcome in RELEASE_OUTCOMES}
    timely = [item['release'] for item in found if item['outcome'] == 'in_time']
    return {'merges': len(found), 'held': sum(1 for item in found if item['held_by']),
            'held_by_ramp': sum(1 for item in found if item['held_by'] and 'ramp' in item['held_by']),
            'held_by_sign': sum(1 for item in found if item['held_by'] and 'sign' in item['held_by']),
            **counts, 'median_release': float(np.median(timely)) if timely else None,
            'releases': [item['release'] for item in found]}
