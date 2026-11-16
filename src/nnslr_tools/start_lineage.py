"""The networks a "Start from" descends from, and the validation rows they trained on (#104).

A network that goes on from an earlier one (--init-from) carries what that one learnt, and what the one before it
learnt, back to the first of the chain. Every import builds a new split: place-v4 pins the test areas and grows the
validation again, so rows that are validation today were train rows for a network of that chain. Measured on them
the new network is flattered (16 Nov 2026: 644 of the 1,651 validation rows of the applicability network in use
were train rows of one of its 8 ancestors, and it found 17 of 18 signs of another road there against 24 of 40 on
the others), and an epoch close to the start may be kept for it. That this happened is not shown: restarts with no
such row keep epoch 1 too (2026-11-04_2341-cutter, 0 of its 1,994 validation frames), and one with half of its
validation trained on keeps epoch 13 (2026-11-11_2144-cutter, 568 of 1,132).

This module follows the chain and says which rows those are, for the cutter (signnet.train_signnet) and the
applicability network (context_experiment.train_experiment_b); model_roles follows it too, for Models' warning.

- ``lineage``: every ancestor of a start, nearest first, by the ``start`` of each result.json. A record is looked
  for by run name in the folder of runs the start is in: the paths older runs recorded are of the repository's
  old place (/root/sunnypilot.speedlimits). An ancestor whose result.json is gone ends the chain as unknown, never
  as complete. A record that holds a complete ``start_lineage`` of its own (a network trained since #104) ends the
  walk: its ancestors are taken from there (``recorded``), so a run removed later than that does not make the
  lineage of what descends from it unknown.
- ``trained_ids``: the annotation ids that were "train" in the split of an ancestor (<data root>/splits/<sha256>.json,
  kept for ever under the sha256 every result.json records). The split, not the ancestor's plan: a cutter keeps no
  plan, and a split names every row its network could learn, so a row is never called clean wrongly (the plan of an
  applicability variant names only the rows it did learn: on the network in use both give the same 644).
- ``account``: result.json's ``start_lineage`` and the validation rows no ancestor trained on. ``choose_on``:
  whether the best epoch is chosen on those (CLEAN_FLOOR, and ``short_of`` with CLEAN_SHARE for a kind of case the
  measure counts apart). ``words``: the same in one line of the log.

Start from stays the default and is never refused here: the split is rebuilt at every import, so a refusal would
end it (the brain can refuse, its split is frozen). No torch, and nothing of the other modules.
"""
from __future__ import annotations

import json
import re
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable

# The fewest cases the choosing measure must count on the clean rows for the best epoch to be chosen on them (the
# reviewed boxes of a cutter's recall, in a Mapillary stage those of the speed signs; the rows of an applicability
# network's reading). Below it the noise would choose: two standard errors of a rate over 200 cases are 6.5 points at
# a rate of 0.7 and 5 at 0.85, over 100 cases 9.2 and 7.1. What #104's review measured on one split
# (2026-11-01_2128-cutter, a restart, against 2026-11-02_0959-cutter from random weights) is 8 points of validation
# recall at the 5 best boxes, 13 as the car reads them, and none on the test. 200 and not 300, the first choice,
# because a Mapillary stage is chosen on the speed signs alone: on 16 Nov 2026 the clean frames of the 12 Mapillary
# stages on disk whose lineage trained on part of their validation hold 250 to 695 reviewed speed boxes, and the three
# with fewer than 300 (250, 282 and 292: 2026-11-11_1344, 2026-11-15_2140 and 2026-11-11_1744) would have gone on
# choosing on the whole validation. Those three find 0.76 to 0.88 of the speed signs of their validation, and at
# those rates two standard errors over their clean boxes are 4.1 to 5.0 points. Fewer than the floor: the best epoch
# is chosen on the whole validation as before, and result.json says so.
CLEAN_FLOOR = 200

# The share of the validation's cases of one kind that the clean rows must hold, when the choosing measure counts that
# kind apart (short_of). The applicability network's score is half the F1 of the signs of another road, and the floor
# counts rows: with rows enough and one or two such signs among them, a single sign caught or missed would decide
# that half of the score, and with it the epoch. Half of them or more on the clean rows (40 of 58 for the network in
# use, 16 Nov 2026), else the whole validation chooses as before and ``why`` says how many of how many.
CLEAN_SHARE = 0.5

# A run's name, and a folder inside it (model_roles.RUN_NAME's letters): nothing else is followed.
NAME = re.compile(r'[A-Za-z0-9][A-Za-z0-9._-]{0,99}')
SHA256 = re.compile(r'[0-9a-f]{64}')


def start_place(start: Any) -> tuple[str, str] | None:
    """Where a start record's network is: its run and the folder of its stage inside the run, from the record's
    ``run`` and ``path`` (.../<run>/<stage>/<weights>). The stage is '' for the run itself, 'pretrain' for the
    Mapillary stage of a two-stage cutter, 'experiment-b/<variant>' for an applicability variant. None for a record
    without them."""
    if not isinstance(start, dict):
        return None
    run, path = start.get('run'), start.get('path')
    if not isinstance(run, str) or not isinstance(path, str) or not NAME.fullmatch(run):
        return None
    folders = PurePosixPath(path).parts[:-1]
    if run not in folders:
        return None
    stage = folders[len(folders) - folders[::-1].index(run):]
    if not all(NAME.fullmatch(part) for part in stage):
        return None
    return run, '/'.join(stage)


def runs_of(start: Any) -> Path | None:
    """The folder of runs a start's weights are in (the parent of its run): where its record and those of its
    ancestors are, by run name."""
    place = start_place(start)
    if place is None:
        return None
    return Path(start['path']).parents[len(PurePosixPath(place[1]).parts) + 1]


def records(runs: Path | None) -> Callable[[str, str], Any]:
    """The reader ``lineage`` follows the chain with: result.json of <runs>/<run>/<stage>, None when it is not there
    or cannot be read."""
    def read(run: str, stage: str) -> Any:
        if runs is None:
            return None
        try:
            return json.loads((Path(runs) / run / stage / 'result.json').read_text(encoding='utf-8'))
        except (OSError, ValueError):
            return None
    return read


def recorded(record: Any) -> list[dict[str, Any]] | None:
    """The ancestors a result.json records for its own start: its ``start_lineage`` (account's, written since #104)
    when that one was followed to its end and its first network is the record's ``start``. None for a record without
    one (a run of before #104, or one from random weights), with one that is not complete, or with one that is not
    shaped as account writes it: lineage then goes on reading the records."""
    if not isinstance(record, dict) or not isinstance(record.get('start_lineage'), dict):
        return None
    found = record['start_lineage']
    entries = found.get('ancestors')
    if found.get('complete') is not True or not isinstance(entries, list) or not entries:
        return None
    ancestors = []
    for entry in entries:
        if not (isinstance(entry, dict) and isinstance(entry.get('run'), str) and NAME.fullmatch(entry['run'])
                and isinstance(entry.get('stage'), str) and isinstance(entry.get('learnt_ours'), bool)
                and all(NAME.fullmatch(part) for part in entry['stage'].split('/') if entry['stage'])):
            return None
        split = entry.get('split_sha256')
        ancestors.append({'run': entry['run'], 'stage': entry['stage'], 'epoch': entry.get('epoch'),
                          'split_sha256': split if isinstance(split, str) else None,
                          'learnt_ours': entry['learnt_ours']})
    if (ancestors[0]['run'], ancestors[0]['stage']) != start_place(record.get('start')):
        return None
    return ancestors


def lineage(start: Any, read: Callable[[str, str], Any]) -> dict[str, Any]:
    """Every network a start descends from, nearest first: {'ancestors', 'complete', 'unknown'}. Each ancestor is
    {'run', 'stage' (start_place), 'epoch', 'split_sha256', 'learnt_ours'}: the epoch whose weights were taken, the
    split it learnt on, and whether it learnt our frames (a cutter's stage on the Mapillary windows alone did not).
    ``read(run, stage)`` gives an ancestor's result.json, whose ``start`` is the next one; a record with no start
    ends the chain (random or ImageNet weights). A record that holds its own complete lineage (``recorded``) ends
    the walk too: the networks before it are taken from there, and their records are not read (they may be gone by
    now). ``complete`` is False, with ``unknown`` saying why, when the chain could not be followed to its end: a
    record is gone, a start names no run, or the records loop."""
    ancestors: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    unknown = None
    while isinstance(start, dict):
        place = start_place(start)
        split = start.get('split_sha256')
        entry = {'run': place[0] if place else None, 'stage': place[1] if place else None, 'epoch': start.get('epoch'),
                 'split_sha256': split if isinstance(split, str) else None,
                 'learnt_ours': not start.get('mapillary_only')}
        ancestors.append(entry)
        if place is None:
            unknown = 'a start names no run'
            break
        name = place[0] + (f' ({place[1]})' if place[1] else '')
        if place in seen:
            unknown = f'{name} comes twice, the records loop'
            break
        seen.add(place)
        record = read(*place)
        if not isinstance(record, dict):
            unknown = f'the record of {name} is gone (its result.json)'
            break
        if entry['split_sha256'] is None and isinstance(record.get('split_sha256'), str):
            entry['split_sha256'] = record['split_sha256']
        before = recorded(record)
        if before is not None and not any((found['run'], found['stage']) in seen for found in before):
            ancestors.extend(before)
            break
        start = record.get('start')
    return {'ancestors': ancestors, 'complete': unknown is None, 'unknown': unknown}


def another_split(ancestors: Any, split_sha256: Any) -> bool:
    """Whether an ancestor learnt our frames on another split than ``split_sha256``: validation rows of this split
    may then be rows it trained on."""
    if not isinstance(split_sha256, str) or not isinstance(ancestors, list):
        return False
    return any(isinstance(entry, dict) and entry.get('learnt_ours', True) and isinstance(entry.get('split_sha256'), str)
               and entry['split_sha256'] != split_sha256 for entry in ancestors)


def trained_ids(ancestors: list[dict[str, Any]], splits: Path, split_sha256: Any = None) -> tuple[set[str] | None, str | None]:
    """The annotation ids that were "train" in the split of at least one ancestor that learnt our frames
    (<splits>/<sha256>.json, "assignments"), and None; or None and why they are not known: an ancestor records no
    split, or its split file is gone. The split ``split_sha256`` (this run's own) is not read: its train rows are
    train rows here too."""
    ids: set[str] = set()
    for entry in ancestors:
        sha = entry.get('split_sha256')
        if not entry.get('learnt_ours', True) or sha == split_sha256 and isinstance(sha, str):
            continue
        name = str(entry.get('run')) + (f" ({entry['stage']})" if entry.get('stage') else '')
        if not isinstance(sha, str) or not SHA256.fullmatch(sha):
            return None, f'{name} records no split'
        try:
            assignments = json.loads((Path(splits) / f'{sha}.json').read_text(encoding='utf-8'))['assignments']
            ids.update(key for key, partition in assignments.items() if partition == 'train')
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            return None, f'the split {sha[:8]} of {name} is gone'
    return ids, None


def account(start: dict[str, Any], split_sha256: Any, splits: Path, *, counted: str,
            validation: list[Iterable[str]], test: list[Iterable[str]],
            read: Callable[[str, str], Any] | None = None) -> tuple[dict[str, Any], list[int] | None]:
    """result.json's ``start_lineage`` of a network that goes on from ``start``, and the indexes of the validation
    rows no ancestor trained on (None when none was, or when the lineage is not known: then the whole validation is
    all there is to measure on).

    ``validation`` and ``test``: the annotation ids of each frame (a cutter, ``counted`` 'frames') or row (an
    applicability network, 'rows') of the partition, in the plan's order; a frame was trained on when one of its
    ids was. The record: ``ancestors`` (lineage), ``complete`` and ``unknown``, ``another_split``, ``counted``, and
    with a known lineage ``validation`` {'all', 'trained_on', 'clean'} and ``test`` {'all', 'trained_on'} (the test
    areas are pinned, so 0 is expected: it is counted, not assumed). ``best_epoch_on`` is 'validation' here, with
    ``why`` when the lineage is unknown; choose_on sets it to 'clean_validation' when the clean rows are enough."""
    found = lineage(start, read or records(runs_of(start)))
    unknown = found['unknown']
    trained = None
    if unknown is None:
        trained, unknown = trained_ids(found['ancestors'], splits, split_sha256)
    record = {'ancestors': found['ancestors'], 'complete': unknown is None, 'unknown': unknown,
              'another_split': another_split(found['ancestors'], split_sha256), 'counted': counted,
              'validation': None, 'test': None, 'floor': CLEAN_FLOOR, 'clean_cases': None,
              'best_epoch_on': 'validation',
              'why': None if unknown is None else f'the lineage is not known: {unknown}'}
    if trained is None:
        return record, None
    clean = [index for index, ids in enumerate(validation) if trained.isdisjoint(ids)]
    record['validation'] = {'all': len(validation), 'trained_on': len(validation) - len(clean), 'clean': len(clean)}
    record['test'] = {'all': len(test), 'trained_on': sum(not trained.isdisjoint(ids) for ids in test)}
    return record, clean if len(clean) < len(validation) else None


def short_of(clean: int, whole: int, what: str) -> str | None:
    """choose_on's ``lacks`` for a kind of case the choosing measure counts apart (``what``: 'signs of another
    road'): None when the clean rows hold CLEAN_SHARE of the validation's cases of that kind or more (``clean`` of
    ``whole``), or the validation has none; else the words that keep the whole validation as the chooser."""
    if whole <= 0 or clean >= CLEAN_SHARE * whole:
        return None
    return f'only {clean} of the {whole} {what} are on the clean rows'


def choose_on(record: dict[str, Any], cases: int, unit: str, lacks: str | None = None) -> bool:
    """Whether the best epoch is chosen on the clean validation rows of ``record`` (account's), written in it as
    ``best_epoch_on`` 'clean_validation', else 'validation' with ``why``. ``cases``: what the choosing measure
    counts on the clean rows, in ``unit`` words ('reviewed boxes'); fewer than CLEAN_FLOOR keep the whole
    validation. ``lacks``: something the measure needs and the clean rows do not have enough of (short_of), which
    keeps it too. Nothing changes when no row was trained on by an ancestor or the lineage is not known."""
    counts = record.get('validation')
    if not record.get('complete') or not counts or not counts['trained_on']:
        return False
    record['clean_cases'] = cases
    if lacks:
        record['why'] = lacks
    elif cases < CLEAN_FLOOR:
        record['why'] = (f"the {counts['clean']} clean {record['counted']} hold {cases} {unit}, fewer than the "
                         f"{CLEAN_FLOOR} a choice needs")
    else:
        record['best_epoch_on'] = 'clean_validation'
        return True
    return False


def words(record: dict[str, Any]) -> str | None:
    """One line for the log of what ``record`` (account's, after choose_on) changes: the validation rows its
    ancestors trained on and where the best epoch is chosen; None when nothing does (no such row, and none on the
    test)."""
    counted, counts, test = record['counted'], record.get('validation'), record.get('test')
    if not record.get('complete'):
        return (f"{record['why']}; which validation {counted} its ancestors trained on cannot be told, so the best "
                "epoch is chosen on the whole validation")
    line = None
    if counts and counts['trained_on']:
        line = (f"{counts['trained_on']} of the {counts['all']} validation {counted} were train {counted} of a network "
                f"it descends from ({len(record['ancestors'])} before it): ")
        if record['best_epoch_on'] == 'clean_validation':
            line += f"the best epoch is chosen on the other {counts['clean']}, and both are measured"
        else:
            line += f"{record['why']}, so the best epoch is chosen on the whole validation; both are measured"
    if test and test['trained_on']:
        line = ((line + '; ') if line else '') + (f"{test['trained_on']} of the {test['all']} test {counted} were "
                                                 "train for one of them too: this test is not fair")
    return line
