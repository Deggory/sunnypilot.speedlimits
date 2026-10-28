# [reviewed-dataset] - START
"""Deterministic partitions of connected groups, never frame randomization.

Groups are connected by place (areas of about 300 m from the GPS position of each frame) when the dataset
has positions, else by route; site, encounter and identical images always join groups. The routes are driven
on the same roads again and again, so a split by route would test the model on signs it learned: by place,
the test areas and a buffer around them never reach the training.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from nnslr_tools.annotations import DatasetError, atomic_write, frame_identity, require, validate_dataset

# Every partition name a split file may use. route-held-out and hard-negative come from the
# first policy (five partitions) and stay valid so its split files can still be read and checked.
SPLITS = ('train', 'validation', 'test', 'route-held-out', 'hard-negative', 'buffer')
# The partitions build_splits assigns now: the training learns from train, picks its best epoch
# on validation and is measured on test; nothing else reads a partition.
POLICY = 'train-validation-test-v2'
POLICY_SPLITS = ('train', 'validation', 'test')
# validation and test each get whole connected groups adding up to about this share of the
# active rows (at least one group each); everything else is train.
EVALUATION_SHARE = .15
MIN_GROUPS = len(POLICY_SPLITS)
# The split by place: areas of about 300 m (0.0027° of latitude, 0.004° of longitude at 47° N). Used when at least
# PLACE_SHARE of the active rows have a GPS position; the areas next to a test or a validation area (also
# diagonally) are the buffer: neither trained on nor measured, since a sign is seen from the next area too.
# place-v3 shielded the test only: its validation shared signs with the training, so the epoch it picked and the
# recall it reported were flattered.
PLACE_POLICY = 'place-v4'
PLACE_CELL = (0.0027, 0.004)
PLACE_SHARE = .75
# The validation by place grows over neighbouring areas in this many compact clusters, each started where it adds
# the most speed-limit values: the buffer around a compact region costs few training frames (28 Oct 2026, 11,700
# rows outside the test: 350 rows of buffer in two clusters against 3,800 for areas scattered by the seed), two
# clusters still sample two roads, and the validation then holds the values 15 to 80 instead of 20 to 60.
VALIDATION_CLUSTERS = 2


def place_of(row: dict) -> tuple[int, int] | None:
    """The area of about 300 m of a row, from its gps, or None without a position."""
    gps = row.get('gps')
    if not isinstance(gps, dict):
        return None
    latitude, longitude = gps.get('latitude'), gps.get('longitude')
    if type(latitude) not in (int, float) or type(longitude) not in (int, float):
        return None
    return int(latitude // PLACE_CELL[0]), int(longitude // PLACE_CELL[1])


def connected_groups(rows: list[dict], *, by_place: bool = False) -> list[list[dict]]:
    """Groups that must stay in one partition. By place: rows of the same area; rows without a position are
    grouped by their route. Else by route. Site, encounter and identical images join groups either way."""
    active = [r for r in rows if r['review_state'] in ('accepted', 'corrected')]
    parent = list(range(len(active)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    tokens = {}
    for i, row in enumerate(active):
        keys = [(field, row[field]) for field in ('site_group', 'encounter_id', 'image_sha256') if row[field] is not None]
        place = place_of(row) if by_place else None
        keys.append(('place', place) if place is not None else ('route_id', row['route_id']))
        for key in keys:
            if key in tokens:
                parent[find(i)] = find(tokens[key])
            else:
                tokens[key] = i
    groups = {}
    for i, row in enumerate(active):
        groups.setdefault(find(i), []).append(row)
    return [sorted(group, key=lambda r: r['annotation_id']) for group in groups.values()]


def validate_splits(rows: list[dict], split: dict, dataset_sha256: str, *, dataset_kind: str = 'gold') -> list[dict]:
    errors = []
    if not isinstance(split, dict) or split.get('schema_version') != 1 or not isinstance(split.get('assignments'), dict):
        return [{'reason': 'invalid_split_schema'}]
    assignments = split['assignments']
    # [model-review] - START
    split_kind = split.get('dataset_kind', 'gold')
    if split_kind not in ('gold', 'training-candidate'):
        return [{'reason': 'unknown_dataset_kind', 'detail': str(split_kind)}]
    if split_kind != dataset_kind:
        return [{'reason': 'split_kind_mismatch', 'detail': 'split and dataset purposes differ'}]
    model_rows = [r for r in rows if isinstance(r.get('provenance'), dict) and r['provenance'].get('label_basis') == 'model_review']
    if split_kind == 'gold' and model_rows:
        return [{'reason': 'split_kind_mismatch', 'detail': 'gold split cannot cover model-reviewed rows'}]
    # [model-review] - END
    expected = {r['annotation_id'] for r in rows if r['review_state'] in ('accepted','corrected')}
    if set(assignments) != expected or any(not isinstance(s,str) or s not in SPLITS for s in assignments.values()):
        return [{'reason': 'invalid_split_membership'}]
    if split.get('dataset_sha256') != dataset_sha256:
        errors.append({'reason': 'split_dataset_mismatch'})
    by_place = 'place' in (split.get('grouping') or ())
    for group in connected_groups(rows, by_place=by_place):
        assigned = {assignments.get(r['annotation_id']) for r in group if r['annotation_id'] in assignments}
        if len(assigned) > 1:
            errors.append({'reason': 'split_leakage', 'routes': sorted({r['route_id'] for r in group})})
    return errors


def _human_reviewed(row: dict) -> bool:
    provenance = row.get('provenance')
    return isinstance(provenance, dict) and provenance.get('label_basis') == 'human_review'


def _human_frame_count(group: list[dict]) -> int:
    """Frames of a group confirmed by a person (provenance label_basis human_review)."""
    return len({frame_identity(r) for r in group if _human_reviewed(r)})


def _pick_groups(candidates: list[list[dict]], wanted: float, *, keep: int,
                 at_least_one: bool = True) -> list[list[dict]]:
    """Whole groups, taken in the order of candidates, adding up to about wanted rows.

    A group is taken when the partition gets closer to wanted with it than without it,
    and only while at least keep groups stay for the other partitions. When no group fits
    (every one is more than twice wanted) and at_least_one, the partition still gets one:
    the group whose size is closest to wanted, the first in the order on a tie.
    """
    chosen = []
    size = 0
    for group in candidates:
        if len(candidates) - len(chosen) - 1 < keep:
            break
        if size + len(group) - wanted < wanted - size:
            chosen.append(group)
            size = size + len(group)
    if not chosen and at_least_one and len(candidates) > keep:
        index = min(range(len(candidates)), key=lambda i: (abs(len(candidates[i]) - wanted), i))
        chosen.append(candidates[index])
    return chosen


def _route_partitions(groups: list[list[dict]], wanted: float, test_routes) -> tuple[list, list]:
    """(test, validation) by whole route groups (policy train-validation-test-v2)."""
    pinned = []
    for route in dict.fromkeys(test_routes):
        for group in groups:
            if any(r['route_id'] == route for r in group) and not any(group is g for g in pinned):
                pinned.append(group)
    rest = [g for g in groups if not any(g is p for p in pinned)]
    require(len(rest) >= 2, 'insufficient_independent_groups',
            f'{len(groups)} groups; with the test routes pinned no group is left for validation and train')
    # Test: the pinned groups, then whole groups up to about the wanted share, those with the most
    # human-reviewed frames first (the seed order on a tie). Two groups stay for validation and train.
    by_human = sorted(rest, key=lambda g: -_human_frame_count(g))
    test = list(pinned)
    tested = sum(len(g) for g in test)
    if not test:
        test = _pick_groups(by_human, wanted, keep=2)
    elif tested < wanted:
        test.extend(_pick_groups(by_human, wanted - tested, keep=2, at_least_one=False))
    rest = [g for g in rest if not any(g is chosen for chosen in test)]
    # Validation: whole groups up to about the wanted share, in the seed order; one group stays for train.
    return test, _pick_groups(rest, wanted, keep=1)


def _places(group: list[dict]) -> set[tuple[int, int]]:
    return {place for place in (place_of(r) for r in group) if place is not None}


def _around(cells) -> set[tuple[int, int]]:
    """The cells and their eight neighbours."""
    return {(a + da, b + db) for a, b in cells for da in (-1, 0, 1) for db in (-1, 0, 1)}


def _speed_values(groups: list[list[dict]]) -> set[int]:
    return {r['value_kph'] for g in groups for r in g if r['sign_family'] == 'maximum_speed' and r['value_kph'] is not None}


def _grow_groups(candidates: list[list[dict]], wanted: float, *, keep: int,
                 clusters: int = VALIDATION_CLUSTERS) -> list[list[dict]]:
    """Whole groups adding up to about wanted rows, in compact clusters. A cluster starts from the candidate that
    adds the most speed-limit values the chosen groups lack (the order of candidates on a tie) and grows over the
    candidates next to it (in or touching its areas) while it is below wanted / clusters. A group is taken when
    the partition gets closer to wanted with it than without it, while at least keep groups stay; when no group
    fits, _pick_groups' one group.
    """
    chosen, size, remaining = [], 0, list(candidates)
    cluster_cells, cluster_size, cap = set(), 0, wanted / max(clusters, 1)
    while remaining and len(remaining) > keep:
        near = _around(cluster_cells) if cluster_size < cap else set()
        have = _speed_values(chosen)
        seeds = sorted((g for g in remaining if not _places(g) & near), key=lambda g: -len(_speed_values([g]) - have))
        order = [g for g in remaining if _places(g) & near] + seeds
        pick = next((g for g in order if size + len(g) - wanted < wanted - size), None)
        if pick is None:
            break
        if not _places(pick) & near:
            cluster_cells, cluster_size = set(), 0
        cluster_cells |= _places(pick)
        cluster_size += len(pick)
        chosen.append(pick)
        size += len(pick)
        remaining = [g for g in remaining if g is not pick]
    return chosen or _pick_groups(candidates, wanted, keep=keep)


def _place_partitions(groups: list[list[dict]], wanted: float, test_routes) -> tuple[list, list, list]:
    """(test, buffer, validation) by areas (policy place-v4).

    The test takes areas up to about the wanted share: areas the test routes drove through when there are
    test routes, the ones with the most frames confirmed by a person first (then the seed order). The areas
    next to a test area are the buffer. Validation takes other areas up to about the wanted share, never next
    to the test, grown from the seed order into VALIDATION_CLUSTERS compact clusters (_grow_groups); the areas
    next to a validation area are the buffer too, so that the training never sees a sign the validation is
    measured on. Everything else is train.
    """
    routes = set(test_routes)
    candidates = [g for g in groups if not routes or any(r['route_id'] in routes for r in g)]
    candidates.sort(key=lambda g: -_human_frame_count(g))
    test = _pick_groups(candidates, wanted, keep=0)
    near = _around(set().union(*(_places(g) for g in test)) if test else set())
    buffer = [g for g in groups if not any(g is t for t in test) and _places(g) & near]
    rest = [g for g in groups if not any(g is t for t in test) and not any(g is b for b in buffer)]
    require(len(rest) >= 2, 'insufficient_independent_groups',
            'no area is left for validation and train next to the test and its buffer')
    validation = _grow_groups(rest, wanted, keep=1)
    near = _around(set().union(*(_places(g) for g in validation)) if validation else set())
    train = [g for g in rest if not any(g is v for v in validation)]
    shield = [g for g in train if _places(g) & near]
    require(len(shield) < len(train), 'insufficient_independent_groups',
            'no area is left for train next to the validation and its buffer')
    return test, buffer + shield, validation


def build_splits(rows: list[dict], root: Path, *, seed: int = 0, output: Path | None = None,
                 dataset_kind: str = 'gold', publish_latest: bool = True,
                 test_routes: list[str] | tuple[str, ...] = ()) -> dict:
    """Freeze train / validation / test by whole connected groups (policy train-validation-test-v2).

    validation and test each get whole groups adding up to about 15% of the active rows,
    at least one group each; every other group is train, so all the reviewed routes are
    used. Reviewed negatives (not_a_sign) stay with their group, in whatever partition.
    The test prefers the groups with the most frames confirmed by a person (label_basis
    human_review), so it is as trustworthy as possible; test_routes pins the groups of
    those routes to test first. Ties and the validation order follow the seed.
    """
    # [model-review] - START
    report = validate_dataset(rows, root, dataset_kind=dataset_kind)
    # [model-review] - END
    require(report['valid'], 'invalid_dataset', json.dumps(report['errors']))
    active_rows = [r for r in rows if r['review_state'] in ('accepted', 'corrected')]
    located = sum(1 for r in active_rows if place_of(r) is not None)
    by_place = bool(active_rows) and located >= PLACE_SHARE * len(active_rows)
    groups = connected_groups(rows, by_place=by_place)
    require(len(groups) >= MIN_GROUPS, 'insufficient_independent_groups',
            f'{len(groups)} groups; train, validation and test need at least {MIN_GROUPS}. '
            f'Keep connected routes/sites together.')
    require(type(seed) is int, 'invalid_seed')

    def rank(group):
        ids = ','.join(r['annotation_id'] for r in group)
        return hashlib.sha256(f'{seed}:{ids}'.encode()).hexdigest()

    groups.sort(key=rank)
    wanted = EVALUATION_SHARE * sum(len(g) for g in groups)
    for route in dict.fromkeys(test_routes):
        require(any(r['route_id'] == route for g in groups for r in g), 'unknown_test_route',
                f'{route} has no accepted or corrected rows in the dataset')
    if by_place:
        test, buffer, validation = _place_partitions(groups, wanted, test_routes)
    else:
        test, validation = _route_partitions(groups, wanted, test_routes)
        buffer = []
    assignments = {}
    for name, chosen in (('test', test), ('buffer', buffer), ('validation', validation)):
        for group in chosen:
            assignments.update({r['annotation_id']: name for r in group})
    for group in groups:
        for row in group:
            assignments.setdefault(row['annotation_id'], 'train')
    active = [r for g in groups for r in g]
    names = POLICY_SPLITS + (('buffer',) if by_place else ())
    human = {name: 0 for name in names}
    routes = {name: set() for name in names}
    for row in active:
        name = assignments[row['annotation_id']]
        human[name] += int(_human_reviewed(row))
        routes[name].add(row['route_id'])
    # [model-review] - START
    base = root / 'datasets' / 'training_candidate_dataset' if dataset_kind == 'training-candidate' else root
    payload = {'schema_version': 1, 'dataset_kind': dataset_kind, 'dataset_sha256': report['dataset_sha256'], 'seed': seed,
    # [model-review] - END
               'policy': PLACE_POLICY if by_place else POLICY, 'test_routes': sorted(set(test_routes)),
               'grouping': (['place', 'site_group', 'encounter_id', 'image_sha256'] if by_place
                            else ['route_id', 'site_group', 'encounter_id', 'image_sha256']),
               'assignments': dict(sorted(assignments.items())),
               'excluded_rejected_ids': sorted(r['annotation_id'] for r in rows if r['review_state'] == 'rejected'),
               'counts': {name: sum(s == name for s in assignments.values()) for name in names},
               'human_reviewed_counts': human,
               'routes': {name: sorted(values) for name, values in routes.items()},
               'warnings': report['warnings'], 'leakage_checked': True,
               'hard_negative_policy': 'reviewed negatives stay in their connected group, in whatever partition'}
    errors = validate_splits(rows, payload, report['dataset_sha256'], dataset_kind=dataset_kind)
    require(not errors, 'split_leakage', json.dumps(errors))
    content = (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    digest = hashlib.sha256(content).hexdigest()
    # [model-review] - START
    target = output if output is not None else base / 'splits' / (digest + '.json')
    # [model-review] - END
    if target.exists():
        require(target.read_bytes() == content, 'immutable_split_conflict', str(target))
    else:
        atomic_write(target, content)
    # [model-review] - START
    # A staged split (e.g. the model-review runner's rename-to-final staging dir)
    # must not repoint the durable pointer at a path that is about to disappear;
    # the caller publishes the pointer once the split is durably in place.
    if publish_latest:
        atomic_write(base / 'splits/latest.json', (json.dumps({'path': str(target.relative_to(root)) if target.is_relative_to(root) else str(target),
                                                           'sha256': digest}, sort_keys=True) + '\n').encode())
    # [model-review] - END
    return {'valid': True, 'dataset_kind': dataset_kind, 'dataset_sha256': report['dataset_sha256'], 'split_sha256': digest,
            'split_path': str(target.relative_to(root)) if target.is_relative_to(root) else str(target),
            'policy': payload['policy'], 'counts': payload['counts'], 'human_reviewed_counts': human,
            'routes': payload['routes'], 'warnings': report['warnings']}
# [reviewed-dataset] - END
