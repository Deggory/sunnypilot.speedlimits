# [reviewed-dataset] - START
"""Deterministic route/site/encounter/image-connected partitions, never frame randomization."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from nnslr_tools.annotations import DatasetError, atomic_write, frame_identity, require, validate_dataset

# Every partition name a split file may use. route-held-out and hard-negative come from the
# first policy (five partitions) and stay valid so its split files can still be read and checked.
SPLITS = ('train', 'validation', 'test', 'route-held-out', 'hard-negative')
# The partitions build_splits assigns now: the training learns from train, picks its best epoch
# on validation and is measured on test; nothing else reads a partition.
POLICY = 'train-validation-test-v2'
POLICY_SPLITS = ('train', 'validation', 'test')
# validation and test each get whole connected groups adding up to about this share of the
# active rows (at least one group each); everything else is train.
EVALUATION_SHARE = .15
MIN_GROUPS = len(POLICY_SPLITS)


def connected_groups(rows: list[dict]) -> list[list[dict]]:
    active = [r for r in rows if r['review_state'] in ('accepted', 'corrected')]
    parent = list(range(len(active)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    tokens = {}
    for i, row in enumerate(active):
        for field in ('route_id', 'site_group', 'encounter_id', 'image_sha256'):
            value = row[field]
            if value is None:
                continue
            key = field, value
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
    for group in connected_groups(rows):
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
    groups = connected_groups(rows)
    require(len(groups) >= MIN_GROUPS, 'insufficient_independent_groups',
            f'{len(groups)} groups; train, validation and test need at least {MIN_GROUPS}. '
            f'Keep connected routes/sites together.')
    require(type(seed) is int, 'invalid_seed')

    def rank(group):
        ids = ','.join(r['annotation_id'] for r in group)
        return hashlib.sha256(f'{seed}:{ids}'.encode()).hexdigest()

    groups.sort(key=rank)
    wanted = EVALUATION_SHARE * sum(len(g) for g in groups)
    pinned = []
    for route in dict.fromkeys(test_routes):
        found = [g for g in groups if any(r['route_id'] == route for r in g)]
        require(bool(found), 'unknown_test_route', f'{route} has no accepted or corrected rows in the dataset')
        for group in found:
            if not any(group is g for g in pinned):
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
    validation = _pick_groups(rest, wanted, keep=1)
    assignments = {}
    for name, chosen in (('test', test), ('validation', validation)):
        for group in chosen:
            assignments.update({r['annotation_id']: name for r in group})
    for group in groups:
        for row in group:
            assignments.setdefault(row['annotation_id'], 'train')
    active = [r for g in groups for r in g]
    human = {name: 0 for name in POLICY_SPLITS}
    routes = {name: set() for name in POLICY_SPLITS}
    for row in active:
        name = assignments[row['annotation_id']]
        human[name] += int(_human_reviewed(row))
        routes[name].add(row['route_id'])
    # [model-review] - START
    base = root / 'datasets' / 'training_candidate_dataset' if dataset_kind == 'training-candidate' else root
    payload = {'schema_version': 1, 'dataset_kind': dataset_kind, 'dataset_sha256': report['dataset_sha256'], 'seed': seed,
    # [model-review] - END
               'policy': POLICY, 'test_routes': sorted(set(test_routes)),
               'grouping': ['route_id', 'site_group', 'encounter_id', 'image_sha256'],
               'assignments': dict(sorted(assignments.items())),
               'excluded_rejected_ids': sorted(r['annotation_id'] for r in rows if r['review_state'] == 'rejected'),
               'counts': {name: sum(s == name for s in assignments.values()) for name in POLICY_SPLITS},
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
            'policy': POLICY, 'counts': payload['counts'], 'human_reviewed_counts': human,
            'routes': payload['routes'], 'warnings': report['warnings']}
# [reviewed-dataset] - END
