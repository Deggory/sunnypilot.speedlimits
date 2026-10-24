"""The Train tab's data: what the dataset holds, where a training can start, and the start option."""
from __future__ import annotations

import json
import time

import pytest

from nnslr_tools.panel import Panel, PanelError
from test_panel import checkout, finished, full_log, only_this_checkout, root  # noqa: F401  (fixtures and helpers)

ENV_TRAIN = '#!/usr/bin/env bash\necho "train $* init=${NNSLR_TRAIN_INIT:-} from=${NNSLR_TRAIN_FROM_RUN:-}"\n'


def write_dataset(root):
    (root / 'annotations').mkdir(parents=True, exist_ok=True)
    rows = [
        {'route_id': 'r1', 'image_path': 'f/1.png', 'sign_family': 'maximum_speed', 'provenance': {'label_basis': 'human_review'}},
        {'route_id': 'r1', 'image_path': 'f/1.png', 'sign_family': 'not_a_sign', 'provenance': {'label_basis': 'ai_review'}},
        {'route_id': 'r2', 'image_path': 'f/2.png', 'sign_family': 'other_sign', 'provenance': {'label_basis': 'ai_review'}},
    ]
    (root / 'annotations' / 'objects.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    (root / 'splits').mkdir(exist_ok=True)
    (root / 'splits' / 'abc.json').write_text(json.dumps({'counts': {'train': 2, 'test': 1}, 'warnings': ['w']}))
    (root / 'splits' / 'latest.json').write_text(json.dumps({'path': 'splits/abc.json'}))
    (root / 'logs').mkdir(exist_ok=True)
    (root / 'logs' / 'import.json').write_text(json.dumps({
        'imported_frame_count': 1, 'already_imported_frame_count': 4, 'changed_after_import': ['a', 'b']}))


def test_the_dataset_summary_counts_frames_objects_families_and_reviewers(root, checkout):
    write_dataset(root)
    summary = Panel(root, repo=checkout).dataset_summary()
    assert (summary['frames'], summary['objects'], summary['routes'], summary['human'], summary['ai']) == (2, 3, 2, 1, 2)
    assert [(item['key'], item['count']) for item in summary['families']][0] == ('maximum_speed', 1)
    assert {item['label'] for item in summary['families']} == {'Speed limit', 'Not a sign', 'Other sign'}
    assert summary['per_route'][0] == {'route': 'r1', 'frames': 1, 'objects': 2, 'speed_signs': 1, 'human': 1, 'ai': 1}
    assert summary['split']['counts'] == {'train': 2, 'test': 1} and summary['split']['warnings'] == ['w']
    assert summary['last_import'] == {'at': summary['last_import']['at'], 'new_frames': 1, 'already': 4,
                                      'changed_after_import': 2, 'warnings': []}


def test_training_starts_list_imagenet_random_stable_and_finished_runs(root, checkout):
    run = root / 'runs' / '2026-10-24_1014'
    for task in ('detector', 'reader'):
        (run / task).mkdir(parents=True)
        (run / task / (task + '-best.pt')).write_bytes(b'x')
    (root / 'runs' / 'unfinished' / 'detector').mkdir(parents=True)
    panel = Panel(root, repo=checkout)
    panel._stable = (time.monotonic(), False)
    starts = {item['key']: item for item in panel.training_starts()}
    assert list(starts) == ['imagenet', 'random', 'stable', 'run:2026-10-24_1014']
    assert starts['imagenet']['default'] and starts['imagenet']['available']
    assert not starts['stable']['available'] and starts['stable']['reason'].startswith('No stable model')


@pytest.mark.parametrize('start, expected', [(None, 'train  init= from='), ('random', 'train  init=random from='),
                                             ('stable', 'train --continue init= from=')])
def test_the_start_of_a_training_becomes_a_flag_or_the_environment(root, checkout, start, expected, monkeypatch):
    (checkout / 'helpers' / '3_train.sh').write_text(ENV_TRAIN)
    panel = Panel(root, repo=checkout)
    # A real training running on this machine would make the panel refuse to start another one.
    monkeypatch.setattr(panel, 'activities', lambda: [])
    panel._stable = (time.monotonic(), True)
    job = panel.start('train', {'start': start} if start else None)
    finished(panel, job['id'])
    assert expected in full_log(panel, job['id'])


def test_a_bad_or_unavailable_start_is_refused(root, checkout, monkeypatch):
    panel = Panel(root, repo=checkout)
    monkeypatch.setattr(panel, 'activities', lambda: [])
    panel._stable = (time.monotonic(), False)
    for options, status in (({'start': 'rm -rf'}, 400), ({'start': 'stable'}, 409), ({'start': 'run:nope'}, 400)):
        with pytest.raises(PanelError) as refused:
            panel.start('train', options)
        assert refused.value.status == status, options
    with pytest.raises(PanelError) as other:
        panel.start('prepare', {'start': 'random'})
    assert other.value.status == 400
