"""The package of our three networks (nnslr_tools.networks_package): its check, its model card and the catalog, on
a package of small stand-in files (the ONNX export itself needs torch: tests/dataset/test_context_experiment.py)."""
from __future__ import annotations

import hashlib
import json

import pytest

from nnslr_tools.networks_package import (PACKAGE_FILES, PackageError, ROLES, add_to_catalog, drive_countries,
                                          mark_published, measured, model_card, package_license, packages, pipeline,
                                          verify_package)


SIGNNET_CONTRACT = {'task': 'signnet', 'stride': 8, 'input': {'name': 'images', 'shape': [1, 3, 288, 896]},
                    'output': {'name': 'out', 'shape': [1, 5, 36, 112]}}


def _package(folder, name='speedlimits-20261025-2200', cutter=None):
    folder.mkdir(parents=True)
    sums = {}
    networks = {}
    for role in ROLES:
        onnx = folder / f'{role}.onnx'
        onnx.write_bytes(role.encode() * 100)
        digest = hashlib.sha256(onnx.read_bytes()).hexdigest()
        (folder / f'{role}.onnx.contract.json').write_text(json.dumps(
            {'onnx_sha256': digest, 'checkpoint': f'{role}:run/{role}', 'verification': {'passed': True},
             'ownership_families': ['maximum_speed']}))
        networks[role] = {'id': f'{role}:run/{role}', 'file': f'{role}.onnx', 'sha256': digest}
    manifest = {'schema_version': 1, 'kind': 'nnslr-networks', 'name': name, 'created': '2026-10-25T22:00:00Z',
                'license': 'MIT', 'experimental': True, 'networks': networks,
                'pipeline': pipeline({'ownership_families': ['maximum_speed']}, .3, None, 160, SIGNNET_CONTRACT)
                if cutter else pipeline({'ownership_families': ['maximum_speed']}, .1, .5, 160),
                'measured': measured({'cutter': cutter or {'id': 'cutter:run/cutter', 'test': {'found': 5, 'truth_boxes': 10}},
                                      'classifier': {'chain': {'truth': 8, 'correct': 4, 'cutter_id': 'other'}},
                                      'decider': {'test': {'other_road_precision': .5, 'other_road_recall': .25}}}),
                'training_data': {'boxes': 1234, 'frames': 99, 'drives': 3, 'dataset_sha256': 'abc'}}
    (folder / 'networks.json').write_text(json.dumps(manifest))
    (folder / 'README.md').write_text(model_card(manifest))
    for file in PACKAGE_FILES:
        if file != 'sha256sums.json':
            sums[file] = hashlib.sha256((folder / file).read_bytes()).hexdigest()
    (folder / 'sha256sums.json').write_text(json.dumps(sums))
    return folder


def test_a_whole_package_passes_and_a_changed_one_does_not(tmp_path):
    folder = _package(tmp_path / 'publish' / 'networks' / 'speedlimits-20261025-2200')
    report = verify_package(folder)
    assert report['valid'] and report['networks']['decider'] == 'decider:run/decider'
    (folder / 'decider.onnx').write_bytes(b'changed')
    with pytest.raises(PackageError, match='decider.onnx: sha256'):
        verify_package(folder)


def test_a_package_never_holds_a_path_of_this_server_or_an_extra_file(tmp_path):
    folder = _package(tmp_path / 'p')
    (folder / 'notes.txt').write_text('x')
    with pytest.raises(PackageError, match='extra'):
        verify_package(folder)
    (folder / 'notes.txt').unlink()
    contract = json.loads((folder / 'cutter.onnx.contract.json').read_text())
    contract['checkpoint'] = '/root/data/runs/x/detector-best.pt'
    (folder / 'cutter.onnx.contract.json').write_text(json.dumps(contract))
    sums = json.loads((folder / 'sha256sums.json').read_text())
    sums['cutter.onnx.contract.json'] = hashlib.sha256((folder / 'cutter.onnx.contract.json').read_bytes()).hexdigest()
    (folder / 'sha256sums.json').write_text(json.dumps(sums))
    with pytest.raises(PackageError, match='path of this server'):
        verify_package(folder)


def test_the_model_card_says_what_it_is_how_to_run_it_and_what_it_measured(tmp_path):
    card = (_package(tmp_path / 'p') / 'README.md').read_text()
    assert card.startswith('---\nlicense: mit') and '**Experimental.**' in card
    assert '`decider:run/decider`' in card and 'score >= 0.1' in card and 'Drop the boxes called not_a_sign' in card
    assert 'found 5 of 10 (50%) signs' in card and '(measured with another cutter)' in card
    assert '1,234 reviewed sign boxes' in card and 'No image is published' in card


def test_a_package_with_a_network_that_learnt_mapillarys_pictures_has_their_licence_and_says_so(tmp_path):
    """The Mapillary Traffic Sign Dataset is CC BY-NC-SA 4.0: weights that learnt from it, themselves or through the
    weights they went on from, are not published as MIT."""
    folder = _package(tmp_path / 'p')
    manifest = json.loads((folder / 'networks.json').read_text())
    assert package_license(manifest['networks']) == 'MIT'
    for role in ('cutter', 'classifier'):
        manifest['networks'][role]['uses_mapillary'] = True
    manifest['networks']['decider']['uses_mapillary'] = False
    manifest['license'] = package_license(manifest['networks'])
    assert manifest['license'] == 'CC BY-NC-SA 4.0'
    manifest['training_data']['countries'] = {'CH': 43, 'DE': 10, 'FR': 3, 'US': 2}
    card = model_card(manifest)
    assert card.startswith('---\nlicense: cc-by-nc-sa-4.0\n')
    assert 'drives in Switzerland, Germany, France and the USA, recorded by comma devices' in card
    assert ('Before these drives the cutter and classifier learnt from the [Mapillary Traffic Sign Dataset]'
            '(https://www.mapillary.com/dataset/trafficsign)') in card and '© Mapillary, CC BY-NC-SA 4.0' in card
    assert 'CC BY-NC-SA 4.0: free to use, change and share for non-commercial purposes' in card
    assert 'MIT. ' not in card.split('## Licence')[1].split('The code that runs them is MIT')[0]
    # Without it: ours, and a manifest of before, with no countries, keeps the two of its time.
    plain = (folder / 'README.md').read_text()
    assert '## Licence\n\nMIT. ' in plain and 'Mapillary' not in plain and 'drives in Switzerland and Germany, recorded' in plain


def test_the_model_card_says_what_each_number_of_the_classifier_means(tmp_path):
    """The classifier gives a number a class: without their names nobody else could read it."""
    manifest = json.loads((_package(tmp_path / 'p') / 'networks.json').read_text())
    assert '## What the classifier says' not in model_card(manifest)  # a manifest of before, without the classes
    manifest['classes'] = ['cancellation', 'maximum_speed:100', 'maximum_speed:30', 'maximum_speed:unknown',
                           'maximum_speed_hours:30', 'maximum_speed_hours:30:22-6', 'not_a_sign', 'other_sign',
                           'other_sign:give_way', 'other_sign:roundabout']
    card = model_card(manifest)
    assert 'one score for each of its 10 classes' in card and '`output.classes` of `classifier.onnx.contract.json`' in card
    assert '| `maximum_speed` | a speed limit, with the number written on the sign' in card
    assert '| `30`, `100`, `unknown` |' in card and '| `30`, `30:22-6` |' in card
    assert '| `other_sign` | another traffic sign or a traffic light, with its kind; alone, one of no kind in the list | ' \
           '`give_way`, `roundabout` |' in card
    assert '| `cancellation` | the end of a speed limit |  |' in card
    assert card.index('## What the classifier says') < card.index('## Measured')


def test_the_countries_of_a_package_are_those_its_drives_were_driven_in(tmp_path):
    for drive, codes in (('a', ['CH', 'CH', 'DE']), ('b', ['CH', None]), ('c', [])):
        folder = tmp_path / 'derived' / 'gps' / drive
        folder.mkdir(parents=True)
        (folder / 'positions.jsonl').write_text(''.join(json.dumps({'country': code}) + '\n' for code in codes))
    assert drive_countries(tmp_path, {'a', 'b', 'c', 'gone', None}) == {'CH': 2, 'DE': 1}


def test_the_model_card_of_a_signnet_cutter_says_how_the_car_runs_it(tmp_path):
    signnet = {'id': 'cutter:s1/signnet', 'kind': 'signnet', 'test': {
        'found': 9, 'truth_boxes': 20, 'score_threshold': 0.1, 'ms_per_frame': None, 'device': None,
        'budget': {'boxes': 5, 'truth_boxes': 20, 'found': 11, 'recall': 0.55},
        'car': {'max_boxes': 5, 'score_threshold': 0.3, 'truth_boxes': 20, 'found': 8, 'recall': 0.4,
                'false_per_frame': 0.22}}}
    numbers = measured({'cutter': signnet})['cutter']
    # As the car reads it (its best 5 boxes of 0.3 or more), never the 5 best at any score.
    assert numbers == {'kind': 'signnet', 'as_the_car': True, 'boxes': 5, 'score_threshold': 0.3, 'signs': 20,
                       'found': 8, 'recall': 0.4, 'false_boxes_per_frame': 0.22, 'ms_per_frame': None, 'device': None}
    steps = pipeline({'ownership_families': ['maximum_speed']}, .3, None, 160, SIGNNET_CONTRACT)
    assert (steps['cutter_kind'], steps['band'], steps['stride'], steps['max_boxes'], steps['max_sign_side']) == (
        'signnet', [896, 288], 8, 5, None)
    card = (_package(tmp_path / 'p', cutter=signnet) / 'README.md').read_text()
    assert 'SignNet, the network the comma runs' in card and 'top 288 rows of the frame scaled to 896 px' in card
    assert 'scaled to 896 px wide (x 896/1344)' in card and '1 x 3 x 288 x 896' in card
    assert 'heat map (channel 0 of out) of score >= 0.3, best first, at most 5' in card
    assert 'times frame width / 896 in frame pixels' in card and 'SSDlite' not in card
    # Measured as the car reads it, with no time when none was measured (never "on a None").
    assert ('- Cutter: finds 8 of 20 (40%) signs as the car reads them: its best 5 boxes of each frame at score 0.3 '
            'or more, 0.22 false boxes per frame.') in card and 'None' not in card and '11 of 20' not in card
    # A SignNet whose entry has no car numbers: its 5 best at any score, said so.
    without = {**signnet, 'test': {key: value for key, value in signnet['test'].items() if key != 'car'}}
    assert measured({'cutter': without})['cutter']['as_the_car'] is False
    card = (_package(tmp_path / 'r', cutter=without) / 'README.md').read_text()
    assert ('- Cutter among its 5 best boxes of each frame at any score (the car keeps only those of 0.3 or more): '
            'found 11 of 20 (55%) signs.') in card
    assert 'SignNet is trained from scratch' in card
    # An old Detector still in use keeps its text.
    old = (_package(tmp_path / 'q') / 'README.md').read_text()
    assert 'SSDlite MobileNetV3-Large on the whole 1344 x 760 frame' in old and 'whose longest side is at most 0.5' in old


def test_the_catalog_lists_every_file_with_its_url_newest_first(tmp_path):
    first = _package(tmp_path / 'a', 'pkg-a')
    second = _package(tmp_path / 'b', 'pkg-b')
    catalog = tmp_path / 'speed_networks.json'
    add_to_catalog(catalog, first, 'https://huggingface.co/u/r/resolve/main/')
    add_to_catalog(catalog, second, 'https://huggingface.co/u/r/resolve/main')
    add_to_catalog(catalog, first, 'https://huggingface.co/u/r/resolve/main')
    content = json.loads(catalog.read_text())
    assert [item['name'] for item in content['packages']] == ['pkg-a', 'pkg-b']
    files = {item['name']: item for item in content['packages'][0]['files']}
    assert sorted(files) == sorted(PACKAGE_FILES)
    assert files['cutter.onnx']['url'] == 'https://huggingface.co/u/r/resolve/main/networks/pkg-a/cutter.onnx'
    assert content['packages'][0]['networks']['classifier']['id'] == 'classifier:run/classifier'


def test_the_packages_here_say_where_they_were_published(tmp_path):
    folder = _package(tmp_path / 'publish' / 'networks' / 'speedlimits-20261025-2200')
    assert packages(tmp_path)[0]['published'] is None
    mark_published(folder, 'u/r', 'https://huggingface.co/u/r/tree/main/networks/x')
    found = packages(tmp_path)[0]
    assert found['published']['repo'] == 'u/r' and found['networks']['cutter'] == 'cutter:run/cutter'
    # The record of the publication is not one of its files: the package still passes its check.
    assert verify_package(folder)['valid']
