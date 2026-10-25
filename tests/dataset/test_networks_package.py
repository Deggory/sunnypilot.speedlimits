"""The package of our three networks (nnslr_tools.networks_package): its check, its model card and the catalog, on
a package of small stand-in files (the ONNX export itself needs torch: tests/dataset/test_context_experiment.py)."""
from __future__ import annotations

import hashlib
import json

import pytest

from nnslr_tools.networks_package import (PACKAGE_FILES, PackageError, ROLES, add_to_catalog, mark_published,
                                          measured, model_card, packages, pipeline, verify_package)


def _package(folder, name='speedlimits-20261025-2200'):
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
                'pipeline': pipeline({'ownership_families': ['maximum_speed']}, .1, .5, 160),
                'measured': measured({'cutter': {'id': 'cutter:run/cutter', 'test': {'found': 5, 'truth_boxes': 10}},
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
