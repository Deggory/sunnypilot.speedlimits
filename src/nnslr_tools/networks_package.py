"""The package of our three networks for others (docs/NEURAL_NETWORK.md): the cutter, the classifier and the decider
in use (Models tab), exported to ONNX, with a model card and a manifest, published together on Hugging Face in a
catalog the car reads (scripts/publish_networks.sh, the Publish card of the Models tab).

A package is a folder <data root>/publish/networks/<name>/ holding exactly PACKAGE_FILES:

- ``cutter.onnx``, ``classifier.onnx``, ``decider.onnx`` and the contract of each (``<file>.contract.json``: inputs,
  outputs, classes, the sha256 of the ONNX), written by nnslr_tools.exporting with a parity check against PyTorch;
- ``networks.json``: the manifest, the id of each network on this server, how to run them in a row, what they
  measured on the test drives;
- ``README.md``: the model card;
- ``sha256sums.json``: the sha256 of every other file.

No frame, dataset, checkpoint or path of this server is in it: the contracts keep the id of the model in place of
its checkpoint path. prepare_package needs torch (the GPU environment); verify_package and the catalog only the
standard library.

The catalog, ``speed_networks.json`` at the root of the Hugging Face repository: ``{"schema_version": 1,
"packages": [...]}``, the newest first, each with its name, date, the id and sha256 of its three networks and every
file with its size, sha256 and download url. The same name again replaces the package.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
from typing import Any, Callable

KIND = 'nnslr-networks'
SCHEMA_VERSION = 1
PACKAGE_DIR = Path('publish') / 'networks'
CATALOG_NAME = 'speed_networks.json'
NAME = re.compile(r'[A-Za-z0-9][A-Za-z0-9._-]{0,99}')
ROLES = ('cutter', 'classifier', 'decider')
NETWORK_FILES = tuple(f'{role}.onnx' for role in ROLES)
PACKAGE_FILES = (*NETWORK_FILES, *(f'{name}.contract.json' for name in NETWORK_FILES), 'networks.json', 'README.md',
                 'sha256sums.json')
# Written next to the package once it is on Hugging Face (not one of its files).
PUBLISHED_FILE = 'published.json'
LICENSE = 'MIT'
# The lowest cutter score kept, as Prepare's "ours" proposer (nnslr_tools.preannotate.proposer_setup).
CUTTER_SCORE = 0.1


class PackageError(ValueError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')


def package_path(root: Path, name: str) -> Path:
    if not isinstance(name, str) or not NAME.fullmatch(name):
        raise PackageError(f'invalid package name {name!r}: letters, digits, _ . - (starting with a letter or digit)')
    return Path(root) / PACKAGE_DIR / name


def new_name(now: datetime | None = None) -> str:
    """speedlimits-<date>-<time>, UTC: a new name for every package."""
    return 'speedlimits-' + (now or datetime.now(timezone.utc)).strftime('%Y%m%d-%H%M')


# -- what the package says about the networks (stdlib) --------------------------------------------------------------

def _share(part: Any, whole: Any) -> str:
    return f'{part} of {whole}' + (f' ({100 * part / whole:.0f}%)' if isinstance(part, int) and whole else '')


def measured(entries: dict[str, dict]) -> dict[str, Any]:
    """The test numbers of the three networks from their Models-tab entries (nnslr_tools.model_roles)."""
    cutter, classifier, decider = (entries.get(role) or {} for role in ROLES)
    result: dict[str, Any] = {}
    test = cutter.get('test') or {}
    if test:
        result['cutter'] = {'score_threshold': test.get('score_threshold'), 'signs': test.get('truth_boxes'),
                            'found': test.get('found'), 'recall': test.get('recall'),
                            'ms_per_frame': test.get('ms_per_frame'), 'device': test.get('device')}
    chain = classifier.get('chain') or {}
    if chain:
        result['cutter_and_classifier'] = {
            'speed_signs': chain.get('truth'), 'found': chain.get('found'), 'read_right': chain.get('correct'),
            'false_boxes_per_frame': chain.get('surviving_false_per_frame'), 'frames': chain.get('images'),
            # The chain is measured with the cutter the classifier was trained with: say when it is another one.
            'with_this_cutter': chain.get('cutter_id') == cutter.get('id')}
    test = decider.get('test') or {}
    if test:
        result['decider'] = {'ownership_boxes': test.get('ownership_samples'), 'macro_f1': test.get('macro_f1'),
                             'another_road_precision': test.get('other_road_precision'),
                             'another_road_recall': test.get('other_road_recall'),
                             'another_road_boxes': test.get('other_road_support'),
                             'abstain_rate': test.get('abstain_rate')}
    return result


def pipeline(decider_contract: dict, cutter_score: float, max_sign_side: float, classifier_size: int) -> dict:
    """How the three run in a row, as Prepare's "ours" proposer runs them (nnslr_tools._chain_proposer)."""
    return {
        'camera': 'comma narrow road camera (fcamera), 1344 x 760 RGB',
        'steps': [
            {'network': 'cutter', 'input': 'the frame as RGB values 0..1, 1 x 3 x 760 x 1344',
             'then': f'Keep the boxes of score >= {cutter_score} whose longest side is at most {max_sign_side} of '
                     'the frame height'},
            {'network': 'classifier', 'input': f'each box cut from the frame (rounded outwards, at least 2 px), '
                                               f'resized to {classifier_size} x {classifier_size}, ImageNet '
                                               'normalisation; all the boxes of a frame in one batch',
             'then': 'Drop the boxes called not_a_sign'},
            {'network': 'decider', 'input': 'the speed-sign boxes of the frame (families '
                                            + ', '.join(decider_contract.get('ownership_families', [])) + ') with '
                                            'their inputs as decider.onnx.contract.json says, in one call',
             'then': 'own_road: the sign applies to the car; other_road: it does not; unresolved: the decider '
                     'abstains'},
        ],
        'cutter_score': cutter_score,
        'max_sign_side': max_sign_side,
    }


def _count(value: Any) -> str:
    return f'{value:,}' if isinstance(value, int) else '?'


def _number(value: Any, digits: int = 2) -> str:
    return '–' if not isinstance(value, (int, float)) else f'{value:.{digits}f}'


def model_card(manifest: dict) -> str:
    """README.md of the package."""
    networks, numbers, data = manifest['networks'], manifest.get('measured', {}), manifest.get('training_data', {})
    cutter = numbers.get('cutter', {})
    chain = numbers.get('cutter_and_classifier', {})
    decider = numbers.get('decider', {})
    lines = [
        '---', f'license: {LICENSE.lower()}', 'tags:', '- onnx', '- traffic-sign-recognition', '- openpilot',
        '- sunnypilot', '---', '',
        f'# Speed-limit networks {manifest["name"]}', '',
        'Three small networks that find speed signs on the narrow road camera of a comma device, read them and say '
        'whether they apply to the car, for sunnypilot\'s Speed Limit Assist '
        '([sunnypilot.speedlimits](https://github.com/cristianku/sunnypilot.speedlimits)).', '',
        '**Experimental.** A help, never a guarantee: the driver always stays responsible for the speed.', '',
        '| Network | What it does | File | Version |', '|---|---|---|---|',
        f'| Cutter | SSDlite MobileNetV3-Large on the whole 1344 x 760 frame: a box on everything that could be a '
        f'sign | `cutter.onnx` | `{networks["cutter"]["id"]}` |',
        f'| Classifier | MobileNetV3-Small on each box: what it is (a speed limit and its number, end of limit, '
        f'variable display, other signs, not a sign) | `classifier.onnx` | `{networks["classifier"]["id"]}` |',
        f'| Decider | MobileNetV3-Small with the box position, the car speed and the scene: whom a speed sign '
        f'applies to (our road, another road, or it abstains) | `decider.onnx` | `{networks["decider"]["id"]}` |',
        '', '## How to run them', '',
    ]
    for number, step in enumerate(manifest['pipeline']['steps'], 1):
        lines.append(f'{number}. **{step["network"].capitalize()}**: {step["input"]}. {step["then"]}.')
    lines += ['', 'Each `*.contract.json` gives the exact inputs, outputs and classes of its network.', '',
              '## Measured', '', 'On test drives kept apart by place (never trained on):', '']
    if cutter:
        found = _share(cutter.get('found'), cutter.get('signs'))
        lines.append(f'- Cutter at score {cutter.get("score_threshold")}: found {found} '
                     f'signs, {_number(cutter.get("ms_per_frame"), 1)} ms a frame on a {cutter.get("device")}.')
    if chain:
        right = _share(chain.get('read_right'), chain.get('speed_signs'))
        lines.append(f'- Cutter and classifier: {right} speed signs '
                     f'found and read right; {_number(chain.get("false_boxes_per_frame"))} false boxes left per frame'
                     + ('' if chain.get('with_this_cutter') else ' (measured with another cutter)') + '.')
    if decider:
        lines.append(f'- Decider: "another road" precision {_number(decider.get("another_road_precision"))} and recall '
                     f'{_number(decider.get("another_road_recall"))} on {decider.get("another_road_boxes")} such '
                     'boxes; '
                     f'ownership macro-F1 {_number(decider.get("macro_f1"))}.')
    lines += ['', '## Trained on', '',
              f'{_count(data.get("boxes"))} reviewed sign boxes on {_count(data.get("frames"))} frames of '
              f'{_count(data.get("drives"))} drives in Switzerland and Germany, checked by a person or by an AI '
              'reviewer. No image is published. '
              f'Dataset sha256 `{data.get("dataset_sha256")}`.', '',
              '## Licence', '', f'{LICENSE}. The networks start from torchvision\'s ImageNet weights.', '']
    return '\n'.join(lines)


def training_data(root: Path, dataset_sha256: str | None) -> dict:
    """Counts of the reviewed boxes of the dataset (annotations/objects.jsonl): boxes, frames, drives."""
    path = Path(root) / 'annotations' / 'objects.jsonl'
    boxes, frames, drives = 0, set(), set()
    if path.is_file():
        for line in path.open(encoding='utf-8'):
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get('review_state') not in ('accepted', 'corrected'):
                continue
            frames.add(row.get('image_path'))
            drives.add(row.get('route_id'))
            if row.get('bbox_xyxy') and row.get('sign_family') not in (None, 'not_a_sign'):
                boxes += 1
    return {'boxes': boxes, 'frames': len(frames), 'drives': len(drives), 'dataset_sha256': dataset_sha256}


# -- verify (stdlib) ------------------------------------------------------------------------------------------------

def verify_package(folder: Path) -> dict:
    """The package is whole and unchanged: exactly its files, every sha256 right, the manifest and the contracts
    agreeing with the ONNX files. Raises PackageError on the first problem."""
    folder = Path(folder)
    if not folder.is_dir():
        raise PackageError(f'{folder} is not a folder')
    present = sorted(p.name for p in folder.iterdir() if p.name != PUBLISHED_FILE)
    if present != sorted(PACKAGE_FILES):
        missing, extra = sorted(set(PACKAGE_FILES) - set(present)), sorted(set(present) - set(PACKAGE_FILES))
        raise PackageError(f'the files are not those of a package: missing {missing}, extra {extra}')
    sums = json.loads((folder / 'sha256sums.json').read_text(encoding='utf-8'))
    if sorted(sums) != sorted(set(PACKAGE_FILES) - {'sha256sums.json'}):
        raise PackageError('sha256sums.json does not list every file')
    for name, digest in sums.items():
        if _sha256(folder / name) != digest:
            raise PackageError(f'{name}: sha256 differs from sha256sums.json')
    manifest = json.loads((folder / 'networks.json').read_text(encoding='utf-8'))
    if manifest.get('kind') != KIND or manifest.get('schema_version') != SCHEMA_VERSION:
        raise PackageError('networks.json is not a manifest of this kind and version')
    for role in ROLES:
        entry = (manifest.get('networks') or {}).get(role) or {}
        contract = json.loads((folder / f'{role}.onnx.contract.json').read_text(encoding='utf-8'))
        if entry.get('file') != f'{role}.onnx' or entry.get('sha256') != sums[f'{role}.onnx'] \
                or contract.get('onnx_sha256') != sums[f'{role}.onnx']:
            raise PackageError(f'{role}: the manifest or the contract does not match {role}.onnx')
        if not (contract.get('verification') or {}).get('passed'):
            raise PackageError(f'{role}: its ONNX was not checked against PyTorch')
        if str(contract.get('checkpoint', '')).startswith('/'):
            raise PackageError(f'{role}: the contract holds a path of this server')
    return {'valid': True, 'name': manifest.get('name'), 'folder': str(folder),
            'networks': {role: manifest['networks'][role]['id'] for role in ROLES},
            'bytes': sum((folder / name).stat().st_size for name in PACKAGE_FILES)}


# -- prepare (torch) ------------------------------------------------------------------------------------------------

def example_frame(root: Path) -> Path | None:
    """A reviewed 1344 x 760 frame of the dataset with a sign, to check the cutter's ONNX on (None without one)."""
    path = Path(root) / 'annotations' / 'objects.jsonl'
    if not path.is_file():
        return None
    for line in path.open(encoding='utf-8'):
        try:
            row = json.loads(line)
        except ValueError:
            continue
        image = Path(root) / str(row.get('image_path'))
        if (row.get('sign_family') == 'maximum_speed' and row.get('width') == 1344 and row.get('height') == 760
                and image.is_file()):
            return image
    return None


def prepare_package(root: Path, name: str, *, log: Callable[[str], None] = print) -> dict:
    """Export the three networks in use into a new package <data root>/publish/networks/<name>/ (replacing one of
    the same name), write its manifest, model card and sums, and verify it."""
    from nnslr_tools.chain import MAX_SIGN_SIDE, resolve_model
    from nnslr_tools.exporting import export_decider_onnx, export_detector_onnx, export_reader_onnx
    from nnslr_tools.model_roles import ModelIndex
    from nnslr_tools.preannotate import ours_missing
    from nnslr_tools.runs import RunsIndex

    root = Path(root)
    target = package_path(root, name)
    missing = ours_missing(root)
    if missing:
        raise PackageError(missing)
    resolved = {}
    for role in ROLES:
        try:
            resolved[role] = resolve_model('active', role, root)
        except ValueError as exc:
            raise PackageError(f'{role}: {exc}') from exc
    index = ModelIndex(RunsIndex(root))
    artifacts = index.artifacts()
    entries = {role: next((e for e in artifacts[role] if e['id'] == resolved[role]['id']), {}) for role in ROLES}

    work = target.with_name('.' + name + '.tmp')
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    try:
        exports = {}
        frame = example_frame(root)
        for number, role in enumerate(ROLES, 1):
            log(f'== {number}/4 Export the {role} ({resolved[role]["id"]})')
            if role == 'cutter':
                exports[role] = export_detector_onnx(resolved[role]['path'], work / 'cutter.onnx', example_image=frame)
            else:
                export = export_reader_onnx if role == 'classifier' else export_decider_onnx
                exports[role] = export(resolved[role]['path'], work / f'{role}.onnx')
            log(f'{role}: ONNX checked against PyTorch')
        log('== 4/4 Write the manifest and the model card, and check the package')
        contracts = {}
        for role in ROLES:
            path = work / f'{role}.onnx.contract.json'
            contract = json.loads(path.read_text(encoding='utf-8'))
            # The id of the model, never a path of this server.
            contract['checkpoint'] = resolved[role]['id']
            _write_json(path, contract)
            contracts[role] = contract
        classifier_size = int(contracts['classifier']['input']['resize'][0])
        manifest = {
            'schema_version': SCHEMA_VERSION, 'kind': KIND, 'name': name,
            'created': datetime.now(timezone.utc).isoformat(timespec='seconds').replace('+00:00', 'Z'),
            'license': LICENSE, 'experimental': True,
            'networks': {role: {'id': resolved[role]['id'], 'file': f'{role}.onnx',
                                'sha256': exports[role]['onnx_sha256'], 'contract': f'{role}.onnx.contract.json',
                                'checkpoint_sha256': contracts[role].get('checkpoint_sha256'),
                                'dataset_sha256': contracts[role].get('dataset_sha256'),
                                'split_sha256': contracts[role].get('split_sha256')} for role in ROLES},
            'pipeline': pipeline(contracts['decider'], CUTTER_SCORE, MAX_SIGN_SIDE, classifier_size),
            'measured': measured(entries),
            'training_data': training_data(root, contracts['decider'].get('dataset_sha256')),
        }
        _write_json(work / 'networks.json', manifest)
        (work / 'README.md').write_text(model_card(manifest), encoding='utf-8')
        _write_json(work / 'sha256sums.json', {name_: _sha256(work / name_) for name_ in PACKAGE_FILES
                                               if name_ != 'sha256sums.json'})
        verify_package(work)
        shutil.rmtree(target, ignore_errors=True)
        os.replace(work, target)
    except BaseException:
        shutil.rmtree(work, ignore_errors=True)
        raise
    report = verify_package(target)
    log(f'Package ready: {target} ({report["bytes"] / 1e6:.1f} MB). Nothing was uploaded.')
    return report


# -- the packages here and the catalog (stdlib) ---------------------------------------------------------------------

def packages(root: Path) -> list[dict]:
    """The packages prepared here, newest first: name, date, networks, size, and where it was published."""
    folder = Path(root) / PACKAGE_DIR
    found = []
    if not folder.is_dir():
        return found
    for path in folder.iterdir():
        if not path.is_dir() or not NAME.fullmatch(path.name):
            continue
        try:
            manifest = json.loads((path / 'networks.json').read_text(encoding='utf-8'))
        except (OSError, ValueError):
            continue
        try:
            published = json.loads((path / PUBLISHED_FILE).read_text(encoding='utf-8'))
        except (OSError, ValueError):
            published = None
        found.append({'name': path.name, 'created': manifest.get('created'),
                      'networks': {role: ((manifest.get('networks') or {}).get(role) or {}).get('id')
                                   for role in ROLES},
                      'measured': manifest.get('measured'),
                      'bytes': sum(p.stat().st_size for p in path.iterdir() if p.is_file()),
                      'published': published})
    return sorted(found, key=lambda item: str(item['created']), reverse=True)


def mark_published(folder: Path, repo: str, url: str) -> dict:
    record = {'repo': repo, 'url': url,
              'at': datetime.now(timezone.utc).isoformat(timespec='seconds').replace('+00:00', 'Z')}
    _write_json(Path(folder) / PUBLISHED_FILE, record)
    return record


def add_to_catalog(catalog_path: Path, folder: Path, base_url: str) -> dict:
    """The catalog (a missing file starts one) with this package first, replacing one of the same name."""
    report = verify_package(folder)
    folder = Path(folder)
    manifest = json.loads((folder / 'networks.json').read_text(encoding='utf-8'))
    name = manifest['name']
    catalog_path = Path(catalog_path)
    catalog = {'schema_version': SCHEMA_VERSION, 'packages': []}
    if catalog_path.is_file():
        catalog = json.loads(catalog_path.read_text(encoding='utf-8'))
        if catalog.get('schema_version') != SCHEMA_VERSION or not isinstance(catalog.get('packages'), list):
            raise PackageError(f'{catalog_path} is not a catalog of version {SCHEMA_VERSION}')
    base = base_url.rstrip('/')
    entry = {'name': name, 'created': manifest['created'], 'license': manifest['license'],
             'experimental': manifest['experimental'],
             'networks': {role: {'id': manifest['networks'][role]['id'], 'file': f'{role}.onnx',
                                 'sha256': manifest['networks'][role]['sha256']} for role in ROLES},
             'files': [{'name': file, 'size': (folder / file).stat().st_size, 'sha256': _sha256(folder / file),
                        'url': f'{base}/networks/{name}/{file}'} for file in PACKAGE_FILES]}
    catalog['packages'] = [entry] + [item for item in catalog['packages'] if item.get('name') != name]
    _write_json(catalog_path, catalog)
    return {'catalog': str(catalog_path), 'packages': len(catalog['packages']), 'name': name,
            'bytes': report['bytes']}
