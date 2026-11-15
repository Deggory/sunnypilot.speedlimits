"""The networks that read the signs a brain learns from, chosen in Train › Brain (#31, 10 Nov 2026): the cutter (a SignNet
the comma runs) and the classifier, those in use on the Models tab unless others are chosen.

The brain is given the signs the car reads. On the car speedvisiond reads them with the networks of its release, every
one of them: no applicability network leaves out those of another road since 15 Nov 2026 (#90), the brain learns whose
road a sign is of itself, from the car's motion and the picture, with its applicability head trained on the reviews.
The brain's training reads them the same way with the two networks chosen here, for its inputs (road_type_signs): the
car's own vision.py, with the two networks exported as Publish exports them (ONNX, contracts, speed_limits.json with
the car's settings of today), into derived/road_type_signs/sets/<key>/, one folder per choice (set_key).

scripts/run_brain.sh writes the choice into its run (FILE, networks.json: the ids and the exported folder), every step
reads it from there, and the brain's result.json records the ids. A brain trained from 10 to 15 Nov 2026 recorded an
applicability network too (decider): its signs were read with it, so of_brain and load still take such a choice, for
Regression and Models › Drive; Publish refuses that brain (car_release._check_sign_networks)."""
from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from pathlib import Path

PARTS = ('cutter', 'classifier')
# The applicability network a brain of 10 to 15 Nov 2026 read its signs with: only in the choices of such brains.
OLD_PART = 'decider'
WORDS = {'cutter': 'cutter', 'classifier': 'classifier', 'decider': 'applicability network'}
FILE = 'networks.json'
SETS_RELATIVE = Path('derived') / 'road_type_signs' / 'sets'
SOURCE_FILE = 'source.json'
MANIFEST = 'speed_limits.json'


def parts_of(ids: dict) -> tuple[str, ...]:
    """The parts of a choice: the cutter and the classifier, and the applicability network of a brain of 10 to 15 Nov
    2026 that read its signs with one."""
    return PARTS + ((OLD_PART,) if ids.get(OLD_PART) else ())


def in_use(root: Path) -> dict[str, str]:
    """The ids of the cutter and the classifier in use (models/active.json), those missing left out."""
    from nnslr_tools.preannotate import networks_in_use
    return {part: value for part, value in networks_in_use(root).items() if part in PARTS}


def resolve(root: Path, cutter: str | None = None, classifier: str | None = None) -> tuple[dict[str, str], dict]:
    """The ids of the two networks (those not given: the ones in use) and car_release.choose's networks for them,
    checked as Publish checks them: a finished SignNet of the car's band and grid, a finished classifier. Raises
    ValueError when one is missing or does not fit."""
    from nnslr_tools import car_release
    from nnslr_tools.model_roles import MODEL_ID, SIGNNET_PART

    used = in_use(root)
    ids = {'cutter': cutter or used.get('cutter'), 'classifier': classifier or used.get('classifier')}
    missing = [WORDS[part] for part in PARTS if not ids[part]]
    if missing:
        raise ValueError('the brain reads its signs with a cutter and a classifier: no '
                         + ', '.join(missing) + ' is in use or chosen (Models tab)')
    match = MODEL_ID.fullmatch(ids['cutter'])
    if match is None or match.group(1) != 'cutter' or match.group(3) != SIGNNET_PART:
        raise ValueError(f"{ids['cutter']} is not a SignNet: the comma reads its signs with a SignNet")
    return ids, car_release.choose(root, match.group(2), ids['classifier'], brain=False)


def set_key(ids: dict[str, str]) -> str:
    """The folder of a choice: the first 16 hex of the sha256 of its ids (with the applicability network of an old
    choice, so its folder stays the same)."""
    return hashlib.sha256(json.dumps([ids[part] for part in parts_of(ids)]).encode()).hexdigest()[:16]


def set_dir(root: Path, ids: dict[str, str]) -> Path:
    return Path(root) / SETS_RELATIVE / set_key(ids)


def exported(root: Path, ids: dict[str, str]) -> Path | None:
    """The folder of a choice when it is exported whole (its source.json names these ids and every file is there),
    else None."""
    from nnslr_tools.car_release import manifest_of
    folder = set_dir(root, ids)
    try:
        source = json.loads((folder / SOURCE_FILE).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None
    files = manifest_of('brain-signs', decider=bool(ids.get(OLD_PART)))
    names = [entry[key] for entry in files.values() if isinstance(entry, dict) for key in ('file', 'contract')]
    if source != {part: ids[part] for part in parts_of(ids)} or not all((folder / name).is_file() for name in names):
        return None
    return folder if (folder / MANIFEST).is_file() else None


def export(root: Path, ids: dict[str, str], found: dict, log=print) -> Path:
    """The networks of a choice exported as Publish exports them (car_release.export: each checked against
    PyTorch), with the car's settings of today, into set_dir; nothing when they are there already, only
    speed_limits.json again when the car's settings alone changed. In the GPU environment: the export is PyTorch's."""
    from nnslr_tools import car_release, car_settings

    target = set_dir(root, ids)
    manifest = car_release.manifest_of('brain-signs', car_settings.values(root), brain=False,
                                       decider=bool(found.get('decider')))
    text = json.dumps(manifest, indent=2) + '\n'
    if exported(root, ids) is not None:
        if (target / MANIFEST).read_text(encoding='utf-8') != text:
            _write(target / MANIFEST, text)
            log("the car's settings changed: written again for " + ', '.join(ids[part] for part in parts_of(ids)))
        else:
            log('the networks of the signs are exported already: ' + ', '.join(ids[part] for part in parts_of(ids)))
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix=target.name + '-', dir=target.parent))
    try:
        car_release.export(found, folder, drive_check=False)
        (folder / MANIFEST).write_text(text, encoding='utf-8')
        (folder / SOURCE_FILE).write_text(json.dumps({part: ids[part] for part in parts_of(ids)}, indent=2) + '\n',
                                          encoding='utf-8')
        old = target.with_name(target.name + '.old')
        shutil.rmtree(old, ignore_errors=True)
        if target.exists():
            target.rename(old)
        folder.rename(target)
        shutil.rmtree(old, ignore_errors=True)
    finally:
        shutil.rmtree(folder, ignore_errors=True)
    log('the networks of the signs exported: ' + ', '.join(ids[part] for part in parts_of(ids)))
    return target


def _write(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(text, encoding='utf-8')
    temporary.replace(path)


def write(run_dir: Path, ids: dict[str, str], models: Path) -> Path:
    """The run's choice (FILE): the ids and the exported folder."""
    path = Path(run_dir) / FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    _write(path, json.dumps({**{part: ids[part] for part in parts_of(ids)}, 'models': str(models)}, indent=2) + '\n')
    return path


def load(path: Path) -> dict:
    """A run's choice: {'cutter', 'classifier' (ids), 'decider' only in that of a brain of 10 to 15 Nov 2026, 'models'
    (the exported folder, a Path)}. Raises ValueError when it is not whole."""
    try:
        found = json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise ValueError(f'the networks of the signs cannot be read from {path}: {exc}') from exc
    if not isinstance(found, dict) or not all(isinstance(found.get(key), str) and found[key] for key in (*PARTS, 'models')):
        raise ValueError(f'{path} does not name the networks of the signs and their folder')
    return {**{part: found[part] for part in parts_of(found)}, 'models': Path(found['models'])}


def of_brain(root: Path, recorded: dict | None) -> dict | None:
    """The networks a brain's signs were read with (its result.json's sign_networks), as load's, for what runs it along a
    drive (Publish's check, Regression): None for a brain from before 10 Nov 2026, which learnt from the car's release of
    the sunnypilot clone. Raises ValueError when their folder is gone."""
    if not recorded:
        return None
    ids = {part: recorded[part] for part in parts_of(recorded)}
    folder = exported(root, ids)
    if folder is None:
        if ids.get(OLD_PART):
            raise ValueError('the networks its signs were read with are not exported any more, and its applicability '
                             f"network {ids[OLD_PART]} reads no brain's signs since 15 Nov 2026: train a brain again")
        raise ValueError('the networks its signs were read with are not exported any more: nnslr brain-networks '
                         f"--run-dir D --cutter {ids['cutter']} --classifier {ids['classifier']}")
    return {**ids, 'models': folder}


def prepare(root: Path, run_dir: Path, *, cutter: str | None = None, classifier: str | None = None, log=print) -> dict:
    """scripts/run_brain.sh, step 3: the run's choice, written once (a run started again keeps the networks it chose,
    their folder exported again when it went), else the networks given or in use, exported and written. Returns load's.
    A run of before 15 Nov 2026 started again keeps its cutter and classifier and drops its applicability network: its
    drives are read again with every sign."""
    path = Path(run_dir) / FILE
    if path.is_file():
        chosen = load(path)
        if chosen.get(OLD_PART):
            log(f"this run chose the applicability network {chosen[OLD_PART]}: no more since 15 Nov 2026 (#90), its "
                'signs are read again with every sign')
        ids = {part: chosen[part] for part in PARTS}
        if exported(root, ids) is None:
            ids, found = resolve(root, **ids)
            export(root, ids, found, log)
        log('the networks of the signs of this run: ' + ', '.join(f'{WORDS[part]} {ids[part]}' for part in PARTS))
        return load(write(run_dir, ids, set_dir(root, ids)))
    ids, found = resolve(root, cutter, classifier)
    folder = export(root, ids, found, log)
    log('the networks of the signs: ' + ', '.join(f'{WORDS[part]} {ids[part]}' for part in PARTS))
    return load(write(run_dir, ids, folder))
