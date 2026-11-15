"""The versions of Fit box (#94): what is a version, the one in use, the choice and the record of a training."""
from __future__ import annotations

import json
import shutil

import pytest

from nnslr_tools.box_fit import versions

RUN = '2026-11-16_0900-box-fit'


@pytest.fixture
def root(tmp_path):
    versions._in_use_cache.clear()
    return tmp_path


def a_version(root, run=RUN, *, files=versions.FILES):
    folder = root / 'runs' / run
    folder.mkdir(parents=True)
    for name in files:
        (folder / name).write_bytes(b'x')
    return folder


def test_only_a_real_run_folder_with_both_rankers_is_a_version(root):
    assert versions.run_folder(root, RUN) is None
    folder = a_version(root)
    assert versions.run_folder(root, RUN) == folder
    assert versions.folder_of(root, 'box_fit:' + RUN) == folder
    assert versions.folder_of(root, versions.CODE_ID) == versions.HERE
    for bad in ('../' + RUN, RUN + '/..', '.' + RUN + '.partial', '2026-11-16_0900-road-labeler', None, 5):
        assert versions.run_folder(root, bad) is None, bad
    for bad in ('box_fit:../x', 'cutter:' + RUN, RUN, 'box_fit:', None):
        assert versions.folder_of(root, bad) is None, bad
    # A folder without one of the rankers, a linked folder, a linked ranker: not versions.
    a_version(root, '2026-11-16_1000-box-fit', files=('colour.npz', 'refiner.onnx'))
    (root / 'runs' / '2026-11-16_1100-box-fit').symlink_to(folder)
    linked = a_version(root, '2026-11-16_1200-box-fit', files=('colour.npz',))
    (linked / 'edges.npz').symlink_to(folder / 'edges.npz')
    (root / 'runs' / f'.{RUN}.partial').mkdir()
    assert [version['id'] for version in versions.versions(root)] == ['box_fit:' + RUN, versions.CODE_ID]


def test_runs_that_lead_out_of_the_data_root_are_not_read(root, tmp_path_factory):
    elsewhere = tmp_path_factory.mktemp('elsewhere')
    a_version(elsewhere)
    (root / 'runs').symlink_to(elsewhere / 'runs')
    assert versions.run_folder(root, RUN) is None
    assert [version['id'] for version in versions.versions(root)] == [versions.CODE_ID]


def test_the_code_is_in_use_until_someone_chooses_and_a_choice_holds_from_the_next_fit(root):
    used = versions.in_use(root)
    assert (used['id'], used['folder'], used['chosen'], used['found'], used['problem']) == (
        versions.CODE_ID, versions.HERE, None, True, None)
    assert versions.folder_in_use(root) == versions.HERE
    folder = a_version(root)
    entry = versions.choose(root, 'box_fit:' + RUN, who='x' * 300, via='token')
    assert entry['id'] == 'box_fit:' + RUN and len(entry['who']) == versions.MAX_WHO and entry['via'] == 'token'
    assert versions.folder_in_use(root) == folder
    # Back to the code: the next fit reads it, without a restart.
    versions.choose(root, versions.CODE_ID, who=None, via='local')
    assert versions.folder_in_use(root) == versions.HERE
    versions.choose(root, 'box_fit:' + RUN, who=None, via='local')
    assert versions.folder_in_use(root) == folder
    # Its folder removed: the code's files, at once.
    moved = root / 'elsewhere'
    shutil.move(folder, moved)
    assert versions.folder_in_use(root) == versions.HERE
    used = versions.in_use(root)
    assert used['found'] is False and used['chosen']['id'] == 'box_fit:' + RUN and RUN in used['problem']
    # Back (a restore puts the folder back and leaves the choice as it is): the chosen version again, at the next fit.
    shutil.move(moved, folder)
    assert versions.folder_in_use(root) == folder


def test_a_choice_is_refused_unless_it_names_a_version_here(root):
    a_version(root)
    for value, reason in (('box_fit:2026-11-17_0000-box-fit', 'unknown_version'), ('box_fit:x', 'bad_id'),
                          ('box_fit:' + RUN + 'x' * 80, 'bad_id'), (['box_fit:code'], 'bad_id'), (None, 'bad_id')):
        with pytest.raises(versions.ChoiceError) as refused:
            versions.choose(root, value, who=None, via='local')
        assert refused.value.reason == reason, value
    with pytest.raises(ValueError):
        versions.choose(root, versions.CODE_ID, who=None, via='reviewer')
    assert not (root / 'models' / 'box-fit.json').exists()


@pytest.mark.parametrize('content, problem', [
    ('{"id": ', 'cannot be read'), ('[]', 'does not name'), ('{"id": "cutter:code"}', 'does not name')])
def test_a_choice_file_that_cannot_be_used_is_said_and_never_overwritten(root, content, problem):
    (root / 'models').mkdir()
    (root / 'models' / 'box-fit.json').write_text(content)
    used = versions.in_use(root)
    assert used['id'] == versions.CODE_ID and problem in used['problem']
    with pytest.raises(versions.ChoiceError) as refused:
        versions.choose(root, versions.CODE_ID, who=None, via='local')
    assert refused.value.reason == 'choice_unreadable'
    assert (root / 'models' / 'box-fit.json').read_text() == content


def test_a_linked_choice_file_or_models_folder_is_not_used(root, tmp_path_factory):
    elsewhere = tmp_path_factory.mktemp('elsewhere')
    (elsewhere / 'box-fit.json').write_text(json.dumps({'id': versions.CODE_ID}))
    (root / 'models').mkdir()
    (root / 'models' / 'box-fit.json').symlink_to(elsewhere / 'box-fit.json')
    assert 'not a file' in versions.in_use(root)['problem']
    (root / 'models' / 'box-fit.json').unlink()
    (root / 'models').rmdir()
    (root / 'models').symlink_to(elsewhere)
    assert 'outside' in versions.in_use(root)['problem']
    with pytest.raises(versions.ChoiceError) as refused:
        versions.choose(root, versions.CODE_ID, who=None, via='local')
    assert refused.value.reason == 'bad_models_folder'


def test_the_record_of_a_training_keeps_its_measures_and_nothing_else(root):
    folder = a_version(root)
    rankers = {'boxes': 4100, 'routes': 31, 'folds': 4,
               'cross_validation': {'overall': {'mean_iou': 0.81, 'rough_mean_iou': 0.6, 'other': 1}}}
    refiner = {'refiner': 'carried', 'train_boxes': 3000, 'held_out_boxes': 900, 'best_epoch': True, 'epochs': 40,
               'in_use': {'id': versions.CODE_ID, 'problem': None}, 'better_than_in_use': False,
               'left_out': {'all': {'cases': 10, 'version': 0.8}, 'kind:ring': {'cases': 2}, 'fair': {'cases': 3}}}
    data = versions.write_record(folder, RUN, rankers, refiner, refiner_from=versions.CODE_ID)
    assert json.loads((folder / 'box-fit.json').read_text()) == data
    assert (data['boxes'], data['routes'], data['rankers']) == (4100, 31, {'mean_iou': 0.81, 'rough_mean_iou': 0.6})
    assert (data['refiner'], data['refiner_from'], data['refiner_boxes'], data['best_epoch']) == (
        'carried', versions.CODE_ID, 3000, None)
    assert data['left_out'] == {'all': {'cases': 10, 'version': 0.8}, 'fair': {'cases': 3}}
    assert (data['in_use_then'], data['better_than_in_use']) == (versions.CODE_ID, False)
    data = versions.write_record(folder, RUN, {}, {'refiner': 'anything', 'better_than_in_use': 'yes'})
    assert (data['refiner'], data['better_than_in_use'], data['left_out'], data['boxes']) == ('none', None, {}, None)
    assert versions.versions(root)[0]['at'] == data['at']


def test_the_script_reads_the_version_in_use_and_writes_the_record(root, capsys):
    folder = a_version(root)
    assert versions.main(['in-use', '--data-root', str(root)]) == 0
    assert capsys.readouterr().out == versions.CODE_ID + '\t' + str(versions.HERE) + '\n'
    (root / 'rankers.json').write_text(json.dumps({'boxes': 7}))
    (root / 'refiner.json').write_text('a progress line\n' + json.dumps({'refiner': 'trained'}) + '\n\n')
    assert versions.main(['record', '--folder', str(folder), '--run', RUN, '--rankers', str(root / 'rankers.json'),
                          '--refiner', str(root / 'refiner.json')]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert (printed['boxes'], printed['refiner']) == (7, 'trained')
    assert json.loads((folder / 'box-fit.json').read_text())['refiner'] == 'trained'
    with pytest.raises(SystemExit):
        versions.main(['record', '--folder', str(folder), '--run', 'nope', '--rankers', str(root / 'rankers.json')])
    (root / 'models').mkdir()
    (root / 'models' / 'box-fit.json').write_text('{')
    assert versions.main(['in-use', '--data-root', str(root)]) == 0
    out = capsys.readouterr()
    assert out.out.startswith(versions.CODE_ID + '\t') and 'cannot be read' in out.err


def test_the_measure_runs_a_version_from_rankers_that_never_saw_its_drives_when_it_has_them(root):
    folder = a_version(root)
    assert versions.measure_folder(folder) == (folder, False)
    measure = folder / versions.MEASURE
    measure.mkdir()
    for name in versions.RANKERS:
        (measure / name).write_bytes(b'm')
    # The version has a refiner and its measure folder has no copy: not the same version, not used.
    assert versions.measure_folder(folder) == (folder, False)
    (measure / versions.REFINER).write_bytes(b'r')
    assert versions.measure_folder(folder) == (measure, True)
    (folder / versions.REFINER).unlink()
    assert versions.measure_folder(folder) == (folder, False)
    (measure / versions.REFINER).unlink()
    assert versions.measure_folder(folder) == (measure, True)
    (measure / 'edges.npz').unlink()
    (measure / 'edges.npz').symlink_to(folder / 'edges.npz')
    assert versions.measure_folder(folder) == (folder, False)
    assert versions.measure_folder(versions.HERE) == (versions.HERE, False)


def test_the_routes_left_out_are_the_same_every_time_and_about_a_quarter():
    routes = [f'{n:016x}--{n:010x}' for n in range(400)]
    left_out = [route for route in routes if versions.held_out(route)]
    assert left_out == [route for route in routes if versions.held_out(route)]
    assert 60 < len(left_out) < 140


def test_the_record_says_which_arm_was_measured_clean(root):
    folder = a_version(root)
    data = versions.write_record(folder, RUN, {}, {'clean': {'version': True, 'in_use': False, 'other': True}})
    assert data['measured_clean'] == {'version': True, 'in_use': False}
    assert versions.write_record(folder, RUN, {}, {'clean': 'yes'})['measured_clean'] == {}
