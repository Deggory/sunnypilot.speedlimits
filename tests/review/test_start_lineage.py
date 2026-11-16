"""What a "Start from" descends from, and the validation rows those networks trained on (nnslr_tools.start_lineage,
#104), and what Models says of it (model_roles): made-up runs folders and split files, nothing is trained."""
from __future__ import annotations

import json
from pathlib import Path

from nnslr_tools import start_lineage
from nnslr_tools.model_roles import (ModelIndex, decider_digest, lineage_digest, network_start_digest, signnet_digest,
                                     start_link)
from nnslr_tools.runs import RunsIndex
from nnslr_tools.start_lineage import (account, another_split, choose_on, lineage, recorded, records, short_of,
                                       start_place, trained_ids, words)
from test_panel_models import make_signnet
from test_panel_runs import experiment_result, write

A, B, C, D = 'a' * 64, 'b' * 64, 'c' * 64, 'd' * 64
OLD = '/root/sunnypilot.speedlimits/data/runs'  # where the older runs say their starts are: not on this machine


def cutter_start(run: str, runs: str, split: str, *, stage: bool = False, epoch: int = 3) -> dict:
    """A SignNet's start record (signnet.signnet_start): the weights of <runs>/<run>, or of its Mapillary stage."""
    folder = f'{runs}/{run}/pretrain' if stage else f'{runs}/{run}'
    return {'path': f'{folder}/signnet-best.pt', 'run': run, 'mapillary_stage': stage, 'mapillary_only': stage,
            'epoch': epoch, 'split_sha256': split, 'sha256': 'f' * 64}


def cutters(root: Path) -> dict:
    """Three cutters in a row under <root>/runs: ``first`` from random weights on split A; ``two`` in two stages on
    split B, its Mapillary stage from ``first`` and itself from that stage (both recorded with the repository's old
    path); ``third`` on split C from ``two``. Returns the start of a fourth that goes on from ``third``."""
    runs = root / 'runs'
    write(runs / 'first/result.json', {'task': 'signnet', 'split_sha256': A, 'start': None})
    write(runs / 'two/pretrain/result.json', {'task': 'signnet', 'split_sha256': B, 'mapillary_only': True,
                                              'start': cutter_start('first', OLD, A, epoch=24)})
    write(runs / 'two/result.json', {'task': 'signnet', 'split_sha256': B,
                                     'start': cutter_start('two', OLD, B, stage=True, epoch=2)})
    write(runs / 'third/result.json', {'task': 'signnet', 'split_sha256': C, 'start': cutter_start('two', OLD, B, epoch=7)})
    return cutter_start('third', str(runs), C, epoch=5)


def split_file(root: Path, sha: str, assignments: dict) -> None:
    write(root / 'splits' / f'{sha}.json', {'schema_version': 1, 'assignments': assignments})


def test_the_place_of_a_start_is_its_run_and_its_stage():
    assert start_place(cutter_start('r1', '/d/runs', A)) == ('r1', '')
    assert start_place(cutter_start('r1', OLD, A, stage=True)) == ('r1', 'pretrain')
    variant = {'run': 'e1', 'path': '/d/runs/e1/experiment-b/full/full-best.pt'}
    assert start_place(variant) == ('e1', 'experiment-b/full')
    assert start_lineage.runs_of(variant) == Path('/d/runs')
    assert start_lineage.runs_of(cutter_start('r1', '/d/runs', A, stage=True)) == Path('/d/runs')
    # A folder above the runs with the run's own name: the run is the last one named so.
    above = {'run': 'r1', 'path': '/d/r1/runs/r1/x.pt'}
    assert start_place(above) == ('r1', '') and start_lineage.runs_of(above) == Path('/d/r1/runs')
    above = {'run': 'r1', 'path': '/d/r1/runs/r1/pretrain/x.pt'}
    assert start_place(above) == ('r1', 'pretrain') and start_lineage.runs_of(above) == Path('/d/r1/runs')
    # Nothing but a run's name and folders inside it is followed.
    for start in (None, {}, {'run': 'r1'}, {'path': '/d/runs/r1/x.pt'}, {'run': 'r1', 'path': '/d/runs/other/x.pt'},
                  {'run': '..', 'path': '/d/runs/../x.pt'}, {'run': 'r1', 'path': '/d/runs/r1/../../etc/x.pt'}):
        assert start_place(start) is None and start_lineage.runs_of(start) is None


def test_the_lineage_follows_every_start_by_run_name(tmp_path):
    start = cutters(tmp_path)
    found = lineage(start, records(start_lineage.runs_of(start)))
    assert found['complete'] is True and found['unknown'] is None
    # Nearest first: the run, then the two stages of the two-stage run (found though their recorded path is of the
    # repository's old place), then the first of all.
    assert [(entry['run'], entry['stage'], entry['epoch'], entry['split_sha256'], entry['learnt_ours'])
            for entry in found['ancestors']] == [('third', '', 5, C, True), ('two', '', 7, B, True),
                                                 ('two', 'pretrain', 2, B, False), ('first', '', 24, A, True)]
    assert another_split(found['ancestors'], C) is True and another_split(found['ancestors'][:1], C) is False
    assert another_split(found['ancestors'], None) is False
    # An applicability variant: each run keeps it in experiment-b/<variant>.
    runs = tmp_path / 'runs'
    write(runs / 'e1/experiment-b/full/result.json', {'task': 'experiment_b', 'split_sha256': A, 'start': None})
    write(runs / 'e2/experiment-b/full/result.json', {
        'task': 'experiment_b', 'split_sha256': B,
        'start': {'run': 'e1', 'variant': 'full', 'path': f'{OLD}/e1/experiment-b/full/full-best.pt', 'epoch': 14,
                  'split_sha256': A}})
    variant = {'run': 'e2', 'variant': 'full', 'path': str(runs / 'e2/experiment-b/full/full-best.pt'), 'epoch': 4,
               'split_sha256': B}
    found = lineage(variant, records(start_lineage.runs_of(variant)))
    assert found['complete'] and [(entry['run'], entry['stage'], entry['split_sha256']) for entry in found['ancestors']] == [
        ('e2', 'experiment-b/full', B), ('e1', 'experiment-b/full', A)]


def test_an_ancestor_whose_record_is_gone_is_unknown_never_guessed(tmp_path):
    start = cutters(tmp_path)
    (tmp_path / 'runs/two/pretrain/result.json').unlink()
    found = lineage(start, records(tmp_path / 'runs'))
    # The stage is known from the start that names it; what it went on from is not.
    assert [entry['run'] + '/' + entry['stage'] for entry in found['ancestors']] == ['third/', 'two/', 'two/pretrain']
    assert found['complete'] is False and 'two (pretrain)' in found['unknown'] and 'result.json' in found['unknown']
    # No folder of runs at all, a start that names no run, and records that loop.
    assert lineage(start, records(None))['complete'] is False
    nameless = lineage({'epoch': 3, 'split_sha256': A}, records(tmp_path / 'runs'))
    assert nameless['complete'] is False and nameless['ancestors'][0]['run'] is None
    cutters(tmp_path)
    write(tmp_path / 'runs/first/result.json', {'task': 'signnet', 'split_sha256': A,
                                                'start': cutter_start('third', OLD, C)})
    reads = []

    def bounded(run: str, stage: str):
        """records(), failing after 50 reads: a walk that no longer stops on a loop fails here instead of hanging."""
        reads.append((run, stage))
        assert len(reads) <= 50, 'the walk does not stop on records that loop'
        return records(tmp_path / 'runs')(run, stage)

    looped = lineage(start, bounded)
    assert looped['complete'] is False and 'loop' in looped['unknown'] and len(looped['ancestors']) == 5
    # Each record read once: third, two, its Mapillary stage, first; third comes again and is not read again.
    assert reads == [('third', ''), ('two', ''), ('two', 'pretrain'), ('first', '')]
    # A start record without its split takes the one its run records.
    write(tmp_path / 'runs/first/result.json', {'task': 'signnet', 'split_sha256': A, 'start': None})
    bare = {key: value for key, value in cutter_start('first', OLD, A).items() if key != 'split_sha256'}
    assert lineage(bare, records(tmp_path / 'runs'))['ancestors'][0]['split_sha256'] == A


def test_the_rows_an_ancestor_trained_on_are_the_train_rows_of_its_split(tmp_path):
    start = cutters(tmp_path)
    ancestors = lineage(start, records(tmp_path / 'runs'))['ancestors']
    split_file(tmp_path, A, {'r1': 'train', 'r2': 'validation', 'r3': 'test', 'r9': 'train'})
    split_file(tmp_path, B, {'r1': 'validation', 'r2': 'train', 'r3': 'test', 'r4': 'buffer'})
    split_file(tmp_path, C, {'r5': 'train', 'r1': 'validation'})
    # This run's own split D: every ancestor's train rows, in any of their splits.
    assert trained_ids(ancestors, tmp_path / 'splits', D) == ({'r1', 'r2', 'r5', 'r9'}, None)
    # On split C itself the rows of C are train rows here too: its file is not read.
    (tmp_path / 'splits' / f'{C}.json').unlink()
    assert trained_ids(ancestors, tmp_path / 'splits', C) == ({'r1', 'r2', 'r9'}, None)
    # A stage on the Mapillary windows alone learnt none of our frames: without the stage that learnt them on B,
    # the split B is not read.
    (tmp_path / 'splits' / f'{B}.json').unlink()
    assert trained_ids([entry for entry in ancestors if entry['run'] != 'two' or entry['stage']],
                       tmp_path / 'splits', C) == ({'r1', 'r9'}, None)
    # A split file that is gone, or an ancestor that records none: not known, with the reason.
    ids, why = trained_ids(ancestors, tmp_path / 'splits', C)
    assert ids is None and 'bbbbbbbb' in why and 'two' in why
    ids, why = trained_ids([{**ancestors[0], 'split_sha256': None}], tmp_path / 'splits', D)
    assert ids is None and 'third records no split' in why


def test_the_account_marks_the_clean_validation_rows_and_counts_the_test(tmp_path):
    start = cutters(tmp_path)
    split_file(tmp_path, A, {'r1': 'train', 'r2': 'validation', 'r3': 'test'})
    split_file(tmp_path, B, {'r1': 'validation', 'r2': 'train', 'r3': 'test'})
    split_file(tmp_path, C, {'r1': 'validation', 'r2': 'validation', 'r5': 'train'})
    # A cutter's validation frames by their annotation ids: a frame was trained on when one of its rows was.
    validation = [('r1',), ('r6', 'r7'), ('r7', 'r5'), (), ('r2',)]
    record, clean = account(start, D, tmp_path / 'splits', counted='frames', validation=validation,
                            test=[('r3',), ('r8',)])
    assert clean == [1, 3]
    assert record['validation'] == {'all': 5, 'trained_on': 3, 'clean': 2} and record['test'] == {'all': 2, 'trained_on': 0}
    assert (record['complete'], record['unknown'], record['another_split'], record['counted']) == (True, None, True, 'frames')
    assert (record['best_epoch_on'], record['why'], record['floor']) == ('validation', None, start_lineage.CLEAN_FLOOR)
    assert [entry['run'] for entry in record['ancestors']] == ['third', 'two', 'two', 'first']
    # A test row an ancestor trained on is counted, not assumed away.
    record, _ = account(start, D, tmp_path / 'splits', counted='rows', validation=[('r6',)], test=[('r1',), ('r8',)])
    assert record['test'] == {'all': 2, 'trained_on': 1} and 'this test is not fair' in words(record)
    # No validation row trained on: nothing to leave out, the whole validation is the clean one.
    record, clean = account(start, D, tmp_path / 'splits', counted='rows', validation=[('r6',), ('r7',)], test=[])
    assert clean is None and record['validation'] == {'all': 2, 'trained_on': 0, 'clean': 2} and words(record) is None
    # A lineage that cannot be followed: no count, no clean rows, and the reason.
    (tmp_path / 'runs/two/result.json').unlink()
    record, clean = account(start, D, tmp_path / 'splits', counted='frames', validation=validation, test=[])
    assert clean is None and record['validation'] is None and record['test'] is None and record['complete'] is False
    assert record['best_epoch_on'] == 'validation' and 'the record of two is gone' in record['why']
    assert words(record).startswith('the lineage is not known') and 'whole validation' in words(record)


def test_the_best_epoch_is_chosen_on_the_clean_rows_only_from_the_floor(tmp_path, monkeypatch):
    start = cutters(tmp_path)
    for sha in (A, B, C):
        split_file(tmp_path, sha, {'r1': 'train'})

    def fresh() -> dict:
        return account(start, D, tmp_path / 'splits', counted='frames', validation=[('r1',), ('r2',), ('r3',)],
                       test=[])[0]

    assert start_lineage.CLEAN_FLOOR == 200
    record = fresh()
    assert choose_on(record, 200, 'reviewed boxes') is True
    assert (record['best_epoch_on'], record['why'], record['clean_cases']) == ('clean_validation', None, 200)
    assert words(record) == ('1 of the 3 validation frames were train frames of a network it descends from (4 before '
                             'it): the best epoch is chosen on the other 2, and both are measured')
    # One case fewer than the floor: the whole validation chooses, as before, and the record says why.
    record = fresh()
    assert choose_on(record, 199, 'reviewed boxes') is False
    assert record['best_epoch_on'] == 'validation' and record['clean_cases'] == 199
    assert record['why'] == 'the 2 clean frames hold 199 reviewed boxes, fewer than the 200 a choice needs'
    assert words(record).endswith('fewer than the 200 a choice needs, so the best epoch is chosen on the whole '
                                  'validation; both are measured')
    # Something the measure counts apart is too rare on the clean rows (short_of), however many rows they are.
    record = fresh()
    assert choose_on(record, 5000, 'rows', short_of(2, 58, 'signs of another road')) is False
    assert record['why'] == 'only 2 of the 58 signs of another road are on the clean rows'
    assert (record['best_epoch_on'], record['clean_cases']) == ('validation', 5000)
    assert words(record) == ('1 of the 3 validation frames were train frames of a network it descends from (4 before '
                             'it): only 2 of the 58 signs of another road are on the clean rows, so the best epoch is '
                             'chosen on the whole validation; both are measured')
    # No row trained on, or a lineage not known: nothing to choose, the record stays as account wrote it.
    untouched = account(start, D, tmp_path / 'splits', counted='frames', validation=[('r2',)], test=[])[0]
    assert choose_on(untouched, 5000, 'reviewed boxes') is False and untouched['clean_cases'] is None
    unknown = account(start, D, tmp_path / 'nowhere', counted='frames', validation=[('r1',)], test=[])[0]
    assert unknown['complete'] is False and choose_on(unknown, 5000, 'reviewed boxes') is False
    assert unknown['best_epoch_on'] == 'validation' and 'is gone' in unknown['why']
    monkeypatch.setattr(start_lineage, 'CLEAN_FLOOR', 2)
    record = fresh()
    assert choose_on(record, 2, 'reviewed boxes') is True and record['best_epoch_on'] == 'clean_validation'


def test_a_kind_the_measure_counts_apart_needs_half_of_its_cases_on_the_clean_rows():
    """The applicability network's score is half the F1 of the signs of another road: the clean rows choose only
    when they hold at least half of the validation's (start_lineage.CLEAN_SHARE)."""
    assert start_lineage.CLEAN_SHARE == 0.5
    # The network in use on 16 Nov 2026: 40 of its 58 signs of another road are on the clean rows.
    assert short_of(40, 58, 'signs of another road') is None
    assert short_of(29, 58, 'signs of another road') is None  # half: enough
    assert short_of(28, 58, 'signs of another road') == 'only 28 of the 58 signs of another road are on the clean rows'
    assert short_of(0, 58, 'signs of another road') == 'only 0 of the 58 signs of another road are on the clean rows'
    assert short_of(1, 3, 'signs of another road') == 'only 1 of the 3 signs of another road are on the clean rows'
    assert short_of(2, 3, 'signs of another road') is None and short_of(1, 1, 'signs of another road') is None
    # A validation with none of them: nothing to hold a share of, the floor alone decides.
    assert short_of(0, 0, 'signs of another road') is None


def test_the_page_and_the_docs_name_the_floor_and_the_share_the_code_has():
    """Train's Start from note and the documents say the numbers in words: they change with the constants."""
    root = Path(__file__).resolve().parents[2]
    floor = start_lineage.CLEAN_FLOOR
    assert start_lineage.CLEAN_SHARE == 0.5  # "at least half", in each of them
    note = (root / 'web/src/control/network-start.ts').read_text(encoding='utf-8')
    assert f"'they hold {floor} reviewed boxes or more (in a stage on Mapillary alone, boxes of speed signs)'" in note
    assert f"'they are {floor} or more and hold at least half of the validation’s signs of another road'" in note
    commands = (root / 'docs/COMMANDS.md').read_text(encoding='utf-8')
    assert f'when they hold {floor} reviewed boxes or more (`start_lineage.CLEAN_FLOOR`' in commands
    assert (f'when they are {floor} or more (`start_lineage.CLEAN_FLOOR`) and hold at least half of the validation\'s '
            'signs of another road (`start_lineage.CLEAN_SHARE`') in commands
    guide = (root / 'docs/GUIDE.md').read_text(encoding='utf-8')
    assert (f'for a cutter {floor} reviewed boxes or more (in its stage on Mapillary alone, boxes of speed signs), for '
            f'applicability {floor} rows or more that hold at least half of the validation\'s signs of another road') in guide
    network = (root / 'docs/NEURAL_NETWORK.md').read_text(encoding='utf-8')
    assert f'The clean frames choose only when they hold {floor} reviewed boxes or more (`CLEAN_FLOOR`' in network
    assert "hold at least half of the validation's signs of another road (`CLEAN_SHARE`)" in network


def test_a_start_that_records_its_lineage_keeps_it_when_an_ancestor_is_removed(tmp_path):
    """A network trained since #104 writes the whole chain into its result.json (start_lineage): what goes on from
    it takes the chain from there, so a run removed afterwards does not make every later lineage unknown."""
    start = cutters(tmp_path)
    for sha in (A, B, C):
        split_file(tmp_path, sha, {'r1': 'train'})
    runs = tmp_path / 'runs'
    # ``fourth`` is trained on split D from ``third``, with this code: its result.json records its lineage.
    kept, _ = account(start, D, tmp_path / 'splits', counted='frames', validation=[('r1',), ('r2',)], test=[])
    assert kept['complete'] is True and len(kept['ancestors']) == 4
    assert recorded({'start': start, 'start_lineage': kept}) == kept['ancestors']
    write(runs / 'fourth/result.json', {'task': 'signnet', 'split_sha256': D, 'start': start, 'start_lineage': kept})
    fifth = cutter_start('fourth', str(runs), D, epoch=9)
    whole = lineage(fifth, records(runs))
    assert whole['complete'] is True and [(entry['run'], entry['stage']) for entry in whole['ancestors']] == [
        ('fourth', ''), ('third', ''), ('two', ''), ('two', 'pretrain'), ('first', '')]
    # The records behind ``fourth`` are not read at all: its own is the only one.
    reads = []
    assert lineage(fifth, lambda run, stage: reads.append((run, stage)) or records(runs)(run, stage)) == whole
    assert reads == [('fourth', '')]
    # An ancestor's folder removed after ``fourth`` was trained: the lineage stays complete, its split files (kept
    # under their sha256) still say which rows it trained on.
    import shutil
    shutil.rmtree(runs / 'two')
    assert lineage(fifth, records(runs)) == whole
    split_file(tmp_path, D, {'r2': 'train'})
    record, clean = account(fifth, 'e' * 64, tmp_path / 'splits', counted='frames',
                            validation=[('r1',), ('r2',), ('r3',)], test=[])
    assert (record['complete'], record['unknown'], clean) == (True, None, [2])
    assert record['validation'] == {'all': 3, 'trained_on': 2, 'clean': 1} and len(record['ancestors']) == 5
    # Without the record ``fourth`` kept, the same removal leaves the lineage unknown, as for a run of before #104.
    write(runs / 'fourth/result.json', {'task': 'signnet', 'split_sha256': D, 'start': start})
    walked = lineage(fifth, records(runs))
    assert walked['complete'] is False and 'the record of two is gone' in walked['unknown']
    # A recorded lineage that was not complete, or is not shaped as account writes it, is not taken: the walk goes on.
    for broken in ({**kept, 'complete': False}, {**kept, 'ancestors': []}, {**kept, 'ancestors': 'third'},
                   {**kept, 'ancestors': [{'run': 'third'}]}, {**kept, 'ancestors': kept['ancestors'][1:]},
                   {**kept, 'ancestors': [{**kept['ancestors'][0], 'run': '../third'}, *kept['ancestors'][1:]]},
                   {'complete': True}, 'complete'):
        assert recorded({'start': start, 'start_lineage': broken}) is None
        write(runs / 'fourth/result.json', {'task': 'signnet', 'split_sha256': D, 'start': start, 'start_lineage': broken})
        assert lineage(fifth, records(runs)) == walked
    assert recorded(None) is None and recorded({'start': start}) is None
    # A recorded lineage that names a network already walked: not taken either, the walk finds the loop itself.
    cutters(tmp_path)
    write(runs / 'first/result.json', {'task': 'signnet', 'split_sha256': A, 'start': cutter_start('third', OLD, C),
                                       'start_lineage': {'complete': True, 'ancestors': [
                                           {'run': 'third', 'stage': '', 'epoch': 3, 'split_sha256': C, 'learnt_ours': True}]}})
    looped = lineage(start, records(runs))
    assert looped['complete'] is False and 'loop' in looped['unknown']


# -- what Models says -------------------------------------------------------------------------------------------
def test_the_digest_flags_a_start_whose_ancestor_learnt_on_another_split():
    start = {'run': 's2', 'epoch': 1, 'split_sha256': D, 'path': '/d/runs/s2/pretrain/signnet-best.pt'}
    # One start back it is this run's own split (the first stage of a two-stage cutter): nothing to say, as before.
    assert network_start_digest(start, D) == {'run': 's2', 'epoch': 1, 'another_split': False}
    ancestors = [{'run': 's2', 'stage': 'pretrain', 'split_sha256': D, 'learnt_ours': False},
                 {'run': 's1', 'stage': '', 'split_sha256': A, 'learnt_ours': True}]
    assert network_start_digest(start, D, {'ancestors': ancestors}) == {'run': 's2', 'epoch': 1, 'another_split': True}
    # A stage on the Mapillary windows alone learnt none of our frames, on whatever split.
    windows = [{'run': 's2', 'stage': 'pretrain', 'split_sha256': A, 'learnt_ours': False}]
    assert network_start_digest(start, D, {'ancestors': windows})['another_split'] is False
    assert network_start_digest({**start, 'split_sha256': A}, D, {'ancestors': []})['another_split'] is True
    assert lineage_digest(None) is None and lineage_digest({'complete': True}) is None
    record = {'ancestors': ancestors, 'complete': True, 'unknown': None, 'another_split': True, 'counted': 'frames',
              'validation': {'all': 9, 'trained_on': 4, 'clean': 5}, 'test': {'all': 7, 'trained_on': 0},
              'floor': 200, 'clean_cases': 6, 'best_epoch_on': 'validation', 'why': '6 reviewed boxes ...'}
    assert lineage_digest(record) == {
        'ancestors': 2, 'complete': True, 'counted': 'frames', 'validation': {'all': 9, 'trained_on': 4, 'clean': 5},
        'test': {'all': 7, 'trained_on': 0}, 'best_epoch_on': 'validation', 'why': '6 reviewed boxes ...'}
    # A lineage that could not be followed stays not complete, with nothing counted: Models says so.
    gone = lineage_digest({**record, 'complete': False, 'unknown': 'the record of s1 is gone', 'validation': None,
                           'test': None})
    assert gone['complete'] is False and gone['validation'] is None and gone['test'] is None
    assert start_link({'split_sha256': D, 'start': {**start, 'sha256': 'f' * 64}}) == {
        'split_sha256': D, 'start': {'run': 's2', 'path': start['path'], 'epoch': 1, 'split_sha256': D,
                                     'mapillary_only': None}}


def test_models_follows_the_lineage_of_a_run_trained_before_it_was_recorded(tmp_path):
    """#104: the second stage of a two-stage cutter starts from its own first stage, of its own split; the cutter
    that stage went on from learnt on another split, and Models never said so."""
    root = tmp_path
    make_signnet(root, 's1')
    run = make_signnet(root, 's2')
    result = json.loads((run / 'result.json').read_text())
    write(root / 'runs/s1/result.json', {**result, 'split_sha256': A})
    write(run / 'pretrain/result.json', {**result, 'split_sha256': D, 'mapillary_only': True,
                                         'start': cutter_start('s1', OLD, A, epoch=9)})
    write(run / 'result.json', {**result, 'split_sha256': D, 'start': cutter_start('s2', OLD, D, stage=True, epoch=1)})
    index = ModelIndex(RunsIndex(root))

    def versions() -> dict:
        return {entry['id']: entry for role in ('cutter', 'decider') for entry in index.artifacts((role,))[role]}

    two = versions()['cutter:s2/signnet']
    assert two['start'] == {'run': 's2', 'epoch': 1, 'another_split': True}
    assert two['lineage'] == {'ancestors': 2, 'complete': True, 'counted': None, 'validation': None, 'test': None,
                              'best_epoch_on': 'validation', 'why': None}
    assert two['validation_clean'] is None
    first = versions()['cutter:s1/signnet']
    assert (first['start'], first['lineage']) == (None, None)
    # The record of the first stage gone: what is known stays (its own split), and the chain is not called whole.
    (run / 'pretrain/result.json').unlink()
    two = versions()['cutter:s2/signnet']
    assert two['start']['another_split'] is False and two['lineage']['complete'] is False
    # An applicability variant, two runs back.
    for name, split, start in (('e1', A, None), ('e2', D, {'run': 'e1', 'variant': 'crop', 'epoch': 3, 'split_sha256': A,
                                                           'path': f'{OLD}/e1/experiment-b/crop/crop-best.pt'}),
                               ('e3', D, {'run': 'e2', 'variant': 'crop', 'epoch': 5, 'split_sha256': D,
                                          'path': f'{OLD}/e2/experiment-b/crop/crop-best.pt'})):
        write(root / 'runs' / name / 'experiment-b/crop/result.json',
              {**experiment_result('crop', 0.9, 0.6), 'split_sha256': split, 'start': start})
        write(root / 'runs' / name / 'experiment-b/crop/crop-best.pt', 'weights')
    found = versions()
    assert found['decider:e3/experiment-b/crop']['start'] == {'run': 'e2', 'epoch': 5, 'another_split': True}
    assert found['decider:e3/experiment-b/crop']['lineage']['ancestors'] == 2
    assert found['decider:e2/experiment-b/crop']['start']['another_split'] is True
    assert found['decider:e1/experiment-b/crop']['lineage'] is None


def test_models_reads_the_lineage_and_the_clean_validation_a_training_recorded(tmp_path):
    root = tmp_path
    run = make_signnet(root, 's1')
    result = json.loads((run / 'result.json').read_text())
    record = {'ancestors': [{'run': 's0', 'stage': '', 'epoch': 4, 'split_sha256': A, 'learnt_ours': True}],
              'complete': True, 'unknown': None, 'another_split': True, 'counted': 'frames',
              'validation': {'all': 40, 'trained_on': 15, 'clean': 25}, 'test': {'all': 30, 'trained_on': 0},
              'floor': 200, 'clean_cases': 420, 'best_epoch_on': 'clean_validation', 'why': None}
    clean = {key: value for key, value in result['test'].items()}
    write(run / 'result.json', {**result, 'split_sha256': D, 'validation': result['test'], 'validation_clean': clean,
                                'start': cutter_start('s0', OLD, D, epoch=4), 'start_lineage': record})
    digest = signnet_digest(json.loads((run / 'result.json').read_text()))
    # The direct start is of this split: only the recorded lineage says an ancestor was not.
    assert digest['start'] == {'run': 's0', 'epoch': 4, 'another_split': True}
    assert digest['lineage']['validation'] == {'all': 40, 'trained_on': 15, 'clean': 25}
    assert digest['lineage']['best_epoch_on'] == 'clean_validation'
    assert digest['validation_clean']['budget'] == digest['validation']['budget'] and digest['validation_clean']['car']
    entry = {item['id']: item for item in ModelIndex(RunsIndex(root)).artifacts(('cutter',))['cutter']}['cutter:s1/signnet']
    # Recorded: not followed again (s0 is not even on disk).
    assert entry['lineage'] == digest['lineage'] and entry['start'] == digest['start']
    assert entry['validation_clean'] == digest['validation_clean']
    # A result.json of before: the same digest as ever, with nothing of the clean validation.
    old = signnet_digest(result)
    assert (old['start'], old['lineage'], old['validation_clean']) == (None, None, None)
    # An applicability variant.
    variant = experiment_result('crop', 0.9, 0.6)
    report = {**variant['metrics']['validation'], 'box_count': 25}
    digest = decider_digest({**variant, 'split_sha256': D, 'validation_clean': report,
                             'start': {'run': 'e1', 'epoch': 3, 'split_sha256': D}, 'start_lineage': {**record, 'counted': 'rows'}})
    assert digest['partitions']['validation']['boxes'] == 40 and digest['partitions']['validation_clean']['boxes'] == 25
    assert digest['lineage']['counted'] == 'rows' and digest['start']['another_split'] is True
    old = decider_digest(variant)
    assert old['partitions']['validation_clean'] is None and old['lineage'] is None and old['start'] is None
    # Its entry in Models (the Clean column of applicability's table reads it), and one of before beside it.
    recorded_start = {'run': 'e1', 'variant': 'crop', 'epoch': 3, 'split_sha256': D,
                      'path': f'{OLD}/e1/experiment-b/crop/crop-best.pt'}
    for name, more in (('e1', {}), ('e2', {'validation_clean': report, 'start': recorded_start,
                                           'start_lineage': {**record, 'counted': 'rows'}})):
        write(root / 'runs' / name / 'experiment-b/crop/result.json', {**variant, 'split_sha256': D, **more})
        write(root / 'runs' / name / 'experiment-b/crop/crop-best.pt', 'weights')
    entries = {item['id']: item for item in ModelIndex(RunsIndex(root)).artifacts(('decider',))['decider']}
    entry = entries['decider:e2/experiment-b/crop']
    assert entry['validation_clean'] == digest['partitions']['validation_clean'] and entry['validation_clean']['boxes'] == 25
    assert entry['validation']['boxes'] == 40 and entry['lineage'] == digest['lineage']
    assert entry['start'] == {'run': 'e1', 'epoch': 3, 'another_split': True}
    before = entries['decider:e1/experiment-b/crop']
    assert (before['validation_clean'], before['lineage'], before['start']) == (None, None, None)
