"""import-annotations --from-db: same canonical dataset as the JSONL export, imported once."""
from __future__ import annotations

import hashlib
import json

from nnslr_tools import cli
from nnslr_tools.annotations import jsonl_bytes
from nnslr_tools.review_db import ReviewStore, db_path, save_frame_review
from review_helpers import OTHER_ROUTE, ROUTE, RUN_ID, make_report, page_body, page_export_row


def invoke(capsys, *args):
    try:
        code = cli.main(list(args))
    except SystemExit as exc:
        code = exc.code
    output = capsys.readouterr()
    return code, output


def save(root, store, row, body, route=ROUTE):
    return save_frame_review(store, root, route_id=route, run_id=RUN_ID, report_row=row, body=body, client='test')


def canonical(path):
    rows = []
    for line in path.read_text().splitlines():
        rows.append(json.loads(line))
    return rows


def test_from_db_gives_the_same_dataset_as_the_jsonl_export(tmp_path, capsys):
    rows = make_report(tmp_path)
    store = ReviewStore(db_path(tmp_path))
    bodies = [
        page_body(rows[0], value_kph=50, manual_box=[3, 3, 12, 14]),
        page_body(rows[1]),
        page_body(rows[2], drop_proposals=True),
    ]
    exported = []
    for row, body in zip(rows, bodies):
        review = save(tmp_path, store, row, body)
        expected = page_export_row(row, body, review['payload']['review_timestamp'])
        assert review['payload'] == expected
        exported.append(expected)
    source = tmp_path / 'annotations-export.jsonl'
    source.write_bytes(jsonl_bytes(exported))
    from_file = tmp_path / 'from-file.jsonl'
    from_db = tmp_path / 'from-db.jsonl'
    code, out = invoke(capsys, 'import-annotations', str(source), '--data-root', str(tmp_path), '--output', str(from_file))
    assert code == 0, out.out + out.err
    code, out = invoke(capsys, 'import-annotations', '--from-db', '--data-root', str(tmp_path), '--output', str(from_db))
    assert code == 0, out.out + out.err
    report = json.loads(out.out)
    assert report['valid'] and report['source'] == 'review-db' and report['imported_frame_count'] == 3
    assert from_db.read_bytes() == from_file.read_bytes()
    records = canonical(from_db)
    states = []
    for record in records:
        states.append((record['output_index'], record['sign_family'], record['review_state']))
    assert sorted(states) == [(0, 'maximum_speed', 'corrected'), (0, 'unreadable', 'corrected'),
                              (1, 'not_a_sign', 'accepted'), (2, 'maximum_speed', 'rejected'),
                              (2, 'not_a_sign', 'accepted')]
    archives = list((tmp_path / 'annotations/imports').glob('*.jsonl'))
    assert len(archives) == 1 and archives[0].read_bytes() == source.read_bytes()


def replaced_archives(root):
    return sorted((root / 'annotations/replaced').glob('*.jsonl'))


def test_from_db_default_output_is_idempotent_and_replaces_late_changes(tmp_path, capsys):
    rows = make_report(tmp_path)
    store = ReviewStore(db_path(tmp_path))
    save(tmp_path, store, rows[0], page_body(rows[0]))
    code, out = invoke(capsys, 'import-annotations', '--from-db', '--data-root', str(tmp_path))
    assert code == 0, out.out + out.err
    dataset = tmp_path / 'annotations/objects.jsonl'
    first = dataset.read_bytes()
    assert json.loads(out.out)['imported_count'] == 1
    code, out = invoke(capsys, 'import-annotations', '--from-db', '--data-root', str(tmp_path))
    assert code == 0, out.out + out.err
    report = json.loads(out.out)
    assert report['imported_count'] == 0 and report['already_imported_frame_count'] == 1
    assert report['replaced_frame_count'] == 0 and report['removed_withdrawn_frame_count'] == 0
    assert dataset.read_bytes() == first and not replaced_archives(tmp_path)
    save(tmp_path, store, rows[0], page_body(rows[0], value_kph=80))
    save(tmp_path, store, rows[1], page_body(rows[1]))
    code, out = invoke(capsys, 'import-annotations', '--from-db', '--data-root', str(tmp_path))
    assert code == 0, out.out + out.err
    report = json.loads(out.out)
    # The new frame is added and the frame reviewed again replaces its old rows.
    assert report['imported_frame_count'] == 1 and report['replaced_frame_count'] == 1
    assert [c['frame_key'] for c in report['changed_after_import']] == ['0/0']
    assert report['removed_count'] == 1 and not any('append-only' in w for w in report['warnings'])
    values = []
    for record in canonical(dataset):
        values.append((record['output_index'], record['value_kph']))
    assert sorted(values, key=str) == sorted([(0, 80), (1, None)], key=str)
    # The replaced rows are archived exactly as they were, under their content hash.
    [archive] = replaced_archives(tmp_path)
    assert archive.read_bytes() == first and str(archive) == report['replaced_archive']
    assert archive.name == hashlib.sha256(first).hexdigest() + '.jsonl'
    code, out = invoke(capsys, 'import-annotations', '--from-db', '--data-root', str(tmp_path))
    report = json.loads(out.out)
    assert code == 0 and report['already_imported_frame_count'] == 2 and report['replaced_frame_count'] == 0


def frame_records(path, output_index):
    records = []
    for record in canonical(path):
        if record['output_index'] == output_index:
            records.append(record)
    return records


def import_change_and_import_again(tmp_path, capsys, rows, index, first_body, second_body):
    """Import frame <index>, save it again changed plus the new frame 2, import again."""
    store = ReviewStore(db_path(tmp_path))
    save(tmp_path, store, rows[index], first_body)
    code, out = invoke(capsys, 'import-annotations', '--from-db', '--data-root', str(tmp_path))
    assert code == 0, out.out + out.err
    dataset = tmp_path / 'annotations/objects.jsonl'
    before = frame_records(dataset, index)
    save(tmp_path, store, rows[index], second_body)
    save(tmp_path, store, rows[2], page_body(rows[2]))
    code, out = invoke(capsys, 'import-annotations', '--from-db', '--data-root', str(tmp_path))
    assert code == 0, out.out + out.err
    report = json.loads(out.out)
    # The new frame is imported, the changed one replaces all its old rows.
    assert report['imported_frame_count'] == 1 and report['replaced_frame_count'] == 1
    assert [c['frame_key'] for c in report['changed_after_import']] == [f'0/{index}']
    after = frame_records(dataset, index)
    # None of the old rows is left: the rows of the frame all come from the new review.
    assert not [record for record in after if record in before]
    assert len({record['provenance']['source_review_sha256'] for record in after}) == 1
    assert len(frame_records(dataset, 2)) == 1
    [archive] = replaced_archives(tmp_path)
    assert canonical(archive) == before
    code, out = invoke(capsys, 'validate-dataset', '--data-root', str(tmp_path))
    assert code == 0, out.out + out.err
    return before, after


def test_empty_frame_imported_then_box_added_replaces_the_negative(tmp_path, capsys):
    rows = make_report(tmp_path)
    first = page_body(rows[1])
    second = page_body(rows[1], manual_box=[3, 3, 12, 14])
    before, after = import_change_and_import_again(tmp_path, capsys, rows, 1, first, second)
    assert len(before) == 1 and before[0]['sign_family'] == 'not_a_sign' and before[0]['bbox_xyxy'] is None
    # The whole-frame negative is gone: it never coexists with the new box.
    assert [(record['sign_family'], record['bbox_xyxy']) for record in after] == [('unreadable', [3, 3, 12, 14])]


def test_manual_box_drawn_again_replaces_the_old_box(tmp_path, capsys):
    rows = make_report(tmp_path)
    first = page_body(rows[1], manual_box=[3, 3, 12, 14])
    second = page_body(rows[1], manual_box=[4, 4, 13, 15])
    second['detections'][0]['detection_id'] = 'manual-2'
    before, after = import_change_and_import_again(tmp_path, capsys, rows, 1, first, second)
    assert len(before) == 1 and before[0]['bbox_xyxy'] == [3, 3, 12, 14]
    assert [record['bbox_xyxy'] for record in after] == [[4, 4, 13, 15]]


def test_manual_box_deleted_leaves_the_dataset(tmp_path, capsys):
    rows = make_report(tmp_path)
    first = page_body(rows[0], manual_box=[3, 3, 12, 14])
    second = page_body(rows[0])
    before, after = import_change_and_import_again(tmp_path, capsys, rows, 0, first, second)
    boxes = []
    for record in before:
        boxes.append(record['bbox_xyxy'])
    assert sorted(boxes) == [[1, 2, 15, 20], [3, 3, 12, 14]]
    assert [record['bbox_xyxy'] for record in after] == [[1, 2, 15, 20]]


def test_frame_withdrawn_after_its_import_leaves_the_dataset(tmp_path, capsys):
    rows = make_report(tmp_path)
    store = ReviewStore(db_path(tmp_path))
    save(tmp_path, store, rows[0], page_body(rows[0]))
    save(tmp_path, store, rows[1], page_body(rows[1]))
    code, out = invoke(capsys, 'import-annotations', '--from-db', '--data-root', str(tmp_path))
    assert code == 0, out.out + out.err
    dataset = tmp_path / 'annotations/objects.jsonl'
    withdrawn_rows = frame_records(dataset, 1)
    store.withdraw_review(route_id=ROUTE, run_id=RUN_ID, frame_key='0/1', reviewer='Cristian', client='test')
    # Only a removal, no new frame: the dataset is still written.
    code, out = invoke(capsys, 'import-annotations', '--from-db', '--data-root', str(tmp_path))
    assert code == 0, out.out + out.err
    report = json.loads(out.out)
    assert report['imported_frame_count'] == 0 and report['already_imported_frame_count'] == 1
    assert report['removed_withdrawn_frame_count'] == 1 and report['replaced_frame_count'] == 0
    assert [c['frame_key'] for c in report['removed_withdrawn']] == ['0/1']
    assert sorted({record['output_index'] for record in canonical(dataset)}) == [0]
    [archive] = replaced_archives(tmp_path)
    assert canonical(archive) == withdrawn_rows
    # Nothing left to do: the next import changes nothing.
    before = dataset.read_bytes()
    code, out = invoke(capsys, 'import-annotations', '--from-db', '--data-root', str(tmp_path))
    report = json.loads(out.out)
    assert code == 0 and report['removed_withdrawn_frame_count'] == 0 and dataset.read_bytes() == before


def test_withdrawn_frame_imported_from_an_older_run_stays(tmp_path, capsys):
    old_run = '20261020T100000Z-00000000'
    rows = make_report(tmp_path, run_id=old_run)
    store = ReviewStore(db_path(tmp_path))
    save_frame_review(store, tmp_path, route_id=ROUTE, run_id=old_run, report_row=rows[1],
                      body=page_body(rows[1], run_id=old_run), client='test')
    code, out = invoke(capsys, 'import-annotations', '--from-db', '--data-root', str(tmp_path))
    assert code == 0, out.out + out.err
    dataset = tmp_path / 'annotations/objects.jsonl'
    before = dataset.read_bytes()
    # A new pre-annotation run of the route, whose review of the same frame is withdrawn.
    import shutil
    shutil.rmtree(tmp_path / 'derived/preannotations' / ROUTE)
    rows = make_report(tmp_path)
    save(tmp_path, store, rows[1], page_body(rows[1]))
    store.withdraw_review(route_id=ROUTE, run_id=RUN_ID, frame_key='0/1', reviewer='Cristian', client='test')
    save(tmp_path, store, rows[0], page_body(rows[0]))
    code, out = invoke(capsys, 'import-annotations', '--from-db', '--data-root', str(tmp_path))
    assert code == 0, out.out + out.err
    report = json.loads(out.out)
    assert report['imported_frame_count'] == 1 and report['removed_withdrawn_frame_count'] == 0
    assert set(before.decode().splitlines()) <= set(dataset.read_text().splitlines())
    assert not replaced_archives(tmp_path)


def test_propose_again_carries_the_confirmed_frames_onto_the_new_run(tmp_path, capsys):
    old_run = '20261020T100000Z-00000000'
    rows = make_report(tmp_path, run_id=old_run)
    store = ReviewStore(db_path(tmp_path))
    for row in rows[:2]:
        save_frame_review(store, tmp_path, route_id=ROUTE, run_id=old_run, report_row=row,
                          body=page_body(row, run_id=old_run), client='test')
    store.withdraw_review(route_id=ROUTE, run_id=old_run, frame_key='0/1', reviewer='Cristian', client='test')
    save_frame_review(store, tmp_path, route_id=ROUTE, run_id=old_run, report_row=rows[1],
                      body=page_body(rows[1], run_id=old_run), client='test')
    code, out = invoke(capsys, 'import-annotations', '--from-db', '--data-root', str(tmp_path))
    assert code == 0, out.out + out.err
    dataset = tmp_path / 'annotations/objects.jsonl'
    before = dataset.read_bytes()
    # Propose again: a new report of the same frames; the confirmed frames are copied onto it, once.
    make_report(tmp_path)
    keys = [row['frame_key'] for row in rows]
    assert store.carry_confirmed(route_id=ROUTE, from_run=old_run, to_run=RUN_ID, frame_keys=keys, client='t') == 2
    assert store.carry_confirmed(route_id=ROUTE, from_run=old_run, to_run=RUN_ID, frame_keys=keys, client='t') == 0
    old = {review['frame_key']: review for review in store.reviews(ROUTE, old_run)}
    new = {review['frame_key']: review for review in store.reviews(ROUTE, RUN_ID)}
    assert sorted(new) == ['0/0', '0/1'] and all(review['confirmed'] for review in new.values())
    for key, review in new.items():
        assert review['payload'] == old[key]['payload'] and review['reviewer'] == old[key]['reviewer']
        assert review['updated_at'] == old[key]['updated_at'] and review['revision'] == 1
        assert [line['client'] for line in store.history(ROUTE, RUN_ID, key)] == ['t']
    assert store.confirmed_count_in_other_runs(ROUTE, RUN_ID) == 0
    # Already in the dataset: nothing new, no warning about older runs.
    code, out = invoke(capsys, 'import-annotations', '--from-db', '--data-root', str(tmp_path))
    report = json.loads(out.out)
    assert code == 0 and report['imported_frame_count'] == 0 and report['already_imported_frame_count'] == 2
    assert not [w for w in report['warnings'] if 'older pre-annotation runs' in w] and dataset.read_bytes() == before
    # A carried frame withdrawn on the new run leaves the dataset.
    store.withdraw_review(route_id=ROUTE, run_id=RUN_ID, frame_key='0/1', reviewer='Cristian', client='test')
    code, out = invoke(capsys, 'import-annotations', '--from-db', '--data-root', str(tmp_path))
    report = json.loads(out.out)
    assert code == 0 and report['removed_withdrawn_frame_count'] == 1
    assert sorted({record['output_index'] for record in canonical(dataset)}) == [0]


def test_from_db_dry_run_reports_and_writes_nothing(tmp_path, capsys):
    rows = make_report(tmp_path)
    store = ReviewStore(db_path(tmp_path))
    save(tmp_path, store, rows[0], page_body(rows[0]))
    save(tmp_path, store, rows[1], page_body(rows[1]))
    code, out = invoke(capsys, 'import-annotations', '--from-db', '--dry-run', '--data-root', str(tmp_path))
    assert code == 0, out.out + out.err
    report = json.loads(out.out)
    assert report['dry_run'] and report['valid'] and report['imported_frame_count'] == 2
    assert not (tmp_path / 'annotations/objects.jsonl').exists()
    assert not (tmp_path / 'annotations/imports').exists() and not (tmp_path / 'annotations/replaced').exists()
    code, out = invoke(capsys, 'import-annotations', '--from-db', '--data-root', str(tmp_path))
    assert code == 0 and not json.loads(out.out)['dry_run']
    dataset = tmp_path / 'annotations/objects.jsonl'
    before = dataset.read_bytes()
    imports = sorted((tmp_path / 'annotations/imports').iterdir())
    save(tmp_path, store, rows[0], page_body(rows[0], value_kph=80))
    store.withdraw_review(route_id=ROUTE, run_id=RUN_ID, frame_key='0/1', reviewer='Cristian', client='test')
    save(tmp_path, store, rows[2], page_body(rows[2]))
    code, out = invoke(capsys, 'import-annotations', '--from-db', '--dry-run', '--data-root', str(tmp_path))
    assert code == 0, out.out + out.err
    report = json.loads(out.out)
    assert (report['imported_frame_count'], report['replaced_frame_count'],
            report['removed_withdrawn_frame_count']) == (1, 1, 1)
    assert report['replaced_archive'].endswith('.jsonl')
    assert dataset.read_bytes() == before and sorted((tmp_path / 'annotations/imports').iterdir()) == imports
    assert not (tmp_path / 'annotations/replaced').exists()
    code, out = invoke(capsys, 'import-annotations', 'reviews.jsonl', '--dry-run', '--data-root', str(tmp_path))
    assert code != 0 and 'invalid_arguments' in out.out


def test_a_replacement_that_fails_validation_writes_nothing(tmp_path, capsys):
    rows = make_report(tmp_path)
    store = ReviewStore(db_path(tmp_path))
    for row in rows:
        save(tmp_path, store, row, page_body(row))
    code, out = invoke(capsys, 'import-annotations', '--from-db', '--data-root', str(tmp_path))
    assert code == 0, out.out + out.err
    dataset = tmp_path / 'annotations/objects.jsonl'
    before = dataset.read_bytes()
    imports = sorted((tmp_path / 'annotations/imports').iterdir())
    save(tmp_path, store, rows[0], page_body(rows[0], value_kph=80))
    store.withdraw_review(route_id=ROUTE, run_id=RUN_ID, frame_key='0/1', reviewer='Cristian', client='test')
    # The image of frame 0/2, which stays in the dataset as it is, changed: the whole new dataset is
    # validated, so the replacement and the removal are not written either.
    (tmp_path / rows[2]['image_path']).write_bytes(b'changed after review')
    code, out = invoke(capsys, 'import-annotations', '--from-db', '--data-root', str(tmp_path))
    assert code != 0 and 'hash_mismatch' in out.out
    assert dataset.read_bytes() == before and not replaced_archives(tmp_path)
    assert sorted((tmp_path / 'annotations/imports').iterdir()) == imports


def test_from_db_skips_withdrawn_and_filters_by_route(tmp_path, capsys):
    rows = make_report(tmp_path)
    other_rows = make_report(tmp_path, route=OTHER_ROUTE, count=2)
    store = ReviewStore(db_path(tmp_path))
    save(tmp_path, store, rows[0], page_body(rows[0]))
    save(tmp_path, store, rows[1], page_body(rows[1]))
    store.withdraw_review(route_id=ROUTE, run_id=RUN_ID, frame_key='0/1', reviewer='Cristian', client='test')
    save(tmp_path, store, other_rows[0], page_body(other_rows[0]), route=OTHER_ROUTE)
    output = tmp_path / 'only-route.jsonl'
    code, out = invoke(capsys, 'import-annotations', '--from-db', '--route', ROUTE, '--data-root', str(tmp_path),
                       '--output', str(output))
    assert code == 0, out.out + out.err
    routes = set()
    frames = set()
    for record in canonical(output):
        routes.add(record['route_id'])
        frames.add(record['output_index'])
    assert routes == {ROUTE} and frames == {0}


def test_from_db_errors_leave_no_dataset(tmp_path, capsys):
    make_report(tmp_path)
    ReviewStore(db_path(tmp_path))
    code, out = invoke(capsys, 'import-annotations', '--from-db', '--data-root', str(tmp_path))
    assert code != 0 and 'empty_dataset' in out.out
    code, out = invoke(capsys, 'import-annotations', '--route', ROUTE, '--data-root', str(tmp_path))
    assert code != 0 and '--from-db' in out.out
    code, out = invoke(capsys, 'import-annotations', 'some.jsonl', '--from-db', '--data-root', str(tmp_path))
    assert code != 0 and 'invalid_arguments' in out.out
    assert not (tmp_path / 'annotations/objects.jsonl').exists()


def test_from_db_rejects_frames_whose_image_changed(tmp_path, capsys):
    rows = make_report(tmp_path)
    store = ReviewStore(db_path(tmp_path))
    save(tmp_path, store, rows[0], page_body(rows[0]))
    (tmp_path / rows[0]['image_path']).write_bytes(b'changed after review')
    code, out = invoke(capsys, 'import-annotations', '--from-db', '--data-root', str(tmp_path))
    assert code != 0 and 'hash_mismatch' in out.out
    assert not (tmp_path / 'annotations/objects.jsonl').exists()
