"""import-annotations --from-db: same canonical dataset as the JSONL export, imported once."""
from __future__ import annotations

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


def test_from_db_default_output_is_idempotent_and_reports_late_changes(tmp_path, capsys):
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
    assert dataset.read_bytes() == first
    save(tmp_path, store, rows[0], page_body(rows[0], value_kph=80))
    save(tmp_path, store, rows[1], page_body(rows[1]))
    code, out = invoke(capsys, 'import-annotations', '--from-db', '--data-root', str(tmp_path))
    assert code == 0, out.out + out.err
    report = json.loads(out.out)
    assert report['imported_frame_count'] == 1
    assert [c['frame_key'] for c in report['changed_after_import']] == ['0/0']
    assert any('append-only' in w for w in report['warnings'])
    values = []
    for record in canonical(dataset):
        values.append((record['output_index'], record['value_kph']))
    assert sorted(values, key=str) == sorted([(0, 30), (1, None)], key=str)


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
    # The new frame is imported, the changed one is listed and keeps its old rows.
    assert report['imported_frame_count'] == 1
    assert [c['frame_key'] for c in report['changed_after_import']] == [f'0/{index}']
    assert frame_records(dataset, index) == before
    assert len(frame_records(dataset, 2)) == 1
    return before


def test_empty_frame_imported_then_box_added_is_changed_not_new(tmp_path, capsys):
    rows = make_report(tmp_path)
    first = page_body(rows[1])
    second = page_body(rows[1], manual_box=[3, 3, 12, 14])
    before = import_change_and_import_again(tmp_path, capsys, rows, 1, first, second)
    assert len(before) == 1 and before[0]['sign_family'] == 'not_a_sign' and before[0]['bbox_xyxy'] is None


def test_manual_box_drawn_again_is_changed_not_added(tmp_path, capsys):
    rows = make_report(tmp_path)
    first = page_body(rows[1], manual_box=[3, 3, 12, 14])
    second = page_body(rows[1], manual_box=[4, 4, 13, 15])
    second['detections'][0]['detection_id'] = 'manual-2'
    before = import_change_and_import_again(tmp_path, capsys, rows, 1, first, second)
    assert len(before) == 1 and before[0]['bbox_xyxy'] == [3, 3, 12, 14]


def test_manual_box_deleted_is_changed_not_imported(tmp_path, capsys):
    rows = make_report(tmp_path)
    first = page_body(rows[0], manual_box=[3, 3, 12, 14])
    second = page_body(rows[0])
    before = import_change_and_import_again(tmp_path, capsys, rows, 0, first, second)
    boxes = []
    for record in before:
        boxes.append(record['bbox_xyxy'])
    assert sorted(boxes) == [[1, 2, 15, 20], [3, 3, 12, 14]]


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
