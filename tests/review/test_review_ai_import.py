"""AI reviews confirmed through the REST API or MCP are dataset labels: no person confirms them again.

The import takes them as they are and keeps their provenance (reviewer "ai/<name>",
label_basis ai_review, reviewer_kind, channel). Frames nobody confirmed stay out,
human reviews and the save checks are unchanged, and saving never imports or trains.
"""
from __future__ import annotations

import json

import pytest

from nnslr_tools import cli
from nnslr_tools.annotations import DatasetError, import_review_sources, jsonl_bytes, load_jsonl
from nnslr_tools.review_db import ReviewStore, db_path, import_from_db, save_frame_review
from nnslr_tools.review_mcp import ReviewTools
from review_helpers import ROUTE, RUN_ID, TOKEN, RunningServer, json_of, make_report, page_body, page_export_row

AUTH = {'Authorization': 'Bearer ' + TOKEN}
# Frames 0/0 and 0/2 have a speed proposal, 0/1 and 0/3 none.
KEYS = [(0, 0), (0, 1), (0, 2), (0, 3)]


@pytest.fixture
def site(tmp_path, monkeypatch):
    monkeypatch.setenv('NNSLR_REVIEW_TOKEN', TOKEN)
    rows = make_report(tmp_path, keys=KEYS)
    by_key = {}
    for row in rows:
        by_key[row['frame_key']] = row
    server = RunningServer(tmp_path)
    yield server, tmp_path, by_key
    server.close()


def invoke(capsys, *args):
    # What the review server printed before (its save log) is not the command's report.
    capsys.readouterr()
    try:
        code = cli.main(list(args))
    except SystemExit as exc:
        code = exc.code
    return code, capsys.readouterr()


def rest_save(server, key, body):
    status, response, data = server.request('POST', f'/api/v1/routes/{ROUTE}/frames/{key}/review', body=body,
                                             headers=AUTH)
    return status, json_of(data)


def ai_body(base_revision=0, detections=None, **extra):
    body = {'run_id': RUN_ID, 'base_revision': base_revision, 'reviewer': 'Claude', 'reviewer_kind': 'ai',
            'confirmed': True, 'detections': [] if detections is None else detections}
    body.update(extra)
    return body


def mcp_save(store, root, row, *, base_revision=0, confirmed=True, detections=None):
    """What the save_frame_review MCP tool does: an AI review on channel mcp."""
    body = {'run_id': RUN_ID, 'base_revision': base_revision, 'reviewer': 'Codex', 'reviewer_kind': 'ai',
            'confirmed': confirmed, 'detections': [] if detections is None else detections}
    return save_frame_review(store, root, route_id=ROUTE, run_id=RUN_ID, report_row=row, body=body, client='test',
                             channel='mcp', require_versioning=True)


def by_frame(records):
    frames = {}
    for record in records:
        frames.setdefault(record['output_index'], []).append(record)
    return frames


def revisions(store, keys):
    result = {}
    for key in keys:
        result[key] = (store.review(ROUTE, RUN_ID, key)['revision'], len(store.history(ROUTE, RUN_ID, key)))
    return result


def test_rest_and_mcp_confirmations_are_imported_without_a_human_confirmation(site, capsys):
    server, root, rows = site
    status, saved = rest_save(server, '0/0', ai_body(detections=[
        {'detection_id': '0/0/0', 'value_kph': 60, 'applicability': 'own_road'},
        {'bbox_xyxy': [16, 2, 30, 20], 'family': 'zone', 'value_kph': 30, 'applicability': 'other_road'}]))
    assert status == 200, saved
    tools = ReviewTools(server.app, client='test')
    frame = tools.get_frame(ROUTE, '0/1')
    saved = tools.save_frame_review(ROUTE, '0/1', reviewer='Codex', detections=[], **frame['save'])
    assert saved['review']['reviewer'] == 'ai/Codex' and saved['review']['channel'] == 'mcp'
    # Confirming imports nothing and trains nothing: the dataset does not exist until the import is run.
    dataset = root / 'annotations' / 'objects.jsonl'
    assert not dataset.exists() and not (root / 'annotations' / 'imports').exists()
    store = ReviewStore(db_path(root))
    before = revisions(store, ['0/0', '0/1'])

    code, out = invoke(capsys, 'import-annotations', '--from-db', '--data-root', str(root))
    assert code == 0, out.out + out.err
    report = json.loads(out.out)
    assert report['valid'] and report['imported_frame_count'] == 2 and report['imported_ai_frame_count'] == 2
    assert 'skipped_ai_reviews' not in report and report['changed_after_import'] == []
    # No human confirmation happened in between, and the import changed nothing in the review database.
    assert revisions(store, ['0/0', '0/1']) == before
    assert store.review(ROUTE, RUN_ID, '0/0')['reviewer_kind'] == 'ai'

    frames = by_frame(load_jsonl(dataset))
    assert sorted(frames) == [0, 1]
    for record in frames[0]:
        assert record['reviewer'] == 'ai/Claude'
        assert record['provenance']['label_basis'] == 'ai_review'
        assert record['provenance']['reviewer_kind'] == 'ai' and record['provenance']['review_channel'] == 'rest'
    states = []
    for record in frames[0]:
        states.append((record['sign_family'], record['value_kph'], record['applicability'], record['review_state'],
                       record['proposal_source']))
    assert sorted(states) == [('maximum_speed', 60, 'own_road', 'corrected', 'preannotation:' + RUN_ID),
                              ('zone', 30, 'other_road', 'corrected', 'ai_review')]
    [negative] = frames[1]
    assert negative['sign_family'] == 'not_a_sign' and negative['bbox_xyxy'] is None
    assert negative['reviewer'] == 'ai/Codex' and negative['proposal_source'] == 'ai_review'
    assert negative['provenance']['label_basis'] == 'ai_review' and negative['provenance']['review_channel'] == 'mcp'

    code, out = invoke(capsys, 'validate-dataset', '--data-root', str(root))
    assert code == 0, out.out + out.err
    code, out = invoke(capsys, 'import-annotations', '--from-db', '--data-root', str(root))
    report = json.loads(out.out)
    assert code == 0 and report['imported_frame_count'] == 0 and report['already_imported_frame_count'] == 2


def test_ai_reviews_saved_before_this_change_are_imported(tmp_path):
    rows = make_report(tmp_path)
    store = ReviewStore(db_path(tmp_path))
    output = tmp_path / 'annotations' / 'objects.jsonl'
    # Human review 0/1, already in the dataset: the previous import took human reviews only.
    human = save_frame_review(store, tmp_path, route_id=ROUTE, run_id=RUN_ID, report_row=rows[1],
                              body=page_body(rows[1]), client='test')
    import_review_sources([('review-db', jsonl_bytes([human['payload']]))], tmp_path, output)
    # AI review 0/0 as the REST API stored it, left in the database and never confirmed by a person.
    body = page_body(rows[0], reviewer='ai/Claude', value_kph=50)
    stored = page_export_row(rows[0], body, '2026-10-22T09:00:00.000Z', reviewer_kind='ai', channel='rest')
    store.save_review(route_id=ROUTE, run_id=RUN_ID, frame_key='0/0', segment_index=0, output_index=0,
                      payload=stored, reviewer='ai/Claude', client='test', base_revision=0, reviewer_kind='ai',
                      channel='rest')
    # Page review 0/2 of an AI, reclassified with review-reclassify-ai (channel web).
    save_frame_review(store, tmp_path, route_id=ROUTE, run_id=RUN_ID, report_row=rows[2],
                      body=page_body(rows[2], reviewer='Codex'), client='test')
    assert len(store.reclassify_as_ai('Codex', client='review-reclassify-ai')) == 1
    before = revisions(store, ['0/0', '0/1', '0/2'])
    dataset_before = output.read_bytes()

    report = import_from_db(tmp_path, output, store=store)
    assert report['valid'] and report['imported_frame_count'] == 2 and report['imported_ai_frame_count'] == 2
    # The human rows written by the previous version are the same rows: nothing "changed after import".
    assert report['already_imported_frame_count'] == 1 and report['changed_after_import'] == []
    assert revisions(store, ['0/0', '0/1', '0/2']) == before
    frames = by_frame(load_jsonl(output))
    for record in frames[1]:
        assert record['reviewer'] == 'Cristian' and record['provenance']['label_basis'] == 'human_review'
        assert 'reviewer_kind' not in record['provenance']
    old_lines = set(dataset_before.decode().splitlines())
    assert old_lines <= set(output.read_text().splitlines())
    [rest] = frames[0]
    assert rest['reviewer'] == 'ai/Claude' and rest['value_kph'] == 50
    assert (rest['provenance']['label_basis'], rest['provenance']['review_channel']) == ('ai_review', 'rest')
    for record in frames[2]:
        assert record['reviewer'] == 'ai/Codex'
        assert (record['provenance']['label_basis'], record['provenance']['review_channel']) == ('ai_review', 'web')


def test_frames_nobody_confirmed_stay_out(tmp_path, capsys):
    rows = make_report(tmp_path)
    store = ReviewStore(db_path(tmp_path))
    output = tmp_path / 'annotations' / 'objects.jsonl'
    # An AI confirmation withdrawn later is not a label.
    mcp_save(store, tmp_path, rows[0])
    mcp_save(store, tmp_path, rows[0], base_revision=1, confirmed=False)
    with pytest.raises(DatasetError, match='no confirmed reviews'):
        import_from_db(tmp_path, output, store=store)
    assert not output.exists()
    mcp_save(store, tmp_path, rows[2], detections=[{'detection_id': '0/2/0', 'applicability': 'own_road'}])
    report = import_from_db(tmp_path, output, store=store)
    assert report['imported_frame_count'] == 1
    # 0/0 withdrawn and 0/1 never reviewed: only the confirmed 0/2 is in the dataset.
    assert sorted(by_frame(load_jsonl(output))) == [2]
    dataset = output.read_bytes()
    # The model's proposals (the report lines) and an unconfirmed AI line are not reviews.
    report_lines = tmp_path / 'derived/preannotations' / ROUTE / RUN_ID / 'preannotations.jsonl'
    code, out = invoke(capsys, 'import-annotations', str(report_lines), '--data-root', str(tmp_path))
    assert code != 0 and 'unreviewed_proposal' in out.out
    pending = page_export_row(rows[1], page_body(rows[1], reviewer='ai/Claude'), '2026-10-22T09:00:00.000Z',
                              reviewer_kind='ai', channel='mcp')
    pending['review_state'] = 'pending'
    source = tmp_path / 'pending.jsonl'
    source.write_bytes(jsonl_bytes([pending]))
    code, out = invoke(capsys, 'import-annotations', str(source), '--data-root', str(tmp_path))
    assert code != 0 and 'unreviewed_proposal' in out.out
    assert output.read_bytes() == dataset


def test_human_reviews_and_save_checks_still_hold(site):
    server, root, rows = site
    assert server.post_frame(ROUTE, '0/0', page_body(rows['0/0'], reviewer='cristianku'))[0] == 200
    status, problem = rest_save(server, '0/0', ai_body(base_revision=1))
    assert status == 409 and problem['error'] == 'human_review_exists'
    status, problem = rest_save(server, '0/0', ai_body(base_revision=1, confirmed=False))
    assert status == 409 and problem['error'] == 'human_review_exists'
    assert rest_save(server, '0/2', ai_body())[0] == 200
    status, problem = rest_save(server, '0/2', ai_body(base_revision=0))
    assert status == 409 and problem['error'] == 'revision_conflict'
    status, problem = rest_save(server, '0/2', ai_body(base_revision=1, run_id='20261022T100000Z-89abcdef'))
    assert status == 409 and problem['error'] == 'stale_report'
    for detection, reason in [({'detection_id': '0/2/0', 'value_kph': 3}, 'illegal_speed'),
                              ({'bbox_xyxy': [1, 2, 99, 20], 'family': 'zone', 'value_kph': 30}, 'invalid_bbox'),
                              ({'bbox_xyxy': [1, 2, 9, 20], 'family': 'logo', 'value_kph': None}, 'unknown_enum')]:
        status, problem = rest_save(server, '0/2', ai_body(base_revision=1, detections=[detection]))
        assert status == 400 and problem['error'] == reason, detection
    store = ReviewStore(db_path(root))
    assert store.review(ROUTE, RUN_ID, '0/0')['reviewer'] == 'cristianku'
    assert store.review(ROUTE, RUN_ID, '0/2')['revision'] == 1

    report = import_from_db(root, root / 'annotations' / 'objects.jsonl', store=store)
    assert report['imported_frame_count'] == 2 and report['imported_ai_frame_count'] == 1
    frames = by_frame(load_jsonl(root / 'annotations' / 'objects.jsonl'))
    for record in frames[0]:
        assert record['reviewer'] == 'cristianku' and record['provenance']['label_basis'] == 'human_review'
    for record in frames[2]:
        assert record['reviewer'] == 'ai/Claude' and record['provenance']['label_basis'] == 'ai_review'


def test_ai_review_of_an_image_changed_after_the_review_is_not_imported(tmp_path, capsys):
    rows = make_report(tmp_path)
    store = ReviewStore(db_path(tmp_path))
    mcp_save(store, tmp_path, rows[0])
    (tmp_path / rows[0]['image_path']).write_bytes(b'changed after review')
    code, out = invoke(capsys, 'import-annotations', '--from-db', '--data-root', str(tmp_path))
    assert code != 0 and 'hash_mismatch' in out.out
    assert not (tmp_path / 'annotations/objects.jsonl').exists()


def test_page_export_of_an_ai_review_imports_like_the_database(site, capsys):
    server, root, rows = site
    status, saved = rest_save(server, '0/0', ai_body(detections=[{'detection_id': '0/0/0', 'value_kph': 60,
                                                                  'applicability': 'own_road'}]))
    assert status == 200, saved
    review = ReviewStore(db_path(root)).review(ROUTE, RUN_ID, '0/0')
    # The page export of an AI review (its boxes as the page reads them back) is the stored line.
    restored = json_of(server.request('GET', f'/api/route/{ROUTE}/reviews')[2])['reviews'][0]
    assert restored['reviewer_kind'] == 'ai' and restored['channel'] == 'rest'
    body = {'reviewer': restored['reviewer'], 'detections': restored['detections'], 'site_group': None,
            'encounter_id': None}
    exported = page_export_row(rows['0/0'], body, restored['review_timestamp'], reviewer_kind='ai',
                               channel=restored['channel'])
    assert exported == review['payload']
    page = server.request('GET', f'/route/{ROUTE}/')[2].decode()
    assert 'line.label_basis = "ai_review"' in page and 'line.review_channel = s.channel' in page
    assert 'non esportati' not in page
    source = root / 'annotations-export.jsonl'
    source.write_bytes(jsonl_bytes([exported]))
    from_file = root / 'from-file.jsonl'
    from_db = root / 'from-db.jsonl'
    code, out = invoke(capsys, 'import-annotations', str(source), '--data-root', str(root), '--output', str(from_file))
    assert code == 0, out.out + out.err
    code, out = invoke(capsys, 'import-annotations', '--from-db', '--data-root', str(root), '--output', str(from_db))
    assert code == 0, out.out + out.err
    assert from_db.read_bytes() == from_file.read_bytes()
    assert {r['provenance']['label_basis'] for r in load_jsonl(from_db)} == {'ai_review'}
