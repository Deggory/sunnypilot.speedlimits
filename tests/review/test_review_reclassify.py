"""review-reclassify-ai: reviews an AI saved through the page become AI reviews, with history."""
from __future__ import annotations

import json

from nnslr_tools import cli
from nnslr_tools.annotations import load_jsonl
from nnslr_tools.review_db import ReviewStore, db_path, import_from_db, save_frame_review
from review_helpers import ROUTE, RUN_ID, make_report, page_body


def save(root, store, row, reviewer):
    return save_frame_review(store, root, route_id=ROUTE, run_id=RUN_ID, report_row=row,
                             body=page_body(row, reviewer=reviewer), client='test')


def run(capsys, root, *args):
    code = cli.main(['review-reclassify-ai', '--data-root', str(root), *args])
    return code, json.loads(capsys.readouterr().out)


def test_reclassify_codex_reviews_as_ai(tmp_path, capsys):
    rows = make_report(tmp_path)
    store = ReviewStore(db_path(tmp_path))
    first = save(tmp_path, store, rows[0], 'Codex')
    save(tmp_path, store, rows[1], 'Cristian')
    save(tmp_path, store, rows[2], 'Codex')
    store.withdraw_review(route_id=ROUTE, run_id=RUN_ID, frame_key='0/2', reviewer='Codex', client='test',
                          base_revision=1)
    code, report = run(capsys, tmp_path, '--reviewer', 'Codex', '--dry-run')
    assert code == 0 and report['dry_run'] and report['reviewers']['Codex']['frames'] == 2
    assert store.review(ROUTE, RUN_ID, '0/0')['reviewer_kind'] == 'human'
    assert len(store.history(ROUTE, RUN_ID, '0/0')) == 1
    code, report = run(capsys, tmp_path, '--reviewer', 'Codex', '--reviewer', 'ChatGPT')
    assert code == 0 and report['reviewers'] == {
        'Codex': {'frames': 2, 'by_route_run': {ROUTE + ' ' + RUN_ID: 2}},
        'ChatGPT': {'frames': 0, 'by_route_run': {}}}
    review = store.review(ROUTE, RUN_ID, '0/0')
    assert review['reviewer'] == 'ai/Codex' and review['reviewer_kind'] == 'ai' and review['channel'] == 'web'
    assert review['revision'] == 2 and review['confirmed']
    payload = review['payload']
    assert payload['label_basis'] == 'ai_review' and payload['reviewer'] == 'ai/Codex'
    assert payload['detections'] == first['payload']['detections']
    assert payload['review_timestamp'] == first['payload']['review_timestamp']
    assert payload['reclassified_from']['label_basis'] == 'human_review'
    history = store.history(ROUTE, RUN_ID, '0/0')
    assert [(h['revision'], h['reviewer'], h['reviewer_kind'], h['client']) for h in history] == [
        (1, 'Codex', 'human', 'test'), (2, 'ai/Codex', 'ai', 'review-reclassify-ai')]
    withdrawn = store.review(ROUTE, RUN_ID, '0/2')
    assert withdrawn['reviewer_kind'] == 'ai' and not withdrawn['confirmed'] and withdrawn['revision'] == 3
    person = store.review(ROUTE, RUN_ID, '0/1')
    assert person['reviewer'] == 'Cristian' and person['reviewer_kind'] == 'human' and person['revision'] == 1
    # Running it again changes nothing.
    code, report = run(capsys, tmp_path, '--reviewer', 'Codex')
    assert report['reviewers']['Codex']['frames'] == 0 and store.review(ROUTE, RUN_ID, '0/0')['revision'] == 2
    # The import takes both, the reclassified frame as an AI review (channel web).
    result = import_from_db(tmp_path, tmp_path / 'objects.jsonl', store=store)
    assert result['imported_frame_count'] == 2 and result['imported_ai_frame_count'] == 1
    for record in load_jsonl(tmp_path / 'objects.jsonl'):
        provenance = record['provenance']
        if record['output_index'] == 0:
            assert record['reviewer'] == 'ai/Codex' and provenance['label_basis'] == 'ai_review'
            assert provenance['review_channel'] == 'web'
        else:
            assert record['reviewer'] == 'Cristian' and provenance['label_basis'] == 'human_review'


def test_frames_reclassified_after_their_import_are_replaced_as_ai_reviews(tmp_path, capsys):
    rows = make_report(tmp_path)
    store = ReviewStore(db_path(tmp_path))
    save(tmp_path, store, rows[0], 'Codex')
    save(tmp_path, store, rows[1], 'Cristian')
    output = tmp_path / 'objects.jsonl'
    first = import_from_db(tmp_path, output, store=store)
    assert first['imported_frame_count'] == 2 and first['imported_ai_frame_count'] == 0
    # Imported as a person's review, then marked as what it was: an AI review.
    code, report = run(capsys, tmp_path, '--reviewer', 'Codex')
    assert code == 0 and report['reviewers']['Codex']['frames'] == 1
    result = import_from_db(tmp_path, output, store=store)
    assert result['replaced_frame_count'] == 1 and result['already_imported_frame_count'] == 1
    assert [item['frame_key'] for item in result['changed_after_import']] == ['0/0']
    bases = {}
    for record in load_jsonl(output):
        bases.setdefault(record['output_index'], set()).add((record['reviewer'], record['provenance']['label_basis']))
    assert bases == {0: {('ai/Codex', 'ai_review')}, 1: {('Cristian', 'human_review')}}
    archived = load_jsonl(next((tmp_path / 'annotations' / 'replaced').glob('*.jsonl')))
    assert {(r['reviewer'], r['provenance']['label_basis']) for r in archived} == {('Codex', 'human_review')}
