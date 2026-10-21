"""Review database: schema version, append-only history, concurrent saves."""
from __future__ import annotations

import sqlite3
import threading

import pytest

from nnslr_tools.review_db import (SCHEMA_VERSION, ReviewRejected, ReviewStore, ReviewStoreError, db_path,
                                   save_frame_review)
from review_helpers import ROUTE, RUN_ID, RunningServer, make_report, page_body


def test_schema_version_wal_and_location(tmp_path):
    store = ReviewStore(db_path(tmp_path))
    assert store.path == tmp_path / 'annotations' / 'review.sqlite'
    assert store.schema_version() == SCHEMA_VERSION
    connection = sqlite3.connect(str(store.path))
    try:
        assert connection.execute('PRAGMA journal_mode').fetchone()[0] == 'wal'
        tables = set()
        for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'"):
            tables.add(row[0])
        assert {'schema_version', 'frame_reviews', 'frame_review_history'} <= tables
        connection.execute('INSERT INTO schema_version (version, applied_at) VALUES (?, ?)',
                           (SCHEMA_VERSION + 1, 'later'))
        connection.commit()
    finally:
        connection.close()
    with pytest.raises(ReviewStoreError, match='schema version'):
        ReviewStore(db_path(tmp_path))


def test_history_is_append_only(tmp_path):
    rows = make_report(tmp_path)
    store = ReviewStore(db_path(tmp_path))
    save_frame_review(store, tmp_path, route_id=ROUTE, run_id=RUN_ID, report_row=rows[0],
                      body=page_body(rows[0]), client='test')
    connection = sqlite3.connect(str(store.path))
    try:
        with pytest.raises(sqlite3.DatabaseError, match='append-only'):
            connection.execute("UPDATE frame_review_history SET reviewer = 'someone else'")
        with pytest.raises(sqlite3.DatabaseError, match='append-only'):
            connection.execute('DELETE FROM frame_review_history')
    finally:
        connection.close()
    assert len(store.history(ROUTE, RUN_ID, '0/0')) == 1


def test_invalid_frame_is_not_stored(tmp_path):
    rows = make_report(tmp_path)
    store = ReviewStore(db_path(tmp_path))
    body = page_body(rows[0])
    body['detections'][0]['bbox_xyxy'] = [-5, 0, 3, 3]
    with pytest.raises(ReviewRejected) as caught:
        save_frame_review(store, tmp_path, route_id=ROUTE, run_id=RUN_ID, report_row=rows[0], body=body,
                          client='test')
    assert caught.value.status == 400 and caught.value.reason == 'invalid_bbox'
    assert store.reviews(ROUTE, RUN_ID) == []
    assert store.history(ROUTE, RUN_ID, '0/0') == []


def test_report_image_changed_since_extraction_is_rejected(tmp_path):
    rows = make_report(tmp_path)
    (tmp_path / rows[0]['image_path']).write_bytes(b'changed')
    store = ReviewStore(db_path(tmp_path))
    with pytest.raises(ReviewRejected) as caught:
        save_frame_review(store, tmp_path, route_id=ROUTE, run_id=RUN_ID, report_row=rows[0],
                          body=page_body(rows[0]), client='test')
    assert caught.value.reason == 'hash_mismatch' and 'estrazione' in caught.value.message


def test_two_threads_saving_at_once_lose_nothing(tmp_path):
    rows = make_report(tmp_path, count=8)
    store = ReviewStore(db_path(tmp_path))
    errors = []
    start = threading.Barrier(2)

    def worker(name, frame_rows):
        try:
            start.wait()
            for repeat in range(3):
                for row in frame_rows:
                    save_frame_review(store, tmp_path, route_id=ROUTE, run_id=RUN_ID, report_row=row,
                                      body=page_body(row, reviewer=name), client=name)
                # Both threads also write the same frame.
                save_frame_review(store, tmp_path, route_id=ROUTE, run_id=RUN_ID, report_row=rows[0],
                                  body=page_body(rows[0], reviewer=name), client=name)
        except Exception as exc:
            errors.append(exc)

    first = threading.Thread(target=worker, args=('anna', rows[:4]))
    second = threading.Thread(target=worker, args=('bruno', rows[4:]))
    first.start()
    second.start()
    first.join()
    second.join()
    assert errors == []
    reviews = store.reviews(ROUTE, RUN_ID)
    assert len(reviews) == 8 and store.confirmed_count(ROUTE, RUN_ID) == 8
    shared = store.history(ROUTE, RUN_ID, '0/0')
    # rows[0] belongs to anna's set too: 3 saves in her loop + 3 + 3 shared saves.
    assert len(shared) == 9
    assert [h['revision'] for h in shared] == list(range(1, 10))
    assert store.review(ROUTE, RUN_ID, '0/0')['revision'] == 9
    for review in reviews:
        if review['frame_key'] != '0/0':
            assert review['revision'] == 3


def test_concurrent_http_saves(tmp_path):
    rows = make_report(tmp_path, count=6)
    server = RunningServer(tmp_path)
    results = []
    lock = threading.Lock()

    def worker(frame_rows):
        for row in frame_rows:
            status = server.post_frame(ROUTE, row['frame_key'], page_body(row))[0]
            with lock:
                results.append(status)

    try:
        threads = [threading.Thread(target=worker, args=(rows[:3],)), threading.Thread(target=worker, args=(rows[3:],))]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    finally:
        server.close()
    assert results == [200] * 6
    assert ReviewStore(db_path(tmp_path)).confirmed_count(ROUTE, RUN_ID) == 6
