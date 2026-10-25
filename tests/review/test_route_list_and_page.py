"""The route list (/routes) and the data of the review page: what the cards count, the AI's share, one link to set
up the AI reviewer, who proposed the signs, the drive's two lines in the route menu."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from nnslr_tools.pages import render_index
from nnslr_tools.review_db import ReviewStore, db_path
from review_helpers import OTHER_ROUTE, ROUTE, RUN_ID, RunningServer, make_report


def ai_save(app, route, key, detections):
    """A frame confirmed by an AI reviewer through the REST API."""
    body = {'run_id': RUN_ID, 'base_revision': 0, 'reviewer': 'Claude', 'reviewer_kind': 'ai', 'confirmed': True,
            'detections': detections}
    return app.save_review(route, key, body, client='test', channel='rest')


@pytest.fixture
def served(tmp_path):
    make_report(tmp_path, count=3)
    make_report(tmp_path, route=OTHER_ROUTE, count=2)
    server = RunningServer(tmp_path)
    yield server, tmp_path
    server.close()


def card_of(page: str, route: str) -> str:
    return next(item for item in page.split('<li>') if 'href="/route/' + route + '/"' in item)


def test_route_list_counts_proposals_while_to_review_and_the_share_of_the_ai(served):
    server, root = served
    ai_save(server.app, ROUTE, '0/1', [])
    ai_save(server.app, OTHER_ROUTE, '0/0', [{'detection_id': '0/0/0', 'applicability': 'own_road'}])
    ai_save(server.app, OTHER_ROUTE, '0/1', [])
    store = ReviewStore(db_path(root))
    assert store.confirmed_count(ROUTE, RUN_ID) == 1 and store.confirmed_count(ROUTE, RUN_ID, reviewer_kind='ai') == 1
    assert store.confirmed_count(ROUTE, RUN_ID, reviewer_kind='human') == 0
    page = server.request('GET', '/routes')[2].decode()
    doing = card_of(page, ROUTE)
    # The model's proposals, not signs: said only while there is something to review.
    assert '2 with proposals' in doing and 'with signs' not in page
    assert '<span class="by-ai">1 by AI</span>' in doing
    # The bar: none by people, a third by the AI, hatched.
    assert '<span style="width:0%"></span><span class="ai" style="width:33%"></span>' in doing
    done = card_of(page, OTHER_ROUTE)
    assert 'with proposals' not in done and '2 by AI' in done and 'class="ai" style="width:100%"' in done
    # No AI reviewer in Settings: one link to set it up, no button on every drive.
    assert page.count('Set up AI review') == 1 and 'href="/account#ai-reviewer"' in page
    assert 'data-ai-route="' not in page


def test_route_list_offers_review_with_ai_on_each_drive_left_once_a_reviewer_is_chosen():
    def summary(route, confirmed, by_ai=0):
        return {'route_id': route, 'run_id': RUN_ID, 'frame_count': 4, 'frames_with_proposals': 2,
                'confirmed_count': confirmed, 'confirmed_by_ai': by_ai, 'suspicious_count': 0, 'hint_frames': 0,
                'url': '/route/' + route + '/', 'drive': {'title': 'Herznach → Zürich', 'when': 'Fri 23 Oct 2026 · 08:01',
                                                          'minutes': 49}, 'proposer': 'qwen'}
    page = render_index([summary(ROUTE, 1), summary(OTHER_ROUTE, 4, by_ai=4)], Path('review.sqlite'), ai_reviewer=True)
    assert 'Set up AI review' not in page
    assert page.count('data-ai-route=') == 1 and 'data-ai-route="' + ROUTE + '"' in page
    assert 'signs by Qwen3-VL' in page and '4 by AI' in page
    # Without one, nobody is offered a button that cannot work.
    page = render_index([summary(ROUTE, 1)], Path('review.sqlite'), ai_reviewer=False)
    assert 'data-ai-route="' not in page and 'Set up AI review' in page
    # Our own networks, mid-sentence: in lower case.
    page = render_index([{**summary(ROUTE, 1), 'proposer': 'ours'}], Path('review.sqlite'))
    assert 'signs by our networks' in page
    # Every drive done: nothing to review with an AI, no link either.
    page = render_index([summary(OTHER_ROUTE, 4, by_ai=4)], Path('review.sqlite'), ai_reviewer=False)
    assert 'Set up AI review' not in page


def payload_of(page: str) -> dict:
    found = re.search(r'<script type="application/json" id="payload">(.*?)</script>', page, re.S)
    assert found
    return json.loads(found.group(1))


def test_review_page_knows_who_proposed_the_signs_and_the_drive_apart(served):
    server, root = served
    run = root / 'derived/preannotations' / ROUTE / RUN_ID / 'run.json'
    run.write_text(json.dumps({'run_id': RUN_ID, 'route_id': ROUTE, 'proposer': 'qwen'}))
    payload = payload_of(server.request('GET', f'/route/{ROUTE}/')[2].decode())
    assert payload['proposer'] == 'qwen'
    link = next(item for item in payload['routes'] if item['route_id'] == ROUTE)
    # The two lines of the route menu (nothing is known of the drive in the test).
    assert {'title', 'when', 'minutes', 'name'} <= set(link) and link['title'] is None
    assert payload_of(server.request('GET', f'/route/{OTHER_ROUTE}/')[2].decode())['proposer'] == 'vietnam'
