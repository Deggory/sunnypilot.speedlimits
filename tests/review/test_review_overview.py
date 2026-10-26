from pathlib import Path

from nnslr_tools.pages import render_index


def _summary(route: str, frames: int, confirmed: int, **extra) -> dict:
    return {'route_id': route, 'frame_count': frames, 'confirmed_count': confirmed, 'frames_with_proposals': 3,
            'url': '/route/' + route + '/', **extra}


def test_the_route_list_opens_on_an_overview_with_a_tile_per_group():
    page = render_index([_summary('0000011e--0c8825628a', 100, 40, confirmed_by_ai=10, waiting_for_person=5),
                         _summary('0000011f--77d17f9ee3', 50, 50)], Path('/tmp/review.sqlite'))
    # 90 of 150 frames confirmed, 10 of them by AI assistants; 5 saved by the site's AI wait for a person.
    assert '<strong>60%</strong>' in page and '90 of 150 in 2 drives' in page
    assert '80 by you · 10 by AI assistants' in page and '5 by AI to confirm' in page
    assert 'data-open-group="doing"' in page and 'data-open-group="todo" aria-controls="group-todo" disabled' in page
    # Each group folded, opened by its tile; no group without drives.
    assert '<details class="group" id="group-doing" data-group="doing">' in page and 'id="group-todo"' not in page
    assert '60 frames left' in page and '50 frames confirmed' in page
