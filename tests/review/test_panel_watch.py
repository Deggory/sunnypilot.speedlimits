"""The Watch panel of the Evaluate tab: the frames of one route of an evaluation, with the boxes to draw over each
(nnslr_tools.runs.frames_digest, RunsIndex.frames) and GET /api/panel/runs/<run>/frames.

The reports are synthetic, built with the real matching and per-frame helpers of nnslr_tools.evaluation: in the form
written now (every box of the cutter from 0.05, per_frame_boxes) and in the forms written before (a detector report
with its boxes at its own threshold only, a chain report with its verdicts at its main score only).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from nnslr_tools.evaluation import (PER_FRAME_MAX_BOXES, PER_FRAME_STEPS, greedy_matching, match_detections,
                                    per_frame_descriptor, per_frame_record)
from nnslr_tools.panel import Panel, PanelError
from nnslr_tools.review_service import ReviewService
from nnslr_tools.runs import PER_FRAME_STEPS as RUNS_STEPS, RunsIndex, WatchError
from review_helpers import TOKEN, RunningServer, json_of, make_report, sign_in
from test_panel_runs import AI, HUMAN, make_classifier_run, make_cutter_run, make_run, write

ROUTE = '0000012d--6552a45378'
OTHER = '00000123--8fe1b32468'
# The frames of ROUTE in the report, out of order (the panel plays them in the order of the drive).
KEYS = [(3, 1), (2, 7), (2, 0), (3, 0)]


def frames_of(root: Path, route: str, keys) -> dict:
    """The pre-annotation report of a route (its previews), by frame key."""
    rows = make_report(root, route=route, keys=keys)
    return {row['frame_key']: row for row in rows}


def box(x: float, y: float, side: float = 20.0) -> list[float]:
    return [x, y, x + side, y + side]


# Per frame of ROUTE (by key): the reviewed boxes (box, family) and the cutter's boxes (box, score).
FRAMES = {
    '2/0': ([], [(box(500, 300), 0.07)]),
    '2/7': ([(box(100, 100), 'maximum_speed'), (box(300, 100), 'other_sign')],
            # The speed sign found at 0.3, the other sign only at 0.08; a false box at 0.6.
            [(box(101, 101), 0.3), (box(301, 99), 0.0812345), (box(700, 400), 0.6)]),
    '3/0': ([(box(50, 50), 'cancellation')], []),
    '3/1': ([], []),
}


def detector_image(route: str, row: dict, truth, predictions, *, new: bool, threshold: float = 0.25) -> dict:
    """An image of evaluate_detector: at the report's threshold, and (new) every box from 0.05 for the panel."""
    truth_boxes = [item[0] for item in truth]
    selected = [{'box': b, 'score': s} for b, s in predictions if s >= threshold]
    matched = match_detections(selected, truth_boxes, iou_threshold=0.5)
    image = {'image_path': row['image_path'], 'image_sha256': row['frame_sha256'], 'route_id': route,
             'segment_index': row['segment_index'], 'label_basis': HUMAN, 'truth_boxes': truth_boxes,
             'truth_families': [item[1] for item in truth], 'predictions': selected, 'truth_count': len(truth),
             'prediction_count': len(selected), 'true_positives': matched['true_positives'],
             'false_positives': matched['false_positives'], 'false_negatives': matched['false_negatives'],
             'matches': matched['matches'], 'false_positive_details': matched['false_positive_details']}
    if new:
        low = [{'box': b, 'score': s} for b, s in predictions if s >= 0.05]
        pairs = {position: index for position, index, _ in greedy_matching(low, truth_boxes, iou_threshold=0.5) if index >= 0}
        image.update(size=[1344, 760], **per_frame_record(low, pairs))
    return image


def detector_report(root: Path, *, new: bool, partition: str = 'test') -> dict:
    rows = frames_of(root, ROUTE, KEYS)
    other = frames_of(root, OTHER, [(0, 5)])
    images = [detector_image(ROUTE, rows[key], *FRAMES[key], new=new) for key in ('3/1', '2/7', '3/0', '2/0')]
    images.append(detector_image(OTHER, other['0/5'], [], [(box(10, 10), 0.9)], new=new))
    report = {'schema_version': 1, 'task': 'detector', 'partition': partition, 'score_threshold': 0.25,
              'iou_threshold': 0.5, 'images': images}
    if new:
        report['per_frame_boxes'] = per_frame_descriptor(chain=False, min_score=0.05)
    return report


@pytest.fixture
def root(tmp_path):
    return tmp_path / 'data'


@pytest.fixture
def index(root):
    value = RunsIndex(root)
    value.running_runs = lambda: {}
    return value


def previews(root: Path):
    service = ReviewService(root, None)
    return lambda route: service.report(route)[2]


def test_the_steps_of_the_slider_are_the_same_numbers_in_both_modules():
    assert PER_FRAME_STEPS == RUNS_STEPS and PER_FRAME_STEPS[0] == 0.05 and PER_FRAME_STEPS[-1] == 0.5 and len(PER_FRAME_STEPS) == 10


def test_the_reports_a_run_can_play_are_listed_in_its_detail(root, index):
    from test_panel_models import make_size
    run = make_run(root, '2026-10-24_1452')
    make_size(run, '640x360')
    watch = index.detail(run.name)['watch']
    assert [(item['model'], item['label'], item['kind'], item['partitions']) for item in watch] == [
        ('detector', 'Detector of the pipeline', 'cutter', ['test']),
        ('detector-sizes/640x360', 'Cutter 640 × 360', 'cutter', ['test', 'validation'])]
    classifier = make_classifier_run(root)
    assert [(item['model'], item['partitions']) for item in index.detail(classifier.name)['watch']] == [
        ('chain', ['test', 'validation']), ('chain-old-reader', ['test'])]
    cutter = make_cutter_run(root, upto='detector')
    assert index.detail(cutter.name)['watch'] == []


def test_an_older_detector_report_plays_its_boxes_from_its_own_threshold(root, index):
    run = make_cutter_run(root)
    write(run / 'detector-sizes/960x544/evaluation/detector-test.json', detector_report(root, new=False))
    payload = index.frames(run.name, 'detector-sizes/960x544', 'test', None, previews(root))
    assert (payload['level'], payload['min_score'], payload['default_score'], payload['steps']) == ('above', 0.25, 0.25, None)
    assert (payload['kind'], payload['label'], payload['route'], payload['route_not_here']) == ('cutter', 'Cutter 960 × 544', ROUTE, None)
    # Every route of the report, with its signs and its misses at the score the panel starts at.
    assert payload['routes'] == [
        {'route': OTHER, 'frames': 1, 'signs': 0, 'frames_with_signs': 0, 'missed': 0, 'frames_with_misses': 0},
        {'route': ROUTE, 'frames': 4, 'signs': 3, 'frames_with_signs': 2, 'missed': 2, 'frames_with_misses': 2}]
    # In the order of the drive, each with its preview of the review page.
    frames = payload['frames']
    assert [frame['key'] for frame in frames] == ['2/0', '2/7', '3/0', '3/1']
    order = {key: position for position, key in enumerate(f'{s}/{i}' for s, i in KEYS)}
    assert [frame['image'] for frame in frames] == [f'/route/{ROUTE}/images/{order[key]:08d}.jpg' for key in ('2/0', '2/7', '3/0', '3/1')]
    assert payload['with_image'] == 4 and (frames[0]['width'], frames[0]['height']) == (32, 24)
    signs = frames[1]
    assert signs['truth'] == [{'box': [100, 100, 120, 120], 'family': 'maximum_speed', 'label': None},
                              {'box': [300, 100, 320, 120], 'family': 'other_sign', 'label': None}]
    # Boxes of 0.25 and more, best first, each with the reviewed box it found (-1: none): x1, y1, x2, y2, score, truth.
    assert signs['boxes'] == [[700, 400, 720, 420, 0.6, -1], [101, 101, 121, 121, 0.3, 0]]
    assert frames[0]['boxes'] == [] and signs['counts'] is None
    # The paths of the frames on disk never leave the server.
    assert 'derived/frames' not in json.dumps(payload)


def test_a_new_detector_report_keeps_every_box_from_0_05_and_counts_them_at_each_step(root, index):
    run = make_cutter_run(root)
    report = detector_report(root, new=True)
    write(run / 'detector-sizes/960x544/evaluation/detector-test.json', report)
    payload = index.frames(run.name, 'detector-sizes/960x544', 'test', ROUTE, previews(root))
    assert (payload['level'], payload['min_score'], payload['default_score']) == ('all', 0.05, 0.1)
    assert payload['steps'] == list(PER_FRAME_STEPS) and payload['max_boxes'] == PER_FRAME_MAX_BOXES
    # At 0.1 the other sign (0.08) is missed too.
    assert [row for row in payload['routes'] if row['route'] == ROUTE][0]['missed'] == 2
    frames = {frame['key']: frame for frame in payload['frames']}
    signs = frames['2/7']
    # The score is rounded down: 0.0812345 stays under 0.082, compared with a step it stays on its side.
    assert signs['boxes'] == [[700, 400, 720, 420, 0.6, -1], [101, 101, 121, 121, 0.3, 0], [301, 99, 321, 119, 0.081, 1]]
    assert signs['counts'] == [3, 2, 2, 2, 2, 2, 1, 1, 1, 1] and signs['surviving'] is None
    assert (signs['width'], signs['height']) == (1344, 760)
    assert frames['2/0']['boxes'] == [[500, 300, 520, 320, 0.07, -1]]
    assert frames['2/0']['counts'] == [1] + [0] * 9
    # The report itself: one line per box, the old keys as they were.
    image = next(item for item in report['images'] if item['image_path'].endswith('2/frame_00000007.png'))
    assert image['cutter_boxes'] == ['700 400 720 420 0.600 -1', '101 101 121 121 0.300 0', '301 99 321 119 0.081 1']
    assert image['predictions'] == [{'box': box(101, 101), 'score': 0.3}, {'box': box(700, 400), 'score': 0.6}]


def test_a_frame_keeps_every_box_on_a_sign_and_the_best_others_up_to_its_cap():
    truth = [box(0, 0)]
    # Many false boxes, and the sign found only by a box of a low score.
    predictions = [{'box': box(100 + 30 * n, 300), 'score': 0.9 - n * 0.01} for n in range(PER_FRAME_MAX_BOXES + 20)]
    predictions.append({'box': box(1, 1), 'score': 0.051})
    pairs = {position: index for position, index, _ in greedy_matching(predictions, truth, iou_threshold=0.5) if index >= 0}
    record = per_frame_record(predictions, pairs)
    kept = record['cutter_boxes']
    assert len(kept) == PER_FRAME_MAX_BOXES and kept[-1] == '1 1 21 21 0.051 0'
    # The best false boxes, best first; the counts are of every box, kept or not.
    assert kept[0].endswith(' 0.900 -1') and record['cutter_box_counts'][0] == PER_FRAME_MAX_BOXES + 21
    # A score just under a step stays under it once rounded.
    assert per_frame_record([{'box': box(0, 0), 'score': 0.0999999}], {})['cutter_boxes'] == ['0 0 20 20 0.099 -1']
    assert per_frame_record([{'box': box(0, 0), 'score': 0.0999999}], {})['cutter_box_counts'][1] == 0


def test_greedy_matching_is_the_matching_of_the_evaluation():
    truth = [box(0, 0), box(100, 0)]
    predicted = [{'box': box(1, 1), 'score': 0.2}, {'box': box(0, 0), 'score': 0.9}, {'box': box(400, 0), 'score': 0.5}]
    assert greedy_matching(predicted, truth, iou_threshold=0.5) == [(1, 0, 1.0), (2, -1, 0.0), (0, -1, pytest.approx(0.0))]
    matched = match_detections(predicted, truth, iou_threshold=0.5)
    assert matched['false_positive_indexes'] == [2, 0] and matched['matches'][0]['truth_index'] == 0
    # The matching at a score is the matching from the lowest score, restricted to the boxes at that score.
    full = {position: index for position, index, _ in greedy_matching(predicted, truth, iou_threshold=0.5)}
    high = [item for item in predicted if item['score'] >= 0.5]
    assert {predicted.index(high[p]): i for p, i, _ in greedy_matching(high, truth, iou_threshold=0.5)} == {
        p: i for p, i in full.items() if predicted[p]['score'] >= 0.5}


def chain_image(route: str, row: dict, truth, predictions, *, new: bool) -> dict:
    """An image of evaluate_chain at its main score 0.1: truth (box, label), predictions (box, score, predicted)."""
    objects = [{'box': b, 'label': label, 'family': label.split(':')[0], 'speed_sign': label.startswith('maximum_speed'),
                'label_basis': AI} for b, label in truth]
    items = [{'box': b, 'score': s, 'predicted': p, 'confidence': 0.8} for b, s, p in predictions]
    selected = [item for item in items if item['score'] >= 0.1]
    from nnslr_tools.chain import match_boxes
    pairs = match_boxes(selected, objects, 0.5)
    by_truth = {index: selected[position] for position, index in pairs.items()}
    image = {'image_path': row['image_path'], 'image_sha256': row['frame_sha256'], 'route_id': route,
             'segment_index': row['segment_index'], 'label_basis': AI, 'boxes': len(selected),
             'truth': [{'box': item['box'], 'label': item['label'], 'speed_sign': item['speed_sign'],
                        'label_basis': item['label_basis'], 'found': index in by_truth,
                        'predicted': by_truth[index]['predicted'] if index in by_truth else None,
                        'confidence': by_truth[index]['confidence'] if index in by_truth else None,
                        'correct': index in by_truth and by_truth[index]['predicted'] == item['label']}
                       for index, item in enumerate(objects)],
             'surviving_false': [{'box': item['box'], 'score': item['score'], 'predicted': item['predicted'],
                                  'confidence': item['confidence']} for position, item in enumerate(selected)
                                 if position not in pairs and item['predicted'] != 'not_a_sign']}
    if new:
        for item, source in zip(image['truth'], objects):
            item['family'] = source['family']
        image.update(size=[1344, 760], **per_frame_record(items, match_boxes(items, objects, 0.5), chain=True))
    return image


CHAIN_FRAMES = {
    '2/7': ([(box(100, 100), 'maximum_speed:80'), (box(300, 100), 'maximum_speed:60')],
            [(box(101, 101), 0.3, 'maximum_speed:80'), (box(301, 101), 0.2, 'maximum_speed:80'),
             (box(700, 400), 0.6, 'not_a_sign'), (box(900, 400), 0.15, 'cancellation'), (box(900, 100), 0.06, 'other_sign')]),
    '2/0': ([], [(box(500, 300), 0.07, 'not_a_sign')]),
}


def chain_report(root: Path, *, new: bool) -> dict:
    rows = frames_of(root, ROUTE, [(2, 0), (2, 7)])
    report = {'schema_version': 1, 'task': 'chain', 'partition': 'test', 'iou_threshold': 0.5, 'main_score_threshold': 0.1,
              'score_thresholds': [0.05, 0.1, 0.25], 'classifier': {'rejects': True, 'class_scheme': 'v2'},
              'images': [chain_image(ROUTE, rows[key], *CHAIN_FRAMES[key], new=new) for key in ('2/7', '2/0')]}
    if new:
        report['per_frame_boxes'] = per_frame_descriptor(chain=True, min_score=0.05)
    return report


def test_an_older_chain_report_plays_its_verdicts_at_its_main_score(root, index):
    run = make_classifier_run(root)
    write(run / 'evaluation/chain-test.json', chain_report(root, new=False))
    payload = index.frames(run.name, 'chain', 'test', None, previews(root))
    assert (payload['kind'], payload['level'], payload['min_score'], payload['default_score'], payload['rejects']) == (
        'chain', 'verdicts', 0.1, 0.1, True)
    frame = next(item for item in payload['frames'] if item['key'] == '2/7')
    assert frame['truth'] == [
        {'box': [100, 100, 120, 120], 'family': 'maximum_speed', 'label': 'maximum_speed:80', 'found': True,
         'verdict': 'maximum_speed:80', 'confidence': 0.8},
        {'box': [300, 100, 320, 120], 'family': 'maximum_speed', 'label': 'maximum_speed:60', 'found': True,
         'verdict': 'maximum_speed:80', 'confidence': 0.8}]
    # Only the false boxes it let through are in the report.
    assert frame['boxes'] == [[900, 400, 920, 420, 0.15, -1, 'cancellation', 0.8]]
    assert payload['routes'][0]['missed'] == 0


def test_a_new_chain_report_keeps_every_box_with_the_classifiers_answer(root, index):
    run = make_classifier_run(root)
    write(run / 'evaluation/chain-test.json', chain_report(root, new=True))
    payload = index.frames(run.name, 'chain', 'test', None, previews(root))
    assert (payload['level'], payload['min_score'], payload['default_score'], payload['steps']) == ('all', 0.05, 0.1, list(PER_FRAME_STEPS))
    frame = next(item for item in payload['frames'] if item['key'] == '2/7')
    assert [item['label'] for item in frame['truth']] == ['maximum_speed:80', 'maximum_speed:60'] and 'found' not in frame['truth'][0]
    # x1, y1, x2, y2, score, truth, label, confidence.
    assert frame['boxes'] == [[700, 400, 720, 420, 0.6, -1, 'not_a_sign', 0.8],
                              [101, 101, 121, 121, 0.3, 0, 'maximum_speed:80', 0.8],
                              [301, 101, 321, 121, 0.2, 1, 'maximum_speed:80', 0.8],
                              [900, 400, 920, 420, 0.15, -1, 'cancellation', 0.8],
                              [900, 100, 920, 120, 0.06, -1, 'other_sign', 0.8]]
    # Every box at each step, and the false boxes the classifier let through (the 0.15 and the 0.06 ones).
    assert frame['counts'] == [5, 4, 4, 3, 2, 2, 1, 1, 1, 1]
    assert frame['surviving'] == [2, 1, 1, 0, 0, 0, 0, 0, 0, 0]


def test_a_report_without_boxes_still_plays_its_signs(root, index):
    run = make_cutter_run(root)
    report = detector_report(root, new=False)
    for image in report['images']:
        del image['predictions']
    write(run / 'detector-sizes/960x544/evaluation/detector-test.json', report)
    payload = index.frames(run.name, 'detector-sizes/960x544', 'test', None, previews(root))
    assert (payload['level'], payload['default_score'], payload['steps']) == ('none', None, None)
    assert [row['missed'] for row in payload['routes']] == [None, None]
    assert all(frame['boxes'] == [] for frame in payload['frames']) and payload['frames'][1]['truth']


def test_a_preview_of_another_image_is_not_shown(root, index):
    run = make_cutter_run(root)
    report = detector_report(root, new=True)
    for image in report['images']:
        image['image_sha256'] = 'f' * 64
    write(run / 'detector-sizes/960x544/evaluation/detector-test.json', report)
    payload = index.frames(run.name, 'detector-sizes/960x544', 'test', ROUTE, previews(root))
    assert payload['with_image'] == 0 and all(frame['image'] is None for frame in payload['frames'])
    # Without a pre-annotation report of the route, no preview either.
    payload = index.frames(run.name, 'detector-sizes/960x544', 'test', ROUTE, lambda route: {})
    assert payload['with_image'] == 0


@pytest.mark.parametrize('model, partition, route, status, reason', [
    ('../evaluation', 'test', None, 400, 'invalid_model'),
    ('detector-sizes/960x544/../../x', 'test', None, 400, 'invalid_model'),
    ('detector-sizes/0x1', 'test', None, 400, 'invalid_model'),
    ('reader', 'test', None, 400, 'invalid_model'),
    (None, 'test', None, 400, 'invalid_model'),
    ('detector-sizes/960x544', 'train', None, 400, 'invalid_partition'),
    ('detector-sizes/960x544', 'test', '../../etc', 400, 'invalid_route'),
    ('detector-sizes/640x360', 'test', None, 404, 'no_report'),
    ('chain-old-reader', 'validation', None, 404, 'no_report'),
])
def test_what_the_frames_refuse(root, index, model, partition, route, status, reason):
    run = make_cutter_run(root)
    write(run / 'detector-sizes/960x544/evaluation/detector-test.json', detector_report(root, new=True))
    with pytest.raises(WatchError) as error:
        index.frames(run.name, model, partition, route)
    assert (error.value.status, error.value.reason) == (status, reason)


def test_an_unknown_run_an_unreadable_report_and_a_route_of_another_report(root, index):
    run = make_cutter_run(root)
    path = run / 'detector-sizes/960x544/evaluation/detector-test.json'
    write(path, detector_report(root, new=True))
    with pytest.raises(WatchError) as error:
        index.frames('nope', 'detector', 'test')
    assert error.value.reason == 'unknown_run'
    # A route this report has no frame of: the panel keeps it and the server shows the report's own first route.
    payload = index.frames(run.name, 'detector-sizes/960x544', 'test', '00000999--0000000000')
    assert (payload['route'], payload['route_not_here']) == (ROUTE, '00000999--0000000000')
    path.write_text('{"task": ')
    with pytest.raises(WatchError) as error:
        index.frames(run.name, 'detector-sizes/960x544', 'test')
    assert (error.value.status, error.value.reason) == (409, 'unreadable_report')
    panel = Panel(root, repo=root / 'no-checkout', actions_enabled=False)
    with pytest.raises(PanelError) as error:
        panel.frames('nope', 'detector', 'test')
    assert (error.value.status, error.value.reason) == (404, 'unknown_run')


def test_a_report_outside_its_run_is_never_read(root, index, tmp_path):
    run = make_cutter_run(root)
    outside = tmp_path / 'elsewhere.json'
    write(outside, detector_report(root, new=True))
    target = run / 'detector-sizes/960x544/evaluation/detector-test.json'
    target.unlink()
    target.symlink_to(outside)
    with pytest.raises(WatchError) as error:
        index.frames(run.name, 'detector-sizes/960x544', 'test')
    assert error.value.reason == 'no_report'


def test_frames_over_http_with_the_sign_in(tmp_path):
    run = make_cutter_run(tmp_path)
    write(run / 'detector-sizes/960x544/evaluation/detector-test.json', detector_report(tmp_path, new=True))
    server = RunningServer(tmp_path, token=TOKEN, require_token=True)
    server.app.panel.runs_index.running_runs = lambda: {}
    address = f'/api/panel/runs/{run.name}/frames?model=detector-sizes%2F960x544&partition=test'
    try:
        assert server.request('GET', address)[0] == 401
        assert server.request('GET', f'/route/{ROUTE}/images/00000000.jpg')[0] == 401
        cookie = sign_in(server)
        status, response, data = server.request('GET', address, headers=cookie)
        payload = json_of(data)
        assert status == 200 and payload['route'] == ROUTE and payload['level'] == 'all' and len(payload['frames']) == 4
        image = payload['frames'][0]['image']
        status, response, data = server.request('GET', image, headers=cookie)
        assert status == 200 and data == b'jpeg preview' and response.getheader('Content-Type') == 'image/jpeg'
        status, response, data = server.request('GET', address + '&route=' + OTHER, headers=cookie)
        assert status == 200 and json_of(data)['route'] == OTHER and len(json_of(data)['frames']) == 1
        for bad, code in ((f'/api/panel/runs/{run.name}/frames?model=..%2F..%2Fsplits&partition=test', 400),
                          (f'/api/panel/runs/{run.name}/frames?partition=test', 400),
                          (f'/api/panel/runs/{run.name}/frames?model=detector-sizes%2F960x544&partition=x', 400),
                          (f'/api/panel/runs/{run.name}/frames?model=chain&partition=test', 404),
                          ('/api/panel/runs/nope/frames?model=detector&partition=test', 404),
                          ('/api/panel/runs/..%2F..%2Fsplits/frames?model=detector&partition=test', 404)):
            status, response, data = server.request('GET', bad, headers=cookie)
            assert status == code, (bad, status, data)
        # The control pages may show the previews of this server, and nothing from elsewhere.
        status, response, data = server.request('GET', '/evaluate', headers=cookie)
        assert status == 200 and "img-src 'self'" in response.getheader('Content-Security-Policy')
        assert server.request('GET', f'/api/panel/runs/{run.name}', headers=cookie)[0] == 200
    finally:
        server.close()


def test_the_report_descriptor_says_how_a_box_is_written():
    plain, chained = per_frame_descriptor(chain=False, min_score=0.05), per_frame_descriptor(chain=True, min_score=0.05)
    assert plain['box'] == 'x1 y1 x2 y2 score truth' and chained['box'] == 'x1 y1 x2 y2 score truth label confidence'
    assert plain['steps'] == list(PER_FRAME_STEPS) and plain['max_boxes'] == PER_FRAME_MAX_BOXES
    assert all(math.isfinite(step) for step in plain['steps'])
