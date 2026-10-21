"""Second-opinion hints (read-only) and the "suspicious" frame rules a-d of the web review."""
from __future__ import annotations

import copy
import json
import os
import shutil
from pathlib import Path

import pytest

from nnslr_tools.review_hints import (MAX_NOTE_LENGTH, empty_stats, frame_sizes_of, load_hints,
                                      suspicious_frames)

FIXTURES = Path(__file__).parent / 'fixtures'
REAL_ROUTE = '0000011e--0c8825628a'
REAL_ROWS_FILE = FIXTURES / 'preannotations_0000011e_3-25_3-33.jsonl'
REAL_HINTS_FILE = FIXTURES / 'hints_0000011e_3-25_3-29.jsonl'
HINT_KEYS = {'type', 'value_kph', 'box', 'confidence', 'pass', 'side', 'note', 'source'}


def real_rows() -> list[dict]:
    rows = []
    for line in REAL_ROWS_FILE.read_text().splitlines():
        rows.append(json.loads(line))
    return rows


def hints_dir(root: Path, route: str = REAL_ROUTE) -> Path:
    directory = root / 'derived' / 'hints' / route
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def real_hints(root: Path) -> dict:
    shutil.copy(REAL_HINTS_FILE, hints_dir(root) / 'claude-vision.jsonl')
    by_frame, stats = load_hints(root, REAL_ROUTE, frame_sizes_of(real_rows()))
    return by_frame


def sign(**changes) -> dict:
    value = {'pass': 'A', 'type': 'maximum_speed', 'value_kph': 60, 'box': [100, 100, 120, 120],
             'confidence': 'high', 'side': 'right', 'note': 'round sign'}
    value.update(changes)
    return value


def line(key: str = '3/27', signs=None, **changes) -> str:
    segment, output = key.split('/')
    row = {'route_id': REAL_ROUTE, 'frame_key': key, 'segment_index': int(segment),
           'output_index': int(output), 'source': 'claude_vision_second_opinion', 'passes': ['A'],
           'signs': [sign()] if signs is None else signs}
    row.update(changes)
    return json.dumps(row)


# ---------- loading ----------

def test_real_hint_lines_load_with_only_the_known_fields(tmp_path):
    shutil.copy(REAL_HINTS_FILE, hints_dir(tmp_path) / 'claude-vision.jsonl')
    by_frame, stats = load_hints(tmp_path, REAL_ROUTE, frame_sizes_of(real_rows()))
    assert sorted(by_frame) == ['3/25', '3/26', '3/27', '3/28', '3/29']
    assert stats == {'files': 1, 'lines': 5, 'frames': 5, 'signs': 20, 'skipped_files': 0, 'skipped_lines': 0,
                     'skipped_signs': 0, 'unknown_frames': 0}
    first = by_frame['3/27'][0]
    assert set(first) == HINT_KEYS
    assert first['type'] == 'maximum_speed' and first['value_kph'] == 60 and first['confidence'] == 'high'
    assert first['pass'] == 'A' and first['source'] == 'claude_vision_second_opinion'
    assert len(first['box']) == 4 and first['box'][0] < first['box'][2] and first['box'][1] < first['box'][3]
    assert isinstance(first['note'], str) and first['note']


def test_malformed_lines_and_signs_are_skipped_and_counted(tmp_path):
    sizes = {'3/27': (1344, 760), '3/28': (1344, 760)}
    lines = [
        line('3/27', extra_field='ignored', signs=[sign(unknown='ignored'), sign(type='advertisement'),
                                                   sign(confidence='certain'), sign(value_kph=3), sign(value_kph=True),
                                                   sign(value_kph=60.5), sign(box=[1, 2, 3]), sign(box='0,0,10,10'),
                                                   sign(box=[2000, 900, 2100, 950]), sign(box=[5, 5, 5, 40]),
                                                   'not a sign', sign(box=[-10, 700, 30, 800], value_kph=None)]),
        'not json at all',
        '[1, 2, 3]',
        '{"route_id": "' + REAL_ROUTE + '", "frame_key": "3/28", "signs": [{"value_kph": NaN}]}',
        line('3/28', route_id='00000001--abcdef1234'),
        line('3/28', frame_key='3/x'),
        line('3/28', frame_key='../3/28'),
        line('3/28', segment_index=4),
        line('3/28', output_index=True),
        line('3/28', signs='none'),
        line('3/28', signs=[sign()] * 51),
        line('9/9'),
        '',
    ]
    directory = hints_dir(tmp_path)
    (directory / 'mixed.jsonl').write_bytes(('\n'.join(lines) + '\n').encode() + b'\xff\xfe broken utf-8\n')
    (directory / 'ignored.txt').write_text(line('3/28'))
    by_frame, stats = load_hints(tmp_path, REAL_ROUTE, sizes)
    assert sorted(by_frame) == ['3/27']
    kept = by_frame['3/27']
    assert len(kept) == 2
    assert set(kept[0]) == HINT_KEYS and kept[0]['value_kph'] == 60
    # A box partly outside the image is clipped to it.
    assert kept[1]['box'] == [0, 700, 30, 760] and kept[1]['value_kph'] is None
    assert stats['files'] == 1 and stats['lines'] == 13
    assert stats['skipped_lines'] == 11 and stats['unknown_frames'] == 1
    assert stats['skipped_signs'] == 10 and stats['signs'] == 2


def test_hint_text_is_cleaned_and_bounded(tmp_path):
    note = '<script>alert(1)</script>\x1b[31m' + 'x' * 400
    text = line('3/27', signs=[sign(note=note, side='\x00left', **{'pass': 7})], source='\n')
    (hints_dir(tmp_path) / 'notes.jsonl').write_text(text + '\n')
    by_frame, stats = load_hints(tmp_path, REAL_ROUTE, {'3/27': (1344, 760)})
    hint = by_frame['3/27'][0]
    # Kept as plain text (the page only uses textContent), without control characters.
    assert hint['note'].startswith('<script>alert(1)</script> [31m')
    assert len(hint['note']) == MAX_NOTE_LENGTH + 1 and hint['note'].endswith('…')
    assert hint['side'] == 'left' and hint['pass'] is None and hint['source'] == 'second_opinion'


def test_route_without_hints_and_invalid_route(tmp_path):
    by_frame, stats = load_hints(tmp_path, REAL_ROUTE, {'3/27': (1344, 760)})
    assert by_frame == {} and stats == empty_stats()
    hints_dir(tmp_path, '00000001--abcdef1234')
    assert load_hints(tmp_path, '00000001--abcdef1234', {}) == ({}, empty_stats())
    for route in ['not-a-route', '../' + REAL_ROUTE, REAL_ROUTE + '/..']:
        with pytest.raises(ValueError):
            load_hints(tmp_path, route, {})


def test_hints_outside_the_data_root_are_refused(tmp_path):
    root = tmp_path / 'data'
    outside = tmp_path / 'outside'
    outside.mkdir(parents=True)
    (outside / 'stolen.jsonl').write_text(line('3/27') + '\n')
    sizes = {'3/27': (1344, 760)}
    # The route folder is a symlink that leaves the data root.
    (root / 'derived' / 'hints').mkdir(parents=True)
    os.symlink(outside, root / 'derived' / 'hints' / REAL_ROUTE)
    by_frame, stats = load_hints(root, REAL_ROUTE, sizes)
    assert by_frame == {} and stats['skipped_files'] == 1 and stats['files'] == 0
    # The whole hints folder is a symlink that leaves the data root.
    (root / 'derived' / 'hints' / REAL_ROUTE).unlink()
    (root / 'derived' / 'hints').rmdir()
    (outside / REAL_ROUTE).mkdir()
    (outside / REAL_ROUTE / 'stolen.jsonl').write_text(line('3/27') + '\n')
    os.symlink(outside, root / 'derived' / 'hints')
    by_frame, stats = load_hints(root, REAL_ROUTE, sizes)
    assert by_frame == {} and stats['files'] == 0
    # A file symlink pointing outside, and a directory named *.jsonl, are skipped; a real file is read.
    (root / 'derived' / 'hints').unlink()
    directory = hints_dir(root)
    os.symlink(outside / 'stolen.jsonl', directory / 'a-link.jsonl')
    (directory / 'b-folder.jsonl').mkdir()
    (directory / 'c-real.jsonl').write_text(line('3/27', signs=[sign(value_kph=80)]) + '\n')
    by_frame, stats = load_hints(root, REAL_ROUTE, sizes)
    assert stats['skipped_files'] == 2 and stats['files'] == 1
    assert [h['value_kph'] for h in by_frame['3/27']] == [80]


def test_several_files_are_merged_per_frame(tmp_path):
    directory = hints_dir(tmp_path)
    (directory / 'a.jsonl').write_text(line('3/27', signs=[sign(value_kph=60)]) + '\n')
    (directory / 'b.jsonl').write_text(line('3/27', signs=[sign(value_kph=50, confidence='low')]) + '\n')
    by_frame, stats = load_hints(tmp_path, REAL_ROUTE, {'3/27': (1344, 760)})
    values = []
    for hint in by_frame['3/27']:
        values.append(hint['value_kph'])
    assert values == [60, 50] and stats['files'] == 2 and stats['frames'] == 1


# ---------- suspicious frames ----------

def reasons(result: dict, key: str) -> list[tuple[str, str]]:
    found = []
    for reason in result.get(key, []):
        found.append((reason['rule'], reason['text']))
    return found


def test_rules_on_real_frames(tmp_path):
    rows = real_rows()
    hints = real_hints(tmp_path)
    result = suspicious_frames(rows, hints, {}, hints_available=True)
    assert result == {
        # a) the second reading is sure of a limit, the model proposes nothing
        '3/25': [{'rule': 'a', 'text': 'seconda lettura vede un limite illeggibile, il modello niente'}],
        '3/26': [{'rule': 'a', 'text': 'seconda lettura vede 60, il modello niente'}],
        # c) both read a value, never the same
        '3/27': [{'rule': 'c', 'text': 'seconda lettura vede 60, il modello 40'}],
        '3/28': [{'rule': 'c', 'text': 'seconda lettura vede 60, il modello 40'}],
        '3/29': [{'rule': 'c', 'text': 'seconda lettura vede 60, il modello 40 e 80'}],
        # d) the previous frame has a sure hint, this one nothing at all
        '3/30': [{'rule': 'd', 'text': 'nel fotogramma precedente la seconda lettura vede 60, qui niente'}],
        # b) the model proposes a limit, the second reading sees nothing
        '3/32': [{'rule': 'b', 'text': 'il modello vede 40, la seconda lettura niente'}],
        '3/33': [{'rule': 'b', 'text': 'il modello vede 40, la seconda lettura niente'}],
    }


def test_rule_b_needs_a_second_reading_for_the_route(tmp_path):
    rows = real_rows()
    result = suspicious_frames(rows, {}, {}, hints_available=False)
    assert result == {}


def test_confirmed_frames_are_never_suspicious_and_feed_rule_d(tmp_path):
    rows = real_rows()
    hints = real_hints(tmp_path)
    confirmed = {'3/27': [], '3/32': [{'family': 'maximum_speed', 'value_kph': 40}]}
    result = suspicious_frames(rows, hints, confirmed, hints_available=True)
    assert '3/27' not in result and '3/32' not in result
    assert reasons(result, '3/31') == [('d', 'nel fotogramma successivo c’è un cartello confermato (40), qui niente')]
    # A confirmed neighbour without a speed box does not make a frame suspicious.
    confirmed = {'3/32': [{'family': 'other_sign', 'value_kph': None}]}
    result = suspicious_frames(rows, hints, confirmed, hints_available=True)
    assert '3/31' not in result


def synthetic(key: str, detections=None) -> dict:
    segment, output = key.split('/')
    return {'frame_key': key, 'segment_index': int(segment), 'output_index': int(output), 'width': 100,
            'height': 100, 'detections': detections or []}


def hint(kind='maximum_speed', value=60, confidence='high') -> dict:
    return {'type': kind, 'value_kph': value, 'box': [1, 1, 9, 9], 'confidence': confidence, 'pass': 'A',
            'side': None, 'note': None, 'source': 'test'}


def proposal(family='maximum_speed', value=40) -> dict:
    return {'family': family, 'value_kph': value, 'bbox_xyxy': [1, 1, 9, 9], 'model_label': 'x', 'score': 0.5}


def test_rule_a_only_for_sure_hints_and_speed_proposals():
    rows = [synthetic('0/0'), synthetic('0/1'), synthetic('0/2', [proposal('other_sign', None)]),
            synthetic('0/3', [proposal('road_marking_candidate', 60)]), synthetic('0/4'),
            synthetic('0/5', [proposal('cancellation', None)])]
    hints = {'0/0': [hint(confidence='low')], '0/1': [hint('unsure', None, 'medium')],
             '0/2': [hint('zone', 30, 'medium')], '0/3': [hint('road_marking', 60, 'high')],
             '0/4': [hint('zone_end', None, 'high'), hint('cancellation', 60, 'high')],
             '0/5': [hint('cancellation', None, 'high')]}
    result = suspicious_frames(rows, hints, {}, hints_available=True)
    assert '0/0' not in result and '0/1' not in result and '0/5' not in result
    # other_sign and the road number OCR are no speed proposal of the model.
    assert reasons(result, '0/2') == [('a', 'seconda lettura vede zona 30, il modello niente')]
    assert reasons(result, '0/3') == [('a', 'seconda lettura vede numero a terra 60, il modello niente')]
    assert reasons(result, '0/4') == [('a', 'seconda lettura vede fine zona e fine limite 60, il modello niente')]


def test_rule_c_needs_values_with_nothing_in_common():
    rows = [synthetic('0/0', [proposal(value=40), proposal(value=60)]), synthetic('0/1', [proposal(value=40)]),
            synthetic('0/2', [proposal(value=None)])]
    hints = {'0/0': [hint(value=60)], '0/1': [hint(value=None, confidence='low'), hint(value=50, confidence='low')],
             '0/2': [hint(value=50)]}
    result = suspicious_frames(rows, hints, {}, hints_available=True)
    assert '0/0' not in result and '0/2' not in result
    assert reasons(result, '0/1') == [('c', 'seconda lettura vede 50, il modello 40')]


def test_rule_d_crosses_to_the_adjacent_segment_only():
    rows = [synthetic('1/59'), synthetic('2/0'), synthetic('2/1'), synthetic('4/0'), synthetic('5/0')]
    confirmed = {'1/59': [{'family': 'zone', 'value_kph': 30}], '2/1': [], '5/0': [{'family': 'cancellation'}]}
    hints = {'2/1': [hint(value=80, confidence='medium')]}
    result = suspicious_frames(rows, hints, confirmed, hints_available=True)
    assert reasons(result, '2/0') == [
        ('d', 'nel fotogramma precedente c’è un cartello confermato (zona 30), qui niente'),
        ('d', 'nel fotogramma successivo c’è un cartello confermato (…), qui niente'.replace('(…)', '(fine limite)'))
    ] or reasons(result, '2/0')[0][1].startswith('nel fotogramma precedente')
    # 2/1 is confirmed without boxes: it hides its own hint from rule d; segment 4 is not next to 2.
    assert reasons(result, '2/0') == [('d', 'nel fotogramma precedente c’è un cartello confermato (zona 30), qui niente')]
    assert reasons(result, '4/0') == [('d', 'nel fotogramma successivo c’è un cartello confermato (fine limite), qui niente')]
    assert '1/59' not in result and '5/0' not in result


def test_neighbour_hint_and_malformed_rows_do_not_break_the_rules():
    rows = [synthetic('0/0'), synthetic('0/1'), {'frame_key': None}, 'not a row', synthetic('0/2', 'bad detections')]
    hints = {'0/1': [hint(value=50, confidence='medium')]}
    result = suspicious_frames(copy.deepcopy(rows), hints, {}, hints_available=True)
    assert reasons(result, '0/0') == [('d', 'nel fotogramma successivo la seconda lettura vede 50, qui niente')]
    assert reasons(result, '0/1') == [('a', 'seconda lettura vede 50, il modello niente')]
    assert reasons(result, '0/2') == [('d', 'nel fotogramma precedente la seconda lettura vede 50, qui niente')]
