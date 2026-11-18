"""Measure a trained brain's response to a changed sign and compare ways to decode its speed, without training."""
from __future__ import annotations

import numpy as np

from nnslr_tools.brain_targets import (BRAIN_INPUTS, RATE_HZ, SWAPS, SWAP_TOLERANCE_KPH, VISION, _matching_slots, _slot_rows,
                                       _swap_slot_numbers, class_kph, learning_pieces, loss_weights, pack_inputs,
                                       sign_kph, swapped)
from nnslr_tools.brain_train import CHUNK, RIGHT_KPH, _measured, _piece, _speed_measure
from nnslr_tools.road_type_net import _torch
from nnslr_tools.road_types import ROAD_TYPES

DECODERS = ('argmax', 'mean', 'median')
WINDOWS = {'0-1s': (0, 1), '2-5s': (2, 5)}
SCALE = BRAIN_INPUTS.SIGN_KPH_SCALE
RAMP = ROAD_TYPES.index('ramp')


def speed_probabilities(model, drive: dict, store, *, chunk: int, device, states_at=()) -> dict:
    """Per tick: speed probabilities, raw trust(h), and probabilities without the sign's direct number term.

    states_at: ticks whose memory before them is also returned ('states': {tick: state or None after a cut}), so that one
    pass over a drive gives every passage of paired_replay its starting state.

    As in predict, the GRU sees all the inputs and starts again only at cuts. The ablation keeps the same GRU output
    and the map term, if the checkpoint has one. On older checkpoints it removes the sign's one-class term instead.
    """
    if chunk < 1:
        raise ValueError('chunk must be positive')
    torch = _torch()
    model.eval()
    state, probs, trust, without, states = None, [], [], [], {}
    use_bell = model.number_path == 'bell'
    wanted = sorted(set(int(at) for at in states_at))
    pieces = []
    for first, last, _learn, fresh in learning_pieces(['train'] * len(drive['frames']), chunk, cuts=drive['cuts']):
        # Splitting a piece where a state is wanted changes nothing of the GRU's run: it only stops there.
        edges = [first, *(at for at in wanted if first < at < last), last]
        pieces += [(a, b, fresh and a == first) for a, b in zip(edges, edges[1:])]
    with torch.no_grad():
        for first, last, fresh in pieces:
            if first in wanted:
                states[first] = None if fresh or state is None else state[0].clone()
            images, extra = _piece(drive, first, last, store, device)
            extra = extra.to(images.dtype)
            # The same encoder and GRU step as BrainNet.forward, but keep the memory output at every tick.
            encoded = model.encode(images.flatten(0, 1)).reshape(*images.shape[:2], -1)
            previous = None if fresh else state
            out, state = model.gru(torch.cat([encoded, extra], dim=-1), previous)
            weight = model.trust(out)
            bare = model.speed_head(out)
            if model.trust_map is not None:
                number_map = model.map_bell(extra) if use_bell else model.map_number(extra)
                bare = bare + model.trust_map(out) * number_map
            probs.append(model.speed(out, extra)[0].softmax(dim=-1).cpu().numpy())
            without.append(bare[0].softmax(dim=-1).cpu().numpy())
            trust.append(weight[0, :, 0].cpu().numpy())
    empty = np.empty((0, len(model.class_kph)), dtype=np.float32)
    return {'probs': np.concatenate(probs) if probs else empty,
            'without_sign': np.concatenate(without) if without else empty.copy(),
            'trust': np.concatenate(trust) if trust else np.empty(0, dtype=np.float32), 'states': states}


def decode(probs: np.ndarray, kph_table: np.ndarray) -> dict[str, np.ndarray]:
    """Read each row of class probabilities in km/h, keeping the first class on an argmax or a median tie."""
    probs, table = np.asarray(probs), np.asarray(kph_table)
    cdf = np.cumsum(probs, axis=-1, dtype=np.float64)
    return {'argmax': table[probs.argmax(axis=-1)], 'mean': (probs * table).sum(axis=-1),
            'median': table[(cdf < .5).sum(axis=-1).clip(max=len(table) - 1)]}


def _single_matches(signs: np.ndarray, number: int) -> np.ndarray:
    return (signs[:, 9] > 0) & (np.rint(signs[:, 8] * SCALE) == number)


def _limits(drive: dict) -> np.ndarray:
    """Known confirmed limits in km/h, -1 for no active number (including an end)."""
    signs = drive['inputs']['signs']
    return np.where((signs[:, 1] > 0) & (signs[:, 6] == 0), sign_kph(signs), -1)


def _tick_times(drive: dict) -> np.ndarray:
    """Video time on the replay grid, or the implied 4 Hz grid for drives without video timestamps."""
    return np.array([frame.get('seconds', at / RATE_HZ) for at, frame in enumerate(drive['frames'])], dtype=np.float64)


def _passage(drive: dict, passage) -> tuple[int, int, int, int]:
    """(first reading tick, confirmation tick, exclusive end, number) for one swap_passages range.

    Use the single reading and matching slots, including the read times encoded in their ages. Walk back from the
    confirmation through readings close enough to belong to the same attempt (the reader's confirmation interval).
    Without visible pre-confirmation evidence the first available tick is confirmation. Stop when the held limit ends,
    even if the stored number has not changed yet.
    """
    passage = np.asarray(passage)
    if (passage.ndim != 1 or not len(passage) or not np.issubdtype(passage.dtype, np.integer)
            or passage[0] < 0 or passage[-1] >= len(drive['frames']) or not (np.diff(passage) == 1).all()):
        raise ValueError('passage must contain consecutive tick indices in the drive')
    signs, rows = drive['inputs']['signs'], drive['inputs'].get('sign_slots')
    number = int(sign_kph(signs[passage[:1]])[0])
    limits = _limits(drive)
    confirmed = passage[limits[passage] == number]
    if number not in SWAPS or not len(confirmed):
        raise ValueError('passage has no known, swappable limit')
    confirm = int(confirmed[0])
    end = int(passage[-1]) + 1
    changed = np.flatnonzero(limits[confirm:end] != number)
    if len(changed):
        end = confirm + int(changed[0])
    single = _single_matches(signs, number)
    matching = None if rows is None else _matching_slots(_slot_rows(rows), number).any(axis=1)
    times, first = _tick_times(drive), confirm
    next_read = times[confirm]
    for at in range(confirm, -1, -1):
        if next_read - times[at] > VISION.CONFIRM_SECONDS + 1e-6:
            break
        ages = [signs[at, 10]] if single[at] else []
        if matching is not None and matching[at]:
            ages.append(rows[at, -1])
        for age in ages:
            read_at = times[at] - np.expm1(float(age) * BRAIN_INPUTS.SIGN_AGE_SCALE)
            if next_read - read_at <= VISION.CONFIRM_SECONDS + 1e-6:
                first = at
                next_read = min(next_read, read_at)
    return first, confirm, end, number


def coherent_swap(drive: dict, passage, groups: tuple[str, ...]) -> dict:
    """Swap one passage's limit, agreeing map and nearby answers, and its matching readings from the first exposure."""
    return _coherent_swap(drive, _passage(drive, passage), groups)


def _coherent_swap(drive: dict, bounds: tuple[int, int, int, int], groups: tuple[str, ...]) -> dict:
    first, confirm, end, number = bounds
    window = slice(first, end)
    copy = swapped(drive, [np.arange(confirm, end)], groups)
    inputs, original = copy['inputs'], drive['inputs']
    # Start with the original single readings: swapped also changes a matching number whose known bit is off.
    single = inputs['signs'][window, 8]
    single[:] = original['signs'][window, 8]
    single[_single_matches(original['signs'][window], number)] = SWAPS[number] / SCALE
    if first < confirm and 'sign_slots' in inputs:
        _swap_slot_numbers(inputs['sign_slots'], slice(first, confirm), number)
    # A legacy checkpoint can take fewer road_signs columns. Preserve the width the caller packed for it.
    width = sum(inputs[name].shape[-1] for name in groups)
    missing = width - drive['extra'].shape[-1] if 'extra' in drive else 0
    sizes = None
    if missing:
        sizes = {name: inputs[name].shape[-1] for name in groups}
        if 'road_signs' not in sizes or not 0 < missing < sizes['road_signs'] or missing % 2:
            raise ValueError('the drive extra width does not match its groups')
        sizes['road_signs'] -= missing
    return {**copy, 'extra': pack_inputs(inputs, groups, sizes)}


_UNSET = object()


def paired_replay(model, drive: dict, passage, store, groups: tuple[str, ...], *, device, state=_UNSET) -> dict:
    """Decode the original and counterfactual branch from one copy each of the state before the first reading.

    state: the GRU memory before the first reading (speed_probabilities' states_at), to skip running the drive up to it.

    Return aligned tick indices, decoded km/h by branch and confirmed limits (-1 when unknown), up to the first cut
    after branching. A cut on the first reading starts both branches with an empty state, like predict.
    """
    bounds = _passage(drive, passage)
    first, confirm, end, _number = bounds
    changed = _coherent_swap(drive, bounds, groups)
    cuts = drive['cuts']
    end = min((cut for cut in cuts if first < cut < end), default=end)
    table = model.class_kph.detach().cpu().numpy()
    model.eval()
    with _torch().no_grad():
        if first in cuts:
            state = None
        elif state is _UNSET:
            state = None
            for start, stop, _learn, fresh in learning_pieces(['train'] * first, CHUNK, cuts=cuts):
                images, extra = _piece(drive, start, stop, store, device)
                _, _, state = model(images, extra, None if fresh else state)
        readings = {}
        for name, branch in (('original', drive), ('swapped', changed)):
            previous = None if state is None else state.clone()
            probs = []
            for start in range(first, end, CHUNK):
                images, extra = _piece(branch, start, min(start + CHUNK, end), store, device)
                logits, _, previous = model(images, extra, previous)
                probs.append(logits[0].softmax(dim=-1).cpu().numpy())
            readings[name] = decode(np.concatenate(probs), table)
    limits, changed_limits = _limits(drive), _limits(changed)
    delta = changed_limits[confirm] - limits[confirm]
    return {**readings, 'ticks': np.arange(first, end), 'first_reading_tick': first, 'confirmation_tick': confirm,
            'original_limit': limits[first:end], 'swapped_limit': changed_limits[first:end],
            'direction': 'up' if delta > 0 else 'down' if delta < 0 else None}


def response_ratio(replay: dict, drive: dict, partition: str) -> dict:
    """Median response per decoder in the half-open 0–1 s and 2–5 s windows, within the confirmation's partition run.

    A 'said' tick has a labelled answer within SWAP_TOLERANCE_KPH of the original limit. On said ticks, a ramp label
    (drive.road_type) takes precedence over a detected lead (inputs.lead's known bit). Otherwise free road requires a
    known non-ramp type, and only those said ticks count for its ratio. Without said ticks, ramp/lead may be reported
    from the window but free road cannot. An unclassified window is 'unknown', not evidence for free road.
    """
    ticks, confirm = replay['ticks'], replay['confirmation_tick']
    delta = replay['swapped_limit'] - replay['original_limit']
    valid = (replay['original_limit'] > 0) & (replay['swapped_limit'] > 0) & (delta != 0)
    parts = np.asarray(drive['partitions'])
    boundary = np.flatnonzero(parts[confirm:] != partition)
    stop = confirm + int(boundary[0]) if len(boundary) else len(parts)
    said = ((loss_weights(drive, partition)[0][ticks] > 0) & (drive['speed'][ticks] >= 0)
            & (np.abs(class_kph(drive['speed'][ticks]) - replay['original_limit']) <= SWAP_TOLERANCE_KPH))
    types = drive.get('road_type', np.full(len(parts), -1))[ticks]
    lead = drive['inputs'].get('lead')
    has_lead = np.zeros(len(ticks), dtype=bool) if lead is None else lead[ticks, 2] > 0
    times = _tick_times(drive)
    elapsed = times[ticks] - times[confirm]
    windows = {}
    for name, (start, end) in WINDOWS.items():
        inside = (elapsed >= start) & (elapsed < end) & (ticks < stop)
        answered = inside & said
        context = answered if answered.any() else inside
        free = answered & (types >= 0) & (types != RAMP)
        kind = ('ramp' if (context & (types == RAMP)).any() else 'lead car' if (context & has_lead).any()
                else 'free road' if lead is not None and free.any() else 'unknown')
        use = (free if kind == 'free road' else inside) & valid
        ratios = {key: _median((replay['swapped'][key][use] - replay['original'][key][use]) / delta[use])
                  for key in DECODERS}
        windows[name] = {'kind': kind, 'said_frames': int(answered.sum()), 'frames': int(use.sum()), 'median_r': ratios}
    return {'route': drive['route'], 'partition': partition, 'direction': replay['direction'], 'windows': windows}


def _median(values) -> float | None:
    return float(np.median(values)) if len(values) else None


def _partition_runs(drive: dict, partition: str):
    """Yield (first, end, has_previous) of contiguous runs separated by cuts, missing ticks or partition edges."""
    parts = np.asarray(drive['partitions'])
    if not len(parts):
        return
    times = _tick_times(drive)
    gaps = ~np.isclose(np.diff(times), 1 / RATE_HZ, rtol=0, atol=1e-6)
    cuts = drive['cuts']
    edges = sorted({0, len(parts), *cuts, *(np.flatnonzero(gaps | (parts[1:] != parts[:-1])) + 1)})
    for first, last in zip(edges, edges[1:]):
        if first < last <= len(parts) and parts[first] == partition:
            yield first, last, first > 0 and first not in cuts and not gaps[first - 1]


def _steady_pairs(own: np.ndarray) -> np.ndarray:
    """Adjacent pairs in at least one 3 s window with a dashboard-speed range of at most 2 km/h."""
    length = int(3 * RATE_HZ)
    if len(own) < length:
        return np.zeros(max(0, len(own) - 1), dtype=bool)
    windows = np.lib.stride_tricks.sliding_window_view(own, length)
    starts = np.flatnonzero(np.ptp(windows, axis=1) <= RIGHT_KPH)
    edges = np.zeros(len(own), dtype=np.int64)
    edges[starts] += 1
    edges[starts + length - 1] -= 1
    return edges.cumsum()[:-1] > 0


def decoder_metrics(drives, decoded, partition: str) -> dict:
    """Error on the same labelled/measured frames as measure, plus 4 Hz steady-speed jumps and sign-change delay.

    Delay starts at a different known confirmed number (not the first number or a cancellation). Count it only within
    the requested partition; look for the start of four consecutive ticks within 2 km/h before the limit ends or a
    cut, gap, or partition boundary. Report the median of reached delays and the count that never reached.
    """
    errors = {name: {'said': [], 'truth': [], 'jumps': [], 'delays': [], 'changes': 0} for name in DECODERS}
    for drive, speeds in zip(drives, decoded, strict=True):
        asked = (loss_weights(drive, partition)[0] > 0) & _measured(drive)
        truth, limits = class_kph(drive['speed']), _limits(drive)
        own = drive.get('own_kph', np.full(len(limits), np.nan))
        for name, metrics in errors.items():
            metrics['said'].append(speeds[name][asked])
            metrics['truth'].append(truth[asked])
        for first, last, has_previous in _partition_runs(drive, partition):
            pairs = _steady_pairs(own[first:last])
            start = first if has_previous else first + 1
            changes = np.flatnonzero(limits[start:last] != limits[start - 1:last - 1]) + start
            for name, metrics in errors.items():
                speed = speeds[name]
                metrics['jumps'].extend(np.abs(np.diff(speed[first:last]))[pairs])
                for at, end in zip(changes, (*changes[1:], last)):
                    if limits[at - 1] <= 0 or limits[at] <= 0:
                        continue
                    metrics['changes'] += 1
                    near = np.abs(speed[at:end] - limits[at]) <= RIGHT_KPH
                    second = int(RATE_HZ)
                    if len(near) >= second:
                        reached = np.flatnonzero(np.lib.stride_tricks.sliding_window_view(near, second).all(axis=1))
                        if len(reached):
                            metrics['delays'].append(reached[0] / RATE_HZ)
    return {name: {**_speed_measure(np.concatenate(m['said'] or [[]]), np.concatenate(m['truth'] or [[]])),
                   'p95_jump_kph': float(np.percentile(m['jumps'], 95)) if m['jumps'] else None,
                   'delay_s': _median(m['delays']), 'changes': m['changes'],
                   'never_reached': m['changes'] - len(m['delays'])}
            for name, m in errors.items()}


def bootstrap_by_drive(values_per_drive, n: int = 2000, seed: int = 0) -> dict:
    """Passage median and percentile interval: resample whole drives, keeping every passage of each sampled drive."""
    if n < 1:
        raise ValueError('n must be positive')
    rows = values_per_drive.values() if hasattr(values_per_drive, 'values') else values_per_drive
    rows = [np.asarray(row, dtype=np.float64).ravel() for row in rows]
    rows = [row[np.isfinite(row)] for row in rows]
    rows = [row for row in rows if len(row)]
    found = {'median': None, 'ci95': None, 'drives': len(rows), 'passages': sum(len(row) for row in rows)}
    if rows:
        rng = np.random.default_rng(seed)
        estimate = [np.median(np.concatenate([rows[i] for i in rng.integers(len(rows), size=len(rows))]))
                    for _ in range(n)]
        found.update(median=_median(np.concatenate(rows)), ci95=tuple(float(x) for x in np.percentile(estimate, [2.5, 97.5])))
    return found


def decision(results: dict) -> dict:
    """Choose from results['responses'] (response_ratio rows) and results['decoders']['validation'] (decoder_metrics).

    Use the current argmax response for the training decision, separately from choosing a decoder. A missing metric
    cannot establish that a new decoder meets its guardrail, nor can extra never-reached changes count as no delay loss.
    """
    by_direction: dict[str, dict[str, list[float]]] = {'up': {}, 'down': {}}
    for row in results.get('responses', ()):
        if row['partition'] != 'validation' or row['direction'] not in by_direction:
            continue
        window = row['windows']['2-5s']
        ratio = window['median_r']['argmax']
        if window['kind'] == 'free road' and window['said_frames'] and ratio is not None:
            by_direction[row['direction']].setdefault(row['route'], []).append(ratio)
    response = {direction: bootstrap_by_drive(drives) for direction, drives in by_direction.items()}
    if all(m['median'] is not None and m['median'] >= .5 for m in response.values()):
        training = 'no training needed for this'
    elif any(m['ci95'] is not None and m['ci95'][1] < .5 for m in response.values()):
        training = 'train: coherent swap, fresh init, best checkpoint'
    else:
        training = 'need more passages'

    metrics = results.get('decoders', {}).get('validation', {})
    baseline = metrics.get('argmax', {})
    candidates = []
    for name in DECODERS[1:]:
        other = metrics.get(name, {})
        keys = ('p95_jump_kph', 'mean_error_kph', 'right', 'delay_s')
        if not all(m.get(key) is not None for m in (baseline, other) for key in keys):
            continue
        if (other['p95_jump_kph'] < baseline['p95_jump_kph']
                and other['mean_error_kph'] <= baseline['mean_error_kph'] + .5
                and other['right'] >= baseline['right'] - .01 and other['delay_s'] <= baseline['delay_s'] + .5
                and other.get('never_reached', 0) <= baseline.get('never_reached', 0)):
            candidates.append(name)
    decoder = min(candidates, key=lambda name: (metrics[name]['p95_jump_kph'], metrics[name]['mean_error_kph']),
                  default='argmax')
    return {'training': training, 'decoder': decoder, 'validation_response': response}


# The number in force, as the reader has it (#116): a reading of SITUATION_FRESH_S or newer, an older one, none read yet
# on this drive, none after the end of a limit.
SITUATION_FRESH_S = 60
NUMBER_STATES = ('number, fresh', 'number, old', 'none yet', 'after an end')
SITUATION_MIN_TICKS = 20  # a number and road type with fewer measured ticks is left out of the within-number comparison


def _number_states(drive: dict) -> np.ndarray:
    signs = drive['inputs']['signs']
    known, ended = signs[:, 1] > 0, signs[:, 6] > 0
    age = np.expm1(signs[:, 5].astype(np.float64) * BRAIN_INPUTS.SIGN_AGE_SCALE)
    return np.select([known & ~ended & (age < SITUATION_FRESH_S), known & ~ended, ~known], NUMBER_STATES[:3],
                     NUMBER_STATES[3])


def _asked(drive: dict, partition: str) -> np.ndarray:
    return (loss_weights(drive, partition)[0] > 0) & _measured(drive) & (np.asarray(drive['partitions']) == partition)


def situation_baseline(drives, *, all_ticks: bool = False) -> dict:
    """The driver's median speed in train for each number in force, and for none: what a brain that only reads the
    number would say (#116). Replay uses the measured ticks; training can use every answered train tick."""
    by = {}
    for drive in drives:
        asked = (loss_weights(drive, 'train')[0] > 0) if all_ticks else _asked(drive, 'train')
        truth, limits = class_kph(drive['speed'])[asked], _limits(drive)[asked]
        for limit, speed in zip(limits, truth):
            by.setdefault(int(limit) if limit > 0 else None, []).append(speed)
    return {key: float(np.median(values)) for key, values in by.items()}


def _type_name(at) -> str:
    return ROAD_TYPES[at] if 0 <= at < len(ROAD_TYPES) else 'unknown'


def situation_type_baseline(drives, *, all_ticks: bool = False) -> dict:
    """The driver's median speed in train for each number in force (or none) and road type, keyed 'limit|type' ('none'
    for no number): what a brain that only reads the number and knows the road type would say (#116, Astra). The road
    type is the offline label, so this reference knows more than the car does. Ticks as situation_baseline."""
    by = {}
    for drive in drives:
        asked = (loss_weights(drive, 'train')[0] > 0) if all_ticks else _asked(drive, 'train')
        truth, limits = class_kph(drive['speed'])[asked], _limits(drive)[asked]
        types = np.asarray(drive.get('road_type', np.full(len(asked), -1)))[asked]
        for limit, kind, speed in zip(limits, types, truth):
            by.setdefault(_type_key(limit, _type_name(kind)), []).append(speed)
    return {key: float(np.median(values)) for key, values in by.items()}


def _type_key(limit, kind: str) -> str:
    return f"{int(limit) if limit > 0 else 'none'}|{kind}"


def _errors(said, truth, baseline_said, cells, type_said=None) -> dict:
    error = np.abs(said - truth)
    if not len(error):
        return {'ticks': 0}
    by_place = {}
    for cell, value in zip(cells, error):
        by_place.setdefault(cell, []).append(value)
    found = {'ticks': int(len(error)), 'places': len(by_place), 'mean_error_kph': float(error.mean()),
             'right': float((error <= RIGHT_KPH).mean()), 'p90_error_kph': float(np.percentile(error, 90)),
             'mean_error_by_place_kph': float(np.mean([np.mean(values) for values in by_place.values()])),
             'baseline_mean_error_kph': float(np.abs(baseline_said - truth).mean())}
    if type_said is not None:
        found['type_baseline_mean_error_kph'] = float(np.abs(type_said - truth).mean())
    return found


def situation_groups(drive: dict, asked: np.ndarray) -> dict[str, np.ndarray]:
    """The number state, road type and lead of each selected tick, shared by the measures and the loss balance."""
    types = np.asarray(drive.get('road_type', np.full(len(asked), -1)))[asked]
    lead = drive['inputs'].get('lead')
    return {'state': _number_states(drive)[asked], 'type': np.array([_type_name(at) for at in types]),
            'lead': (np.where(lead[asked, 2] > 0, 'lead car', 'no lead detected') if lead is not None
                     else np.full(int(asked.sum()), 'unknown'))}


def situation_keys(groups: dict[str, np.ndarray]) -> np.ndarray:
    """The joint situation key of each tick, in the same form as the metrics breakdown."""
    keys = np.char.add(groups['state'].astype(str), ' | ')
    return np.char.add(np.char.add(keys, np.char.add(groups['type'].astype(str), ' | ')), groups['lead'].astype(str))


def situation_metrics(drives, said_kph, partition: str, baseline: dict, type_baseline: dict | None = None) -> dict:
    """How well it holds the situation on one partition (#116), on the measured answered ticks as measure: the error by
    the number in force (NUMBER_STATES), by road type (the stored offline label, 'unknown' without one), by lead car
    (detected or not: none detected is not a free road), and by the three together; each against the baseline (the
    driver's train median for that number). Within a number: on each number and road type, how far his mean speed is
    from his mean for that number, and how far the brain's is from its own (does it slow in a curve, on a ramp, as he
    did?), with the least-squares slope of the second on the first, weighted by ticks (1: the full difference).
    With type_baseline (situation_type_baseline) each group also has the error of the driver's train median for the
    number and road type, the number's alone where that pair was never seen in train. 'after_end' splits the ticks after
    an end of limit by the seconds since it, 'entries' is the profile around each entry into a ramp or a merge
    (_entry_profiles)."""
    rows = {'said': [], 'truth': [], 'base': [], 'state': [], 'type': [], 'lead': [], 'limit': [], 'cell': [],
            'since_end': []}
    entries = []
    for drive, said in zip(drives, said_kph, strict=True):
        asked = _asked(drive, partition)
        if not asked.any():
            continue
        limits = _limits(drive)[asked]
        rows['said'].append(np.asarray(said)[asked])
        rows['truth'].append(class_kph(drive['speed'])[asked])
        rows['base'].append(np.array([baseline.get(int(limit) if limit > 0 else None, np.nan) for limit in limits]))
        for name, values in situation_groups(drive, asked).items():
            rows[name].append(values)
        rows['since_end'].append(_seconds_since_end(drive)[asked])
        entries.extend(_entry_profiles(drive, np.asarray(said), asked))
        rows['limit'].append(limits)
        cells = drive.get('cells')
        rows['cell'].extend((drive['route'], None) if cells is None or cells[at] is None else tuple(cells[at])
                            for at in np.flatnonzero(asked))
    if not rows['said']:
        return {'ticks': 0}
    data = {key: np.concatenate(value) if key != 'cell' else value for key, value in rows.items()}
    cells = np.empty(len(data['cell']), dtype=object)
    cells[:] = data['cell']
    # A number never seen in train has no baseline: its train median for no number, else the number itself.
    data['base'] = np.where(np.isnan(data['base']), np.where(data['limit'] > 0, data['limit'], baseline.get(None, 0)),
                            data['base'])
    typed = None
    if type_baseline is not None:
        typed = np.array([type_baseline.get(_type_key(limit, kind), base)
                          for limit, kind, base in zip(data['limit'], data['type'], data['base'])])

    def errors(mask):
        return _errors(data['said'][mask], data['truth'][mask], data['base'][mask], cells[mask],
                       None if typed is None else typed[mask])

    found = {'all': errors(np.ones(len(data['said']), dtype=bool)), 'by': {}}
    for name in ('state', 'type', 'lead'):
        found['by'][name] = {str(value): errors(data[name] == value) for value in np.unique(data[name])}
    combined = situation_keys(data)
    found['by']['state, type, lead'] = {value: errors(combined == value) for value in np.unique(combined)}
    pairs = []
    for limit in np.unique(data['limit'][data['limit'] > 0]):
        same = data['limit'] == limit
        for kind in np.unique(data['type'][same]):
            group = same & (data['type'] == kind)
            if kind == 'unknown' or group.sum() < SITUATION_MIN_TICKS:
                continue
            pairs.append({'limit': int(limit), 'type': str(kind), 'ticks': int(group.sum()),
                          'driver_kph': float(data['truth'][group].mean() - data['truth'][same].mean()),
                          'brain_kph': float(data['said'][group].mean() - data['said'][same].mean())})
    found['within_number'] = {'groups': pairs, 'slope': None}
    weights = np.array([pair['ticks'] for pair in pairs], dtype=np.float64)
    driver = np.array([pair['driver_kph'] for pair in pairs])
    if len(pairs) >= 2 and (weights * driver ** 2).sum() > 0:
        brain = np.array([pair['brain_kph'] for pair in pairs])
        found['within_number']['slope'] = float((weights * driver * brain).sum() / (weights * driver ** 2).sum())
    found['after_end'] = {}
    for low, high in AFTER_END_BINS_S:
        group = (data['state'] == NUMBER_STATES[3]) & (data['since_end'] >= low) & (data['since_end'] < high)
        found['after_end'][f'{low}-{high}s' if high < np.inf else f'{low}s+'] = errors(group)
    found['entries'] = _entry_summary(entries)
    return found


# Seconds since an end of limit, in bins: does the error after an end grow with time (a memory too short) or sit at
# the start (a slow answer to the end)? (#116, Astra)
AFTER_END_BINS_S = ((0, 10), (10, 30), (30, 60), (60, 120), (120, np.inf))
# The profile around entering a ramp or a merge, seconds from the entry: does the brain slow late, or too little?
ENTRY_TYPES = ('ramp', 'merge')
ENTRY_WINDOW_S = (-20, 30)
ENTRY_PHASES_S = {'before': (-10, 0), 'first 10 s': (0, 10), 'after': (10, 30)}


def _seconds_since_end(drive: dict) -> np.ndarray:
    """At each tick after an end of limit, the seconds since the tick the end first held; nan elsewhere."""
    signs = drive['inputs']['signs']
    ended, times = (signs[:, 1] > 0) & (signs[:, 6] > 0), _tick_times(drive)
    found, start = np.full(len(ended), np.nan), None
    for at, now in enumerate(ended):
        start = (start if start is not None else times[at]) if now else None
        if now:
            found[at] = times[at] - start
    return found


def _entry_profiles(drive: dict, said: np.ndarray, asked: np.ndarray) -> list[dict]:
    """Each entry into a road type of ENTRY_TYPES (a tick of that type after a tick of another known type): the asked
    ticks within ENTRY_WINDOW_S of it, as (whole seconds from the entry, driver km/h, brain km/h)."""
    types = np.asarray(drive.get('road_type', np.full(len(asked), -1)))
    times, truth, found = _tick_times(drive), class_kph(drive['speed']), []
    for at in range(1, len(types)):
        kind = _type_name(types[at])
        if kind not in ENTRY_TYPES or types[at - 1] == types[at] or _type_name(types[at - 1]) == 'unknown':
            continue
        offset = times - times[at]
        near = asked & (offset >= ENTRY_WINDOW_S[0]) & (offset < ENTRY_WINDOW_S[1])
        if near.any():
            found.append({'type': kind, 'offset': np.floor(offset[near]).astype(int), 'driver': truth[near],
                          'brain': said[near]})
    return found


def _entry_summary(entries: list[dict]) -> dict:
    """By road type: the entries, the mean driver and brain km/h for each second from the entry, and in each phase of
    ENTRY_PHASES_S the mean signed error brain - driver (positive: the brain is faster than he was)."""
    summary = {}
    for kind in ENTRY_TYPES:
        own = [entry for entry in entries if entry['type'] == kind]
        if not own:
            continue
        offset = np.concatenate([entry['offset'] for entry in own])
        driver = np.concatenate([entry['driver'] for entry in own])
        brain = np.concatenate([entry['brain'] for entry in own])
        profile = [{'second': int(second), 'ticks': int((offset == second).sum()),
                    'driver_kph': float(driver[offset == second].mean()),
                    'brain_kph': float(brain[offset == second].mean())} for second in np.unique(offset)]
        phases = {}
        for name, (low, high) in ENTRY_PHASES_S.items():
            inside = (offset >= low) & (offset < high)
            phases[name] = ({'ticks': int(inside.sum()), 'brain_minus_driver_kph': float((brain - driver)[inside].mean()),
                             'mean_error_kph': float(np.abs(brain - driver)[inside].mean())}
                            if inside.any() else {'ticks': 0})
        summary[kind] = {'entries': len(own), 'phases': phases, 'profile': profile}
    return summary


def _number_use(drive: dict, run: dict, table: np.ndarray, partition: str) -> dict:
    """On the partition's ticks with a number in force: the raw trust(h), and how the answer moves without the sign's term."""
    held = (np.asarray(drive['partitions']) == partition) & (_limits(drive) > 0)
    if not held.any():
        return {'ticks': 0}
    with_sign, without = decode(run['probs'][held], table), decode(run['without_sign'][held], table)
    trust = run['trust'][held]
    return {'ticks': int(held.sum()), 'trust': trust.tolist(),
            'argmax_changed': int((with_sign['argmax'] != without['argmax']).sum()),
            'mean_shift_kph': np.abs(with_sign['mean'] - without['mean']).tolist()}


# The replay of a brain, beside its weights: Models › brain reads it (model_roles.brain_replay_digest).
REPLAY_FILE = 'replay-{partition}.json'


def run_replay(root, run_dir, merged_of, *, device, partition: str = 'validation', log=print) -> dict:
    """The paired replay of one trained brain on one partition (#116): no weight moves. The drives as its training read
    them (routes.json, its sign networks, its groups and sizes), one pass per drive for the decoders and the number's use,
    then each swappable passage from the memory before its first reading."""
    return run_replays(root, run_dir, merged_of, device=device, partitions=(partition,), log=log)[partition]


def run_replays(root, run_dir, merged_of, *, device, partitions=('validation', 'test'), log=print) -> dict:
    """run_replay on several partitions with the drives read once: {partition: found}."""
    import json
    from pathlib import Path

    from nnslr_tools.brain_export import load_brain
    from nnslr_tools.brain_targets import brain_sizes, dataset, swap_passages
    from nnslr_tools.brain_train import CHECKPOINT, ROUTES, _pictures
    from nnslr_tools.road_type_net import FrameStore
    from nnslr_tools.sign_networks import of_brain
    root, run_dir = Path(root), Path(run_dir)
    result = json.loads((run_dir / 'result.json').read_text(encoding='utf-8'))
    routes = json.loads((run_dir / ROUTES).read_text(encoding='utf-8'))
    model, saved = load_brain(run_dir / CHECKPOINT)
    model = model.to(device)
    groups = tuple(saved['groups'])
    drives = dataset(root, merged_of, routes, log=log, rate=RATE_HZ, need_pictures=True, types=True,
                     networks=of_brain(root, result.get('sign_networks')),
                     other_road_boxes=bool(saved.get('other_road_boxes', False)))
    baseline, type_baseline = situation_baseline(drives), situation_type_baseline(drives)
    drives = [drive for drive in drives if set(partitions) & set(drive['partitions'])]
    for drive in drives:
        drive['extra'] = pack_inputs(drive['inputs'], groups, brain_sizes(saved))
    table = model.class_kph.detach().cpu().numpy()
    every = drives
    results = {}
    for partition in partitions:
        drives = [drive for drive in every if partition in set(drive['partitions'])]
        log(f'== {partition}: {len(drives)} drives with {partition} frames ==')
        # The pictures of one partition at a time on the card (a drive's whole video at 4 Hz, about 110 kB a frame).
        store = FrameStore(_pictures(root, drives, dry_run=False, log=log), device) if model.picture else None
        results[partition] = _replay_partition(model, drives, store, groups, table, baseline, type_baseline, partition,
                                               run_dir, device=device, log=log)
        del store
        if str(device).startswith('cuda'):
            _torch().cuda.empty_cache()
    return results


def _replay_partition(model, drives, store, groups, table, baseline, type_baseline, partition, run_dir, *, device,
                      log=print) -> dict:
    """run_replays on the drives of one partition."""
    decoded, responses, uses, skipped = [], [], [], 0
    for at, drive in enumerate(drives, 1):
        bounds = []
        for passage in swap_passages(drive, partition):
            try:
                bounds.append((passage, _passage(drive, passage)))
            except ValueError:
                skipped += 1
        whole = speed_probabilities(model, drive, store, chunk=CHUNK, device=device,
                                    states_at=[found[0] for _, found in bounds])
        decoded.append(decode(whole['probs'], table))
        uses.append(_number_use(drive, whole, table, partition))
        for passage, found in bounds:
            replay = paired_replay(model, drive, passage, store, groups, device=device, state=whole['states'][found[0]])
            responses.append(response_ratio(replay, drive, partition))
        log(f'{at}/{len(drives)} {drive["route"]}: {len(bounds)} passages')
    trust = np.concatenate([np.asarray(use['trust']) for use in uses if use['ticks']] or [np.empty(0)])
    shift = np.concatenate([np.asarray(use['mean_shift_kph']) for use in uses if use['ticks']] or [np.empty(0)])
    held = sum(use['ticks'] for use in uses)
    number = {'ticks': held, 'trust_median': _median(trust), 'trust_negative_share': float((trust < 0).mean()) if held else None,
              'argmax_changed_share': sum(use.get('argmax_changed', 0) for use in uses) / held if held else None,
              'mean_shift_kph_median': _median(shift)}
    found = {'run': run_dir.name, 'partition': partition, 'passages': len(responses), 'passages_skipped': skipped,
             'number_use': number, 'responses': responses,
             'decoders': {partition: decoder_metrics(drives, decoded, partition)},
             'situation': situation_metrics(drives, [speeds['argmax'] for speeds in decoded], partition, baseline,
                                            type_baseline),
             'baseline': {str(key): value for key, value in baseline.items()}, 'type_baseline': type_baseline}
    found['decision'] = decision(found) if partition == 'validation' else None
    return found


def main(argv=None) -> int:
    import argparse
    import json
    from pathlib import Path

    from nnslr_tools.drive_review import DriveReviewStore
    from nnslr_tools.preannotate import resolve_data_root
    parser = argparse.ArgumentParser(description='Paired replay of a trained brain (#116): no training.')
    parser.add_argument('--run', required=True, help='the brain run, e.g. 2026-11-17_1935-brain-s0')
    parser.add_argument('--data-root')
    parser.add_argument('--partition', default='both', choices=('validation', 'test', 'both'))
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--out', help='the file of one partition; by default <run>/replay-<partition>.json (REPLAY_FILE), '
                                      'which Models › brain shows')
    args = parser.parse_args(argv)
    partitions = ('validation', 'test') if args.partition == 'both' else (args.partition,)
    if args.out and len(partitions) > 1:
        parser.error('--out names the file of one partition')
    torch = _torch()
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise SystemExit('no GPU found: the replay runs only on the V100s')
    root = resolve_data_root(Path(args.data_root) if args.data_root else None)
    run_dir = root / 'runs' / args.run
    if not (run_dir / 'result.json').is_file():
        raise SystemExit(f'no brain run {args.run}: {run_dir}/result.json is missing')
    store = DriveReviewStore(root)
    found = run_replays(root, run_dir, lambda route: store.merged(route, keep=False),
                        device=torch.device(args.device), partitions=partitions)
    for partition, replay in found.items():
        out = Path(args.out) if args.out else run_dir / REPLAY_FILE.format(partition=partition)
        out.parent.mkdir(parents=True, exist_ok=True)
        # Written whole and then renamed: Models never reads half a file.
        part = out.with_name(out.name + '.part')
        part.write_text(json.dumps(replay, indent=1, default=float), encoding='utf-8')
        part.replace(out)
        print(f'written {out}')
        print(json.dumps({key: replay[key] for key in ('partition', 'passages', 'passages_skipped', 'number_use',
                                                       'decoders', 'decision')}
                         | {'situation': {key: replay['situation'].get(key) for key in ('all',)}},
                         indent=1, default=float))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
