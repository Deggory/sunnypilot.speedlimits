"""The brain's training: a person's labels into the weights, and what it is worth on sites it never learned. Started only on
request (nnslr train-brain); a dry run is two steps on the CPU, nothing kept but its own folder.

An epoch goes drive after drive in a random order, each drive in its order with the state carried from piece to piece, so
that approach, crossing and exit are one sequence and the network sees only the past, as on the car. The pieces move their
cuts every epoch, and the gradient stops at each of them. Every epoch the drives get another clock and half of their numbers
swapped, and those with a plate of hours go through once more at night (brain_targets.epoch_drives): the measure has only
the real hour and the real numbers.

The last epoch is what is kept. Validation is a few stretches of two drives today: choosing on it rewards the epoch that
remembers those roads, the opposite of what the swap asks. It is shown every epoch and it chooses nothing.

A picture of test or validation runs in no training step and in no memory of one: the training skips those frames, and the
memory starts again after them (brain_targets.learning_pieces). A frame of the buffer runs under no_grad, only to keep the
memory going. What does cross from a held-out stretch into train is the number read there, as an input, the way the car
carries it: never its picture, never its answer.

The measure (measure), at the end on train, validation and test, frames with an answer only; train against test is the gap
between roads seen and not seen:

- speed: right (within RIGHT_KPH of the answer: at a km/h a class nobody says the dashboard's speed to the km/h, and on the
  5 km/h table of the brains before it is the same as the class itself), and the mean error in km/h, on all of them and on
  those without gas in the last second: the frames where the driver gives nothing away, so a network that follows the pedal
  shows here;
- speeds_said: every speed the network said there: one alone is a network that learned only how often each speed comes;
- one_speed: the error of the speed most frames have, said everywhere: what the network has to beat;
- follow_the_sign: on the frames with a number in force, the error of that number and the network's own there: the other
  thing it has to beat, or it adds nothing to the car's rule;
- far_from_own_speed: the same on the frames where the car's speed is OWN_SPEED_GAP_KPH or more from his label, and how
  many of them the network puts nearer the car's speed than the label. Leaving pedals and speed out of the inputs is not
  enough: the video shows how fast the car goes, and at the end of works the label and the car's speed rise together, so
  a network that follows the car looks right there. Here following the car is the wrong answer;
- in_works: the same inside the marks;
- in_blocks: the same inside the blocks of Speed review with a speed, the only frames whose answer is a person's correction
  and not the speed he drove: everywhere else his speed is the answer, and a network that only knows how fast the car goes
  is right there. copy_own_speed: of those frames, where the car's speed is more than RIGHT_KPH from the block's, the share
  where the network said the car's speed (within RIGHT_KPH) and not the block's;
- works: precision and recall of works_head, on the frames with an answer of its own, None where the partition has no
  frame to tell them by: with no "no" a constant yes would read 1.0;
- road: the same for each kind of road (road_head), and in_road the speed inside each kind's marks;
- types: the share of frames on which type_head names the road type right, on all and per type, None without the head;
  in_type the speed on each type, the head or not;
- applicability: whose road a speed sign is of, on the slots of sign_slots a reviewed box answers
  (brain_targets.applicability_answers), every frame with one (a reviewed frame is not one a second): precision and
  recall of "of another road" of applicability_head (head; None without it), with how many slots say our road and
  how many another. Only against the reviews: no applicability network reads the brain's signs since 15 Nov 2026 (#90);
- sites: per mark of the partition, how long after his end tap the speed reaches (within RIGHT_KPH) the speed he drove
  after it, and how much of the mark was above his label: said too early, the end of works that is not there. The speed
  after is the 90th percentile of his answers in END_WINDOW_S after the tap, the one he reaches, not his first answer there,
  which is still on the way up; there is an end only where it is more than RIGHT_KPH above his median inside the mark.

predict goes through a whole drive as on the car, so a held-out stretch starts from the memory of the train frames before it,
and a network that only holds its last speed scores well there. So the result has each partition twice: as on the car, and
cold, with the memory starting again where the stretch out of train begins.

test_swapped and validation_swapped ask the network what the swap asks in training, on roads it never saw: the frames whose
passage says its number (brain_targets.swap_passages: within SWAP_TOLERANCE_KPH), with that number changed to the other of its
family and his answers moved as much. A network deaf to the sign is still right on the frames that go to the speed it says, so
moved counts the answers that changed by more than RIGHT_KPH: the network reads the sign when they follow it.

validation_no_map and test_no_map, of a brain that learns the map's limit: the same partitions with no map on any frame
(brain_targets.without_map), as past a border with no map downloaded. Beside validation and test they say what the map
is worth to the brain and what is left of it without the map.

The brain steps at RATE_HZ, every reading of the car (brain_targets): its pictures are those vision.BrainInput makes on the
car, kept while the signs are read. Each measure counts one frame a second, the frames of the old 1 Hz brains, so that the
numbers of a brain of today and of one before read the same.

No limit ahead since 12 Nov 2026: the brain says the speed to hold and nothing else (brain_net), so a training has no
next limit, no metres to it, no weight for them (next_balance) and no measure of them. The result.json of a brain of 7
to 11 Nov 2026 keeps its 'ahead' measures, which nothing here makes any more. predict still gives the next limit and the
metres of such a brain, for the check of its export along a drive (brain_export.drive_rollout).

With types (the usual since 8 Nov 2026, his rule) only the seconds with a road type teach anything (brain_targets'
types): the measure is on those too, so a brain with types and one without measure different frames.

The comparison he asked for is the same training twice, with works and with works=False, on the same split. His pedals
and the car's speed are never inputs (brain_targets.LEFT_OUT, since 10 Nov 2026; off by default since 8 Nov): the answer is
his own speed, and with the car's among the inputs the network learns to say it back, on the car its own. Nor are the
metres to the sign ahead (sign_distance, LEFT_OUT since 12 Nov 2026): the brain gets the boxes of the signs (sign_boxes).

The crossings (speed_review.EVENT_KINDS, since 17 Nov 2026, #112): a person or a vehicle crossing our path is a kind of
road of Speed review, marked as every kind (by a person, or by the road labeler once a head of that kind passed its
gate), and road_head learns it as the others (road and in_road measure it). A mark is a few seconds in hours of driving:
road_pos_weight lifts its frames, per kind, by the frames of train answered no over those answered yes, at most
brain_net.ROAD_EVENT_POS_WEIGHT_MAX; every other kind keeps 1, and its loss is what it was. The log says the weights and
result.json keeps them (road_pos_weight). The road users' head of 12 to 17 Nov 2026 (users_head, a COCO detector's
answers and a rule of crossings) went with it: a brain of those days still loads, its head left out.

Whose road a sign is of (applicability, the usual since 15 Nov 2026, #89; --no-applicability, the comparison):
applicability_head on the memory learns it from the reviewed boxes (brain_net), and the brain is given the boxes of
every speed sign read, those the applicability network gives to another road too (the checkpoint's other_road_boxes,
which the export puts in the contract for the car). Without it the brain gets the signs taken, as every brain before,
and only the applicability network is measured. Stage 1: the head goes to no car, the applicability network still
decides which signs are taken; the result says which of the two tells a sign of another road better.

A training can go on from an earlier brain (brain_start, nnslr train-brain --init-from, Train › Brain's "Start from"):
its weights instead of ImageNet's and random ones, the rest of the training the same. Only a brain stepped as this one
(rate, picture, pool, sizes) on the same split; an input it lacks starts at zero, so the new one starts as the old.

The number's path (number_path, since 16 Nov 2026, #104; brain_net's text): a training gives the number in force to the
answer as a bell around it ('bell', the usual one), or on its class alone as every brain before ('class': nnslr
train-brain --number-path class, Train › Brain's Number, the comparison). The checkpoint and result.json keep the word.
Going on from a brain of the other path, the trust in the number starts again (at zero for 'bell', so the new brain
starts from what the start's speed_head says); from one of the same path it is kept. Whether the path was learnt is read
from two things in result.json: moved of test_swapped, and trust (trust_size: the L2 norm of the weights of the trust
layer and its bias, at the start and at the epoch kept; trust(h) lies within the bias plus or minus the L1 norm), which
the training says together in the last line it logs.
"""
from __future__ import annotations

import hashlib
import json
import random
import time
from functools import partial
from pathlib import Path

import numpy as np

from nnslr_tools import road_type_data as data
from nnslr_tools.brain_net import (APPLICABILITY_POS_WEIGHT_MAX, NUMBER_PATHS, ROAD_EVENT_POS_WEIGHT_MAX, brain_loss,
                                   build_brain_net, number_path_of, outputs)
from nnslr_tools.brain_targets import (INPUT_SIZES, LEFT_OUT, MAP_GROUP, METRES_SCALE, NO_TYPE, NS, OPTIONAL, RATE_HZ,
                                       ROAD_SIGN_KINDS, brain_sizes,
                                       applicability_counts, applicability_counts_text, applicability_weights,
                                       check_drives, class_kph, dataset, epoch_drives, hours_in_train, inputs_sha256,
                                       learning_pieces, loss_weights, night_frames, pack_inputs, plates_in_train, read_split,
                                       road_weights, sign_kph, swap_passages, swapped, training_counts,
                                       training_counts_text, type_weights, wet_in_train, without_map)
from nnslr_tools.road_types import ROAD_TYPES
from nnslr_tools.speed_review import EVENT_KINDS
from nnslr_tools.road_type_net import FEATURES, FrameStore, HIDDEN, LEARNING_RATE, WEIGHT_DECAY, _augment, _print, _torch
from nnslr_tools.road_type_signs import load_pictures

EPOCHS = 40
CHUNK = int(64 * RATE_HZ)  # a piece of 64 s, as the 64 frames of the 1 Hz brains
CLIP = 5.0
CHECKPOINT = 'brain-last.pt'
RESULT = 'result.json'
# The drives a training took (dataset's, before the dry run's cut), written first: scripts/run_brain.sh keeps the dry
# run's for the whole run, so that every step and every seed has the same drives whatever drive arrives meanwhile.
ROUTES = 'routes.json'
END_WINDOW_S = 60
OWN_SPEED_GAP_KPH = 20
RIGHT_KPH = 2  # a speed said this near the answer is right (the site says "within 2 km/h")
AFTER_PERCENTILE = 90  # the speed he reaches after the end of works (_site)
DRY_FRAMES = 12
DRY_STEPS = 2
LEARNED_INPUTS = tuple(name for name in INPUT_SIZES if name not in LEFT_OUT and name not in OPTIONAL)
NUMBER_PATH = NUMBER_PATHS[1]  # a training of today: the number in force as a bell (brain_net, #104)
# The layers that say how far a number is believed, the sign's and the map's: what they multiply is the number path's.
NUMBER_TRUSTS = ('trust.', 'trust_map.')


def learned_inputs(map_limit: bool = False) -> tuple[str, ...]:
    """The groups a brain learns from: LEARNED_INPUTS, and after them the optional ones asked for (map_limit: the map's
    limit, #85)."""
    return LEARNED_INPUTS + (('map_limit',) if map_limit else ())


def _piece(drive: dict, first: int, last: int, store: FrameStore, device):
    torch = _torch()
    images = store.batch([(drive['route'], list(range(first, last)))])
    return images, torch.from_numpy(drive['extra'][first:last])[None].to(device)


def _same_augment(torch, images, seed: int, device):
    """_augment with the numbers of seed: one look for a whole drive, since the car never sees the light jump between two
    pieces. Only the centre of the contrast is the piece's own, its mean light: a step far smaller than the look."""
    with torch.random.fork_rng(devices=[device] if device.type == 'cuda' else []):
        torch.manual_seed(seed)
        return _augment(torch, images)


def train_epoch(model, optimizer, drives: list[dict], store: FrameStore, *, chunk: int, rng: random.Random, device,
                augment: bool = True, steps: int | None = None, road_pos_weight=None, sign_pos_weight=None) -> float:
    """One pass over the train frames: the mean error of its steps. road_pos_weight: road_pos_weight's, a tensor on
    device (kinds,) in the order of the kinds, or None (every kind 1). sign_pos_weight: applicability_pos_weight's, a
    scalar tensor on device, or None."""
    torch = _torch()
    model.train()
    errors = []
    for drive in rng.sample(drives, len(drives)):
        speed_weight, works_weight = loss_weights(drive, 'train')
        if model.works_head is None:
            works_weight = np.zeros_like(works_weight)
        road = road_weights(drive, 'train') if getattr(model, 'road_head', None) is not None else None
        typed = type_weights(drive, 'train') if getattr(model, 'type_head', None) is not None else None
        whose = applicability_weights(drive, 'train') if getattr(model, 'applicability_head', None) is not None else None
        look, state = rng.getrandbits(32), None
        for first, last, learn, fresh in learning_pieces(drive['partitions'], chunk, rng.randrange(chunk), drive['cuts']):
            state = None if fresh else state
            images, extra = _piece(drive, first, last, store, device)
            images = _same_augment(torch, images, look, device) if augment else images
            answers = speed_weight[first:last].sum() + works_weight[first:last].sum()
            if road is not None:
                answers += road[first:last].sum()
            if typed is not None:
                answers += typed[first:last].sum()
            if whose is not None:
                answers += whose[first:last].sum()
            if not learn or not answers or (steps is not None and len(errors) >= steps):
                with torch.no_grad():
                    _speed, _works, state = model(images, extra, state)
                continue

            def piece(values):
                return torch.from_numpy(values[first:last])[None].to(device)

            asked = {'road': road is not None, 'types': typed is not None, 'applicability': whose is not None}
            said = outputs(model(images, extra, state, **asked), **asked)
            speed_logits, works_logit, state = said['speed'], said['works'], said['state']
            kinds = None if road is None else (said['road'], piece(drive['road']), piece(road), road_pos_weight)
            types = None if typed is None else (said['types'], piece(drive['road_type']), piece(typed))
            signs = None if whose is None else (said['applicability'], piece(drive['applicability']), piece(whose),
                                                sign_pos_weight)
            loss = brain_loss(speed_logits, works_logit,
                              *(piece(values) for values in (drive['speed'], speed_weight, drive['works'], works_weight)),
                              frames=chunk, kph=model.class_kph, road=kinds, types=types, applicability=signs)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            optimizer.step()
            state = state.detach()
            errors.append(float(loss.detach()))
    return float(np.mean(errors)) if errors else 0.0


def _stretches_out_of_train(partitions: list[str]) -> set[int]:
    return {at for at in range(1, len(partitions)) if partitions[at] != 'train' and partitions[at - 1] == 'train'}


def predict(model, drive: dict, store: FrameStore, *, chunk: int, device, cold: bool = False) -> dict:
    """What the network says along a whole drive, as on the car: {'kph': per frame, the km/h of its class in the network's
    own table (class_kph), 'works': chance per frame, or None, 'road': chance per frame and kind of road, or None, 'types':
    chance per frame and road type (road_types.ROAD_TYPES), or None, 'applicability': the chance that the speed sign of each slot of sign_slots is of another road (frames,
    columns), or None}; a network with the limit ahead (a brain of 7 to 11
    Nov 2026, loaded from its checkpoint: brain_export.load_brain) also 'next' (a class of that table per frame, 'none'
    for none: the last one, len(class_kph)) and 'metres'. cold: the memory also starts again where a stretch out of
    train begins."""
    torch = _torch()
    model.eval()
    asked = {'ahead': getattr(model, 'ahead', False), 'road': getattr(model, 'road_head', None) is not None,
             'types': getattr(model, 'type_head', None) is not None,
             'applicability': getattr(model, 'applicability_head', None) is not None}
    cuts = drive['cuts'] | (_stretches_out_of_train(drive['partitions']) if cold else set())
    state, speed, works, road, types, following, metres = None, [], [], [], [], [], []
    whose = []
    with torch.no_grad():
        for first, last, _learn, fresh in learning_pieces(['train'] * len(drive['frames']), chunk, cuts=cuts):
            state = None if fresh else state
            images, extra = _piece(drive, first, last, store, device)
            said = outputs(model(images, extra, state, **asked), **asked)
            state = said['state']
            if asked['ahead']:
                following.append(said['next'][0].argmax(dim=-1).cpu().numpy())
                metres.append(said['metres'][0].clamp_min(0).cpu().numpy() * METRES_SCALE)
            if asked['road']:
                road.append(torch.sigmoid(said['road'][0]).cpu().numpy())
            if asked['types']:
                types.append(torch.softmax(said['types'][0], dim=-1).cpu().numpy())
            if asked['applicability']:
                whose.append(torch.sigmoid(said['applicability'][0]).cpu().numpy())
            speed.append(said['speed'][0].argmax(dim=-1).cpu().numpy())
            if said['works'] is not None:
                works.append(torch.sigmoid(said['works'][0]).cpu().numpy())
    table = np.rint(model.class_kph.detach().cpu().numpy()).astype(np.int64)
    found = {'kph': table[np.concatenate(speed)], 'works': np.concatenate(works) if works else None,
             'road': np.concatenate(road) if road else None, 'types': np.concatenate(types) if types else None,
             'applicability': np.concatenate(whose) if whose else None}
    ahead = asked['ahead']
    if ahead:
        found['next'], found['metres'], found['none'] = np.concatenate(following), np.concatenate(metres), len(table)
    return found


def _speed_measure(said: np.ndarray, truth: np.ndarray) -> dict:
    if not len(truth):
        return {'frames': 0, 'right': None, 'mean_error_kph': None}
    error = np.abs(said - truth)
    return {'frames': int(len(truth)), 'right': float((error <= RIGHT_KPH).mean()), 'mean_error_kph': float(error.mean())}


def _one_speed(truth: np.ndarray) -> dict:
    """The speed most frames have, said everywhere."""
    if not len(truth):
        return {'kph': None, 'mean_error_kph': None}
    values, counts = np.unique(truth, return_counts=True)
    kph = values[counts.argmax()]
    return {'kph': int(kph), 'mean_error_kph': float(np.abs(truth - kph).mean())}


def _follow_the_sign(said: np.ndarray, truth: np.ndarray, number: np.ndarray) -> dict:
    """The frames with a number in force (-1: none): that number said there, and the network on the same frames."""
    held = number >= 0
    return {'sign': _speed_measure(number[held], truth[held]), 'network': _speed_measure(said[held], truth[held])}


def _far_from_own_speed(said: np.ndarray, truth: np.ndarray, own: np.ndarray) -> dict:
    """The frames where following the car is the wrong answer. An unknown speed (NaN) is not among them."""
    far = np.abs(own - truth) >= OWN_SPEED_GAP_KPH
    said, truth, own = said[far], truth[far], own[far]
    nearer = float((np.abs(said - own) < np.abs(said - truth)).mean()) if far.any() else None
    return {**_speed_measure(said, truth), 'nearer_own_speed': nearer}


def _in_blocks(said: np.ndarray, truth: np.ndarray, own: np.ndarray) -> dict:
    """The frames inside a block with a speed: the network against the block's, and how often it said the car's instead,
    where the two are apart. An unknown speed of the car (NaN) is not among those."""
    apart = np.abs(own - truth) > RIGHT_KPH
    copied = (np.abs(said - own) <= RIGHT_KPH) & (np.abs(said - truth) > RIGHT_KPH)
    return {**_speed_measure(said, truth), 'copy_own_speed': float(copied[apart].mean()) if apart.any() else None}


def _site(drive: dict, said: dict, mark: dict, truth: np.ndarray, asked: np.ndarray) -> dict:
    """A mark: the frames inside it above his label, and how long after its end the speed reaches the one he drove after it
    (see the module's text)."""
    ns = drive['ns']
    inside = (ns >= mark['start_ns']) & (ns < mark['end_ns']) & asked
    window = (ns >= mark['end_ns']) & (ns < mark['end_ns'] + END_WINDOW_S * NS)
    after = window & asked
    site = {'route': drive['route'], 'frames': int(inside.sum()),
            'above_label': float((said['kph'][inside] > truth[inside]).mean()) if inside.any() else None, 'end_delay_s': None}
    if after.any() and inside.any():
        reached_kph = float(np.percentile(truth[after], AFTER_PERCENTILE))
        if reached_kph > float(np.median(truth[inside])) + RIGHT_KPH:
            reached = np.flatnonzero(window & (said['kph'] >= reached_kph - RIGHT_KPH))
            if len(reached):
                site['end_delay_s'] = float((ns[reached[0]] - mark['end_ns']) / NS)
    return site


def _in_force(signs: np.ndarray) -> np.ndarray:
    return (signs[:, 1] > 0) & (signs[:, 6] == 0)


def _measured(drive: dict) -> np.ndarray:
    """The frames a measure counts: one a second."""
    return drive.get('measured', np.ones(len(drive['frames']), dtype=bool))


def measure(drives: list[dict], answers: list[dict], partition: str) -> dict:
    """What the network is worth on one partition: see the module's text."""
    said_all, truth_all, own_all, number_all, free, works_frames, blocked, sites = [], [], [], [], [], [], [], []
    hits = {'true_positive': 0, 'false_positive': 0, 'false_negative': 0}
    told = {'yes': 0, 'no': 0}
    kinds = next((drive['road_kinds'] for drive in drives if drive.get('road_kinds')), [])
    road_hits = {kind: ({'true_positive': 0, 'false_positive': 0, 'false_negative': 0}, {'yes': 0, 'no': 0}) for kind in kinds}
    road_frames: dict[str, list] = {kind: [] for kind in kinds}
    type_all, type_said, type_truth = [], [], []
    for drive, said in zip(drives, answers):
        speed_weight, works_weight = loss_weights(drive, partition)
        asked = (speed_weight > 0) & _measured(drive)
        type_all.append(drive['road_type'][asked] if 'road_type' in drive else np.full(int(asked.sum()), -1))
        typed = type_weights(drive, partition)
        if typed is not None and said.get('types') is not None:
            known = (typed > 0) & _measured(drive)
            type_said.append(said['types'][known].argmax(axis=-1))
            type_truth.append(drive['road_type'][known])
        road_weight = road_weights(drive, partition)
        for column, kind in enumerate(kinds):
            if road_weight is None:
                road_frames[kind].append(np.zeros(int(asked.sum()), dtype=bool))
                continue
            road_frames[kind].append(drive['road'][asked, column] > 0)
            if said.get('road') is not None:
                _count(said['road'][:, column], drive['road'][:, column], (road_weight[:, column] > 0) & _measured(drive),
                       *road_hits[kind])
        truth = class_kph(drive['speed'])
        said_all.append(said['kph'][asked])
        truth_all.append(truth[asked])
        own_all.append(drive['own_kph'][asked])
        signs = drive['inputs']['signs'][asked]
        number_all.append(np.where(_in_force(signs), sign_kph(signs), -1))
        free.append(drive['inputs']['pedals'][asked, 0] == 0)
        works_frames.append(drive['works'][asked] > 0)
        blocked.append(drive['blocked'][asked] if 'blocked' in drive else np.zeros(int(asked.sum()), dtype=bool))
        if said['works'] is not None:
            _count(said['works'], drive['works'], (works_weight > 0) & _measured(drive), hits, told)
        held = np.array([name == partition for name in drive['partitions']])
        for mark in drive['marks']:
            if held[(drive['ns'] >= mark['start_ns']) & (drive['ns'] < mark['end_ns'])].any():
                sites.append(_site(drive, said, mark, truth, (drive['speed_weight'] > 0) & _measured(drive)))
    said_all, truth_all, own_all, number_all = (np.concatenate(part or [[]]) for part in
                                                (said_all, truth_all, own_all, number_all))
    free, works_frames, blocked = (np.concatenate(part or [[]]).astype(bool) for part in (free, works_frames, blocked))
    road_frames = {kind: np.concatenate(part or [[]]).astype(bool) for kind, part in road_frames.items()}
    type_all = np.concatenate(type_all or [[]]).astype(np.int64)
    return {'speed': _speed_measure(said_all, truth_all), 'speeds_said': sorted({int(kph) for kph in said_all}),
            'one_speed': _one_speed(truth_all), 'follow_the_sign': _follow_the_sign(said_all, truth_all, number_all),
            'no_gas': _speed_measure(said_all[free], truth_all[free]),
            'far_from_own_speed': _far_from_own_speed(said_all, truth_all, own_all),
            'in_works': _speed_measure(said_all[works_frames], truth_all[works_frames]),
            'in_blocks': _in_blocks(said_all[blocked], truth_all[blocked], own_all[blocked]),
            'works': _told(hits, told), 'sites': sites,
            'road': {kind: _told(*road_hits[kind]) for kind in kinds},
            'in_road': {kind: _speed_measure(said_all[road_frames[kind]], truth_all[road_frames[kind]]) for kind in kinds},
            'types': _types_measure(type_said, type_truth),
            'in_type': {kind: _speed_measure(said_all[type_all == column], truth_all[type_all == column])
                        for column, kind in enumerate(ROAD_TYPES) if (type_all == column).any()},
            'applicability': _applicability_measure(drives, answers, partition)}


def _applicability_measure(drives: list[dict], answers: list[dict], partition: str) -> dict | None:
    """Whose road a speed sign is of, on the slots of sign_slots a reviewed box answers in a partition, every frame
    with one: {'ours', 'another': the columns of each; 'head': precision and recall of "of another road" as
    applicability_head says it (_told), None for a network without the head}. Only against the reviews (#90). None for
    drives without those answers."""
    if not any('applicability_weight' in drive for drive in drives):
        return None
    head = ({'true_positive': 0, 'false_positive': 0, 'false_negative': 0}, {'yes': 0, 'no': 0})
    ours = another = 0
    learned = False
    for drive, said in zip(drives, answers):
        weight = applicability_weights(drive, partition)
        if weight is None:
            continue
        known = weight > 0
        real = drive['applicability'][known] > 0
        another, ours = another + int(real.sum()), ours + int((~real).sum())
        if said.get('applicability') is not None:
            learned = True
            _count(said['applicability'], drive['applicability'], known, *head)
    return {'ours': ours, 'another': another, 'head': _told(*head) if learned else None}


def applicability_pos_weight(drives: list[dict]) -> float | None:
    """How much a sign of another road is lifted in the loss: the answered slots of train of our road over those of
    another, between 1 and APPLICABILITY_POS_WEIGHT_MAX. None with no sign of another road in train, or none of ours:
    one side alone teaches nothing (_applicability_needs_both)."""
    counts = applicability_counts(drives)['partitions']['train']
    if not counts['another'] or not counts['ours']:
        return None
    return float(np.clip(counts['ours'] / counts['another'], 1.0, APPLICABILITY_POS_WEIGHT_MAX))


def _applicability_needs_both(drives: list[dict], log) -> None:
    """_works_needs_both for whose road a sign is of: with only signs of our road in train, or only of another, or none,
    the task leaves the loss of train: one side alone would teach a constant answer."""
    counts = applicability_counts(drives)['partitions']['train']
    if counts['another'] and counts['ours']:
        return
    log('No reviewed speed sign in train: whose road a sign is of teaches nothing yet.'
        if not counts['another'] and not counts['ours'] else
        f"Whose road a sign is of, in train: {counts['ours']} of our road and {counts['another']} of another, one side "
        'alone teaches nothing: left out.')
    for drive in drives:
        if 'applicability_weight' in drive:
            drive['applicability_weight'] = drive['applicability_weight'] * (np.array(drive['partitions']) != 'train')[:, None]


def _types_measure(said: list[np.ndarray], truth: list[np.ndarray]) -> dict | None:
    """type_head on the frames with a road type: the share it names right, on all and per type; None for a network
    without it."""
    if not said:
        return None
    said, truth = np.concatenate(said), np.concatenate(truth)
    per_type = {kind: {'frames': int((truth == column).sum()),
                       'right': float((said[truth == column] == column).mean()) if (truth == column).any() else None}
                for column, kind in enumerate(ROAD_TYPES) if (truth == column).any()}
    return {'frames': int(len(truth)), 'right': float((said == truth).mean()) if len(truth) else None, 'per_type': per_type}


def _types_words(types: dict | None) -> str:
    """Of an epoch line: " val_types 0.8123 val_type_town 0.9100 ...", the road type named right on validation, on all
    and on each type with frames there; nothing for a network without type_head or a validation without a road type."""
    if not types or types['right'] is None:
        return ''
    return f' val_types {types["right"]:.4f}' + ''.join(f' val_type_{kind} {one["right"]:.4f}'
                                                     for kind, one in types['per_type'].items() if one['right'] is not None)


def _applicability_words(found: dict | None) -> str:
    """Of an epoch line: " val_another_road_recall 0.7500 val_another_road_precision 0.6000 (4 signs)", what
    applicability_head finds of the signs of another road of validation; nothing for a network without the head or a
    validation without such a sign."""
    head = (found or {}).get('head')
    if not head or head['recall'] is None:
        return ''
    precision = '-' if head['precision'] is None else f"{head['precision']:.4f}"
    return (f" val_another_road_recall {head['recall']:.4f} val_another_road_precision {precision} "
            f"({found['another']} sign{'' if found['another'] == 1 else 's'})")


def _count(chance: np.ndarray, answer: np.ndarray, known: np.ndarray, hits: dict, told: dict) -> None:
    """A head of yes or no (works_head, a kind of road_head) against the answers on the frames known."""
    yes, real = chance[known] >= .5, answer[known] > 0
    hits['true_positive'] += int((yes & real).sum())
    hits['false_positive'] += int((yes & ~real).sum())
    hits['false_negative'] += int((~yes & real).sum())
    told['yes'] += int(real.sum())
    told['no'] += int((~real).sum())


def _told(hits: dict, told: dict) -> dict:
    """Precision and recall of _count, None where the partition has no frame to tell them by."""
    found = hits['true_positive']
    return {**hits, 'precision': found / max(found + hits['false_positive'], 1) if told['no'] else None,
            'recall': found / max(found + hits['false_negative'], 1) if told['yes'] else None}


def _short(drive: dict, length: int) -> dict:
    """The first length frames from the first answer of train on: for a dry run."""
    first = next((i for i, name in enumerate(drive['partitions']) if name == 'train' and drive['speed_weight'][i]), 0)
    part = slice(first, first + length)
    cut = {name: drive[name][part] for name in ('frames', 'ns', 'partitions', 'cells', 'speed', 'speed_out', 'speed_weight',
                                                'works', 'works_weight', 'road', 'road_weight', 'road_type',
                                                'type_weight', 'typed_by_person', 'own_kph', 'measured',
                                                'blocked', 'applicability',
                                                'applicability_weight') if name in drive}
    return {**drive, **cut, 'inputs': {name: value[part] for name, value in drive['inputs'].items()},
            'cuts': {at - first for at in drive['cuts'] if first < at < first + length}}


class _BrainPictures:
    """The brain's pictures of a drive at RATE_HZ, read only when FrameStore copies them: one drive in memory at a time."""

    def __init__(self, root: Path, drive: dict, dry_run: bool):
        self.root, self.drive, self.dry_run = root, drive, dry_run

    def __len__(self) -> int:
        return len(self.drive['frames'])

    def __array__(self, *_args, **_options) -> np.ndarray:
        """uint8 (frames, H, W, 3): the rows of the segments' pictures (road_type_signs.picture_path), zeros where one is
        missing; in a dry run, a frame without its picture has the one of its second (the 1 Hz PNG)."""
        frames = self.drive['frames']
        found = np.zeros((len(frames), data.FRAME_HEIGHT, data.FRAME_WIDTH, 3), dtype=np.uint8)
        segments: dict[int, list[int]] = {}
        for at, frame in enumerate(frames):
            segments.setdefault(frame['segment'], []).append(at)
        for segment, rows in segments.items():
            pictures = load_pictures(self.root, self.drive['route'], segment)
            kept = [at for at in rows if frames[at]['present']] if pictures is not None else []
            if kept:
                found[kept] = pictures[[frames[at]['picture'] for at in kept]]
            if self.dry_run:
                for at in set(rows) - set(kept):
                    if Path(frames[at]['image']).is_file():
                        found[at] = data.resized_frame(frames[at]['image'])
        return found


def _pictures(root: Path, drives: list[dict], *, dry_run: bool, log) -> dict:
    """{route: uint8 (frames, H, W, 3)} for FrameStore: at RATE_HZ the brain's pictures (_BrainPictures); at 1 Hz the frame
    cache (made when missing), or from the PNGs for a dry run, nothing written."""
    brain = {drive['route']: _BrainPictures(root, drive, dry_run) for drive in drives if drive.get('rate', 1.0) == RATE_HZ}
    drives = [drive for drive in drives if drive['route'] not in brain]
    if not drives:
        return brain
    if dry_run:
        return {**brain, **{drive['route']: np.stack([data.resized_frame(frame['image']) if frame['present'] else
                                                       np.zeros((data.FRAME_HEIGHT, data.FRAME_WIDTH, 3), dtype=np.uint8)
                                                       for frame in drive['frames']]) for drive in drives}}
    paths = data.build_frame_cache(root, {drive['route']: drive['frames'] for drive in drives}, log=log)
    return {**brain, **{route: np.load(path) for route, path in paths.items()}}


def _works_needs_both(drives: list[dict], log) -> None:
    """With only marks in train, or nothing, the works task leaves the loss of train: its marks alone would teach a constant
    yes to the memory the speed shares."""
    weights = [loss_weights(drive, 'train')[1] for drive in drives]
    yes = sum(float((weight * drive['works']).sum()) for weight, drive in zip(weights, drives))
    no = sum(float((weight * (1 - drive['works'])).sum()) for weight, drive in zip(weights, drives))
    if yes and no:
        return
    log('No answer on roadworks in train: the works task teaches nothing yet.' if not yes and not no else
        f'Roadworks in train: {int(yes)} frames inside and {int(no)} outside, one side alone teaches nothing: left out. '
        'A mark needs "checked to here" in Speed review.')
    for drive in drives:
        drive['works_weight'] = drive['works_weight'] * (np.array(drive['partitions']) != 'train')


def _road_needs_both(drives: list[dict], log) -> None:
    """_works_needs_both for each kind of road: a kind with only marks in train, or only frames without, leaves the loss of
    train; a kind with nothing says nothing."""
    kinds = next((drive['road_kinds'] for drive in drives if drive.get('road_kinds')), [])
    road = [(road_weights(drive, 'train'), drive) for drive in drives if 'road_weight' in drive]
    for column, kind in enumerate(kinds):
        yes = sum(float((weight[:, column] * drive['road'][:, column]).sum()) for weight, drive in road)
        no = sum(float((weight[:, column] * (1 - drive['road'][:, column])).sum()) for weight, drive in road)
        if yes and no:
            log(f'Road "{kind}" in train: {int(yes)} frames inside and {int(no)} outside.')
            continue
        if yes or no:
            log(f'Road "{kind}" in train: {int(yes)} frames inside and {int(no)} outside, one side alone teaches nothing: '
                'left out. A mark needs "checked to here" in Speed review.')
        for _weight, drive in road:
            drive['road_weight'][:, column] *= np.array(drive['partitions']) != 'train'


def road_pos_weight(drives: list[dict], kinds: list[str]) -> dict[str, float]:
    """{kind: pos_weight} of road_head's loss, in the order of the kinds: for a crossing (speed_review.EVENT_KINDS), the
    frames of train with an answer (weight above 0) of no over those of yes, between 1 and ROAD_EVENT_POS_WEIGHT_MAX;
    1 for every other kind, and for a crossing with no frame of yes in train (one _road_needs_both left out)."""
    road = [(road_weights(drive, 'train'), drive) for drive in drives if 'road_weight' in drive]
    lifts = {}
    for column, kind in enumerate(kinds):
        yes = sum(int(((weight[:, column] > 0) & (drive['road'][:, column] > 0)).sum()) for weight, drive in road)
        no = sum(int(((weight[:, column] > 0) & (drive['road'][:, column] <= 0)).sum()) for weight, drive in road)
        lifts[kind] = (float(np.clip(no / yes, 1.0, ROAD_EVENT_POS_WEIGHT_MAX)) if kind in EVENT_KINDS and yes
                       else 1.0)
    return lifts


def _types_in_train(drives: list[dict], log) -> None:
    """What the road type leaves to learn from, in the log: the seconds of train with a type and those cut without one.
    Only the drives built with the road types (drive_targets' types) count."""
    drives = [drive for drive in drives if drive.get('types')]
    if not drives:
        return
    measured = [(drive, _measured(drive) & (np.array(drive['partitions']) == 'train')) for drive in drives]
    typed = sum(int((drive['type_weight'] * here).sum()) for drive, here in measured)
    cut = sum(int((drive['speed_out'][here] == NO_TYPE).sum()) for drive, here in measured)
    bare = [drive['route'] for drive in drives if not drive['type_weight'].any()]
    log(f'Road types in train: {typed} seconds with one; {cut} seconds with a speed and no road type, left out.')
    if bare:
        log(f"No road type on {len(bare)} drive{'s' if len(bare) > 1 else ''} ({', '.join(bare)}): nothing of "
            f"{'them' if len(bare) > 1 else 'it'} is learned. Read the map in Data › Prepare.")


def _swapped_measure(model, drives: list[dict], answers: list[dict], partition: str, groups: tuple[str, ...], store: FrameStore, *,
                     chunk: int, device) -> dict:
    """follow_the_sign on the frames of a partition whose number was swapped, and how many answers moved: see the module's
    text."""
    held, said, moved = [], [], 0
    for drive, before in zip(drives, answers):
        passages = swap_passages(drive, partition)
        if passages:
            inside = np.zeros_like(drive['speed_weight'])
            inside[np.concatenate(passages)] = 1
            held.append({**swapped(drive, passages, groups), 'speed_weight': drive['speed_weight'] * inside})
            said.append(predict(model, held[-1], store, chunk=chunk, device=device))
            asked = (loss_weights(held[-1], partition)[0] > 0) & _in_force(held[-1]['inputs']['signs']) & _measured(drive)
            moved += int((np.abs(said[-1]['kph'][asked] - before['kph'][asked]) > RIGHT_KPH).sum())
    return {**measure(held, said, partition)['follow_the_sign'], 'moved': moved}


def _spoken(speeds: list[int]) -> str:
    """The speeds said, for the log: how many and from where to where (the whole list is in result.json)."""
    if not speeds:
        return 'nothing'
    return f'{len(speeds)} speed{"s" if len(speeds) > 1 else ""}, {speeds[0]}-{speeds[-1]} km/h'


def _group_sizes(said: dict) -> list[tuple[str, int]]:
    """The groups of a brain with the columns of each, as its checkpoint says of itself (brain_sizes: an earlier one has
    fewer kinds of other sign)."""
    sizes = brain_sizes(said)
    return [(name, sizes[name]) for name in said['groups']]


def brain_start(torch, model, path: Path, source: dict) -> dict:
    """The weights of an earlier brain (its brain-last.pt, nnslr train-brain --init-from) in this fresh one: every layer of
    the same name and shape; the rows of the speed head by km/h, of the road and type heads by name (one the
    start lacks keeps the fresh row); the memory's input by group, an input the start lacks at zero and one it took and
    this one does not left out, so the network starts saying what the start said. A group that grew at its end since the
    start (road_signs: a kind of other sign more) keeps the start's columns and has the new ones at zero. A head the start has not stays fresh,
    and a head only the start has is not taken: the next limit and the metres of a brain of 7 to 11 Nov 2026.
    The trust in the number (NUMBER_TRUSTS) is taken only from a start of this brain's number path (the start's
    'number_path', 'class' when it does not say; source['number_path']): on the other path the same layer multiplies
    another thing, so it stays as this brain was built, at zero for the 'bell' path (#104), and the brain starts from
    what the start's speed_head says wherever a number is in force.
    Refuses a start stepped otherwise (rate, picture, pool, features, memory), learnt on another split, or whose inputs
    are not as wide as its groups say. Returns what result.json records of it: its file and sha256, run, epoch, split,
    the layers left fresh, the inputs added, grown (name: [columns of the start, columns now]) and left out, whether
    the car's file of the inputs was the same, and the start's number path."""
    path = Path(path)
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(saved, dict) or not isinstance(saved.get('weights'), dict) or not saved.get('groups'):
        raise ValueError(f'the start is not a brain: {path}')
    theirs = {'rate_hz': saved.get('rate_hz', 1.0), 'picture': saved.get('picture', 'road_type'), 'pool': saved.get('pool'),
              'features': saved.get('features'), 'hidden': saved.get('hidden')}
    other = [key for key, value in theirs.items() if value != source[key]]
    if other:
        raise ValueError(f'the start is a brain stepped otherwise ({", ".join(f"{key} {theirs[key]}" for key in other)}, '
                         f'this one {", ".join(str(source[key]) for key in other)}): {path}')
    if saved.get('split') and saved['split'] != source['split']:
        raise ValueError(f'the start learnt on another split: its train may hold the frames of this test: {path}')
    weights = saved['weights']
    their_path, our_path = number_path_of(saved), number_path_of(source)
    state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    # The memory's input: the picture's features, then each group's columns in its order (pack_inputs).
    columns, at = {}, source['features']
    for name, size in _group_sizes(saved):
        columns[name] = (at, size)
        at += size
    memory = 'gru.weight_ih_l0'
    # This brain's own groups: those of a training of today when its source does not say its kinds of other sign.
    ours = _group_sizes(source if source.get('road_sign_kinds') is not None else {**source, 'road_sign_kinds': ROAD_SIGN_KINDS})
    if model.state_dict()[memory].shape[1] != source['features'] + sum(size for _name, size in ours):
        raise ValueError(f'this brain takes {model.state_dict()[memory].shape[1] - source["features"]} inputs, its groups say '
                         f'{sum(size for _name, size in ours)}')
    if weights[memory].shape[1] != at:
        raise ValueError(f'the start takes {weights[memory].shape[1] - source["features"]} inputs, its groups say '
                         f'{at - source["features"]}: {path}')
    table = [int(round(value)) for value in weights['class_kph'].tolist()] if 'class_kph' in weights else []
    own = [int(round(value)) for value in model.class_kph.tolist()]
    rows = {'speed_head.': ({kph: row for row, kph in enumerate(table)}, own),
            'road_head.': ({kind: row for row, kind in enumerate(saved.get('road_kinds') or [])}, source['road_kinds']),
            'type_head.': ({kind: row for row, kind in enumerate(saved.get('road_types') or [])}, source['road_types'])}
    fresh, added, grown = [], [], {}
    for key, current in state.items():
        if key in ('class_kph', 'mean', 'std'):
            continue  # its own table and the ImageNet constants
        head = next((prefix for prefix in rows if key.startswith(prefix)), None)
        if key == memory:
            current[:, :source['features']] = weights[key][:, :source['features']]
            at = source['features']
            for name, size in ours:
                current[:, at:at + size] = 0
                if name in columns:
                    # The start's columns, the first of this group: one that grew at its end has its new ones at zero.
                    theirs, kept = columns[name][0], min(columns[name][1], size)
                    current[:, at:at + kept] = weights[key][:, theirs:theirs + kept]
                    if columns[name][1] != size:
                        grown[name] = [columns[name][1], size]
                else:
                    added.append(name)
                at += size
        elif head is not None and key in weights:
            found, wanted = rows[head]
            missing = [str(name) for name in wanted if name not in found]
            for row, name in enumerate(wanted):
                if name in found:
                    current[row] = weights[key][found[name]]
            if missing:
                fresh.append(f'{key} ({", ".join(missing[:5])}{"..." if len(missing) > 5 else ""})')
        elif key.startswith(NUMBER_TRUSTS) and their_path != our_path:
            fresh.append(key)  # the trust of another number path: as this brain was built
        elif key in weights and tuple(weights[key].shape) == tuple(current.shape):
            state[key] = weights[key]
        else:
            fresh.append(key)
    if any(key.startswith('backbone.') for key in fresh):
        raise ValueError(f'the start has another picture encoder: {path}')
    model.load_state_dict(state, strict=True)
    ours = set(source['groups'])
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'run': path.parent.name,
            'epoch': saved.get('epoch'), 'split_sha256': saved.get('split'), 'fresh': fresh, 'inputs_added': added,
            'inputs_grown': grown,
            'inputs_left_out': [name for name in columns if name not in ours],
            'same_inputs_file': saved.get('inputs_sha256') == source['inputs_sha256'],
            'number_path': their_path, 'same_number_path': their_path == our_path}


def _start_text(start: dict) -> str:
    """The log's line of what a training goes on from."""
    epoch = '' if start['epoch'] is None else f' (epoch {start["epoch"] + 1})'
    return (f'Starts from the brain {start["run"]}{epoch}: every layer'
            + (f' but {", ".join(start["fresh"])}' if start['fresh'] else '')
            + (f'; new inputs at zero: {", ".join(start["inputs_added"])}' if start['inputs_added'] else '')
            + ''.join(f'; {name} grew from {before} to {now} numbers, the new ones at zero'
                      for name, (before, now) in (start.get('inputs_grown') or {}).items())
            + (f'; inputs it took, left out: {", ".join(start["inputs_left_out"])}' if start['inputs_left_out'] else '')
            + ('' if start.get('same_number_path', True) else
               f'; its number path is another ({start["number_path"]}): the trust in the number starts again')
            + ('' if start['same_inputs_file'] else '; the car’s file of the inputs has changed since') + '.')


def trust_size(model) -> dict:
    """How much the number's path holds (brain_net's text): for the trust in the sign's number, and in the map's limit
    of a brain with that path ('trust_map'), the L2 and L1 norms of the layer's weights and its bias. The memory's
    output is within -1 and 1, so trust(h) lies within the bias plus or minus the L1 norm; a layer of the 'bell' path
    starts at zero, so all of it was learnt, and a fresh one of the 'class' path starts near 0.58 (L2)."""
    found = {}
    for name in ('trust', 'trust_map'):
        layer = getattr(model, name, None)
        if layer is not None:
            weight = layer.weight.detach().float()
            found[name] = {'weights_l2': float(weight.norm()), 'weights_l1': float(weight.abs().sum()),
                           'bias': float(layer.bias.detach().float().sum())}
    return found


def _trust_text(path: str, before: dict, after: dict, swapped_test: dict) -> str:
    """The log's last line of a training: what the number's path holds at its start and at the epoch kept, and on how
    many test frames a swapped number moved the answer."""
    parts = [f"{'the map’s limit' if name == 'trust_map' else 'the sign’s number'}: weights L2 "
             f"{before[name]['weights_l2']:.2f} -> {size['weights_l2']:.2f}, bias {before[name]['bias']:+.2f} -> "
             f"{size['bias']:+.2f}" for name, size in after.items()]
    frames = swapped_test['network']['frames']
    return (f"The number’s path ({path}), trust in " + '; '.join(parts)
            + f". A swapped number moved the answer on {swapped_test['moved']} of {frames} test frames.")


def train_brain(data_root: Path, run_dir: Path, merged_of, *, pool: str = 'grid', works: bool = True,
                road: bool = True, types: bool = True, epochs: int = EPOCHS, chunk: int = CHUNK, device: str | None = None,
                dry_run: bool = False, seed: int = 0, routes: list[str] | None = None, init_from: Path | None = None,
                sign_networks: dict | None = None, map_limit: bool = False, applicability: bool = True,
                number_path: str = NUMBER_PATH, log=_print) -> dict:
    """Train on train, keep the last epoch, measure once. Writes routes.json (the drives it takes, ROUTES), brain-last.pt
    and result.json into run_dir; the checkpoint
    says what it came from, its table of the speeds too (the class_kph of its weights), and its kinds of road. Refuses what
    would train wrong: brain_targets.check_drives. It learns from LEARNED_INPUTS, never his pedals nor the car's
    speed, nor the metres to a sign. road: with the kinds of road of Speed review as a task of their own (road_head), the
    crossings lifted (road_pos_weight); False, the comparison.
    types: with the road type as a task of its own (type_head), and only the seconds that have one (brain_targets'
    types); False, the comparison. routes: only these drives (those of the run's start, run_brain.sh), and of them only
    the ones still in training_routes; None, every drive of training_routes. init_from: an earlier brain-last.pt whose
    weights it goes on from (brain_start; the dry run checks it), else ImageNet and random weights. sign_networks: the
    networks its signs were read with (sign_networks.load: the cutter, classifier and applicability network chosen in
    Train › Brain, for its inputs), recorded in result.json; None, the car's release of the sunnypilot
    clone. map_limit: the map's limit the comma had is one more input (learned_inputs, #85: Train › Brain's switch, off
    unless asked); the brain is then the one case where the vision side reads the map. applicability: with whose road
    a sign is of as a task of its own (applicability_head, brain_net), and sign_boxes with every speed sign read, those
    of another road too (the checkpoint's other_road_boxes); False, the comparison: the signs taken, as every brain of
    before, and only the applicability network measured. number_path: how the number in force reaches the answer
    (brain_net.NUMBER_PATHS, #104): 'bell', the usual one, or 'class', the path of every brain before 16 Nov 2026, the
    comparison; the checkpoint and result.json keep the word, and result.json what its trust holds (trust_size)."""
    if number_path not in NUMBER_PATHS:
        raise ValueError(f'the number path is one of {NUMBER_PATHS}: {number_path}')
    torch = _torch()
    root, run_dir = Path(data_root), Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    torch.manual_seed(seed)
    device = torch.device(device or ('cpu' if dry_run else 'cuda'))
    groups = learned_inputs(map_limit)
    if map_limit:
        log("The map's limit is an input of this brain (liveMapDataSP, as the comma had it): on the car it will read the "
            'map, which no other brain does.')
    # Every dashcam drive, each whole (brain_targets.training_routes): how many and how long, before the dry run keeps
    # only a piece of the first.
    counts = training_counts(root)
    log(training_counts_text(counts))
    if routes is not None:
        log(f'Only the {len(routes)} drive{"" if len(routes) == 1 else "s"} of the start of this run (--route): '
            'one added since waits for the next run.')
    if sign_networks is not None:
        log('The signs read by the cutter ' + sign_networks['cutter'] + ' and the classifier ' + sign_networks['classifier']
            + ': every one of them, of another road too; ' + ('the brain learns whose road a sign is of from your reviews.'
                                                             if applicability else 'the brain is not taught whose road a sign is of.'))
    drives = dataset(root, merged_of, routes, log=log, rate=RATE_HZ, need_pictures=not dry_run, types=types,
                     networks=sign_networks, other_road_boxes=applicability)
    if not drives and not counts['drives']:
        raise ValueError('no dashcam drive to learn from: drive with sunnylink › Toggles › Enable sunnypilot off, '
                         'then Read the logs on Data › Speed review')
    if not drives:
        raise ValueError(f'none of the {counts["drives"]} dashcam drives gave the brain a drive to learn: see the '
                         'lines above for why each was left out'
                         + ('' if routes is None else ' (only those of the start of this run: a new NAME takes the '
                                                      'drives of today)'))
    # On every drive, before the dry run keeps a piece of one: the split may have put some of them all in val and test.
    if not any(loss_weights(drive, 'train')[0].any() for drive in drives):
        raise ValueError('no speed labels in train: the dashcam drives fall in val and test of the split; drive more')
    (run_dir / ROUTES).write_text(json.dumps([drive['route'] for drive in drives]), encoding='utf-8')
    # The same of whose road a sign is of: what the reviews answer, on every drive, with the head or not.
    log(applicability_counts_text(applicability_counts(drives))
        + ('. The brain is given every speed sign read, those of another road too, and learns whose road each is of'
           if applicability else '. applicability_head off (--no-applicability): the brain gets the signs taken, and '
                                 'only the applicability network is measured'))
    if dry_run:
        # The first drive with an answer in train (one all in val or test would leave the dry run nothing to learn).
        drives = [_short(next(drive for drive in drives if loss_weights(drive, 'train')[0].any()), DRY_FRAMES)]
        epochs, chunk = 1, DRY_FRAMES // 2
    else:
        check_drives(drives)
    for drive in drives:
        drive['extra'] = pack_inputs(drive['inputs'], groups)
    if works:
        _works_needs_both(drives, log)
    kinds = list(drives[0].get('road_kinds', ())) if road and drives else []
    if kinds:
        _road_needs_both(drives, log)
    if types:
        _types_in_train(drives, log)
    if applicability:
        # A dry run keeps a few frames of one drive: what it lacks says nothing of the training.
        _applicability_needs_both(drives, (lambda _text: None) if dry_run else log)
    sign_lift = applicability_pos_weight(drives) if applicability else None
    if sign_lift is not None:
        log(f'A sign of another road weighs {sign_lift:.1f} times one of ours (pos_weight)')
    road_lift = road_pos_weight(drives, kinds)
    if any(kind in EVENT_KINDS for kind in kinds):
        log(f'The crossings weigh in road_head (pos_weight, frames of no over yes in train, at most '
            f'{ROAD_EVENT_POS_WEIGHT_MAX:.0f}): '
            + ', '.join(f'{kind} {road_lift[kind]:.1f}' for kind in kinds if kind in EVENT_KINDS))
    store = FrameStore(_pictures(root, drives, dry_run=dry_run, log=log), device)
    # A brain that learns the map's limit has its path straight to the answer (brain_net: trust_map).
    model = build_brain_net(groups, pool=pool, works=works, pretrained=not dry_run, road_kinds=len(kinds),
                            road_types=len(ROAD_TYPES) if types else 0, map_path=map_limit,
                            applicability=applicability, number_path=number_path).to(device)
    # None when every kind weighs 1: road_head's loss is then the one of before, to the last bit.
    lifted = [road_lift[kind] for kind in kinds]
    kinds_pos_weight = (torch.tensor(lifted, dtype=torch.float32, device=device) if any(value != 1.0 for value in lifted)
                        else None)
    sign_pos_weight = None if sign_lift is None else torch.tensor(sign_lift, device=device)
    optimizer = torch.optim.AdamW([value for value in model.parameters() if value.requires_grad], lr=LEARNING_RATE,
                                  weight_decay=WEIGHT_DECAY)
    plates = 'plate' in groups and plates_in_train(drives)
    hours = 'plate_hours' in groups and hours_in_train(drives)
    wet = 'plate_wet' in groups and wet_in_train(drives)
    # rate_hz and picture: how the car steps it (brain_export); a checkpoint without them is a 1 Hz brain of PIL pictures.
    # No 'ahead': only the checkpoints of 7 to 11 Nov 2026 have it, with their heads of the limit ahead.
    source = {'groups': groups, 'pool': pool, 'works': works, 'road_kinds': kinds,
              'road_types': list(ROAD_TYPES) if types else [],
              # Whose road a sign is of as a head, and what sign_boxes held: every sign read, those of another road too
              # (brain_export puts it in the contract: the car gives such a brain the same).
              'applicability': applicability, 'other_road_boxes': applicability,
              'features': FEATURES, 'hidden': HIDDEN, 'plates': plates,
              'hours': hours, 'wet': wet,
              'rate_hz': drives[0].get('rate', 1.0), 'picture': 'brain' if drives[0].get('rate', 1.0) == RATE_HZ else 'road_type',
              'split': read_split(root)['sha256'], 'reader': drives[0]['reader'], 'inputs_sha256': inputs_sha256(),
              # The kinds of other sign of road_signs (brain_targets.brain_sizes): the export tells the car, which gives a
              # brain the kinds it was trained with when more are added.
              'road_sign_kinds': ROAD_SIGN_KINDS,
              # How the number in force reaches the answer (brain_net.NUMBER_PATHS, #104): whoever builds the network
              # from this file reads it (brain_net.number_path_of); a checkpoint of before 16 Nov 2026 does not say.
              'number_path': number_path}
    start = None
    if init_from is not None:
        start = brain_start(torch, model, Path(init_from), source)
        log(_start_text(start))
    trust_start = trust_size(model)
    log(f'Frames with a night answer: {sum(int(night_frames(drive)[0].sum()) for drive in drives)}')
    history = []
    name = 'dry_run' if dry_run else f'brain_s{seed}'
    for epoch in range(epochs):
        started = time.monotonic()
        error = train_epoch(model, optimizer, epoch_drives(drives, groups, rng), store, chunk=chunk, rng=rng, device=device,
                            steps=DRY_STEPS if dry_run else None, road_pos_weight=kinds_pos_weight,
                            sign_pos_weight=sign_pos_weight)
        answers = [predict(model, drive, store, chunk=chunk, device=device) for drive in drives]
        validation = measure(drives, answers, 'validation')
        score = validation['speed']['mean_error_kph']
        history.append({'epoch': epoch, 'error': error, 'validation': validation['speed'], 'types': validation['types'],
                        'applicability': validation['applicability']})
        # Read by nnslr_tools.job_progress: the seed's curve in Train › Brain, with the road type type_head names right on
        # validation, on all and on each type (only measured: the last epoch is kept whatever it says).
        right, spoken = validation['speed']['right'], validation['speeds_said']
        log(f'{name} epoch {epoch + 1}/{epochs} train_loss {error:.4f} val_error_kph {"-" if score is None else f"{score:.2f}"} '
            f'val_accuracy {"-" if right is None else f"{right:.4f}"}{_types_words(validation["types"])}'
            f'{_applicability_words(validation["applicability"])} '
            f'on {validation["speed"]["frames"]} frames '
            f'(always {validation["one_speed"]["kph"]}: {validation["one_speed"]["mean_error_kph"]}, '
            f'the sign: {validation["follow_the_sign"]["sign"]["mean_error_kph"]}), says {_spoken(spoken)} '
            f'{time.monotonic() - started:.0f}s')
    torch.save({'weights': model.state_dict(), 'epoch': epochs - 1, **source}, run_dir / CHECKPOINT)
    cold = [predict(model, drive, store, chunk=chunk, device=device, cold=True) for drive in drives]
    swapped_measure = partial(_swapped_measure, model, drives, answers, groups=groups, store=store, chunk=chunk, device=device)
    # Of a brain that learns the map's limit: the same drives with no map on any frame, as past a border with no map.
    no_map = {}
    if MAP_GROUP in groups:
        blind = [without_map(drive, groups) for drive in drives]
        said = [predict(model, drive, store, chunk=chunk, device=device) for drive in blind]
        no_map = {'validation_no_map': measure(blind, said, 'validation'), 'test_no_map': measure(blind, said, 'test')}
    table = [int(round(value)) for value in model.class_kph.tolist()]
    # What the number's path holds, at the start and at the epoch kept: beside test_swapped, whether it was learnt.
    trust = {'start': trust_start, 'kept': trust_size(model)}
    # rate_hz, hidden and kph: how the car steps it and what it says, for Models (model_roles.brain_digest).
    result = {'groups': list(groups), 'road_sign_kinds': source['road_sign_kinds'], 'map_limit': map_limit, 'map_path': map_limit, 'pool': pool, 'works': works, 'road_kinds': kinds,
              'number_path': number_path, 'trust': trust,
              'road_types': list(ROAD_TYPES) if types else [],
              'road_pos_weight': {kind: round(value, 3) for kind, value in road_lift.items()},
              'applicability': applicability, 'other_road_boxes': applicability,
              'applicability_pos_weight': sign_lift, 'applicability_answers': applicability_counts(drives),
              'rate_hz': source['rate_hz'],
              'hidden': HIDDEN, 'kph': {'from': table[0], 'to': table[-1], 'step': table[1] - table[0] if len(table) > 1 else 0},
              'plates': plates, 'hours': hours, 'wet': wet, 'dry_run': dry_run,
              'seed': seed, 'right_kph': RIGHT_KPH, 'history': history,
              'split_sha256': source['split'], 'start': start,
              'sign_networks': {part: sign_networks[part] for part in ('cutter', 'classifier', 'decider')
                                if sign_networks.get(part)} if sign_networks is not None else None,
              'answers': {drive['route']: int((drive['speed_weight'] > 0).sum()) for drive in drives},
              'train': measure(drives, answers, 'train'), 'validation': measure(drives, answers, 'validation'),
              'test': measure(drives, answers, 'test'), 'validation_cold': measure(drives, cold, 'validation'),
              'test_cold': measure(drives, cold, 'test'),
              'validation_swapped': swapped_measure('validation'), 'test_swapped': swapped_measure('test'), **no_map}
    (run_dir / RESULT).write_text(json.dumps(result, indent=2), encoding='utf-8')
    log(_trust_text(number_path, trust['start'], trust['kept'], result['test_swapped']))
    return result
