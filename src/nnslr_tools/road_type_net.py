"""The road-type network with memory: the kind of road from the camera, one frame a second, and its changes.

Each second the car gives it the road frame (resized to 256 x 144, road_type_data) and its speed; a small pretrained
encoder (torchvision's MobileNetV3-small, ImageNet weights) makes 128 numbers of the frame, the speed is added, and a
GRU of 128 keeps what the network saw before: the merge of a ramp, the first houses of a town and the end of them are
found from how the picture changes over seconds, not from one frame. The output is the probability of each class
(road_type_data.CLASSES). ``memory='none'`` is the same encoder and speed with a small head per frame: the network
without memory it is compared with (and a causal majority vote of 5 seconds over its answers, the cheap memory).

Training (train_road_type): windows of WINDOW seconds cut from the train sequences, FOCUS_SHARE of
them around a change of road type; half the batches start from the memory of BURN_IN seconds before the window (as on
the car, where the memory is never empty after the first seconds), the others from zeros; per sequence the same
random colour and framing; a second dropped now and then; the speed scaled by +-5 % and, in one sequence in ten,
unknown. Loss: cross-entropy weighted by the frame's weight (road_type_data) and by 1 / sqrt(class frequency); the
best epoch by the validation macro F1 (weight-1 frames, the whole sequences run with their memory, as on the car).

The ONNX export (export_road_type_onnx) is one STEP: image (1, 3, 144, 256) float RGB 0..1, speed (1, 2) and hidden
(1, 128) in; probabilities (1, 6) and hidden_out (1, 128) out. The car keeps hidden_out and gives it back the next
second (zeros at the start). The GRU is written out with matrix products, sigmoid and tanh (no ONNX GRU node), the
ImageNet normalisation is inside. The contract beside it says how to make the input from the NV12 road frame.
Needs the GPU environment (.venv-gpu); the dry run works on the CPU.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np

from nnslr_tools import road_type_data as data
from nnslr_tools import road_type_signs as signs_data
from nnslr_tools.road_type_metrics import (AFTER_MERGE_SECONDS, LATE_SECONDS, change_points, comma_release, debounced,
                                           majority, ramp_entries, release_summary, sequence_metrics)
from nnslr_tools.road_type_signs import NO_SIGN

SIGN_INPUTS = 2

FEATURES = 128
HIDDEN = 128
# Every train window has the same length: cuDNN plans the half-precision backward of every new batch shape anew, about
# 6 s each on the V100 (29 Oct 2026), longer than the step itself.
WINDOW = 48
BURN_IN = 32
FOCUS_SHARE = 0.35
DROP_FRAME = 0.04
NO_SPEED_SHARE = 0.1
# The merge, where the ramp's limit ends (road_type_metrics.comma_release): MERGE_SHARE of the windows hold one (the
# ramp in the window or in its burn-in, the carriageway after it), and the loss weighs the ramp and the first
# AFTER_MERGE_SECONDS of the carriageway MERGE_WEIGHT times.
MERGE_SHARE = 0.25
MERGE_WEIGHT = 3.0
# The sign input (road_type_signs): in SIGN_DROP_SHARE of the windows the car missed the sign (none seen), in
# SIGN_FAKE_SHARE of those without one a sign is confirmed where there is none (a misreading).
SIGN_DROP_SHARE = 0.25
SIGN_FAKE_SHARE = 0.05
# The sign of a train window comes up to SIGN_JITTER seconds earlier or later: the reviewed 4.01 signs were seen from
# 7 s before to 6 s after the first second of the map's ramp, and the car confirms a sign while it sees it.
SIGN_JITTER = 4.0
MIN_DRIVE = 30
# One letter a second for the answers around a merge in the result (release_evaluation).
ANSWER_LETTERS = {'town': 'T', 'rural': 'R', 'expressway': 'E', 'motorway': 'M', 'ramp': '/', 'parking': 'P'}
BATCH = 12
EPOCHS = 60
PATIENCE = 15
LEARNING_RATE = 1e-3
BACKBONE_LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
SMOOTH_SECONDS = 5
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CHECKPOINT = 'road-type-best.pt'
MEMORY_NAMES = {'gru': 'road_type_gru', 'none': 'road_type_frame'}


def network_name(memory: str, signs: bool = True) -> str:
    """The name of a training in its log lines (job_progress reads them): road_type_gru, road_type_gru_nosign, ..."""
    return MEMORY_NAMES[memory] + ('' if signs else '_nosign')


def _torch():
    import torch
    return torch


def _print(text: str) -> None:
    print(text, flush=True)


def build_road_type_net(memory: str = 'gru', pretrained: bool = True, features: int = FEATURES, hidden: int = HIDDEN,
                        signs: bool = False):
    """The network (torch.nn.Module): forward(images (B, T, 3, H, W) float 0..1, speed (B, T, 2), hidden (B, hidden)
    or None, sign (B, T, 2) or None) -> (logits (B, T, classes), hidden (B, hidden)). With signs, the input of the
    memory has the sign input too (road_type_signs.sign_features; None: no sign seen)."""
    torch = _torch()
    from torch import nn
    from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

    if memory not in MEMORY_NAMES:
        raise ValueError(f'memory must be one of {sorted(MEMORY_NAMES)}')
    extra = 2 + (SIGN_INPUTS if signs else 0)

    class RoadTypeNet(nn.Module):
        def __init__(self):
            super().__init__()
            backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None)
            self.memory, self.hidden_size, self.signs = memory, hidden, signs
            self.register_buffer('mean', torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
            self.register_buffer('std', torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
            self.backbone = backbone.features
            self.project = nn.Sequential(nn.Linear(576, features), nn.Hardswish(), nn.Dropout(0.2))
            if memory == 'gru':
                self.gru = nn.GRU(features + extra, hidden, batch_first=True)
                self.head = nn.Linear(hidden, len(data.CLASSES))
            else:
                self.head = nn.Sequential(nn.Linear(features + extra, hidden), nn.ReLU(), nn.Linear(hidden, len(data.CLASSES)))

        def encode(self, images):
            """(N, 3, H, W) 0..1 -> (N, features)."""
            x = self.backbone((images - self.mean) / self.std)
            return self.project(x.mean(dim=(2, 3)))

        def inputs(self, encoded, speed, sign=None):
            """The input of the memory (..., features + 2 [+ 2]): the frame's numbers, the speed, the sign input."""
            parts = [encoded, speed.to(encoded.dtype)]
            if self.signs:
                if sign is None:
                    sign = encoded.new_tensor(NO_SIGN).expand(*encoded.shape[:-1], SIGN_INPUTS)
                parts.append(sign.to(encoded.dtype))
            return torch.cat(parts, dim=-1)

        def forward(self, images, speed, hidden=None, sign=None):
            b, t = images.shape[:2]
            return self.run(self.inputs(self.encode(images.flatten(0, 1)).view(b, t, -1), speed, sign), hidden)

        def run(self, x, hidden=None):
            """The memory's input (B, T, features + 2 [+ 2]) -> (logits, hidden)."""
            if self.memory == 'gru':
                out, last = self.gru(x, None if hidden is None else hidden[None].contiguous())
                return self.head(out), last[0]
            b = x.shape[0]
            return self.head(x), hidden if hidden is not None else x.new_zeros(b, self.hidden_size)

    return RoadTypeNet()


def gru_step(gru, x, hidden):
    """One step of a torch.nn.GRU (one layer) with matrix products, sigmoid and tanh: the same numbers, no GRU node in
    the ONNX graph (the comma compiles the graph with tinygrad)."""
    torch = _torch()
    import torch.nn.functional as F
    gates_x = F.linear(x, gru.weight_ih_l0, gru.bias_ih_l0)
    gates_h = F.linear(hidden, gru.weight_hh_l0, gru.bias_hh_l0)
    xr, xz, xn = gates_x.chunk(3, dim=1)
    hr, hz, hn = gates_h.chunk(3, dim=1)
    reset = torch.sigmoid(xr + hr)
    update = torch.sigmoid(xz + hz)
    candidate = torch.tanh(xn + reset * hn)
    return (1 - update) * candidate + update * hidden


def step_module(model):
    """The one-second step of the car: (image (1, 3, H, W) 0..1, speed (1, 2), [sign (1, 2),] hidden (1, hidden)) ->
    (probabilities (1, classes), hidden_out (1, hidden)); the sign input only for a network trained with it."""
    torch = _torch()
    from torch import nn

    def advance(image, speed, sign, hidden):
        x = model.inputs(model.encode(image), speed, sign)
        if model.memory == 'gru':
            hidden_out = gru_step(model.gru, x, hidden)
            logits = model.head(hidden_out)
        else:
            hidden_out = hidden * 1.0
            logits = model.head(x)
        return torch.softmax(logits, dim=1), hidden_out

    class RoadTypeStep(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model

        def forward(self, image, speed, hidden):
            return advance(image, speed, None, hidden)

    class RoadTypeSignStep(RoadTypeStep):
        def forward(self, image, speed, sign, hidden):
            return advance(image, speed, sign, hidden)

    return (RoadTypeSignStep() if getattr(model, 'signs', False) else RoadTypeStep()).eval()


def step_example(model, torch=None):
    """Zero inputs of one step, in the order of step_module's forward."""
    torch = torch or _torch()
    image, speed, hidden = torch.zeros(1, 3, data.FRAME_HEIGHT, data.FRAME_WIDTH), torch.zeros(1, 2), torch.zeros(1, model.hidden_size)
    if getattr(model, 'signs', False):
        return image, speed, torch.tensor([NO_SIGN]), hidden
    return image, speed, hidden


def count_parameters(model) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def step_flops(model, width: int = data.FRAME_WIDTH, height: int = data.FRAME_HEIGHT) -> int:
    """Floating point operations of one step (torch's FlopCounterMode: 2 per multiply-add of convolutions and matrix
    products)."""
    torch = _torch()
    import copy
    from torch.utils.flop_counter import FlopCounterMode
    step = step_module(copy.deepcopy(model).cpu())
    example = step_example(model, torch)
    with torch.no_grad(), FlopCounterMode(display=False) as counter:
        step(*example)
    return int(counter.get_total_flops())


class FrameStore:
    """The resized pictures of the drives in one uint8 tensor (frames, H, W, 3) on the device, and each drive's
    first row in it."""

    def __init__(self, arrays: dict[str, np.ndarray], device):
        torch = _torch()
        self.offset, rows, start = {}, [], 0
        for route, array in arrays.items():
            self.offset[route] = start
            rows.append(array)
            start += len(array)
        self.pixels = torch.from_numpy(np.concatenate(rows)).to(device)

    def batch(self, route_positions: list[tuple[str, list[int]]]):
        """(B, T, 3, H, W) float 0..1 of equal-length position lists."""
        torch = _torch()
        index = torch.tensor([[self.offset[route] + p for p in positions] for route, positions in route_positions],
                             device=self.pixels.device)
        return self.pixels[index].permute(0, 1, 4, 2, 3).float().div_(255)


def attach_signs(dataset: dict, root: Path, source: str = signs_data.DEFAULT_SOURCE) -> dict:
    """The sign input of every frame (dataset['signs'][route], (frames, 2)) from the motorway signs of source
    (road_type_signs.SIGN_SOURCES: 'car', those the car's code confirmed, read before by nnslr road-type-signs, a drive
    without them: none seen; 'map', the first second of every entry ramp of the map), their times
    (dataset['sign_times']), the time of every frame (dataset['seconds']) and the loss emphasis of the merges
    (dataset['emphasis'][route]). Returns the coverage."""
    if source not in signs_data.SIGN_SOURCES:
        raise ValueError(f'unknown sign source {source!r}: one of {", ".join(signs_data.SIGN_SOURCES)}')
    dataset['signs'], dataset['sign_times'], dataset['emphasis'], dataset['seconds'] = {}, {}, {}, {}
    read = []
    for route, frames in dataset['frames'].items():
        if source == 'car':
            record = signs_data.load_signs(root, route)
            if record:
                read.append(route)
            times = signs_data.sign_times(record)
        else:
            read.append(route)
            times = signs_data.map_sign_times(frames, data.CLASSES)
        dataset['sign_times'][route] = times
        dataset['seconds'][route] = np.array([signs_data.frame_seconds(f) for f in frames], dtype=np.float64)
        dataset['signs'][route] = signs_data.frame_sign_features(frames, times)
        labels = np.array([f['label'] for f in frames], dtype=np.int64)
        emphasis = np.ones(len(frames), dtype=np.float32)
        for entry in ramp_entries(labels, data.CLASSES):
            emphasis[entry['ramp']:entry['merge'] + AFTER_MERGE_SECONDS] = MERGE_WEIGHT
        dataset['emphasis'][route] = emphasis
    coverage = {'source': source, 'drives': len(dataset['frames']), 'read': len(read),
                'signs': sum(len(t) for t in dataset['sign_times'].values()),
                'drives_with_signs': sum(1 for t in dataset['sign_times'].values() if t)}
    dataset['sign_coverage'] = coverage
    return coverage


def _sequence_arrays(dataset: dict, run: dict) -> dict:
    route, positions = run['route'], run['positions']
    frames = dataset['frames'][route]
    chosen = [frames[p] for p in positions]
    signs = dataset.get('signs', {}).get(route)
    emphasis = dataset.get('emphasis', {}).get(route)
    seconds = dataset.get('seconds', {}).get(route)
    return {'labels': np.array([f['label'] for f in chosen], dtype=np.int64),
            'weights': np.array([f['weight'] for f in chosen], dtype=np.float32),
            'speed': np.array([data.speed_features(f['speed_mps']) for f in chosen], dtype=np.float32),
            'sign': (signs[positions] if signs is not None else np.tile(np.float32(NO_SIGN), (len(chosen), 1))
                     ).astype(np.float32),
            'emphasis': (emphasis[positions] if emphasis is not None else np.ones(len(chosen))).astype(np.float32),
            'seconds': (seconds[positions] if seconds is not None else
                        np.array([signs_data.frame_seconds(f) for f in chosen], dtype=np.float64)),
            'keys': [f['key'] for f in chosen]}


def class_weights(dataset: dict, runs: list[dict]) -> np.ndarray:
    """1 / sqrt(frequency) of each class over the train frames (by their weights), mean 1 over the frames; 0 for a
    class without frames."""
    totals = np.zeros(len(data.CLASSES))
    for run in runs:
        arrays = _sequence_arrays(dataset, run)
        for label, weight in zip(arrays['labels'], arrays['weights']):
            if label >= 0:
                totals[label] += weight
    weights = np.where(totals > 0, 1 / np.sqrt(np.maximum(totals, 1e-9)), 0.0)
    scale = (totals * weights).sum() / max(totals.sum(), 1e-9)
    return (weights / scale).astype(np.float32) if scale > 0 else weights.astype(np.float32)


def _augment(torch, images, generator_seed: int | None = None):
    """The same colour and framing on every frame of a sequence, different per sequence: brightness, contrast and
    saturation +-25 %, a zoom of up to 12 % and a shift and turn of a few pixels and degrees. No mirror: traffic keeps
    right."""
    import torch.nn.functional as F
    b, t, c, h, w = images.shape
    device = images.device
    brightness = torch.empty(b, 1, 1, 1, 1, device=device).uniform_(0.75, 1.25)
    contrast = torch.empty(b, 1, 1, 1, 1, device=device).uniform_(0.75, 1.25)
    saturation = torch.empty(b, 1, 1, 1, 1, device=device).uniform_(0.7, 1.3)
    mean = images.mean(dim=(1, 2, 3, 4), keepdim=True)
    images = (images - mean) * contrast + mean
    images = images * brightness
    gray = images.mean(dim=2, keepdim=True)
    images = (images - gray) * saturation + gray
    scale = torch.empty(b, device=device).uniform_(1 / 1.12, 1.0)
    angle = torch.empty(b, device=device).uniform_(-2, 2) * math.pi / 180
    shift_x = torch.empty(b, device=device).uniform_(-1, 1) * (1 - scale)
    shift_y = torch.empty(b, device=device).uniform_(-1, 1) * (1 - scale)
    theta = torch.zeros(b, 2, 3, device=device)
    theta[:, 0, 0] = scale * torch.cos(angle)
    theta[:, 0, 1] = -scale * torch.sin(angle) * h / w
    theta[:, 1, 0] = scale * torch.sin(angle) * w / h
    theta[:, 1, 1] = scale * torch.cos(angle)
    theta[:, 0, 2], theta[:, 1, 2] = shift_x, shift_y
    theta = theta.repeat_interleave(t, dim=0)
    flat = images.reshape(b * t, c, h, w)
    grid = F.affine_grid(theta, flat.shape, align_corners=False)
    flat = F.grid_sample(flat, grid, mode='bilinear', padding_mode='reflection', align_corners=False)
    return flat.view(b, t, c, h, w).clamp_(0, 1)


class WindowSampler:
    """Train windows: (route, positions, labels, weights, speed, sign, emphasis) of length burn + window, a second
    dropped now and then, MERGE_SHARE of them around a merge and FOCUS_SHARE around a change of road type; the sign
    input missed or misread now and then (SIGN_DROP_SHARE, SIGN_FAKE_SHARE) and confirmed up to SIGN_JITTER seconds
    earlier or later."""

    def __init__(self, dataset: dict, runs: list[dict], rng: random.Random):
        self.rng = rng
        self.times = dataset.get('sign_times', {})
        self.runs = [(run, _sequence_arrays(dataset, run)) for run in runs]
        self.changes = [(i, change['time']) for i, (run, arrays) in enumerate(self.runs)
                        for change in change_points(arrays['labels'], data.CLASSES)]
        self.merges = [(i, entry['merge']) for i, (run, arrays) in enumerate(self.runs)
                       for entry in ramp_entries(arrays['labels'], data.CLASSES)]
        self.lengths = [len(run['positions']) for run, _ in self.runs]

    def window(self, length: int):
        rng = self.rng
        fits = [i for i, n in enumerate(self.lengths) if n >= length]
        if not fits:
            raise ValueError(f'no train sequence of {length} frames')
        fitting = set(fits)
        focus = [(i, t) for i, t in self.changes if i in fitting]
        merges = [(i, t) for i, t in self.merges if i in fitting]
        draw = rng.random()
        if merges and draw < MERGE_SHARE:
            index, moment = rng.choice(merges)
            start = min(max(moment - rng.randrange(min(10, length - 1), max(length - 10, 11)), 0),
                        self.lengths[index] - length)
        elif focus and draw < MERGE_SHARE + FOCUS_SHARE:
            index, moment = rng.choice(focus)
            start = min(max(moment - rng.randrange(length), 0), self.lengths[index] - length)
        else:
            index = rng.choices(fits, weights=[self.lengths[i] for i in fits])[0]
            start = rng.randrange(self.lengths[index] - length + 1)
        run, arrays = self.runs[index]
        picks, cursor = [], start
        while len(picks) < length and cursor < self.lengths[index]:
            picks.append(cursor)
            cursor += 2 if rng.random() < DROP_FRAME else 1
        while len(picks) < length:
            if picks[0] > 0:
                picks.insert(0, picks[0] - 1)
            else:
                picks.append(picks[-1])
        return (run['route'], [run['positions'][p] for p in picks], arrays['labels'][picks],
                arrays['weights'][picks], arrays['speed'][picks].copy(),
                self._sign(arrays['sign'][picks].copy(), arrays['seconds'][picks], self.times.get(run['route'], [])),
                arrays['emphasis'][picks])

    def _sign(self, sign: np.ndarray, seconds: np.ndarray, times: list[float]) -> np.ndarray:
        """The sign input of a window, missed (SIGN_DROP_SHARE) or with a misreading (SIGN_FAKE_SHARE) now and then,
        its signs moved by up to SIGN_JITTER seconds."""
        rng = self.rng
        if rng.random() < SIGN_DROP_SHARE:
            sign[:] = NO_SIGN
            return sign
        if len(times) and SIGN_JITTER:
            shift = rng.uniform(-SIGN_JITTER, SIGN_JITTER)
            sign = signs_data.features_at(seconds, [t + shift for t in times])
        if rng.random() < SIGN_FAKE_SHARE:
            at = rng.randrange(len(sign))
            for k in range(at, len(sign)):
                fake = signs_data.sign_features(float(k - at))
                if sign[k, 1] == 0 or sign[k, 0] > fake[0]:
                    sign[k] = fake
        return sign


def _speed_noise(rng: random.Random, speed: np.ndarray) -> np.ndarray:
    if rng.random() < NO_SPEED_SHARE:
        return np.zeros_like(speed)
    speed[:, 0] *= rng.uniform(0.95, 1.05)
    return speed


def _predict_sequence(torch, model, store: FrameStore, route: str, positions: list[int], speed: np.ndarray,
                      device, chunk: int = 256, sign: np.ndarray | None = None) -> np.ndarray:
    """Probabilities (T, classes) of a whole sequence run from zeros with its memory, as the car would."""
    model.eval()
    with torch.no_grad():
        features = []
        for start in range(0, len(positions), chunk):
            images = store.batch([(route, positions[start:start + chunk])])[0]
            features.append(model.encode(images))
        x = model.inputs(torch.cat(features), torch.from_numpy(speed).to(device),
                         None if sign is None else torch.from_numpy(sign).to(device))[None]
        logits, _ = model.run(x)
        return torch.softmax(logits[0].float(), dim=1).cpu().numpy()


def evaluate(torch, model, store: FrameStore, dataset: dict, runs: list[dict], device, *, smooth: bool = False) -> dict:
    """sequence_metrics of a partition's sequences (and of the 5-second majority vote of the answers with smooth),
    plus the answers per sequence."""
    found, smoothed, answers = [], [], []
    for run in runs:
        arrays = _sequence_arrays(dataset, run)
        probabilities = _predict_sequence(torch, model, store, run['route'], run['positions'], arrays['speed'], device,
                                          sign=arrays['sign'])
        predicted = probabilities.argmax(axis=1)
        found.append({'labels': arrays['labels'], 'weights': arrays['weights'], 'predicted': predicted})
        if smooth:
            smoothed.append({**found[-1], 'predicted': majority(predicted, SMOOTH_SECONDS, len(data.CLASSES))})
        answers.append({'route': run['route'], 'keys': arrays['keys'], 'labels': arrays['labels'].tolist(),
                        'predicted': predicted.tolist(),
                        'probabilities': np.round(probabilities, 4).tolist()})
    result = {'metrics': sequence_metrics(found, data.CLASSES), 'answers': answers}
    if smooth:
        result['smoothed'] = sequence_metrics(smoothed, data.CLASSES)
    return result


def drive_runs(dataset: dict, min_length: int = MIN_DRIVE) -> list[dict]:
    """Every drive as the car drives it: the runs of its frames without a missing second, whatever their partition."""
    return [run for frames in dataset['frames'].values()
            for run in data.sequences(frames, ['drive'] * len(frames), min_length=min_length)]


def merge_partition(partitions: list[str]) -> str | None:
    """The partition a merge is measured in: none when a second of its ramp or of the LATE_SECONDS after it was
    trained on, else test or validation (the buffer between them is never trained on)."""
    if 'train' in partitions:
        return None
    return 'test' if 'test' in partitions else 'validation' if 'validation' in partitions else None


def release_evaluation(torch, model, store: FrameStore, dataset: dict, device, *, answers: dict | None = None) -> dict:
    """The comma's rule (road_type_metrics.comma_release) on every merge of validation and test: each drive run whole
    from zeros, as on the car, the answer debounced; the age of the last motorway sign at each second from the signs
    attached (attach_signs: dataset['sign_times'], the network's sign input too). {'sign_source', 'sign_coverage',
    'validation': release_summary, 'test': ..., 'merges': [...]}; answers: the answers per drive run already computed
    ({(route, start): predicted}), else the model runs."""
    found = []
    for run in drive_runs(dataset):
        route, positions = run['route'], run['positions']
        arrays = _sequence_arrays(dataset, run)
        entries = ramp_entries(arrays['labels'], data.CLASSES)
        partitions = dataset['split']['partitions'][route]
        measured = [(entry, merge_partition([partitions[positions[t]] for t in
                                             range(entry['ramp'], min(len(positions), entry['merge'] + LATE_SECONDS))]))
                    for entry in entries]
        measured = [(entry, partition) for entry, partition in measured if partition]
        if not measured:
            continue
        key = (route, positions[0])
        if answers is not None and key in answers:
            predicted = np.asarray(answers[key])
        else:
            predicted = _predict_sequence(torch, model, store, route, positions, arrays['speed'], device,
                                          sign=arrays['sign']).argmax(axis=1)
        held = debounced(predicted)
        age = signs_data.sign_ages(arrays['seconds'], dataset.get('sign_times', {}).get(route, []))
        for entry, partition in measured:
            result = comma_release(held, age, entry, data.CLASSES)
            said = [data.CLASSES[int(v)] for v in held[entry['ramp']:min(len(held), entry['merge'] + LATE_SECONDS)]]
            found.append({'route': route, 'partition': partition, 'ramp_key': arrays['keys'][entry['ramp']],
                          'merge_key': arrays['keys'][entry['merge']], 'before': entry['before'],
                          'ramp_seconds': entry['merge'] - entry['ramp'], **result,
                          'held_answers': ''.join(ANSWER_LETTERS[name] for name in said)})
    out = {partition: release_summary([m for m in found if m['partition'] == partition])
           for partition in ('validation', 'test')}
    coverage = dataset.get('sign_coverage') or {}
    out.update(merges=found, sign_source=coverage.get('source'), sign_coverage=coverage)
    return out


def car_release(torch, model, store: FrameStore, dataset: dict, root: Path, device) -> dict | None:
    """release_evaluation with the signs the car's code confirmed (nnslr road-type-signs), the network's sign input
    too: the comma's rule as the car runs today. None when those signs were not read on every drive."""
    view = dict(dataset)
    coverage = attach_signs(view, root, 'car')
    if coverage['read'] < coverage['drives']:
        return None
    return release_evaluation(torch, model, store, view, device)


def _dataset_sha256(dataset: dict) -> str:
    digest = hashlib.sha256()
    for route in sorted(dataset['frames']):
        for frame in dataset['frames'][route]:
            digest.update(f"{route}/{frame['key']}:{frame['label']}:{frame['weight']}:{frame['speed_mps']}\n".encode())
    return digest.hexdigest()


def _load_pictures(root: Path, dataset: dict, runs: list[dict], *, dry_run: bool, workers: int, log) -> dict:
    """{route: uint8 (frames, H, W, 3)}: the frame cache (built when missing), or for a dry run only the frames of its
    sequences read from the PNGs (nothing written)."""
    routes = sorted({run['route'] for run in runs})
    if not dry_run:
        paths = data.build_frame_cache(root, {r: dataset['frames'][r] for r in routes}, workers=workers, log=log)
        return {route: np.load(paths[route]) for route in routes}
    arrays = {}
    for route in routes:
        frames = dataset['frames'][route]
        array = np.zeros((len(frames), data.FRAME_HEIGHT, data.FRAME_WIDTH, 3), dtype=np.uint8)
        for run in runs:
            if run['route'] == route:
                for p in run['positions']:
                    array[p] = data.resized_frame(frames[p]['image'])
        arrays[route] = array
    return arrays


def _dry_runs(runs: list[dict]) -> list[dict]:
    """A dry run's few short sequences: two of train, one of validation and of test, 40 frames each."""
    picked = []
    for partition, count in (('train', 2), ('validation', 1), ('test', 1)):
        for run in [r for r in runs if r['partition'] == partition][:count]:
            picked.append({**run, 'positions': run['positions'][:40]})
    return picked


def train_road_type(data_root: Path, run_dir: Path, *, memory: str = 'gru', epochs: int = EPOCHS,
                    patience: int = PATIENCE, batch: int = BATCH, device: str | None = None, dry_run: bool = False,
                    pretrained: bool | None = None, workers: int = 8, seed: int = 0, signs: bool = True,
                    sign_source: str = signs_data.DEFAULT_SOURCE, log=_print) -> dict:
    """Train the road-type network on the drives with road types and measure it on validation and test (see the
    module's text). Writes into run_dir: split.json, dataset.json, road-type-best.pt, result.json and the answers on
    test and validation (answers-test.json, answers-validation.json). A dry run: the CPU, a few short sequences, one
    epoch of two steps, no ImageNet download, nothing written outside run_dir."""
    torch = _torch()
    data_root, run_dir = Path(data_root), Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(device or ('cpu' if dry_run else 'cuda'))
    pretrained = (not dry_run) if pretrained is None else pretrained
    dataset = data.load_dataset(data_root)
    coverage = attach_signs(dataset, data_root, sign_source)
    if signs and coverage['read'] < coverage['drives'] and not dry_run:
        raise ValueError(f"the motorway signs were read on {coverage['read']} of {coverage['drives']} drives: run "
                         'nnslr road-type-signs first')
    runs = _dry_runs(dataset['sequences']) if dry_run else dataset['sequences']
    by_partition = {name: [run for run in runs if run['partition'] == name] for name in data.PARTITIONS}
    if not by_partition['train'] or not by_partition['validation']:
        raise ValueError('the road types give no train or validation sequence: run nnslr road-types on more drives')
    (run_dir / 'split.json').write_text(json.dumps({**dataset['split'], 'sha256': dataset['split_sha256']},
                                                   sort_keys=True) + '\n', encoding='utf-8')
    (run_dir / 'dataset.json').write_text(json.dumps(dataset['summary'], indent=2, sort_keys=True) + '\n',
                                          encoding='utf-8')
    # A dry run's lines start with "dry run": the panel draws the curves of the real epochs only (job_progress).
    name = ('dry run ' if dry_run else '') + network_name(memory, signs)
    log(f'{name}: {len(by_partition["train"])} train, {len(by_partition["validation"])} validation, '
        f'{len(by_partition["test"])} test sequences; split {dataset["split_sha256"][:12]}; '
        + (f"sign input ({sign_source}): {coverage['signs']} motorway signs on {coverage['drives_with_signs']} of "
           f"{coverage['drives']} drives" if signs else 'no sign input'))
    pictures = runs if dry_run else runs + drive_runs(dataset)
    store = FrameStore(_load_pictures(data_root, dataset, pictures, dry_run=dry_run, workers=workers, log=log), device)
    model = build_road_type_net(memory, pretrained=pretrained, signs=signs).to(device)
    weights_of_class = torch.tensor(class_weights(dataset, by_partition['train']), device=device)
    backbone = list(model.backbone.parameters())
    others = [p for n, p in model.named_parameters() if not n.startswith('backbone.')]
    optimizer = torch.optim.AdamW([{'params': backbone, 'lr': BACKBONE_LEARNING_RATE},
                                   {'params': others, 'lr': LEARNING_RATE}], weight_decay=WEIGHT_DECAY)
    sampler = WindowSampler(dataset, by_partition['train'], rng)
    train_frames = sum(len(run['positions']) for run in by_partition['train'])
    steps = 2 if dry_run else max(10, math.ceil(train_frames / (batch * WINDOW)))
    window = 8 if dry_run else WINDOW
    burn = 4 if dry_run else BURN_IN
    epochs = 1 if dry_run else epochs
    batch = 2 if dry_run else batch
    total_steps, warmup = epochs * steps, steps * 2
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(s, total_steps) / total_steps)))
    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    history, best, best_epoch, since = [], None, 0, 0
    config = {'memory': memory, 'epochs': epochs, 'patience': patience, 'batch': batch, 'steps_per_epoch': steps,
              'window': window, 'burn_in': burn, 'focus_share': FOCUS_SHARE, 'drop_frame': DROP_FRAME,
              'no_speed_share': NO_SPEED_SHARE, 'merge_share': MERGE_SHARE, 'merge_weight': MERGE_WEIGHT,
              'signs': signs, 'sign_source': sign_source, 'sign_drop_share': SIGN_DROP_SHARE,
              'sign_fake_share': SIGN_FAKE_SHARE, 'sign_jitter': SIGN_JITTER, 'sign_coverage': coverage,
              'learning_rate': LEARNING_RATE,
              'backbone_learning_rate': BACKBONE_LEARNING_RATE, 'weight_decay': WEIGHT_DECAY, 'seed': seed,
              'pretrained': pretrained, 'width': data.FRAME_WIDTH, 'height': data.FRAME_HEIGHT,
              'features': FEATURES, 'hidden': HIDDEN, 'unsure_weight': data.UNSURE_WEIGHT,
              'ramp_weight': data.RAMP_WEIGHT, 'class_weights': [round(float(w), 4) for w in weights_of_class]}
    for epoch in range(1, epochs + 1):
        model.train()
        started, losses = time.time(), []
        for _ in range(steps):
            length = window
            warm = memory == 'gru' and rng.random() < 0.5
            windows = [sampler.window(length + (burn if warm else 0)) for _ in range(batch)]
            images = store.batch([(route, positions) for route, positions, *_ in windows])
            images = _augment(torch, images)
            speed = torch.from_numpy(np.stack([_speed_noise(rng, w[4]) for w in windows])).to(device)
            sign = torch.from_numpy(np.stack([w[5] for w in windows])).to(device)
            labels = torch.from_numpy(np.stack([w[2] for w in windows])).to(device)
            frame_weights = torch.from_numpy(np.stack([w[3] * w[6] for w in windows])).to(device)
            hidden = None
            if warm:
                with torch.no_grad(), torch.autocast(device.type, enabled=use_amp):
                    _, hidden = model(images[:, :burn], speed[:, :burn], None, sign[:, :burn])
                hidden = hidden.float()
                images, speed, sign, labels, frame_weights = (images[:, burn:], speed[:, burn:], sign[:, burn:],
                                                              labels[:, burn:], frame_weights[:, burn:])
            with torch.autocast(device.type, enabled=use_amp):
                logits, _ = model(images, speed, hidden, sign)
            known = labels >= 0
            per_frame = torch.nn.functional.cross_entropy(logits.float().flatten(0, 1), labels.clamp(min=0).flatten(),
                                                          reduction='none').view_as(frame_weights)
            weight = frame_weights * weights_of_class[labels.clamp(min=0)] * known
            loss = (per_frame * weight).sum() / weight.sum().clamp(min=1e-6)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            losses.append(loss.item())
        validation = evaluate(torch, model, store, dataset, by_partition['validation'], device)['metrics']
        score = validation['weight_1']['macro_f1'] or 0.0
        point = {'epoch': epoch, 'train_loss': round(float(np.mean(losses)), 4), 'val_macro_f1': score,
                 'val_accuracy': validation['weight_1']['accuracy'],
                 'val_switches_per_hour': validation['false_switches_per_hour'],
                 'seconds': round(time.time() - started, 1)}
        history.append(point)
        log(f"{name} epoch {epoch}/{epochs} train_loss {point['train_loss']:.4f} val_macro_f1 {score:.4f} "
            f"val_accuracy {point['val_accuracy'] or 0:.4f} {int(point['seconds'])}s")
        if best is None or score > best:
            best, best_epoch, since = score, epoch, 0
            torch.save({'task': 'road_type', 'architecture': f'mobilenet_v3_small+{memory}', 'memory': memory,
                        'signs': signs, 'sign_source': sign_source, 'classes': list(data.CLASSES),
                        'state_dict': model.state_dict(), 'epoch': epoch,
                        'config': config, 'split_sha256': dataset['split_sha256'],
                        'dataset_sha256': _dataset_sha256(dataset)}, run_dir / CHECKPOINT)
        else:
            since += 1
            if since >= patience:
                log(f'{name} stops early after epoch {epoch}: no better validation macro F1 for {patience} epochs')
                break
    saved = torch.load(run_dir / CHECKPOINT, map_location=device, weights_only=False)
    model.load_state_dict(saved['state_dict'])
    result = {'task': 'road_type', 'memory': memory, 'signs': signs, 'sign_source': sign_source, 'name': name,
              'dry_run': dry_run,
              'architecture': saved['architecture'], 'classes': list(data.CLASSES), 'best_epoch': best_epoch,
              'history': history, 'config': config, 'parameters': count_parameters(model),
              'flops_per_step': step_flops(model), 'split_sha256': dataset['split_sha256'],
              'dataset_sha256': saved['dataset_sha256'], 'dataset': dataset['summary'],
              'finished_at': time.strftime('%Y-%m-%dT%H:%M:%S')}
    for partition in ('validation', 'test'):
        measured = evaluate(torch, model, store, dataset, by_partition[partition], device, smooth=memory == 'none')
        result[partition] = measured['metrics']
        if 'smoothed' in measured:
            result[f'{partition}_smoothed'] = measured['smoothed']
        (run_dir / f'answers-{partition}.json').write_text(json.dumps(measured['answers']) + '\n', encoding='utf-8')
    if not dry_run:
        result['release'] = release_evaluation(torch, model, store, dataset, device)
        log(f"{name}: the comma's rule on test ({sign_source} signs): " + release_text(result['release']['test']))
        if sign_source != 'car':
            today = car_release(torch, model, store, dataset, data_root, device)
            if today:
                result['release_car'] = today
                log(f"{name}: the comma's rule on test (the car's signs): " + release_text(today['test']))
    result['step_seconds'] = _step_seconds(torch, model, device)
    (run_dir / 'result.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    log(f"{name}: best epoch {best_epoch}, test macro F1 {result['test']['weight_1']['macro_f1']}, "
        f"{result['parameters']:,} parameters, {result['flops_per_step'] / 1e6:.0f} MFLOPs a step")
    return result


def _step_seconds(torch, model, device) -> dict:
    """The time of one step at batch 1 on this machine's device (warm), a hint only: the car's GPU is another."""
    step = step_module(model).to(device)
    example = [t.to(device) for t in step_example(model, torch)]
    example[0] = torch.rand_like(example[0])
    with torch.no_grad():
        for _ in range(3):
            step(*example)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(20):
            step(*example)
        if device.type == 'cuda':
            torch.cuda.synchronize()
    return {'device': str(device), 'milliseconds': round((time.perf_counter() - started) / 20 * 1000, 2)}


def load_road_type(checkpoint: Path, device: str = 'cpu'):
    """(model in eval mode, the saved checkpoint dict)."""
    torch = _torch()
    saved = torch.load(Path(checkpoint), map_location=device, weights_only=False)
    if saved.get('task') != 'road_type':
        raise ValueError(f'{checkpoint} is not a road-type checkpoint')
    model = build_road_type_net(saved['memory'], pretrained=False, signs=bool(saved.get('signs', False)))
    model.load_state_dict(saved['state_dict'])
    return model.to(device).eval(), saved


def release_of_checkpoint(data_root: Path, checkpoint: Path, output: Path | None = None, device: str | None = None,
                          sign_source: str = signs_data.DEFAULT_SOURCE) -> dict:
    """The comma's rule (release_evaluation) and the measure per class (with the confusion) on validation and test for
    any road-type checkpoint, the first version too: {'checkpoint', 'memory', 'signs', 'release' (with the signs of
    sign_source), 'release_car' (with the car's, when read), 'metrics': {partition: sequence_metrics}}, written to
    output when given."""
    torch = _torch()
    data_root = Path(data_root)
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    model, saved = load_road_type(checkpoint, device=str(device))
    dataset = data.load_dataset(data_root)
    attach_signs(dataset, data_root, sign_source)
    store = FrameStore(_load_pictures(data_root, dataset, dataset['sequences'] + drive_runs(dataset), dry_run=False,
                                      workers=8, log=_print), device)
    result = {'checkpoint': str(checkpoint), 'memory': saved['memory'], 'signs': bool(saved.get('signs', False)),
              'split_sha256': dataset['split_sha256'], 'metrics': {},
              'release': release_evaluation(torch, model, store, dataset, device)}
    if sign_source != 'car':
        today = car_release(torch, model, store, dataset, data_root, device)
        if today:
            result['release_car'] = today
    for partition in ('validation', 'test'):
        runs = [run for run in dataset['sequences'] if run['partition'] == partition]
        result['metrics'][partition] = evaluate(torch, model, store, dataset, runs, device)['metrics']
    if output is not None:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(result, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return result


def release_text(summary: dict) -> str:
    """One line of release_summary: 'N merges: held H (ramp R, sign S); on the ramp O, in time I (median M s), late
    L, not held X'."""
    median = summary['median_release']
    return (f"{summary['merges']} merges: held {summary['held']} (ramp {summary['held_by_ramp']}, sign "
            f"{summary['held_by_sign']}); released on the ramp {summary['on_ramp']}, in time {summary['in_time']}"
            + (f' (median {median:g} s after the merge)' if median is not None else '')
            + f", late {summary['late']}, never held {summary['not_held']}")


def _numbers(value: float) -> str:
    return '-' if value is None else f'{value:.3f}' if isinstance(value, float) else str(value)


COMPARED = (('gru', 'gru'), ('gru-nosign', 'gru-nosign'), ('frame', 'frame'))
# The comma's rule with the signs of the map (where 4.01 stands: what the sign input gives once the car reads it) and
# with the signs the car's code confirms today.
_SCENARIOS = (('release', 'With the sign at the start of every entry ramp of the map (the car reading every 4.01):'),
              ('release_car', "With the signs the car's code confirms today:"))
BASELINE = 'baseline'  # <run>/road-type/baseline/release.json: an earlier version measured by nnslr release-road-type


def _variant_label(result: dict) -> str:
    if result.get('memory') == 'none':
        return 'without memory' + (' (sign)' if result.get('signs') else '')
    return 'with memory and the sign' if result.get('signs') else 'with memory, no sign'


def compare_road_type(run_root: Path) -> str:
    """The comparison on test (and validation for the comma's rule), as text, also written to
    run_root/road-type/comparison.txt: the comma's rule (when the ramp's limit ends) of every variant and of the
    baseline (an earlier version, road-type/baseline/release.json), the F1 per class and the confusion of the ramp, the
    changes of road type, the false switches and the first minute after a merge."""
    run_root = Path(run_root)
    out_dir = run_root / 'road-type'
    results = {}
    for key, folder in COMPARED:
        path = out_dir / folder / 'result.json'
        if path.is_file():
            results[key] = json.loads(path.read_text(encoding='utf-8'))
    baseline = None
    if (out_dir / BASELINE / 'release.json').is_file():
        baseline = json.loads((out_dir / BASELINE / 'release.json').read_text(encoding='utf-8'))
    rows = []  # (label, test metrics, validation metrics, {scenario: release})
    if baseline:
        rows.append((f"baseline ({Path(baseline['checkpoint']).parent.parent.parent.name})",
                     baseline['metrics']['test'], baseline['metrics']['validation'],
                     {'release': baseline.get('release'), 'release_car': baseline.get('release_car')}))
    for key, result in results.items():
        rows.append((_variant_label(result), result['test'], result['validation'],
                     {'release': result.get('release'), 'release_car': result.get('release_car')}))
    for key, result in results.items():
        if 'test_smoothed' in result:
            rows.append((_variant_label(result) + f', {SMOOTH_SECONDS} s majority', result['test_smoothed'],
                         result.get('validation_smoothed'), {}))
    width = max([32] + [len(row[0]) + 1 for row in rows])
    lines = ["The comma's rule: a number read on the ramp (its 3rd second) is the ramp's limit when the road type says "
             f'ramp then or a motorway sign was confirmed within 45 s; it ends after 8 s of motorway in a row '
             '(debounced answer). Per merge: released on the ramp (a 120 too early), in time (within 60 s of the '
             'merge, median seconds after it), late (later or never), never held.']
    for scenario, title in _SCENARIOS:
        if not any((releases.get(scenario) or {}).get('test') for *_, releases in rows):
            continue
        lines += ['', title]
        for partition in ('test', 'validation'):
            for label, _, _, releases in rows:
                release = releases.get(scenario)
                if release and release.get(partition):
                    lines.append(f'{partition:<10} {label:{width}} {release_text(release[partition])}')
    lines += ['', 'Per merge (test and validation): release seconds after the merge, how it was held, the debounced '
              'answer from the ramp on (/ ramp, M motorway, E expressway, R rural, T town, P parking)']
    for scenario, title in _SCENARIOS:
        for label, _, _, releases in rows:
            release = releases.get(scenario)
            if not release:
                continue
            lines.append(f"{label} - {title.split(':')[0].lower()}")
            for merge in release.get('merges', []):
                when = '-' if merge['release'] is None else f"{merge['release']:+d} s"
                lines.append(f"  {merge['partition']:<10} {merge['route']} {merge['ramp_key']:>6}->"
                             f"{merge['merge_key']:<6} {merge['ramp_seconds']:>3} s ramp from {merge['before'] or '?':<10} "
                             f"held by {merge['held_by'] or '-':<9} {merge['outcome']:<8} {when:>6}  "
                             f"{merge.get('held_answers', '')}")
    lines += ['', 'Road type on test (areas of the held-out drive, weight-1 frames): F1']
    lines.append(f"{'':{width}} {'macro F1':>8} " + ' '.join(f'{name[:9]:>9}' for name in data.CLASSES))
    for label, metrics, _, _ in rows:
        f1 = metrics['weight_1']
        lines.append(f'{label:{width}} {_numbers(f1["macro_f1"]):>8} '
                     + ' '.join(f'{_numbers(f1["classes"][c]["f1"]):>9}' for c in data.CLASSES))
    ramp = data.CLASSES.index('ramp')
    lines += ['', 'The ramp on test and validation (weight-1 frames): seconds of ramp answered as each class']
    for label, test, validation, _ in rows:
        for partition, metrics in (('test', test), ('validation', validation)):
            matrix = (metrics or {}).get('confusion')
            if matrix:
                said = ', '.join(f'{name} {count}' for name, count in zip(data.CLASSES, matrix[ramp]) if count)
                taken = sum(row[ramp] for i, row in enumerate(matrix) if i != ramp)
                lines.append(f'{partition:<10} {label:{width}} {sum(matrix[ramp])} s: {said or "-"}; '
                             f'other roads called ramp {taken} s')
    lines += ['', 'Transitions on test: found / count, median delay in seconds (negative: early)']
    for label, metrics, _, _ in rows:
        parts = [f"{kind} {entry['found']}/{entry['count']} {_numbers(entry['median_delay'])} s"
                 for kind, entry in metrics['transitions'].items() if entry['count']]
        lines.append(f'{label:{width}} ' + '; '.join(parts))
    lines += ['', 'False switches per hour inside one road type on test (raw / after 3 s debounce)']
    for label, metrics, _, _ in rows:
        lines.append(f"{label:{width}} {_numbers(metrics['false_switches_per_hour'])} / "
                     f"{_numbers(metrics['false_switches_per_hour_debounced'])}")
    lines += ['', f'After a merge: the first {AFTER_MERGE_SECONDS} s on the motorway after a ramp, frames right (raw / '
              'after 3 s debounce) and what was said instead']
    for label, test, validation, _ in rows:
        for partition, metrics in (('test', test), ('validation', validation)):
            merge = (metrics or {}).get('after_merge')
            if not merge:
                continue
            said = ', '.join(f'{name} {count}' for name, count in merge['said'].items() if count)
            lines.append(f"{partition:<10} {label:{width}} {_numbers(merge['accuracy'])} / "
                         f"{_numbers(merge['accuracy_debounced'])} over {merge['merges']} merges, {merge['frames']} s"
                         + (f' (said: {said})' if said else ''))
    lines.append('')
    for key, result in results.items():
        coverage = result.get('config', {}).get('sign_coverage') or {}
        lines.append(f"{result.get('name', key)}: {result['parameters']:,} parameters, "
                     f"{result['flops_per_step'] / 1e6:.0f} MFLOPs a step, best epoch {result['best_epoch']}"
                     + (f"; sign input from the {coverage.get('source', 'car')}: {coverage.get('signs')} motorway "
                        f"signs on {coverage.get('drives_with_signs', coverage.get('read'))} of "
                        f"{coverage.get('drives')} drives" if result.get('signs') else ''))
    for releases in ([row[3] for row in rows]):
        car = (releases.get('release_car') or {}).get('sign_coverage')
        if car:
            lines.append(f"The car's own signs (nnslr road-type-signs): {car['signs']} motorway signs confirmed on "
                         f"{car['drives_with_signs']} of {car['drives']} drives")
            break
    text = '\n'.join(lines) + '\n'
    out = out_dir / 'comparison.txt'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding='utf-8')
    return text


def car_input_spec(width: int = data.FRAME_WIDTH, height: int = data.FRAME_HEIGHT, signs: bool = False) -> dict:
    """How the car makes the inputs of a step from the narrow road camera (sunnypilot's speed_limits/vision.py) and,
    with signs, from speedvisiond's memory of the signs."""
    sign_steps = [
        'age = now - memory._road_sign_at  # speedvisiond\'s SpeedLimitMemory: when it last confirmed '
        'other_sign:motorway_start or expressway_start (read 5 times a second, MIN_CONFIDENCE, CONFIRM_FRAMES within '
        'CONFIRM_SECONDS); -inf before the first: age inf',
        f'sign = [[min(age, {signs_data.SIGN_HORIZON:g}) / {signs_data.SIGN_SCALE:g}, 1.0]] if age <= '
        f'{signs_data.SIGN_HORIZON:g} else [{list(NO_SIGN)}]  # road_type_signs.sign_features',
    ] if signs else []
    return {
        'camera': 'VISION_STREAM_ROAD (the narrow road camera), the whole frame: 1344 x 760 on the comma 4, as the '
                  'frames of the dataset (frames.jsonl width 1344, height 760, camera_stream narrow_road)',
        'steps': [
            'y, uv = vision.nv12_planes(buf.data, buf.width, buf.height, buf.stride, buf.uv_offset)',
            'planes = vision.yuv_to_rgb(y, uv, 0, buf.height, 0, buf.width)  # uint8 (3, 760, 1344), ffmpeg BT.601 '
            'limited range, as the dataset PNGs',
            f'planes = vision.resize(planes, {width}, {height})  # PIL BILINEAR (antialiased), columns then rows: '
            'bit-exact with the training cache (road_type_data.resized_frame)',
            'image = planes[None].astype(np.float32) / 255.0  # (1, 3, 144, 256) RGB 0..1, NCHW: no mean/std, the '
            'network normalises inside',
            f'speed = [[min(max(vEgo, 0), {data.SPEED_MAX}) / {data.SPEED_SCALE}, 1.0]]  # carState.vEgo m/s; '
            '[[0, 0]] when unknown',
            *sign_steps,
            'hidden = zeros((1, 128), float32) at the start of the drive (and after more than 5 s without a step); '
            'then the hidden_out of the previous step',
            'probabilities, hidden = session.run(None, {"image": image, "speed": speed, '
            + ('"sign": sign, ' if signs else '') + '"hidden": hidden})',
        ],
        'rate': 'one step a second (the training frames are 1 s apart; 2 Hz would make its memory twice as fast: '
                'keep 1 Hz)',
        'decode': 'road_type = classes[argmax(probabilities)], classes exactly road_types.ROAD_TYPES (town, rural, '
                  'expressway, motorway, ramp, parking); feed SpeedLimitMemory.update(road_type=...) only after the '
                  'same class 3 steps in a row (the measure\'s debounce). "ramp" is an on- or off-ramp: ramp then '
                  'motorway or expressway is the merge, where the ramp\'s limit ends without a sign',
    }


def export_road_type_onnx(checkpoint: Path, output: Path, *, verify: bool = True, data_root: Path | None = None) -> dict:
    """The one-second step as ONNX and its contract (<output>.contract.json): see the module's text. The parity with
    PyTorch is checked with onnxruntime on random inputs (rtol 1e-4, atol 1e-5) and over a rollout of 120 real
    seconds of the test when data_root is given (the hidden state fed back, the largest difference reported)."""
    torch = _torch()
    from nnslr_tools.exporting import sha256_file
    checkpoint, output = Path(checkpoint), Path(output)
    model, saved = load_road_type(checkpoint)
    step = step_module(model)
    hidden_size, signs = model.hidden_size, model.signs
    example = (torch.rand(1, 3, data.FRAME_HEIGHT, data.FRAME_WIDTH), torch.tensor([[0.8, 1.0]]),
               *([torch.tensor([[0.25, 1.0]])] if signs else []), torch.zeros(1, hidden_size))
    output.parent.mkdir(parents=True, exist_ok=True)
    names_in = ['image', 'speed', *(['sign'] if signs else []), 'hidden']
    names_out = ['probabilities', 'hidden_out']
    try:
        program = torch.onnx.export(step, example, input_names=names_in, output_names=names_out, dynamo=True,
                                    external_data=False)
        program.save(str(output))
    except ImportError as exc:
        raise ValueError('onnx_dependencies_missing: install onnx and onnxscript in the training environment') from exc
    except Exception as exc:  # torch.export refuses the graph: the TorchScript exporter
        print(f'torch.export could not export the road-type step ({type(exc).__name__}); using TorchScript', flush=True)
        torch.onnx.export(step, example, str(output), input_names=names_in, output_names=names_out, dynamo=False,
                          opset_version=17)
    verification = {'enabled': verify, 'passed': None, 'max_abs_error': None, 'rtol': 1e-4, 'atol': 1e-5,
                    'provider': 'CPUExecutionProvider'}
    if verify:
        import onnx
        import onnxruntime
        onnx.checker.check_model(onnx.load(str(output)))
        session = onnxruntime.InferenceSession(str(output), providers=['CPUExecutionProvider'])
        generator = torch.Generator().manual_seed(0)
        worst, passed = 0.0, True
        for _ in range(4):
            inputs = (torch.rand(1, 3, data.FRAME_HEIGHT, data.FRAME_WIDTH, generator=generator),
                      torch.rand(1, 2, generator=generator),
                      *([torch.rand(1, 2, generator=generator) * 2] if signs else []),
                      torch.rand(1, hidden_size, generator=generator) * 2 - 1)
            with torch.no_grad():
                expected = [t.numpy() for t in step(*inputs)]
            actual = session.run(None, dict(zip(names_in, (t.numpy() for t in inputs))))
            for e, a in zip(expected, actual):
                worst = max(worst, float(np.abs(e - a).max()))
                passed = passed and bool(np.allclose(a, e, rtol=1e-4, atol=1e-5))
        verification.update(passed=passed, max_abs_error=worst)
        if data_root is not None:
            verification['rollout'] = _rollout_check(torch, step, session, Path(data_root), hidden_size, signs=signs)
        if not passed:
            raise ValueError(f'road-type ONNX differs from PyTorch: max abs error {worst}')
    contract = {
        'schema_version': 1, 'task': 'road_type', 'architecture': saved['architecture'],
        'checkpoint': str(checkpoint), 'checkpoint_sha256': sha256_file(checkpoint),
        'dataset_sha256': saved.get('dataset_sha256'), 'split_sha256': saved.get('split_sha256'),
        'onnx_sha256': sha256_file(output), 'epoch': saved.get('epoch'),
        'input': {'name': 'image', 'shape': [1, 3, data.FRAME_HEIGHT, data.FRAME_WIDTH], 'dtype': 'float32',
                  'layout': 'NCHW', 'color': 'RGB', 'range': '0..1',
                  'normalization': 'none: ImageNet mean and std are applied inside the network',
                  'resize': {'method': 'PIL BILINEAR (antialiased), vision.resize', 'from': list(data.SOURCE_SIZE),
                             'to': [data.FRAME_WIDTH, data.FRAME_HEIGHT], 'crop': 'none, the whole frame'},
                  'source': 'narrow road camera NV12 -> vision.yuv_to_rgb -> vision.resize'},
        'inputs': [
            {'name': 'image', 'shape': [1, 3, data.FRAME_HEIGHT, data.FRAME_WIDTH], 'dtype': 'float32'},
            {'name': 'speed', 'shape': [1, 2], 'dtype': 'float32',
             'meaning': [f'min(max(vEgo m/s, 0), {data.SPEED_MAX}) / {data.SPEED_SCALE}', '1 when the speed is known, '
                         'else 0 (and the first value 0)']},
            *([{'name': 'sign', 'shape': [1, 2], 'dtype': 'float32',
                'meaning': [f'min(age, {signs_data.SIGN_HORIZON:g}) / {signs_data.SIGN_SCALE:g}: age = seconds since '
                            'SpeedLimitMemory last confirmed a motorway or expressway sign (now - _road_sign_at)',
                            f'1 when that is at most {signs_data.SIGN_HORIZON:g} s ago, else 0; never or longer ago: '
                            f'{list(NO_SIGN)}']}] if signs else []),
            {'name': 'hidden', 'shape': [1, hidden_size], 'dtype': 'float32',
             'meaning': 'the memory: zeros at the start, then the previous step\'s hidden_out'}],
        'output': {'name': 'probabilities', 'shape': [1, len(data.CLASSES)], 'classes': list(data.CLASSES),
                   'decode': 'softmax probabilities; the class is the argmax'},
        'outputs': [
            {'name': 'probabilities', 'shape': [1, len(data.CLASSES)], 'dtype': 'float32'},
            {'name': 'hidden_out', 'shape': [1, hidden_size], 'dtype': 'float32',
             'meaning': 'give it back as hidden at the next step'}],
        'step': {'rate_hz': 1.0, 'memory': saved['memory'], 'signs': signs},
        'car_input': car_input_spec(signs=signs),
        'verification': verification,
    }
    contract_path = output.with_name(output.name + '.contract.json')
    contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')
    return {'onnx': str(output), 'contract': str(contract_path), 'onnx_sha256': contract['onnx_sha256'],
            'verification': verification}


def _rollout_check(torch, step, session, data_root: Path, hidden_size: int, seconds: int = 120,
                   signs: bool = False) -> dict:
    """The step run for seconds of a test sequence by PyTorch and by onnxruntime, each with its own hidden state fed
    back: the largest difference of the probabilities and whether the answers agree. With the sign input: from 30 s
    before the first motorway sign the car confirmed on the test (the first test sequence when there is none)."""
    dataset = data.load_dataset(data_root)
    attach_signs(dataset, data_root)
    tests = [run for run in dataset['sequences'] if run['partition'] == 'test']
    if not tests:
        return {'checked': False}
    run, begin = tests[0], 0
    if signs:
        for candidate in tests:
            seen = np.flatnonzero(_sequence_arrays(dataset, candidate)['sign'][:, 1] > 0)
            if len(seen):
                run, begin = candidate, max(0, int(seen[0]) - 30)
                break
    arrays = _sequence_arrays(dataset, run)
    frames = dataset['frames'][run['route']]
    hidden_t = torch.zeros(1, hidden_size)
    hidden_o = np.zeros((1, hidden_size), dtype=np.float32)
    worst, agree, count = 0.0, 0, 0
    for offset, position in enumerate(run['positions'][begin:begin + seconds]):
        frame = frames[position]
        image = data.resized_frame(frame['image']).transpose(2, 0, 1)[None].astype(np.float32) / 255
        speed = np.array([data.speed_features(frame['speed_mps'])], dtype=np.float32)
        sign = arrays['sign'][begin + offset][None]
        inputs = {'image': image, 'speed': speed, **({'sign': sign} if signs else {}), 'hidden': hidden_o}
        with torch.no_grad():
            p_t, hidden_t = step(*(torch.from_numpy(inputs[name]) for name in inputs if name != 'hidden'), hidden_t)
        p_o, hidden_o = session.run(None, inputs)
        worst = max(worst, float(np.abs(p_t.numpy() - p_o).max()))
        agree += int(p_t.numpy().argmax() == p_o.argmax())
        count += 1
    return {'checked': True, 'route': run['route'], 'first_key': arrays['keys'][begin], 'seconds': count,
            'max_abs_error': worst, 'same_answer': agree,
            'sign_seconds': int((arrays['sign'][begin:begin + seconds, 1] > 0).sum())}
