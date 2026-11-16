"""The brain: one network with memory that says the speed to hold, from the road camera and what was read, and nothing of a
map. Here are the network and its loss, the end of the road from a person's labels to the weights
(nnslr_tools.brain_targets makes the answers, nnslr_tools.brain_train trains it, only on request).

The network, per frame: encoder (MobileNetV3-small, as the road-type network) -> features, with the inputs of the frame
(brain_targets.INPUT_SIZES) -> GRU -> h, then two heads on h:

    speed  = speed_head(h) + trust(h) * number   the speed to hold, a class of the table: the only output on the car
    works  = works_head(h)                       are we inside roadworks: only in training
    road   = road_head(h)                        which kinds of road this is, one logit a kind: only in training
    type   = type_head(h)                        the road type, a logit a type of road_types.ROAD_TYPES: only in training
    users  = users_head(h)                       who is on the road and where, a presence and a size a kind and column:
                                                 only in training
    whose  = applicability_head(h)               whose road the speed sign of each column of sign_boxes is of, a logit a
                                                 column (above 0: of another road): only in training, for now

The table is the network's own (build_brain_net's kph, the class_kph buffer, kept in every checkpoint): a class a km/h from
1 to 200 since 8 Nov 2026 (brain_targets.SPEED_CLASSES), a class every 5 km/h before. A brain of before keeps its own: its
checkpoint loads with it (brain_export.load_brain), and its contract says it to the car.

loss = cross entropy(speed, his speed) + WORKS_LOSS * binary cross entropy(works, his marks), each only on the frames with
an answer of its own. The works error goes back through h into the GRU and the layer on the picture's features, which the
speed uses too: that is how a mark changes the network, without ever being an input. works_head is one layer, left out of the car's network
(car_weights), and no code goes from works to speed. Whether it helps is measured: the same training without it
(works=False), on sites of test and validation.

The kinds of road (a narrow road, a country road, a gravel road, potholes, speed bumps, and those added in Speed review:
brain_targets.road_kinds) teach the memory the same way: road_head is one layer with a logit a kind, loss += ROAD_LOSS *
binary cross entropy of each kind on its own frames, so each kind weighs as the roadworks do. It is left out of the car's
network too (TRAINING_ONLY), so the car's brain is the same network with or without it. A kind is a column in the order of
the kinds when the training started: a kind added later is a new column of the next training.

The road users (users=True, since 12 Nov 2026, his option B): users_head is one layer on h with, for each of
usernet.KINDS (person, two-wheeler, vehicle) and each of road_users.COLUMNS (8) vertical stripes of the road frame, a
presence logit and a size (the log of the tallest one's height over the frame's). loss += USERS_LOSS * (binary cross
entropy of the presence + smooth L1 of the log size where one is there), each kind a mean over its answered cells. The
answers are brain_targets.users_answers: persons and two-wheelers the COCO detector's on every frame it read (nnslr
road-users), a vehicle only where a person confirmed a crossing the rule proposed, "nobody" only on a drive whose
proposals he reviewed whole, and weight 0 everywhere else: a frame or a drive nobody read is unknown, never nobody there.
pos_weight (brain_train) lifts the rare presences, per kind and column, at most USERS_POS_WEIGHT_MAX times. The point is
the memory: a person at the kerb, a car about to cross, are what a driver slows for before any sign, and h has to keep
them to answer here; no road user ever reaches the network's inputs, nor the car: users_head is TRAINING_ONLY.
Its design limits: 8 stripes of 168 pixels say left, centre and right, not the lane; one size a stripe, the tallest;
nothing of how far ahead but the size; the detector misses what is under COMPLETE_HEIGHT pixels, and the rule
proposes only vehicles that move across our path faster than road_users.CROSSING_MPS while we drive straight.

Whose road a sign is of (applicability=True, since 15 Nov 2026, #89: Cristian's "applicability is a head of the brain,
trained on our reviews"). Until now a small network of its own (the applicability network, one box at a time, no memory)
gave each speed sign to our road or to another, and the brain never saw a sign given to another road. The brain has what
that network has not: its memory, the turn signals, the steering wheel, the whole scene. So applicability_head is one
layer on h with a logit for each of the BOX_COLUMNS columns of sign_boxes: the speed sign the brain is given in that
column is of another road (above 0) or of ours. loss += APPLICABILITY_LOSS * binary cross entropy on the columns a
reviewed box answers (brain_targets.applicability_answers: the reviews of the gold dataset, our road or another), a
sum over those columns divided by the frames of the piece: a reviewed sign weighs as a frame of his speed, since few
frames have one and each is a person's word. pos_weight (brain_train.applicability_pos_weight) lifts the rare side,
another road, by the columns of our road over those of another in train, at most APPLICABILITY_POS_WEIGHT_MAX times. A
brain that learns it is given the boxes of every speed sign read, those of another road too (the checkpoint's
other_road_boxes; the car gives the same, brain.BrainReader), since it can only judge what it sees. The head is
TRAINING_ONLY for now (stage 1): on the car the applicability network still decides which signs are taken, and the head
is measured beside it on the same signs. Its design limits: a column, not a sign (two speed signs in one column get one
answer); the picture is 256 x 144, where a sign is a few pixels, so it judges by where the box is and how big, the
scene, the motion and what it remembers, not by the face of the sign; it answers only of a sign SignNet found.

The road type (town, rural, motorway... of the map, with a person's corrections in Speed review: brain_targets.type_answers)
teaches the memory as well: type_head is one layer with a logit a type, loss += TYPE_LOSS * cross entropy of the type on
the frames that have one. The map never reaches the network's inputs, nor the car: type_head is TRAINING_ONLY, so the car's
speed is the same network with or without it; the export gives its logits apart, only to show the road type on the comma's
screen (brain_export, since 12 Nov 2026). The memory learns that a motorway is driven faster than a town, from the
picture and the signs, and says it with the speed.

His speed is no single class but a bell over the table around it (speed_targets, SPEED_SIGMA_KPH wide): at a km/h a
class, a hard answer would cost 52 for 53 as much as 52 for 120, and a speed off the dashboard is only so exact. With the
bell a near speed is a small error, so the network learns the order of the speeds; the car still decodes by argmax. A frame
of weight 0 adds nothing, whatever its answer says.

The number goes straight to the answer too. As one value among the GRU's inputs it hardly reached the answer: with the
numbers swapped in training (5 Nov 2026) the answer moved on almost no frame when the number changed, and on few even with
one constant picture in place of the road. What `number` is in the formula at the top is the brain's number path (NUMBER_PATHS),
a word its checkpoint keeps ('number_path'; one that does not say has the first):

- 'class', every brain until 16 Nov 2026: number is 1 on the class of the number in force, so trust(h) is added to that
  class alone. On the table of 5 km/h, with one class as the answer, that could be the whole answer. Since 8 Nov 2026 the
  answer is a bell 2 km/h wide on a class a km/h, of which one class holds 0.2 at the most: with speed_head a bell at 50,
  the sign at 60 and his speed 60, the best trust takes the loss from 14.61 to 12.30 and no further. So little gradient
  reached trust: its layer stayed near its random start (L2 0.70 to 0.84, 0.58 fresh; 1.44 to 1.58 in the brains of 5 and
  6 Nov) and a swapped number moved the answer on 4 to 22 % of the test frames, 62 to 100 % before (#104). That this path
  is the cause is read from the arithmetic and the weights: the dashboard speed as the answer and the 4 steps a second
  came the same day, and no training changed the path alone.
- 'bell', the trainings since 16 Nov 2026 (#104): number has the shape of the answer, the log of a bell NUMBER_SIGMA_KPH
  wide around the number in force, on every class, and nothing on a frame without one:

      number = max(-(class_kph - the number in force)^2 / (2 * NUMBER_SIGMA_KPH^2), NUMBER_FLOOR)

  A trust of 1 adds the answer of a driver who holds the number, and the case above reaches 5.78 with speed_head as it
  was, 58 said for the 60: the rest is for speed_head and the memory to learn. NUMBER_FLOOR (-18: 6 sigma, 12 km/h from
  the number) takes the same from every class far from the number, so speed_head still says which of them, and keeps the
  values small for the comma, which runs the network with half floats (they end at 65504): it is computed as the distance
  cut at 12 km/h and then squared, the same number. For a number a sign or a map says nothing on the way is above 200 (a
  distance in km/h), and past the cut nothing is above 144 (its square) whatever comes in; the square first would pass
  65504 from 256 km/h off. Its trust starts at zero: it adds nothing until it has learnt to, and a brain that goes on
  from one of the 'class' path starts saying what that one's speed_head says (brain_train.brain_start; from a brain of
  the 'bell' path the trust is kept). brain_train.trust_size is what it learnt, in result.json.

The network still decides, with either: trust comes from the memory and is 0 on a sign it has reason to doubt, and
speed_head can say any speed. Below 0 the two differ: on the 'class' path it takes the number's one class out, on the
'bell' path it lifts every class 12 km/h and more from the number by 18 x |trust| over the number's own, so the answer
moves away from the number, to either side. It is all comparisons and arithmetic, which export as they are. A comparison with
no signs among the groups has no number and no such path.

The map's limit has the same path (map_path=True, a brain trained with the group map_limit since 14 Nov 2026 evening):

    speed = speed_head(h) + trust(h) * number + trust_map(h) * map_number

with map_number 1 on the class of the limit the map has, all 0 with no map; in a brain of the 'bell' path, the same bell
around the map's limit. The first brain with the map as one more value
among the GRU's inputs did not read it: with the map turned off its answer moved by more than 2 km/h on 4 % of the
validation frames and its error stayed the same. trust_map starts at zero, so a brain that goes on from another starts
saying what that one said, and it comes from the memory as trust does: the network decides where the map is to be believed.

The limit ahead (ahead=True, the brains of 7 to 11 Nov 2026 only): two more heads on the same h,

    next   = next_head(h) + trust_next(h) * number   the next limit: a class of the speed's table, or the last one, none
    metres = metres_head(h) + metres_gate(h) * sign  the metres to it, over brain_targets.METRES_SCALE

with sign vision.SignDistance's metres to the sign ahead (sign_distance among the inputs). No brain is trained with them
since 12 Nov 2026, his rule: the speed to hold is the brain's only word to openpilot, and it lowers that speed itself
before a sign. So they have no answer and no loss here, and a new brain is built without them. They are still built for a
checkpoint that has them (its 'ahead'), so that such a brain loads, replays and exports as it was (brain_export): the
car leaves the two outputs unread. Their number is the 'class' path's: no brain of the 'bell' path has them.

The picture's features. pool 'mean' is the road-type network's: the mean of the 8 x 5 map keeps what is in a frame and loses
most of where, and a truck in the lane beside ours is a matter of where. pool 'grid' keeps the map: a 1 x 1 convolution to
GRID_CHANNELS, then all its cells into one layer. Which one sees works is measured too.

Batch norm stays frozen in training: its statistics of a batch would let the frames after a moment reach it. On the car the
network only ever has the past, so here too. With its statistics frozen and one sequence a step nothing brings a moving
encoder back: in the first run (5 Nov 2026) its output grew 250 times, the GRU saturated, the signs reached nothing and the
network said one speed on every frame. So the encoder does not learn, for now: a few drives would only teach it their roads.
The picture's features go through a LayerNorm, so that they keep the scale of the inputs beside them in the GRU.
"""
from __future__ import annotations

from nnslr_tools.brain_targets import BRAIN_INPUTS, INPUT_SIZES, MAP_GROUP, SPEED_CLASSES, class_kph
from nnslr_tools.road_users import COLUMNS as _USER_COLUMNS
from nnslr_tools.usernet import KINDS as _USER_KINDS
from nnslr_tools.road_type_net import FEATURES, HIDDEN, IMAGENET_MEAN, IMAGENET_STD, _torch

WORKS_LOSS = 0.3
ROAD_LOSS = 0.3  # each kind of road
TYPE_LOSS = 0.3  # the road type
USERS_LOSS = 0.3  # each kind of road user: its presence and size over the columns
USERS_POS_WEIGHT_MAX = 10.0  # the most a presence of a rare kind and column is lifted (brain_train.users_pos_weight)
USERS_MIN_HEIGHT = 1e-3  # the size's floor before its log: a present one has at least this height over the frame's
APPLICABILITY_LOSS = 1.0  # whose road a sign is of: a reviewed sign weighs as a frame of his speed
APPLICABILITY_POS_WEIGHT_MAX = 10.0  # the most a sign of another road is lifted (brain_train.applicability_pos_weight)
APPLICABILITY_COLUMNS = BRAIN_INPUTS.BOX_COLUMNS  # applicability_head's outputs: the columns of sign_boxes
SPEED_SIGMA_KPH = 2.0  # the width of the bell around his speed (speed_targets)
# How the number in force reaches the answer (the module's text): added to its class alone, or as a bell around it.
# The first is the path of a checkpoint that does not say: every brain trained before 16 Nov 2026 (#104).
NUMBER_PATHS = ('class', 'bell')
NUMBER_FLOOR = -18.0  # the least the 'bell' path gives a class: its bell 6 sigma from the number, and beyond
# The width of the 'bell' path's bell: the answer's own width when the path was made (SPEED_SIGMA_KPH on 16 Nov 2026).
# It and NUMBER_FLOOR belong to the brains trained with the path, as near_kph belongs to a table: a brain is built
# again from the word alone, so another width or floor is another word of NUMBER_PATHS, never another number here. A
# change of the loss's SPEED_SIGMA_KPH leaves them alone.
NUMBER_SIGMA_KPH = 2.0
POOLS = ('mean', 'grid')
GRID_CHANNELS = 32
GRID_CELLS = 8 * 5  # the encoder's map of a 256 x 144 frame
BACKBONE_CHANNELS = 576
# Out of the car's network.
TRAINING_ONLY = ('works_head.', 'road_head.', 'type_head.', 'users_head.', 'applicability_head.')
# users_head's layout: (presence, size) x kinds x columns.
USER_KINDS = _USER_KINDS
USER_COLUMNS = _USER_COLUMNS
USER_OUTPUTS = 2 * len(USER_KINDS) * USER_COLUMNS


def training_table() -> list[float]:
    """The km/h of each class of the brains trained today (brain_targets): 1 to 200."""
    return [float(class_kph(index)) for index in range(SPEED_CLASSES)]


def _table(kph) -> list[float]:
    """kph as a list of km/h in order, one a class: training_table() for None."""
    if kph is None:
        return training_table()
    values = [float(value) for value in (kph.flatten().tolist() if hasattr(kph, 'flatten') else kph)]
    if not values or any(b <= a for a, b in zip(values, values[1:])):
        raise ValueError(f'a table of the speeds is km/h going up, one a class: {values[:5]}...')
    return values


def number_path_of(saved: dict) -> str:
    """The number path of a brain, from what its checkpoint or its result.json says of itself ('number_path'): 'class'
    for one that does not say. A word this code does not know is refused: built as another path, the weights would say
    other speeds without a word."""
    path = saved.get('number_path') or NUMBER_PATHS[0]
    if path not in NUMBER_PATHS:
        raise ValueError(f'this brain has a number path this code does not know ({path!r}): one of {NUMBER_PATHS}')
    return path


def build_brain_net(groups: tuple[str, ...], *, pool: str = 'grid', works: bool = True, ahead: bool = False,
                    pretrained: bool = True, features: int = FEATURES, hidden: int = HIDDEN, kph=None, road_kinds: int = 0,
                    road_types: int = 0, users: bool = False, map_path: bool = False, applicability: bool = False,
                    sizes: dict | None = None, number_path: str = NUMBER_PATHS[0]):
    """The network (torch.nn.Module): forward(images (B, T, 3, H, W) float 0..1, extra (B, T, inputs of the groups), state
    (B, hidden) or None) -> (speed logits (B, T, classes), works logit (B, T) or None, state); with ahead=True asked of
    forward (a network built with ahead: only for a checkpoint of 7 to 11 Nov 2026 that has those heads, never for a
    training) also next logits (B, T, classes + 1) and metres (B, T); with road=True asked of
    forward, then the logits of the kinds of road (B, T, road_kinds), or None for a network built without them; with
    types=True, the logits of the road type (B, T, road_types), or None for a network built without them; with users=True,
    last, users_head's (B, T, 2, kinds, columns): presence logits ([:, :, 0]) and log sizes ([:, :, 1]) of the road
    users, or None for a network built without it; with applicability=True, last, applicability_head's logits (B, T,
    BOX_COLUMNS), above 0 where the speed sign of that column of sign_boxes is of another road, or None for a network
    built without it (outputs reads them by name).
    kph: the km/h of each class, in order (a list, or the class_kph buffer of a checkpoint): the heads have as many classes,
    and a number in force falls on the class within half the table's step of it. None: the table of today's training.
    map_path: the map's limit goes straight to the answer too (trust_map; the module's text): only with map_limit among
    the groups. A checkpoint has it when it has trust_map's weights. applicability: with applicability_head, whose
    road the speed sign of each column of sign_boxes is of (the module's text): only with sign_boxes among the groups.
    number_path: how the number in force, and the map's limit of a network with map_path, reach the answer (the module's
    text): 'class', the path of every checkpoint that does not say (number_path_of), or 'bell', a training of today's
    (brain_train); never 'bell' with ahead."""
    torch = _torch()
    from torch import nn
    from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

    if pool not in POOLS:
        raise ValueError(f'pool must be one of {POOLS}')
    if number_path not in NUMBER_PATHS:
        raise ValueError(f'number_path must be one of {NUMBER_PATHS}')
    if ahead and number_path != NUMBER_PATHS[0]:
        raise ValueError('the heads of the limit ahead are of the brains of 7 to 11 Nov 2026, all of the class path')
    table = _table(kph)
    classes = len(table)
    # A plain number, no buffer: a checkpoint of before has none to load, and the export writes it as a constant.
    near_kph = (min(b - a for a, b in zip(table, table[1:])) if classes > 1 else table[0]) / 2
    # The 'bell' path: how far from the number its bell reaches NUMBER_FLOOR (12 km/h), and what a squared distance is
    # multiplied by. Plain numbers too.
    far_kph = NUMBER_SIGMA_KPH * (-2 * NUMBER_FLOOR) ** 0.5
    bell_scale = -1 / (2 * NUMBER_SIGMA_KPH ** 2)
    # The sizes of this brain's groups (brain_targets.brain_sizes): an earlier one has fewer kinds of other sign.
    sizes = INPUT_SIZES if sizes is None else sizes
    extra = sum(sizes[name] for name in groups)
    signs_at = sum(sizes[name] for name in groups[:groups.index('signs')]) if 'signs' in groups else None
    distance_at = (sum(sizes[name] for name in groups[:groups.index('sign_distance')])
                   if 'sign_distance' in groups else None)
    if map_path and MAP_GROUP not in groups:
        raise ValueError(f'the map’s limit has a path to the answer only when {MAP_GROUP} is among the groups')
    map_at = sum(sizes[name] for name in groups[:groups.index(MAP_GROUP)]) if map_path else None
    if applicability and 'sign_boxes' not in groups:
        raise ValueError('whose road a sign is of is asked of the columns of sign_boxes: it must be among the groups')

    class BrainNet(nn.Module):
        def __init__(self):
            super().__init__()
            backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None)
            self.groups, self.pool = groups, pool
            self.register_buffer('mean', torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
            self.register_buffer('std', torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
            self.backbone = backbone.features.requires_grad_(False)
            if pool == 'grid':
                self.squeeze = nn.Conv2d(BACKBONE_CHANNELS, GRID_CHANNELS, 1)
            width = BACKBONE_CHANNELS if pool == 'mean' else GRID_CHANNELS * GRID_CELLS
            self.project = nn.Sequential(nn.Linear(width, features), nn.LayerNorm(features), nn.Hardswish(), nn.Dropout(0.2))
            self.gru = nn.GRU(features + extra, hidden, batch_first=True)
            self.speed_head = nn.Linear(hidden, classes)
            self.trust = nn.Linear(hidden, 1)
            self.number_path = number_path
            if number_path == 'bell':
                # At zero, as trust_map: the bell adds nothing until trust has learnt to, so what the layer holds at
                # the end is what it learnt (brain_train.trust_size).
                nn.init.zeros_(self.trust.weight)
                nn.init.zeros_(self.trust.bias)
            self.trust_map = nn.Linear(hidden, 1) if map_path else None
            if self.trust_map is not None:
                # At zero: the answer of a brain that goes on from another starts as that one's, the map adding nothing yet.
                nn.init.zeros_(self.trust_map.weight)
                nn.init.zeros_(self.trust_map.bias)
            self.works_head = nn.Linear(hidden, 1) if works else None
            self.road_head = nn.Linear(hidden, road_kinds) if road_kinds else None
            self.type_head = nn.Linear(hidden, road_types) if road_types else None
            self.users_head = nn.Linear(hidden, USER_OUTPUTS) if users else None
            self.applicability_head = nn.Linear(hidden, APPLICABILITY_COLUMNS) if applicability else None
            self.ahead = ahead
            if ahead:
                self.next_head = nn.Linear(hidden, classes + 1)
                self.trust_next = nn.Linear(hidden, 1)
                self.metres_head = nn.Linear(hidden, 1)
                self.metres_gate = nn.Linear(hidden, 1)
            self.register_buffer('class_kph', torch.tensor(table, dtype=torch.float32))
            self.near_kph = near_kph

        def train(self, mode: bool = True):
            super().train(mode)
            for layer in self.backbone.modules():
                if isinstance(layer, nn.BatchNorm2d):
                    layer.eval()
            return self

        def encode(self, images):
            """(N, 3, H, W) 0..1 -> (N, features)."""
            x = self.backbone((images - self.mean) / self.std)
            x = x.mean(dim=(2, 3)) if self.pool == 'mean' else self.squeeze(x).flatten(1)
            return self.project(x)

        def number_in_force(self, extra_input):
            """(B, T, inputs) -> (B, T, classes): 1 on the class of the number in force, all 0 with none."""
            if signs_at is None:
                return extra_input.new_zeros(*extra_input.shape[:-1], classes)
            signs = extra_input[..., signs_at:signs_at + INPUT_SIZES['signs']]
            held = (signs[..., 1:2] > 0) & (signs[..., 6:7] == 0)
            near = (signs[..., :1] * BRAIN_INPUTS.SIGN_KPH_SCALE - self.class_kph).abs() < self.near_kph
            return (near & held).to(extra_input.dtype)

        def map_number(self, extra_input):
            """(B, T, inputs) -> (B, T, classes): 1 on the class of the limit the map has, all 0 with no map. Only a
            network with the map's path (trust_map)."""
            limit = extra_input[..., map_at:map_at + INPUT_SIZES[MAP_GROUP]]
            near = (limit[..., :1] * BRAIN_INPUTS.SIGN_KPH_SCALE - self.class_kph).abs() < self.near_kph
            return (near & (limit[..., 1:2] > 0)).to(extra_input.dtype)

        def bell(self, number, held):
            """(B, T, 1) a number over SIGN_KPH_SCALE and whether it counts -> (B, T, classes): the 'bell' path's
            number (the module's text), max(-(class_kph - number)^2 / (2 * NUMBER_SIGMA_KPH^2), NUMBER_FLOOR) where it
            counts and 0 where not. The distance is cut at far_kph before its square, which is the same number: past
            the cut no value is above 144, whatever comes in, for the comma's half floats."""
            away = (self.class_kph - number * BRAIN_INPUTS.SIGN_KPH_SCALE).clamp(-far_kph, far_kph)
            return away * away * bell_scale * held.to(number.dtype)

        def number_bell(self, extra_input):
            """(B, T, inputs) -> (B, T, classes): the bell around the number in force, all 0 with none. Only for a
            network of the 'bell' path."""
            if signs_at is None:
                return extra_input.new_zeros(*extra_input.shape[:-1], classes)
            signs = extra_input[..., signs_at:signs_at + INPUT_SIZES['signs']]
            return self.bell(signs[..., :1], (signs[..., 1:2] > 0) & (signs[..., 6:7] == 0))

        def map_bell(self, extra_input):
            """(B, T, inputs) -> (B, T, classes): the bell around the limit the map has, all 0 with no map. Only for a
            network of the 'bell' path with the map's path (trust_map)."""
            limit = extra_input[..., map_at:map_at + INPUT_SIZES[MAP_GROUP]]
            return self.bell(limit[..., :1], limit[..., 1:2] > 0)

        def speed(self, out, extra_input):
            """The speed's logits from the memory's output and the inputs: forward's and the car's step (brain_export)."""
            if self.number_path == 'bell':
                said = self.speed_head(out) + self.trust(out) * self.number_bell(extra_input)
                return said if self.trust_map is None else said + self.trust_map(out) * self.map_bell(extra_input)
            said = self.speed_head(out) + self.trust(out) * self.number_in_force(extra_input)
            return said if self.trust_map is None else said + self.trust_map(out) * self.map_number(extra_input)

        def sign_metres(self, extra_input):
            """(B, T, inputs) -> (B, T): vision.SignDistance's metres over METRES_SCALE, 0 without a sign or the input:
            what the metres head of a brain with the limit ahead adds to."""
            if distance_at is None:
                return extra_input.new_zeros(extra_input.shape[:-1])
            return extra_input[..., distance_at]

        def forward(self, images, extra_input, state=None, ahead: bool = False, road: bool = False, types: bool = False,
                    users: bool = False, applicability: bool = False):
            b, t = images.shape[:2]
            extra_input = extra_input.to(images.dtype)
            x = torch.cat([self.encode(images.flatten(0, 1)).view(b, t, -1), extra_input], dim=-1)
            out, last = self.gru(x, None if state is None else state[None].contiguous())
            number = self.number_in_force(extra_input)
            speed = self.speed(out, extra_input)
            works = None if self.works_head is None else self.works_head(out)[..., 0]
            found = (speed, works, last[0])
            if ahead:
                following = self.next_head(out) + self.trust_next(out) * torch.cat([number, number[..., :1] * 0], dim=-1)
                metres = self.metres_head(out)[..., 0] + self.metres_gate(out)[..., 0] * self.sign_metres(extra_input)
                found += (following, metres)
            if road:
                found += (None if self.road_head is None else self.road_head(out),)
            if types:
                found += (None if self.type_head is None else self.type_head(out),)
            if users:
                found += (None if self.users_head is None else
                          self.users_head(out).view(b, t, 2, len(USER_KINDS), USER_COLUMNS),)
            if applicability:
                found += (None if self.applicability_head is None else self.applicability_head(out),)
            return found

    return BrainNet()


def outputs(said: tuple, *, ahead: bool = False, road: bool = False, types: bool = False, users: bool = False,
            applicability: bool = False) -> dict:
    """What forward gave, asked so, by name: speed, works, state, and next, metres, road, types, users, applicability
    where asked."""
    found = dict(zip(('speed', 'works', 'state'), said[:3]))
    rest = list(said[3:])
    if ahead:
        found['next'], found['metres'] = rest.pop(0), rest.pop(0)
    if road:
        found['road'] = rest.pop(0)
    if types:
        found['types'] = rest.pop(0)
    if users:
        found['users'] = rest.pop(0)
    if applicability:
        found['applicability'] = rest.pop(0)
    return found


def speed_targets(speed, kph=None, sigma_kph: float = SPEED_SIGMA_KPH):
    """(..., classes) the answer of each frame as chances over the classes: a bell of sigma_kph around the km/h of its class,
    cut at the ends of the table and summing to 1. speed: classes (long), -1 (no class) as the first one, a frame that
    has no weight anyway. kph: the km/h of each class (the network's class_kph), None the table of today's training."""
    torch = _torch()
    table = (torch.tensor(training_table(), device=speed.device) if kph is None else
             torch.as_tensor(kph, device=speed.device).flatten()).to(torch.float32)
    answer = table[speed.clamp(0, len(table) - 1)]
    bell = torch.exp(-0.5 * ((table - answer[..., None]) / sigma_kph) ** 2)
    return bell / bell.sum(dim=-1, keepdim=True)


def brain_loss(speed_logits, works_logit, speed, speed_weight, works, works_weight, *, works_share: float = WORKS_LOSS,
               frames: int | None = None, kph=None, road: tuple | None = None, types: tuple | None = None,
               users: tuple | None = None, applicability: tuple | None = None):
    """The error of a batch, (B, T) answers and weights of brain_targets: each task a mean over its own frames, 0 with none.
    A frame of weight 0 gives no gradient: not examined is not an answer, and an unmarked stretch is not "no works".
    The speed's answer is speed_targets' bell over the table kph (the network's class_kph; None: today's training's).
    frames: divide by this many instead, so that a frame weighs the same in a piece with one answer as in a full one.
    road: (road logits, road answers, road_weight), (B, T, kinds), of a network with the kinds of road: ROAD_LOSS a kind.
    types: (type logits (B, T, types), the type's class (B, T), -1 none, type_weight), of a network with the road type.
    users: (users_head's (B, T, 2, kinds, columns), presence, height over the frame's, weight, each (B, T, kinds, columns),
    pos_weight (kinds, columns) or None), of a network with the road users: USERS_LOSS a kind, the presence's binary cross
    entropy over its answered cells and the log size's smooth L1 over those with one there; with frames, each kind is
    divided by frames x columns, so that a frame with every column answered weighs as a frame of a kind of road.
    applicability: (applicability_head's logits, the answers, 1 of another road, their weight, each (B, T, columns),
    pos_weight (a scalar tensor) or None), of a network that learns whose road a sign is of: APPLICABILITY_LOSS times
    the binary cross entropy over the answered columns; with frames, their sum over frames, so that a reviewed sign
    weighs as a frame of his speed."""
    torch = _torch()
    from torch.nn import functional

    def mean(errors, weight):
        return (errors * weight).sum() / (frames or weight.sum().clamp_min(1))

    weight = speed_weight.flatten()
    errors = functional.cross_entropy(speed_logits.flatten(0, 1), speed_targets(speed.flatten(), kph), reduction='none')
    total = mean(torch.where(weight > 0, errors, torch.zeros_like(errors)), weight)
    if works_logit is not None:
        total = total + works_share * mean(functional.binary_cross_entropy_with_logits(works_logit, works, reduction='none'),
                                           works_weight)
    if road is not None and road[0] is not None:
        road_logits, answers, road_weight = road
        errors = functional.binary_cross_entropy_with_logits(road_logits, answers, reduction='none')
        for kind in range(road_logits.shape[-1]):
            total = total + ROAD_LOSS * mean(errors[..., kind], road_weight[..., kind])
    if types is not None and types[0] is not None:
        type_logits, answers, type_weight = types
        errors = functional.cross_entropy(type_logits.flatten(0, 1), answers.flatten().clamp_min(0), reduction='none')
        total = total + TYPE_LOSS * mean(errors, type_weight.flatten())
    if users is not None and users[0] is not None:
        said, presence, height, user_weight, pos_weight = users
        columns = said.shape[-1]
        found = functional.binary_cross_entropy_with_logits(said[:, :, 0], presence, pos_weight=pos_weight,
                                                            reduction='none')
        there = user_weight * (presence > 0).to(user_weight.dtype)
        size = functional.smooth_l1_loss(said[:, :, 1], height.clamp_min(USERS_MIN_HEIGHT).log(), reduction='none')

        def cells(errors, weight):
            return (errors * weight).sum() / ((frames * columns) if frames else weight.sum().clamp_min(1))

        for kind in range(said.shape[3]):
            total = total + USERS_LOSS * (cells(found[..., kind, :], user_weight[..., kind, :])
                                          + cells(size[..., kind, :], there[..., kind, :]))
    if applicability is not None and applicability[0] is not None:
        said, answers, sign_weight, pos_weight = applicability
        errors = functional.binary_cross_entropy_with_logits(said, answers, pos_weight=pos_weight, reduction='none')
        total = total + APPLICABILITY_LOSS * mean(errors, sign_weight)
    return total


def car_weights(model) -> dict:
    """The weights that go to the car: all but works_head, road_head, type_head, users_head and applicability_head (the
    heads of the limit ahead, of a brain that has them, go too)."""
    return {name: value for name, value in model.state_dict().items() if not name.startswith(TRAINING_ONLY)}
