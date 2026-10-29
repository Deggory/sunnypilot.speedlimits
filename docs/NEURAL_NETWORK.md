# The speed-limit pipeline: cutter → classifier → decider

A plan, started on 24 October 2026 from Codex's "teacher and student" proposal, Cristian's scheme of three separate networks, and the real data on the server. The first runs of that day are measured below.

![The pipeline](neural-network.png)

## Where things run

**All training happens on the server**, on the V100. The comma never trains anything.

- **The networks for the car** are trained on the server from the reviews, then copied to the car. They run on **Chestnut**, comma's external GPU, not on the comma four's own chip (decided on 24 October 2026): the chip would need the frame shrunk until distant signs disappear, while Chestnut takes the full 1344 × 760 frame. Chestnut also runs openpilot's driving model, which always comes first: the room left next to it is still to be measured. In the car they learn nothing and see only the past, since there are no future frames while driving.
- **The same networks prepare new drives on the server** (Prepare, the "Our networks" proposer). There a drive can also be looked at with its later frames, its GPS and the map: that extra step (the teacher below) makes better proposals than the car can, without needing a different network.

## Three networks, each trained and measured on its own

1. **Cut: the cutter finds every box that could be a sign.**
   - It looks at the frame of the main road camera (1344 × 760 on the comma four, full resolution), scaled down, plus the band where signs appear.
   - It must miss nothing. False boxes are fine, because the classifier drops them, so it can be a small network.
   - It is measured on **signs missed, by size**, and on how many boxes per frame it passes on.
   - A new cutter also cuts the other traffic signs (`--cutter-families all`, the default; `speed` is the speed signs only, as before), so that the classifier sees them and says what they are: 34 more boxes in train, 12 in validation and 54 in test, all reviewed by people. Its speed-sign recall is still reported alone (`recall_by_family`), to compare with the older cutters.
   - Today's detector plays this role and finds only 6% of the test signs (see the results).
2. **Classify: the classifier says what each box is.**
   - It takes the box cut from the full-resolution frame, never from the scaled-down one.
   - Its answers are a speed value, end of limit, zone, variable display, "?" (a limit it cannot read), another sign with its kind (town entry, direction, give way…), or **not a sign**.
   - Today's reader plays this role. It still lacks "not a sign": the 2,431 model boxes that reviewers removed, plus the cutter's own mistakes, teach it.
   - `scripts/run_classifier.sh` (the **Train the classifier** button) trains it with those classes (`nnslr train --classes v2`): the removed proposals that overlap no kept box, the boxes marked "not a sign" and up to 3 × the signs of false boxes mined with the cutter in use (`nnslr mine-negatives`), darker crops and jittered boxes. It then measures it alone and the whole chain on the test (`nnslr evaluate-chain`: signs found **and** read right, false boxes per frame that survive), next to the old reader. First results below.
   - Frames an AI reviewed before the other-sign rule have their other signs without a box. There, the other-sign proposals the AI removed and the cutter's boxes scoring 0.3 or more are not used as "not a sign" (see the results).
   - It is measured on right answers per sign.
3. **Decide: the decider says whom each sign applies to.**
   - It takes the list of recognised signs with their position and size, the scene as a small image, and the car's speed and steering. The signs of a frame look at each other, as with two 80s on one gantry.
   - It answers ours, another or abstain, with how sure it is.
   - Experiment B supports it: position and speed raised the "Another" signs found from 59% to 77%, while the full image made the network memorise places.
4. **Over time**, with the past only: the same sign is followed across frames, as "Apply previous" does (the tracks of 323 signs are already computed with `nnslr tracks`). The reading improves as the car gets closer, and the sign is either ahead or passed.
5. **The limit now** is rules and explicit state, not a network, so the reason for a limit is always visible.
   - The country comes from the GPS, and its rules interpret what the networks read: a town entry in Poland means 50.
   - The logic is written and tested here, in `speed_vision_core`, and replayed offline on the recorded routes.
   - Only then does it go into sunnypilot, as a separate change: the networks run on Chestnut through openpilot's model process (`modeld` owns Chestnut: no second process may start it), and Speed Limit Assist takes it as a third source next to the OSM map and the car's CAN, which the 3008 does not have. At first the limit is only shown on the screen.

## Why the cutter needs pixels

The cutter has to find a sign before anything can cut it out, and the median sign is only about 20 px wide in the frame:

| Cutter input | A 20 px sign becomes | Cost compared with 320 |
|---|---|---|
| 320 × 320 (today, squashed) | about 5 × 8 px | 1× |
| 640 × 360 | about 10 px | about 2.3× |
| 960 × 544 | about 14 px | about 5× |
| 1344 × 760 (full) | 20 px | about 10× |

The cutter only answers "could this be a sign?" and may leave false boxes, so it can stay small. There are three ways to keep it cheap:
- give it the full resolution only on the band where signs appear;
- run it a few times a second instead of 20;
- once a sign is found, follow it with the tracker and crop it at full resolution.

`scripts/run_detector_sizes.sh` measures each size: signs missed by size at low score thresholds, boxes per frame and time per frame.

## First results (24 October 2026)

Measured on the test areas of route 0000012d, which no training sees. 35% of those frames are at dusk; only 7% of the training frames are dark.

**Detector (run `2026-10-24_1452`, input 320 × 320)**

| Sign width | Found |
|---|---|
| Under 40 px | 0 of 68 |
| 40 px and wider | 5 of 12 |
| All | 6%, with 529 false boxes |

Its validation loss was best at epoch 8 of 20: after that it memorised the training images. Training now stops after 5 epochs without improvement.

**Cutter comparison (`scripts/run_detector_sizes.sh 2026-10-24_1452`)**

All three sizes use small default boxes (the smallest is 0.02 of the image side), 10% of the sign-free frames per epoch, and the best epoch chosen by validation recall. Scores are at the cutter's low threshold (0.1), on the 80 test signs.

| Input | Signs found | < 12 px | 12–20 px | 20–40 px | ≥ 40 px | Hand-reviewed | Boxes per frame |
|---|---|---|---|---|---|---|---|
| 320 × 320, default boxes 0.2 (the run above) | 6% | 0% | 0% | 0% | 42% | | |
| 320 × 320 | 38% | 11% | 19% | 58% | 92% | 48% | 80 |
| 640 × 360 | 62% | 17% | 61% | 89% | 92% | 78% | 237 |
| **960 × 544** | **79%** | **44%** | **81%** | **100%** | 92% | **96%** | **42** |

- **960 × 544 is the cutter in use** (Models tab). The small default boxes alone take 320 × 320 from 6% to 38%, and the larger input takes it to 79%.
- **Cost:** on the V100, 11.1, 11.3 and 13.3 ms per frame, which says little about the comma; 960 × 544 has about 5× the pixels of 320 × 320.
- **Every box goes to the classifier, and almost all are junk.** Dropping them is the classifier's job.
- **Still hard:** signs under 12 px. The next steps are the full-resolution band and more drives.

**Reader (same run)**

| | Right answers |
|---|---|
| All test signs | 71% |
| Signs reviewed by hand | 60% |
| Speed limit 60 | 81% |
| Speed limit, number unreadable | 91% |
| Other signs | 57% |

Other signs were mostly taken for variable displays: the AI reviewer used to leave them out of the boxes, while people box them. The rule now asks for them too, with their kind.

**Classifier (run `2026-10-24_1750-classifier`, cutter 960 × 544 at score 0.1, test)**

| | New classifier | Old reader |
|---|---|---|
| Speed signs found **and** read right | 61% (49 of 80) | 66% (53 of 80) |
| … of those a person confirmed | 70% (16 of 23) | 65% (15 of 23) |
| False boxes per frame left after the classifier | **0.26** (from 41.5) | 41.5 (it has no "not a sign") |
| Other signs read right | **0 of 54** | |

- **It drops 99.4% of the cutter's false boxes,** which is its job. The ones left are mostly read as "speed, number unreadable" or "variable display".
- **It called every other sign "not a sign".** The training had 34 other signs against 5,906 "not a sign" crops, and many of those crops were real signs:
  - before the rule to box other signs, the AI reviewer removed the other-sign proposals: 1,339 of the 1,434 removed proposals of train were proposed as other signs, mostly pedestrian crossings, give way, no parking and direction signs;
  - about 15 of the 40 highest-scoring cutter boxes mined in those frames were real signs: no entry, no overtaking, roadworks, end of motorway.
- **Fixed:** in those frames, removed other-sign proposals and mined boxes scoring 0.3 or more are no longer "not a sign" (`BOXES_OTHER_SIGNS`, `mine-negatives --sign-score`). Run `2026-10-24_1835-classifier` (cutter 1344 × 760, score 0.1, test) measures the fix: other signs read right went from 0 to **18 of 54**; speed signs found and read right **66%** (53 of 80; old reader 68%); the cutter's 210 false boxes per frame drop to **0.49**. In train, 650 likely signs were left out of the mined boxes.
- Those 1,339 removed proposals are other signs waiting for their kind: reviewing them would give the classifier its first real set of other signs.

**Proposers of Prepare (`helpers/measure_proposer.sh`, 25 October 2026, the 604 test frames)**

| Proposer | Speed signs proposed with the right number | Speed signs found | All signs found | False proposals per frame | Seconds per frame |
|---|---|---|---|---|---|
| Vietnamese YOLO (Hugging Face) | 2 of 80 | 27 of 80 | 45 of 134 | 0.08 | 0.5 |
| Our cutter + classifier | **51 of 80** | **60 of 80** | 66 of 134 | 0.45 | 0.8 |

The Vietnamese model, which proposed the signs of every route so far, finds a third of the test speed signs and reads almost none right: reviewers drew and labelled the rest by hand. Our networks propose most of them with the right number, at the cost of about one false proposal every two frames to delete.

**Experiment B (run `2026-10-24_expB`, on the reviewed boxes)**

| Variant | Reading on validation | "Another" found on validation | Reading on test |
|---|---|---|---|
| crop | 85% | 59% | 62% |
| crop + position and speed | 82% | 77% | 63% |
| crop + position + full image | 69% | 59% | 46% |

The full image reached 99.7% on training: it learned the places. The test has no "Another" sign yet, so "whom it applies to" can only be measured on validation, which has 17 cases.

## Cameras and devices

- The comma four records two road cameras, both 1344 × 760: the main road camera (`fcamera.hevc`) and the wide-angle one (`ecamera.hevc`). We use the main one, because its narrower view gives each sign more pixels. The wide one may help at the moment of passing, when a sign at the side has left the main view.
- The comma 3X has three cameras (main road, wide road, driver), and its road cameras record 1928 × 1208 with another sensor and lens. Networks trained on comma four frames do not work on a 3X by themselves. Three things make them work:
  - every frame is converted into a common "standard camera" from the device's calibration, so a sign at 100 m has the same size on every device, as openpilot does for its driving model;
  - training uses variations of size and colour;
  - drives recorded with a 3X are added when available.

## The teacher

The teacher has bigger versions of the three networks, at full resolution everywhere. It also sees the later frames, the GPS and the map. It is used in three separate ways:
- **new labels**, which a person checks on the site;
- **distillation**: the networks for the car also learn how sure the teacher is;
- **starting weights**.

The test stays human: route 0000012d, reviewed by hand, split by place, never trained on.

## The plan

| Step | What | Why |
|---|---|---|
| 1. Cutter | Compare input sizes and the full-resolution band; judge on signs missed (by size) at low thresholds and boxes per frame | No sign is cut out if the cutter misses it |
| 2. Classifier | Add "not a sign" (removed proposals, cutter mistakes) and the kinds of other signs as classes; darker images in training; a button to train it alone | It drops the cutter's false boxes, and the test has many dark frames and other signs |
| 3. Decider | A small network over the classifier's answers, positions, speed and a small scene image; collect "Another" cases (ramps, parallel roads, exits), proposed from GPS and map | Position helped in experiment B; the full image memorised places; 125 "Another" against 1463 "Ours" |
| 4. Site | Every network by role (cutter, classifier, decider): versions, numbers, a training button, the version in use; the whole chain measured end to end | Each network is replaced on its own, and the site must show what is in use |
| 5. Time and state | Tracks, "passed or not" and the limit state on whole routes; time with a wrong limit on the screen measured | The limit must stay valid when the sign is out of sight |
| 6. Chestnut | Export of the networks for the car; latency and memory measured on Chestnut next to the driving model (Cristian installs it) | The driving model comes first |

Data comes before all of these. Route 0000012d is reviewed by hand as the test, drives on new roads and at night are needed, and the split is by place: 75–100% of every route runs on the same roads as another route.

## Second opinion (strata)

strata, a local model, agrees on these points:
- a human test set first;
- crops at full resolution;
- a finder at 480–640 px followed by full-resolution crops;
- an explicit limit state that says "unknown";
- 1 frame per second is too sparse for tracking.

It added two points, both taken: "Unclear" is taught as abstain rather than dropped, and 125 "Another" are too few for a network alone, so that case also needs geometry, rules and targeted collection. It also wanted experiment B only after the detector; B was run first anyway, because it uses the reviewed boxes and so measures context without the detector's errors.

## Decisions taken (24 October 2026)

- **Three networks** (cutter, classifier, decider), each trained, measured and replaced on its own, as Cristian proposed. The cutter favours missing nothing; the classifier drops what is not a sign; the decider works on the recognised signs and their positions.
- **Test.** Route 0000012d is reviewed entirely by hand: about 800 frames with proposals and 200+ signs, on motorway and in town. The final test also needs drives on new roads.
- **Split by place.** Areas of about 300 m. Every frame in an area goes to the same partition, whatever route it comes from. The test areas are those of the test route, and frames of any route next to them are kept out of training (the buffer). The validation has its own buffer too since 28 Oct 2026 (policy `place-v4`): two compact stretches of other roads, each started where it adds the most speed-limit values, where before its areas were scattered and shared their signs with the training. strata asked for 250–500 m areas with a 100–200 m buffer.
- **"Ours".** The sign is meant for the traffic on the carriageway we drive on, in our direction, whatever the lane. **"Another"** is an exit ramp, a side road, a parallel road or the opposite direction. The decider answers which applies; the label only fixes what the answer means. A lane label comes later, if gantries with different values per lane show up.
- **"Unclear"** is taught as "abstain", not dropped.
- **Other traffic signs** get a box with their kind, from people and from the AI reviewer alike.
- **On a vehicle** (25 October 2026): a sign-like disc or panel on a vehicle, such as the top-speed sticker on the back of a truck, is boxed as "Not a sign" with the kind `on_vehicle` (key V). Cut to its box it is the same pixels as a sign, so the classifier does not learn from it (it would learn to reject signs); the decider, which sees the scene, learns it as not ours (`other_road`).
- **Painted on the road** (28 October 2026): an other sign painted on the road, such as the children's triangle with "SCHULE" in front of a school, keeps its kind (Children) and is marked "Painted on the road" on the review page (`painted: true`, other signs only). The networks learn it from the picture; the flag keeps these cases countable apart. A painted number is not this: it is a road marking.
- **Road markings are speed limits** (26 October 2026): a number painted on the road ("30", "30 ZONE") counts as a speed sign. It is often the only one a driver gets, for example when leaving a parking space inside a 30 zone, past its entry sign. The cutter learns to box it (both "Every sign" and "Speed signs only"), the classifier reads it (`road_marking:30`) and the decider says whose road it is on. Older reviews left many painted numbers without a box, so until they are boxed, those frames teach the cutter that the paint is background.
- **Published together, on Hugging Face** (25 October 2026): the three networks in use go out as one package (ONNX, contracts, model card, `networks.json`) in the catalog `speed_networks.json` of `cristianku/sunnypilot-speedlimits-models`, so that other people can use them; a car will download from that catalog, as sunnypilot's model picker does from its own. Prepared and published from the Models tab (`scripts/publish_networks.sh`).
- **Our networks propose only whole** (25 October 2026): Prepare offers "Our networks" only when the cutter, the classifier and the decider are all in use (`preannotate.ours_missing`). The decider sets the road (ours, another, or unresolved when it abstains) of every speed sign proposed, with the car's speed from `nnslr gps` when the drive has it (`_chain_proposer`); the other signs stay unresolved, since it learnt ownership only on the speed signs.

## The road type with memory (29 October 2026)

**Why.** Mobileye, and today's sunnypilot, keep a sign's limit until another sign ends it. On a Swiss motorway on-ramp a 60 sign is never ended by a sign: the motorway's 120 starts where the ramp merges. The same holds for the end of a town (80 or 100 after the town's 50), for an expressway (100) and for a car park. The car needs the kind of road it is on, and the moment it changes, from the camera. One frame is often not enough (the first metres of a motorway look like a rural road), so the network has a memory of the seconds before.

**The network** (`src/nnslr_tools/road_type_net.py`, `nnslr train-road-type`):
- one step a second: the whole road frame, resized to 256 × 144, and the car's speed;
- encoder: torchvision's MobileNetV3-small with ImageNet weights, pooled, then 128 numbers (Linear, Hardswish, Dropout);
- the speed as two numbers, `min(max(v, 0), 60) / 30` and 1 when known (GPS speed in training, `carState.vEgo` on the car);
- memory: a GRU of 128; output: the probability of each of `road_types.ROAD_TYPES` (town, rural, expressway, motorway, ramp, parking), in that order and with those names;
- the version without memory, to compare: the same encoder and speed with a small head per frame (`--memory none`), also measured with a causal 5-second majority vote over its answers, the cheapest memory;
- size: 1,101,478 parameters (1,018,406 without memory), about 84 MFLOPs a step, 3.9 ms a step on the V100 at batch 1. At 1 Hz that is a small fraction of Chestnut, next to the driving model; it is still to be measured there.

**Labels and weights** (`src/nnslr_tools/road_type_data.py`). The road type of each frame comes from the map (`nnslr road-types`) with the corrections of Data › Roads over it (`road_types.load_road_types`). The weight of a frame in the loss:
- no road type (no GPS, no road on the map, "Can't tell"): 0. It is out of the loss but still goes through the memory;
- sure: 1;
- a ramp from its OSM link (`highway motorway_link` or `trunk_link`): 1. Every ramp is marked "check it" because its limit is unclear, not because the road is: an OSM link is the ramp. All 832 ramp frames are unsure and none was corrected yet, so masking them would erase the class;
- every other unsure frame (a 60 or 70 road taken as rural, a gap of the map or the GPS filled, a frame smoothed into its neighbours): 0.3. These are mostly at the changes, where the memory matters; they count less because the map may be wrong there.
- On top, the loss weighs each class by 1 / √(its frequency). Every measure reads the weight-1 frames only.

**Split** (`road-type-place-v1`, split sha256 `a1681d1cef42…`). The drives pass the same roads again and again, so it is by place, like the signs' split: areas of about 300 m. Test is every area of the held-out test drive of `data/splits/latest.json` (`0000012d--6552a45378`, the same drive as the signs' test), so that drive is measured as it was driven; the areas next to it are a buffer. Validation is two compact clusters of other areas (about 15 % of the frames) with their own buffer. A sequence is a run of frames of one drive, without a gap, in one partition, of at least 30 seconds: the network is trained and measured on whole stretches, never on lone frames.

| Partition | Sequences | Labelled frames | Town | Rural | Motorway | Ramp | Parking |
|---|---|---|---|---|---|---|---|
| Train | 18 | 10,031 | 2,595 | 2,171 | 3,902 | 365 | 178 |
| Validation | 11 | 1,814 | 328 | 509 | 783 | 82 | 0 |
| Test | 17 | 6,646 | 1,757 | 937 | 2,840 | 327 | 0 |

There is no expressway frame anywhere (the drives have none on the map), and no parking in validation or test. The changes of road type (10 s of each side): ramp → motorway 2 in train, 1 in validation, 4 in test; motorway → ramp 2, 1, 2; town entry 13, 3, 11; town exit 11, 3, 10. Merges (a ramp of 3 s or more, then the motorway): 5 in train, 1 in validation, 7 in test.

Unsure frames, by class (to confirm first in Data › Roads): town 343, rural 1,352, motorway 397, ramp 832, parking 13.

**Training.** Windows of 48 seconds, 35 % of them around a change of road type; half of the batches start from the memory of the 32 seconds before the window, as on the car, the others from zeros. Per sequence the same random colour and framing (no mirror: traffic keeps right), a second dropped now and then, the speed ±5 % and unknown in one sequence in ten. AdamW, the best epoch by the validation macro F1, run over whole sequences with their memory. `scripts/run_road_type.sh` does both networks, the comparison and the export in about 15 minutes on the V100.

**First results** (run `2026-10-29_0937-road-type`, test, weight-1 frames, 5,029 seconds):

| | Macro F1 | Accuracy | Town | Rural | Motorway | Ramp |
|---|---|---|---|---|---|---|
| With memory (GRU, epoch 37) | **0.584** | **0.790** | **0.932** | 0.576 | **0.827** | 0.000 |
| Without memory (epoch 30) | 0.548 | 0.749 | 0.831 | 0.553 | 0.801 | 0.006 |
| Without memory, 5 s majority | 0.571 | | 0.846 | 0.612 | 0.824 | 0.000 |

| Change (test) | With memory: found, median delay | Without memory |
|---|---|---|
| Ramp → motorway (4) | 4, −7.5 s | 3, −10 s |
| Motorway → ramp (2) | 0 | 0 |
| Town entry (11) | 11, +3 s | 11, 0 s |
| Town exit (10) | 10, +1.5 s | 9, −2 s |
| Other (8) | 3, +7 s | 4, +1.5 s |

- **False switches** inside one road type, per hour: with memory 58 raw, 33 after the car's 3-step debounce; without memory 483 and 75; with the 5 s majority 136 and 80. The memory is what makes the answer steady.
- **After a merge**, the first 60 s on the motorway after a ramp, the share said motorway: with memory **91.5 %** (90.9 % debounced), without memory 79.9 % (81.0 %), with the majority 81.6 % (83.0 %), over 7 merges and 364 s of test. Without memory most of the misses are "town" (57 s); with memory 22 s rural and 9 s town. Validation has a single merge (60 s, all right for every version). **These are few merges**: the trend is clear, the number is not yet a measure.
- **Ramp is never recognised** (F1 0): of the 327 ramp seconds of test it says motorway for 180, rural for 82, town for 62. There are about 12 ramps in train, all with unsure labels. So the "ramp → motorway" change is found early (−7.5 s): the network says motorway while still on the ramp.
- **Motorway taken as rural**: 652 of the 2,840 motorway seconds of test, mostly in two stretches (0000012e segments 33–45, 350 s; 0000012d, 231 s).

**On the car** (the ONNX step, `nnslr export-road-type`; `road-type/export/road-type.onnx` and `road-type.onnx.contract.json` in the run). One step: `image` (1, 3, 144, 256), `speed` (1, 2), `hidden` (1, 128) in; `probabilities` (1, 6) and `hidden_out` (1, 128) out. The GRU is written as matrix products, sigmoid and tanh, with no ONNX GRU node; the ImageNet normalisation is inside. Checked against PyTorch with onnxruntime: largest error 6e-7 on random inputs, and the same answer on all 120 seconds of a real drive run step by step. The input, from sunnypilot's `speed_limits/vision.py`:
1. `y, uv = vision.nv12_planes(buf.data, buf.width, buf.height, buf.stride, buf.uv_offset)` from `VISION_STREAM_ROAD`, the whole 1344 × 760 frame;
2. `planes = vision.yuv_to_rgb(y, uv, 0, buf.height, 0, buf.width)` (BT.601, as the dataset PNGs);
3. `planes = vision.resize(planes, 256, 144)` (PIL bilinear: bit-exact with the training cache);
4. `image = planes[None].astype(np.float32) / 255.0`, RGB, NCHW, no mean or std;
5. `speed = [[min(max(vEgo, 0), 60) / 30, 1.0]]`, or `[[0, 0]]` when unknown;
6. `hidden`: zeros at the start of the drive and after more than 5 s without a step, then the previous `hidden_out`;
7. one step a second (the frames of the training are 1 s apart; a faster rate would make the memory faster too);
8. `road_type = classes[argmax(probabilities)]`, taken only after the same class 3 steps in a row.

**What is missing.**
- Ramps: confirmed ramp frames in Data › Roads, and more drives with on- and off-ramps. Until the network says "ramp", the car must not lift a ramp's limit on "motorway" alone: it says motorway on the ramp itself.
- Expressways: no frame yet. Parking: no held-out frame.
- More merges in validation and test, to make the after-merge number a measure.
- The stretches of motorway taken as rural, to look at (the frames, or the map's labels there).
- The car: nothing uses it. It needs `SpeedLimitMemory` to take the road type, the timing on Chestnut, and a decision on how the network and the map combine.
