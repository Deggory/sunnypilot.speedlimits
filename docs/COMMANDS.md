# Helpers, commands and scripts

This page is the technical reference behind the portal. Daily work happens in the portal (see the [README](../README.md)), and its buttons run the helpers and scripts listed here. Run everything from the repository root. Before running anything, check its exact options with `--help`: this page only condenses them. Several helpers print their messages in Italian.

The **Effects** columns use these words:

| Effect | Meaning |
|---|---|
| read-only | changes nothing |
| writes data | writes in the data root (`data/` by default) |
| CPU-heavy | keeps the CPU busy for minutes to hours |
| GPU | trains or runs networks on the GPU |
| installs | installs Python packages into a virtual environment |
| network | talks to the comma over SSH, to an AI provider's server, or downloads from the internet |
| restarts | restarts the portal |
| system | changes the system configuration (a systemd service) |
| publishes | uploads to a public place |

## Where things are

| What | Where |
|---|---|
| Data root | `data/` in the clone, or `NNSLR_DATA_ROOT` / `--data-root`. Git ignores it. |
| Comma recordings | `COMMA_VIDEOS_DIR`, by default `../comma-videos` next to the clone. Each file is stored once and linked into the data root, so other projects on the same machine can share them. |
| Portal access token | `NNSLR_REVIEW_TOKEN` in the `.env` of the clone (mode 600, ignored by git). It is the setup code of the first sign-in and the key of the REST API and MCP. Never copy it into files, logs or reports. |
| Python environments | `.venv` for the core and the portal, and `.venv-gpu` for training. Git ignores both. |
| AI providers | `web/private/llm-providers.json` in the data root (mode 600, folder 700), or the file named by `NNSLR_LLM_PROVIDERS` (a full path). It holds the OpenAI-compatible servers of **Settings → AI reviewer**, their API keys and which one reviews. Only the portal's Settings and `nnslr llm-provider` write it. A key never leaves the server: no page, API answer, log or report shows it. |

Inside the data root:

| Path | Content |
|---|---|
| `raw/routes/<route>/<segment>/` | links to the recordings (`fcamera.hevc`, `ecamera.hevc`, `rlog.zst`) |
| `derived/frames/<route>/` | the extracted frames, one per second |
| `derived/preannotations/` | the boxes proposed by the pre-annotation models |
| `derived/gps/<route>/positions.jsonl` | position, speed, heading and country of every frame |
| `derived/gps/borders.json.gz` | finer country borders (GISCO 1:1 million, non-commercial), from `scripts/make_country_borders.py --gisco` |
| `derived/tracks/<route>/tracks.jsonl` | the same sign followed across confirmed frames |
| `derived/hints/<route>/` | second readings shown on the review page |
| `derived/osm/` | OpenStreetMap roads (`roads.jsonl`) and car parks (`parking.jsonl`) near the drives |
| `derived/road_types/<route>.jsonl` | the road type of every frame from the map (town, rural, expressway, motorway, parking or unknown), and why |
| `annotations/review.sqlite` | every review, with its history |
| `annotations/road_types/<route>.jsonl` | a person's road types (Data › Roads, a drive opened): one line per correction, `{from, to, road_type, reviewer, at}`, the last one covering a frame wins; kept apart from `derived/road_types/`, so a new `nnslr road-types` keeps them |
| `annotations/objects.jsonl` | the dataset, written by the import |
| `annotations/replaced/` | dataset rows replaced by a newer review |
| `splits/` | the frozen splits (`latest.json` names the current one) |
| `runs/<run>/` | trainings, evaluations, exports and bundles |
| `models/active.json` | the version in use for each network (Models tab) |
| `models/preannotation/<sha256>/` | the downloaded pre-annotation models |
| `logs/` | logs of the helpers, the portal's jobs and the trainings |
| `ai_reviews/<route>/` | what the AI suggested for each frame in every route review run (`nnslr ai-review`), with or without saving |
| `evaluations/ai_reviewers/` | the latest measure of each AI provider (`nnslr measure-ai-reviewer`), and a timestamped copy of every measure |

## Install

[INSTALL.md](INSTALL.md) has the full procedure. In short:

```sh
apt install -y git curl rsync openssh-client python3 python3-venv ffmpeg zstd jq
python3 -m venv .venv && .venv/bin/pip install -e . pytest    # or: uv venv .venv && uv pip install --python .venv/bin/python -e . pytest
.venv/bin/nnslr selftest                                      # synthetic checks on the CPU
.venv/bin/python -m pytest -q
```

The core needs Python 3.11 or newer and has no dependencies. The optional extras are:

| Extra | Adds | Needed for |
|---|---|---|
| `.[preannotate]` | onnxruntime, Pillow, numpy, rapidocr 3.4.0 | `nnslr preannotate` on the CPU |
| `.[preannotate-gpu]` | onnxruntime-gpu below 1.27, with CUDA and cuDNN wheels | the same on an NVIDIA GPU. Never install it together with `preannotate`. |
| `.[log-reader]` | pycapnp, zstandard | reading comma logs: `nnslr gps`, `log-metadata`, `align-route`, and the comma download |
| `.[mcp]` | the MCP SDK | `nnslr mcp`, and the portal's `/mcp` endpoint |
| `.[training]` | wandb, huggingface-hub | training logs and model downloads; `scripts/setup_v100_training_env.sh` installs it into `.venv-gpu` |

## The helpers

These helpers are the daily flow, and the portal's buttons run them. Each one skips work that is already done and ends by printing the next command.

| Command | What it does | Effects |
|---|---|---|
| `helpers/fetch_comma_segments.sh [--host=...] [<route> ...]` | Copies drives from the comma over SSH into `COMMA_VIDEOS_DIR` and links them into the data root. Without arguments it takes every route still on the comma, drops trips under 300 s, the first and last two segments of each trip, and any segment with reverse gear. Explicit routes and segments can also be given. Needs `.[log-reader]` in `.venv`; the log schema of the comma is downloaded once into `~/.cache/comma-fetch-log` and checked by SHA-256 (`NNSLR_OPENPILOT_ROOT` / `NNSLR_OPENDBC_ROOT` choose other schemas). Portal: **Download from the comma**. | network, writes data |
| `helpers/1_prepare.sh [--jobs=N] [--background] [<route> ...]` | For every new route, it links the videos, extracts one frame per second, has the models propose signs, and reads the GPS position and country of every frame. A route that already has proposals is never processed again. Portal: **Prepare new drives**. | CPU-heavy, writes data |
| `helpers/road_types.sh [--again]` | The road type of every frame of the prepared drives (town, rural, expressway, motorway, parking), from the GPS of the comma and the OpenStreetMap roads and car parks, read offline: a. `nnslr gps --all --missing`, b. `scripts/osm_roads.py` when a drive or the map is newer, or `parking.jsonl` is missing (`--again`: always), c. `nnslr road-types --all`. No review is needed first, and the reviews of the signs are not touched. Needs the Geofabrik extracts in `raw/osm/`; about 2 minutes when the map is read, seconds otherwise. Portal: **Find the road types** in Data › Roads. | writes data |
| `helpers/2_review.sh [--local] [--foreground] [--stop]` | Starts the portal on port 8765, on the LAN with sign-in, or with `--local` on 127.0.0.1 only. Creates the token in `.env` the first time. When the systemd service is running, it only prints its address. | opens a port, writes `.env` |
| `helpers/3_train.sh [--dataset-only] [--background] [--continue \| --resume=<run>]` | Imports the confirmed reviews, validates the dataset and freezes the split. Without `--dataset-only` it also creates `.venv-gpu` the first time and runs `scripts/run_v100_baseline.sh`, the older detector-and-reader baseline. It refuses a busy GPU (`--force` to share it). Portal: **Import reviews** (`--dataset-only`), **Whole pipeline**. | writes data; GPU and installs without `--dataset-only` |
| `helpers/status.sh` | Shows, for each route, its segments, frames, reviewed frames and objects in the dataset, then what is running, the GPU, the disk, and the next command. | read-only |
| `helpers/update.sh [--no-link]` | Runs `git pull --ff-only`, updates `.venv` when `pyproject.toml` changed, reinstalls the service unit if it changed, and restarts the portal. Jobs started from the portal keep running. Portal: **Update the code**. | network, installs, restarts |
| `helpers/reset_password.sh` | For a forgotten password: stops the portal, moves `web/auth/admin.json` of the data root aside as `admin.json.<time>.old`, and starts the portal again. The next visit asks for the setup code, then a new name and password. Reviews and settings stay. The sign-in page points to it under **Forgot the password?**. | restarts, moves the administrator file |
| `helpers/build_web.sh [--check]` | Type-checks the Vue 3 and TypeScript pages in `web/src` and builds them into `src/nnslr_tools/web/js`. The built files are committed, so the server never needs Node. Node comes from PyPI through uv, and nothing is installed system-wide. `--check` only verifies that the built files are current. | installs into `web/node_modules`, writes `src/nnslr_tools/web/js` |
| `helpers/mcp.sh [--read-only]` | Runs the MCP server of the review over stdio (`nnslr mcp`), for Claude Code, Codex and other MCP clients, on this machine or through SSH. Its saves are AI reviews. `--read-only` offers no save tool. | writes reviews unless `--read-only` |
| `helpers/ai_review.sh <route> [--provider=ID] [--save] [--limit=N]` | Reviews the frames of a route that nobody confirmed with an AI provider of Settings (`nnslr ai-review`; default: the reviewer). Without `--save` it is a trial: the suggestions go to `ai_reviews/<route>/`, nothing is saved, and the review page of the route shows them (**AI trial**). `--save` needs a provider that passed its last test. With `--save`, every frame the AI judged completely is saved as an AI review, `ai/<provider>/<model>`. That save is final, and the next import puts it into training. Frames a person confirmed are never touched. Portal: **Review with AI** on the route list. | network; writes reviews with `--save` |
| `helpers/measure_ai_reviewer.sh [--provider=ID] [--limit=N]` | Measures an AI provider on up to 60 frames people reviewed (`nnslr measure-ai-reviewer`). Every provider gets the same sample, so their numbers compare. Nothing is saved; the report goes to `evaluations/ai_reviewers/`. Portal: **Measure** in Evaluate → AI reviewers. | network, writes data |
| `helpers/setup_mcp.sh [--claude] [--effort=...] [--restart]` | Installs `.[mcp]`. `--claude` registers the MCP server in Claude Code for this folder and writes the git-ignored `.claude/settings.local.json`. `--restart` backs up the review database when it will be migrated, then restarts the portal. | installs; restarts with `--restart` |

## The `nnslr` command

`nnslr` is the project's command-line tool, and `nnslr <subcommand> --help` gives the exact options. Commands marked GPU run from `.venv-gpu` (`.venv-gpu/bin/nnslr`), and most of them accept `--dry-run` for a CPU-only plan.

**Environment and checks**

| Subcommand | Purpose | Effects |
|---|---|---|
| `version` | package versions | read-only |
| `env` (alias `check-environment`) | environment report: data root, videos folder, CPU. `--gpu-smoke` also probes the GPU. | read-only |
| `selftest` | the bundled synthetic checks, CPU only | read-only |
| `validate-batch` | validates an observation-batch JSON file against the core contract | read-only |

**Drives and frames**

| Subcommand | Purpose | Effects |
|---|---|---|
| `sync-routes --route R [--dry-run]` | copies one route over SSH/rsync, without the download helper's filters | network, writes data |
| `link-videos [--route R] [--dry-run]` | links segments already in `COMMA_VIDEOS_DIR` into `raw/routes` | writes data |
| `extract-frames --route R` | extracts frames with ffmpeg, one per second by default | CPU-heavy, writes data |
| `route-manifest`, `inspect-manifest`, `video-probe`, `log-metadata`, `align-route`, `alignment-report` | inventory of local segments, video probing, and the alignment of video frames to the logs | read-only or writes reports |
| `find-candidates`, `make-clips` | map transitions as search hints, and short review clips around them (older tools) | writes data |

**Proposals, positions and review**

| Subcommand | Purpose | Effects |
|---|---|---|
| `preannotate --route R` | proposes sign boxes on every frame of a route. Downloads the models on first use and checks their SHA-256. | network on first use, CPU-heavy (GPU with `.[preannotate-gpu]`), writes data |
| `gps --all --missing` | GPS position and country of every frame (`derived/gps/`). The review page proposes that country. | writes data |
| `gps --all --countries` | The country of every frame again, from the positions already read and the borders now (after `make_country_borders.py --gisco`); the logs are not read | writes data |
| `road-types --all` (or `--route R`) | the road type of every frame from its GPS position, speed and heading and the map roads and car parks (`derived/osm/roads.jsonl`, `parking.jsonl`): `derived/road_types/<route>.jsonl`. Parking: on a parking aisle, or inside a car park outline on a service road, on no road, or slower than 3 m/s (not over an underground car park); its limit is always 0 to sunnypilot unless a sign is read. A proposal of the map, corrected by a person in Data › Roads (a drive opened; the corrections stay in `annotations/road_types/`, never touched here, a broken line is skipped). Data › Roads shows it drive by drive. | writes data |
| `tracks --all` | follows the same sign across consecutive confirmed frames (`derived/tracks/`) | writes data |
| `review [--host H] [--port P] [--route R]` | serves the portal. By default it listens on 127.0.0.1:8765 with no sign-in. `--host 0.0.0.0` opens it to the LAN and requires the token. | serves the portal, writes reviews |
| `mcp [--read-only]` | the MCP server of the review, over stdio | writes reviews unless `--read-only` |
| `review-reclassify-ai --reviewer NAME [--dry-run]` | marks the page reviews saved by an AI agent as AI reviews (`ai/<name>`) | writes reviews |

**AI reviewer** (an OpenAI-compatible server that sees images; see below)

| Subcommand | Purpose | Effects |
|---|---|---|
| `llm-provider list` | the providers, their address and model, `key: set\|none`, and which one reviews. Never a key. | read-only |
| `llm-provider add --name N --base-url URL [--model M] [--key-file PATH] [--reviewer]` | adds a provider, or updates the one whose id comes from that name. The key is read from the file and never printed. Without `--key-file` a new provider has no key and an update keeps its key. Without `--model` it takes the first model the server lists. `--reviewer` tests it and makes it the reviewer only if the test passes. | writes the providers file; network without `--model` or with `--reviewer` |
| `llm-provider remove ID`, `llm-provider use ID\|none` | removes a provider and its key; chooses the reviewer, which must have passed its last test (`llm-provider test ID`) | writes the providers file |
| `llm-provider models ID` | the models the provider's server lists, and whether each one sees images | network |
| `llm-provider test ID` | asks the provider about one reviewed frame: does it answer in JSON, and does it see the image? Only the result is saved. | network, writes the providers file |
| `ai-review ROUTE [--provider ID] [--save] [--limit N]` | the AI review of a route (above, `helpers/ai_review.sh`). Progress goes to stderr and the summary as JSON to stdout. | network; writes reviews with `--save` |
| `measure-ai-reviewer [--provider ID] [--limit 60]` | the measure of an AI provider (above). The report goes to stdout as JSON. | network, writes data |

**Dataset and splits**

| Subcommand | Purpose | Effects |
|---|---|---|
| `import-annotations --from-db [--dry-run]` | turns the confirmed reviews into `annotations/objects.jsonl` | writes data |
| `validate-dataset` | checks labels, images, hashes and provenance, and prints a report | read-only |
| `build-splits [--test-route R] [--seed N]` | freezes the train, validation and test split | writes data |

**Training and evaluation**

| Subcommand | Purpose | Effects |
|---|---|---|
| `train --task detector\|reader` | Trains the cutter (`detector`) or the classifier (`reader`). Main options: `--input-size WxH`, `--cutter-families all\|speed`, `--classes v1\|v2`, `--negatives`, `--select-by recall\|loss`, `--patience`, `--mode fresh\|continue\|resume`, and `--dry-run`. | GPU, writes data |
| `train-signnet --run-dir D [--stride 4\|8] [--widths 16,32,48,64] [--band-width 1344\|896\|672]` | SignNet, the sign finder made for the comma's Adreno: plain convolutions on the top band of the frame (the top 416 of 760 rows), a heat map of sign centres and each sign's size, no NMS. Same frames, split and scoring as the cutter; `result.json` has recall at 3, 5 and 10 boxes a frame, by width and family. `--band-width 672` is half the resolution. About 2 minutes an epoch on the V100. | GPU, writes data |
| `evaluate` | measures a detector or reader checkpoint on a frozen partition | GPU, writes data |
| `mine-negatives` | runs the cutter on the reviewed frames and keeps its false boxes as "not a sign" crops | GPU, writes data |
| `evaluate-chain` | runs cutter and classifier end to end: signs found and read right, and false boxes per frame, at several cutter thresholds | GPU, writes data |
| `experiment-b --variant crop\|crop_geom\|full` | the decider prototype, trained on the reviewed boxes | GPU, writes data |
| `mine-hard-examples` | extracts the errors from an evaluation report | writes data |
| `replay` | runs detector then reader offline over the extracted frames of a route | GPU, writes data |
| `drive-video ROUTE` | joins the comma's minutes of road video (`fcamera.hevc`, 1344 × 760, 20 fps) into one H.264 MP4 960 px wide with the start of each minute, `data/derived/video/<route>.mp4` and `.json`: what the Drive view of Results plays. The site makes it by itself the first time Drive shows the drive (low priority, 8 threads, about 4 minutes for 45 minutes of drive, about 3 MB a minute); `--again` makes it again | CPU, writes data |

**Export and packaging**

| Subcommand | Purpose | Effects |
|---|---|---|
| `export-onnx --task reader\|detector\|signnet --checkpoint P --output X.onnx` | exports a checkpoint to ONNX and checks it against PyTorch (onnxruntime), with a `.contract.json` beside it. SignNet's ONNX has one output `out` (heat and the sign's size in one tensor): two outputs cost the Adreno twice the time. | writes data |
| `package-model`, `verify-bundle` | packages detector and reader into an immutable bundle, and verifies its files, hashes, provenance and advisory-only flag | writes data / read-only |
| `export-core` | exports and verifies the dependency-free `speed_vision_core` | writes files |

## Training and experiment scripts

All of these scripts use the GPU and write only under `data/runs/<name>/`. Each one refuses a GPU that is already busy unless `NNSLR_ALLOW_BUSY_GPU=1`, picks the GPU with `NNSLR_GPU_INDEX` (default 0), and never imports reviews, deploys anything or touches a comma. Run the dry run first.

| Script | What it does | Dry run first |
|---|---|---|
| `bash scripts/run_detector_sizes.sh NAME [320x320 640x360 960x544 ...]` | Trains one cutter per input size with the same settings, evaluates each on test and validation, and writes `comparison.txt`: recall by sign width at low thresholds, boxes per frame, and time per frame. `NNSLR_CUTTER_FAMILIES=all\|speed` sets what it boxes. Run it again with the same name to finish. Portal: **Train the cutter**. | `.venv/bin/nnslr train --task detector --input-size 640x360 --dry-run` |
| `bash scripts/run_classifier.sh NAME` | Mines the false boxes of the cutter in use (Models tab), trains the classifier with `--classes v2` and "not a sign", then evaluates it alone, in the chain, and against the older reader (`comparison.txt`). Portal: **Train the classifier**. | `.venv/bin/nnslr mine-negatives --dry-run` and `.venv/bin/nnslr train --task reader --classes v2 --dry-run` |
| `bash scripts/run_experiment_b.sh NAME` | Experiment B: trains the decider variants `crop`, `crop_geom` and `full`, then compares reading, "another road" precision and recall, and abstentions. | `.venv/bin/nnslr experiment-b --variant full --dry-run` |
| `bash scripts/run_v100_baseline.sh RUN_ID [fresh\|continue\|resume]` | The older full baseline: validates the dataset and split, trains detector and reader, evaluates, mines errors, exports ONNX and packages a verified bundle. The run ID must be new. | `.venv/bin/nnslr train --task detector --dry-run` |

The GPU environment comes from `bash scripts/setup_v100_training_env.sh`. It creates `.venv-gpu` with torch 2.14.0 and torchvision 0.29.0 for CUDA 12.6, the last line that supports the Tesla V100 (sm_70). It installs packages, but no driver, and runs no GPU job. Afterwards, `.venv-gpu/bin/nnslr env --gpu-smoke --gpu-device 0` checks the GPU.

`scripts/setup_vlm.sh` (optional) adds Qwen3-VL-8B-Instruct, a model that sees images (Apache-2.0, about 17.5 GB), to this server: `transformers`, `accelerate` and `qwen-vl-utils` go into `.venv-gpu` without changing any package already there, the weights go into `data/models/vlm/Qwen3-VL-8B-Instruct` at a pinned revision, and it is tried on the GPU on one frame (`--no-test` skips that). Nothing uses it yet: it is there to be measured as a reader of the cutter's boxes and as a proposer.

`nnslr preannotate --proposer ours` (or `NNSLR_PROPOSER=ours helpers/1_prepare.sh`, the **Find the signs → Found by** choice of Data › Drives) proposes the signs with our cutter and classifier instead of the Vietnamese YOLO; `--proposer qwen` asks Qwen3-VL-8B for every sign of each frame (installed by `scripts/setup_vlm.sh`; its boxes have no score). Both run in `.venv-gpu`. `NNSLR_TRAIN_WITHOUT_REVIEWERS="ai/llama,ai/strata"` before `nnslr train --task reader --classes v2` leaves the frames those reviewers reviewed (names starting so) out of the classifier's train partition, validation and test whole: to measure what their reviews do; the plan names them in `train_without_reviewers`. `helpers/1_prepare.sh --again` (**Propose again**) proposes again the frames not confirmed yet of every route: the new report gets the confirmed frames of the old one as they are (`nnslr preannotate --keep-reviews`, copied before it becomes the current one, so nothing is lost if it fails); a route with every frame confirmed is left out, and each route is read only until its new proposals are in.

`helpers/measure_proposer.sh [vietnam|ours|qwen]` (the **Measure** button of the Proposers view of Evaluate, `nnslr evaluate-proposer`) runs a proposer on the reviewed test frames exactly as Prepare would and compares its proposals with the kept boxes: speed signs found and read right (overall, by width, on the frames a person reviewed), every sign found, false proposals per frame, seconds per frame. Painted road numbers are left out. Nothing is written into the reviews; the report goes to `data/evaluations/proposers/`.

**The AI reviewer.** A review can be done by hand or by an AI that sees images, through any OpenAI-compatible server (`/v1/models` and `/v1/chat/completions`): a provider such as strata, a llama.cpp `llama-server` with a vision model and its `--mmproj` file (for example Qwen3-VL-32B-Instruct), or another service. Providers are added in the portal under **Account → Settings → AI reviewer**, or from a terminal:

```sh
.venv/bin/nnslr llm-provider add --name strata --base-url https://strata.farm.14bodhi.com/v1 --key-file /root/.config/strata-api-key
.venv/bin/nnslr llm-provider test strata      # answers in JSON? sees the image?
.venv/bin/nnslr llm-provider use strata       # the one that reviews
```

**Test** sends one frame that a person reviewed, without its proposals, and checks whether the model reads its speed sign. A server that lists the model as text only is reported as such, and the model is not asked anything. On the review page, **Ask AI** (key I) shows the AI's suggestion for the frame. It saves nothing: **Apply** turns it into edits that you check, and your **Confirm** saves them as your review. On the route list, **Review with AI** runs `helpers/ai_review.sh` as a job: **Try without saving** checks 20 frames, and **Review and save** saves AI reviews, which are final. The model gets the image, the numbered boxes and the second reading as hints, in coordinates from 0 to 1000. It never sees the neighbouring frames, and it is told to ignore any instruction written in the image. Only a browser signed in to the portal, or the portal on 127.0.0.1, can change the providers or use the AI. The access token can only read the settings, never a key.

**Training modes** (`--mode`, or the second argument of `run_v100_baseline.sh`):

- **fresh** (the default) is a new run, started from torchvision's ImageNet-pretrained MobileNetV3 weights.
- **continue** is a new run, started from the published stable weights. No weights are published yet, so this mode fails clearly for now.
- **resume** picks up an interrupted run in the same folder, from its `training-state.pt`. It refuses a changed dataset, split or configuration.

## Portal service

| Command | What it does | Effects |
|---|---|---|
| `scripts/install_review_service.sh --dry-run` | shows what an install would change | read-only |
| `sudo scripts/install_review_service.sh` | Installs or updates the portal as the systemd service `speedlimits-web`, using the token in `.env`. A restart stops only the web server, never a job started from it. `--uninstall` removes the service. | system |

The portal also serves a REST API at `/api/v1`: routes, paged frames, the next frame to review, images, history, and saves with `run_id` and `base_revision`, which answer 409 on a conflict. With `.[mcp]` it also serves MCP at `/mcp`. Both need the token in an `Authorization: Bearer` header, never in the URL, even on 127.0.0.1. The AI reviewer's endpoints (`/api/settings/ai`, `/api/ai/state`, `/api/route/<route>/frame/<s>/<i>/ai-suggest`, `/api/route/<route>/ai-trial`, `/api/panel/ai-reviewers`) belong to the portal, not to `/api/v1`: the token may only read them. `NNSLR_PANEL_ACTIONS=0` makes the portal's buttons read-only. Keep the portal on a LAN or behind an SSH tunnel. Details are in [API_MCP.md](API_MCP.md) (in Italian).

## Publishing (when there is something to publish)

Nothing is published yet. The commands below upload to Hugging Face, so start with their dry run.

| Command | What it does | Effects |
|---|---|---|
| `scripts/publish_networks.sh [--name=<name>]` | **Our networks**: exports the cutter, classifier and decider in use (Models tab) to ONNX, checks each against PyTorch with onnxruntime, and writes the model card, the manifest `networks.json` and the sha256 sums into `<data root>/publish/networks/<name>/`. Uploads nothing. The portal's **Prepare the package** (Models → Publish) runs it. | writes data |
| `scripts/publish_networks.sh --upload --name=<name>` | Publishes that package: checks it again, adds it to the catalog `speed_networks.json`, creates the repository (public) if it is missing, and uploads `networks/<name>/*`, the catalog and the model card as `README.md` in one commit (`HF_REPO`, default `cristianku/sunnypilot-speedlimits-models`). Only the package's files: no frame, dataset, checkpoint or server path. The portal's **Publish on Hugging Face** runs it, after a question. | **publishes** |
| `nnslr networks-package`, `nnslr verify-networks <folder>`, `nnslr networks-catalog <speed_networks.json> --package <folder> --base-url <url>` | The steps of the script: the export (GPU environment), the check of a package, the catalog. | writes data |
| `scripts/publish_model.sh --no-upload [<bundle>]` | Verifies a bundle with `nnslr verify-bundle`, stages its files and updates the catalog `speed_models.json`, without uploading. Without `--no-upload` it publishes the bundle (`HF_REPO`, default `cristianku/sunnypilot-speedlimits-models`). Only model files are uploaded, never videos, logs, datasets or checkpoints. | writes data; publishes without `--no-upload` |
| `python scripts/publish_training_weights.py ...` | Publishes fine-tunable PyTorch weights. It is a dry run by default and refuses optimizer checkpoints. `--upload --stable` uploads and moves `training/stable.json`. | publishes with `--upload` |

Before publishing:

- The ONNX contracts of a bundle contain the absolute checkpoint path of the training machine, so that path, and the user name in it, become public.
- The weights start from ImageNet-pretrained weights and learn from private drives. Their licence must be assessed before any release (see the README).

## Map and country data

| Command | What it does | Effects |
|---|---|---|
| `uv run --no-project --with osmium python scripts/osm_places.py` | Keeps the names of the Swiss cities, towns, villages and districts of the same local map in `derived/osm/places.jsonl`; the Review list names each drive by them ("Herznach → Zürich", with the day and time from the last segment video). Run it again after a newer map. | writes data |
| `uv run --no-project --with osmium python scripts/osm_roads.py` | Keeps the OpenStreetMap roads near the drives (from the local Geofabrik extracts in `raw/osm/`, read offline) in `derived/osm/roads.jsonl`, with the kind of a service road (`parking_aisle`, `driveway`...), and the car parks near them (`amenity=parking` outlines, ways and multipolygons, without the parking along a street) in `derived/osm/parking.jsonl`. The roads are a possible input for "another road"; both make the road types. About 2 minutes for Switzerland and Baden-Württemberg. Map data © OpenStreetMap contributors, ODbL. | writes data |
| `python scripts/make_country_borders.py [<file.geojson>]` | Rebuilds `src/nnslr_tools/data/countries.json.gz`, the Natural Earth borders (public domain) used to propose the country of a frame | network (downloads the source), writes the file |
| `python scripts/make_country_borders.py --gisco [<file.geojson>]` | Writes finer borders, GISCO 1:1 million, into `<data root>/derived/gps/borders.json.gz`, used instead of Natural Earth when there (Natural Earth puts Bad Säckingen and Rheinfelden in Switzerland). Administrative boundaries © EuroGeographics, non-commercial use only: never in git. Then `nnslr gps --all --countries` | network (64 MB download), writes the file |

## Rules the tools follow

- Only frames explicitly confirmed by a person or an AI reviewer become labels. Model proposals and unconfirmed frames are never imported. Proposals that reviewers removed are kept as records, and used as "not a sign" examples.
- AI confirmations are final, and stay recognisable in the dataset (`ai/<name>`, `label_basis: ai_review`). The portal's AI reviewer saves as `ai/<provider>/<model>`, with channel `web`, only from **Review and save**. A suggestion on the review page saves nothing until a person confirms it, and then it is that person's review.
- The dataset follows the latest review of each frame. Rows it replaces are archived in `annotations/replaced/`.
- Splits are frozen and identified by a hash of their content. When most frames have a GPS position, the split is by place: areas of about 300 m, with a buffer of neighbouring areas around the test that is neither trained on nor measured. Otherwise the split is by route group. It needs at least three independent groups, and the test prefers frames confirmed by a person.
- Confirming starts nothing. Import and training are separate, explicit steps.

## Other tests and documents

- Offline tests of the download helper: `python -m unittest discover -s helpers -p 'test_*.py'`.
- Tests of the frozen sunnypilot runtime snapshot:

  ```sh
  PYTHONPATH="$PWD/integrations/sunnypilot/snapshot" python -m pytest -q \
    --confcutdir=integrations/sunnypilot/snapshot \
    integrations/sunnypilot/snapshot/openpilot/sunnypilot/speed_vision/tests
  ```

- [INSTALL.md](INSTALL.md): system packages, environments and data folders.
- [NEURAL_NETWORK.md](NEURAL_NETWORK.md): the pipeline and the plan, in detail.
- [GUIDA.md](GUIDA.md) (in Italian): a step-by-step guide to the helpers and the service. Parts of it are older than this page.
- [../integrations/sunnypilot/README.md](../integrations/sunnypilot/README.md): what the runtime snapshot contains, and what it does not.
