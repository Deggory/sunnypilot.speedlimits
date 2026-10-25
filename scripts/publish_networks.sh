#!/usr/bin/env bash
# Our three networks (the cutter, the classifier and the decider in use, Models tab) as one package, and that
# package published on Hugging Face for others (docs/NEURAL_NETWORK.md, src/nnslr_tools/networks_package.py).
#
# Usage:
#   scripts/publish_networks.sh [--name=<name>]           prepare: export to ONNX, check against PyTorch, write the
#                                                         model card; <data root>/publish/networks/<name>/. Uploads
#                                                         nothing. Default name: speedlimits-<UTC date>-<time>.
#   scripts/publish_networks.sh --upload --name=<name>    publish the package prepared under that name.
#
# Publishing: the package is checked again; the catalog speed_networks.json of the repository is read (a repository
# or catalog that does not exist yet starts a new one) and gets the package; the repository is created, public, when
# it is missing; one commit uploads networks/<name>/*, the catalog and the package's README.md at the root (the
# model card of the newest package); <name>/published.json records it.
#
# Env: HF_REPO (default cristianku/sunnypilot-speedlimits-models), NNSLR_DATA_ROOT (default <repo>/data),
#      NNSLR_GPU_VENV (default <repo>/.venv-gpu: torch for the export).
# The Hugging Face login of this server is used (hf auth login): no token is passed, printed or written.
# Needs uvx (uv) for the upload only.

set -euo pipefail

HF_REPO="${HF_REPO:-cristianku/sunnypilot-speedlimits-models}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_ROOT="${NNSLR_DATA_ROOT:-$ROOT/data}"
VENV="${NNSLR_GPU_VENV:-$ROOT/.venv-gpu}"
CATALOG_NAME="speed_networks.json"
# The huggingface_hub that uvx runs.
HF_CLI="huggingface_hub>=1,<3"

die() { echo "error: $*" >&2; exit 1; }

usage() {
  sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'
  exit 0
}

name=""
upload=0
for argument in "$@"; do
  case "$argument" in
    -h|--help) usage ;;
    --upload) upload=1 ;;
    --name=*) name="${argument#--name=}" ;;
    *) die "unknown argument: $argument" ;;
  esac
done
if [ -n "$name" ] && [[ ! "$name" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$ ]]; then
  die "invalid name: $name (letters, digits, _ . -, starting with a letter or a digit)"
fi
[ -x "$VENV/bin/nnslr" ] || die "GPU environment missing: $VENV/bin/nnslr (scripts/setup_v100_training_env.sh)"
NN="$VENV/bin/nnslr"
export NNSLR_DATA_ROOT="$DATA_ROOT"

if [ "$upload" -eq 0 ]; then
  echo "== Package of the networks${name:+ $name}"
  # One CPU thread per export is plenty, and leaves the server to the rest.
  CUDA_VISIBLE_DEVICES="" "$NN" networks-package ${name:+--name "$name"} --data-root "$DATA_ROOT"
  exit 0
fi

[ -n "$name" ] || die "--upload needs --name=<the package to publish>"
package="$DATA_ROOT/publish/networks/$name"
[ -d "$package" ] || die "no package $name: prepare it first ($0 --name=$name)"
command -v uvx >/dev/null 2>&1 || die "uvx is missing: install uv (https://docs.astral.sh/uv/)"

echo "== 1/3 Check the package $name"
"$NN" verify-networks "$package" >/dev/null || die "the package $name does not pass its check: not published"
echo "Package checked"

work="$(mktemp -d "$DATA_ROOT/tmp/publish-networks.XXXXXX" 2>/dev/null || mktemp -d)"
trap 'rm -rf "$work"' EXIT

echo "== 2/3 The catalog $CATALOG_NAME of $HF_REPO"
uvx -q --from "$HF_CLI" python - "$HF_REPO" "$CATALOG_NAME" "$work" <<'EOF'
import shutil
import sys

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

repo, name, work = sys.argv[1:4]
try:
    shutil.copy(hf_hub_download(repo, name, repo_type="model"), f"{work}/{name}")
    print("Catalog read")
except (RepositoryNotFoundError, EntryNotFoundError):
    print("No catalog yet: a new one")
EOF
"$NN" networks-catalog "$work/$CATALOG_NAME" --package "$package" \
  --base-url "https://huggingface.co/$HF_REPO/resolve/main" >/dev/null

echo "== 3/3 Upload to https://huggingface.co/$HF_REPO"
uvx -q --from "$HF_CLI" python - "$HF_REPO" "$package" "$name" "$work/$CATALOG_NAME" <<'EOF'
import json
import sys
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

repo, package, name, catalog = sys.argv[1:5]
package = Path(package)
sums = json.loads((package / "sha256sums.json").read_text())
files = sorted(sums) + ["sha256sums.json"]
api = HfApi()
api.create_repo(repo, repo_type="model", private=False, exist_ok=True)
operations = [CommitOperationAdd(path_in_repo=f"networks/{name}/{file}", path_or_fileobj=str(package / file))
              for file in files]
operations.append(CommitOperationAdd(path_in_repo="speed_networks.json", path_or_fileobj=catalog))
operations.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=str(package / "README.md")))
api.create_commit(repo, operations=operations, commit_message=f"Publish the networks {name}", repo_type="model")
print(f"Uploaded {len(files)} files")
EOF
"$NN" networks-published "$package" --repo "$HF_REPO" \
  --url "https://huggingface.co/$HF_REPO/tree/main/networks/$name" >/dev/null
echo "Published: https://huggingface.co/$HF_REPO/tree/main/networks/$name"
