#!/usr/bin/env bash
# Publish one trained speed-limit model bundle on Hugging Face (cristianku/sunnypilot-speedlimits-models):
# verify the bundle, upload its files and the catalog speed_models.json in one commit.
#
# Usage: scripts/publish_model.sh [--name=<nome>] [--no-upload] [<cartella bundle>]
#   scripts/publish_model.sh                                  (l'ultimo <data root>/runs/*/bundle)
#   scripts/publish_model.sh data/runs/baseline-001/bundle --name=prova
#   scripts/publish_model.sh --no-upload                      (prepara tutto, non carica niente)
#
# Steps:
#   1. verify: the bundle is checked with "nnslr verify-bundle" (file list, sha256, provenance,
#      advisory capability). A bundle that fails is not published.
#   2. stage: only the files the bundle defines (REQUIRED_FILES in src/nnslr_tools/bundle.py) are
#      copied to <data root>/publish/<nome>/models/<nome>/ and the copy is verified again.
#   3. catalog: downloads speed_models.json from Hugging Face and adds/replaces this model
#      (src/nnslr_tools/model_catalog.py) with size, sha256 and download url of every file.
#   4. one Hugging Face commit with models/<nome>/* and speed_models.json. Everything is also left
#      in <data root>/publish/<nome>/.
# Only model artifacts are uploaded: never videos, logs, datasets or checkpoints. The two contracts
# (detector.contract.json, reader.contract.json) do contain the absolute path of the checkpoint on the
# training machine (e.g. /home/<utente>/.../data/runs/<id>/...): it becomes public with the model.
# Nothing is compiled or installed on the comma: the on-device backend does not exist yet.
#
# Hugging Face, once on this machine:
#   uvx --from 'huggingface_hub>=1,<3' hf auth login       (token that can write the repo)
#   uvx --from 'huggingface_hub>=1,<3' hf repos create cristianku/sunnypilot-speedlimits-models
# Until the repo exists Hugging Face answers 401 to the catalog download and the script stops.
# huggingface_hub is not a dependency of this repo: uvx (from uv) runs it in its own environment.
# The version is pinned to 1.x/2.x, which have the same "hf auth", "hf repos" and "hf upload".
#
# --name: name of the model in the catalog and in the paths (default: the run folder that contains
#         the bundle, e.g. baseline-001). Letters, digits, _ . - starting with a letter or a digit.
#         The same name again replaces the model.
# --no-upload: stops before Hugging Face (uvx is not needed).
# -h, --help: this help.
# Env: HF_REPO (default cristianku/sunnypilot-speedlimits-models)
#      NNSLR_DATA_ROOT (default <repo>/data): where runs/*/bundle is searched and publish/ is written

set -euo pipefail

HF_REPO="${HF_REPO:-cristianku/sunnypilot-speedlimits-models}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_ROOT="${NNSLR_DATA_ROOT:-$ROOT/data}"
CATALOG_NAME="speed_models.json"
# huggingface_hub version that uvx runs; keep it equal to the one in the help block above.
HF_CLI="huggingface_hub>=1,<3"

die() { echo "ERROR: $*" >&2; exit 1; }

# Prints the comment block at the top of this file.
usage() {
  sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'
  exit 0
}

# Runs Python with this repo's src/ first on the path. Verification and catalog are standard
# library only, so any Python 3.11+ works: the local venv when there is one, else python3.
run_python() {
  local python
  if [ -x "$ROOT/.venv/bin/python" ]; then
    python="$ROOT/.venv/bin/python"
  elif [ -x "$ROOT/.venv-gpu/bin/python" ]; then
    python="$ROOT/.venv-gpu/bin/python"
  else
    python="python3"
  fi
  PYTHONPATH="$ROOT/src" "$python" "$@"
}

name=""
upload=1
bundle=""

while [ $# -gt 0 ]; do
  if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    usage
  elif [ "$1" = "--no-upload" ]; then
    upload=0
  elif [ "$1" = "--name" ]; then
    [ $# -gt 1 ] || die "--name vuole un nome"
    shift
    name="$1"
    [ -n "$name" ] || die "--name vuole un nome"
  elif [[ "$1" == --name=* ]]; then
    name="${1#--name=}"
    [ -n "$name" ] || die "--name vuole un nome"
  elif [ -z "$bundle" ]; then
    bundle="$1"
  else
    die "argomento in piu': $1"
  fi
  shift
done

if [ "$upload" -eq 1 ]; then
  command -v uvx >/dev/null 2>&1 \
    || die "manca uvx: installa uv (https://docs.astral.sh/uv/) oppure usa --no-upload"
fi

if [ -z "$bundle" ]; then
  bundle="$(ls -td "$DATA_ROOT"/runs/*/bundle 2>/dev/null | head -1 || true)"
  [ -n "$bundle" ] || die "nessun $DATA_ROOT/runs/*/bundle: passa la cartella del bundle"
fi

[ -d "$bundle" ] || die "$bundle non e' una cartella"
bundle="$(cd "$bundle" && pwd)"
# The bundle sits inside its run folder: data/runs/<run id>/bundle.
run_id="$(basename "$(dirname "$bundle")")"

if [ -z "$name" ]; then
  name="$run_id"
fi
[[ "$name" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] \
  || die "nome non valido: $name (solo lettere, numeri, _ . - e all'inizio una lettera o un numero)"
stage="$DATA_ROOT/publish/$name"
model_dir="$stage/models/$name"

echo "Modello: $name (run $run_id)"
echo "Bundle: $bundle"

# 1. Verify with the same check as "nnslr verify-bundle"; its error goes to stderr.
echo "== 1. verifico il bundle"
if ! run_python -m nnslr_tools.cli verify-bundle "$bundle" >/dev/null; then
  die "bundle non valido, non lo pubblico: $bundle"
fi
echo "Bundle verificato"

# 2. Stage only the files the bundle defines, then verify the copy that is going to be uploaded.
echo "== 2. preparo i file in $stage"
# The stage is deleted and rebuilt: a bundle inside it would be deleted before being copied.
# Compare the real paths, so symlinks and relative paths do not hide it.
mkdir -p "$DATA_ROOT/publish"
real_stage="$(cd "$DATA_ROOT/publish" && pwd -P)/$name"
real_bundle="$(cd "$bundle" && pwd -P)"
if [[ "$real_bundle/" == "$real_stage/"* ]]; then
  die "il bundle e' dentro $stage: copialo altrove prima di pubblicarlo"
fi
rm -rf "$stage"
mkdir -p "$model_dir"
bundle_files="$(run_python -c 'from nnslr_tools.bundle import REQUIRED_FILES; print("\n".join(REQUIRED_FILES))')"
while IFS= read -r file_name; do
  cp -p "$bundle/$file_name" "$model_dir/$file_name"
done <<< "$bundle_files"
if ! run_python -m nnslr_tools.cli verify-bundle "$model_dir" >/dev/null; then
  die "la copia in $model_dir non passa la verifica"
fi

# 3. Catalog: the one on Hugging Face plus this model.
echo "== 3. catalogo"
catalog_url="https://huggingface.co/$HF_REPO/resolve/main/$CATALOG_NAME"
http_code="$(curl -sS -L -w '%{http_code}' -o "$stage/$CATALOG_NAME" "$catalog_url" || true)"
if [ "$http_code" = "404" ]; then
  rm -f "$stage/$CATALOG_NAME"
  echo "Nessun catalogo su $HF_REPO: ne creo uno nuovo"
elif [ "$http_code" = "401" ]; then
  die "HTTP 401 da $catalog_url: il repo $HF_REPO non esiste o e' privato (crealo con: uvx --from '$HF_CLI' hf repos create $HF_REPO)"
elif [ "$http_code" != "200" ]; then
  die "non riesco a scaricare $catalog_url (HTTP $http_code)"
fi
run_python -m nnslr_tools.model_catalog "$stage/$CATALOG_NAME" \
  --name "$name" --run-id "$run_id" --model-dir "$model_dir" \
  --base-url "https://huggingface.co/$HF_REPO/resolve/main"

if [ "$upload" -eq 0 ]; then
  echo "--no-upload: niente caricato. File pronti in $stage"
  exit 0
fi

# 4. One commit with the model files and the new catalog.
echo "== 4. carico su https://huggingface.co/$HF_REPO"
uvx --from "$HF_CLI" hf upload "$HF_REPO" "$stage" . --commit-message "Publish model $name"

echo "Pubblicato. Catalogo: $catalog_url"
