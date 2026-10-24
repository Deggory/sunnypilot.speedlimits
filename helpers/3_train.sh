#!/usr/bin/env bash
# Step 3 of 3: from the reviews to a trained model, with no questions asked.
#   a. imports the frames confirmed in the review site            (nnslr import-annotations --from-db)
#   b. checks the dataset and freezes the train/validation/test split (validate-dataset, build-splits)
#   c. creates the GPU environment .venv-gpu the first time         (scripts/setup_v100_training_env.sh)
#   d. trains detector + reader, evaluates, exports and packages the bundle
#                                                                     (scripts/run_v100_baseline.sh)
# Next: scripts/publish_model.sh data/runs/<run>/bundle --no-upload, then without --no-upload.
#
# Usage: helpers/3_train.sh [--background] [--name=<run>] [--continue | --resume=<run>] [--dataset-only]
#   helpers/3_train.sh                       (dataset + training da zero, in primo piano: parte da
#                                             MobileNetV3 pre-allenata su ImageNet, non da pesi casuali)
#   helpers/3_train.sh --background          (continua anche se chiudi la ssh)
#   helpers/3_train.sh --dataset-only        (solo a+b: importa e controlla, niente GPU)
#   helpers/3_train.sh --continue            (parte dai pesi "stable" su Hugging Face)
#   helpers/3_train.sh --resume=2026-10-21_2100   (riprende un training interrotto)
#
# --name: name of the run, folder data/runs/<run>/ (default: date and time, e.g. 2026-10-21_2100).
# --continue: a new run from the PyTorch weights of training/stable.json on Hugging Face
#             (NNSLR_HF_REPO, default cristianku/sunnypilot-speedlimits-models).
# --resume: the same run, interrupted: it skips a+b, because a resumed run refuses a dataset
#           or split that changed.
# The GPU must be free: while another training (e.g. openpilot.distill) uses it, it stops
# before starting. --force starts anyway (NNSLR_ALLOW_BUSY_GPU=1: the two slow each other down).
# The log of the training is data/logs/train-<run>.log.
# -h, --help: this help.
# Env: NNSLR_GPU_INDEX (default 0), NNSLR_DATA_ROOT (default data/)

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export COMMA_VIDEOS_DIR="${COMMA_VIDEOS_DIR:-$(dirname "$ROOT")/comma-videos}"
DATA_ROOT="${NNSLR_DATA_ROOT:-$ROOT/data}"
NN="$ROOT/.venv/bin/nnslr"
PY="$ROOT/.venv/bin/python"
GPU_INDEX="${NNSLR_GPU_INDEX:-0}"
HF_REPO="${NNSLR_HF_REPO:-cristianku/sunnypilot-speedlimits-models}"
LOG_DIR="$DATA_ROOT/logs"

die() { echo "ERROR: $*" >&2; exit 1; }

# Prints the comment block at the top of this file.
usage() {
  sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'
  exit 0
}

# Prints the fields of the JSON report of an nnslr command (file $1) with the Python expression $2.
report() {
  "$PY" -c 'import json, sys
r = json.load(open(sys.argv[1]))
print(eval(sys.argv[2], {}, {"r": r}))' "$1" "$2"
}

# The errors of a failed nnslr report, one per line.
report_errors() {
  "$PY" -c 'import json, sys
r = json.load(open(sys.argv[1]))
for e in r.get("errors", [])[:10]:
    print("  -", e.get("reason", ""), e.get("detail", ""))' "$1" 2>/dev/null || cat "$1"
}

background=0
mode="fresh"
name=""
dataset_only=0
force=0

while [ $# -gt 0 ]; do
  if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    usage
  elif [ "$1" = "--background" ]; then
    background=1
  elif [ "$1" = "--dataset-only" ]; then
    dataset_only=1
  elif [ "$1" = "--force" ]; then
    force=1
  elif [ "$1" = "--continue" ]; then
    [ "$mode" != "resume" ] || die "--continue e --resume insieme non hanno senso"
    mode="continue"
  elif [[ "$1" == --resume=* ]]; then
    [ "$mode" != "continue" ] || die "--continue e --resume insieme non hanno senso"
    mode="resume"
    name="${1#--resume=}"
    name="${name%/}"
    name="${name##*/}"
  elif [[ "$1" == --name=* ]]; then
    name="${1#--name=}"
  else
    die "opzione non valida: $1 (vedi --help)"
  fi
  shift
done

[ -x "$NN" ] || die "manca $NN: crea l'ambiente come in docs/INSTALL.md"
[ "$mode" != "resume" ] || [ -n "$name" ] || die "--resume vuole il nome del run (la cartella in data/runs/)"
[ "$mode" != "resume" ] || [ "$dataset_only" -eq 0 ] || die "--resume non rifa' il dataset: --dataset-only non serve"
if [ -z "$name" ]; then
  name="$(date '+%Y-%m-%d_%H%M')"
fi
[[ "$name" =~ ^[A-Za-z0-9._-]+$ ]] || die "nome del run non valido: $name (lettere, numeri, . _ -)"
mkdir -p "$LOG_DIR"

if [ "$background" -eq 1 ]; then
  args=()
  [ "$dataset_only" -eq 0 ] || args+=(--dataset-only)
  [ "$force" -eq 0 ] || args+=(--force)
  if [ "$mode" = "continue" ]; then args+=(--continue); fi
  if [ "$mode" = "resume" ]; then args+=(--resume="$name"); else args+=(--name="$name"); fi
  NNSLR_TRAIN_LOGGED=1 nohup "$0" "${args[@]}" < /dev/null >> "$LOG_DIR/train-$name.log" 2>&1 &
  echo "Partito in background (pid $!). Segui con:  tail -f $LOG_DIR/train-$name.log"
  exit 0
fi

# --continue without stable weights on Hugging Face would stop only after the dataset steps.
if [ "$mode" = "continue" ]; then
  stable_url="https://huggingface.co/$HF_REPO/resolve/main/training/stable.json"
  http_code="$(curl -sS -L -o /dev/null -w '%{http_code}' "$stable_url" || true)"
  # Hugging Face answers 401 (not 404) for a repository that does not exist yet.
  if [ "$http_code" = "404" ] || [ "$http_code" = "401" ]; then
    die "su $HF_REPO non ci sono ancora pesi stable (training/stable.json): allena prima da zero"
  fi
  [ "$http_code" = "200" ] || die "non riesco a leggere $stable_url (HTTP $http_code)"
fi

if [ "$mode" = "resume" ]; then
  [ -d "$DATA_ROOT/runs/$name" ] || die "non c'e' il run $DATA_ROOT/runs/$name da riprendere"
  echo "== a+b. dataset saltato: --resume riprende con il dataset e lo split del run"
else
  [ ! -e "$DATA_ROOT/runs/$name" ] || die "il run $name esiste gia' in $DATA_ROOT/runs/: scegli un altro --name"

  echo "== a. importo le revisioni confermate"
  if ! "$NN" import-annotations --from-db < /dev/null > "$LOG_DIR/import.json" 2> "$LOG_DIR/import.err"; then
    if grep -q empty_dataset "$LOG_DIR/import.json"; then
      die "nessun frame confermato nel sito di revisione: prima helpers/2_review.sh"
    fi
    report_errors "$LOG_DIR/import.json" >&2
    die "import fallito (report completo: $LOG_DIR/import.json)"
  fi
  echo "Frame nuovi nel dataset: $(report "$LOG_DIR/import.json" 'r.get("imported_frame_count", 0)'), gia' presenti: $(report "$LOG_DIR/import.json" 'r.get("already_imported_frame_count", "-")'), rivisti e sostituiti: $(report "$LOG_DIR/import.json" 'r.get("replaced_frame_count", 0)'), tolti perche' ritirati: $(report "$LOG_DIR/import.json" 'r.get("removed_withdrawn_frame_count", 0)')"
  report "$LOG_DIR/import.json" '"\n".join("  attenzione: " + w for w in r.get("warnings", []))' | sed '/^$/d'

  echo "== b. controllo il dataset e fisso lo split"
  if ! "$NN" validate-dataset < /dev/null > "$LOG_DIR/validate.json" 2> "$LOG_DIR/validate.err"; then
    report_errors "$LOG_DIR/validate.json" >&2
    die "dataset non valido (report completo: $LOG_DIR/validate.json)"
  fi
  # The test routes (reviewed entirely by hand): annotations/test-routes.txt, one route per line. The split is
  # by place: their areas are the test, the areas next to them are kept out of training.
  test_routes=()
  if [ -f "$DATA_ROOT/annotations/test-routes.txt" ]; then
    while read -r line; do
      if [[ "$line" =~ ^[0-9a-f]{8}--[0-9a-f]{10}$ ]]; then test_routes+=(--test-route "$line"); fi
    done < "$DATA_ROOT/annotations/test-routes.txt"
  fi
  if ! "$NN" build-splits "${test_routes[@]}" < /dev/null > "$LOG_DIR/splits.json" 2> "$LOG_DIR/splits.err"; then
    # Three disjoint partitions (train, validation, test) of independent route/site groups:
    # with fewer groups the split cannot be made.
    groups="$(sed -n 's/.*: \([0-9][0-9]*\) groups.*/\1/p' "$LOG_DIR/splits.json" | head -1)"
    if [ -n "$groups" ]; then
      die "per dividere il dataset servono almeno 3 route (o luoghi) indipendenti revisionate, ora ce ne sono $groups.
Prepara e revisiona altre route (helpers/1_prepare.sh, helpers/2_review.sh), poi rilancia.
Le revisioni fatte sono gia' nel dataset: non si perde niente."
    fi
    report_errors "$LOG_DIR/splits.json" >&2
    die "split non creato (report completo: $LOG_DIR/splits.json)"
  fi
  echo "Split: $(report "$LOG_DIR/splits.json" '", ".join(f"{k} {v}" for k, v in r["counts"].items())')"
  if [ "$dataset_only" -eq 1 ]; then
    echo "--dataset-only: dataset pronto, training non lanciato."
    exit 0
  fi
fi

echo "== c. ambiente GPU"
if [ ! -x "$ROOT/.venv-gpu/bin/nnslr" ]; then
  echo "Manca .venv-gpu: lo creo (scarica PyTorch per la V100, qualche minuto)."
  bash "$ROOT/scripts/setup_v100_training_env.sh" < /dev/null > "$LOG_DIR/setup-gpu.log" 2>&1 \
    || die "creazione di .venv-gpu fallita (log: $LOG_DIR/setup-gpu.log)"
fi
echo "Ok: $ROOT/.venv-gpu"

busy="$(nvidia-smi -i "$GPU_INDEX" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^$/d' || true)"
if [ -n "$busy" ]; then
  if [ "$force" -eq 0 ]; then
    who="$(echo "$busy" | wc -l) processi"
    if pgrep -f "supervised.train" > /dev/null; then who="il training di openpilot.distill"; fi
    die "la GPU $GPU_INDEX e' occupata ($who): aspetta che finisca, o --force per partire lo stesso (piu' lenti tutti e due)"
  fi
  export NNSLR_ALLOW_BUSY_GPU=1
fi

echo "== d. training $name ($mode) -> $DATA_ROOT/runs/$name"
if [ -n "${NNSLR_TRAIN_LOGGED:-}" ]; then
  # --background: this whole output already goes to the log.
  NNSLR_GPU_INDEX="$GPU_INDEX" bash "$ROOT/scripts/run_v100_baseline.sh" "$name" "$mode" < /dev/null
else
  echo "Log anche in $LOG_DIR/train-$name.log"
  NNSLR_GPU_INDEX="$GPU_INDEX" bash "$ROOT/scripts/run_v100_baseline.sh" "$name" "$mode" < /dev/null 2>&1 \
    | tee -a "$LOG_DIR/train-$name.log"
fi

echo
echo "Fatto. Bundle: $DATA_ROOT/runs/$name/bundle"
echo "Per pubblicarlo:  scripts/publish_model.sh $DATA_ROOT/runs/$name/bundle --no-upload   (poi senza --no-upload)"
