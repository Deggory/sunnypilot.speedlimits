#!/usr/bin/env bash
# Where you are: every route (segments, frames, reviewed frames, objects in the dataset), what
# is running now (prepare, review site, training), GPU and disk, and the next command to run.
# Read-only: it changes nothing.
#
# Usage: helpers/status.sh
#
# The three steps:
#   helpers/1_prepare.sh   new videos -> frames -> proposed signs   (automatic)
#   helpers/2_review.sh    review site: you confirm the signs      (the only manual step)
#   helpers/3_train.sh     reviews -> dataset -> trained model      (automatic)
# -h, --help: this help.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export COMMA_VIDEOS_DIR="${COMMA_VIDEOS_DIR:-$(dirname "$ROOT")/comma-videos}"
DATA_ROOT="${NNSLR_DATA_ROOT:-$ROOT/data}"
PY="$ROOT/.venv/bin/python"

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'
  exit 0
fi
[ -x "$PY" ] || { echo "ERROR: manca $PY: crea l'ambiente come in docs/INSTALL.md" >&2; exit 1; }

(cd "$ROOT" && "$PY" -m helpers.route_status)

echo
echo "In corso:"
running=0
if pgrep -f "nnslr (extract-frames|preannotate)" > /dev/null; then
  echo "  preparazione: $(pgrep -af "nnslr (extract-frames|preannotate)" | sed 's/.*nnslr //' | head -3 | paste -sd ';' -)"
  running=1
fi
if pgrep -f "nnslr review" > /dev/null || systemctl is-active --quiet nnslr-review 2>/dev/null; then
  echo "  sito di revisione acceso (helpers/2_review.sh ristampa il link)"
  running=1
fi
if pgrep -f "run_v100_baseline.sh" > /dev/null; then
  echo "  training speedlimits: $(pgrep -af "run_v100_baseline.sh" | awk '{print $NF, $(NF-1)}' | head -1)"
  echo "    log: $(ls -t "$DATA_ROOT"/logs/train-*.log 2>/dev/null | head -1)"
  running=1
fi
[ "$running" -eq 1 ] || echo "  niente"

gpu="$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null | head -1 || true)"
# In a container nvidia-smi cannot name the processes ("[Not Found]"): count them instead.
apps="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^$/d' | wc -l || true)"
if [ -n "$gpu" ]; then
  users=""
  if [ "${apps:-0}" -gt 0 ]; then
    users=" (in uso: $apps processi"
    if pgrep -f "supervised.train" > /dev/null; then users="$users, c'e' il training di openpilot.distill"; fi
    users="$users)"
  fi
  echo "GPU: $gpu$users"
fi
echo "Disco: $(df -h "$DATA_ROOT" | awk 'NR==2 {print $4 " liberi su " $2}')"
