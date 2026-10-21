#!/usr/bin/env bash
# Step 1 of 3: get every new route ready for the review, with no questions asked.
#   a. links the segments of the shared folder (COMMA_VIDEOS_DIR, the same videos
#      openpilot.distill downloads) into data/raw/routes   (nnslr link-videos)
#   b. extracts the frames (1 per second) of the routes not extracted yet   (nnslr extract-frames)
#   c. proposes the signs on the frames of the routes without a report     (nnslr preannotate)
# Already done work is skipped, so it can be run after every fetch. Next: helpers/2_review.sh.
#
# Usage: helpers/1_prepare.sh [--jobs=<n>] [--background] [<route> ...]
#   helpers/1_prepare.sh                     (tutte le route nuove, una alla volta)
#   helpers/1_prepare.sh --jobs=4            (4 route in parallelo: piu' core, piu' veloce)
#   helpers/1_prepare.sh --background        (continua anche se chiudi la ssh)
#   helpers/1_prepare.sh 0000011e--0c8825628a
#
# A route that already has a pre-annotation report is never processed again: a new report
# would leave its reviews on the old one, and they would no longer be imported. Segments of
# it that arrive later stay unused (helpers/status.sh shows them).
# Every route writes its own log in data/logs/prepare/<route>.log.
# -h, --help: this help.
# Env: COMMA_VIDEOS_DIR (default ../comma-videos next to this repo), NNSLR_DATA_ROOT (default data/)

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export COMMA_VIDEOS_DIR="${COMMA_VIDEOS_DIR:-$(dirname "$ROOT")/comma-videos}"
DATA_ROOT="${NNSLR_DATA_ROOT:-$ROOT/data}"
NN="$ROOT/.venv/bin/nnslr"
PY="$ROOT/.venv/bin/python"
LOG_DIR="$DATA_ROOT/logs/prepare"

die() { echo "ERROR: $*" >&2; exit 1; }

# Prints the comment block at the top of this file.
usage() {
  sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'
  exit 0
}

# Routes still to extract / to pre-annotate (helpers/route_status.py), one per line.
todo() {
  (cd "$ROOT" && "$PY" -m helpers.route_status --todo="$1")
}

# Extract (when needed) and pre-annotate one route; all its output goes to its log.
# Re-extracting is safe here: only routes without a report get this far.
# < /dev/null: ffmpeg reads commands from stdin and would eat the terminal (or the caller's script).
prepare_route() {
  local route="$1"
  local log="$LOG_DIR/$route.log"
  local worked=0
  if todo extract | grep -qx "$route"; then
    echo "$route: estraggo i frame..."
    "$NN" extract-frames --route "$route" --overwrite < /dev/null >> "$log" 2>&1 \
      || { echo "$route: estrazione fallita (vedi $log)"; return 1; }
    worked=1
  fi
  if todo preannotate | grep -qx "$route"; then
    echo "$route: propongo i cartelli..."
    "$NN" preannotate --route "$route" < /dev/null >> "$log" 2>&1 \
      || { echo "$route: pre-annotazione fallita (vedi $log)"; return 1; }
    echo "$route: pronta per la revisione"
    worked=1
  fi
  if [ "$worked" -eq 0 ]; then
    echo "$route: niente da fare (gia' pre-annotata, o senza video collegati)"
  fi
}

jobs=1
background=0
one=""
routes=()

while [ $# -gt 0 ]; do
  if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    usage
  elif [ "$1" = "--background" ]; then
    background=1
  elif [[ "$1" == --jobs=* ]]; then
    jobs="${1#--jobs=}"
  elif [[ "$1" == --one=* ]]; then
    # Internal: one route of a parallel run (--jobs).
    one="${1#--one=}"
  elif [[ "$1" =~ ^[0-9a-f]{8}--[0-9a-f]{10}$ ]]; then
    routes+=("$1")
  else
    die "opzione o route non valida: $1 (vedi --help)"
  fi
  shift
done

[[ "$jobs" =~ ^[1-9][0-9]*$ ]] || die "--jobs vuole un numero da 1 in su"
[ -x "$NN" ] || die "manca $NN: crea l'ambiente come in docs/INSTALL.md (python -m venv .venv, pip install -e '.[preannotate]')"
mkdir -p "$LOG_DIR"

if [ -n "$one" ]; then
  prepare_route "$one"
  exit $?
fi

if [ "$background" -eq 1 ]; then
  args=(--jobs="$jobs" "${routes[@]}")
  nohup "$0" "${args[@]}" < /dev/null > "$DATA_ROOT/logs/prepare.log" 2>&1 &
  echo "Partito in background (pid $!). Segui con:  tail -f $DATA_ROOT/logs/prepare.log"
  echo "A che punto sei:  helpers/status.sh"
  exit 0
fi

echo "== a. collego i video di $COMMA_VIDEOS_DIR"
"$NN" link-videos < /dev/null > "$LOG_DIR/link-videos.json" 2> "$LOG_DIR/link-videos.err" \
  || die "nnslr link-videos fallito: vedi $LOG_DIR/link-videos.json e .err"

if [ "${#routes[@]}" -eq 0 ]; then
  mapfile -t routes < <( (todo extract; todo preannotate) | sed '/^$/d' | sort -u)
fi
if [ "${#routes[@]}" -eq 0 ]; then
  echo "Niente da preparare: nessuna route nuova."
else
  echo "== b+c. ${#routes[@]} route da preparare, $jobs alla volta (log in $LOG_DIR/)"
  failed=0
  if [ "$jobs" -eq 1 ]; then
    for route in "${routes[@]}"; do
      prepare_route "$route" || failed=1
    done
  else
    # xargs reads the routes from the pipe; the commands it starts get /dev/null as stdin.
    printf '%s\n' "${routes[@]}" | xargs -P "$jobs" -I{} "$0" --one={} || failed=1
  fi
  [ "$failed" -eq 0 ] || echo "Qualche route e' fallita: guarda i log in $LOG_DIR/"
fi

echo
(cd "$ROOT" && "$PY" -m helpers.route_status)
