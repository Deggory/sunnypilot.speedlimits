#!/usr/bin/env bash
# Update this clone from GitHub and put the new code to work, always the same way:
#   a. git pull --ff-only of the current branch (stops on local changes or a diverged branch)
#   b. when pyproject.toml changed, or the .venv has the wrong onnxruntime for this machine:
#      updates the .venv (with uv when installed, otherwise pip). With an NVIDIA GPU the
#      pre-annotation uses onnxruntime-gpu (extra preannotate-gpu), otherwise onnxruntime
#      (extra preannotate); the two install the same "onnxruntime" module, so the other one
#      is removed first.
#   c. with the systemd service speedlimits-web installed (or still its old name,
#      nnslr-review): scripts/install_review_service.sh (rewrites the unit only when
#      deploy/speedlimits-web.service changed, replaces the old service) and a restart
#      of the service; without it, restarts the site started by helpers/2_review.sh, if on
# A preparation or a training started from the panel keeps running: the service stops
# only the web server (KillMode=process). Full procedure: docs/GUIDA.md.
#
# Usage: helpers/update.sh [--no-link]
# --no-link: do not print the link with the token at the end (the panel uses it, so the
#            token never lands in its log).
# -h, --help: this help.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SERVICE="speedlimits-web"
OLD_SERVICE="nnslr-review"

die() { echo "ERROR: $*" >&2; exit 1; }

# pip of the .venv: uv when installed (the .venv made by uv has no pip), otherwise pip.
venv_pip() {
  local uv=""
  if command -v uv > /dev/null 2>&1; then
    uv="$(command -v uv)"
  elif [ -x "$HOME/.local/bin/uv" ]; then
    uv="$HOME/.local/bin/uv"
  fi
  if [ -n "$uv" ]; then
    "$uv" pip "$1" --python "$ROOT/.venv/bin/python" "${@:2}" < /dev/null
  elif [ "$1" = "uninstall" ]; then
    "$ROOT/.venv/bin/python" -m pip uninstall -y "${@:2}" < /dev/null
  else
    "$ROOT/.venv/bin/python" -m pip "$@" < /dev/null
  fi
}

# True when the distribution $1 is installed in the .venv.
venv_has() {
  "$ROOT/.venv/bin/python" -c 'import importlib.metadata, sys; importlib.metadata.version(sys.argv[1])' "$1" > /dev/null 2>&1
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'
  exit 0
fi
print_link=1
if [ "${1:-}" = "--no-link" ]; then
  print_link=0
  shift
fi
[ $# -eq 0 ] || die "opzione non valida: $1 (vedi --help)"

cd "$ROOT"
[ -z "$(git status --porcelain --untracked-files=no)" ] \
  || die "ci sono modifiche locali ai file del repo (git status): sistemale prima, non le sovrascrivo"

before="$(git rev-parse HEAD)"
echo "== a. git pull ($(git branch --show-current))"
git pull --ff-only || die "git pull non riuscito (branch divergente?): niente e' cambiato"
after="$(git rev-parse HEAD)"
if [ "$before" = "$after" ]; then
  echo "Gia' aggiornato: $(git log --oneline -1)"
else
  git log --oneline "$before..$after"
fi

extra="preannotate"
if command -v nvidia-smi > /dev/null 2>&1 && nvidia-smi -L > /dev/null 2>&1; then
  extra="preannotate-gpu"
fi
wrong=""
if [ "$extra" = "preannotate-gpu" ] && venv_has onnxruntime; then
  wrong="onnxruntime"
fi
if [ "$extra" = "preannotate" ] && venv_has onnxruntime-gpu; then
  wrong="onnxruntime-gpu"
fi
changed=0
if [ "$before" != "$after" ] && git diff --name-only "$before" "$after" | grep -qx 'pyproject.toml'; then
  changed=1
fi
if [ "$changed" -eq 1 ] || [ -n "$wrong" ]; then
  echo "== b. aggiorno .venv (.[$extra])"
  if [ -n "$wrong" ]; then
    echo "tolgo $wrong: su questa macchina serve l'altro onnxruntime"
    venv_pip uninstall "$wrong" || die "non riesco a togliere $wrong da .venv"
  fi
  venv_pip install -q -e "$ROOT[$extra]" \
    || die "aggiornamento di .venv fallito: rilancia helpers/update.sh, o a mano: uv pip install --python .venv/bin/python -e '.[$extra]'"
fi

installed=0
if systemctl list-unit-files "$SERVICE.service" 2>/dev/null | grep -q "^$SERVICE.service"; then
  installed=1
fi
if systemctl list-unit-files "$OLD_SERVICE.service" 2>/dev/null | grep -q "^$OLD_SERVICE.service"; then
  installed=1
fi

if [ "$installed" -eq 1 ]; then
  echo "== c. servizio $SERVICE"
  bash "$ROOT/scripts/install_review_service.sh" < /dev/null | grep -v -e '^Apri nel browser' -e '^  http' || true
  systemctl restart "$SERVICE"
  sleep 2
  systemctl is-active --quiet "$SERVICE" || die "il servizio non riparte: journalctl -u $SERVICE -n 50"
  echo "Servizio riavviato con il codice nuovo."
elif [ -f "$ROOT/data/logs/review.pid" ] && kill -0 "$(cat "$ROOT/data/logs/review.pid")" 2>/dev/null; then
  echo "== c. riavvio il sito di revisione acceso da helpers/2_review.sh"
  local_flag=()
  if grep -q "in ascolto su 127.0.0.1" "$ROOT/data/logs/review.log" 2>/dev/null; then local_flag=(--local); fi
  "$ROOT/helpers/2_review.sh" --stop
  "$ROOT/helpers/2_review.sh" "${local_flag[@]}" > /dev/null
  echo "Sito riavviato."
else
  echo "== c. nessun sito di revisione acceso: niente da riavviare (helpers/2_review.sh lo accende)"
  exit 0
fi

# The site is on (service or helpers/2_review.sh): 2_review.sh only prints its link.
[ "$print_link" -eq 1 ] || exit 0
echo
"$ROOT/helpers/2_review.sh" 2>/dev/null | grep -e '^Apri' -e '^Pannello' || true
