#!/usr/bin/env bash
# Our networks back from GitHub, from the private repository cristianku/smartspeed-networks that
# scripts/save_networks_github.sh fills (#92): after a lost data root, the networks saved there put back, and the ones
# in use with them (src/nnslr_tools/networks_github.py). Models › Publish › GitHub › Restore from GitHub.
#
# Usage:
#   scripts/restore_networks_github.sh   bring the clone of the repository up to date (cloned when missing), check
#                                        every saved file against its manifest, then put back in the data root every
#                                        file missing there, with its time and mode. Never overwrites a file of the
#                                        data root: one that differs is kept and listed. The records (models/active.json,
#                                        on-car.json, train-settings.json, box-fit.json) come back only when missing.
#                                        The versions of Fit box (runs/<run>-box-fit/) come back as the other runs; the
#                                        code's Fit box (box_fit:code) is only compared with this checkout (it comes
#                                        back with the code repository). The record
#                                        <data root>/publish/github/restored.json. Any data root of any machine
#                                        will do: a fresh clone of the code repository and its .venv, then this script,
#                                        and the site shows the networks in use and on the car of the last save.
#
# Env: NNSLR_DATA_ROOT (default <repo>/data), NNSLR_NETWORKS_REPO (default cristianku/smartspeed-networks),
#      NNSLR_NETWORKS_CLONE (default smartspeed-networks beside this checkout), NNSLR_NETWORKS_REMOTE (default
#      git@github.com:<repo>.git). git reads over SSH (a key that may read the private repository); no token is
#      needed.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_ROOT="${NNSLR_DATA_ROOT:-$ROOT/data}"
NN="$ROOT/.venv/bin/nnslr"

die() { echo "error: $*" >&2; exit 1; }

usage() {
  sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'
  exit 0
}

for argument in "$@"; do
  case "$argument" in
    -h|--help) usage ;;
    *) die "unknown argument: $argument" ;;
  esac
done
[ -x "$NN" ] || die "nnslr is missing: $NN (helpers/update.sh makes .venv)"
command -v git >/dev/null 2>&1 || die "git is missing"
# The steps (== N/M) are printed by nnslr networks-github itself.
exec "$NN" networks-github restore --data-root "$DATA_ROOT"
