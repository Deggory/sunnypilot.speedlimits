#!/usr/bin/env bash
# Our networks saved on GitHub, in the private repository cristianku/smartspeed-networks (#92), so that a lost data root
# can start again from networks that work: those in use and on the car, the sign networks of each brain, the road
# labeler, the splits and the records of what is in use, and Fit box (src/nnslr_tools/networks_github.py). Models ›
# Publish › GitHub › Save on GitHub. scripts/restore_networks_github.sh puts them back.
#
# Usage:
#   scripts/save_networks_github.sh          check the repository on GitHub (created, private, when missing; never a
#                                            push to a public one), bring its clone up to date, copy the networks new
#                                            or changed into it, commit and push them over SSH; the record
#                                            <data root>/publish/github/saved.json
#   scripts/save_networks_github.sh --plan   only say what a save copies: no GitHub, nothing copied
#
# Env: NNSLR_DATA_ROOT (default <repo>/data), NNSLR_NETWORKS_REPO (default cristianku/smartspeed-networks),
#      NNSLR_NETWORKS_CLONE (default smartspeed-networks beside this checkout), NNSLR_NETWORKS_REMOTE (default
#      git@github.com:<repo>.git), NNSLR_GITHUB_TOKEN_FILE (default /root/.secrets/github-token: GitHub's API only, to
#      check that the repository is private and to create it; never printed nor written).

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_ROOT="${NNSLR_DATA_ROOT:-$ROOT/data}"
NN="$ROOT/.venv/bin/nnslr"

die() { echo "error: $*" >&2; exit 1; }

usage() {
  sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'
  exit 0
}

mode=save
for argument in "$@"; do
  case "$argument" in
    -h|--help) usage ;;
    --plan) mode=plan ;;
    *) die "unknown argument: $argument" ;;
  esac
done
[ -x "$NN" ] || die "nnslr is missing: $NN (helpers/update.sh makes .venv)"
command -v git >/dev/null 2>&1 || die "git is missing"
# The steps (== N/M) are printed by nnslr networks-github itself.
exec "$NN" networks-github "$mode" --data-root "$DATA_ROOT"
