#!/usr/bin/env bash
# Where each photo of the Mapillary Traffic Sign Dataset (MTSD) was taken, with its country and continent (nnslr
# mapillary-locations, nnslr_tools.mapillary): <data root>/external/mapillary/locations.jsonl. It lets Train ›
# Classifier take the Mapillary signs of some continents or countries only (scripts/run_classifier.sh with
# NNSLR_CLASSIFIER_MAPILLARY_REGION).
#
# MTSD's annotations say nothing of where a photo was taken, so each one is asked of Mapillary's Graph API. MTSD
# names its photos by the keys of Mapillary's old API; a table of old keys and ids of today (made by a Mapillary
# user, downloaded once into <data root>/external/mapillary/image_ids.csv) gives the ids the Graph API takes. About a
# fifth of MTSD's photos have no id of today (deleted since): they get no position and stay out of every region but
# "All the world".
#
# It needs the client token of a Mapillary app (https://www.mapillary.com/dashboard/developers) as MAPILLARY_TOKEN in
# the repository's .env; the token is read from there and never printed.
#
# usage: helpers/mapillary_locations.sh [nnslr mapillary-locations options: --limit N, --again, --dry-run]
# Network: about 650 requests to graph.mapillary.com, at most 4 a second (a few minutes). No GPU, little CPU, does
# not read the MTSD zips. Run it again after a stop: the photos already placed are not asked again.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="${NNSLR_DATA_ROOT:-$ROOT/data}"
NN="$ROOT/.venv/bin/nnslr"
if [ -f "$ROOT/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    . "$ROOT/.env"
    set +a
fi
if [ -z "${MAPILLARY_TOKEN:-}" ] && [[ " $* " != *" --dry-run "* ]]; then
    echo "mapillary: no MAPILLARY_TOKEN in $ROOT/.env (the client token of a Mapillary app)" >&2
    exit 2
fi
echo "== Mapillary: where the photos were taken"
nice -n 10 "$NN" mapillary-locations --data-root "$DATA" "$@"
