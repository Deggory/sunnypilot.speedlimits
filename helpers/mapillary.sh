#!/usr/bin/env bash
# The signs of the Mapillary Traffic Sign Dataset (MTSD) that are ours, cut into small crops for the classifier's
# training (nnslr mapillary-crops, nnslr_tools.mapillary): <data root>/external/mapillary/crops/ and crops.jsonl.
# The Mapillary switch of Train › Classifier (scripts/run_classifier.sh with NNSLR_CLASSIFIER_MAPILLARY=1) adds them
# to train; validation and test stay our drives only.
#
# MTSD comes from https://www.mapillary.com/dataset/trafficsign (log in, accept the licence, CC BY-NC-SA 4.0: the
# links come by email). Leave the zips as they come: the annotations (<key>.json) and the photos (<key>.jpg) are
# read straight from them, or from unzipped folders, searched with their subfolders. By default the source is
#
#   /data/mapillary   (NNSLR_MAPILLARY_DIR; on the server the TrueNAS share, the zips in /data/mapillary/zips/)
#
# and, when it is not there, <data root>/external/mapillary/annotations/ and <data root>/external/mapillary/images/.
#
# usage: helpers/mapillary.sh [ANNOTATIONS_DIR IMAGES_DIR]
# No network, no GPU; the CPU for a while, the share read once. Run it again after more zips are copied, or after a
# read error of the share: the photos already cut are not read again.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="${NNSLR_DATA_ROOT:-$ROOT/data}"
NN="$ROOT/.venv/bin/nnslr"
if [ "$#" -ne 0 ] && [ "$#" -ne 2 ]; then
    echo "usage: helpers/mapillary.sh [ANNOTATIONS_DIR IMAGES_DIR]" >&2
    exit 2
fi
SOURCE="${NNSLR_MAPILLARY_DIR:-/data/mapillary}"
if [ "$#" -eq 0 ] && [ -d "$SOURCE" ]; then
    # On the server /data/mapillary is a share mounted into the container: an empty local folder means it is not.
    if [ "$SOURCE" = /data/mapillary ] && ! mountpoint -q "$SOURCE"; then
        echo "mapillary: /data/mapillary is not mounted (the TrueNAS share): ask cluster-admin, do not copy around it" >&2
        exit 1
    fi
    ANNOTATIONS="$SOURCE"
    IMAGES="$SOURCE"
else
    ANNOTATIONS="${1:-$DATA/external/mapillary/annotations}"
    IMAGES="${2:-$DATA/external/mapillary/images}"
fi
echo "== Mapillary: the signs that are ours, from $ANNOTATIONS and $IMAGES"
nice -n 10 "$NN" mapillary-crops --annotations "$ANNOTATIONS" --images "$IMAGES" --data-root "$DATA"
