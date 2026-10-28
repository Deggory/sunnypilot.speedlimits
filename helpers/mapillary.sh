#!/usr/bin/env bash
# The signs of the Mapillary Traffic Sign Dataset (MTSD) that are ours, cut into small crops for the classifier's
# training (nnslr mapillary-crops, nnslr_tools.mapillary): <data root>/external/mapillary/crops/ and crops.jsonl.
# The Mapillary switch of Train › Classifier (scripts/run_classifier.sh with NNSLR_CLASSIFIER_MAPILLARY=1) adds them
# to train; validation and test stay our drives only.
#
# MTSD comes from https://www.mapillary.com/dataset/trafficsign (log in, accept the licence, CC BY-NC-SA 4.0: the
# links come by email). Unzip the "fully annotated" archives so that the annotations are <key>.json files in one
# folder and the photos <key>.jpg files in another; by default
#
#   <data root>/external/mapillary/annotations/   and   <data root>/external/mapillary/images/
#
# usage: helpers/mapillary.sh [ANNOTATIONS_DIR IMAGES_DIR]
# No network, no GPU; a few minutes on the CPU. Run it again after more photos are unzipped: it cuts everything again.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="${NNSLR_DATA_ROOT:-$ROOT/data}"
NN="$ROOT/.venv/bin/nnslr"
if [ "$#" -ne 0 ] && [ "$#" -ne 2 ]; then
    echo "usage: helpers/mapillary.sh [ANNOTATIONS_DIR IMAGES_DIR]" >&2
    exit 2
fi
ANNOTATIONS="${1:-$DATA/external/mapillary/annotations}"
IMAGES="${2:-$DATA/external/mapillary/images}"
echo "== Mapillary: the signs that are ours, from $ANNOTATIONS and $IMAGES"
nice -n 10 "$NN" mapillary-crops --annotations "$ANNOTATIONS" --images "$IMAGES" --data-root "$DATA"
