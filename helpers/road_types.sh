#!/usr/bin/env bash
# The type of road of every frame of the prepared drives (town, rural, expressway, motorway, parking), from the GPS
# of the comma and the OpenStreetMap roads and car parks, read offline: the labels of the road-type head, prepared
# without a person (nnslr road-types; the button "Find the road types" of Data › Roads starts this). A person corrects
# them later in the road-type review; nothing here touches the reviews of the signs.
#
#   a. the GPS position of the frames of the drives prepared without it            (nnslr gps --all --missing)
#   b. the map roads and car parks near the drives, again when a drive or the map is newer (scripts/osm_roads.py,
#      osmium from uv's cache for this run only: the environments of the site and of the training stay as they are)
#   c. the road type of every frame                                                (nnslr road-types --all)
#
# usage: helpers/road_types.sh [--again]   (--again: the map roads read again even when nothing is newer)
# No network but uv's first download of osmium; about 2 minutes when the map is read, seconds otherwise.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="${NNSLR_DATA_ROOT:-$ROOT/data}"
NN="$ROOT/.venv/bin/nnslr"
again=no
for argument in "$@"; do
    case "$argument" in
        --again) again=yes ;;
        *) echo "usage: helpers/road_types.sh [--again]" >&2; exit 2 ;;
    esac
done
MAPS="$DATA/raw/osm"
ROADS="$DATA/derived/osm/roads.jsonl"
PARKING="$DATA/derived/osm/parking.jsonl"
if [ -z "$(find "$MAPS" -maxdepth 1 -name '*.osm.pbf' -print -quit 2>/dev/null)" ]; then
    echo "The map is missing: no $MAPS/*.osm.pbf (the Geofabrik extracts, see docs/COMMANDS.md)" >&2
    exit 2
fi

echo "== a. GPS of the drives"
# A drive whose logs cannot be read keeps no position: it is said, and the others go on.
"$NN" gps --all --missing --data-root "$DATA" < /dev/null \
    || echo "GPS position not read for some drives: their frames stay without a road type"

echo "== b. Map roads and car parks near the drives"
newer=no
if [ "$again" = yes ] || [ ! -f "$ROADS" ]; then
    newer=yes
elif [ -n "$(find "$MAPS" -maxdepth 1 -name '*.osm.pbf' -newer "$ROADS" -print -quit)" ]; then
    # A map added or downloaded again.
    newer=yes
elif [ -n "$(find "$DATA/derived/gps" -name positions.jsonl -newer "$ROADS" -print -quit)" ]; then
    newer=yes
elif ! grep -q -m 1 '"maxspeed_type"' "$ROADS"; then
    # A file written before the kind of limit was kept.
    newer=yes
elif [ ! -f "$PARKING" ]; then
    # A map read before the car parks (and the kind of a service road) were kept.
    newer=yes
fi
if [ "$newer" = yes ]; then
    # The site starts this with the service's PATH, without ~/.local/bin where uv usually is.
    UV="$(command -v uv || true)"
    if [ -z "$UV" ] && [ -x "$HOME/.local/bin/uv" ]; then UV="$HOME/.local/bin/uv"; fi
    if [ -z "$UV" ]; then
        echo "uv is missing: it reads the map (see docs/INSTALL.md)" >&2
        exit 2
    fi
    (cd "$ROOT" && nice "$UV" run --no-project --with osmium python scripts/osm_roads.py --data-root "$DATA")
else
    echo "The map roads and car parks are up to date."
fi

echo "== c. Road type of every frame"
"$NN" road-types --all --data-root "$DATA" < /dev/null
echo "Done: the road types are in $DATA/derived/road_types/."
