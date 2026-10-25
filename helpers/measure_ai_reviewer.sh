#!/usr/bin/env bash
# Measures an AI provider of Settings as a reviewer (nnslr measure-ai-reviewer): on up to 60 frames people reviewed
# (the same sample for every provider), the speed signs it finds and reads right, the frames it says have no sign,
# the false boxes per frame, the seconds per frame. The result is the AI reviewers view of the Evaluate tab (its
# "Measure" buttons start this). Nothing is written into the reviews or the routes; the report goes to
# data/evaluations/ai_reviewers/.
#
# usage: helpers/measure_ai_reviewer.sh [--provider=ID] [--limit=N]   (default provider: the reviewer of Settings)
# The provider's key is read from its file by the command, never passed here. No GPU here: the frames go to the
# provider's server.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
echo "Measuring the AI reviewer on the frames people reviewed..."
"$ROOT/.venv/bin/nnslr" measure-ai-reviewer "$@" > /dev/null
echo "Done: see the AI reviewers view of the Evaluate tab."
