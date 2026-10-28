#!/usr/bin/env bash
# MCP server (stdio) of the sign review, for Claude Code, Codex and other MCP clients started on
# this machine or through ssh. stdout carries the MCP messages: this script prints nothing else there.
# Same rules and same database (data/annotations/review.sqlite) as the review site; the panel actions
# (prepare, import, train, update) are not exposed. Saves are AI reviews ("ai/<name>").
#
# Usage: helpers/mcp.sh [--read-only]
#   --read-only   only reading tools: save_frame_review is not offered
# From another computer, in the MCP client configuration (see docs/API_MCP.md):
#   ssh <server> ~/sunnypilot.speedlimits/helpers/mcp.sh
# The review site also serves MCP over HTTP at /mcp (token in the header).
# Needs the optional dependencies once:  .venv/bin/python -m pip install -e ".[mcp]"
# -h, --help: this help.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export COMMA_VIDEOS_DIR="${COMMA_VIDEOS_DIR:-$(dirname "$ROOT")/comma-videos}"
export NNSLR_DATA_ROOT="${NNSLR_DATA_ROOT:-$ROOT/data}"
PY="$ROOT/.venv/bin/python"

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'
  exit 0
fi
[ -x "$PY" ] || { echo "ERROR: $PY is missing: create the environment as in docs/INSTALL.md" >&2; exit 1; }

cd "$ROOT"
exec "$PY" -m nnslr_tools.cli mcp "$@"
