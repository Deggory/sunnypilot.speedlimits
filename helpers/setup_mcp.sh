#!/usr/bin/env bash
# Sets up, on this machine, what the review REST API and MCP server need and git cannot carry:
#   1. the optional MCP SDK in .venv (extra [mcp]; uv when installed, otherwise pip);
#   2. with --claude, Claude Code in this folder, for review sessions started here
#      (also the remote ones of the bridge environment):
#        - the MCP server "nnslr-review" (stdio, helpers/mcp.sh), local scope;
#        - .claude/settings.local.json: the nnslr-review tools allowed without asking,
#          and the effort of the sessions (--effort, default medium);
#        - .claude/settings.local.json listed in .git/info/exclude (never committed);
#   3. with --restart, a copy of data/annotations/review.sqlite when the new code will
#      migrate it, then systemctl restart speedlimits-web (otherwise it prints the command).
# Running it again changes nothing that is already in place.
#
# Usage: helpers/setup_mcp.sh [--claude] [--effort=low|medium|high|xhigh|max] [--restart]
# Effects: installs packages in .venv; --claude writes ~/.claude.json and .claude/;
#          --restart restarts the review site for a few seconds (panel jobs keep running).
# Documentation: docs/API_MCP.md, section 5.
# -h, --help: this help.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
DATA_ROOT="${NNSLR_DATA_ROOT:-$ROOT/data}"
SERVICE=speedlimits-web
claude_setup=0
restart=0
effort=medium

die() { echo "ERROR: $*" >&2; exit 1; }

for arg in "$@"; do
  case "$arg" in
    -h|--help) sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    --claude) claude_setup=1 ;;
    --restart) restart=1 ;;
    --effort=*) effort="${arg#--effort=}" ;;
    *) die "unknown option: $arg (see --help)" ;;
  esac
done
case "$effort" in
  low|medium|high|xhigh|max) ;;
  *) die "--effort must be low, medium, high, xhigh or max" ;;
esac
[ -x "$PY" ] || die "$PY is missing: create the environment as in docs/INSTALL.md"

echo "== 1. MCP SDK in .venv"
if "$PY" -c 'import mcp.server.mcpserver' 2> /dev/null; then
  echo "already installed: $("$PY" -c 'import importlib.metadata as m; print("mcp", m.version("mcp"))')"
else
  # The .venv made by uv has no pip; installing the extra never removes other packages.
  if command -v uv > /dev/null 2>&1; then
    uv pip install -q --python "$PY" -e "$ROOT[mcp]" < /dev/null || die "installing .[mcp] failed"
  else
    "$PY" -m pip install -q -e "$ROOT[mcp]" < /dev/null || die "installing .[mcp] failed"
  fi
  "$PY" -c 'import mcp.server.mcpserver' || die "MCP SDK installed but it cannot be imported"
  echo "installed: $("$PY" -c 'import importlib.metadata as m; print("mcp", m.version("mcp"))')"
fi

if [ "$claude_setup" -eq 1 ]; then
  echo "== 2. Claude Code in $ROOT"
  command -v claude > /dev/null 2>&1 || die "claude not found in the PATH: install Claude Code or drop --claude"
  cd "$ROOT"
  if claude mcp get nnslr-review > /dev/null 2>&1; then
    echo "MCP server nnslr-review already registered"
  else
    claude mcp add -s local nnslr-review -- "$ROOT/helpers/mcp.sh" > /dev/null
    echo "MCP server nnslr-review registered (stdio, helpers/mcp.sh)"
  fi
  mkdir -p "$ROOT/.claude"
  "$PY" - "$ROOT/.claude/settings.local.json" "$effort" <<'PYEOF'
import json, sys
from pathlib import Path
path, effort = Path(sys.argv[1]), sys.argv[2]
settings = json.loads(path.read_text()) if path.exists() else {}
allow = settings.setdefault('permissions', {}).setdefault('allow', [])
if 'mcp__nnslr-review' not in allow:
    allow.append('mcp__nnslr-review')
settings['effortLevel'] = effort
path.write_text(json.dumps(settings, indent=2) + '\n')
print(f'{path}: mcp__nnslr-review tools allowed, effort {effort}')
PYEOF
  exclude="$(git -C "$ROOT" rev-parse --git-path info/exclude)"
  case "$exclude" in /*) ;; *) exclude="$ROOT/$exclude" ;; esac
  grep -qx '.claude/settings.local.json' "$exclude" 2> /dev/null || echo '.claude/settings.local.json' >> "$exclude"
  echo ".claude/settings.local.json excluded from git (.git/info/exclude)"
fi

if [ "$restart" -eq 1 ]; then
  echo "== 3. restarting the site"
  database="$DATA_ROOT/annotations/review.sqlite"
  if [ -f "$database" ]; then
    # A copy only when the code will change the schema of the database (one-way migration).
    "$PY" - "$database" <<'PYEOF'
import sqlite3, sys
from datetime import datetime, timezone
from nnslr_tools.review_db import SCHEMA_VERSION
path = sys.argv[1]
source = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
version = source.execute('SELECT MAX(version) FROM schema_version').fetchone()[0] or 0
if version < SCHEMA_VERSION:
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    target = sqlite3.connect(f'{path}.bak-v{version}-{stamp}')
    source.backup(target)
    target.close()
    print(f'database copy (version {version}): {path}.bak-v{version}-{stamp}')
else:
    print(f'database already at version {version}: no copy')
PYEOF
  fi
  if systemctl list-unit-files "$SERVICE.service" 2> /dev/null | grep -q "^$SERVICE.service"; then
    systemctl restart "$SERVICE" || die "restarting $SERVICE failed: journalctl -u $SERVICE -n 50"
    sleep 2
    systemctl is-active --quiet "$SERVICE" || die "$SERVICE is not active: journalctl -u $SERVICE -n 50"
    echo "$SERVICE restarted"
  else
    echo "service $SERVICE not installed: restart the site with helpers/2_review.sh --stop and helpers/2_review.sh"
  fi
else
  echo "To turn on /mcp in the running site: helpers/setup_mcp.sh --restart (or systemctl restart $SERVICE)."
fi
echo "Check: curl -H \"Authorization: Bearer \$NNSLR_REVIEW_TOKEN\" http://127.0.0.1:8765/api/v1  -> \"mcp\": true"
