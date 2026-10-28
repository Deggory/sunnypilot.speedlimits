#!/usr/bin/env bash
# Forgotten password of the site: the administrator starts over. It stops the site, moves the
# administrator file (web/auth/admin.json of the data folder) aside as admin.json.<time>.old and
# starts the site again. The next visit asks for the setup code (NNSLR_REVIEW_TOKEN in the .env of
# this clone), then a new name and password. Every browser has to sign in again; the reviews, the
# AI reviewer and the other settings stay as they are.
# Works with the systemd service speedlimits-web, or with the site started by helpers/2_review.sh.
# Guide: docs/GUIDE.md.
#
# Usage: helpers/reset_password.sh
# -h, --help: this help.
# Env: NNSLR_DATA_ROOT (default: the one of /etc/speedlimits-web.env, else data/)

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SERVICE="speedlimits-web"
SERVICE_ENV="/etc/speedlimits-web.env"

die() { echo "ERROR: $*" >&2; exit 1; }

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'
  exit 0
fi
[ $# -eq 0 ] || die "unknown option: $1 (see --help)"

# The data folder of the service when its environment file names one (only that line is read).
DATA_ROOT="${NNSLR_DATA_ROOT:-}"
if [ -z "$DATA_ROOT" ] && [ -r "$SERVICE_ENV" ]; then
  DATA_ROOT="$(sed -n 's/^NNSLR_DATA_ROOT=//p' "$SERVICE_ENV" | tail -1 | tr -d "\"'")"
fi
DATA_ROOT="${DATA_ROOT:-$ROOT/data}"
ADMIN="$DATA_ROOT/web/auth/admin.json"
PID_FILE="$DATA_ROOT/logs/review.pid"

if [ ! -f "$ADMIN" ]; then
  echo "No administrator yet ($ADMIN): the next visit already asks for the setup code."
  exit 0
fi

service=0
if command -v systemctl > /dev/null 2>&1 \
    && systemctl list-unit-files "$SERVICE.service" 2>/dev/null | grep -q "^$SERVICE.service"; then
  service=1
fi
manual=0
local_flag=()
if [ "$service" -eq 0 ] && [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  manual=1
  if grep -q "in ascolto su 127.0.0.1" "$DATA_ROOT/logs/review.log" 2>/dev/null; then local_flag=(--local); fi
fi

# The site keeps the administrator in memory: it is stopped first and started again after.
if [ "$service" -eq 1 ]; then
  echo "== stopping the service $SERVICE"
  systemctl stop "$SERVICE"
elif [ "$manual" -eq 1 ]; then
  echo "== stopping the site started by helpers/2_review.sh"
  "$ROOT/helpers/2_review.sh" --stop
fi

old="$ADMIN.$(date -u +%Y%m%dT%H%M%SZ).old"
mv "$ADMIN" "$old"
chmod 600 "$old"
echo "The old administrator is kept in $old"

if [ "$service" -eq 1 ]; then
  systemctl start "$SERVICE"
  sleep 2
  systemctl is-active --quiet "$SERVICE" || die "the service does not start again: journalctl -u $SERVICE -n 50"
  echo "Service started again."
elif [ "$manual" -eq 1 ]; then
  "$ROOT/helpers/2_review.sh" "${local_flag[@]}" > /dev/null
  echo "Site started again."
else
  echo "No site is running: the next start (the service or helpers/2_review.sh) asks for the setup code."
fi
echo "Open the site: it asks for the setup code (NNSLR_REVIEW_TOKEN in the .env of this clone), then a new name and password."
