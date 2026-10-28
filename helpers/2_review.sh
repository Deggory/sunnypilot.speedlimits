#!/usr/bin/env bash
# Step 2 of 3: the review web site (nnslr review). It starts in the background and prints the
# link to open from the PC: there you confirm or correct the signs proposed by
# helpers/1_prepare.sh. Every confirmation is saved at once in data/annotations/review.sqlite.
# Next: helpers/3_train.sh.
#
# Usage: helpers/2_review.sh [--local] [--foreground] [--stop]
#   helpers/2_review.sh                (starts it, or prints the link again when it is already on)
#   helpers/2_review.sh --stop         (stops it)
#   helpers/2_review.sh --local        (only from this server: from the PC an ssh tunnel is needed)
#   helpers/2_review.sh --foreground   (stays attached to the terminal, Ctrl+C to stop)
#
# By default it listens on the LAN (0.0.0.0) and every request needs the access token: the
# link printed here already has it. The token is created once in the .env of this clone
# (NNSLR_REVIEW_TOKEN=..., permissions 600, ignored by git), the same one the service uses,
# so the link stays the same after a restart and can be bookmarked. Guide: docs/GUIDE.md.
# When the systemd service speedlimits-web (scripts/install_review_service.sh) is active it does
# not start a second server: it prints the link of the service.
# -h, --help: this help.
# Env: NNSLR_REVIEW_PORT (default 8765), NNSLR_DATA_ROOT (default data/)

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export COMMA_VIDEOS_DIR="${COMMA_VIDEOS_DIR:-$(dirname "$ROOT")/comma-videos}"
DATA_ROOT="${NNSLR_DATA_ROOT:-$ROOT/data}"
NN="$ROOT/.venv/bin/nnslr"
PY="$ROOT/.venv/bin/python"
PORT="${NNSLR_REVIEW_PORT:-8765}"
REPO_ENV="${NNSLR_REVIEW_REPO_ENV:-$ROOT/.env}"
LOG="$DATA_ROOT/logs/review.log"
PID_FILE="$DATA_ROOT/logs/review.pid"
SERVICE_ENV="/etc/speedlimits-web.env"

die() { echo "ERROR: $*" >&2; exit 1; }

# Prints the comment block at the top of this file.
usage() {
  sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'
  exit 0
}

server_ip() {
  hostname -I 2>/dev/null | awk '{print $1}'
}

# Pid of the server started by this script, when it is still running.
running_pid() {
  local pid
  [ -f "$PID_FILE" ] || return 1
  pid="$(cat "$PID_FILE")"
  [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null && grep -q "review" "/proc/$pid/cmdline" 2>/dev/null \
    && echo "$pid"
}

port_busy() {
  (exec 3<>"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null
}

# The access token: the NNSLR_REVIEW_TOKEN line of the .env of the clone, created once
# (permissions 600) and appended to the other lines of that file when it already exists.
token() {
  local value=""
  if [ -f "$REPO_ENV" ]; then
    value="$(sed -n 's/^NNSLR_REVIEW_TOKEN=//p' "$REPO_ENV" | tail -1)"
  fi
  if [ -z "$value" ]; then
    value="$("$PY" -c 'import secrets; print(secrets.token_urlsafe(24))')"
    (
      umask 077
      if [ ! -f "$REPO_ENV" ]; then
        echo "# Local secrets of this clone: the file is in .gitignore, it never goes into git." > "$REPO_ENV"
        echo "# Access token of the review site (service speedlimits-web and helpers/2_review.sh)." >> "$REPO_ENV"
      fi
      echo "NNSLR_REVIEW_TOKEN=$value" >> "$REPO_ENV"
    )
    chmod 600 "$REPO_ENV"
    echo "Token created and saved in $REPO_ENV" >&2
  fi
  echo "$value"
}

# [site-login] - START
# $1, the token, is not printed: the browser signs in with name and password; the token is only the
# setup code of the first visit (and the key of the API), and it stays in .env.
print_lan_link() {
  echo "Open from the PC:  http://$(server_ip):$PORT/"
  echo "Panel:             http://$(server_ip):$PORT/pannello"
  echo "Sign in with name and password. The first visit creates the administrator with the setup"
  echo "code: it is the token NNSLR_REVIEW_TOKEN of $REPO_ENV (grep NNSLR_REVIEW_TOKEN $REPO_ENV)."
}
# [site-login] - END

print_local_link() {
  echo "From the PC open the tunnel first (leave that terminal open):"
  echo "  ssh -L $PORT:127.0.0.1:$PORT root@$(server_ip)"
  echo "then in the browser:  http://localhost:$PORT/"
}

local_only=0
foreground=0
stop=0

while [ $# -gt 0 ]; do
  if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    usage
  elif [ "$1" = "--local" ]; then
    local_only=1
  elif [ "$1" = "--foreground" ]; then
    foreground=1
  elif [ "$1" = "--stop" ]; then
    stop=1
  else
    die "invalid option: $1 (see --help)"
  fi
  shift
done

[ -x "$NN" ] || die "$NN is missing: create the environment as in docs/INSTALL.md"
mkdir -p "$(dirname "$LOG")"

if [ "$stop" -eq 1 ]; then
  if pid="$(running_pid)"; then
    kill "$pid"
    rm -f "$PID_FILE"
    echo "Review site stopped (pid $pid)."
  else
    echo "No review site started by this script."
    if systemctl is-active --quiet speedlimits-web 2>/dev/null; then
      echo "The speedlimits-web service is installed: stop it with  systemctl stop speedlimits-web"
    fi
  fi
  exit 0
fi

# Already on: the systemd service, or a server started earlier by this script.
if systemctl is-active --quiet speedlimits-web 2>/dev/null; then
  echo "The review site already runs as the speedlimits-web service (port 8765)."
  if [ -r "$SERVICE_ENV" ]; then
    PORT=8765
    print_lan_link "$(sed -n 's/^NNSLR_REVIEW_TOKEN=//p' "$SERVICE_ENV")"
  else
    echo "The token is in $SERVICE_ENV (readable by root only)."
  fi
  exit 0
fi
if pid="$(running_pid)"; then
  echo "The review site is already on (pid $pid, log $LOG)."
  if grep -q "in ascolto su 127.0.0.1" "$LOG" 2>/dev/null; then
    print_local_link
  else
    print_lan_link "$(token)"
  fi
  exit 0
fi
port_busy && die "port $PORT is already in use by another program (NNSLR_REVIEW_PORT=<other> to change it)"

if [ "$local_only" -eq 1 ]; then
  host="127.0.0.1"
  unset NNSLR_REVIEW_TOKEN
else
  host="0.0.0.0"
  NNSLR_REVIEW_TOKEN="$(token)"
  export NNSLR_REVIEW_TOKEN
fi
command=("$NN" review --host "$host" --port "$PORT")

if [ "$foreground" -eq 1 ]; then
  if [ "$local_only" -eq 1 ]; then print_local_link; else print_lan_link "$NNSLR_REVIEW_TOKEN"; fi
  echo "Ctrl+C to stop."
  exec "${command[@]}"
fi

nohup "${command[@]}" < /dev/null > "$LOG" 2>&1 &
echo $! > "$PID_FILE"
# The server stops at once when it cannot start (database, port): wait for its first line.
for _ in 1 2 3 4 5 6 7 8 9 10; do
  grep -q "in ascolto" "$LOG" 2>/dev/null && break
  kill -0 "$(cat "$PID_FILE")" 2>/dev/null || break
  sleep 0.5
done
if ! kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  rm -f "$PID_FILE"
  tail -5 "$LOG" >&2
  die "the review site did not start (full log: $LOG)"
fi

echo "Review site started in the background (pid $(cat "$PID_FILE"), log $LOG)."
if [ "$local_only" -eq 1 ]; then print_local_link; else print_lan_link "$NNSLR_REVIEW_TOKEN"; fi
echo "Stop it with:  helpers/2_review.sh --stop"
