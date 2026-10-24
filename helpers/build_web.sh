#!/bin/bash
# The Vue and TypeScript of the site pages: type check of web/src (vue-tsc, also the .vue components),
# then one script per page into src/nnslr_tools/web/js/<page>.js (web/src/pages/<page>.ts and what it
# imports, built by Vite in web/build.mjs), which pages.py puts inline in the page with its nonce. The
# built files are committed: the server and helpers/update.sh never need Node.
#
#   helpers/build_web.sh          install the tools if missing (web/node_modules), check, build
#   helpers/build_web.sh --check  check, and fail if the committed scripts are not the build of the sources
#
# Node comes from PyPI through uv (nodejs-wheel), nothing installed in the system; Vue, Vite, vue-tsc and
# TypeScript are the versions of web/package-lock.json.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
WEB=$REPO/web
OUT=$REPO/src/nnslr_tools/web/js
NODE=(uvx --quiet --from 'nodejs-wheel==24.*')
CHECK=0
case "${1:-}" in
  --check) CHECK=1 ;;
  '') ;;
  *) echo "uso: $0 [--check]" >&2; exit 2 ;;
esac

if [ ! -f "$WEB/node_modules/vite/package.json" ] || [ ! -f "$WEB/node_modules/vue-tsc/bin/vue-tsc.js" ]; then
  echo "Installo Vue, Vite e TypeScript in web/node_modules (npm ci)…"
  (cd "$WEB" && "${NODE[@]}" npm ci --no-audit --no-fund)
fi

echo "Controllo dei tipi (vue-tsc)…"
(cd "$WEB" && "${NODE[@]}" node node_modules/vue-tsc/bin/vue-tsc.js --noEmit -p .)

target=$OUT
if [ "$CHECK" = 1 ]; then
  target=$(mktemp -d)
  trap 'rm -rf "$target"' EXIT
fi
mkdir -p "$target"
echo "Compilo le pagine (Vite)…"
(cd "$WEB" && "${NODE[@]}" node build.mjs "$target")

if [ "$CHECK" = 1 ]; then
  if ! diff -r "$target" "$OUT" >/dev/null; then
    echo "src/nnslr_tools/web/js non corrisponde ai sorgenti di web/src: lancia helpers/build_web.sh e committa." >&2
    diff -rq "$target" "$OUT" >&2 || true
    exit 1
  fi
  echo "Script delle pagine aggiornati."
else
  # A page removed from web/src/pages leaves no stale script behind.
  for built in "$OUT"/*.js; do
    [ -e "$WEB/src/pages/$(basename "$built" .js).ts" ] || rm -f "$built"
  done
  echo "Script delle pagine in src/nnslr_tools/web/js:"
  ls "$OUT"
fi
