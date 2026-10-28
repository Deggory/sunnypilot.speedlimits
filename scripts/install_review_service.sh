#!/usr/bin/env bash
# Install (or remove) the speed limits web site (nnslr review: review pages and control
# panel) as the systemd service "speedlimits-web" on the LXC. Safe to run again: the token
# file is created only when it is missing, the unit is rewritten only when it changed.
# The service used to be called "nnslr-review": an install replaces it (the old token file
# /etc/nnslr-review.env becomes /etc/speedlimits-web.env, with the same content).
#
#   sudo scripts/install_review_service.sh             install / update and start
#   scripts/install_review_service.sh --dry-run        show what would be done, change nothing
#   sudo scripts/install_review_service.sh --uninstall stop and remove the service (token file kept)
#
# The token is printed only at the end of an install, in the addresses to open.
# It is also kept in the .env of this clone (NNSLR_REVIEW_TOKEN=..., permissions 600,
# ignored by git), the one place to look it up, shared with helpers/2_review.sh: the
# first install takes the token already there, otherwise it creates one and writes it
# in both files. Full procedure: docs/GUIDE.md.
# NNSLR_REVIEW_ENV_FILE, NNSLR_REVIEW_UNIT_FILE and NNSLR_REVIEW_REPO_ENV override the
# three paths (used by the tests).
set -euo pipefail

SERVICE_NAME="speedlimits-web"
PORT=8765
ENV_FILE="${NNSLR_REVIEW_ENV_FILE:-/etc/speedlimits-web.env}"
UNIT_FILE="${NNSLR_REVIEW_UNIT_FILE:-/etc/systemd/system/${SERVICE_NAME}.service}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
REPO_ENV="${NNSLR_REVIEW_REPO_ENV:-$REPO/.env}"
TEMPLATE="$REPO/deploy/speedlimits-web.service"
# The same service under its old name: replaced by an install, removed by --uninstall.
OLD_SERVICE_NAME="nnslr-review"
OLD_UNIT_FILE="/etc/systemd/system/${OLD_SERVICE_NAME}.service"
OLD_ENV_FILE="/etc/nnslr-review.env"
NNSLR="$REPO/.venv/bin/nnslr"
PYTHON="$REPO/.venv/bin/python"

usage() {
  cat <<USAGE
Uso: $(basename "$0") [--dry-run] [--uninstall] [--help]

Installa il sito web di speedlimits (nnslr review: revisione e pannello) come
servizio systemd "$SERVICE_NAME", in ascolto su 0.0.0.0:$PORT, avviato a ogni boot.
Se c'e' ancora il vecchio servizio "$OLD_SERVICE_NAME", lo sostituisce.

  (nessuna opzione)  installa o aggiorna e avvia il servizio (serve root)
  --dry-run          mostra cosa farebbe, senza cambiare niente
  --uninstall        ferma e rimuove il servizio; il file del token resta
  --help             questo aiuto

File usati:
  token     $ENV_FILE (creato solo se manca, permessi 600)
            e copia in $REPO_ENV (NNSLR_REVIEW_TOKEN=..., permessi 600, fuori da git)
  servizio  $UNIT_FILE
  clone     $REPO
USAGE
}

say() { printf '%s\n' "$*"; }
fail() { printf 'Errore: %s\n' "$*" >&2; exit 1; }

MODE="install"
DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --uninstall) MODE="uninstall" ;;
    --dry-run) DRY_RUN=1 ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Opzione sconosciuta: %s\n\n' "$arg" >&2; usage >&2; exit 2 ;;
  esac
done

# The paths go into the unit through sed and into ExecStart unquoted: plain characters only.
check_plain_path() {
  case "$1" in
    *[!A-Za-z0-9._/+-]*) fail "percorso con caratteri non supportati (solo lettere, cifre, . _ / + -): $1" ;;
  esac
}

need_writable_dir() {
  if [ ! -w "$(dirname "$1")" ]; then
    fail "non posso scrivere in $(dirname "$1"): rilancia con sudo."
  fi
}

need_systemctl() {
  if ! command -v systemctl >/dev/null 2>&1; then
    fail "systemctl non trovato: questo script serve su una macchina con systemd (la LXC)."
  fi
}

render_unit() {
  sed -e "s|__NNSLR_REPO__|$REPO|g" -e "s|__NNSLR_ENV_FILE__|$ENV_FILE|g" "$TEMPLATE"
}

new_token() {
  "$PYTHON" - <<'PY'
import secrets
import string

alphabet = string.ascii_letters + string.digits
token = ''
for _ in range(32):
    token = token + secrets.choice(alphabet)
print(token)
PY
}

read_token() {
  local file="${1:-$ENV_FILE}"
  local line
  [ -f "$file" ] || return 0
  while IFS= read -r line || [ -n "$line" ]; do
    case "$line" in
      NNSLR_REVIEW_TOKEN=*) printf '%s\n' "${line#NNSLR_REVIEW_TOKEN=}"; return 0 ;;
    esac
  done < "$file"
  return 0
}

# Writes the token in the .env of the clone, replacing an older NNSLR_REVIEW_TOKEN line
# and keeping every other line.
save_repo_token() {
  local token="$1"
  local temporary="$REPO_ENV.tmp"
  (
    umask 077
    if [ -f "$REPO_ENV" ]; then
      grep -v '^NNSLR_REVIEW_TOKEN=' "$REPO_ENV" > "$temporary" || true
    else
      say '# Segreti locali di questo clone: il file e'"'"' in .gitignore, non va mai in git.' > "$temporary"
      say '# Token di accesso al sito di revisione (servizio speedlimits-web e helpers/2_review.sh).' >> "$temporary"
    fi
    say "NNSLR_REVIEW_TOKEN=$token" >> "$temporary"
    mv "$temporary" "$REPO_ENV"
  )
  chmod 600 "$REPO_ENV"
}

valid_token() {
  case "$1" in
    ''|*[!A-Za-z0-9_-]*) return 1 ;;
  esac
  [ "${#1}" -ge 16 ]
}

lan_ips() {
  local found=""
  found="$(hostname -I 2>/dev/null || true)"
  if [ -z "${found// /}" ] && command -v ip >/dev/null 2>&1; then
    found="$(ip -4 -o addr show scope global 2>/dev/null | awk '{print $4}' | cut -d/ -f1 || true)"
  fi
  if [ -z "${found// /}" ]; then
    found="$(hostname)"
  fi
  local address
  for address in $found; do
    case "$address" in
      *:*) ;;          # IPv6: the server listens on IPv4 0.0.0.0
      *) say "$address" ;;
    esac
  done
}

write_env_file() {
  local token="$1"
  (
    umask 077
    {
      say '# Token di accesso di "nnslr review" (servizio speedlimits-web). Tienilo segreto.'
      say '# Per cambiarlo: scrivi un nuovo valore (almeno 16 caratteri tra lettere, cifre, - e _)'
      say '# e riavvia: systemctl restart speedlimits-web'
      say "NNSLR_REVIEW_TOKEN=$token"
      say '# Cartella dei dati, se non è <clone>/data:'
      say '# NNSLR_DATA_ROOT=/percorso/dei/dati'
    } > "$ENV_FILE"
  )
  chmod 600 "$ENV_FILE"
}

# Stops and removes the service under its old name, nnslr-review. KillMode=process there too:
# a preparation or a training started from the panel keeps running.
remove_old_service() {
  if [ ! -e "$OLD_UNIT_FILE" ]; then
    return 0
  fi
  if [ "$DRY_RUN" -eq 1 ]; then
    say "[prova] fermerei e toglierei il vecchio servizio $OLD_SERVICE_NAME ($OLD_UNIT_FILE)."
    return 0
  fi
  need_writable_dir "$OLD_UNIT_FILE"
  systemctl disable --now "$OLD_SERVICE_NAME.service" >/dev/null 2>&1 || true
  rm -f "$OLD_UNIT_FILE"
  systemctl daemon-reload
  say "Tolto il vecchio servizio $OLD_SERVICE_NAME (ora si chiama $SERVICE_NAME)."
}

# The token file of the old service becomes the new one, so its token and settings
# (NNSLR_DATA_ROOT, NNSLR_PANEL_ACTIONS) stay. Only with the default paths.
move_old_env_file() {
  if [ -n "${NNSLR_REVIEW_ENV_FILE:-}" ] || [ -e "$ENV_FILE" ] || [ ! -e "$OLD_ENV_FILE" ]; then
    return 0
  fi
  if [ "$DRY_RUN" -eq 1 ]; then
    say "[prova] sposterei $OLD_ENV_FILE in $ENV_FILE (stesso token, stesse impostazioni)."
    return 0
  fi
  need_writable_dir "$ENV_FILE"
  mv "$OLD_ENV_FILE" "$ENV_FILE"
  sed -i 's/nnslr-review/speedlimits-web/g' "$ENV_FILE"
  chmod 600 "$ENV_FILE"
  say "Spostato il file del token: $OLD_ENV_FILE -> $ENV_FILE"
}

uninstall_service() {
  if [ "$DRY_RUN" -eq 1 ]; then
    say "[prova] fermerei e disabiliterei $SERVICE_NAME, cancellerei $UNIT_FILE e rileggerei systemd."
    remove_old_service
    say "[prova] il file del token $ENV_FILE resterebbe."
    return 0
  fi
  need_systemctl
  if [ -e "$UNIT_FILE" ]; then
    need_writable_dir "$UNIT_FILE"
  fi
  systemctl disable --now "$SERVICE_NAME.service" >/dev/null 2>&1 || true
  if [ -e "$UNIT_FILE" ]; then
    rm -f "$UNIT_FILE"
    say "Servizio rimosso: $UNIT_FILE"
  else
    say "Servizio non installato: niente da rimuovere."
  fi
  systemctl daemon-reload
  remove_old_service
  if [ -e "$ENV_FILE" ]; then
    say "Il file del token resta: $ENV_FILE (cancellalo a mano se non serve più)."
  fi
}

install_service() {
  [ -f "$TEMPLATE" ] || fail "manca il modello del servizio: $TEMPLATE"
  [ -x "$NNSLR" ] || fail "manca $NNSLR: crea prima l'ambiente .venv (docs/INSTALL.md, sezione 2)."
  [ -x "$PYTHON" ] || fail "manca $PYTHON: crea prima l'ambiente .venv (docs/INSTALL.md, sezione 2)."
  check_plain_path "$REPO"
  check_plain_path "$ENV_FILE"

  if [ "$DRY_RUN" -eq 1 ]; then
    remove_old_service
    move_old_env_file
    if [ -e "$ENV_FILE" ]; then
      say "[prova] il file del token $ENV_FILE esiste già: lo lascerei com'è (permessi 600)."
    elif valid_token "$(read_token "$REPO_ENV")"; then
      say "[prova] creerei $ENV_FILE con il token già presente in $REPO_ENV (permessi 600)."
    else
      say "[prova] creerei $ENV_FILE con un token casuale di 32 caratteri (permessi 600)."
    fi
    say "[prova] terrei lo stesso token anche in $REPO_ENV (riga NNSLR_REVIEW_TOKEN=, permessi 600, fuori da git)."
    say "[prova] scriverei $UNIT_FILE con questo contenuto:"
    render_unit
    say "[prova] poi: systemctl daemon-reload; systemctl enable --now $SERVICE_NAME.service"
    say "[prova] alla fine stamperei gli indirizzi da aprire (http://<IP>:$PORT/) e come entrare la prima volta."
    return 0
  fi

  need_systemctl
  need_writable_dir "$ENV_FILE"
  need_writable_dir "$UNIT_FILE"
  # First the old service goes: it holds the port 8765 the new one needs.
  remove_old_service
  move_old_env_file

  if [ -e "$ENV_FILE" ]; then
    chmod 600 "$ENV_FILE"
    say "File del token già presente: $ENV_FILE (lasciato com'è)."
  else
    # The first install keeps the token already in the .env of the clone (helpers/2_review.sh).
    local first
    first="$(read_token "$REPO_ENV")"
    if ! valid_token "$first"; then
      first="$(new_token)"
    fi
    write_env_file "$first"
    say "Creato il file del token: $ENV_FILE (permessi 600)."
  fi
  local token
  token="$(read_token)"
  if ! valid_token "$token"; then
    fail "in $ENV_FILE manca NNSLR_REVIEW_TOKEN valido (almeno 16 caratteri tra lettere, cifre, - e _): correggilo, oppure cancella il file e rilancia."
  fi
  if [ "$(read_token "$REPO_ENV")" != "$token" ]; then
    save_repo_token "$token"
    say "Token salvato anche in $REPO_ENV (permessi 600, fuori da git)."
  fi

  local was_active=0
  if systemctl is-active --quiet "$SERVICE_NAME.service"; then
    was_active=1
  fi
  local rendered
  rendered="$(mktemp)"
  render_unit > "$rendered"
  local changed=0
  if [ -f "$UNIT_FILE" ] && cmp -s "$rendered" "$UNIT_FILE"; then
    say "Servizio già aggiornato: $UNIT_FILE"
  else
    install -m 644 "$rendered" "$UNIT_FILE"
    changed=1
    say "Servizio scritto: $UNIT_FILE"
  fi
  rm -f "$rendered"

  systemctl daemon-reload
  systemctl enable --now "$SERVICE_NAME.service"
  if [ "$changed" -eq 1 ] && [ "$was_active" -eq 1 ]; then
    systemctl restart "$SERVICE_NAME.service"
    say "Servizio riavviato con la nuova configurazione."
  fi
  if systemctl is-active --quiet "$SERVICE_NAME.service"; then
    say "Servizio $SERVICE_NAME attivo, si riavvia da solo a ogni boot."
  else
    say "Attenzione: il servizio non risulta attivo. Guarda: journalctl -u $SERVICE_NAME -n 50"
  fi

  say ""
  # [site-login] - START
  say "Apri nel browser:"
  local address
  while IFS= read -r address; do
    say "  http://$address:$PORT/"
  done < <(lan_ips)
  say "Si entra con nome e password. La prima visita crea l'amministratore con il codice di configurazione:"
  say "e' il token NNSLR_REVIEW_TOKEN (grep NNSLR_REVIEW_TOKEN $REPO_ENV). Non condividerlo: e' anche la chiave"
  say "dell'API /api/v1 e di MCP."
  # [site-login] - END
}

if [ "$MODE" = "uninstall" ]; then
  uninstall_service
else
  install_service
fi
