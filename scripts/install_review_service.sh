#!/usr/bin/env bash
# Install (or remove) the NNSLR review web server as the systemd service
# "nnslr-review" on the LXC. Safe to run again: the token file is created only
# when it is missing, the unit is rewritten only when it changed.
#
#   sudo scripts/install_review_service.sh             install / update and start
#   scripts/install_review_service.sh --dry-run        show what would be done, change nothing
#   sudo scripts/install_review_service.sh --uninstall stop and remove the service (token file kept)
#
# The token is printed only at the end of an install, in the addresses to open.
# NNSLR_REVIEW_ENV_FILE and NNSLR_REVIEW_UNIT_FILE override the two system
# paths (used by the tests).
set -euo pipefail

SERVICE_NAME="nnslr-review"
PORT=8765
ENV_FILE="${NNSLR_REVIEW_ENV_FILE:-/etc/nnslr-review.env}"
UNIT_FILE="${NNSLR_REVIEW_UNIT_FILE:-/etc/systemd/system/${SERVICE_NAME}.service}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
TEMPLATE="$REPO/deploy/nnslr-review.service"
NNSLR="$REPO/.venv/bin/nnslr"
PYTHON="$REPO/.venv/bin/python"

usage() {
  cat <<USAGE
Uso: $(basename "$0") [--dry-run] [--uninstall] [--help]

Installa il server di revisione web (nnslr review) come servizio systemd
"$SERVICE_NAME", in ascolto su 0.0.0.0:$PORT, avviato a ogni boot.

  (nessuna opzione)  installa o aggiorna e avvia il servizio (serve root)
  --dry-run          mostra cosa farebbe, senza cambiare niente
  --uninstall        ferma e rimuove il servizio; il file del token resta
  --help             questo aiuto

File usati:
  token     $ENV_FILE (creato solo se manca, permessi 600)
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
  local line
  while IFS= read -r line || [ -n "$line" ]; do
    case "$line" in
      NNSLR_REVIEW_TOKEN=*) printf '%s\n' "${line#NNSLR_REVIEW_TOKEN=}"; return 0 ;;
    esac
  done < "$ENV_FILE"
  return 0
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
      say '# Token di accesso di "nnslr review" (servizio nnslr-review). Tienilo segreto.'
      say '# Per cambiarlo: scrivi un nuovo valore (almeno 16 caratteri tra lettere, cifre, - e _)'
      say '# e riavvia: systemctl restart nnslr-review'
      say "NNSLR_REVIEW_TOKEN=$token"
      say '# Cartella dei dati, se non è <clone>/data:'
      say '# NNSLR_DATA_ROOT=/percorso/dei/dati'
    } > "$ENV_FILE"
  )
  chmod 600 "$ENV_FILE"
}

uninstall_service() {
  if [ "$DRY_RUN" -eq 1 ]; then
    say "[prova] fermerei e disabiliterei $SERVICE_NAME, cancellerei $UNIT_FILE e rileggerei systemd."
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
    if [ -e "$ENV_FILE" ]; then
      say "[prova] il file del token $ENV_FILE esiste già: lo lascerei com'è (permessi 600)."
    else
      say "[prova] creerei $ENV_FILE con un token casuale di 32 caratteri (permessi 600)."
    fi
    say "[prova] scriverei $UNIT_FILE con questo contenuto:"
    render_unit
    say "[prova] poi: systemctl daemon-reload; systemctl enable --now $SERVICE_NAME.service"
    say "[prova] alla fine stamperei gli indirizzi da aprire (http://<IP>:$PORT/?token=...)."
    return 0
  fi

  need_systemctl
  need_writable_dir "$ENV_FILE"
  need_writable_dir "$UNIT_FILE"

  if [ -e "$ENV_FILE" ]; then
    chmod 600 "$ENV_FILE"
    say "File del token già presente: $ENV_FILE (lasciato com'è)."
  else
    write_env_file "$(new_token)"
    say "Creato il file del token: $ENV_FILE (permessi 600)."
  fi
  local token
  token="$(read_token)"
  if ! valid_token "$token"; then
    fail "in $ENV_FILE manca NNSLR_REVIEW_TOKEN valido (almeno 16 caratteri tra lettere, cifre, - e _): correggilo, oppure cancella il file e rilancia."
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
  say "Apri nel browser (l'indirizzo contiene il token: non condividerlo):"
  local address
  while IFS= read -r address; do
    say "  http://$address:$PORT/?token=$token"
  done < <(lan_ips)
  say "Dopo la prima visita il token resta in un cookie e sparisce dalla barra degli indirizzi."
  say "Per rileggere il token: sudo grep NNSLR_REVIEW_TOKEN $ENV_FILE"
}

if [ "$MODE" = "uninstall" ]; then
  uninstall_service
else
  install_service
fi
