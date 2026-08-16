#!/bin/bash
# ERNIE-Image-Turbo Feldlauf, losgelöst von worker2 gegen einen bereits laufenden
# Container auf einer anderen Maschine (Adresse in ERNIE_ENDPOINT). Danach Vergleichsseite + Hub. Container am Ende abräumen.
set -u
export PATH="/usr/local/bin:/usr/bin:/bin:$PATH"
# Pfade nicht fest verdrahten: Repo relativ zu diesem Skript, die
# Geschwister-Repos daneben. Ueberschreibbar, falls die Ablage abweicht.
IMG="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS="${SOUTHBYTE_RESULTS:-$(dirname "$IMG")/southbyte-results}"
# Judge-Adresse und Schluessel kommen aus der gitignorierten .env, nicht aus
# diesem Skript — es liegt in einem oeffentlichen Repository.
set -a; . "$IMG/.env" 2>/dev/null; set +a
CO='Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_019YpS7jJGxYt9cD5nRMS7fH'
EL=$IMG/results/ernie_field.log
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$EL"; }
push(){ local repo="$1" msg="$2"; shift 2; cd "$repo" || return
  git add "$@" 2>/dev/null
  git commit -q -m "$msg

$CO" 2>/dev/null && log "commit: $msg" || log "nichts zu committen ($repo)"
  git pull --rebase origin main >/dev/null 2>&1 || true
  git push origin main 2>&1 | tail -1 | tee -a "$EL"
}

cd "$IMG"
log "=== ERNIE-FIELD START (Eval gegen $ERNIE_ENDPOINT, 8 Steps) ==="
python3 "$IMG/eval/image_eval.py" \
  --endpoint "${ERNIE_ENDPOINT:?ERNIE_ENDPOINT fehlt — Adresse des laufenden Containers}" \
  --model-name ERNIE-Image-Turbo \
  --steps 8 --seed 42 --timeout 600 >> "$EL" 2>&1
RC=$?
log "EVAL rc=$RC"
log "summary: $(cat "$IMG"/results/*_ERNIE-Image-Turbo/summary.json 2>/dev/null | tr -d '\n' | sed 's/  */ /g')"

log "Vergleichsseite bauen (make_docs)"
python3 "$IMG/eval/make_docs.py" >> "$EL" 2>&1
push "$IMG" "ERNIE-Image-Turbo: Feldlauf ergänzt (worker1, 4 Modelle)" docs

log "Hub aktualisieren (build_site)"
python3 "$RESULTS/build_site.py" >> "$EL" 2>&1
push "$RESULTS" "Hub: Image-Vergleich 4 Modelle (ERNIE ergänzt)" build_site.py docs

log "ERNIE-Container abräumen (GPU freigeben)"
ssh -o BatchMode=yes "${ERNIE_SSH_HOST:?ERNIE_SSH_HOST fehlt}" 'docker rm -f southbyte-image-ernie' >/dev/null 2>&1 && log "Container entfernt" || log "Container-Entfernen fehlgeschlagen"

log "ERNIE_FIELD_DONE rc=$RC"
log "=== ERNIE-FIELD ENDE ==="
