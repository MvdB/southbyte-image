#!/usr/bin/env bash
# southbyte-image — Feldlauf: alle Zielmodelle nacheinander über das Testset rendern.
# Nur ein ~54-GB-Modell passt gleichzeitig → Container je Modell starten, generieren,
# stoppen. Danach Vergleichsseite bauen. Bewertung (OCR/Judge) ist ein separater
# Schritt (Judge-VLM muss erst serviert werden).
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="${REPO}/results/field_run.log"
mkdir -p "${REPO}/results"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

# Modell-Verzeichnis  ⟶  Label (für results/<datum>_<label>/)
MODELS=(
  "black-forest-labs--FLUX.1-schnell|FLUX.1-schnell"
  "Qwen--Qwen-Image-2512|Qwen-Image-2512"
  "nvidia--Qwen-Image-Flash|Qwen-Image-Flash"
)

log "═══ Feldlauf Start (${#MODELS[@]} Modelle) ═══"
for entry in "${MODELS[@]}"; do
  dir="${entry%%|*}"; label="${entry##*|}"
  log "── Modell: ${label} (${dir}) ──"

  MODEL_DIR="${dir}" bash "${REPO}/serving/run_image.sh" >>"$LOG" 2>&1 || { log "FEHLER Start ${label}"; continue; }

  # Health abwarten (54-GB-Ladezeit; bis ~15 min)
  ok=0
  for i in $(seq 1 90); do
    if curl -sf http://localhost:8010/health 2>/dev/null | grep -q '"status": *"ok"'; then ok=1; break; fi
    sleep 10
  done
  if [ "$ok" -ne 1 ]; then
    log "FEHLER: ${label} nicht healthy — überspringe. Letzte Logs:"
    docker logs --tail 6 southbyte-image >>"$LOG" 2>&1
    docker rm -f southbyte-image >/dev/null 2>&1
    continue
  fi
  log "${label} healthy nach ~$((i*10))s — generiere 22 Fälle"

  python3 "${REPO}/eval/image_eval.py" --model-name "${label}" --timeout 600 >>"$LOG" 2>&1 \
    && log "${label} fertig: $(cat "${REPO}"/results/*_"${label}"/summary.json 2>/dev/null | tr -d '\n' | sed 's/  */ /g')" \
    || log "FEHLER Eval ${label}"

  docker rm -f southbyte-image >/dev/null 2>&1
  log "${label} Container gestoppt"
done

log "── Vergleichsseite bauen ──"
python3 "${REPO}/eval/make_docs.py" >>"$LOG" 2>&1 && log "docs/index.html gebaut" || log "FEHLER make_docs"
log "═══ Feldlauf Ende ═══"
