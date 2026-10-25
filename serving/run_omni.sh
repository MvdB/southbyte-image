#!/usr/bin/env bash
# Startet ein Bildmodell über vLLM-Omni, wie es sein Rezept auf recipes.vllm.ai
# vorsieht, und davor auf Wunsch den Prompt-Upsampler auf normalem vLLM.
# FLUX.3 klein: Day-0-Image vllm/vllm-omni:flux-3-image-klein, der Upsampler
# flux-3-image-klein-pup (Qwen3.5-9B) auf vllm/vllm-openai:v0.29.0.
# Ming-Image: vllm/vllm-omni:v0.32.0rc1 mit der Deploy-Config des Rezepts, auf eine
# GPU umgestellt (serving/omni/ming_image_einzel.yaml, im Container unter
# /southbyte-omni), uebergeben als OMNI_ARGS (JSON-Liste aus omni_args der Registry).
#
# Nacheinander, nicht gleichzeitig: vLLM misst beim Start den freien Speicher
# und bricht ab, wenn sich der unterdessen ändert. Der Upsampler muss bereit
# sein, bevor das Bildmodell lädt.
set -euo pipefail

HF_MODELS_DIR="${HF_MODELS_DIR:-$HOME/hf_models}"
OMNI_CONFIGS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/omni"
CONTAINER_NAME="${CONTAINER_NAME:-southbyte-image}"
HOST_PORT="${HOST_PORT:-8010}"
IMAGE="${IMAGE:?IMAGE fehlt, etwa vllm/vllm-omni:flux-3-image-klein}"
MODEL_DIR="${MODEL_DIR:?MODEL_DIR fehlt}"
SERVED_NAME="${SERVED_NAME:-$MODEL_DIR}"
PUP_DIR="${PUP_DIR:-}"
PUP_IMAGE="${PUP_IMAGE:-vllm/vllm-openai:v0.29.0}"
PUP_PORT="${PUP_PORT:-8014}"
# Das Rezept gibt dem Upsampler 0.90 einer eigenen GPU. Auf dem GB10 teilen sich
# alle Prozesse 128 GB; ein Viertel reicht für 18 GB Gewichte plus KV-Cache.
PUP_GPU_ANTEIL="${PUP_GPU_ANTEIL:-0.25}"
# Zusatzargumente fuer vllm serve als JSON-Liste, z. B. '["--deploy-config", "..."]'.
mapfile -t OMNI_ZUSATZ < <(python3 -c 'import json, os, sys; sys.stdout.write("".join(a + "\n" for a in json.loads(os.environ.get("OMNI_ARGS") or "[]")))')

for c in "${CONTAINER_NAME}" "${CONTAINER_NAME}-pup"; do
  docker rm -f "$c" >/dev/null 2>&1 || true
done

bereit() {  # $1 = URL, $2 = Container
  for _ in $(seq 180); do
    curl -sf --max-time 5 "$1" >/dev/null && return 0
    docker ps --format '{{.Names}}' | grep -qx "$2" || { docker logs --tail 15 "$2"; return 1; }
    sleep 5
  done
  return 1
}

if [[ -n "${PUP_DIR}" ]]; then
  docker run -d --name "${CONTAINER_NAME}-pup" --gpus all --ipc=host \
    -p "${PUP_PORT}:8000" -v "${HF_MODELS_DIR}:/hf_models:ro" -e HF_HUB_OFFLINE=1 \
    "${PUP_IMAGE}" \
    --model "/hf_models/${PUP_DIR}" --served-model-name "${PUP_DIR}" \
    --max-model-len 32768 --gpu-memory-utilization "${PUP_GPU_ANTEIL}" --reasoning-parser qwen3 >/dev/null
  echo "Upsampler startet: ${PUP_DIR} auf :${PUP_PORT}"
  bereit "http://localhost:${PUP_PORT}/v1/models" "${CONTAINER_NAME}-pup" || { echo "Upsampler nicht bereit"; exit 1; }
  echo "Upsampler bereit"
fi

# MODEL_AUSBLENDEN (Leerzeichen-getrennt): Eintraege des Modellordners, die vLLM-Omni
# nicht sehen soll. Es bekommt dann eine Ansicht aus Verweisen auf alle uebrigen
# Eintraege (results/_ansichten/, nicht im Repo); der Modellordner selbst bleibt,
# wie ihn der Collection-Sync liefert. Anlass: Ming-Image hat seit 2026-10-14 eine
# config.json im Wurzelverzeichnis ("Hub download tracking"); mit ihr erkennt
# vLLM-Omni die Sprachstufe als Modell und laedt die Diffusionsstufe ueber die
# Z-Image-Pipeline — die VAE-Gewichte passen dann nicht (384 gegen 64).
MODELL_PFAD="/hf_models/${MODEL_DIR}"; ANSICHT_MOUNT=()
if [[ -n "${MODEL_AUSBLENDEN:-}" ]]; then
  ANSICHT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/results/_ansichten/${MODEL_DIR}"
  mkdir -p "${ANSICHT}"
  for e in "${HF_MODELS_DIR}/${MODEL_DIR}"/*; do
    n="$(basename "$e")"
    [[ " ${MODEL_AUSBLENDEN} " == *" ${n} "* ]] && continue
    ln -sfn "/hf_models/${MODEL_DIR}/${n}" "${ANSICHT}/${n}"
  done
  for n in ${MODEL_AUSBLENDEN}; do
    [[ -L "${ANSICHT}/${n}" ]] && { echo "Ansicht enthaelt ${n} noch — bitte ${ANSICHT} pruefen" >&2; exit 1; }
  done
  ANSICHT_MOUNT=(-v "${ANSICHT}:/modell:ro"); MODELL_PFAD=/modell
  echo "Ansicht ohne: ${MODEL_AUSBLENDEN}"
fi

docker run -d --name "${CONTAINER_NAME}" --gpus all --ipc=host \
  -p "${HOST_PORT}:8010" -v "${HF_MODELS_DIR}:/hf_models:ro" -v "${OMNI_CONFIGS}:/southbyte-omni:ro" \
  ${ANSICHT_MOUNT[@]+"${ANSICHT_MOUNT[@]}"} \
  -e HF_HUB_OFFLINE=1 "${IMAGE}" \
  vllm serve "${MODELL_PFAD}" --omni --served-model-name "${SERVED_NAME}" --port 8010 \
  ${OMNI_ZUSATZ[@]+"${OMNI_ZUSATZ[@]}"} >/dev/null
echo "Bildmodell startet: ${MODEL_DIR} über vLLM-Omni auf :${HOST_PORT}"
