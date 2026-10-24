#!/usr/bin/env bash
# Startet ein Bildmodell über vLLM-Omni, wie es sein Rezept auf recipes.vllm.ai
# vorsieht, und davor auf Wunsch den Prompt-Upsampler auf normalem vLLM.
# Bisher nur FLUX.3 klein: Day-0-Image vllm/vllm-omni:flux-3-image-klein, der
# Upsampler flux-3-image-klein-pup (Qwen3.5-9B) auf vllm/vllm-openai:v0.29.0.
#
# Nacheinander, nicht gleichzeitig: vLLM misst beim Start den freien Speicher
# und bricht ab, wenn sich der unterdessen ändert. Der Upsampler muss bereit
# sein, bevor das Bildmodell lädt.
set -euo pipefail

HF_MODELS_DIR="${HF_MODELS_DIR:-$HOME/hf_models}"
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

docker run -d --name "${CONTAINER_NAME}" --gpus all --ipc=host \
  -p "${HOST_PORT}:8010" -v "${HF_MODELS_DIR}:/hf_models:ro" -e HF_HUB_OFFLINE=1 \
  "${IMAGE}" \
  vllm serve "/hf_models/${MODEL_DIR}" --omni --served-model-name "${SERVED_NAME}" --port 8010 >/dev/null
echo "Bildmodell startet: ${MODEL_DIR} über vLLM-Omni auf :${HOST_PORT}"
