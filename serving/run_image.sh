#!/usr/bin/env bash
# Startet den southbyte-image Text-to-Image-Container (diffusers) auf dem DGX Spark.
# Muster wie southbyte-tts/serving/run_*.sh: env-überschreibbar, Modell read-only aus
# ~/hf_models, vorhandenen Container entfernen, dann detached starten.
set -euo pipefail

HF_MODELS_DIR="${HF_MODELS_DIR:-$HOME/hf_models}"
CONTAINER_NAME="${CONTAINER_NAME:-southbyte-image}"
HOST_PORT="${HOST_PORT:-8010}"
IMAGE="${IMAGE:-spark-southbyte-image:v1}"
MODEL_DIR="${MODEL_DIR:-black-forest-labs--FLUX.1-schnell}"
SPARK_PROFILES_DIR="${SPARK_PROFILES_DIR:-$HOME/southbyte/southbyte-spark-profiles}"

# Profil (Schritte/Guidance/Auflösung/dtype) für dieses Modell laden, falls vorhanden.
PROFILE="${SPARK_PROFILES_DIR}/image/${MODEL_DIR}/image_profile.conf"
if [[ -f "${PROFILE}" ]]; then
  # shellcheck disable=SC1090
  source "${PROFILE}"
  echo "Profil geladen: ${PROFILE}"
else
  echo "Kein Profil unter ${PROFILE} — Adapter-Defaults werden genutzt."
fi

if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  echo "Container '${CONTAINER_NAME}' existiert -> wird entfernt."
  docker rm -f "${CONTAINER_NAME}" >/dev/null
fi

docker run -d --name "${CONTAINER_NAME}" \
  --gpus all \
  --shm-size 8g \
  -p "${HOST_PORT}:8010" \
  -v "${HF_MODELS_DIR}:/hf_models:ro" \
  -e MODEL_DIR="${MODEL_DIR}" \
  -e SERVED_MODEL_NAME="${MODEL_DIR}" \
  -e IMG_LOADER="${PROFILE_LOADER:-}" \
  -e IMG_STEPS="${PROFILE_STEPS:-20}" \
  -e IMG_GUIDANCE="${PROFILE_GUIDANCE:-3.5}" \
  -e IMG_SIZE="${PROFILE_SIZE:-1024x1024}" \
  -e IMG_DTYPE="${PROFILE_DTYPE:-bfloat16}" \
  -e IMG_COMPONENTS_DIR="${PROFILE_COMPONENTS_DIR:-}" \
  -e IMG_TRANSFORMER_FILE="${PROFILE_TRANSFORMER_FILE:-}" \
  "${IMAGE}"

echo "Gestartet: ${CONTAINER_NAME} auf :${HOST_PORT}  (Modell ${MODEL_DIR})"
echo "  Health:  curl -s http://localhost:${HOST_PORT}/health"
echo "  Logs:    docker logs -f ${CONTAINER_NAME}"
