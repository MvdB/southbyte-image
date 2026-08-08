"""southbyte-image — diffusers Serving-Adapter (Text-to-Image) auf dem DGX Spark.

Stellt eine OpenAI-Images-kompatible API vor eine lokale diffusers-Pipeline.
Ein Adapter bedient genau ein Modell (per MODEL_DIR gewählt); die Familie fährt
ohnehin nur ein ~54-GB-Modell gleichzeitig. Modelle kommen read-only aus
/hf_models (Host: ~/hf_models via southbyte-sync), nichts wird zur Laufzeit geladen.

Endpunkte:
  GET  /health                 → {status, model, device}
  GET  /v1/models              → OpenAI-Style Modell-Liste
  POST /v1/images/generations  → OpenAI-Images-kompatibel (+ Erweiterungen)

Konfiguration per Env (Defaults kommen aus dem image_profile.conf, das run_image.sh
in Env-Variablen übersetzt):
  MODEL_DIR      Verzeichnisname unter /hf_models (z.B. black-forest-labs--FLUX.1-schnell)
  IMG_STEPS      num_inference_steps (Default 20)
  IMG_GUIDANCE   guidance_scale bzw. true_cfg_scale (Default 3.5)
  IMG_SIZE       WxH (Default 1024x1024)
  IMG_DTYPE      bfloat16|float16 (Default bfloat16)
"""
from __future__ import annotations

import base64
import inspect
import io
import os
import time
from typing import Any

import torch
import uvicorn
from diffusers import AutoPipelineForText2Image, DiffusionPipeline
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ── Konfiguration ────────────────────────────────────────────────────────────
HF_ROOT = os.environ.get("HF_ROOT", "/hf_models")
MODEL_DIR = os.environ.get("MODEL_DIR", "")
MODEL_PATH = os.path.join(HF_ROOT, MODEL_DIR)
SERVED_NAME = os.environ.get("SERVED_MODEL_NAME", MODEL_DIR)

DEF_STEPS = int(os.environ.get("IMG_STEPS", "20"))
DEF_GUIDANCE = float(os.environ.get("IMG_GUIDANCE", "3.5"))
DEF_W, DEF_H = (int(x) for x in os.environ.get("IMG_SIZE", "1024x1024").lower().split("x"))
_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
DTYPE = _DTYPES.get(os.environ.get("IMG_DTYPE", "bfloat16"), torch.bfloat16)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI(title="southbyte-image", version="0.1.0")
_PIPE: Any = None  # lazy geladen beim Startup


# ── Request/Response-Modelle (OpenAI-Images + Erweiterungen) ─────────────────
class GenRequest(BaseModel):
    prompt: str
    n: int = 1
    size: str | None = None                 # "WxH"; None → Profil-Default
    response_format: str = "b64_json"       # b64_json (url wird nicht unterstützt)
    # Erweiterungen über die OpenAI-Spec hinaus:
    negative_prompt: str | None = None
    steps: int | None = None
    guidance_scale: float | None = None
    seed: int | None = None
    model: str | None = None                # ignoriert (ein Adapter = ein Modell)


def _load_pipeline() -> Any:
    if not MODEL_DIR or not os.path.isdir(MODEL_PATH):
        raise RuntimeError(f"MODEL_DIR ungültig oder nicht gemountet: {MODEL_PATH!r}")
    # AutoPipeline erkennt FluxPipeline / QwenImagePipeline etc. aus model_index.json.
    # Custom-Pipelines (z.B. ErnieImagePipeline) stehen nicht im AutoPipeline-Mapping —
    # dann greift DiffusionPipeline, das die Klasse direkt aus _class_name auflöst.
    try:
        pipe = AutoPipelineForText2Image.from_pretrained(
            MODEL_PATH, torch_dtype=DTYPE, local_files_only=True
        )
    except (ValueError, EnvironmentError):
        pipe = DiffusionPipeline.from_pretrained(
            MODEL_PATH, torch_dtype=DTYPE, local_files_only=True
        )
    print(f"[load] Pipeline: {type(pipe).__name__}", flush=True)
    pipe = pipe.to(DEVICE)
    # Auf dem GB10 (Unified Memory) bleibt alles auf der GPU; kein CPU-Offload nötig.
    return pipe


@app.on_event("startup")
def _startup() -> None:
    global _PIPE
    _PIPE = _load_pipeline()


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok" if _PIPE is not None else "loading",
            "model": SERVED_NAME, "device": DEVICE, "dtype": str(DTYPE),
            "pipeline": type(_PIPE).__name__ if _PIPE is not None else None}


@app.get("/v1/models")
def models() -> dict[str, Any]:
    return {"object": "list",
            "data": [{"id": SERVED_NAME, "object": "model", "owned_by": "southbyte"}]}


@app.post("/v1/images/generations")
def generate(req: GenRequest) -> dict[str, Any]:
    if _PIPE is None:
        raise HTTPException(503, "Pipeline lädt noch")
    if req.response_format != "b64_json":
        raise HTTPException(400, "nur response_format=b64_json wird unterstützt")

    if req.size:
        try:
            w, h = (int(x) for x in req.size.lower().split("x"))
        except ValueError:
            raise HTTPException(400, f"ungültige size {req.size!r}, erwartet WxH")
    else:
        w, h = DEF_W, DEF_H

    steps = req.steps or DEF_STEPS
    guidance = req.guidance_scale if req.guidance_scale is not None else DEF_GUIDANCE
    generator = None
    if req.seed is not None:
        generator = torch.Generator(device=DEVICE).manual_seed(req.seed)

    # QwenImage nutzt true_cfg_scale statt guidance_scale — anhand der echten
    # Call-Signatur entscheiden, welcher Parametername akzeptiert wird.
    kwargs: dict[str, Any] = dict(
        prompt=req.prompt, negative_prompt=req.negative_prompt,
        width=w, height=h, num_inference_steps=steps,
        num_images_per_prompt=req.n, generator=generator,
    )
    sig_params = inspect.signature(_PIPE.__call__).parameters
    if "true_cfg_scale" in sig_params:
        kwargs["true_cfg_scale"] = guidance
    else:
        kwargs["guidance_scale"] = guidance

    t0 = time.perf_counter()
    with torch.inference_mode():
        out = _PIPE(**{k: v for k, v in kwargs.items() if v is not None})
    dt = time.perf_counter() - t0

    data = []
    for img in out.images:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        data.append({"b64_json": base64.b64encode(buf.getvalue()).decode()})

    return {"created": int(t0), "model": SERVED_NAME,
            "gen_seconds": round(dt, 2), "data": data}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8010")))
