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
  IMG_KV_CACHE   1 = use_kv_cache an, wenn die Pipeline es kennt (Default 1)
  IMG_SIZE       WxH (Default 1024x1024)
  IMG_DTYPE      bfloat16|float16 (Default bfloat16)

Antwort-Erweiterungen, nur wenn die Pipeline sie liefert (bisher FLUX.3 mit
Prompt-Upsampler): `data[].revised_prompt` — der Prompt, den das Bildmodell
tatsaechlich bekam, Feldname wie in der OpenAI-Images-API —, dazu
`upsample_seconds` und `upsample_tokens` neben `gen_seconds`. `gen_seconds`
bleibt die Gesamtzeit und enthaelt den Upsampler.
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
KV_CACHE = os.environ.get("IMG_KV_CACHE", "1") not in ("0", "false", "False")
DEF_W, DEF_H = (int(x) for x in os.environ.get("IMG_SIZE", "1024x1024").lower().split("x"))
_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
DTYPE = _DTYPES.get(os.environ.get("IMG_DTYPE", "bfloat16"), torch.bfloat16)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Version an genau dieser Stelle; der Commit kommt beim Bau als Bauargument
# ins Image (SOUTHBYTE_COMMIT) und steht in /health dahinter. Ohne Stempel
# steht die blanke Version da, nie ein erfundener Commit.
VERSION = "0.2.0"
_COMMIT = os.environ.get("SOUTHBYTE_COMMIT", "").strip()
VERSION_VOLL = f"{VERSION}+{_COMMIT}" if _COMMIT else VERSION

app = FastAPI(title="southbyte-image", version=VERSION_VOLL)
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


_LOADER_NAME = ""  # gesetzt beim Laden, für /health


def _load_pipeline() -> Any:
    global _LOADER_NAME
    if not MODEL_DIR or not os.path.isdir(MODEL_PATH):
        raise RuntimeError(f"MODEL_DIR ungültig oder nicht gemountet: {MODEL_PATH!r}")
    # Familie-agnostisch über die Loader-Registry. PROFILE_LOADER → IMG_LOADER wählt die
    # Strategie (auto/diffusion/mage_flow/flux2_singlefile); ohne Angabe Auto-Detect.
    # Zusatz-Optionen kommen als IMG_<KEY> (z.B. IMG_COMPONENTS_DIR) durch.
    from loaders import load_pipeline
    _reserved = {"IMG_LOADER", "IMG_STEPS", "IMG_GUIDANCE", "IMG_SIZE", "IMG_DTYPE"}
    opts = {k[4:]: v for k, v in os.environ.items()
            if k.startswith("IMG_") and k not in _reserved}
    pipe, _LOADER_NAME = load_pipeline(
        MODEL_PATH, loader=os.environ.get("IMG_LOADER"),
        dtype_name=os.environ.get("IMG_DTYPE", "bfloat16"), opts=opts)
    print(f"[load] loader={_LOADER_NAME} pipeline={type(pipe).__name__}", flush=True)
    # Auf dem GB10 (Unified Memory) bleibt alles auf der GPU; kein CPU-Offload nötig.
    return pipe.to(DEVICE)


@app.on_event("startup")
def _startup() -> None:
    global _PIPE
    _PIPE = _load_pipeline()


@app.get("/health")
def health() -> dict[str, Any]:
    antwort = {"status": "ok" if _PIPE is not None else "loading",
               "model": SERVED_NAME, "device": DEVICE, "dtype": str(DTYPE),
               "loader": _LOADER_NAME or None,
               "pipeline": type(_PIPE).__name__ if _PIPE is not None else None,
               "version": VERSION_VOLL}
    # Nur Pipelines mit Upsampler kennen den Schalter (FLUX.3); von aussen soll
    # pruefbar sein, ob er lief.
    upsampler = getattr(_PIPE, "prompt_upsampler_aktiv", None)
    if upsampler is not None:
        antwort["prompt_upsampler"] = upsampler
    return antwort


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
    # Qwen-Image-2.1-Turbo empfiehlt prefix KV caching — es spart Rechenzeit
    # ueber die Denoising-Schritte und aendert das Bild nicht. Nur setzen, wenn
    # die Pipeline es kennt; die Filterung weiter unten wuerde es sonst ohnehin
    # verwerfen, aber eine Warnzeile je Bild waere laestig.
    if KV_CACHE and "use_kv_cache" in sig_params:
        kwargs["use_kv_cache"] = True
    if "true_cfg_scale" in sig_params:
        kwargs["true_cfg_scale"] = guidance
    else:
        kwargs["guidance_scale"] = guidance

    # Nur vom Pipeline-Call akzeptierte Argumente durchreichen: FLUX.2
    # (guidance-distilled) kennt z.B. kein negative_prompt und wirft sonst
    # TypeError → HTTP 500. Pipelines mit **kwargs bleiben unangetastet.
    accepts_var_kw = any(p.kind == p.VAR_KEYWORD for p in sig_params.values())
    if not accepts_var_kw:
        dropped = [k for k in kwargs if k not in sig_params]
        if dropped:
            print(f"[warn] Pipeline akzeptiert nicht: {dropped} — verworfen", flush=True)
        kwargs = {k: v for k, v in kwargs.items() if k in sig_params}

    t0 = time.perf_counter()
    with torch.inference_mode():
        out = _PIPE(**{k: v for k, v in kwargs.items() if v is not None})
    dt = time.perf_counter() - t0

    # Hat ein Upsampler den Prompt umgeschrieben, gehoert der neue Text in die
    # Antwort: sonst ist nicht nachvollziehbar, was das Bildmodell gesehen hat.
    revised = getattr(out, "revised_prompt", None)
    data = []
    for img in out.images:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        eintrag = {"b64_json": base64.b64encode(buf.getvalue()).decode()}
        if revised:
            eintrag["revised_prompt"] = revised
        data.append(eintrag)

    antwort = {"created": int(t0), "model": SERVED_NAME,
               "gen_seconds": round(dt, 2), "data": data}
    for feld in ("upsample_seconds", "upsample_tokens"):
        wert = getattr(out, feld, None)
        if wert is not None:
            antwort[feld] = wert
    return antwort


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8010")))
