"""Loader-Registry für den southbyte-image Serving-Harness.

Jede Text-zu-Bild-Familie hat eine Lade-Strategie. Das Profil (image_profile.conf)
wählt sie über PROFILE_LOADER → Env IMG_LOADER. Alle Loader geben eine
diffusers-kompatible Pipeline zurück (`.to(device)` + `__call__(prompt=..., ...)`),
sodass der Server-Generate-Pfad familienunabhängig bleibt.

Strategien
----------
  auto              AutoPipelineForText2Image        FLUX.1-schnell, Qwen-Image-*
  diffusion         DiffusionPipeline (via _class_name)  ERNIE (ErnieImagePipeline)
  mage_flow         from mage_flow import MageFlowPipeline   Mage-Flow (ext. Lib)
  flux2_singlefile  Flux2Pipeline + Single-File-NVFP4-Transformer  FLUX.2-klein/-dev

`auto` und `diffusion` laufen im aktuellen Container (diffusers 0.38). `mage_flow`
und `flux2_singlefile` brauchen das v2-Image (mage_flow-Paket bzw. nunchaku/torchao
für NVFP4) — der Code ist aber schon hier, damit ein neues Modell nur noch ein
Profil + den Loader-Namen braucht.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable

import torch

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def _dtype(name: str):
    return _DTYPES.get(name, torch.bfloat16)


# ── Strategien ───────────────────────────────────────────────────────────────
def _load_auto(model_path: str, dtype, opts: dict) -> Any:
    from diffusers import AutoPipelineForText2Image
    return AutoPipelineForText2Image.from_pretrained(
        model_path, torch_dtype=dtype, local_files_only=True)


def _load_diffusion(model_path: str, dtype, opts: dict) -> Any:
    # Respektiert _class_name aus model_index.json (Custom-Klassen, die in diffusers
    # existieren, aber nicht im AutoPipeline-Mapping stehen — z.B. ErnieImagePipeline).
    from diffusers import DiffusionPipeline
    return DiffusionPipeline.from_pretrained(
        model_path, torch_dtype=dtype, local_files_only=True)


class _MageFlowAdapter:
    """diffusers-kompatible Hülle um MageFlowPipeline (2026-08-11).

    MageFlowPipeline ist KEINE diffusers-Pipe: `from_pretrained(repo_dir, device='cuda')`
    (kein torch_dtype), lädt schon auf cuda (kein `.to()`), und generiert via
    `generate(prompts, steps=, cfg=, heights=, widths=, neg_prompts=, seeds=) -> [PIL]`.
    Der Server erwartet aber `.to(device)` + `__call__(prompt=..., width, height,
    num_inference_steps, guidance_scale, ...).images`. Diese Hülle übersetzt das.
    """
    def __init__(self, mage: Any):
        self._mage = mage

    def to(self, _device):  # mage_flow lädt in from_pretrained bereits auf cuda
        return self

    def __call__(self, prompt=None, negative_prompt=None, width=1024, height=1024,
                 num_inference_steps=30, guidance_scale=5.0, num_images_per_prompt=1,
                 generator=None, **_ignored):
        n = int(num_images_per_prompt or 1)
        prompts = [prompt] * n
        kw: dict[str, Any] = dict(steps=int(num_inference_steps), cfg=float(guidance_scale),
                                  heights=[int(height)] * n, widths=[int(width)] * n)
        if negative_prompt:
            kw["neg_prompts"] = [negative_prompt] * n
        if generator is not None:  # Server reicht torch.Generator; mage will int-seeds
            try:
                kw["seeds"] = [int(generator.initial_seed()) % (2**32)] * n
            except Exception:
                pass
        images = self._mage.generate(prompts, **kw)
        return type("_Out", (), {"images": images})()


def _load_mage_flow(model_path: str, dtype, opts: dict) -> Any:
    # Braucht das mage_flow-Paket (github.com/microsoft/Mage) im Image.
    from mage_flow import MageFlowPipeline  # type: ignore
    mage = MageFlowPipeline.from_pretrained(model_path)  # device='cuda' default; kein torch_dtype
    return _MageFlowAdapter(mage)


def _load_flux2_singlefile(model_path: str, dtype, opts: dict) -> Any:
    """FLUX.2-klein/-dev: Single-File-NVFP4-Transformer + geteilte FLUX.2-Komponenten.

    opts:
      COMPONENTS_DIR  Pfad zu einer vollständigen FLUX.2-Pipeline (model_index +
                      text_encoder + vae); der Single-File-Transformer wird eingesetzt.
      TRANSFORMER_FILE  .safetensors des quantisierten Transformers (Default: erste im dir).
    """
    from diffusers import Flux2Pipeline, Flux2Transformer2DModel  # type: ignore
    # 2026-08-11: COMPONENTS_DIR kommt aus der Config als bloßer Verzeichnis-NAME
    # (z.B. black-forest-labs--FLUX.2-dev). Relativ zum Modell-Store auflösen (Parent von
    # model_path = /hf_models), sonst sucht from_pretrained im CWD/Hub → schlägt offline fehl.
    comp = opts.get("COMPONENTS_DIR")
    if comp:
        comp = str(Path(comp) if Path(comp).is_absolute() else Path(model_path).parent / comp)
    else:
        comp = model_path
    tf_file = opts.get("TRANSFORMER_FILE")
    if not tf_file:
        cands = sorted(Path(model_path).glob("*.safetensors"))
        if not cands:
            raise RuntimeError(f"kein Single-File-Transformer in {model_path}")
        tf_file = str(cands[0])
    # config= auf die lokale FLUX.2-dev/transformer-Config zeigen (Single-File-Checkpoint hat
    # keine eigene config.json → sonst Hub-Fetch, offline-Fehler). local_files_only offline-safe.
    _tf_cfg = str(Path(comp) / "transformer")
    transformer = Flux2Transformer2DModel.from_single_file(
        tf_file, config=_tf_cfg, local_files_only=True, torch_dtype=dtype)
    return Flux2Pipeline.from_pretrained(
        comp, transformer=transformer, torch_dtype=dtype, local_files_only=True)


LOADERS: dict[str, Callable[[str, Any, dict], Any]] = {
    "auto": _load_auto,
    "diffusion": _load_diffusion,
    "mage_flow": _load_mage_flow,
    "flux2_singlefile": _load_flux2_singlefile,
}


def _autodetect(model_path: str) -> str:
    """Ohne PROFILE_LOADER: aus dem Repo die Strategie raten."""
    mi = Path(model_path) / "model_index.json"
    if mi.exists():
        cls = json.loads(mi.read_text(encoding="utf-8")).get("_class_name", "")
        if cls in ("FluxPipeline", "QwenImagePipeline", "StableDiffusionXLPipeline"):
            return "auto"
        if cls == "MageFlowPipeline":
            return "mage_flow"
        return "diffusion"  # sonstige Custom-Klasse in diffusers
    # kein model_index → Single-File-Checkpoint (nur *.safetensors)
    if list(Path(model_path).glob("*.safetensors")):
        return "flux2_singlefile"
    return "auto"


def load_pipeline(model_path: str, *, loader: str | None = None,
                  dtype_name: str = "bfloat16", opts: dict | None = None) -> tuple[Any, str]:
    """Lädt die Pipeline gemäß Loader (oder Auto-Detect). Gibt (pipe, loader_name)."""
    opts = opts or {}
    name = (loader or os.environ.get("IMG_LOADER") or "").strip() or _autodetect(model_path)
    if name not in LOADERS:
        raise RuntimeError(f"unbekannter IMG_LOADER {name!r}; bekannt: {list(LOADERS)}")
    pipe = LOADERS[name](model_path, _dtype(dtype_name), opts)
    return pipe, name
