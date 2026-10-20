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
  flux3_modular     ModularPipeline + Prompt-Upsampler (lokal)   FLUX.3 [klein]

`auto` und `diffusion` laufen im aktuellen Container (diffusers 0.38). `mage_flow`
und `flux2_singlefile` brauchen das v2-Image (mage_flow-Paket bzw. nunchaku/torchao
für NVFP4) — der Code ist aber schon hier, damit ein neues Modell nur noch ein
Profil + den Loader-Namen braucht.

`flux3_modular` braucht das v4-Image: FLUX.3 steckt nur im diffusers-Hauptzweig
und nur als ModularPipeline.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from types import SimpleNamespace
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
    from diffusers import (Flux2Pipeline, Flux2Transformer2DModel,  # type: ignore
                           NVIDIAModelOptConfig)
    import modelopt.torch.quantization as mtq  # type: ignore
    from modelopt.torch.opt import enable_huggingface_checkpointing  # type: ignore
    # 2026-08-11: Die FLUX.2-NVFP4-Checkpoints sind NVIDIA-ModelOpt-NVFP4. enable_huggingface_
    # checkpointing() patcht diffusers/HF-Laden dafür; ohne die ModelOpt-Quant-Config lädt
    # diffusers die gepackten Weights falsch (Shape-/Parameter-Fehler). Der "-mixed"-Checkpoint
    # kommt durch den Konverter (die voll-nvfp4-Variante hat noch einen diffusers-chunk-Bug).
    enable_huggingface_checkpointing()
    # 2026-08-11: COMPONENTS_DIR kommt aus der Config als bloßer Verzeichnis-NAME
    # (z.B. black-forest-labs--FLUX.2-dev). Relativ zum Modell-Store auflösen (Parent von
    # model_path = /hf_models), sonst sucht from_pretrained im CWD/Hub → schlägt offline fehl.
    comp = opts.get("COMPONENTS_DIR")
    if comp:
        comp = str(Path(comp) if Path(comp).is_absolute() else Path(model_path).parent / comp)
    else:
        comp = model_path
    tf_file = opts.get("TRANSFORMER_FILE")
    if tf_file and not Path(tf_file).is_absolute():
        tf_file = str(Path(model_path) / tf_file)   # bloßer Name → unter model_path
    if not tf_file:
        cands = sorted(Path(model_path).glob("*.safetensors"))
        if not cands:
            raise RuntimeError(f"kein Single-File-Transformer in {model_path}")
        # "-mixed"-Variante bevorzugen: sie kommt durch den diffusers-Konverter; die voll-nvfp4-
        # Datei hat noch den chunk-Bug (torch.chunk auf 0-dim-Scale). Sonst erste.
        mixed = [c for c in cands if "mixed" in c.name.lower()]
        tf_file = str((mixed or cands)[0])
    # config= auf die lokale FLUX.2-dev/transformer-Config zeigen (Single-File-Checkpoint hat
    # keine eigene config.json → sonst Hub-Fetch, offline-Fehler). local_files_only offline-safe.
    _tf_cfg = str(Path(comp) / "transformer")
    # modelopt_config direkt setzen umgeht den diffusers-Bug in get_config_from_quant_type (NVFP4).
    _qc = NVIDIAModelOptConfig(quant_type="NVFP4", modelopt_config=mtq.NVFP4_DEFAULT_CFG)
    transformer = Flux2Transformer2DModel.from_single_file(
        tf_file, config=_tf_cfg, quantization_config=_qc, local_files_only=True, torch_dtype=dtype)
    return Flux2Pipeline.from_pretrained(
        comp, transformer=transformer, torch_dtype=dtype, local_files_only=True)


def _load_flux2_bnb(model_path: str, dtype, opts: dict) -> Any:
    """FLUX.2-dev via bitsandbytes-NF4 (diffusers/FLUX.2-dev-bnb-4bit) + FLUX.2-dev-Komponenten.

    Der funktionierende diffusers-native FLUX.2-Weg (2026-08-11): der NF4-Transformer liegt als
    PROPER diffusers-Format im `transformer/`-Subfolder mit bnb-quantization_config in der
    config.json → normales from_pretrained lädt ihn 4-bit (kein NVFP4-Single-File-Bruch). Die
    restlichen Komponenten (text_encoder/VAE/scheduler/tokenizer) kommen aus COMPONENTS_DIR
    (= black-forest-labs--FLUX.2-dev, voll gesynct).

    opts:
      COMPONENTS_DIR   Verzeichnis der vollen FLUX.2-dev-Pipeline (Name → /hf_models aufgelöst).
      TRANSFORMER_SUBFOLDER  Subfolder des NF4-Transformers im bnb-Repo (Default 'transformer').
    """
    from diffusers import Flux2Pipeline, Flux2Transformer2DModel  # type: ignore
    comp = opts.get("COMPONENTS_DIR")
    if comp:
        comp = str(Path(comp) if Path(comp).is_absolute() else Path(model_path).parent / comp)
    else:
        comp = model_path
    sub = opts.get("TRANSFORMER_SUBFOLDER") or "transformer"
    # bnb-quantization_config steht in <model_path>/<sub>/config.json → from_pretrained lädt 4-bit.
    transformer = Flux2Transformer2DModel.from_pretrained(
        model_path, subfolder=sub, torch_dtype=dtype, local_files_only=True)
    return Flux2Pipeline.from_pretrained(
        comp, transformer=transformer, torch_dtype=dtype, local_files_only=True)


def _synchronisieren() -> None:
    """Wartet auf die GPU, damit eine Zeitmessung die Rechnung enthält und nicht nur den Aufruf."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class _Flux3ModularAdapter:
    """diffusers-kompatible Hülle um die FLUX.3-ModularPipeline (2026-10-20).

    Eine ModularPipeline liefert keinen Output mit `.images`, sondern den
    PipelineState; was man braucht, fordert man mit `output=` an. Die Hülle holt
    die Bilder UND den Prompt, den der Upsampler geschrieben hat. Bei FLUX.3 ist
    der Prompt der Anfrage nicht der, den das Bildmodell sieht — ohne den
    umgeschriebenen Text lässt sich ein Lauf nicht nachvollziehen.

    Die Signatur nennt bewusst weder `negative_prompt` noch `guidance_scale`:
    das Distillat kennt beides nicht (16 Schritte, keine CFG). Der Server
    verwirft die beiden dann mit einer Warnzeile im Log, wie bei FLUX.2.
    """

    def __init__(self, pipe: Any, upsampler: Any = None):
        self._pipe = pipe
        self.prompt_upsampler_aktiv = upsampler is not None
        self._messung = {"sekunden": 0.0, "tokens": 0}
        if upsampler is not None:
            self._generate_messen(upsampler)

    def _generate_messen(self, upsampler: Any) -> None:
        """Misst den Upsampler getrennt, ohne die Pipeline umzubauen.

        Der Upsampler-Block ruft `prompt_upsampler.generate(...)` auf. Die Instanz
        bekommt eine Hülle um genau diese Methode, die Zeit und neu erzeugte
        Tokens mitschreibt. Die Tokens zeigen, ob die Beschreibung an der Grenze
        von `max_new_tokens` (3584) abgeschnitten wurde.
        """
        original = upsampler.generate
        messung = self._messung

        def generate(*args: Any, **kwargs: Any) -> Any:
            _synchronisieren()
            t0 = time.perf_counter()
            out = original(*args, **kwargs)
            _synchronisieren()
            messung["sekunden"] += time.perf_counter() - t0
            ids = kwargs.get("input_ids")
            if ids is not None and hasattr(out, "shape"):
                messung["tokens"] += int(out.shape[-1] - ids.shape[-1])
            return out

        upsampler.generate = generate

    def to(self, device: Any) -> _Flux3ModularAdapter:
        self._pipe = self._pipe.to(device)
        return self

    def __call__(self, prompt: str, width: int = 1024, height: int = 1024,
                 num_inference_steps: int = 16, num_images_per_prompt: int = 1,
                 generator: Any = None) -> SimpleNamespace:
        self._messung.update(sekunden=0.0, tokens=0)
        out = self._pipe(prompt=prompt, width=int(width), height=int(height),
                         num_inference_steps=int(num_inference_steps),
                         num_images_per_prompt=int(num_images_per_prompt or 1),
                         generator=generator, output=["images", "prompt"])
        ergebnis = SimpleNamespace(images=out["images"])
        if self.prompt_upsampler_aktiv:
            ergebnis.revised_prompt = out["prompt"]
            ergebnis.upsample_seconds = round(self._messung["sekunden"], 2)
            ergebnis.upsample_tokens = self._messung["tokens"]
        return ergebnis


def _load_flux3_modular(model_path: str, dtype: Any, opts: dict) -> Any:
    """FLUX.3 [klein] als ModularPipeline, Prompt-Upsampler aus lokalem Verzeichnis.

    FLUX.3 gibt es in diffusers nur als ModularPipeline (Image v4). Die Gewichte
    kommen lazy: `from_pretrained` baut nur die Pipeline, `load_components` lädt.

    Der Upsampler wird NICHT über `modular_model_index.json` geladen: der Index
    verweist per Hub-ID auf `black-forest-labs/flux-3-image-klein-pup`, und das
    lässt sich offline nicht auflösen. Er kommt deshalb ausdrücklich aus
    COMPONENTS_DIR und wird per `update_components` eingesetzt.

    Die VAE rechnet in fp32, der Rest im Profil-dtype — so steht es in der
    diffusers-Doku zu FLUX.3 und im Beispiel von PR #15058.

    opts:
      COMPONENTS_DIR    Verzeichnis des Upsamplers (black-forest-labs--flux-3-image-klein-pup);
                        ein bloßer Name wird relativ zum Modellspeicher aufgelöst.
                        Pflicht, solange der Upsampler an ist.
      PROMPT_UPSAMPLER  0/false/aus = ohne Upsampler: der Block wird aus der
                        Pipeline entfernt, der Prompt geht unverändert an den
                        Text-Encoder. Standard ist an — die Modellkarte sagt
                        ausdrücklich, das Modell nicht ohne zu betreiben. Der
                        Schalter ist für Vergleichsläufe da.
    """
    from diffusers import ModularPipeline  # type: ignore
    from transformers import (  # type: ignore
        AutoProcessor,
        Qwen3_5ForConditionalGeneration,
    )

    schalter = str(opts.get("PROMPT_UPSAMPLER") or "1").strip().lower()
    upsampler_an = schalter not in ("0", "false", "aus", "nein", "no", "off")
    pup = None
    if upsampler_an:
        comp = opts.get("COMPONENTS_DIR")
        if not comp:
            raise RuntimeError("FLUX.3 mit Prompt-Upsampler braucht COMPONENTS_DIR "
                               "(black-forest-labs--flux-3-image-klein-pup); "
                               "ohne Upsampler: PROMPT_UPSAMPLER=0")
        pup = Path(comp) if Path(comp).is_absolute() else Path(model_path).parent / comp
        if not pup.is_dir():
            raise RuntimeError(f"Upsampler-Verzeichnis fehlt: {pup} (kommt über den Collection-Sync)")

    pipe = ModularPipeline.from_pretrained(model_path, local_files_only=True)
    if not upsampler_an:
        blocks = pipe.blocks  # eine Kopie — die Pipeline selbst bleibt unberührt
        blocks.sub_blocks.pop("prompt_upsampler")
        pipe = blocks.init_pipeline(model_path)

    # Alles außer dem Upsampler aus dem Modellverzeichnis. load_components
    # schluckt Ladefehler und schreibt nur eine Warnung; deshalb danach prüfen,
    # sonst scheitert erst das erste Bild.
    upsampler_teile = ("prompt_upsampler", "prompt_upsampler_processor")
    namen = [n for n in pipe.pretrained_component_names if n not in upsampler_teile]
    pipe.load_components(names=namen, dtype={"vae": torch.float32, "default": dtype},
                         local_files_only=True)
    fehlend = [n for n in namen if getattr(pipe, n, None) is None]
    if fehlend:
        raise RuntimeError(f"FLUX.3: Komponenten nicht geladen: {fehlend} — Ursache in den Warnungen darüber")

    upsampler = None
    if pup is not None:
        upsampler = Qwen3_5ForConditionalGeneration.from_pretrained(
            str(pup), dtype=dtype, local_files_only=True).eval()
        processor = AutoProcessor.from_pretrained(str(pup), local_files_only=True)
        pipe.update_components(prompt_upsampler=upsampler, prompt_upsampler_processor=processor)
    print(f"[load] FLUX.3: {type(pipe).__name__}, Komponenten {namen}, "
          f"Prompt-Upsampler {pup if pup is not None else 'AUS'}", flush=True)
    return _Flux3ModularAdapter(pipe, upsampler)


LOADERS: dict[str, Callable[[str, Any, dict], Any]] = {
    "auto": _load_auto,
    "diffusion": _load_diffusion,
    "mage_flow": _load_mage_flow,
    "flux2_singlefile": _load_flux2_singlefile,
    "flux2_bnb": _load_flux2_bnb,
    "flux3_modular": _load_flux3_modular,
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
